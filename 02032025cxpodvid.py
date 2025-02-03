import streamlit as st
from bs4 import BeautifulSoup
import requests
import openai
import tempfile
from moviepy.editor import (
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    concatenate_audioclips,
)
from PIL import Image
from io import BytesIO
import logging
import os
import re  # For splitting conversation segments

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load API keys securely from Streamlit secrets
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Constants
WORDS_PER_MINUTE = 150
# Default voice IDs
VOICE_1_ID = "kPzsL2i3teMYv0FxEYQ6"  # (Lisa's voice)
VOICE_2_ID = "edRtkKm7qEwZ8pH9ggtf"    # (Tony's voice)

# Default texts for single host mode
DEFAULT_INTRO_TEXT = "Welcome to the CollegeXpress Campus Countdown! Let’s get started!"
DEFAULT_CONCLUSION_TEXT = "Until next time, happy college hunting!"

# Image URLs
INTRO_IMAGE_URL = "https://github.com/scooter7/carnegiedailypodcast/blob/main/cx.jpg?raw=true"
CONCLUSION_IMAGE_URL = "https://github.com/scooter7/carnegiedailypodcast/blob/main/cx.jpg?raw=true"

# Initialize session state
if "master_script" not in st.session_state:
    st.session_state.master_script = ""
if "sections" not in st.session_state:
    st.session_state.sections = []
if "section_images" not in st.session_state:
    st.session_state.section_images = {}
if "num_sections" not in st.session_state:
    st.session_state.num_sections = 3
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# ================================
# UI: Host Mode Selection
# ================================
st.title("Custom Video Script and Section Creator")

host_mode = st.selectbox(
    "Select Host Mode",
    [
        f"Voice 1 ({VOICE_1_ID})",
        f"Voice 2 ({VOICE_2_ID})",
        "Both (Co-host - Conversational)"
    ]
)
st.session_state.host_mode = host_mode

# ================================
# Helper Functions
# ================================

def download_image_from_url(url):
    """Download an image from a URL, save it as PNG, and return the file path."""
    if not url:
        logging.error("Empty image URL received.")
        return None
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        img_path = tempfile.mktemp(suffix=".png")
        img.save(img_path, format="PNG")
        if os.path.exists(img_path):
            return img_path
        else:
            logging.error(f"Failed to save image from {url}")
            return None
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logging.error(f"Error scraping text from {url}: {e}")
        return ""

def generate_dynamic_summary(all_text, desired_duration):
    max_words = (desired_duration // 60) * WORDS_PER_MINUTE
    # Use conversational names when in co-host mode.
    if st.session_state.get("host_mode") == "Both (Co-host - Conversational)":
        system_prompt = (
            f"As two podcast hosts named Lisa and Tony engaging in a natural, conversational back-and-forth, "
            f"generate a dynamic script that presents a lively discussion based on the following text. "
            f"The conversation should alternate naturally between Lisa and Tony, including greetings, questions, "
            f"and responses, and feel like a real conversation. Make sure each speaking turn starts with either 'Lisa:' or 'Tony:' and keep the entire conversation within {max_words} words."
        )
    else:
        system_prompt = f"As a show host, summarize the text to fit within {max_words} words."

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": all_text},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return "[Error generating summary]"

def generate_audio_from_script(script, voice_id):
    try:
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
        data = {
            "text": script,
            "voice_id": voice_id,
            "model_id": "eleven_multilingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
        }
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            audio_path = tempfile.mktemp(suffix=".mp3")
            with open(audio_path, "wb") as f:
                f.write(response.content)
            return audio_path
        else:
            st.error(f"ElevenLabs Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception in ElevenLabs API Call: {str(e)}")
        return None

# ================================
# Main UI: Input URLs and Video Duration
# ================================
urls = st.text_area("Enter URLs (one per line):", height=100).splitlines()
video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

# ================================
# Generate Master Script
# ================================
if st.button("Generate Master Script"):
    combined_text = "\n".join([scrape_text_from_url(url) for url in urls])
    if combined_text.strip():
        generated_summary = generate_dynamic_summary(combined_text, video_duration)
        if st.session_state.host_mode == "Both (Co-host - Conversational)":
            # In conversational mode, the generated script should already include dialogue labels.
            full_script = generated_summary
        else:
            full_script = f"{DEFAULT_INTRO_TEXT}\n\n{generated_summary}\n\n{DEFAULT_CONCLUSION_TEXT}"
        st.session_state.master_script = full_script
        st.success("✅ Master script generated successfully!")

if st.session_state.master_script:
    st.subheader("📜 Master Script")
    st.session_state.master_script = st.text_area(
        "Generated Master Script (editable):", st.session_state.master_script, height=300
    )

    st.subheader("🔹 Section Configuration")
    st.session_state.num_sections = st.number_input(
        "Number of Middle Sections:", min_value=1, step=1, value=st.session_state.num_sections
    )

    if st.session_state.host_mode == "Both (Co-host - Conversational)":
        middle_content = st.session_state.master_script
    else:
        paragraphs = st.session_state.master_script.split("\n\n")
        if len(paragraphs) >= 3:
            middle_content = "\n\n".join(paragraphs[1:-1])
        else:
            middle_content = st.session_state.master_script

    sentences = middle_content.split(". ")
    num_sections = st.session_state.num_sections
    section_splits = [" ".join(sentences[i::num_sections]) for i in range(num_sections)]
    if len(st.session_state.sections) != num_sections:
        st.session_state.sections = section_splits

    st.subheader("✏️ Edit Sections & Assign Images")
    for i in range(num_sections):
        st.session_state.sections[i] = st.text_area(
            f"Section {i + 1} Content:", value=st.session_state.sections[i], height=150
        )
        st.session_state.section_images[i] = st.text_input(
            f"Image URL for Section {i + 1}:", value=st.session_state.section_images.get(i, "")
        )

# ================================
# Generate Video
# ================================
if st.button("Create Video"):
    video_clips = []

    # ----------------------------
    # Audio Generation
    # ----------------------------
    if st.session_state.host_mode != "Both (Co-host - Conversational)":
        # Single voice mode
        if st.session_state.host_mode.startswith("Voice 1"):
            selected_voice_id = VOICE_1_ID
        elif st.session_state.host_mode.startswith("Voice 2"):
            selected_voice_id = VOICE_2_ID
        else:
            selected_voice_id = VOICE_1_ID
        audio_path = generate_audio_from_script(st.session_state.master_script, selected_voice_id)
        if not audio_path:
            st.error("Failed to generate audio for the script.")
            st.stop()
        audio_clip = AudioFileClip(audio_path)
    else:
        # In co-host conversational mode, split the script into segments.
        # Each segment should begin with "Lisa:" or "Tony:".
        segments = re.split(r'(?=Lisa:|Tony:)', st.session_state.master_script)
        segments = [seg.strip() for seg in segments if seg.strip()]
        if not segments:
            st.error("No conversation segments detected. Ensure your script contains 'Lisa:' and 'Tony:' labels.")
            st.stop()
        audio_segments = []
        for seg in segments:
            if seg.startswith("Lisa:"):
                voice_id = VOICE_1_ID
            elif seg.startswith("Tony:"):
                voice_id = VOICE_2_ID
            else:
                voice_id = VOICE_1_ID  # Fallback
            seg_audio_path = generate_audio_from_script(seg, voice_id)
            if not seg_audio_path:
                st.error(f"Failed to generate audio for segment: {seg}")
                st.stop()
            audio_segments.append(AudioFileClip(seg_audio_path))
        audio_clip = concatenate_audioclips(audio_segments)

    total_audio_duration = audio_clip.duration
    total_sections = st.session_state.num_sections + 2  # Intro and Outro images
    section_duration = total_audio_duration / total_sections

    # ----------------------------
    # Video Image Clips
    # ----------------------------
    # Add intro image
    intro_img_path = download_image_from_url(INTRO_IMAGE_URL)
    if intro_img_path and os.path.exists(intro_img_path):
        video_clips.append(ImageClip(intro_img_path).set_duration(section_duration).set_fps(24))
    else:
        logging.error("Intro image failed to load.")

    # Add user-defined middle section images
    for i in range(st.session_state.num_sections):
        img_url = st.session_state.section_images.get(i)
        img_path = download_image_from_url(img_url) if img_url else None
        if img_path and os.path.exists(img_path):
            video_clips.append(ImageClip(img_path).set_duration(section_duration).set_fps(24))
        else:
            logging.error(f"Invalid or missing image for section {i + 1}")

    # Add conclusion image
    outro_img_path = download_image_from_url(CONCLUSION_IMAGE_URL)
    if outro_img_path and os.path.exists(outro_img_path):
        video_clips.append(ImageClip(outro_img_path).set_duration(section_duration).set_fps(24))
    else:
        logging.error("Outro image failed to load.")

    if not video_clips:
        st.error("No valid images found. Video cannot be created.")
        st.stop()

    final_video_path = tempfile.mktemp(suffix=".mp4")
    combined_clip = concatenate_videoclips(video_clips, method="compose").set_audio(audio_clip)
    combined_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac", fps=24)

    # ----------------------------
    # Write final audio to a file for download
    # ----------------------------
    temp_audio_file = tempfile.mktemp(suffix=".mp3")
    audio_clip.write_audiofile(temp_audio_file, fps=44100)

    # ----------------------------
    # Display and Download Buttons
    # ----------------------------
    st.video(final_video_path)
    st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
    st.download_button("Download Audio Only", open(temp_audio_file, "rb"), "audio.mp3", mime="audio/mp3")
    st.download_button("Download Script", st.session_state.master_script, "script.txt")
