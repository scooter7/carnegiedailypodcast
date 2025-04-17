import streamlit as st
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
import re
import requests
import PyPDF2
from docx import Document

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load API keys securely from Streamlit secrets
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Constants
WORDS_PER_MINUTE = 150
VOICE_1_ID = "kPzsL2i3teMYv0FxEYQ6"  # Lisa's voice
VOICE_2_ID = "edRtkKm7qEwZ8pH9ggtf"    # Tony's voice

# Initialize session state
if "master_script" not in st.session_state:
    st.session_state.master_script = ""
if "sections" not in st.session_state:
    st.session_state.sections = []
if "section_images" not in st.session_state:
    st.session_state.section_images = {}
if "num_sections" not in st.session_state:
    st.session_state.num_sections = 3

# ================================
# UI: Host Mode and Output Type Selection
# ================================
st.title("Custom Video Script / Podcast Creator")

host_mode = st.selectbox(
    "Select Host Mode",
    [
        f"Voice 1 ({VOICE_1_ID})",
        f"Voice 2 ({VOICE_2_ID})",
        "Both (Co-host - Conversational)"
    ]
)
st.session_state.host_mode = host_mode

output_type = st.radio("Select Output Type", ["Video + Podcast", "Podcast Only"])
st.session_state.output_type = output_type

# ================================
# Helper Functions
# ================================

def download_image_from_url(url):
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
        return img_path if os.path.exists(img_path) else None
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None


def extract_text_from_file(uploaded_file):
    data = uploaded_file.read()
    uploaded_file.seek(0)
    text = ""
    # PDF handling
    if uploaded_file.name.lower().endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(BytesIO(data))
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            logging.error(f"Error reading PDF {uploaded_file.name}: {e}")
    # Word document handling
    elif uploaded_file.name.lower().endswith(".docx"):
        try:
            doc = Document(BytesIO(data))
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            logging.error(f"Error reading DOCX {uploaded_file.name}: {e}")
    else:
        logging.warning(f"Unsupported file type: {uploaded_file.name}")
    return text


def generate_dynamic_summary(all_text, desired_duration):
    max_words = (desired_duration // 60) * WORDS_PER_MINUTE
    if st.session_state.host_mode == "Both (Co-host - Conversational)":
        system_prompt = (
            f"As two podcast hosts, Lisa and Tony, engage in a natural, conversational dialogue based on the following text. "
            f"Do not include explicit speaker names before every turn. Ensure the conversation is structured in clear paragraphs. "
            f"Keep the entire conversation within {max_words} words."
        )
    else:
        system_prompt = f"As a podcast host, summarize the text to fit within {max_words} words."

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
# Main UI: File Uploader and Video Duration
# ================================
uploaded_files = st.file_uploader(
    "Upload documents (PDF or Word)", type=["pdf", "docx"], accept_multiple_files=True
)
video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

# ================================
# Generate Master Script
# ================================
if st.button("Generate Master Script"):
    if uploaded_files:
        combined_text = "".join([extract_text_from_file(file) for file in uploaded_files])
        if combined_text.strip():
            generated_summary = generate_dynamic_summary(combined_text, video_duration)
            st.session_state.master_script = generated_summary
            st.success("✅ Master script generated successfully!")
        else:
            st.error("No text extracted from the uploaded documents.")
    else:
        st.error("Please upload at least one document.")

if st.session_state.master_script:
    st.subheader("📜 Master Script")
    st.session_state.master_script = st.text_area(
        "Generated Master Script (editable):", st.session_state.master_script, height=300
    )

    # Only show section configuration for video output
    if st.session_state.output_type == "Video + Podcast":
        st.subheader("🔹 Section Configuration (for Video)")
        st.session_state.num_sections = st.number_input(
            "Number of Sections:", min_value=1, step=1, value=st.session_state.num_sections
        )
        paragraphs = st.session_state.master_script.split("\n\n")
        middle_content = "\n\n".join(paragraphs) if paragraphs else st.session_state.master_script
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
# Generate Output (Video or Podcast)
# ================================
button_label = "Create " + ("Video" if st.session_state.output_type == "Video + Podcast" else "Podcast")
if st.button(button_label):
    # ------------- Audio Generation (common) -------------
    if st.session_state.host_mode != "Both (Co-host - Conversational)":
        selected_voice_id = VOICE_1_ID if st.session_state.host_mode.startswith("Voice 1") else VOICE_2_ID
        audio_path = generate_audio_from_script(st.session_state.master_script, selected_voice_id)
        if not audio_path:
            st.error("Failed to generate audio for the script.")
            st.stop()
        audio_clip = AudioFileClip(audio_path)
    else:
        segments = [seg.strip() for seg in st.session_state.master_script.split("\n\n") if seg.strip()]
        if len(segments) < 1:
            st.error("The script must contain at least one paragraph.")
            st.stop()
        audio_segments = []
        for i, seg in enumerate(segments):
            voice_id = VOICE_1_ID if i % 2 == 0 else VOICE_2_ID
            seg_audio = generate_audio_from_script(seg, voice_id)
            if not seg_audio:
                st.error(f"Failed to generate audio for segment: {seg}")
                st.stop()
            audio_segments.append(AudioFileClip(seg_audio))
        audio_clip = concatenate_audioclips(audio_segments)

    temp_audio_file = tempfile.mktemp(suffix=".mp3")
    audio_clip.write_audiofile(temp_audio_file, fps=44100)

    # ------------------ Branch based on Output Type ------------------
    if st.session_state.output_type == "Video + Podcast":
        total_audio_duration = audio_clip.duration
        total_sections = st.session_state.num_sections
        section_duration = total_audio_duration / total_sections

        video_clips = []
        for i in range(total_sections):
            img_url = st.session_state.section_images.get(i)
            img_path = download_image_from_url(img_url) if img_url else None
            if img_path:
                video_clips.append(
                    ImageClip(img_path).set_duration(section_duration).set_fps(24)
                )
            else:
                logging.error(f"Invalid or missing image for section {i + 1}")

        if not video_clips:
            st.error("No valid images found. Video cannot be created.")
            st.stop()

        final_video_path = tempfile.mktemp(suffix=".mp4")
        combined_clip = concatenate_videoclips(video_clips, method="compose").set_audio(audio_clip)
        combined_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac", fps=24)

        st.video(final_video_path)
        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
        st.download_button("Download Audio Only", open(temp_audio_file, "rb"), "audio.mp3", mime="audio/mp3")
        st.download_button("Download Script", st.session_state.master_script, "script.txt")
    else:
        st.audio(temp_audio_file)
        st.download_button("Download Audio Only", open(temp_audio_file, "rb"), "audio.mp3", mime="audio/mp3")
        st.download_button("Download Script", st.session_state.master_script, "script.txt")
