import streamlit as st
from bs4 import BeautifulSoup
import requests
import openai
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from PIL import Image
from io import BytesIO
import logging
import json
import os

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load API keys securely from Streamlit secrets
ELEVENLABS_API_KEY = st.secrets["ELEVENLABS_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Constants
WORDS_PER_MINUTE = 150
ELEVENLABS_VOICE_ID = "NYy9s57OPECPcDJavL3T"

INTRO_TEXT = "Welcome to the CollegeXpress Campus Countdown! Let’s get started!"
INTRO_IMAGE_URL = "https://github.com/scooter7/carnegiedailypodcast/blob/main/cx.jpg?raw=true"
CONCLUSION_TEXT = "Until next time, happy college hunting!"
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

# Function to download an image and return a valid file path for MoviePy
def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))

        # Ensure the image is in RGB mode (some images may be in grayscale or RGBA)
        img = img.convert("RGB")

        # Save image as a proper format for MoviePy
        img_path = tempfile.mktemp(suffix=".png")
        img.save(img_path, format="PNG")

        return img_path  # Return the valid file path
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

# Function to scrape text content from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logging.error(f"Error scraping text from {url}: {e}")
        return ""

# Function to generate a dynamic summary
def generate_dynamic_summary(all_text, desired_duration):
    max_words = (desired_duration // 60) * WORDS_PER_MINUTE
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

# Function to generate speech using ElevenLabs
def generate_audio_from_script(script):
    try:
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
        data = {
            "text": script,
            "voice_id": ELEVENLABS_VOICE_ID,
            "model_id": "eleven_multilingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
        }
        response = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}", headers=headers, json=data)
        
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

# UI: Input URLs
st.title("Custom Video Script and Section Creator")
urls = st.text_area("Enter URLs (one per line):", height=100).splitlines()
video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

# Generate Master Script
if st.button("Generate Master Script"):
    combined_text = "\n".join([scrape_text_from_url(url) for url in urls])
    if combined_text.strip():
        generated_summary = generate_dynamic_summary(combined_text, video_duration)
        full_script = f"{INTRO_TEXT}\n\n{generated_summary}\n\n{CONCLUSION_TEXT}"
        st.session_state.master_script = full_script
        st.success("✅ Master script generated successfully!")

# Editable Master Script
if st.session_state.master_script:
    st.subheader("📜 Master Script")
    st.session_state.master_script = st.text_area("Generated Master Script (editable):", st.session_state.master_script, height=300)

    st.session_state.num_sections = st.number_input("Number of Middle Sections:", min_value=1, step=1, value=st.session_state.num_sections)

    middle_content = st.session_state.master_script.replace(INTRO_TEXT, "").replace(CONCLUSION_TEXT, "").strip()
    section_splits = middle_content.split("\n\n")

    if len(section_splits) < st.session_state.num_sections:
        section_splits += [""] * (st.session_state.num_sections - len(section_splits))
    
    st.session_state.sections = section_splits[:st.session_state.num_sections]

    st.subheader("Edit Middle Sections & Assign Images")
    for i in range(st.session_state.num_sections):
        st.session_state.sections[i] = st.text_area(f"Section {i + 1} Content:", value=st.session_state.sections[i], height=150)
        st.session_state.section_images[i] = st.text_input(f"Image URL for Section {i + 1}:")

# Generate Video
if st.button("Create Video & Generate Audio"):
    st.session_state.audio_path = generate_audio_from_script(st.session_state.master_script)
    if not st.session_state.audio_path:
        st.error("Failed to generate audio.")
        st.stop()

    audio = AudioFileClip(st.session_state.audio_path)
    video_clips = []

    # Add Intro Image
    intro_img_path = download_image_from_url(INTRO_IMAGE_URL)
    if intro_img_path:
        video_clips.append(ImageClip(intro_img_path).set_duration(3))

    # Add User-Defined Middle Sections
    for i in range(st.session_state.num_sections):
        img_url = st.session_state.section_images.get(i, "")
        img_path = download_image_from_url(img_url) if img_url else None
        if img_path:
            video_clips.append(ImageClip(img_path).set_duration(5))

    # Add Conclusion Image
    outro_img_path = download_image_from_url(CONCLUSION_IMAGE_URL)
    if outro_img_path:
        video_clips.append(ImageClip(outro_img_path).set_duration(3))

    # Ensure we have video clips before proceeding
    if video_clips:
        final_video_path = tempfile.mktemp(suffix=".mp4")
        combined_clip = concatenate_videoclips(video_clips, method="compose").set_audio(audio)
        combined_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac", fps=24)

        st.video(final_video_path)
        st.download_button("Download Video", open(final_video_path, "rb"), "generated_video.mp4", mime="video/mp4")

    # Download Audio
    st.download_button("Download Audio (MP3)", open(st.session_state.audio_path, "rb"), "generated_audio.mp3", mime="audio/mp3")
