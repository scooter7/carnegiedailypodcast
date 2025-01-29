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

# ElevenLabs API imports
import base64
import os

logging.basicConfig(level=logging.INFO)

# Constants
WORDS_PER_MINUTE = 150
ELEVENLABS_VOICE_ID = "NYy9s57OPECPcDJavL3T"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "your-elevenlabs-api-key")  # Replace with Streamlit secrets

INTRO_TEXT = (
    "Welcome to the CollegeXpress Campus Countdown, where we explore colleges and universities around the country "
    "to help you find great schools to apply to! Let’s get started!"
)
INTRO_IMAGE_URL = "https://github.com/scooter7/carnegiedailypodcast/blob/main/cx.jpg?raw=true"
CONCLUSION_TEXT = (
    "Don’t forget, you can connect with any of our featured colleges by visiting CollegeXpress.com. "
    "Just click the green 'Yes, connect me!' buttons when you see them on the site, and then the schools you’re interested in will reach out to you with more information! "
    "You can find the links to these schools in the description below. Don’t forget to follow us on social media @CollegeXpress. "
    "Until next time, happy college hunting!"
)
CONCLUSION_IMAGE_URL = "https://github.com/scooter7/carnegiedailypodcast/blob/main/cx.jpg?raw=true"

# Initialize session state
if "master_script" not in st.session_state:
    st.session_state.master_script = ""
if "sections" not in st.session_state:
    st.session_state.sections = []
if "section_images" not in st.session_state:
    st.session_state.section_images = {}
if "num_sections" not in st.session_state:
    st.session_state.num_sections = 3  # Default middle sections

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

# Function to dynamically generate a summary script based on duration
def generate_dynamic_summary(all_text, desired_duration):
    max_words = (desired_duration // 60) * WORDS_PER_MINUTE
    system_prompt = (
        f"As a show host, summarize the text to fit within {max_words} words. "
        f"Be enthusiastic, engaging, and include key details such as accolades and testimonials."
    )
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

# Function to generate audio from script using ElevenLabs
def generate_audio_from_script(script):
    try:
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        data = {
            "text": script,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            "voice_id": ELEVENLABS_VOICE_ID,
            "model_id": "eleven_multilingual_v1"
        }
        response = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/{}".format(ELEVENLABS_VOICE_ID),
            headers=headers,
            data=json.dumps(data)
        )
        if response.status_code == 200:
            audio_path = tempfile.mktemp(suffix=".mp3")
            with open(audio_path, "wb") as f:
                f.write(response.content)
            return audio_path
        else:
            logging.error(f"Error generating audio from ElevenLabs: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# Function to download an image from a URL
def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

# UI: Input URLs
st.title("Custom Video Script and Section Creator")
urls = st.text_area("Enter URLs (one per line):", height=100).splitlines()
video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

# Generate Master Script
if st.button("Generate Master Script"):
    combined_text = ""
    for url in urls:
        combined_text += scrape_text_from_url(url)
    if combined_text:
        dynamic_summary = generate_dynamic_summary(combined_text, video_duration)
        st.session_state.master_script = f"{INTRO_TEXT}\n\n{dynamic_summary}\n\n{CONCLUSION_TEXT}"

# Editable Master Script
if st.session_state.master_script:
    st.subheader("Master Script")
    st.session_state.master_script = st.text_area(
        "Generated Master Script (editable):", st.session_state.master_script, height=300
    )

    # User decides the number of middle sections
    st.subheader("Section Configuration")
    st.session_state.num_sections = st.number_input(
        "Number of Middle Sections:", min_value=1, step=1, value=st.session_state.num_sections
    )

    # Create sections dynamically based on user input
    middle_content = st.session_state.master_script.replace(INTRO_TEXT, "").replace(CONCLUSION_TEXT, "").strip()
    section_splits = middle_content.split("\n\n") if middle_content else []

    if len(st.session_state.sections) != st.session_state.num_sections:
        st.session_state.sections = section_splits[:st.session_state.num_sections] + [
            "" for _ in range(max(0, st.session_state.num_sections - len(section_splits)))
        ]
        st.session_state.section_images = {i: "" for i in range(st.session_state.num_sections)}

# Generate Video
if st.button("Create Video"):
    video_clips = []
    audio_path = generate_audio_from_script(st.session_state.master_script)
    
    if not audio_path:
        st.error("Failed to generate audio for the script.")
        st.stop()

    audio = AudioFileClip(audio_path)
    total_audio_duration = audio.duration
    total_sections = st.session_state.num_sections + 2  # Include intro and outro
    section_duration = total_audio_duration / total_sections

    intro_img = download_image_from_url(INTRO_IMAGE_URL)
    if intro_img:
        intro_path = tempfile.mktemp(suffix=".png")
        intro_img.save(intro_path, "PNG")
        video_clips.append(ImageClip(intro_path).set_duration(section_duration).set_fps(24))

    for i, content in enumerate(st.session_state.sections):
        img_url = st.session_state.section_images.get(i)
        if img_url:
            image = download_image_from_url(img_url)
            if image:
                img_path = tempfile.mktemp(suffix=".png")
                image.save(img_path, "PNG")
                video_clips.append(ImageClip(img_path).set_duration(section_duration).set_fps(24))

    final_video_path = tempfile.mktemp(suffix=".mp4")
    combined_clip = concatenate_videoclips(video_clips, method="compose").set_audio(audio)
    combined_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac", fps=24)

    st.video(final_video_path)
    st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
