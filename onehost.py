# Standard Python and library imports
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
import openai
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import logging
import subprocess
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Set API keys (Streamlit secrets or local .env)
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

# Font file for text overlay
font_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/Arial.ttf"
local_font_path = "Arial.ttf"

# Download font file
def download_font(font_url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(font_url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
download_font(font_url, local_font_path)

# Calculate word limit based on duration
def max_words_for_duration(duration_seconds):
    wpm = 150  # Words per minute
    return int((duration_seconds / 60) * wpm)

# Filter valid image formats and URLs
def filter_valid_images(image_urls, min_width=400, min_height=300):
    valid_images = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            if img.width >= min_width and img.height >= min_height and img.mode in ["RGB", "RGBA"]:
                valid_images.append(url)
        except Exception as e:
            logging.warning(f"Invalid image {url}: {e}")
    logging.info(f"Valid images found: {len(valid_images)}")
    return valid_images

# Scrape images and text
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        image_urls = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        text = soup.get_text(separator=" ", strip=True)
        return image_urls, text[:5000]
    except Exception as e:
        logging.error(f"Error scraping content from {url}: {e}")
        return [], ""

# Generate script using OpenAI
def generate_script(text, max_words):
    system_prompt = """
    You are a podcast host for 'CX Overview.' Generate a robust, fact-based summary of the school at the scraped webpage narrated by Lisa. Make sure that the voices are excited and enthusiastic, not flat and overly matter-of-fact. 
    Include relevant statistics, facts, and insights based on the summaries. Every podcast should include information about the school's location (city, state) and type of campus (urban, rural, suburban, beach, mountains, etc.). Include accolades and testimonials if they are available, but do not make them up if not available. 
    The narration should feel conversational and engaging, with occasional natural pauses and fillers like 'um,' and 'you know.' Whenever you discuss a faculty-to-student ratio like 14:1, pronounce it as 14 to 1. At the end of the podcast, always mention that more information about the school can be found at collegexpress.com. 
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this text: {text} Limit: {max_words} words."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating script: {e}")
        return ""

# Generate speech using OpenAI TTS
def generate_audio_with_openai(text, voice="alloy"):
    try:
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as audio_file:
            audio_file.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# Add static end image
def add_static_end_image(end_image_url, duration=5):
    try:
        response = requests.get(end_image_url, timeout=10)
        response.raise_for_status()

        temp_image_path = tempfile.mktemp(suffix=".jpg")
        with open(temp_image_path, "wb") as img_file:
            img_file.write(response.content)

        temp_video_path = tempfile.mktemp(suffix=".mp4")
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", temp_image_path,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
            "-t", str(duration), "-pix_fmt", "yuv420p", temp_video_path
        ], check=True)
        return temp_video_path
    except Exception as e:
        logging.error(f"Error generating static end image video: {e}")
        return None

# Create video with optional text overlay
def create_video_with_audio(logo_url, images, script, audio_path, add_text_overlay_flag):
    try:
        temp_videos = []

        # Add logo video
        logo_video = add_static_end_image(logo_url, duration=5)
        temp_videos.append(logo_video)

        # Add main content
        split_texts = textwrap.wrap(script, width=250)[:len(images)]
        for img_url, text in zip(images, split_texts):
            img_path = add_static_end_image(img_url, duration=5)
            temp_videos.append(img_path)

        # Add end image
        end_video = add_static_end_image("https://raw.githubusercontent.com/scooter7/carnegiedailypodcast/main/cx.jpg", 5)
        temp_videos.append(end_video)

        # Concatenate all videos
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, "w") as f:
            for video in temp_videos:
                f.write(f"file '{video}'\n")

        final_video = "final_video.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", audio_path, "-c:v", "libx264", "-c:a", "aac", "-shortest", final_video
        ], check=True)
        return final_video
    except Exception as e:
        logging.error(f"Error creating final video: {e}")
        return None

# Streamlit Interface
st.title("CX Overview Podcast Generator")
url_input = st.text_input("Enter the webpage URL:")
logo_url = st.text_input("Enter the logo image URL:")
add_text_overlay_flag = st.checkbox("Add text overlays to images")
duration = st.radio("Video duration:", [30, 45, 60])

if st.button("Generate Podcast"):
    images, text = scrape_images_and_text(url_input)
    valid_images = filter_valid_images(images)
    if valid_images and text:
        script = generate_script(text, max_words_for_duration(duration))
        audio = generate_audio_with_openai(script)
        if audio:
            final_video = create_video_with_audio(logo_url, valid_images, script, audio, add_text_overlay_flag)
            if final_video:
                st.video(final_video)
                st.download_button("Download Video", open(final_video, "rb"), "CX_Overview.mp4")
