# Standard Python and library imports
import sys
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
import tempfile
import json
from moviepy.editor import ImageClip, concatenate_videoclips
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap

# Load environment variables
load_dotenv()

# Set API keys (Streamlit secrets or local .env)
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
elevenlabs_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY") or st.secrets["ELEVENLABS_API_KEY"]
)

# Configure speaker voices
speaker_voice_map = {
    "Lisa": "Rachel",
    "Ali": "NYy9s57OPECPcDJavL3T"  # Replace with the ID of your cloned voice
}

# Font file URL and local path
font_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/Arial.ttf"
local_font_path = "Arial.ttf"

# Function to download the font
def download_font(font_url, local_path):
    if not os.path.exists(local_path):
        try:
            response = requests.get(font_url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
            st.info("Font file downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download font file: {e}")
            raise e

download_font(font_url, local_font_path)

# Calculate maximum words for the selected duration
def max_words_for_duration(duration_seconds):
    wpm = 150  # Average words per minute
    return int((duration_seconds / 60) * wpm)

# Validate and filter valid image URLs
def filter_valid_images(image_urls, max_images=5):
    valid_images = []
    for url in image_urls[:max_images]:
        if url.lower().endswith(("png", "jpg", "jpeg", "webp")):
            valid_images.append(url)
    return valid_images

# Download and save an image locally, handling RGBA to RGB conversion
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        img.save(temp_path, "JPEG")
        return temp_path
    except Exception as e:
        st.warning(f"Failed to download image: {url}. Error: {e}")
        return None

# Scrape images and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        images = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        images = filter_valid_images(images)
        images = [download_image(img_url) for img_url in images if download_image(img_url)]

        text = soup.get_text(separator=" ", strip=True)
        return images, text[:5000]
    except Exception as e:
        st.error(f"Error scraping content from {url}: {e}")
        return [], ""

# Summarize content using OpenAI
def summarize_content(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following content into meaningful insights."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error summarizing content: {e}")
        return ""

# Generate podcast script with a word limit
def generate_script(enriched_text, max_words):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{system_prompt} The script should not exceed {max_words} words in total."},
                {"role": "user", "content": enriched_text}
            ]
        )
        raw_content = response.choices[0].message.content.strip()
        conversation_script = json.loads(raw_content)
        truncated_script = []
        total_words = 0
        for part in conversation_script:
            word_count = len(part["text"].split())
            if total_words + word_count <= max_words:
                truncated_script.append(part)
                total_words += word_count
            else:
                break
        return truncated_script
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return []

# Add text overlay to an image
def add_text_overlay(image_path, text, output_path, font_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=30)

        max_text_width = img.width - 40
        wrapped_text = textwrap.fill(text, width=40)

        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        x_start = 20
        y_start = img.height - text_height - 30

        background = Image.new("RGBA", img.size, (255, 255, 255, 0))
        background_draw = ImageDraw.Draw(background)
        background_draw.rectangle(
            [(x_start - 10, y_start - 10), (x_start + text_width + 10, y_start + text_height + 10)],
            fill=(0, 0, 0, 128)
        )
        img = Image.alpha_composite(img, background)
        draw.text((x_start, y_start), wrapped_text, font=font, fill="white")

        img.convert("RGB").save(output_path, "JPEG")
        return output_path
    except Exception as e:
        st.error(f"Failed to add text overlay: {e}")
        return None

# Create video using MoviePy
def create_video(images, script, duration_seconds):
    if not images or not script:
        st.error("No valid images or script provided. Cannot create video.")
        return None

    clips = []
    segment_duration = duration_seconds / len(script)

    for image, part in zip(images, script):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        if add_text_overlay(image, part["text"], output_path, local_font_path):
            clips.append(ImageClip(output_path).set_duration(segment_duration))

    if not clips:
        st.error("No video clips could be created.")
        return None

    video_file = "video_short.mp4"
    concatenate_videoclips(clips).write_videofile(video_file, codec="libx264", fps=24)
    return video_file

# Streamlit app interface
st.title("CX Podcast and Video Generator")
parent_url = st.text_input("Enter the URL of the page:")
duration = st.radio("Select Duration (seconds)", [15, 30, 45, 60], index=0)

if st.button("Generate Content"):
    if not parent_url.strip():
        st.error("Please enter a valid URL.")
    else:
        images, text = scrape_images_and_text(parent_url.strip())
        if text:
            summary = summarize_content(text)
            if summary:
                max_words = max_words_for_duration(duration)
                conversation_script = generate_script(summary, max_words)
                if conversation_script:
                    video_file = create_video(images, conversation_script, duration)
                    if video_file:
                        st.success("Video created successfully!")
                        st.video(video_file)
                    else:
                        st.error("Failed to create video.")
