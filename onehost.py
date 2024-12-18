# Standard Python and library imports
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
import openai
import av  # PyAV for FFmpeg-based video processing
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import logging
import numpy as np
import subprocess
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Set OpenAI API key
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

# Add text overlay to an image
def add_text_overlay(image_url, text, font_path):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=28)
        lines = textwrap.wrap(text, width=40)

        # Draw background box
        box_height = len(lines) * font.size + 20
        draw.rectangle([(0, img.height - box_height), (img.width, img.height)], fill=(0, 0, 0, 128))

        # Draw text
        y = img.height - box_height + 10
        for line in lines:
            text_width = draw.textlength(line, font=font)
            draw.text(((img.width - text_width) // 2, y), line, font=font, fill="white")
            y += font.size

        # Save temporary image
        temp_img_path = tempfile.mktemp(suffix=".png")
        img.save(temp_img_path)
        return temp_img_path
    except Exception as e:
        logging.error(f"Failed to add text overlay: {e}")
        return None

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
        logging.error(f"Error generating audio with OpenAI TTS: {e}")
        return None

# Generate script with OpenAI
def generate_script(text, max_words):
    try:
        system_prompt = """
        You are a podcast narrator for 'CX Overview.' Generate a robust, fact-based, news-oriented narrative 
        about the given topic. Include relevant statistics, insights, and accolades if available. 
        Make it engaging and conversational. The script should be no longer than the specified word count.
        Format the response as plain text without any additional formatting.
        """
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"The content: {text} \nLimit: {max_words} words"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating script: {e}")
        return ""

# Create video using PyAV and FFmpeg
def create_video_with_audio(images, text, audio_path, add_text_overlay):
    try:
        temp_video_files = []

        # Split the text into smaller parts for each image
        split_texts = textwrap.wrap(text, width=250)[:len(images)]
        for idx, (image_url, split_text) in enumerate(zip(images, split_texts)):
            # Add text overlay
            image_path = add_text_overlay(image_url, split_text, local_font_path)

            # Generate silent video segment
            temp_video = tempfile.mktemp(suffix=".mp4")
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-c:v", "libx264", 
                "-t", "5", "-pix_fmt", "yuv420p", temp_video
            ], check=True)
            temp_video_files.append(temp_video)

        # Concatenate videos and add audio
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, "w") as f:
            for video in temp_video_files:
                f.write(f"file '{video}'\n")

        final_video = "final_video_pyav.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", audio_path, "-c:v", "libx264", "-c:a", "aac", "-shortest", final_video
        ], check=True)

        return final_video
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        return None

# Streamlit App Interface
st.title("CX Podcast and Video Generator")

# Input for URL
url_input = st.text_input("Enter the URL of the page to scrape text and images:")
add_text_overlay = st.checkbox("Add text overlays to video", value=True)

# Button to trigger generation
if st.button("Generate Content"):
    if not url_input.strip():
        st.error("Please enter a valid URL.")
    else:
        # Simulate scraped content for demo purposes
        images = [
            "https://via.placeholder.com/1280x720.png?text=Image1",
            "https://via.placeholder.com/1280x720.png?text=Image2",
            "https://via.placeholder.com/1280x720.png?text=Image3"
        ]
        sample_text = "Welcome to the CX Overview! Today we explore the most exciting colleges in the nation."

        # Generate script using OpenAI
        max_words = 200
        script = generate_script(sample_text, max_words)
        if not script:
            st.error("Failed to generate script.")
        else:
            # Generate full audio using OpenAI TTS
            audio_path = generate_audio_with_openai(script)
            if not audio_path:
                st.error("Failed to generate audio.")
            else:
                # Create video with PyAV
                video_file = create_video_with_audio(images, script, audio_path, add_text_overlay)
                if video_file:
                    st.video(video_file)
                    st.download_button("Download Video", open(video_file, "rb"), "cx_overview_video.mp4")
                else:
                    st.error("Failed to create video.")
