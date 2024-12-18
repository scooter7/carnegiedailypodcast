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

# Set API keys
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
    wpm = 150
    return int((duration_seconds / 60) * wpm)

# Filter valid images
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
    return valid_images

# Scrape images and text
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
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
    You are a podcast host for 'CX Overview.' Generate a robust, fact-based summary of the school at the scraped webpage narrated by Lisa. 
    Include location, campus type, accolades, and testimonials. End with 'more information can be found at collegexpress.com.'
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
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# Convert image to video
def image_to_video(image_path, duration=2):
    temp_video_path = tempfile.mktemp(suffix=".mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loop", "1", "-i", image_path,
                "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p", temp_video_path
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting image to video: {e}")
        raise
    return temp_video_path

# Add fade effect for transitions
def add_fade_effect(previous_video, next_image_url, duration):
    temp_fade_video = tempfile.mktemp(suffix="_fade.mp4")
    next_image_path = download_image(next_image_url)
    next_video_path = image_to_video(next_image_path, duration=duration)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", previous_video, "-i", next_video_path,
            "-filter_complex", f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset=0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_fade_video
        ], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error in fade effect: {e}")
        raise
    return temp_fade_video

def generate_video_clip(image_url, duration, text=None, filter_option="None"):
    temp_video = tempfile.mktemp(suffix=".mp4")
    vf_filters = {
        "None": "",
        "Grayscale": "format=gray",
        "Sepia": "colorchannelmixer=.393:.769:.189:.349:.686:.168:.272:.534:.131"
    }
    vf = vf_filters.get(filter_option, "")
    overlay_path = add_text_overlay(image_url, text) if text else download_image(image_url)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", overlay_path,
            "-vf", vf if vf else "scale=1280:720", "-c:v", "libx264", "-t", str(duration), "-pix_fmt", "yuv420p", temp_video
        ], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e}")
        raise
    return temp_video

# Combine videos and audio
def create_final_video(logo_url, images, script, audio_path, duration, filter_option, add_text_overlay_flag, transition_option):
    temp_videos = [generate_video_clip(logo_url, 5, None, filter_option)]
    split_texts = textwrap.wrap(script, width=250)[:len(images)]
    for i, img_url in enumerate(images):
        if i > 0:
            if transition_option == "Fade":
                transition_video = add_fade_effect(temp_videos[-1], img_url, 2)
            elif transition_option == "Zoom":
                transition_video = add_zoom_effect(images[i - 1], 2)
            else:
                transition_video = None
            if transition_video:
                temp_videos.append(transition_video)
        temp_videos.append(generate_video_clip(img_url, duration // len(images), split_texts[i] if add_text_overlay_flag else None, filter_option))
    temp_videos.append(generate_video_clip("https://raw.githubusercontent.com/scooter7/carnegiedailypodcast/main/cx.jpg", 5, None, filter_option))
    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for video in temp_videos:
            f.write(f"file '{video}'\n")
    final_video = "final_video.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-i", audio_path,
            "-c:v", "libx264", "-c:a", "aac", "-shortest", final_video
        ], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error in final video creation: {e}")
        raise
    return final_video

# Streamlit Interface
st.title("CX Overview Podcast Generator")
url_input = st.text_input("Enter the webpage URL:")
logo_url = st.text_input("Enter the logo image URL:")
add_text_overlay_flag = st.checkbox("Add text overlays to images")
filter_option = st.selectbox("Select a Video Filter:", ["None", "Grayscale", "Sepia"])
transition_option = st.selectbox("Select Transition Effect:", ["None", "Fade", "Zoom"])
duration = st.radio("Video Duration (seconds):", [30, 45, 60])

if st.button("Generate Podcast"):
    images, text = scrape_images_and_text(url_input)
    valid_images = filter_valid_images(images)
    if valid_images and text:
        script = generate_script(text, max_words_for_duration(duration))
        audio = generate_audio_with_openai(script)
        if audio:
            final_video = create_final_video(logo_url, valid_images, script, audio, duration, filter_option, add_text_overlay_flag, transition_option)
            st.video(final_video)
            st.download_button("Download Video", open(final_video, "rb"), "CX_Overview.mp4")
            st.download_button("Download Script", script, "script.txt")
    else:
        st.error("No valid content found!")
