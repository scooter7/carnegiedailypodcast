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
    wpm = 150  # Words per minute
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
    Include statistics, location, campus type, and accolades. End with 'more info at collegexpress.com'.
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

# Apply video transitions and filters
def generate_video_clip(image_url, duration, filter_option, transition_option):
    temp_video = tempfile.mktemp(suffix=".mp4")
    filter_command = {"None": "", "Grayscale": "format=gray", "Sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131"}
    transition_command = {"Fade": "fade=t=in:st=0:d=1", "Slide": "zoompan=z='zoom+0.05':d=25", "None": ""}

    filter_str = filter_command.get(filter_option, "")
    transition_str = transition_command.get(transition_option, "")
    
    vf_filter = ",".join([f for f in [filter_str, transition_str] if f]) or "scale=iw:ih"

    try:
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", image_url,
            "-vf", vf_filter, "-c:v", "libx264", "-t", str(duration),
            "-pix_fmt", "yuv420p", temp_video
        ], check=True)
        return temp_video
    except Exception as e:
        logging.error(f"Error applying filter/transition: {e}")
        return None

# Create video with audio and effects
def create_video_with_audio(logo_url, images, script, audio_path, duration, filter_option, transition_option):
    temp_videos = []

    # Add logo as the first video clip
    temp_videos.append(generate_video_clip(logo_url, 5, filter_option, transition_option))

    # Generate clips for each image
    split_texts = textwrap.wrap(script, width=250)[:len(images)]
    for img_url, text in zip(images, split_texts):
        temp_videos.append(generate_video_clip(img_url, duration // len(images), filter_option, transition_option))

    # Add static end image
    end_image_url = "https://raw.githubusercontent.com/scooter7/carnegiedailypodcast/main/cx.jpg"
    temp_videos.append(generate_video_clip(end_image_url, 5, filter_option, transition_option))

    # Concatenate all videos and add audio
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

# Streamlit App
st.title("CX Overview Podcast Generator")
url_input = st.text_input("Enter the URL to scrape text and images:")
logo_url = st.text_input("Enter the URL for the logo image:")
duration = st.radio("Select Duration (seconds):", [30, 45, 60])
filter_option = st.selectbox("Select a Video Filter:", ["None", "Grayscale", "Sepia"])
transition_option = st.selectbox("Select a Transition Effect:", ["None", "Fade", "Slide"])

if st.button("Generate Content"):
    images, text = scrape_images_and_text(url_input)
    images = filter_valid_images(images)
    if not images or not text or not logo_url:
        st.error("Invalid inputs or no content found.")
    else:
        max_words = max_words_for_duration(duration)
        script = generate_script(text, max_words)
        audio_path = generate_audio_with_openai(script)
        video_file = create_video_with_audio(logo_url, images, script, audio_path, duration, filter_option, transition_option)
        if video_file:
            st.video(video_file)
            st.download_button("Download Video", open(video_file, "rb"), "CX_Overview.mp4")
            st.download_button("Download Script", script, "script.txt")
