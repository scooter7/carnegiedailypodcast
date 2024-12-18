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

# Helper function to download an image
def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        temp_img_path = tempfile.mktemp(suffix=".jpg")
        with open(temp_img_path, "wb") as f:
            f.write(response.content)
        return temp_img_path
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        return None

# Add text overlay
def add_text_overlay(image_path, text):
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(local_font_path, size=28)
        lines = textwrap.wrap(text, width=40)
        y = img.height - (len(lines) * 35) - 20
        for line in lines:
            text_width = draw.textlength(line, font=font)
            draw.text(((img.width - text_width) // 2, y), line, font=font, fill="white")
            y += 35
        temp_img_path = tempfile.mktemp(suffix=".png")
        img.save(temp_img_path)
        return temp_img_path
    except Exception as e:
        logging.error(f"Failed to add text overlay: {e}")
        return None

# Generate video clip
def generate_video_clip(image_url, duration, text=None, filter_chain="", transition_chain=""):
    try:
        # Download the image locally
        local_image_path = download_image(image_url)
        if not local_image_path:
            raise ValueError("Failed to download the image.")

        # Add text overlay if needed
        if text:
            overlay_image = add_text_overlay(local_image_path, text)
            if overlay_image:
                local_image_path = overlay_image

        temp_video = tempfile.mktemp(suffix=".mp4")
        vf_chain = ",".join(filter(None, [filter_chain, transition_chain]))
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", local_image_path, "-t", str(duration),
            "-vf", vf_chain, "-c:v", "libx264", "-pix_fmt", "yuv420p", temp_video
        ], check=True)
        return temp_video
    except Exception as e:
        logging.error(f"Error generating video clip: {e}")
        return None

# Combine videos and audio
def create_final_video(logo_url, images, script, audio_path, duration, filter_option, transition, add_text_overlay_flag):
    FILTERS = {"None": "", "Grayscale": "format=gray", "Sepia": "colorchannelmixer=.393:.769:.189:.349:.686:.168:.272:.534:.131"}
    TRANSITIONS = {"None": "", "Fade": "fade=t=in:st=0:d=1", "Zoom": "zoompan=z='zoom+0.01':d=25"}

    temp_videos = [generate_video_clip(logo_url, 5, None, FILTERS[filter_option], TRANSITIONS["None"])]
    split_texts = textwrap.wrap(script, width=250)
    per_image_duration = duration // (len(images) + 2)

    for idx, img_url in enumerate(images):
        text = split_texts[idx] if idx < len(split_texts) else None
        temp_videos.append(generate_video_clip(img_url, per_image_duration, text if add_text_overlay_flag else None, FILTERS[filter_option], TRANSITIONS[transition]))

    temp_videos.append(generate_video_clip("https://raw.githubusercontent.com/scooter7/carnegiedailypodcast/main/cx.jpg", 5, None, FILTERS[filter_option], TRANSITIONS["None"]))

    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for video in temp_videos:
            f.write(f"file '{video}'\n")

    final_video = "final_video.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-i", audio_path,
        "-c:v", "libx264", "-c:a", "aac", "-shortest", final_video
    ], check=True)
    return final_video

# Streamlit Interface
st.title("CX Overview Podcast Generator")
url_input = st.text_input("Enter the webpage URL:")
logo_url = st.text_input("Enter the logo image URL:")
add_text_overlay_flag = st.checkbox("Add Text Overlays to Images")
filter_option = st.selectbox("Select a Video Filter:", ["None", "Grayscale", "Sepia"])
transition_option = st.selectbox("Select Image Transition:", ["None", "Fade", "Zoom"])
duration = st.radio("Video Duration (seconds):", [30, 45, 60])

if st.button("Generate Podcast"):
    images, text = scrape_images_and_text(url_input)
    valid_images = filter_valid_images(images)
    if valid_images and text:
        script = generate_script(text, max_words_for_duration(duration))
        audio = generate_audio_with_openai(script)
        if audio:
            final_video = create_final_video(logo_url, valid_images, script, audio, duration, filter_option, transition_option, add_text_overlay_flag)
            st.video(final_video)
            st.download_button("Download Video", open(final_video, "rb"), "CX_Overview.mp4")
            st.download_button("Download Script", script, "script.txt")
    else:
        st.error("No valid content found!")
