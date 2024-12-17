import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import json
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

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
    "Ali": "NYy9s57OPECPcDJavL3T"
}

# System prompt for script generation
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, news-oriented conversation between Ali and Lisa. Make sure that the voices are excited and enthusiastic, not flat and overly matter-of-fact.
Include relevant statistics, facts, and insights based on the summaries. Every podcast should include information about the school's location (city, state) and type of campus (urban, rural, suburban, beach, mountains, etc.). Include accolades and testimonials if they are available, but do not make them up if not available. When mentioning tuition, never make judgmental statements about the cost being high; instead, try to focus on financial aid and scholarship opportunities. 
The conversation should feel conversational and engaging, with occasional natural pauses and fillers like 'um,' and  'you know' (Do not overdo the pauses and fillers, though). Whenever you discuss a faculty-to-student ratio like 14:1, pronounce it as 14 to 1 (or whatever the applicable true number is). At the end of the podcast, always mention that more information about the school can be found at collegexpress.com.Make sure that, anytime, collegexpress is mentioned, it is pronounced as college express. However, at the end of the video, it should be spelled as collegexpress.

Format the response **strictly** as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text, explanations, or formatting.
"""

# Font file for text overlay
font_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/Arial.ttf"
local_font_path = "Arial.ttf"

# Download font file
@st.cache_resource
def download_font(font_url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(font_url)
        with open(local_path, "wb") as f:
            f.write(response.content)
    return local_path

local_font_path = download_font(font_url, local_font_path)

# Calculate word limit based on duration
def max_words_for_duration(duration_seconds):
    words_per_minute = 150
    return int((duration_seconds / 60) * words_per_minute)

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
            logging.warning(f"Skipping image {url}: {e}")
    return valid_images

# Extract logo URL
def extract_logo_url(soup):
    try:
        logo_div = soup.find("div", class_="client-logo")
        style_attr = logo_div["style"]
        start = style_attr.find("url('") + 5
        end = style_attr.find("')", start)
        logo_url = style_attr[start:end]
        if logo_url.startswith("https"):
            return logo_url
    except Exception as e:
        logging.warning("No valid logo found.")
    return None

# Scrape images and text
@st.cache_data
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        logo_url = extract_logo_url(soup)
        image_urls = [img["src"] for img in soup.find_all("img", src=True)]
        text = soup.get_text(separator=" ", strip=True)
        valid_images = filter_valid_images(image_urls)
        return logo_url, valid_images, text[:5000]
    except Exception as e:
        logging.error(f"Error scraping content: {e}")
        return None, [], ""

# Summarize content using OpenAI
@st.cache_data
def summarize_content(text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the content into key points."},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content.strip()

# Generate script using OpenAI
@st.cache_data
def generate_script(enriched_text, max_words):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{system_prompt} Max words: {max_words}"},
            {"role": "user", "content": enriched_text}
        ]
    )
    return json.loads(response.choices[0].message.content)

# Synthesize speech with ElevenLabs
@st.cache_data
def synthesize_cloned_voice(text, speaker):
    try:
        audio_generator = elevenlabs_client.generate(
            text=text, voice=speaker_voice_map[speaker], model="eleven_multilingual_v2"
        )
        audio_content = b"".join(audio_generator)
        return AudioSegment.from_file(BytesIO(audio_content), format="mp3")
    except Exception as e:
        logging.error(f"Error synthesizing speech: {e}")
        return None

# Combine audio with pacing
def combine_audio_with_pacing(script, audio_segments):
    combined_audio = AudioSegment.empty()
    for idx, audio in enumerate(audio_segments):
        combined_audio += audio + AudioSegment.silent(duration=500)
    return combined_audio

# Add text overlay to an image
def add_text_overlay(image_url, text, font_path):
    response = requests.get(image_url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=28)
    wrapped_text = textwrap.fill(text, width=40)
    draw.text((20, 20), wrapped_text, font=font, fill="white")
    return np.array(img)

# Create video with audio
def create_video(filtered_images, script, audio_segments, logo_url, add_overlay):
    """
    Create a video from images, script, and audio.
    """
    clips = []

    # Add logo as the first clip
    if logo_url:
        try:
            logo_image = Image.open(requests.get(logo_url, stream=True).raw)
            logo_clip = ImageClip(np.array(logo_image), duration=2).set_fps(24)
            clips.append(logo_clip)
        except Exception as e:
            logging.warning(f"Error loading logo image: {e}")

    # Process each image and audio
    for idx, (image_url, part, audio) in enumerate(zip(filtered_images, script, audio_segments)):
        try:
            # Add text overlay if enabled
            if add_overlay:
                img = add_text_overlay(image_url, part['text'], local_font_path)
                img = Image.fromarray(img)  # Convert NumPy array back to Pillow image
            else:
                img = Image.open(requests.get(image_url, stream=True).raw)

            temp_img_path = f"temp_image_{idx}.png"
            img.save(temp_img_path)  # Save the image directly

            # Save audio file
            temp_audio_path = f"audio_{idx}.mp3"
            audio.export(temp_audio_path, format="mp3")

            # Create a video clip
            clip = ImageClip(temp_img_path, duration=audio.duration_seconds).set_audio(AudioFileClip(temp_audio_path))
            clip = clip.set_fps(24)
            clips.append(clip)

        except Exception as e:
            logging.error(f"Error processing image {image_url}: {e}")

    # Concatenate all clips
    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile("final_video.mp4", codec="libx264", fps=24, audio_codec="aac")
        return "final_video.mp4"
    else:
        logging.error("No valid clips to combine.")
        return None

# Streamlit app interface
st.title("CX Podcast and Video Generator")

# URL input for scraping text and images
url_input = st.text_input("Enter the URL to scrape:")

# New field to allow manual entry of the logo URL
logo_url_input = st.text_input("Enter Logo URL (optional):")

# Duration selection
duration = st.radio("Duration (seconds):", [15, 30, 45, 60], index=0)

# Checkbox to toggle text overlays
add_overlay = st.checkbox("Add Text Overlays", value=True)

# Generate content button
if st.button("Generate Content"):
    if not url_input:
        st.error("Please provide a valid URL.")
    else:
        logo_url, images, text = scrape_images_and_text(url_input)
        logo_url = logo_url_input or logo_url
        filtered_images = images[:5]
        summary = summarize_content(text)
        max_words = max_words_for_duration(duration)
        script = generate_script(summary, max_words)
        audio_segments = [
            synthesize_cloned_voice(part['text'], part['speaker']) for part in script
        ]
        combined_audio = combine_audio_with_pacing(script, audio_segments)
        video_path = create_video(filtered_images, script, audio_segments, logo_url, add_overlay)
        st.video(video_path)
        st.download_button("Download Video", open(video_path, "rb"), file_name="final_video.mp4")
        podcast_path = "podcast.mp3"
        combined_audio.export(podcast_path, format="mp3")
        st.download_button("Download Podcast (MP3)", open(podcast_path, "rb"), file_name="podcast.mp3")
