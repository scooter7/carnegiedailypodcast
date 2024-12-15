# Standard Python and library imports
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
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import logging
import numpy as np
import cairosvg
from moviepy.video.fx.all import fadein, fadeout
from moviepy.audio.fx.all import audio_fadein, audio_fadeout

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# API Keys
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY") or st.secrets["ELEVENLABS_API_KEY"])

# Speaker voice map
speaker_voice_map = {"Lisa": "Rachel", "Ali": "NYy9s57OPECPcDJavL3T"}

# Font file
font_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/Arial.ttf"
font_path = "Arial.ttf"
if not os.path.exists(font_path):
    with open(font_path, "wb") as f:
        f.write(requests.get(font_url).content)

# System prompt for OpenAI
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, engaging conversation between Ali and Lisa based on provided summaries. 
Each podcast should include school offerings. Format strictly as JSON: [{"speaker": "Lisa", "text": "..."}, {"speaker": "Ali", "text": "..."}].
"""

# Scrape images and text
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        image_urls = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        for tag in soup.find_all(style=True):
            if "background-image" in tag["style"]:
                bg_url = tag["style"].split("url(")[1].split(")")[0].strip("'\"")
                image_urls.append(urljoin(url, bg_url))
        text = soup.get_text(separator=" ", strip=True)
        return list(set(image_urls)), text[:5000]
    except Exception as e:
        logging.error(f"Scraping failed: {e}")
        return [], ""

# Filter valid images
def filter_valid_images(image_urls):
    valid_images = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(cairosvg.svg2png(bytestring=response.content)) if url.endswith(".svg") else BytesIO(response.content))
            if img.width >= 300 and img.height >= 200:
                valid_images.append(url)
        except Exception:
            continue
    return valid_images

# Summarize content
def summarize_content(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Summarize into key points."}, {"role": "user", "content": text}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return ""

# Generate script
def generate_script(summary, max_words):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": f"{system_prompt} Max words: {max_words}."}, {"role": "user", "content": summary}]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        logging.error(f"Script generation failed: {e}")
        return []

# Synthesize speech
def synthesize_voice(text, speaker):
    audio = elevenlabs_client.generate(text=text, voice=speaker_voice_map[speaker], model="eleven_multilingual_v2")
    return AudioSegment.from_file(BytesIO(b"".join(audio)), format="mp3")

# Add text overlay
def add_text_overlay(image_url, text):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content)).convert("RGBA")
    font = ImageFont.truetype(font_path, size=int(img.height * 0.05))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 180))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, img.height - 100, img.width, img.height], fill=(0, 0, 0, 180))
    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    draw.text((20, img.height - 80), textwrap.fill(text, 40), font=font, fill="white")
    return img

# Create video
def create_video(images, script, audio_segments):
    clips = []
    combined_audio = AudioSegment.silent(duration=0)
    for img_url, part, audio in zip(images, script, audio_segments):
        combined_audio += audio + AudioSegment.silent(duration=500)
        img = add_text_overlay(img_url, part["text"])
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        img.save(temp_img)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        audio.export(temp_audio, format="mp3")
        audio_clip = AudioFileClip(temp_audio).fx(audio_fadein, 0.5).fx(audio_fadeout, 0.5)
        image_clip = ImageClip(temp_img, duration=audio_clip.duration).set_audio(audio_clip).set_fps(24)
        clips.append(image_clip)
    final_video = concatenate_videoclips(clips, method="compose")
    combined_audio.export("podcast.mp3", format="mp3")
    final_video.write_videofile("video.mp4", codec="libx264", fps=24, audio_codec="aac")
    return "video.mp4", "podcast.mp3"

# Streamlit App
st.title("Podcast and Video Generator")
url = st.text_input("Enter the webpage URL:")
duration = st.radio("Podcast Duration (seconds):", [15, 30, 45, 60])

if st.button("Generate Content"):
    with st.spinner("Scraping content..."):
        images, text = scrape_images_and_text(url)
        valid_images = filter_valid_images(images)
    if valid_images and text:
        with st.spinner("Summarizing content..."):
            summary = summarize_content(text)
        with st.spinner("Generating script..."):
            script = generate_script(summary, max_words=int(duration * 2.5))
        if script:
            audio_segments = [synthesize_voice(part["text"], part["speaker"]) for part in script]
            with st.spinner("Creating video..."):
                video, podcast = create_video(valid_images, script, audio_segments)
            st.video(video)
            st.download_button("Download Podcast", open(podcast, "rb"), file_name="podcast.mp3")
        else:
            st.error("Script generation failed.")
    else:
        st.error("No valid images or content found.")
