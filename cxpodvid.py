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
from moviepy.editor import ImageClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
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
    "Ali": "NYy9s57OPECPcDJavL3T"
}

# System prompt for script generation
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, conversational dialogue between Ali and Lisa. 
Include relevant statistics and insights. Format the response strictly as a JSON array of objects with 'speaker' and 'text' keys.
"""

# Download font file for text overlay
font_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/Arial.ttf"
local_font_path = "Arial.ttf"

def download_font(font_url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(font_url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
download_font(font_url, local_font_path)

# Word limit based on duration
def max_words_for_duration(duration_seconds):
    wpm = 150  # Words per minute
    return int((duration_seconds / 60) * wpm)

# Scrape images and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        images = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        images = [download_image(img_url) for img_url in images if download_image(img_url)]
        text = soup.get_text(separator=" ", strip=True)
        return images, text[:5000]
    except Exception as e:
        st.error(f"Error scraping content from {url}: {e}")
        return [], ""

# Download and process image
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp_img.name, format="JPEG")
        return temp_img.name
    except Exception as e:
        st.warning(f"Failed to download image: {url}. Error: {e}")
        return None

# Summarize content using OpenAI
def summarize_content(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following content into key points."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error summarizing content: {e}")
        return ""

# Generate script using OpenAI
def generate_script(summary, max_words):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{system_prompt} The script should not exceed {max_words} words."},
                {"role": "user", "content": summary}
            ]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return []

# Synthesize speech with ElevenLabs
def synthesize_cloned_voice(text, speaker):
    try:
        audio_generator = elevenlabs_client.generate(
            text=text,
            voice=speaker_voice_map[speaker],
            model="eleven_multilingual_v2"
        )
        audio_content = b"".join(audio_generator)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_audio_file.name, "wb") as f:
            f.write(audio_content)
        return temp_audio_file.name
    except Exception as e:
        st.error(f"Error synthesizing speech for {speaker}: {e}")
        return None

# Add text overlay to an image
def add_text_overlay(image_path, text, output_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(local_font_path, size=30)
        wrapped_text = textwrap.fill(text, width=40)
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)

        x_start = 20
        y_start = img.height - (text_bbox[3] - text_bbox[1]) - 30

        # Add background rectangle for text
        background = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(background)
        draw.rectangle(
            [(x_start - 10, y_start - 10), (x_start + text_bbox[2] + 10, y_start + text_bbox[3] + 10)],
            fill=(0, 0, 0, 128)
        )
        img = Image.alpha_composite(img, background)
        draw = ImageDraw.Draw(img)
        draw.text((x_start, y_start), wrapped_text, font=font, fill="white")
        img.convert("RGB").save(output_path, "JPEG")
        return output_path
    except Exception as e:
        st.error(f"Failed to add text overlay: {e}")
        return None

# Create video using MoviePy
def create_video(images, script, audio_files, duration_seconds):
    if not images or not script or not audio_files:
        st.error("Insufficient data to create video.")
        return None

    clips = []
    segment_duration = duration_seconds / len(script)

    for image, part, audio_file in zip(images, script, audio_files):
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        if add_text_overlay(image, part["text"], temp_image_path):
            img_clip = ImageClip(temp_image_path).set_duration(segment_duration)
            audio_clip = AudioFileClip(audio_file).subclip(0, segment_duration)
            img_clip = img_clip.set_audio(audio_clip)
            clips.append(img_clip)

    if clips:
        video_file = "final_video.mp4"
        final_video = concatenate_videoclips(clips)
        final_video.write_videofile(video_file, codec="libx264", fps=24)
        return video_file
    else:
        st.error("No video clips created.")
        return None

# Streamlit app interface
st.title("CX Podcast & Video Generator")
url = st.text_input("Enter the URL of the page:")
duration = st.radio("Select Video Duration (seconds)", [15, 30, 45, 60], index=0)

if st.button("Generate Content"):
    if url.strip():
        images, text = scrape_images_and_text(url)
        if images and text:
            summary = summarize_content(text)
            if summary:
                max_words = max_words_for_duration(duration)
                script = generate_script(summary, max_words)
                if script:
                    audio_files = [synthesize_cloned_voice(part["text"], part["speaker"]) for part in script]
                    video_file = create_video(images, script, audio_files, duration)
                    if video_file:
                        st.video(video_file)
                        st.download_button("Download Video", open(video_file, "rb"), file_name="final_video.mp4")
                else:
                    st.error("Failed to generate script.")
            else:
                st.error("Failed to summarize text.")
        else:
            st.error("Failed to scrape content.")
    else:
        st.error("Please enter a valid URL.")
