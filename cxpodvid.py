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

# Load environment variables
load_dotenv()

# Set API keys (Streamlit secrets or local .env)
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
elevenlabs_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY") or st.secrets["ELEVENLABS_API_KEY"]
)

# Speaker voices configuration
speaker_voice_map = {
    "Lisa": "Rachel",
    "Ali": "NYy9s57OPECPcDJavL3T"
}

# Font for image overlay
font_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/Arial.ttf"
local_font_path = "Arial.ttf"

# Download font if not locally available
def download_font(font_url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(font_url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)
download_font(font_url, local_font_path)

# Word limit calculation based on duration
def max_words_for_duration(duration_seconds):
    wpm = 150
    return int((duration_seconds / 60) * wpm)

# Download and process valid images
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp_img.name, format="JPEG")
        return temp_img.name
    except Exception as e:
        st.warning(f"Failed to download or process image: {url}. Error: {e}")
        return None

def filter_valid_images(image_urls, max_images=5):
    valid_images = []
    for url in image_urls[:max_images]:
        if url.lower().endswith(("svg", "bmp")):
            st.warning(f"Skipping unsupported image format: {url}")
            continue
        img_path = download_image(url)
        if img_path:
            valid_images.append(img_path)
    return valid_images

def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        image_urls = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        valid_images = filter_valid_images(image_urls)

        text = soup.get_text(separator=" ", strip=True)
        return valid_images, text[:5000]
    except Exception as e:
        st.error(f"Error scraping content from {url}: {e}")
        return [], ""

# Summarize content
def summarize_content(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following content for podcast creation."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error summarizing content: {e}")
        return ""

# Generate script
def generate_script(summary, max_words):
    system_prompt = """
    You are a podcast host for 'CX Overview.' Generate an engaging, fact-based conversation between Ali and Lisa.
    Format the response strictly as a JSON array of objects, each with 'speaker' and 'text' keys.
    """
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

# Synthesize speech
def synthesize_cloned_voice(text, speaker):
    try:
        audio_generator = elevenlabs_client.generate(
            text=text,
            voice=speaker_voice_map[speaker],
            model="eleven_multilingual_v2"
        )
        audio_content = b"".join(audio_generator)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_audio.name, "wb") as f:
            f.write(audio_content)
        return temp_audio.name
    except Exception as e:
        st.error(f"Error synthesizing voice for {speaker}: {e}")
        return None

# Overlay text onto images
def add_text_overlay(image_path, text, output_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(local_font_path, size=30)
        wrapped_text = textwrap.fill(text, width=40)

        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        x_start = 20
        y_start = img.height - (text_bbox[3] - text_bbox[1]) - 30

        background = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(background)
        draw.rectangle([(x_start - 10, y_start - 10), (x_start + text_bbox[2] + 10, y_start + text_bbox[3] + 10)],
                       fill=(0, 0, 0, 128))
        img = Image.alpha_composite(img, background)
        draw = ImageDraw.Draw(img)
        draw.text((x_start, y_start), wrapped_text, font=font, fill="white")
        img.convert("RGB").save(output_path, "JPEG")
        return output_path
    except Exception as e:
        st.error(f"Failed to add text overlay: {e}")
        return None

# Create video
def create_video(images, script, audio_files, duration_seconds):
    if not images or not script or not audio_files:
        st.error("Insufficient data to create video.")
        return None

    clips = []
    segment_duration = duration_seconds / len(script)
    for img_path, part, audio_path in zip(images, script, audio_files):
        overlay_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        add_text_overlay(img_path, part["text"], overlay_path)
        img_clip = ImageClip(overlay_path).set_duration(segment_duration)
        audio_clip = AudioFileClip(audio_path).subclip(0, segment_duration)
        img_clip = img_clip.set_audio(audio_clip)
        clips.append(img_clip)

    if clips:
        final_video = concatenate_videoclips(clips, method="compose")
        video_file = "final_video.mp4"
        final_video.write_videofile(video_file, codec="libx264", fps=24)
        return video_file
    else:
        st.error("No video clips created.")
        return None

# Streamlit interface
st.title("Podcast & Video Generator")
url = st.text_input("Enter the URL of the page:")
duration = st.radio("Select Duration (seconds)", [15, 30, 45, 60])

if st.button("Generate Content"):
    if url.strip():
        images, text = scrape_images_and_text(url)
        if images and text:
            summary = summarize_content(text)
            max_words = max_words_for_duration(duration)
            script = generate_script(summary, max_words)
            if script:
                audio_files = [synthesize_cloned_voice(part["text"], part["speaker"]) for part in script]
                podcast_file = "podcast.mp3"
                combined_audio = sum([AudioSegment.from_file(audio) for audio in audio_files], AudioSegment.empty())
                combined_audio.export(podcast_file, format="mp3")
                st.success("Podcast created successfully!")
                st.audio(podcast_file)
                video_file = create_video(images, script, audio_files, duration)
                if video_file:
                    st.success("Video created successfully!")
                    st.video(video_file)
                    st.download_button("Download Video", open(video_file, "rb"), "final_video.mp4")
