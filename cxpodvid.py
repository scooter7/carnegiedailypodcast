# Standard Python and library imports
import os
import requests
import json
import streamlit as st
from io import BytesIO
from pydub import AudioSegment
from moviepy.editor import concatenate_videoclips, ImageClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tempfile
import textwrap
import openai
from elevenlabs.client import ElevenLabs

# Load environment variables
openai.api_key = st.secrets["OPENAI_API_KEY"]
elevenlabs_client = ElevenLabs(api_key=st.secrets["ELEVENLABS_API_KEY"])

# Global configurations
font_path = "Arial.ttf"
font_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/Arial.ttf"
speaker_voice_map = {
    "Lisa": "Rachel",  # ElevenLabs voice ID for Lisa
    "Ali": "onyx"      # ElevenLabs voice ID for Ali
}
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, fact-based conversation between Ali and Lisa. 
Include statistics, insights, and natural conversation flow. 
Format the response strictly as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional explanations.
"""

# Ensure font is available
def setup_font(font_path, font_url):
    if not os.path.exists(font_path):
        response = requests.get(font_url)
        response.raise_for_status()
        with open(font_path, "wb") as f:
            f.write(response.content)
setup_font(font_path, font_url)

# Scrape images and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract images
        images = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        images = [download_image(img_url) for img_url in images[:5] if img_url.lower().endswith(("png", "jpg", "jpeg"))]

        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        return images, text[:5000]
    except Exception as e:
        st.error(f"Error scraping content from {url}: {e}")
        return [], ""

# Download and save an image locally
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

# Add text overlay to an image
def add_text_overlay(image_path, text, output_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=30)
        wrapped_text = textwrap.fill(text, width=40)
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x_start = 20
        y_start = img.height - text_height - 30
        background = Image.new("RGBA", img.size, (255, 255, 255, 0))
        background_draw = ImageDraw.Draw(background)
        background_draw.rectangle([(x_start - 10, y_start - 10), (x_start + text_width + 10, y_start + text_height + 10)], fill=(0, 0, 0, 128))
        img = Image.alpha_composite(img, background)
        draw.text((x_start, y_start), wrapped_text, font=font, fill="white")
        img.convert("RGB").save(output_path, "JPEG")
        return output_path
    except Exception as e:
        st.error(f"Failed to add text overlay: {e}")
        return None

# Generate a video from images and script
def create_video(images, script, duration_seconds):
    clips = []
    segment_duration = duration_seconds / len(script) if script else 0
    for image, part in zip(images, script):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        if add_text_overlay(image, part["text"], output_path):
            img_clip = ImageClip(output_path).set_duration(segment_duration)
            clips.append(img_clip)
        else:
            st.warning(f"Failed to process image or text for: {part['text']}")

    if clips:
        video_file = "video_short.mp4"
        final_video = concatenate_videoclips(clips, method="compose")
        final_video.write_videofile(video_file, codec="libx264", fps=24)
        return video_file
    else:
        st.error("No video clips could be created.")
        return None

# Synthesize speech
def synthesize_cloned_voice(text, speaker):
    try:
        audio_generator = elevenlabs_client.generate(text=text, voice=speaker_voice_map[speaker], model="eleven_multilingual_v2")
        audio_content = b"".join(audio_generator)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio_content)
            temp_audio_path = temp_audio_file.name
        return AudioSegment.from_file(temp_audio_path, format="mp3")
    except Exception as e:
        st.error(f"Error synthesizing speech for {speaker}: {e}")
        return None

# Summarize text
def summarize_content(input_text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize this text for a podcast conversation."},
                {"role": "user", "content": input_text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error summarizing content: {e}")
        return ""

# Generate conversation script
def generate_script(summary):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summary}
            ]
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return []

# Main Streamlit App
st.title("CX Podcast and Video Generator")
parent_url = st.text_input("Enter the URL of the page:")
duration = st.slider("Select Video Duration (seconds)", min_value=15, max_value=120, step=15)

if st.button("Generate Podcast and Video"):
    if parent_url.strip():
        # Scrape images and text
        images, scraped_text = scrape_images_and_text(parent_url)
        if images and scraped_text:
            # Summarize text and generate script
            summary = summarize_content(scraped_text)
            if summary:
                script = generate_script(summary)
                if script:
                    # Synthesize audio
                    audio_segments = [synthesize_cloned_voice(part["text"], part["speaker"]) for part in script]
                    combined_audio = sum(audio_segments, AudioSegment.empty())
                    podcast_file = "podcast.mp3"
                    combined_audio.export(podcast_file, format="mp3")
                    st.audio(podcast_file)
                    st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")

                    # Create video
                    video_file = create_video(images, script, duration)
                    if video_file:
                        st.video(video_file)
                        st.download_button("Download Video", open(video_file, "rb"), file_name="video_short.mp4")
                else:
                    st.error("Failed to generate the podcast script.")
            else:
                st.error("Failed to summarize content.")
        else:
            st.error("Failed to scrape content.")
