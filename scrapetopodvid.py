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
        return AudioSegment.from_file(temp_audio_file.name)
    except Exception as e:
        logging.error(f"Error synthesizing speech for {speaker}: {e}")
        return None

# Add text overlay
def add_text_overlay(image_url, text, font_path):
    try:
        # Load the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        # Create overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 180))
        draw = ImageDraw.Draw(overlay)
        font_size = int(img.height * 0.05)
        font = ImageFont.truetype(font_path, size=font_size)

        # Wrap text to fit within image width
        max_chars_per_line = int(img.width / font_size * 1.2)
        wrapped_text = textwrap.fill(text, width=max_chars_per_line)

        # Calculate text size using textbbox
        lines = wrapped_text.split("\n")
        total_height = 0
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            height = bbox[3] - bbox[1]
            line_heights.append(height)
            total_height += height

        # Calculate position to center text at the bottom
        y_start = img.height - total_height - 20
        x_start = 20

        # Draw text on overlay
        for i, line in enumerate(lines):
            draw.text((x_start, y_start + sum(line_heights[:i])), line, font=font, fill="white")

        # Merge overlay with the original image
        img = Image.alpha_composite(img, overlay)
        return img

    except Exception as e:
        logging.error(f"Error adding text overlay: {e}")
        return None

# Create video
from moviepy.video.fx.all import fadein, fadeout
from moviepy.audio.fx.all import audio_fadein, audio_fadeout

def create_video(images, script, audio_segments, font_path, fade_duration=0.5, silence_duration=0.5):
    clips = []
    combined_audio = AudioSegment.empty()

    for img_url, part, audio in zip(images, script, audio_segments):
        silence = AudioSegment.silent(duration=silence_duration * 1000)
        audio_with_gap = silence + audio
        combined_audio += audio_with_gap

        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        audio_with_gap.export(temp_audio_path, format="mp3")

        # Add text overlay (pass font_path explicitly)
        overlay_image = add_text_overlay(img_url, part["text"], font_path)
        if overlay_image is None:
            continue

        temp_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        overlay_image.save(temp_img_path)

        # Combine image and audio into a clip
        audio_clip = AudioFileClip(temp_audio_path).fx(audio_fadein, fade_duration).fx(audio_fadeout, fade_duration)
        image_clip = ImageClip(temp_img_path, duration=audio_clip.duration).set_audio(audio_clip).set_fps(24)

        clips.append(image_clip)

    if not clips:
        raise ValueError("No valid video clips were created.")

    # Match video duration to the full audio
    total_audio_duration = combined_audio.duration_seconds
    current_video_duration = sum(clip.duration for clip in clips)
    if current_video_duration < total_audio_duration:
        extra_duration = total_audio_duration - current_video_duration
        clips[-1] = clips[-1].set_duration(clips[-1].duration + extra_duration)

    # Combine all clips
    final_video = concatenate_videoclips(clips, method="compose")
    final_video_path = "final_video.mp4"
    final_video.write_videofile(final_video_path, codec="libx264", fps=24, audio_codec="aac")

    # Save the combined podcast audio
    podcast_file = "final_podcast.mp3"
    combined_audio.export(podcast_file, format="mp3")

    return final_video_path, podcast_file

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
            with st.spinner("Synthesizing audio..."):
                audio_segments = [synthesize_cloned_voice(part["text"], part["speaker"]) for part in script]
            
            with st.spinner("Creating video..."):
                video, podcast = create_video(valid_images, script, audio_segments, font_path)
                
                script_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
                with open(script_file, "w") as f:
                    json.dump(script, f, indent=4)
                
                st.video(video)
                st.download_button("Download Podcast", open(podcast, "rb"), file_name="podcast.mp3")
                st.download_button("Download Script", open(script_file, "rb"), file_name="script.json")
        else:
            st.error("Script generation failed.")
    else:
        st.error("No valid images or content found.")
