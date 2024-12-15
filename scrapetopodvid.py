
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
    "Ali": "NYy9s57OPECPcDJavL3T"  # Replace with your voice ID
}

# System prompt for script generation
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, news-oriented conversation between Ali and Lisa. 
Include relevant statistics, facts, and insights based on the summaries. Every podcast should include information about the school's offerings gleaned from the web page content. 
The conversation should feel conversational and engaging, with occasional natural pauses and fillers like 'um,' and  'you know' (Do not overdo the pauses and fillers, though). At the end of the podcast, always mention that more information about the school can be found at collegexpress.com.

Format the response **strictly** as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text, explanations, or formatting.
"""

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

# Scrape images and text from a URL
def scrape_images_and_text(url):
    """
    Scrape all image URLs and text from a webpage.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract image URLs, including background images in styles
        image_urls = []

        # Include <img> tags
        for img in soup.find_all("img", src=True):
            image_urls.append(urljoin(url, img["src"]))

        # Include background images from inline styles
        for style_tag in soup.find_all(style=True):
            style = style_tag.get("style")
            if "background-image" in style:
                bg_url = style.split("url(")[1].split(")")[0].strip("'\"")
                image_urls.append(urljoin(url, bg_url))

        # Extract and truncate text
        text = soup.get_text(separator=" ", strip=True)
        return list(set(image_urls)), text[:5000]  # Remove duplicates
    except Exception as e:
        logging.error(f"Error scraping content from {url}: {e}")
        return [], ""

# Filter valid images by size and format
def filter_valid_images(image_urls, min_width=300, min_height=200):
    """
    Filter out images based on dimensions and formats.
    """
    valid_images = []
    for url in image_urls:
        try:
            # Fetch image
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Handle SVG separately
            if url.lower().endswith(".svg"):
                try:
                    # Convert SVG to PNG using cairosvg
                    png_data = cairosvg.svg2png(bytestring=response.content)
                    img = Image.open(BytesIO(png_data))
                except Exception as e:
                    logging.warning(f"Error processing SVG: {url}, Error: {e}")
                    continue
            else:
                # Process non-SVG images
                img = Image.open(BytesIO(response.content))

            # Filter based on dimensions
            if img.width < min_width or img.height < min_height:
                logging.warning(f"Skipping small image: {url} ({img.width}x{img.height})")
                continue

            # Ensure image has proper color channels
            if img.mode not in ["RGB", "RGBA"]:
                logging.warning(f"Skipping non-RGB image: {url} (mode: {img.mode})")
                continue

            # Add valid image URL
            valid_images.append(url)
        except Exception as e:
            logging.warning(f"Error processing image: {url}, Error: {e}")

    logging.info(f"Filtered valid images: {len(valid_images)} out of {len(image_urls)}")
    return valid_images
    
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
        logging.warning(f"Error summarizing content: {e}")
        return ""

# Generate script using OpenAI
def generate_script(enriched_text, max_words):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"{system_prompt} The script should not exceed {max_words} words in total."},
                {"role": "user", "content": enriched_text}
            ]
        )
        raw_content = response.choices[0].message.content.strip()
        logging.info(f"Raw OpenAI response: {raw_content}")

        # Remove surrounding Markdown backticks and potential "json" identifier
        if raw_content.startswith("```") and raw_content.endswith("```"):
            raw_content = raw_content.strip("```").strip()
        if raw_content.lower().startswith("json"):
            raw_content = raw_content[4:].strip()

        logging.info(f"Processed content after cleanup: {raw_content}")
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in API response: {e}")
        st.error("The API response is not valid JSON. Please check the prompt and input content.")
        return []
    except Exception as e:
        logging.error(f"Error generating script: {e}")
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

# Add text overlay to an image
def add_text_overlay_on_fly(image_url, text, font_path):
    """Add captions to an image with proper text wrapping and a semi-transparent background."""
    try:
        # Load the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        # Create drawing context and load font
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=30)

        # Calculate maximum text width (pixels) for wrapping
        max_text_width = img.width - 10  # Padding of 5px on each side
        wrapped_text = textwrap.fill(text, width=40)  # Approx. 40 chars per line

        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        total_text_height = text_height + 20  # Add padding

        # Position the text at the bottom of the image
        x_start = 20  # 20px padding from left
        y_start = img.height - total_text_height - 20  # 20px padding from bottom

        # Create semi-transparent rectangle for text background
        background = Image.new("RGBA", img.size, (255, 255, 255, 0))
        background_draw = ImageDraw.Draw(background)
        background_draw.rectangle(
            [(0, img.height - total_text_height - 40), (img.width, img.height)],
            fill=(0, 0, 0, 128)  # Semi-transparent black
        )

        # Combine overlay and original image
        img = Image.alpha_composite(img, background)

        # Draw the text on the image
        draw = ImageDraw.Draw(img)
        draw.text((x_start, img.height - total_text_height - 30), wrapped_text, font=font, fill="white")

        # Return the final image as a NumPy array
        return np.array(img.convert("RGB"))

    except Exception as e:
        logging.error(f"Failed to add text overlay: {e}")
        return None

from moviepy.video.fx.all import fadein, fadeout
from moviepy.audio.fx.all import audio_fadein, audio_fadeout
from pydub import AudioSegment
import tempfile

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
duration = st.radio("Podcast Duration (minutes):", [1, 2, 3])

if st.button("Generate Content"):
    with st.spinner("Scraping content..."):
        images, text = scrape_images_and_text(url)
        valid_images = filter_valid_images(images)
    if valid_images and text:
        with st.spinner("Summarizing content..."):
            summary = summarize_content(text)
        with st.spinner("Generating script..."):
            script = generate_script(summary, max_words=duration * 150)
        if script:
            audio_segments = [synthesize_voice(part["text"], part["speaker"]) for part in script]
            with st.spinner("Creating video..."):
                video, podcast = create_video(valid_images, script, audio_segments)
            st.video(video)
            st.download_button("Download Podcast", open(podcast, "rb"), file_name="podcast.mp3")
        else:
            st.error("Failed to generate the script.")
    else:
        st.error("No valid images or content found.")
