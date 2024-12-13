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
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, conversational dialogue between Ali and Lisa. 
Include relevant statistics and insights. Format the response strictly as a JSON array of objects with 'speaker' and 'text' keys.
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

# Download and process image
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Handle RGBA images by converting them to RGB
        if img.mode == "RGBA":
            img = img.convert("RGB")

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp_img.name, format="JPEG")
        logging.info(f"Image downloaded and saved: {temp_img.name}")
        return temp_img.name
    except Exception as e:
        logging.warning(f"Failed to download or process image: {url}. Error: {e}")
        return None

# Filter valid image formats and URLs
def filter_valid_images(image_urls, max_images=5):
    valid_images = []
    for url in image_urls[:max_images]:
        # Skip unsupported formats
        if any(url.lower().endswith(ext) for ext in ["svg", "webp", "bmp", "gif"]):
            logging.info(f"Skipping unsupported image format: {url}")
            continue
        image_path = download_image(url)
        if image_path:
            valid_images.append(image_path)
    return valid_images

# Scrape images and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            raise ValueError(f"Received {response.status_code} status code for URL: {url}")

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract images
        image_urls = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        valid_images = filter_valid_images(image_urls)

        # Log downloaded image paths
        logging.info(f"Downloaded valid images: {valid_images}")

        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        if not text:
            logging.warning("No textual content found on the page.")
        return valid_images, text[:5000]
    except Exception as e:
        logging.error(f"Error scraping content from {url}: {e}")
        return [], ""

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
        logging.info(f"Raw OpenAI response: {raw_content}")  # Log the raw response for debugging

        # Remove surrounding Markdown backticks and potential "json" identifier
        if raw_content.startswith("```") and raw_content.endswith("```"):
            raw_content = raw_content.strip("```").strip()
        if raw_content.lower().startswith("json"):
            raw_content = raw_content[4:].strip()

        logging.info(f"Processed content after cleanup: {raw_content}")
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in API response: {e}")
        logging.error(f"Raw response content:\n{raw_content}")
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
def add_text_overlay(image_path, text, output_path, font_path):
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        # Load the font
        font = ImageFont.truetype(font_path, size=30)

        # Wrap the text
        wrapped_text = textwrap.fill(text, width=40)

        # Calculate text dimensions
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Position the text (Ensure it's always within bounds)
        x_start = max((img.width - text_width) // 2, 10)  # Add padding of 10px
        y_start = min(img.height - text_height - 30, img.height - 50)  # Ensure bottom padding

        # Draw background for text
        background = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw_bg = ImageDraw.Draw(background)
        draw_bg.rectangle(
            [(x_start - 10, y_start - 10), (x_start + text_width + 10, y_start + text_height + 10)],
            fill=(0, 0, 0, 200)  # Make background semi-transparent
        )
        img = Image.alpha_composite(img, background)

        # Draw the text on top
        draw = ImageDraw.Draw(img)
        draw.text((x_start, y_start), wrapped_text, font=font, fill="white")

        # Save the image
        img.convert("RGB").save(output_path, "JPEG")
        logging.info(f"Text overlay added to image: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Failed to add text overlay: {e}")
        return None

def create_video_with_audio(images, script, audio_segments):
    if not images or not script or not audio_segments:
        st.error("No valid images, script, or audio segments provided. Cannot create video.")
        return None, []

    clips = []
    temp_files = []  # Track temporary files

    try:
        for idx, (image, part, audio) in enumerate(zip(images, script, audio_segments)):
            logging.info(f"Processing image {idx + 1}/{len(images)} with text overlay...")

            # Add text overlay to the image
            temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            text_overlay_path = add_text_overlay(image, part["text"], temp_image_path, local_font_path)
            if not text_overlay_path:
                logging.error(f"Failed to add text overlay to image: {image}")
                continue
            logging.info(f"Overlay image saved: {text_overlay_path}")
            temp_files.append(text_overlay_path)

            # Create video clip
            audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
            audio.export(audio_path, format="mp3")
            temp_files.append(audio_path)

            audio_clip = AudioFileClip(audio_path)
            image_clip = ImageClip(text_overlay_path, duration=audio_clip.duration)
            image_clip = image_clip.set_audio(audio_clip).set_fps(24)

            logging.info(f"Created video clip for image {idx + 1}")
            clips.append(image_clip)

        # Concatenate video clips
        if clips:
            final_video = concatenate_videoclips(clips, method="compose")
            video_file_path = "final_video.mp4"
            final_video.write_videofile(video_file_path, codec="libx264", fps=24, audio_codec="aac")
            logging.info(f"Video successfully created: {video_file_path}")
            return video_file_path, temp_files
        else:
            st.error("No video clips were created.")
            return None, temp_files
    finally:
        # Cleanup temporary files in case of failure
        if not clips:  # Clean up only if no successful clips
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
# After generating overlays in create_video_with_audio
st.write("Preview images with text overlay:")
for overlay_path in temp_files:
    if overlay_path.endswith(".jpg"):
        st.image(overlay_path, caption=f"Overlay: {overlay_path}")

# Streamlit app interface
st.title("CX Podcast and Video Generator")
url_input = st.text_input("Enter the URL of the page to scrape text and images:")

duration = st.radio("Select Duration (seconds)", [15, 30, 45, 60], index=0)

if st.button("Generate Content"):
    if not url_input.strip():
        st.error("Please enter a valid URL.")
    else:
        images, text = scrape_images_and_text(url_input.strip())
        if text:
            summary = summarize_content(text)
            if summary:
                max_words = max_words_for_duration(duration)
                conversation_script = generate_script(summary, max_words)
                if conversation_script:
                    audio_segments = [synthesize_cloned_voice(part["text"], part["speaker"]) for part in conversation_script]
                    audio_segments = [audio for audio in audio_segments if audio]
                    if audio_segments:
                        combined_audio = sum(audio_segments, AudioSegment.empty())
                        podcast_file = "podcast.mp3"
                        combined_audio.export(podcast_file, format="mp3")
                        st.success("Podcast created successfully!")
                        st.audio(podcast_file)
                        st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")

                        # Generate video and retrieve temp files
                        video_file, temp_files = create_video_with_audio(images, conversation_script, audio_segments)
                        if video_file:
                            st.success("Video created successfully!")
                            st.video(video_file)
                            st.download_button("Download Video", open(video_file, "rb"), file_name="video_with_audio.mp4")

                            # Preview overlayed images
                            st.write("Preview images with text overlay:")
                            for overlay_path in temp_files:
                                if overlay_path.endswith(".jpg"):
                                    st.image(overlay_path, caption=f"Overlay: {overlay_path}")
                        else:
                            st.error("Failed to create the video.")

                    else:
                        st.error("Failed to synthesize audio for the script.")
                else:
                    st.error("Failed to generate the podcast script.")
            else:
                st.error("Failed to summarize content.")
        else:
            st.error("Failed to scrape content.")
