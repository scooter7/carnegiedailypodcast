
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
Include relevant statistics, facts, and insights based on the summaries. Every podcast should include information about the school's location (city, state) and type of campus (urban, rural, suburban, beach, mountains, etc.). Include accolades and testimonials if they are available, but do not make them up if not available. When mentioning tuition, never make judgmental statements about the cost being high; instead, try to focus on financial aid and scholarship opportunities. 
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

# Filter valid image formats and URLs
def filter_valid_images(image_urls, min_width=400, min_height=300):
    valid_images = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))

            # Skip small or greyscale images
            if img.width < min_width or img.height < min_height:
                logging.warning(f"Skipping small image: {url} ({img.width}x{img.height})")
                continue
            if img.mode not in ["RGB", "RGBA"]:
                logging.warning(f"Skipping non-RGB image: {url} (mode: {img.mode})")
                continue

            # Append valid image URL
            valid_images.append(url)
        except Exception as e:
            logging.warning(f"Error processing image: {url}, Error: {e}")
    logging.info(f"Filtered valid images: {len(valid_images)} out of {len(image_urls)}")
    return valid_images

# Scrape images, logo, and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract image URLs
        image_urls = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        valid_images = [url for url in image_urls if any(url.lower().endswith(ext) for ext in ["jpg", "jpeg", "png"])]

        # Extract college logo
        logo_div = soup.find("div", class_="client-logo")
        logo_url = None
        if logo_div and "background-image" in logo_div.attrs.get("style", ""):
            style_content = logo_div["style"]
            logo_url = style_content.split("url('")[1].split("')")[0]

        # Extract and truncate text
        text = soup.get_text(separator=" ", strip=True)
        return valid_images, logo_url, text[:5000]
    except Exception as e:
        logging.error(f"Error scraping content from {url}: {e}")
        return [], None, ""

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

def create_video_with_audio(images, script, audio_segments):
    clips = []

    # Add the college logo as the first image
    logo_url = "https://images.collegexpress.com/wg_school/1100456_logo.jpg"  # Placeholder; dynamically update this based on scrape
    logo_overlay_image = add_text_overlay_on_fly(logo_url, "Welcome to CX Overview", local_font_path)
    if logo_overlay_image is not None:
        temp_logo_path = "temp_logo.png"
        Image.fromarray(logo_overlay_image).save(temp_logo_path)

        # Create a clip for the logo
        logo_clip = ImageClip(temp_logo_path, duration=3).set_fps(24)  # Show for 3 seconds
        clips.append(logo_clip)

    # Add main content images with audio
    for idx, (image_url, part, audio) in enumerate(zip(images, script, audio_segments)):
        # Add text overlay to the image
        overlay_image = add_text_overlay_on_fly(image_url, part["text"], local_font_path)
        if overlay_image is None:
            logging.error(f"Failed to create overlay for image: {image_url}")
            continue

        # Save the overlay image temporarily for MoviePy
        temp_img_path = f"temp_image_{idx}.png"
        Image.fromarray(overlay_image).save(temp_img_path)

        # Save the audio temporarily for MoviePy
        temp_audio_path = f"audio_{idx}.mp3"
        audio.export(temp_audio_path, format="mp3")

        # Create MoviePy audio clip
        audio_clip = AudioFileClip(temp_audio_path)

        # Create MoviePy image clip with the same duration as the audio segment
        image_clip = (
            ImageClip(temp_img_path, duration=audio_clip.duration)
            .set_audio(audio_clip)
            .set_fps(24)
        )

        # Add fade-in and fade-out transitions
        image_clip = fadein(image_clip, 0.5).fx(fadeout, 0.5)

        clips.append(image_clip)

    # Add the static ending image (cx.jpg)
    ending_image_url = "https://github.com/scooter7/carnegiedailypodcast/blob/main/cx.jpg"
    ending_overlay_image = add_text_overlay_on_fly(ending_image_url, "Thank you for watching!", local_font_path)
    if ending_overlay_image is not None:
        temp_ending_path = "temp_ending.png"
        Image.fromarray(ending_overlay_image).save(temp_ending_path)

        # Create a clip for the ending image
        ending_clip = ImageClip(temp_ending_path, duration=3).set_fps(24)  # Show for 3 seconds
        clips.append(ending_clip)

    # Ensure there are valid clips
    if not clips:
        logging.error("No valid video clips could be created.")
        return None

    # Concatenate video clips into the final video
    final_video = concatenate_videoclips(clips, method="compose")

    # Write final video to file
    final_video_path = "final_video.mp4"
    final_video.write_videofile(final_video_path, codec="libx264", fps=24, audio_codec="aac")

    return final_video_path

# Streamlit app interface
st.title("CX Podcast and Video Generator")
url_input = st.text_input("Enter the URL of the page to scrape text and images:")

duration = st.radio("Select Duration (seconds)", [15, 30, 45, 60], index=0)

if st.button("Generate Content"):
    if not url_input.strip():
        st.error("Please enter a valid URL.")
    else:
        images, logo_url, text = scrape_images_and_text(url_input.strip())
        filtered_images = filter_valid_images(images)

        # Prepend the logo to the filtered images
        if logo_url:
            filtered_images.insert(0, logo_url)

        # Add the static final image
        filtered_images.append("https://github.com/scooter7/carnegiedailypodcast/raw/main/cx.jpg")

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
                        st.audio(podcast_file)
                        st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")

                        script_text = "\n\n".join([f"{part['speaker']}: {part['text']}" for part in conversation_script])
                        script_file = "conversation_script.txt"
                        with open(script_file, "w") as f:
                            f.write(script_text)

                        st.download_button("Download Script", open(script_file, "rb"), file_name="conversation_script.txt")

                        video_file = create_video_with_audio(filtered_images, conversation_script, audio_segments)
                        if video_file:
                            st.video(video_file)
                            st.download_button("Download Video", open(video_file, "rb"), file_name="video_with_audio.mp4")
                        else:
                            st.error("Failed to create video.")

                        else:
                            st.error("Failed to create video.")
