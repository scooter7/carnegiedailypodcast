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
    logging.info(f"Filtered valid images: {len(valid_images)} out of {len(image_urls)})")
    return valid_images

# Scrape images and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract image URLs
        image_urls = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        valid_images = [url for url in image_urls if any(url.lower().endswith(ext) for ext in ["jpg", "jpeg", "png"])]

        # Extract and truncate text
        text = soup.get_text(separator=" ", strip=True)
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

# Calculate word limit based on duration
def max_words_for_duration(duration_seconds):
    """Calculate the maximum number of words based on the duration of the video."""
    wpm = 150  # Average words per minute for speech
    return int((duration_seconds / 60) * wpm)

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

        # Wrap text to fit within the image width
        max_width = img.width - 40  # Padding of 20px on each side
        lines = []
        words = text.split()
        line = words[0]

        # Break text into lines that fit within the max_width
        for word in words[1:]:
            line_width = font.getbbox(line + " " + word)[2]  # Calculate line width
            if line_width <= max_width:
                line += " " + word
            else:
                lines.append(line)
                line = word
        lines.append(line)

        # Calculate text height
        line_height = font.getbbox("Ay")[3]  # Get height of a single line
        total_text_height = len(lines) * line_height + 20  # Add padding

        # Calculate background height and text position
        y_start = img.height - total_text_height - 20  # 20px padding from the bottom
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)
        draw_overlay.rectangle(
            [(0, y_start), (img.width, img.height)],
            fill=(0, 0, 0, 128)  # Semi-transparent black
        )

        # Combine the overlay and the image
        img = Image.alpha_composite(img, overlay)

        # Draw the text on the image
        y_text = y_start + 10  # Padding inside the background
        for line in lines:
            text_width = font.getbbox(line)[2]
            x_text = (img.width - text_width) // 2  # Center text horizontally
            draw.text((x_text, y_text), line, font=font, fill="white")
            y_text += line_height

        # Return the final image as a NumPy array
        return np.array(img.convert("RGB"))

    except Exception as e:
        logging.error(f"Failed to add text overlay: {e}")
        return None

# Generate audio sequentially for all speakers
def synthesize_speaker_audio(script, speaker):
    """Concatenate all text for a speaker and synthesize a single audio file."""
    try:
        speaker_text = "\n".join([part["text"] for part in script if part["speaker"] == speaker])
        audio_generator = elevenlabs_client.generate(
            text=speaker_text,
            voice=speaker_voice_map[speaker],
            model="eleven_multilingual_v2"
        )
        audio_content = b"".join(audio_generator)
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        with open(temp_audio_file.name, "wb") as f:
            f.write(audio_content)
        return AudioSegment.from_file(temp_audio_file.name)
    except Exception as e:
        logging.error(f"Error synthesizing audio for {speaker}: {e}")
        return None

def generate_audio(script):
    """Generate audio for all speakers sequentially to avoid overlaps."""
    audio_segments = []
    for speaker in speaker_voice_map.keys():
        speaker_audio = synthesize_speaker_audio(script, speaker)
        if speaker_audio:
            audio_segments.append(speaker_audio)
    return sum(audio_segments, AudioSegment.empty())

# Ensure video duration matches audio duration
def match_video_duration(images, total_duration):
    """Distribute total duration evenly across all images."""
    num_images = len(images)
    if num_images == 0:
        return []
    per_image_duration = total_duration / num_images
    return [per_image_duration] * num_images

# Create video with audio and fade transitions
def create_video_with_audio(images, script, audio_segments, total_duration):
    """Create a video from images and audio, with fade transitions."""
    durations = match_video_duration(images, total_duration)
    clips = []

    for idx, (image_url, part, audio, duration) in enumerate(zip(images, script, audio_segments, durations)):
        # Add text overlay to the image
        overlay_image = add_text_overlay_on_fly(image_url, part["text"], local_font_path)
        if overlay_image is None:
            logging.error(f"Failed to create overlay for image: {image_url}")
            continue

        # Save the overlay image temporarily for MoviePy
        temp_img_path = f"temp_image_{idx}.png"
        Image.fromarray(overlay_image).save(temp_img_path)

        # Save the audio temporarily for MoviePy
        audio_path = f"audio_{idx}.mp3"
        audio.export(audio_path, format="mp3")

        # Create MoviePy clips
        audio_clip = AudioFileClip(audio_path)
        image_clip = (
            ImageClip(temp_img_path, duration=duration)
            .set_audio(audio_clip)
            .fadein(0.5)
            .fadeout(0.5)
            .set_fps(24)
        )
        clips.append(image_clip)

    if clips:
        # Concatenate all clips into the final video
        final_video = concatenate_videoclips(clips, method="compose")
        video_file_path = "final_video.mp4"
        final_video.write_videofile(video_file_path, codec="libx264", fps=24, audio_codec="aac")
        return video_file_path
    return None

# Streamlit app interface
st.title("CX Podcast and Video Generator")
url_input = st.text_input("Enter the URL of the page to scrape text and images:")

duration = st.slider("Select Duration (seconds)", min_value=15, max_value=120, value=60, step=15)

if st.button("Generate Content"):
    if not url_input.strip():
        st.error("Please enter a valid URL.")
    else:
        images, text = scrape_images_and_text(url_input.strip())
        filtered_images = filter_valid_images(images)

        if text:
            summary = summarize_content(text)
            if summary:
                max_words = max_words_for_duration(duration)
                conversation_script = generate_script(summary, max_words)
                if conversation_script:
                    # Generate non-overlapping audio for all speakers
                    combined_audio = generate_audio(conversation_script)

                    if combined_audio:
                        # Export the combined audio to a podcast file
                        podcast_file = "podcast.mp3"
                        combined_audio.export(podcast_file, format="mp3")
                        st.audio(podcast_file)
                        st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")

                        # Save and download the script as a text file
                        script_text = "\n\n".join([f"{part['speaker']}: {part['text']}" for part in conversation_script])
                        script_file = "conversation_script.txt"
                        with open(script_file, "w") as f:
                            f.write(script_text)
                        st.download_button("Download Script", open(script_file, "rb"), file_name="conversation_script.txt")

                        # Create video with audio and text overlays
                        video_file = create_video_with_audio(
                            filtered_images, 
                            conversation_script, 
                            [combined_audio] * len(filtered_images), 
                            duration
                        )
                        if video_file:
                            st.video(video_file)
                            st.download_button("Download Video", open(video_file, "rb"), file_name="video_with_audio.mp4")
                        else:
                            st.error("Failed to create video.")
                else:
                    st.error("Failed to generate the conversation script.")
            else:
                st.error("Failed to summarize the content.")
        else:
            st.error("Failed to scrape valid content from the provided URL.")

