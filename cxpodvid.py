# Standard Python and library imports
import sys
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
from moviepy.editor import ImageClip, concatenate_videoclips, CompositeVideoClip
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

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
    "Ali": "NYy9s57OPECPcDJavL3T"  # Replace with the ID of your cloned voice
}

# System prompt for the podcast script
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, news-oriented conversation between Ali and Lisa. 
Include relevant statistics, facts, and insights based on the summaries. 
Format the response strictly as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text, explanations, or formatting.
"""

# Estimate maximum words for duration
def max_words_for_duration(duration_seconds):
    wpm = 150  # Average words per minute
    return int((duration_seconds / 60) * wpm)

# Filter valid image formats
def filter_valid_images(image_urls, max_images=5):
    valid_images = []
    for url in image_urls[:max_images]:  # Restrict to a maximum number of images
        if url.lower().endswith(("png", "jpg", "jpeg")):
            valid_images.append(url)
    return valid_images

# Download and convert an image to a rasterized format
def download_image(url):
    try:
        response = requests.get(url, timeout=10)  # Set timeout for image download
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(temp_img.name, format="PNG")
        return temp_img.name
    except Exception as e:
        st.warning(f"Failed to process image: {url}. Error: {e}")
        return None

# Scrape images and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)  # Set timeout for scraping
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract images
        images = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        images = filter_valid_images(images)
        images = [download_image(img_url) for img_url in images if download_image(img_url)]

        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        return images, text[:5000]
    except requests.exceptions.Timeout:
        st.warning(f"Request to {url} timed out.")
        return [], ""
    except Exception as e:
        st.warning(f"Error scraping content from {url}: {e}")
        return [], ""

# Summarize content using OpenAI
def summarize_content(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following content into meaningful insights."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error summarizing content: {e}")
        return ""

# Generate podcast script with word limit
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
        try:
            conversation_script = json.loads(raw_content)
            truncated_script = []
            total_words = 0
            for part in conversation_script:
                word_count = len(part["text"].split())
                if total_words + word_count <= max_words:
                    truncated_script.append(part)
                    total_words += word_count
                else:
                    break
            return truncated_script
        except json.JSONDecodeError:
            st.error("The API response is not valid JSON. Please check the prompt and input content.")
            st.warning("Response content:\n" + raw_content)
            return []
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return []

# Synthesize speech using ElevenLabs
def synthesize_cloned_voice(text, speaker):
    try:
        audio_generator = elevenlabs_client.generate(
            text=text,
            voice=speaker_voice_map[speaker],
            model="eleven_multilingual_v2"
        )
        audio_content = b"".join(audio_generator)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio_content)
            temp_audio_path = temp_audio_file.name
        return AudioSegment.from_file(temp_audio_path, format="mp3")
    except Exception as e:
        st.error(f"Error synthesizing speech for {speaker}: {e}")
        return None

# Add text to image using PIL
def add_text_to_image(image_path, text, font_size=24):
    try:
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", font_size)  # Use a default font

        # Calculate text size and position
        text_width, text_height = draw.textsize(text, font=font)
        text_x = (img.width - text_width) // 2
        text_y = img.height - text_height - 20  # Padding from bottom

        # Add background for text
        draw.rectangle(
            [(text_x - 10, text_y - 10), (text_x + text_width + 10, text_y + text_height + 10)],
            fill="black"
        )
        # Add text
        draw.text((text_x, text_y), text, font=font, fill="white")

        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(temp_img.name, format="PNG")
        return temp_img.name
    except Exception as e:
        st.error(f"Error adding text to image: {e}")
        return None

# Create video using images and captions
def create_video(images, script, duration_seconds):
    if not images or not script:
        st.error("No valid images or script provided. Cannot create video.")
        return None

    clips = []
    segment_duration = duration_seconds / len(script) if script else 0

    for i, (image, part) in enumerate(zip(images, script)):
        text_image_path = add_text_to_image(image, part["text"])
        if text_image_path:
            try:
                img_clip = ImageClip(text_image_path).set_duration(segment_duration)
                clips.append(img_clip)
            except Exception as e:
                st.warning(f"Failed to create clip for script part: {part['text']}. Error: {e}")
        else:
            st.warning(f"Failed to add text to image for script part: {part['text']}")

    if not clips:
        st.error("No video clips could be created. Ensure valid images and script are provided.")
        return None

    try:
        final_video = concatenate_videoclips(clips)
        video_file = "video_short.mp4"
        final_video.write_videofile(video_file, codec="libx264", fps=24)
        return video_file
    except Exception as e:
        st.error(f"Failed to concatenate video clips. Error: {e}")
        return None

# Main Streamlit App Interface
st.title("CX College Profile Video & Podcast Creator")
st.write("Enter a CX Profile URL to generate a podcast and video short.")

# Input fields for the URL and duration selection
parent_url = st.text_input("Enter the URL of the page:")
duration = st.radio("Select Duration (seconds)", [15, 30, 45, 60], index=0)

if st.button("Generate Content"):
    if parent_url.strip():
        st.write("Scraping content from the URL...")
        images, scraped_text = scrape_images_and_text(parent_url.strip())
        st.write(f"Scraping complete. Found {len(images)} images.")

        if scraped_text:
            st.write("Summarizing content...")
            summary = summarize_content(scraped_text)
            st.write("Summarization complete.")

            if summary:
                st.write("Generating podcast script...")
                max_words = max_words_for_duration(duration)
                conversation_script = generate_script(summary, max_words)
                st.write(f"Script generation complete. Generated {len(conversation_script)} script parts.")

                if conversation_script:
                    st.write("Generating podcast audio...")
                    audio_segments = []
                    for part in conversation_script:
                        st.write(f"Processing audio for {part['speaker']}...")
                        audio = synthesize_cloned_voice(part["text"], part["speaker"])
                        if audio:
                            audio_segments.append(audio)

                    if audio_segments:
                        st.write("Combining audio segments...")
                        combined_audio = sum(audio_segments, AudioSegment.empty())
                        podcast_file = "podcast.mp3"
                        combined_audio.export(podcast_file, format="mp3")
                        st.success("Podcast generated successfully!")
                        st.audio(podcast_file)
                        st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")

                    if images and conversation_script:
                        st.write("Creating video short...")
                        video_file = create_video(images, conversation_script, duration)
                        if video_file:
                            st.success("Video short created successfully!")
                            st.video(video_file)
                            st.download_button("Download Video", open(video_file, "rb"), file_name="video_short.mp4")
                        else:
                            st.error("Failed to create video short. Ensure valid images and script are available.")
                    else:
                        st.error("No valid images or script available to create video short.")
                else:
                    st.error("Failed to generate the podcast script.")
            else:
                st.error("Failed to summarize the content.")
        else:
            st.error("Failed to scrape content from the URL.")
    else:
        st.error("Please enter a valid URL.")
