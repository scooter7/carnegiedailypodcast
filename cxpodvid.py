
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
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, news-oriented conversation between Ali and Lisa. Make sure that the voices are excited and enthusiastic, not flat and overly matter-of-fact.
Include relevant statistics, facts, and insights based on the summaries. Every podcast should include information about the school's location (city, state) and type of campus (urban, rural, suburban, beach, mountains, etc.). Include accolades and testimonials if they are available, but do not make them up if not available. When mentioning tuition, never make judgmental statements about the cost being high; instead, try to focus on financial aid and scholarship opportunities. 
The conversation should feel conversational and engaging, with occasional natural pauses and fillers like 'um,' and  'you know' (Do not overdo the pauses and fillers, though). Whenever you discuss a faculty-to-student ratio like 14:1, pronounce it as 14 to 1 (or whatever the applicable true number is). At the end of the podcast, always mention that more information about the school can be found at collegexpress.com.Make sure that, anytime, collegexpress is mentioned, it is pronounced as college express. However, at the end of the video, it should be spelled as collegexpress.

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

from bs4 import BeautifulSoup
import requests
import logging

def extract_logo_url(soup):
    """
    Extracts the logo URL from the <div class="client-logo">.
    """
    try:
        # Find the client-logo div
        logo_div = soup.find("div", class_="client-logo")
        if not logo_div or "style" not in logo_div.attrs:
            logging.warning("Logo div or style attribute not found.")
            return None

        # Extract the style attribute
        style_attr = logo_div["style"]

        # Look for the background-image URL
        start_index = style_attr.find("url('") + 5
        end_index = style_attr.find("')", start_index)
        logo_url = style_attr[start_index:end_index]

        # Validate the logo URL
        if not logo_url.startswith("https://images.collegexpress.com/wg_school/"):
            logging.warning("Invalid logo URL format.")
            return None

        return logo_url
    except Exception as e:
        logging.error(f"Error extracting logo URL: {e}")
        return None

def scrape_images_and_text(url):
    """
    Scrapes the CollegeXpress page for the logo, images, and text.
    """
    try:
        # Fetch the page content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract the logo URL
        logo_url = extract_logo_url(soup)

        # Extract other image URLs
        image_urls = [img["src"] for img in soup.find_all("img", src=True)]
        valid_images = [img for img in image_urls if img.endswith((".jpg", ".jpeg", ".png"))]

        # Extract and truncate text from the page
        text = soup.get_text(separator=" ", strip=True)

        return logo_url, valid_images, text[:5000]
    except Exception as e:
        logging.error(f"Error scraping content from {url}: {e}")
        return None, [], ""

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
from pydub import AudioSegment

def synthesize_cloned_voice_with_pacing(text, speaker, pause_duration=2000):
    """
    Synthesizes voice for a speaker with a pause after each sentence.
    """
    try:
        # Generate the speech audio
        audio_generator = elevenlabs_client.generate(
            text=text,
            voice=speaker_voice_map[speaker],
            model="eleven_multilingual_v2"
        )
        audio_content = b"".join(audio_generator)

        # Create an AudioSegment object
        audio = AudioSegment.from_file(BytesIO(audio_content), format="mp3")

        # Add a pause at the end of the audio
        pause = AudioSegment.silent(duration=pause_duration)
        return audio + pause

    except Exception as e:
        logging.error(f"Error synthesizing speech for {speaker}: {e}")
        return None

def combine_audio_with_pacing(script, audio_segments):
    """
    Combines the audio segments of all speakers with natural pacing.
    """
    combined_audio = AudioSegment.empty()

    for idx, (part, audio) in enumerate(zip(script, audio_segments)):
        # Add the current speaker's audio
        combined_audio += audio

        # Add extra silence between speakers
        if idx < len(script) - 1:  # Avoid adding silence after the last speaker
            combined_audio += AudioSegment.silent(duration=2000)  # 2 seconds of silence

    return combined_audio

# Add text overlay to an image
def add_text_overlay_on_fly(image_url, text, font_path):
    """
    Add captions to an image with proper text wrapping and a semi-transparent background.
    Ensures text spans across the textbox width and is center-aligned.
    """
    try:
        # Load the image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGBA")

        # Define text box padding and dimensions
        text_box_padding = 40  # Padding inside the text box
        text_box_height = 120  # Height of the text box

        # Extend the image height to add space for the text box
        canvas = Image.new("RGBA", (img.width, img.height + text_box_height), (255, 255, 255, 255))
        canvas.paste(img, (0, 0))

        # Draw the text box
        draw = ImageDraw.Draw(canvas)
        text_box_start_y = img.height  # Start the text box below the image
        text_box_end_y = img.height + text_box_height
        draw.rectangle(
            [(0, text_box_start_y), (img.width, text_box_end_y)],
            fill=(0, 0, 0, 128),  # Semi-transparent black
        )

        # Load the font
        font = ImageFont.truetype(font_path, size=28)

        # Wrap text dynamically based on the width of the text box
        max_text_width = img.width - 2 * text_box_padding
        lines = []
        words = text.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            text_bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            if text_width <= max_text_width:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))  # Add the last line

        # Calculate the total height of the wrapped text
        total_text_height = sum(
            draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1]
            for line in lines
        )
        start_y = text_box_start_y + (text_box_height - total_text_height) // 2  # Center vertically

        # Draw each line of text
        current_y = start_y
        for line in lines:
            text_bbox = draw.textbbox((0, 0), line, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (img.width - text_width) // 2  # Center-align text horizontally
            draw.text((text_x, current_y), line, font=font, fill="white")
            current_y += text_bbox[3] - text_bbox[1]

        return np.array(canvas.convert("RGB"))

    except Exception as e:
        logging.error(f"Failed to add text overlay: {e}")
        return None

from moviepy.video.fx.all import fadein, fadeout

def create_video_with_audio(images, script, audio_segments, logo_url):
    """
    Creates a video with audio, including the dynamically scraped logo as the first image
    and a static image at the end, with textboxes placed below each image.
    """
    clips = []

    # Add the college logo as the first image
    try:
        logo_overlay_image = add_text_overlay_on_fly(logo_url, "Welcome to CX Overview", local_font_path)
        if logo_overlay_image is not None:
            temp_logo_path = "temp_logo.png"
            Image.fromarray(logo_overlay_image).save(temp_logo_path)

            # Create a silent audio clip for the logo (adjust duration as needed)
            silent_audio = AudioSegment.silent(duration=2000)  # 2 seconds of silence
            temp_audio_path = "logo_audio.mp3"
            silent_audio.export(temp_audio_path, format="mp3")

            # Create MoviePy audio and image clip for the logo
            audio_clip = AudioFileClip(temp_audio_path)
            image_clip = (
                ImageClip(temp_logo_path, duration=audio_clip.duration)
                .set_audio(audio_clip)
                .set_fps(24)
            )

            # Add fade-in and fade-out transitions for the logo
            image_clip = fadein(image_clip, 0.5).fx(fadeout, 0.5)
            clips.append(image_clip)
        else:
            logging.error("Logo overlay image not generated.")
    except Exception as e:
        logging.error(f"Failed to add logo as first image: {e}")

    # Add main content (images + script + audio)
    for idx, (image_url, part, audio) in enumerate(zip(images, script, audio_segments)):
        try:
            # Fetch and process image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))

            # Extend canvas to add text box below the image
            text_box_height = 100  # Adjust height of text box
            text_image = Image.new("RGBA", (img.width, img.height + text_box_height), (255, 255, 255, 255))
            text_image.paste(img, (0, 0))  # Paste original image on top

            # Draw the text below the image
            draw = ImageDraw.Draw(text_image)
            font = ImageFont.truetype(local_font_path, size=20)
            text = textwrap.fill(part["text"], width=50)  # Wrap text to 50 characters

            # Calculate textbox position
            text_x = 20  # Padding from left
            text_y = img.height + 10  # Start text below the image
            draw.text((text_x, text_y), text, fill="black", font=font)

            # Save the extended image temporarily for MoviePy
            temp_img_path = f"temp_image_{idx}.png"
            text_image.convert("RGB").save(temp_img_path)

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

        except Exception as e:
            logging.error(f"Failed to process main content image: {image_url}. Error: {e}")
            continue

    # Add the static CX Overview image at the end
    try:
        cx_image_url = "https://github.com/scooter7/carnegiedailypodcast/raw/main/cx.jpg"
        response = requests.get(cx_image_url, timeout=10)
        response.raise_for_status()
        cx_image = Image.open(BytesIO(response.content)).convert("RGBA")
        temp_cx_path = "temp_cx_image.png"
        cx_image.save(temp_cx_path)

        # Add silent audio for the static image
        silent_audio = AudioSegment.silent(duration=3000)  # 3 seconds of silence
        temp_cx_audio_path = "cx_audio.mp3"
        silent_audio.export(temp_cx_audio_path, format="mp3")

        # Create MoviePy audio and image clip for the CX Overview image
        cx_audio_clip = AudioFileClip(temp_cx_audio_path)
        cx_image_clip = (
            ImageClip(temp_cx_path, duration=cx_audio_clip.duration)
                .set_audio(cx_audio_clip)
                .set_fps(24)
        )

        # Add fade-in and fade-out transitions
        cx_image_clip = fadein(cx_image_clip, 0.5).fx(fadeout, 0.5)
        clips.append(cx_image_clip)

    except Exception as e:
        logging.error(f"Failed to add CX Overview image: {e}")

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
        logo_url, images, text = scrape_images_and_text(url_input.strip())
        filtered_images = filter_valid_images(images)

        if not logo_url:
            st.warning("No logo found for this page. Skipping logo addition.")

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

                        video_file = create_video_with_audio(filtered_images, conversation_script, audio_segments, logo_url)
                        if video_file:
                            st.video(video_file)
                            st.download_button("Download Video", open(video_file, "rb"), file_name="video_with_audio.mp4")
                        else:
                            st.error("Failed to create video.")
