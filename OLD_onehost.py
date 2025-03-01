# Standard Python and library imports
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
import openai
from urllib.parse import urljoin
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import logging
import subprocess
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

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
    wpm = 150
    return int((duration_seconds / 60) * wpm)

# Filter valid images
def filter_valid_images(image_urls, min_width=200, min_height=200):
    valid_images = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content))
            if img.width >= min_width and img.height >= min_height and img.mode in ["RGB", "RGBA", "P"]:
                valid_images.append(url)
        except Exception as e:
            logging.warning(f"Invalid image {url}: {e}")
    return valid_images


# Scrape images and text
def scrape_images_and_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        image_urls = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        text = soup.get_text(separator=" ", strip=True)
        return image_urls, text[:5000]
    except Exception as e:
        logging.error(f"Error scraping content from {url}: {e}")
        return [], ""

# Generate script using OpenAI
def generate_script(text, max_words):
    opening_message = (
        "Welcome to the CollegeXpress Campus Countdown, where we explore colleges and universities around the country to help you find great schools to apply to! "
        "Today we’re highlighting [school or list of schools]. Let’s get started!"
    )
    closing_message = (
        "Don’t forget, you can connect with any of our featured colleges by visiting CollegeXpress.com. "
        "Just click the green “Yes, connect me!” buttons when you see them on the site, and then the schools you’re interested in will reach out to you with more information! "
        "You can find the links to these schools in the description below. Don’t forget to follow us on social media @CollegeXpress. "
        "Until next time, happy college hunting!"
    )
    system_prompt = """
    You are a podcast host for 'CollegeXpress Campus Countdown.' Generate a robust, fact-based summary of the school at the scraped webpage narrated by Lisa. 
    Include location, campus type, accolades, and testimonials. Don't be afraid to show emotion and enthusiasm!
    """
    try:
        # Generate dynamic part of the script
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this text: {text} Limit: {max_words} words."}
            ]
        )
        dynamic_content = response.choices[0].message.content.strip()
        
        # Combine all parts of the script
        full_script = f"{opening_message}\n\n{dynamic_content}\n\n{closing_message}"
        return full_script
    except Exception as e:
        logging.error(f"Error generating script: {e}")
        return f"{opening_message}\n\n[Error generating dynamic content]\n\n{closing_message}"

# Generate speech using OpenAI TTS
def generate_audio_with_openai(text, voice="alloy"):
    try:
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=text)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# Helper function to download an image
def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        temp_img_path = tempfile.mktemp(suffix=".jpg")
        with open(temp_img_path, "wb") as f:
            f.write(response.content)
        return temp_img_path
    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        return None

# Add text overlay
def add_text_overlay(image_path, text):
    try:
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(local_font_path, size=28)
        lines = textwrap.wrap(text, width=40)
        y = img.height - (len(lines) * 35) - 20
        for line in lines:
            text_width = draw.textlength(line, font=font)
            draw.text(((img.width - text_width) // 2, y), line, font=font, fill="white")
            y += 35
        temp_img_path = tempfile.mktemp(suffix=".png")
        img.save(temp_img_path)
        return temp_img_path
    except Exception as e:
        logging.error(f"Failed to add text overlay: {e}")
        return None

# Ensure even dimensions for videos
def ensure_even_dimensions(image_path):
    """
    Ensure image dimensions are even without altering the aspect ratio.
    Pads the image with black borders if necessary.
    """
    try:
        img = Image.open(image_path)
        width, height = img.size

        # Calculate even dimensions
        new_width = width if width % 2 == 0 else width + 1
        new_height = height if height % 2 == 0 else height + 1

        # Create a new image with the even dimensions
        padded_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        # Center the original image on the padded image
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        padded_img.paste(img, (x_offset, y_offset))

        # Save the padded image to a temporary file
        temp_img_path = tempfile.mktemp(suffix=".png")
        padded_img.save(temp_img_path)
        return temp_img_path
    except Exception as e:
        logging.error(f"Error ensuring even dimensions while preserving aspect ratio: {e}")
        return None

# Add dropdown for video effects
effect_option = st.selectbox(
    "Select Video Effect:", 
    ["None", "Cartoon", "Sketch", "Anime"]
)

# Updated generate_video_clip function with effects
def generate_video_clip_with_effects(image_url, duration, text=None, filter_option="None", transition_option="None", effect_option="None"):
    try:
        duration = max(1, round(duration, 2))  # Ensure minimum duration

        # Download the image
        img_path = download_image(image_url)
        if not img_path:
            raise ValueError("Failed to download image.")

        # Add text overlay if specified
        if text:
            img_path = add_text_overlay(img_path, text)

        # Ensure even dimensions and maintain aspect ratio
        img_path = ensure_even_dimensions(img_path)
        if not img_path:
            raise ValueError("Failed to process image dimensions.")

        # Define filters based on selected effect
        effects = {
            "None": None,
            "Cartoon": "geq=lum='p(X,Y)':cb='128+(p(X,Y)-128)*2':cr='128+(p(X,Y)-128)*2'",
            "Sketch": "edgedetect=mode=colormix:high=0.1:low=0.1",
            "Anime": "tblend=all_mode='and'",  # Anime-style blending
        }
        transitions = {
            "None": None,
            "Fade": "fade=t=in:st=0:d=1",
            "Zoom": "zoompan=z='zoom+0.01':d=25",
        }

        # Build the video filter chain
        vf_chain_parts = [
            effects.get(effect_option, None),
            transitions.get(transition_option, None),
        ]
        vf_chain = ",".join([part for part in vf_chain_parts if part])

        # Build FFmpeg command
        temp_video = tempfile.mktemp(suffix=".mp4")
        ffmpeg_command = [
            "ffmpeg", "-y", "-loop", "1", "-i", img_path, "-t", str(duration)
        ]
        if vf_chain:
            ffmpeg_command += ["-vf", vf_chain]
        else:
            ffmpeg_command += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]  # Neutral scaling

        ffmpeg_command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", temp_video]

        # Run FFmpeg command
        subprocess.run(ffmpeg_command, check=True)
        return temp_video
    except subprocess.CalledProcessError as ffmpeg_error:
        logging.error(f"FFmpeg process failed: {ffmpeg_error.stderr}")
        return None
    except Exception as e:
        logging.error(f"Error generating video clip with effect: {e}")
        return None

# Combine videos and audio
def create_final_video(video_clips, audio_path, end_image_url, duration_per_clip, filter_option, transition_option):
    try:
        if not video_clips or None in video_clips:
            raise ValueError("One or more video clips are invalid.")

        # Process the end image
        try:
            logging.info("Adding end image to the video...")
            end_image_path = ensure_even_dimensions(download_image(end_image_url))
            if end_image_path:
                end_clip = generate_video_clip_with_effects(end_image_path, duration_per_clip, None, filter_option, transition_option)
                if end_clip:
                    video_clips.append(end_clip)
                    logging.info("End image successfully processed and added.")
                else:
                    logging.warning("Failed to process the end image. It will be skipped.")
            else:
                logging.warning("Failed to process the end image dimensions. It will be skipped.")
        except Exception as e:
            logging.error(f"Error processing the end image: {e}")
            logging.warning("An error occurred while processing the end image. It will be skipped.")

        # Get audio duration using ffprobe
        audio_duration_command = [
            "ffprobe", "-i", audio_path, "-show_entries", "format=duration",
            "-v", "quiet", "-of", "csv=p=0"
        ]
        audio_duration = float(subprocess.check_output(audio_duration_command).strip())

        # Extend the last video clip to match audio duration if necessary
        total_video_duration = sum([
            float(subprocess.check_output([
                "ffprobe", "-i", clip, "-show_entries", "format=duration",
                "-v", "quiet", "-of", "csv=p=0"
            ]).strip()) for clip in video_clips
        ])
        if total_video_duration < audio_duration:
            last_clip = video_clips[-1]
            extended_clip = extend_video_clip(last_clip, audio_duration - total_video_duration)
            video_clips[-1] = extended_clip

        # Concatenate video clips with audio
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, "w") as f:
            for clip in video_clips:
                f.write(f"file '{clip}'\n")

        final_video = tempfile.mktemp(suffix=".mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-i", audio_path,
            "-c:v", "libx264", "-c:a", "aac", "-shortest", final_video
        ], check=True)
        return final_video
    except Exception as e:
        logging.error(f"Error creating final video: {e}")
        return None

def extend_video_clip(video_path, extra_duration):
    try:
        # Create an extended version of the video to match target duration
        extended_video = tempfile.mktemp(suffix=".mp4")
        subprocess.run([
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", video_path,
            "-t", str(extra_duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", extended_video
        ], check=True)
        return extended_video
    except Exception as e:
        logging.error(f"Error extending video clip: {e}")
        return video_path

# Main Streamlit Interface
st.title("CX Overview Podcast Generator")
url_input = st.text_input("Enter the webpage URL:")
logo_url = st.text_input("Enter the logo image URL:")
add_text_overlay_flag = st.checkbox("Add Text Overlays (display text below video)")
filter_option = st.selectbox("Select a Video Filter:", ["None", "Grayscale", "Sepia"])
transition_option = st.selectbox("Select Image Transition:", ["None", "Fade", "Zoom"])
total_duration = st.number_input("Total Video Duration (seconds):", min_value=10, value=60, step=10)

if st.button("Generate Podcast"):
    images, text = scrape_images_and_text(url_input)
    valid_images = filter_valid_images(images)

    if valid_images and text:
        st.write("Generating podcast script...")
        script = generate_script(text, max_words_for_duration(total_duration))
        
        audio_path = generate_audio_with_openai(script)
        if audio_path:
            st.write("Audio generated. Calculating audio duration...")
            
            # Retrieve audio duration for alignment
            audio_duration_command = [
                "ffprobe", "-i", audio_path, "-show_entries", "format=duration",
                "-v", "quiet", "-of", "csv=p=0"
            ]
            audio_duration = float(subprocess.check_output(audio_duration_command).strip())
            st.write(f"Audio duration: {audio_duration:.2f} seconds")

            # Calculate per-image duration
            num_images = len(valid_images)
            duration_per_clip = total_duration / num_images
            st.write(f"Allocating {duration_per_clip:.2f} seconds per clip for {num_images} images.")

            # Generate video clips
            st.write("Generating video clips with selected effect...")
            video_clips = [
                generate_video_clip_with_effects(logo_url, 5, None, filter_option, transition_option, effect_option)
            ]
            for img_url in valid_images:
                video_clips.append(generate_video_clip_with_effects(img_url, duration_per_clip, None, filter_option, transition_option, effect_option))

            # Define the end image URL
            end_image_url = "https://raw.githubusercontent.com/scooter7/carnegiedailypodcast/main/cx.jpg"

            # Create final video, including the end image
            st.write("Combining video clips and audio...")
            final_video = create_final_video(
                video_clips,
                audio_path,
                end_image_url,
                duration_per_clip,
                filter_option,
                transition_option
            )
            if final_video:
                st.video(final_video)
                if add_text_overlay_flag:
                    st.text_area("Podcast Script", script, height=300)
                st.download_button("Download Video", open(final_video, "rb"), "CX_Overview.mp4")
                st.download_button("Download Script", script, "script.txt")
            else:
                st.error("Error generating final video.")
        else:
            st.error("Error generating audio. Please check your input.")
    else:
        st.error("No valid images or text found. Please check the URL.")
