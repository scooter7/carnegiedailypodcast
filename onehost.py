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

def download_font(font_url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(font_url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

download_font(font_url, local_font_path)

# Preserve image dimensions
def preserve_image_dimensions(image_url, output_path, duration=5):
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(temp_image.name)

    subprocess.run([
        "ffmpeg", "-y", "-loop", "1", "-i", temp_image.name,
        "-vf", f"scale={img.width}:{img.height}", "-t", str(duration),
        "-pix_fmt", "yuv420p", "-c:v", "libx264", output_path
    ], check=True)

    return output_path

# Generate script
def generate_script(text, max_words):
    system_prompt = """
    You are a podcast host for 'CX Overview.' Generate a robust, fact-based summary of the school at the scraped webpage narrated by Lisa. Include information such as:
    - School's location (city, state) and campus type (urban, rural, etc.)
    - Relevant statistics and faculty-to-student ratio
    - Tuition and financial aid opportunities
    - Testimonials and accolades (if available)
    End with: "For more information, visit CollegeXpress.com (pronounced college express)."
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this text: {text} Limit: {max_words} words."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating script: {e}")
        return ""

# Video transitions
TRANSITIONS = {
    "None": "",
    "Fade": "fade=t=out:st=4:d=1",
    "Dissolve": "xfade=transition=dissolve:duration=1"
}

# Video filters
FILTERS = {
    "None": "",
    "Grayscale": "format=gray",
    "Vignette": "vignette"
}

# Static end image for video
END_IMAGE_URL = "https://raw.githubusercontent.com/scooter7/carnegiedailypodcast/main/cx.jpg"

# Create video with audio and static end image
def create_video_with_audio(logo_url, images, script, audio_path, transition, filter_effect):
    try:
        temp_videos = []

        # Add the logo video
        temp_logo_video = tempfile.mktemp(suffix=".mp4")
        preserve_image_dimensions(logo_url, temp_logo_video, duration=5)
        temp_videos.append(temp_logo_video)

        # Add main content (images with optional text overlays)
        split_texts = textwrap.wrap(script, width=250)[:len(images)]
        for img_url, text in zip(images, split_texts):
            temp_video = tempfile.mktemp(suffix=".mp4")
            preserve_image_dimensions(img_url, temp_video, duration=5)
            temp_videos.append(temp_video)

        # Add the static end image as a final frame
        temp_end_video = tempfile.mktemp(suffix=".mp4")
        preserve_image_dimensions(END_IMAGE_URL, temp_end_video, duration=5)
        temp_videos.append(temp_end_video)

        # Write all video parts into a text file for concatenation
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, "w") as f:
            for video in temp_videos:
                f.write(f"file '{video}'\n")

        # Apply transitions and filters
        filter_chain = FILTERS.get(filter_effect, "")
        transition_chain = TRANSITIONS.get(transition, "")

        final_video = "final_video.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-vf", f"{filter_chain},{transition_chain}" if filter_chain else "",
            "-i", audio_path, "-c:v", "libx264", "-c:a", "aac", "-shortest", final_video
        ], check=True)

        return final_video
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        return None

# Streamlit UI
st.title("CX Overview Podcast Generator")
url_input = st.text_input("Enter the webpage URL:")
logo_url = st.text_input("Enter the logo image URL:")
transition = st.selectbox("Select a video transition:", options=list(TRANSITIONS.keys()))
filter_effect = st.selectbox("Select a video filter:", options=list(FILTERS.keys()))
add_text_overlay_flag = st.checkbox("Add text overlays to images", value=True)
duration = st.radio("Video duration (seconds):", [30, 45, 60])

if st.button("Generate Podcast"):
    images, text = scrape_images_and_text(url_input)
    valid_images = filter_valid_images(images)
    if valid_images and text:
        script = generate_script(text, max_words_for_duration(duration))
        audio_path = generate_audio_with_openai(script)
        
        # Script as downloadable text
        st.download_button("Download Script", script, "script.txt")
        
        if audio_path:
            final_video = create_video_with_audio(logo_url, valid_images, script, audio_path, transition, filter_effect)
            if final_video:
                st.video(final_video)
                st.download_button("Download Video", open(final_video, "rb"), "CX_Overview.mp4")
    else:
        st.error("No valid images or text found. Please check the URL.")
