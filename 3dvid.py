import streamlit as st
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import openai
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from PIL import Image
from io import BytesIO
import logging
import cv2
import numpy as np
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to scrape text content from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]
    except Exception as e:
        logging.error(f"Error scraping text from {url}: {e}")
        return ""

# Function to download an image from a URL
def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

# Function to generate a summary using OpenAI
def generate_summary(text, max_words):
    system_prompt = (
        "You are a podcast host. Summarize the text narratively. Include key details and end with an engaging note."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this text: {text} Limit: {max_words} words."},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return ""

# Generate speech using OpenAI TTS
def generate_audio_with_openai(script, voice="alloy"):
    try:
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=script)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# Function to apply a 3D transformation to an image using OpenCV
def apply_3d_transformation(image_path):
    try:
        image = cv2.imread(image_path)
        rows, cols, _ = image.shape

        # Define 3D transformation matrix
        src_points = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
        dst_points = np.float32([[0, 0], [int(0.8 * cols), int(0.2 * rows)], [int(0.2 * cols), int(0.9 * rows)]])
        matrix = cv2.getAffineTransform(src_points, dst_points)

        transformed = cv2.warpAffine(image, matrix, (cols, rows))
        return transformed
    except Exception as e:
        logging.error(f"Error applying 3D transformation: {e}")
        return None

# Process the image with an optional 3D transformation
def process_image_with_3d(image_path, apply_3d=False):
    try:
        if apply_3d:
            transformed_image = apply_3d_transformation(image_path)
            if transformed_image is not None:
                output_path = tempfile.mktemp(suffix=".jpg")
                cv2.imwrite(output_path, transformed_image)
                return output_path
        return image_path
    except Exception as e:
        logging.error(f"Error processing image with 3D transformation: {e}")
        return None

# Function to create a video clip from an image
def create_video_clip_with_effect(image_path, effect, duration=5, fps=24, apply_3d=False):
    try:
        processed_image_path = process_image_with_3d(image_path, apply_3d)
        image = cv2.imread(processed_image_path)
        output_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(output_path, image)
        clip = ImageClip(output_path).set_duration(duration).set_fps(fps)
        return clip
    except Exception as e:
        logging.error(f"Error creating video clip with effect: {e}")
        return None

# Function to generate the final video with transitions
def create_final_video_with_transitions(video_clips, script_audio_path, output_path, transition_type="None", fps=24):
    try:
        # Combine video clips
        combined_clip = concatenate_videoclips(video_clips, method="compose")

        # Add audio if provided
        if script_audio_path:
            audio = AudioFileClip(script_audio_path)
            combined_clip = combined_clip.set_audio(audio)

        # Write the final video
        combined_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        return output_path
    except Exception as e:
        logging.error(f"Error generating final video: {e}")
        return None

# Streamlit UI
st.title("Custom Video and Script Generator with 3D Effects")

# Add a URL input field
def url_input_fields():
    urls = []
    with st.container():
        st.subheader("Enter Page URLs")
        num_urls = st.number_input("Number of URLs", min_value=1, value=1, step=1)
        for i in range(num_urls):
            url = st.text_input(f"URL #{i + 1}", placeholder="Enter a webpage URL")
            urls.append(url)
    return urls

# Add image URL input fields per URL
def image_input_fields(urls):
    url_image_map = {}
    for i, url in enumerate(urls):
        st.subheader(f"Images for {url}")
        num_images = st.number_input(
            f"Number of images for URL #{i + 1}", min_value=1, value=1, step=1, key=f"num_images_{i}"
        )
        images = []
        for j in range(num_images):
            image_url = st.text_input(
                f"Image #{j + 1} for URL #{i + 1}", placeholder="Enter an image URL", key=f"image_url_{i}_{j}"
            )
            images.append(image_url)
        url_image_map[url] = images
    return url_image_map

# Main logic
urls = url_input_fields()
video_clips = []  # Initialize video_clips

if urls:
    url_image_map = image_input_fields(urls)
    apply_3d_option = st.checkbox("Apply 3D Transformation")
    if st.button("Generate Video"):
        for url, images in url_image_map.items():
            for img_url in images:
                image = download_image_from_url(img_url)
                if image:
                    temp_image_path = tempfile.mktemp(suffix=".jpg")
                    image.save(temp_image_path)
                    video_clip = create_video_clip_with_effect(temp_image_path, effect=None, apply_3d=apply_3d_option)
                    if video_clip:
                        video_clips.append(video_clip)

        if video_clips:
            final_video_path = tempfile.mktemp(suffix=".mp4")
            create_final_video_with_transitions(video_clips, None, final_video_path)
            st.video(final_video_path)
