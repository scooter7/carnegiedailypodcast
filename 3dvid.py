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
import open3d as o3d  # New addition for 3D effects
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

# New: Function to create a 3D-like transformation using Open3D
def transform_image_to_3d(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create height map
        height, width = gray.shape
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = gray / 255.0  # Normalize grayscale to create height

        # Create Open3D point cloud
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Visualize and save as image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        output_path = tempfile.mktemp(suffix=".png")
        vis.capture_screen_image(output_path)
        vis.destroy_window()

        return output_path
    except Exception as e:
        logging.error(f"Error generating 3D transformation: {e}")
        return None

# Function to process images with effects, including 3D transformation
def process_image(image_path, effect):
    if effect == "Cartoon":
        return apply_cartoon_effect(image_path)
    elif effect == "Anime":
        return apply_anime_effect(image_path)
    elif effect == "Sketch":
        return apply_sketch_effect(image_path)
    elif effect == "3D":
        return transform_image_to_3d(image_path)
    else:
        return image_path

# Function to create video clips from processed images
def create_video_clip_with_effect(image_path, effect, duration=5, fps=24):
    try:
        processed_image_path = process_image(image_path, effect)
        clip = ImageClip(processed_image_path).set_duration(duration).set_fps(fps)
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

# URL and image input
urls = st.text_input("Enter webpage URLs (comma-separated):")
effect_option = st.selectbox("Select an Effect:", ["None", "Cartoon", "Anime", "Sketch", "3D"])
transition_option = st.selectbox("Select a Transition:", ["None", "Fade", "Slide"])

if st.button("Generate Video"):
    video_clips = []
    url_list = [url.strip() for url in urls.split(",") if url.strip()]

    for url in url_list:
        st.write(f"Processing URL: {url}")
        text = scrape_text_from_url(url)
        summary = generate_summary(text, max_words=150)

        images = st.file_uploader(f"Upload images for {url}", accept_multiple_files=True, type=["jpg", "png"])

        for image_file in images:
            temp_image_path = tempfile.mktemp(suffix=".jpg")
            with open(temp_image_path, "wb") as f:
                f.write(image_file.read())
            clip = create_video_clip_with_effect(temp_image_path, effect_option)
            if clip:
                video_clips.append(clip)

    if video_clips:
        audio_path = generate_audio_with_openai(summary)
        final_video_path = tempfile.mktemp(suffix=".mp4")
        create_final_video_with_transitions(video_clips, audio_path, final_video_path, transition_type=transition_option)
        st.video(final_video_path)
