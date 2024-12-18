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
import subprocess

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

# Effects functions
def apply_cartoon_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_anime_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    anime = cv2.bitwise_and(blurred, blurred, mask=edges)
    return anime

def apply_sketch_effect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def process_image(image_path, effect):
    image = cv2.imread(image_path)
    if effect == "Cartoon":
        return apply_cartoon_effect(image)
    elif effect == "Anime":
        return apply_anime_effect(image)
    elif effect == "Sketch":
        return apply_sketch_effect(image)
    else:
        return image

# Function to transform an image into 3D using Blender
def transform_image_to_3d(image_path):
    try:
        # Temporary output for the 3D-transformed image
        output_path = tempfile.mktemp(suffix=".png")

        # Path to Blender executable and the 3D script
        blender_executable = "/path/to/blender"  # Replace with the actual path to Blender
        blender_script = "/path/to/blender_image_3d.py"  # Replace with the actual Blender script path

        # Run Blender in background mode with the Python script
        subprocess.run([
            blender_executable,
            "--background",  # Run without UI
            "--python", blender_script,
            "--", image_path, output_path
        ], check=True)

        return output_path
    except Exception as e:
        logging.error(f"Error transforming image to 3D: {e}")
        return None

# Function to create a video clip from an image
def create_video_clip_with_effect(image_path, effect, use_3d=False, duration=5, fps=24):
    try:
        if use_3d:
            # Transform the image to 3D using Blender
            image_path = transform_image_to_3d(image_path)

        # Apply additional effects (e.g., Cartoon, Anime)
        processed_image_path = process_image(image_path, effect)
        
        # Create a video clip from the processed image
        clip = ImageClip(processed_image_path).set_duration(duration).set_fps(fps)
        return clip
    except Exception as e:
        logging.error(f"Error creating video clip with effect: {e}")
        return None

# Function to generate the final video with transitions
def create_final_video_with_transitions(video_clips, script_audio_path, output_path, transition_type="None", fps=24):
    try:
        # Apply transitions between clips
        if transition_type == "Fade":
            video_clips = [
                clip.crossfadein(1) if i > 0 else clip
                for i, clip in enumerate(video_clips)
            ]

        # Combine video clips
        combined_clip = concatenate_videoclips(video_clips, method="compose")

        # Add audio if provided
        if script_audio_path:
            audio = AudioFileClip(script_audio_path)
            combined_clip = combined_clip.set_audio(audio)

            # Extend video duration to match audio length
            audio_duration = audio.duration
            combined_clip = combined_clip.loop(duration=audio_duration)

        # Write the final video
        combined_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        return output_path
    except Exception as e:
        logging.error(f"Error generating final video: {e}")
        return None

# Streamlit UI
st.title("Custom Video and Script Generator")

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
    effect_option = st.selectbox("Select an Effect:", ["None", "Cartoon", "Anime", "Sketch"])
    use_3d = st.checkbox("Apply 3D Transformation to Images", value=False)
    transition_option = st.selectbox("Select a Transition:", ["None", "Fade"])

    if st.button("Generate Video"):
        final_script = ""

        for url, images in url_image_map.items():
            st.write(f"Processing URL: {url}")
            text = scrape_text_from_url(url)
            summary = generate_summary(text, max_words=150)
            final_script += f"\n{summary}"

            for img_url in images:
                image = download_image_from_url(img_url)
                if image:
                    st.image(image, caption=f"Processing {img_url}")
                    temp_image_path = tempfile.mktemp(suffix=".jpg")
                    image.save(temp_image_path)
                    video_clip = create_video_clip_with_effect(temp_image_path, effect_option, use_3d=use_3d)
                    if video_clip:
                        video_clips.append(video_clip)

        # Process video clips if they exist
        if video_clips:
            st.write("Generating audio from script...")
            audio_path = generate_audio_with_openai(final_script, voice="alloy")

            if audio_path:
                st.write("Combining video clips into final video...")
                final_video_path = tempfile.mktemp(suffix=".mp4")

                try:
                    # Create final video with transitions
                    final_video_path = create_final_video_with_transitions(video_clips, audio_path, final_video_path, transition_type=transition_option)
                    if final_video_path:
                        st.video(final_video_path)
                        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
                        st.download_button("Download Script", final_script, "script.txt")
                except Exception as e:
                    logging.error(f"Error creating final video: {e}")
                    st.error("Failed to create the final video. Please check the logs.")
            else:
                st.write("No audio generated. Creating a silent video...")
                final_video_path = tempfile.mktemp(suffix=".mp4")

                try:
                    final_video_path = create_final_video_with_transitions(video_clips, None, final_video_path, transition_type=transition_option)
                    if final_video_path:
                        st.video(final_video_path)
                        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
                        st.download_button("Download Script", final_script, "script.txt")
                except Exception as e:
                    logging.error(f"Error creating silent video: {e}")
                    st.error("Failed to create the silent video. Please check the logs.")
        else:
            st.error("No valid video clips were created. Please check your input and try again.")
