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

# Function to create a 3D-like perspective transformation of an image
def apply_3d_transform(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to read image from {image_path}")
        return None

    height, width = image.shape[:2]
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst_points = np.float32([
        [width * 0.1, height * 0.1],
        [width * 0.9, height * 0.2],
        [width * 0.2, height * 0.9],
        [width * 0.8, height * 0.8],
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))

    output_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(output_path, transformed_image)
    return output_path

# Function to create a video clip from an image
def create_video_clip_with_effect(image_path, duration=5, fps=24):
    try:
        transformed_image_path = apply_3d_transform(image_path)
        if not transformed_image_path:
            logging.error("3D transformation failed.")
            return None

        clip = ImageClip(transformed_image_path).set_duration(duration).set_fps(fps)
        return clip
    except Exception as e:
        logging.error(f"Error creating video clip with 3D effect: {e}")
        return None

# Function to generate the final video with transitions
def create_final_video_with_transitions(video_clips, script_audio_path, output_path, fps=24):
    try:
        combined_clip = concatenate_videoclips(video_clips, method="compose")

        if script_audio_path:
            audio = AudioFileClip(script_audio_path)
            combined_clip = combined_clip.set_audio(audio)

        combined_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        return output_path
    except Exception as e:
        logging.error(f"Error generating final video: {e}")
        return None

# Streamlit UI
st.title("Custom Video and Script Generator with 3D Effect")

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
video_clips = []

if urls:
    url_image_map = image_input_fields(urls)

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
                    video_clip = create_video_clip_with_effect(temp_image_path)
                    if video_clip:
                        video_clips.append(video_clip)

        if video_clips:
            st.write("Generating audio from script...")
            audio_path = generate_audio_with_openai(final_script, voice="alloy")

            if audio_path:
                st.write("Combining video clips into final video...")
                final_video_path = tempfile.mktemp(suffix=".mp4")

                try:
                    final_video_path = create_final_video_with_transitions(video_clips, audio_path, final_video_path)
                    if final_video_path:
                        st.video(final_video_path)
                        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
                        st.download_button("Download Script", final_script, "script.txt")
                except Exception as e:
                    logging.error(f"Error creating final video: {e}")
                    st.error("Failed to create the final video. Please check the logs.")
            else:
                st.error("Failed to generate audio. Please check the script and try again.")
        else:
            st.error("No valid video clips were created. Please check your input and try again.")
