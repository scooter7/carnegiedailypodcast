import streamlit as st
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import openai
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip
from PIL import Image
from io import BytesIO
import logging
import os
import replicate
import uuid
import cv2

logging.basicConfig(level=logging.INFO)

# Constants
WORDS_PER_MINUTE = 150

# 🔹 Initialize Streamlit App
st.title("Custom Video and Script Generator")

# Ensure session state for num_urls
if "num_urls" not in st.session_state:
    st.session_state.num_urls = 1  # Default value

# 🔹 Function to scrape text content from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)[:5000]
    except Exception as e:
        logging.error(f"Error scraping text from {url}: {e}")
        return ""

# 🔹 Function to download an image from a URL
def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img.convert("RGB") if img.mode in ("RGBA", "P") else img
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

# 🔹 Function to get URLs from user input
def url_input_fields():
    urls = []
    with st.container():
        st.subheader("Enter Page URLs")
        num_urls = st.number_input("Number of URLs", min_value=1, value=st.session_state.num_urls, step=1, key="unique_num_urls")
        if num_urls != st.session_state.num_urls:
            st.session_state.num_urls = num_urls

        for i in range(num_urls):
            url = st.text_input(f"URL #{i + 1}", placeholder="Enter a webpage URL", key=f"unique_url_{i}")
            if url:
                urls.append(url)
    return urls

# 🔹 Function to get image URLs for each webpage
def image_input_fields(urls):
    url_image_map = {}
    for i, url in enumerate(urls):
        with st.container():
            st.subheader(f"Images for [{url}]({url})")
            num_images = st.number_input(f"Number of images for URL #{i + 1}", min_value=1, value=1, step=1, key=f"unique_num_images_{i}")
            images = []
            for j in range(num_images):
                image_url = st.text_input(f"Image #{j + 1} for URL #{i + 1}", placeholder="Enter an image URL", key=f"unique_image_url_{i}_{j}")
                if image_url:
                    images.append(image_url)
            url_image_map[url] = images
    return url_image_map

# Get URLs
urls = url_input_fields()

# Get image URLs only if URLs exist
url_image_map = image_input_fields(urls) if urls else {}

# 🔹 Function to generate a summary script based on duration
def generate_dynamic_summary_with_duration(all_text, desired_duration, school_name="these amazing schools"):
    opening_message = f"Welcome to the CollegeXpress Campus Countdown! Today we’re highlighting {school_name}. Let’s get started!"
    closing_message = "Visit CollegeXpress.com for more information. Until next time, happy college hunting!"
    
    max_words = (desired_duration // 60) * WORDS_PER_MINUTE
    system_prompt = f"As a show host, summarize the text narratively to fit within {max_words} words."

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Summarize this text: {all_text}"}],
        )
        summary = response.choices[0].message.content.strip()
        return f"{opening_message}\n\n{summary}\n\n{closing_message}"
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return f"{opening_message}\n\n[Error generating summary]\n\n{closing_message}"

# 🔹 Function to generate audio from a script using OpenAI
def generate_audio_with_openai(script, voice="shimmer"):
    try:
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=script)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# 🔹 Function to apply image effects
def apply_image_effect(image_path, effect):
    try:
        if effect == "None":
            return image_path

        replicate_api_key = st.secrets["replicate"]["api_key"]
        client = replicate.Client(api_token=replicate_api_key)
        
        models = {"Cartoon": "catacolabs/cartoonify", "Anime": "cjwbw/videocrafter2-anime", "Sketch": "catacolabs/pencil-sketch"}
        model_name = models.get(effect)
        
        if not model_name:
            logging.warning(f"Effect '{effect}' is not supported.")
            return image_path

        with open(image_path, "rb") as img_file:
            response = client.run(model_name, input={"image": img_file})

        if not response:
            logging.error(f"Replicate API returned no result for effect '{effect}'.")
            return image_path

        output_path = tempfile.mktemp(suffix=".png")
        with open(output_path, "wb") as f:
            f.write(requests.get(response).content)

        return output_path
    except Exception as e:
        logging.error(f"Error applying effect '{effect}': {e}")
        return image_path

# 🔹 Function to create a video clip with optional effects
def create_video_clip(image_path, duration=5, fps=24):
    try:
        return ImageClip(image_path).set_duration(duration).set_fps(fps)
    except Exception as e:
        logging.error(f"Error creating video clip: {e}")
        return None

# 🔹 Function to combine video clips with audio
def create_final_video(video_clips, audio_path, output_path, fps=24):
    try:
        if not video_clips:
            raise ValueError("No video clips provided.")

        audio = AudioFileClip(audio_path)
        total_duration = audio.duration

        final_clips = []
        current_duration = 0
        while current_duration < total_duration:
            for clip in video_clips:
                if current_duration + clip.duration > total_duration:
                    clip = clip.subclip(0, total_duration - current_duration)
                    final_clips.append(clip)
                    current_duration = total_duration
                    break
                final_clips.append(clip)
                current_duration += clip.duration

        final_video = concatenate_videoclips(final_clips, method="compose").set_audio(audio)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        return output_path
    except Exception as e:
        logging.error(f"Error generating final video: {e}")
        return None

# **Process Video Generation**
if st.button("Generate Video"):
    video_clips = []
    combined_text = "\n".join(scrape_text_from_url(url) for url in urls if url)

    if combined_text:
        final_script = generate_dynamic_summary_with_duration(combined_text, 60)
        audio_path = generate_audio_with_openai(final_script)

        if audio_path:
            for url, images in url_image_map.items():
                for img_url in images:
                    image = download_image_from_url(img_url)
                    if image:
                        temp_image_path = tempfile.mktemp(suffix=".png")
                        image.save(temp_image_path, "PNG")
                        video_clip = create_video_clip(temp_image_path, duration=5)
                        if video_clip:
                            video_clips.append(video_clip)

            if video_clips:
                final_video_path = tempfile.mktemp(suffix=".mp4")
                final_video_path = create_final_video(video_clips, audio_path, final_video_path)
                st.video(final_video_path)
                st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
