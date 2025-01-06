import streamlit as st
from bs4 import BeautifulSoup
import requests
import openai
import replicate
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from PIL import Image
from io import BytesIO
import logging
import os

logging.basicConfig(level=logging.INFO)

# Constants
WORDS_PER_MINUTE = 150
REPLICATE_MODELS = {
    "Cartoon": "gpt-viz/cartoon-style",
    "Anime": "cjwbw/videocrafter2-anime",
    "Sketch": "catacolabs/pencil-sketch",
}

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

# Function to dynamically generate a summary script based on duration
def generate_dynamic_summary_with_duration(all_text, desired_duration, school_name="the highlighted schools"):
    opening_message = (
        f"Welcome to the CollegeXpress Campus Countdown, where we explore colleges and universities around the country to help you find great schools to apply to! "
        f"Today we’re highlighting {school_name}. Let’s get started!"
    )
    closing_message = (
        "Don’t forget, you can connect with any of our featured colleges by visiting CollegeXpress.com. "
        "Just click the green “Yes, connect me!” buttons when you see them on the site, and then the schools you’re interested in will reach out to you with more information! "
        "You can find the links to these schools in the description below. Don’t forget to follow us on social media @CollegeXpress. "
        "Until next time, happy college hunting!"
    )
    max_words = (desired_duration // 60) * WORDS_PER_MINUTE
    system_prompt = (
        f"As a show host, summarize the text narratively to fit within {max_words} words. Include key details like location, accolades, and testimonials. "
        f"Speak naturally in terms of pace, and be enthusiastic in your tone."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this text: {all_text}"},
            ],
        )
        dynamic_summary = response.choices[0].message.content.strip()
        return f"{opening_message}\n\n{dynamic_summary}\n\n{closing_message}"
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return f"{opening_message}\n\n[Error generating summary]\n\n{closing_message}"

# Function to generate audio from a script using OpenAI
def generate_audio_with_openai(script, voice="shimmer"):
    try:
        response = openai.Audio.create(model="tts-1", voice=voice, input=script)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# Function to apply effects using Replicate API
def apply_replicate_effect(image_path, effect):
    try:
        model_name = REPLICATE_MODELS.get(effect)
        if not model_name or effect == "None":
            return image_path

        replicate_api_key = st.secrets["replicate"]["api_key"]
        client = replicate.Client(api_token=replicate_api_key)

        with open(image_path, "rb") as img_file:
            response = client.run(model_name, input={"image": img_file})

        if not response:
            logging.error("Replicate API returned an empty response.")
            return image_path

        output_path = tempfile.mktemp(suffix=".jpg")
        with open(output_path, "wb") as f:
            f.write(requests.get(response).content)
        return output_path
    except Exception as e:
        logging.error(f"Error applying effect '{effect}': {e}")
        return image_path

# Function to create a video clip with optional effects
def create_video_clip_with_effect(image_path, effect, duration=5, fps=24):
    try:
        processed_image_path = apply_replicate_effect(image_path, effect)
        return ImageClip(processed_image_path).set_duration(duration).set_fps(fps)
    except Exception as e:
        logging.error(f"Error creating video clip: {e}")
        return None

# Function to combine video clips and synchronize with audio
def create_final_video_with_audio(video_clips, audio_path, output_path, fps=24):
    try:
        if not video_clips:
            raise ValueError("No video clips provided.")

        audio = AudioFileClip(audio_path)
        final_clip = concatenate_videoclips(video_clips, method="compose")
        final_clip = final_clip.set_audio(audio)
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        return output_path
    except Exception as e:
        logging.error(f"Error during final video creation: {e}")
        return None

# Streamlit UI
st.title("AI-Powered Video Generator")

# URL and image input fields
def url_image_inputs():
    urls = st.text_area("Enter URLs (comma-separated)").split(",")
    images_per_url = {}
    for url in urls:
        st.subheader(f"Images for URL: {url}")
        images = []
        num_images = st.number_input(f"Number of images for {url}", min_value=1, value=1)
        for _ in range(num_images):
            image_url = st.text_input("Enter Image URL")
            if image_url:
                images.append(image_url)
        images_per_url[url] = images
    return urls, images_per_url

urls, url_image_map = url_image_inputs()
effect = st.selectbox("Select Effect", ["None", "Cartoon", "Anime", "Sketch"])
video_duration = st.number_input("Video Duration (seconds)", min_value=10, step=5, value=60)

if st.button("Generate Video"):
    video_clips = []
    combined_text = ""

    # Process URLs and Images
    for url, images in url_image_map.items():
        text = scrape_text_from_url(url)
        combined_text += text if text else ""
        for img_url in images:
            image = download_image_from_url(img_url)
            if image:
                temp_img_path = tempfile.mktemp(suffix=".jpg")
                image.save(temp_img_path)
                video_clip = create_video_clip_with_effect(temp_img_path, effect)
                if video_clip:
                    video_clips.append(video_clip)

    # Generate script and audio
    script = generate_dynamic_summary_with_duration(combined_text, video_duration)
    audio_path = generate_audio_with_openai(script)

    # Compile final video
    if video_clips and audio_path:
        final_video_path = tempfile.mktemp(suffix=".mp4")
        final_video = create_final_video_with_audio(video_clips, audio_path, final_video_path)
        if final_video:
            st.video(final_video)
            st.download_button("Download Video", open(final_video, "rb"), "video.mp4")
