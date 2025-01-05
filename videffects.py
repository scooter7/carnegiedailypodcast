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

logging.basicConfig(level=logging.INFO)

def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]
    except Exception as e:
        logging.error(f"Error scraping text from {url}: {e}")
        return ""

def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

def generate_summary_with_narration(text, max_words, school_name="the highlighted schools"):
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
    system_prompt = (
        "You are a podcast host. Summarize the text narratively. Include key details like location, accolades, and testimonials. "
        "End with an engaging note that highlights the school's strengths."
    )
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Summarize this text: {text} Limit: {max_words} words."},
            ],
        )
        dynamic_summary = response.choices[0].message.content.strip()
        full_script = f"{opening_message}\n\n{dynamic_summary}\n\n{closing_message}"
        return full_script
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return f"{opening_message}\n\n[Error generating dynamic summary]\n\n{closing_message}"

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

def create_video_clip_with_effect(image_path, effect, duration=5, fps=24):
    try:
        image = cv2.imread(image_path)
        processed_image = process_image(image_path, effect)
        output_path = tempfile.mktemp(suffix=".jpg")
        cv2.imwrite(output_path, processed_image)
        clip = ImageClip(output_path).set_duration(duration).set_fps(fps)
        return clip
    except Exception as e:
        logging.error(f"Error creating video clip with effect: {e}")
        return None

def create_final_video_with_transitions(video_clips, script_audio_path, output_path, transition_type="None", fps=24):
    try:
        if transition_type == "Fade":
            video_clips = [
                clip.crossfadein(1) if i > 0 else clip
                for i, clip in enumerate(video_clips)
            ]
        combined_clip = concatenate_videoclips(video_clips, method="compose")
        if script_audio_path:
            audio = AudioFileClip(script_audio_path)
            combined_clip = combined_clip.set_audio(audio)
            audio_duration = audio.duration
            combined_clip = combined_clip.loop(duration=audio_duration)
        combined_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        return output_path
    except Exception as e:
        logging.error(f"Error generating final video: {e}")
        return None

st.title("Custom Video and Script Generator")

def url_input_fields():
    urls = []
    with st.container():
        st.subheader("Enter Page URLs")
        num_urls = st.number_input("Number of URLs", min_value=1, value=1, step=1)
        for i in range(num_urls):
            url = st.text_input(f"URL #{i + 1}", placeholder="Enter a webpage URL")
            urls.append(url)
    return urls

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

urls = url_input_fields()
video_clips = []

if urls:
    url_image_map = image_input_fields(urls)
    effect_option = st.selectbox("Select an Effect:", ["None", "Cartoon", "Anime", "Sketch"])
    transition_option = st.selectbox("Select a Transition:", ["None", "Fade", "Slide"])

    if st.button("Generate Video"):
        final_script = ""

        for url, images in url_image_map.items():
            st.write(f"Processing URL: {url}")
            text = scrape_text_from_url(url)
            script = generate_summary_with_narration(text, max_words=150, school_name="this amazing school")
            final_script += f"\n{script}"

            for img_url in images:
                image = download_image_from_url(img_url)
                if image:
                    st.image(image, caption=f"Processing {img_url}")
                    temp_image_path = tempfile.mktemp(suffix=".jpg")
                    image.save(temp_image_path)
                    video_clip = create_video_clip_with_effect(temp_image_path, effect_option)
                    if video_clip:
                        video_clips.append(video_clip)

        if video_clips:
            audio_path = generate_audio_with_openai(final_script, voice="alloy")
            if audio_path:
                final_video_path = tempfile.mktemp(suffix=".mp4")
                try:
                    final_video_path = create_final_video_with_transitions(
                        video_clips, audio_path, final_video_path, transition_type=transition_option
                    )
                    if final_video_path:
                        st.video(final_video_path)
                        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
                        st.download_button("Download Script", final_script, "script.txt")
                except Exception as e:
                    logging.error(f"Error creating final video: {e}")
                    st.error("Failed to create the final video.")
            else:
                st.write("No audio generated. Creating a silent video...")
                final_video_path = tempfile.mktemp(suffix=".mp4")
                try:
                    final_video_path = create_final_video_with_transitions(
                        video_clips, None, final_video_path, transition_type=transition_option
                    )
                    if final_video_path:
                        st.video(final_video_path)
                        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
                        st.download_button("Download Script", final_script, "script.txt")
                except Exception as e:
                    logging.error(f"Error creating silent video: {e}")
                    st.error("Failed to create the silent video.")
        else:
            st.error("No valid video clips were created. Please check your input and try again.")
