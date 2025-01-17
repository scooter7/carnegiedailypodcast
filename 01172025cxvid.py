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
import os

logging.basicConfig(level=logging.INFO)

# Constants
WORDS_PER_MINUTE = 150
INTRO_OUTRO_IMAGE = "https://github.com/scooter7/carnegiedailypodcast/blob/ffe1af9fb3bb7e853bdd4e285d0b699ceb452208/cx.jpg"

# Initialize session state
if "urls" not in st.session_state:
    st.session_state.urls = []
if "script" not in st.session_state:
    st.session_state.script = ""
if "num_sections" not in st.session_state:
    st.session_state.num_sections = 3
if "section_images" not in st.session_state:
    st.session_state.section_images = {}

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
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None

# Function to generate a summary script
def generate_dynamic_summary_with_duration(all_text, desired_duration, num_sections, school_name="the highlighted schools"):
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
        f"As a show host, summarize the text narratively to fit within {max_words} words and split it into {num_sections} sections. "
        f"Include key details like location, accolades, and testimonials. Speak naturally in terms of pace, and be enthusiastic in your tone."
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
        logging.error(f"Error generating dynamic summary: {e}")
        return f"{opening_message}\n\n[Error generating dynamic summary]\n\n{closing_message}"

# Function to create a video clip
def create_video_clip_with_effect(image_path, duration=5, fps=24):
    try:
        return ImageClip(image_path).set_duration(duration).set_fps(fps)
    except Exception as e:
        logging.error(f"Error creating video clip: {e}")
        return None

# UI: Input URLs
st.title("Custom Video and Script Generator with Image Assignment")
urls = st.text_area("Enter URLs (one per line):", value="\n".join(st.session_state.urls), height=100)
st.session_state.urls = urls.splitlines()

# UI: Define video duration and number of sections
video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)
st.session_state.num_sections = st.number_input("Number of Sections for Script:", min_value=1, step=1, value=st.session_state.num_sections)

# Generate Script
if st.button("Generate Script"):
    combined_text = ""
    for url in st.session_state.urls:
        combined_text += scrape_text_from_url(url)

    if combined_text:
        st.session_state.script = generate_dynamic_summary_with_duration(
            combined_text, video_duration, st.session_state.num_sections
        )

# Display Script and Allow Modifications
if st.session_state.script:
    st.subheader("Generated Script")
    st.session_state.script = st.text_area("Edit Script Below:", st.session_state.script, height=300)

# Image Assignment for Sections
if st.session_state.script:
    st.subheader("Assign Images to Script Sections")
    sections = st.session_state.script.split("\n\n")
    st.session_state.section_images = {
        i: st.text_input(f"Image URL for Section {i + 1}:", st.session_state.section_images.get(i, ""))
        for i in range(len(sections))
    }

# Generate Video
if st.button("Create Video"):
    video_clips = []

    # Add intro image
    intro_img = download_image_from_url(INTRO_OUTRO_IMAGE)
    if intro_img:
        intro_path = tempfile.mktemp(suffix=".png")
        intro_img.save(intro_path, "PNG")
        video_clips.append(ImageClip(intro_path).set_duration(5))

    # Add section images
    for i, img_url in st.session_state.section_images.items():
        if img_url:
            image = download_image_from_url(img_url)
            if image:
                image_path = tempfile.mktemp(suffix=".png")
                image.save(image_path, "PNG")
                video_clips.append(create_video_clip_with_effect(image_path, duration=video_duration / len(sections)))

    # Add outro image
    outro_img = download_image_from_url(INTRO_OUTRO_IMAGE)
    if outro_img:
        outro_path = tempfile.mktemp(suffix=".png")
        outro_img.save(outro_path, "PNG")
        video_clips.append(ImageClip(outro_path).set_duration(5))

    # Combine video clips
    if video_clips:
        final_video_path = tempfile.mktemp(suffix=".mp4")
        combined_clip = concatenate_videoclips(video_clips, method="compose")
        combined_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac", fps=24)
        
        # Display and download
        st.video(final_video_path)
        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
        st.download_button("Download Script", st.session_state.script, "script.txt")
