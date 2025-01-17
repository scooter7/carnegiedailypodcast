import streamlit as st
from bs4 import BeautifulSoup
import requests
import openai
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips
from PIL import Image
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)

# Constants
WORDS_PER_MINUTE = 150
INTRO_OUTRO_IMAGE = "https://github.com/scooter7/carnegiedailypodcast/blob/ffe1af9fb3bb7e853bdd4e285d0b699ceb452208/cx.jpg"

# Initialize session state
if "master_script" not in st.session_state:
    st.session_state.master_script = ""
if "sections" not in st.session_state:
    st.session_state.sections = []
if "section_images" not in st.session_state:
    st.session_state.section_images = {}
if "num_sections" not in st.session_state:
    st.session_state.num_sections = 0

# Function to scrape text content from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logging.error(f"Error scraping text from {url}: {e}")
        return ""

# Function to dynamically generate a summary script based on duration
def generate_dynamic_summary_with_duration(all_text, desired_duration, school_name="the highlighted schools"):
    opening_message = (
        f"Welcome to the CollegeXpress Campus Countdown, where we explore colleges and universities around the country to help you find great schools to apply to! "
        f"Today we’re highlighting {school_name}. Let’s get started!"
    )
    closing_message = (
        "Don’t forget, you can connect with any of our featured colleges by visiting CollegeXpress.com. "
        "Just click the green 'Yes, connect me!' buttons when you see them on the site, and then the schools you’re interested in will reach out to you with more information! "
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
        full_script = f"{opening_message}\n\n{dynamic_summary}\n\n{closing_message}"
        return full_script
    except Exception as e:
        logging.error(f"Error generating dynamic summary: {e}")
        return f"{opening_message}\n\n[Error generating dynamic summary]\n\n{closing_message}"

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

# UI: Input URLs
st.title("Custom Video Script and Section Creator")
urls = st.text_area("Enter URLs (one per line):", height=100).splitlines()
video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

# Generate Master Script
if st.button("Generate Master Script"):
    combined_text = ""
    for url in urls:
        combined_text += scrape_text_from_url(url)
    if combined_text:
        st.session_state.master_script = generate_dynamic_summary_with_duration(combined_text, video_duration)

# Editable Master Script
if st.session_state.master_script:
    st.subheader("Master Script")
    st.session_state.master_script = st.text_area(
        "Generated Master Script (editable):", st.session_state.master_script, height=300
    )

    # User decides the number of sections
    st.subheader("Section Configuration")
    st.session_state.num_sections = st.number_input(
        "Number of Sections:", min_value=1, step=1, value=st.session_state.num_sections or 3
    )

    # Create sections dynamically based on user input
    if len(st.session_state.sections) != st.session_state.num_sections:
        st.session_state.sections = ["" for _ in range(st.session_state.num_sections)]
        st.session_state.section_images = {i: "" for i in range(st.session_state.num_sections)}

    # User edits content and assigns images for each section
    st.subheader("Edit Sections and Assign Images")
    for i in range(st.session_state.num_sections):
        st.session_state.sections[i] = st.text_area(
            f"Section {i + 1} Content:", value=st.session_state.sections[i], height=150
        )
        st.session_state.section_images[i] = st.text_input(
            f"Image URL for Section {i + 1}:", value=st.session_state.section_images.get(i, "")
        )

# Generate Video
if st.button("Create Video"):
    video_clips = []

    # Add intro image
    intro_img = download_image_from_url(INTRO_OUTRO_IMAGE)
    if intro_img:
        intro_path = tempfile.mktemp(suffix=".png")
        intro_img.save(intro_path, "PNG")
        video_clips.append(ImageClip(intro_path).set_duration(5))

    # Add user-defined sections with images
    for i, content in enumerate(st.session_state.sections):
        img_url = st.session_state.section_images.get(i)
        if img_url:
            image = download_image_from_url(img_url)
            if image:
                img_path = tempfile.mktemp(suffix=".png")
                image.save(img_path, "PNG")
                section_duration = video_duration / max(len(st.session_state.sections), 1)
                video_clips.append(ImageClip(img_path).set_duration(section_duration))

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
        combined_clip.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
        
        # Display and download
        st.video(final_video_path)
        st.download_button("Download Video", open(final_video_path, "rb"), "video.mp4")
        st.download_button("Download Script", "\n\n".join(st.session_state.sections), "script.txt")
