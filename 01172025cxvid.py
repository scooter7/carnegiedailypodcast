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
        logging.error(f"Error generating dynamic summary: {e}")
        return f"{opening_message}\n\n[Error generating dynamic summary]\n\n{closing_message}"

# Function to create a video clip with optional effects
def create_video_clip_with_effect(image_path, duration=5, fps=24):
    try:
        return ImageClip(image_path).set_duration(duration).set_fps(fps)
    except Exception as e:
        logging.error(f"Error creating video clip: {e}")
        return None

# Function to combine video clips with intro/outro and synchronize with audio
def create_final_video_with_audio_sync(video_clips, script_audio_path, output_path, fps=24):
    try:
        if not video_clips:
            raise ValueError("No video clips provided for final video creation.")

        # Load audio to determine total duration
        if script_audio_path:
            audio = AudioFileClip(script_audio_path)
            total_audio_duration = audio.duration
        else:
            raise ValueError("Audio file is required to create a synchronized video.")

        # Repeat and trim video clips to match total duration
        repeated_clips = []
        current_duration = 0
        while current_duration < total_audio_duration:
            for clip in video_clips:
                if current_duration + clip.duration > total_audio_duration:
                    # Trim the last clip to match the remaining duration
                    clip = clip.subclip(0, total_audio_duration - current_duration)
                    repeated_clips.append(clip)
                    current_duration = total_audio_duration
                    break
                repeated_clips.append(clip)
                current_duration += clip.duration

        # Concatenate all clips
        combined_clip = concatenate_videoclips(repeated_clips, method="compose")

        # Add audio to the video
        combined_clip = combined_clip.set_audio(audio)

        # Ensure the final video length matches the audio duration
        final_clip = combined_clip.subclip(0, total_audio_duration)

        # Write the final video file
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        return output_path
    except Exception as e:
        logging.error(f"Error generating final video: {e}")
        return None

# Streamlit UI
st.title("Custom Video and Script Generator with Image Assignment")

# Step 1: Input URLs and images
urls = st.text_area("Enter URLs (one per line):").splitlines()
video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

if st.button("Generate Script"):
    combined_text = ""
    for url in urls:
        combined_text += scrape_text_from_url(url)
    
    if combined_text:
        script = generate_dynamic_summary_with_duration(combined_text, video_duration)
        user_script = st.text_area("Generated Script (Modify if needed):", script, height=300)

        # Step 2: Image assignment
        st.subheader("Assign Images to Script Sections")
        section_images = {}
        script_sections = user_script.split("\n\n")
        for i, section in enumerate(script_sections):
            st.text_area(f"Script Section {i + 1}:", section, key=f"section_{i}")
            section_images[i] = st.text_input(f"Image URL for Section {i + 1}:", key=f"image_{i}")
        
        # Generate video
        if st.button("Create Video"):
            video_clips = []
            
            # Add intro image
            intro_img = download_image_from_url(INTRO_OUTRO_IMAGE)
            if intro_img:
                intro_path = tempfile.mktemp(suffix=".png")
                intro_img.save(intro_path, "PNG")
                video_clips.append(ImageClip(intro_path).set_duration(5))

            # Add section images
            for section_index, img_url in section_images.items():
                if img_url:
                    image = download_image_from_url(img_url)
                    if image:
                        image_path = tempfile.mktemp(suffix=".png")
                        image.save(image_path, "PNG")
                        video_clips.append(create_video_clip_with_effect(image_path, duration=video_duration / len(script_sections)))
            
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
                st.download_button("Download Script", user_script, "script.txt")
