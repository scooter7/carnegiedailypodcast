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
import replicate

logging.basicConfig(level=logging.INFO)

# Approximate words-per-minute rate for narration
WORDS_PER_MINUTE = 150

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

def download_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        # Ensure all images are in RGBA mode
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        return img
    except Exception as e:
        logging.error(f"Error downloading or processing image from {url}: {e}")
        return None

for img_url in images:
    image = download_image_from_url(img_url)
    if image:
        st.image(image, caption=f"Processing {img_url}")
        temp_image_path = tempfile.mktemp(suffix=".png")  # Always use PNG
        try:
            # Save the image as PNG
            image.save(temp_image_path, "PNG")
            logging.info(f"Image saved as PNG at {temp_image_path}")

            # Create a video clip with the saved image
            video_clip = create_video_clip_with_effect(temp_image_path, effect_option, duration=5)
            if video_clip:
                video_clips.append(video_clip)
                logging.info(f"Video clip created successfully for {img_url}")
        except Exception as e:
            logging.error(f"Error saving or processing image {img_url}: {e}")
    else:
        logging.warning(f"Failed to download or process image from {img_url}")

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
        full_script = f"{opening_message}\n\n{dynamic_summary}\n\n{closing_message}"
        return full_script
    except Exception as e:
        logging.error(f"Error generating dynamic summary: {e}")
        return f"{opening_message}\n\n[Error generating dynamic summary]\n\n{closing_message}"

# Function to generate audio from a script using OpenAI
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

# Apply optional image effects using Replicate API
def apply_image_effect(image_path, effect):
    try:
        if effect == "None":
            return image_path

        replicate_api_key = st.secrets["replicate"]["api_key"]
        client = replicate.Client(api_token=replicate_api_key)

        # Define models for effects
        models = {
            "Cartoon": "catacolabs/cartoonify",
            "Anime": "cjwbw/videocrafter2-anime",
            "Sketch": "catacolabs/pencil-sketch",
        }

        model_name = models.get(effect)
        if not model_name:
            logging.warning(f"Effect '{effect}' is not supported.")
            return image_path

        # Upload image and apply effect
        with open(image_path, "rb") as img_file:
            response = client.run(model_name, input={"image": img_file})

        if not response:
            logging.error(f"Replicate API returned no result for effect '{effect}'.")
            return image_path

        # Save processed image to a new path
        output_path = tempfile.mktemp(suffix=".png")  # Always use PNG
        with open(output_path, "wb") as f:
            f.write(requests.get(response).content)
        return output_path
    except Exception as e:
        logging.error(f"Error applying effect '{effect}': {e}")
        return image_path

# Function to create a video clip with optional effects
def create_video_clip_with_effect(image_path, effect, duration=5, fps=24):
    try:
        processed_image_path = apply_image_effect(image_path, effect)
        return ImageClip(processed_image_path).set_duration(duration).set_fps(fps)
    except Exception as e:
        logging.error(f"Error creating video clip: {e}")
        return None

# Function to combine video clips with transitions and synchronize with audio
def create_final_video_with_audio_sync(video_clips, script_audio_path, output_path, transition_type="None", fps=24):
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

        # Apply transitions between clips
        if transition_type == "Fade":
            video_clips_with_transitions = []
            for i in range(len(repeated_clips) - 1):
                clip = repeated_clips[i]
                next_clip = repeated_clips[i + 1]
                video_clips_with_transitions.append(clip.crossfadeout(1))
                video_clips_with_transitions.append(next_clip.crossfadein(1))
            repeated_clips = video_clips_with_transitions

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
st.title("Custom Video and Script Generator")

# Input fields for URLs
def url_input_fields():
    urls = []
    with st.container():
        st.subheader("Enter Page URLs")
        num_urls = st.number_input("Number of URLs", min_value=1, value=1, step=1)
        for i in range(num_urls):
            url = st.text_input(f"URL #{i + 1}", placeholder="Enter a webpage URL")
            urls.append(url)
    return urls

# Input fields for images
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
    video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

if st.button("Generate Video"):
    combined_text = ""
    for url in urls:
        text = scrape_text_from_url(url)
        if text:
            combined_text += f"\n{text}"

    if combined_text:
        # Generate script based on user-defined duration
        final_script = generate_dynamic_summary_with_duration(combined_text, video_duration, school_name="these amazing schools")
        audio_path = generate_audio_with_openai(final_script, voice="shimmer")

        # Ensure the audio duration matches the user-defined video duration
        if audio_path:
            audio = AudioFileClip(audio_path)
            audio_duration = audio.duration

            for url, images in url_image_map.items():
                for img_url in images:
                    image = download_image_from_url(img_url)
                    if image:
                        st.image(image, caption=f"Processing {img_url}")
                        temp_image_path = tempfile.mktemp(suffix=".png")  # Always use PNG
                        try:
                        # Convert and save as PNG
                        if image.mode != "RGBA":
                            image = image.convert("RGBA")
                        image.save(temp_image_path, "PNG")
                        video_clip = create_video_clip_with_effect(temp_image_path, effect_option, duration=5)
                        if video_clip:
                            video_clips.append(video_clip)
                        except Exception as e:
                            logging.error(f"Error processing image {img_url}: {e}")

            if video_clips:
                final_video_path = tempfile.mktemp(suffix=".mp4")
                try:
                    final_video_path = create_final_video_with_audio_sync(
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
                st.error("No valid video clips were created.")
