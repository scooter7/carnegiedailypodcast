import streamlit as st
from bs4 import BeautifulSoup
import requests
import replicate
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from PIL import Image
from io import BytesIO
import os
import logging

logging.basicConfig(level=logging.INFO)

# Approximate words-per-minute rate for narration
WORDS_PER_MINUTE = 150

# Function to scrape text content from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]  # Limit to 5000 characters
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

# Function to apply effects using Replicate API
def apply_replicate_effect(image_path, effect):
    try:
        replicate_models = {
            "Cartoon": "catacolabs/cartoonify",
            "Anime": "cjwbw/videocrafter2-anime",
            "Sketch": "catacolabs/pencil-sketch",
        }
        model_name = replicate_models.get(effect)
        if not model_name:
            logging.warning(f"No Replicate model found for effect: {effect}")
            return image_path  # Return original image if no effect is selected

        replicate_api_key = st.secrets["replicate"]["api_key"]
        client = replicate.Client(api_token=replicate_api_key)

        logging.info(f"Applying effect '{effect}' using model '{model_name}'")
        with open(image_path, "rb") as img_file:
            response = client.run(
                model_name,
                input={"image": img_file}
            )

        # Ensure the response is valid
        if not response or not isinstance(response, str):
            logging.error(f"Invalid response from Replicate API: {response}")
            return image_path  # Return original image if API fails

        # Download the processed image and save it
        output_path = tempfile.mktemp(suffix=".jpg")
        response_content = requests.get(response).content
        with open(output_path, "wb") as out_file:
            out_file.write(response_content)

        logging.info(f"Effect applied successfully. Processed image saved to: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error applying effect '{effect}': {e}")
        return image_path  # Return the original image in case of error

# Function to create video clip from processed image
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
            raise ValueError("No video clips provided for final video creation.")

        # Load audio to determine total duration
        audio = AudioFileClip(audio_path)
        total_audio_duration = audio.duration

        # Repeat and trim clips to match audio duration
        repeated_clips = []
        current_duration = 0
        while current_duration < total_audio_duration:
            for clip in video_clips:
                if current_duration + clip.duration > total_audio_duration:
                    clip = clip.subclip(0, total_audio_duration - current_duration)
                    repeated_clips.append(clip)
                    current_duration = total_audio_duration
                    break
                repeated_clips.append(clip)
                current_duration += clip.duration

        # Concatenate video clips and synchronize with audio
        final_clip = concatenate_videoclips(repeated_clips, method="compose")
        final_clip = final_clip.set_audio(audio)
        final_clip = final_clip.subclip(0, total_audio_duration)

        # Save the final video
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=fps)
        logging.info(f"Final video successfully created at {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Error during final video creation: {e}")
        return None

# Streamlit UI
st.title("Custom Video and Script Generator with AI Effects")

# Input URLs
def url_input_fields():
    urls = []
    st.subheader("Enter URLs")
    num_urls = st.number_input("Number of URLs", min_value=1, value=1, step=1)
    for i in range(num_urls):
        url = st.text_input(f"URL #{i + 1}", placeholder="Enter a webpage URL")
        urls.append(url)
    return urls

# Input images for each URL
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

if urls:
    url_image_map = image_input_fields(urls)
    effect_option = st.selectbox("Select an Effect:", ["None", "Cartoon", "Anime", "Sketch"])
    video_duration = st.number_input("Desired Video Duration (in seconds):", min_value=10, step=5, value=60)

if st.button("Generate Video"):
    video_clips = []
    combined_text = ""

    # Process each URL and its associated images
    for url, images in url_image_map.items():
        text = scrape_text_from_url(url)
        combined_text += f"\n{text}" if text else ""
        for img_url in images:
            image = download_image_from_url(img_url)
            if image:
                temp_image_path = tempfile.mktemp(suffix=".jpg")
                image.save(temp_image_path)

                # Apply effect and create video clip
                video_clip = create_video_clip_with_effect(temp_image_path, effect_option)
                if video_clip:
                    video_clips.append(video_clip)

    # Generate audio (simulating audio for now)
    audio_path = tempfile.mktemp(suffix=".mp3")
    with open(audio_path, "wb") as f:
        f.write(b"Simulated audio content")

    # Create final video
    if video_clips and os.path.exists(audio_path):
        final_video_path = tempfile.mktemp(suffix=".mp4")
        final_video_path = create_final_video_with_audio(video_clips, audio_path, final_video_path)

        # Verify final video and display
        if os.path.exists(final_video_path):
            st.video(final_video_path)
        else:
            st.error("Failed to create the final video. Please check the processing pipeline.")
