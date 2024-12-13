# Standard Python and library imports
import sys
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv
import openai
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
import tempfile
import json
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip
from urllib.parse import urljoin
import time

# Load environment variables
load_dotenv()

# Set API keys (Streamlit secrets or local .env)
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
elevenlabs_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY") or st.secrets["ELEVENLABS_API_KEY"]
)

# Configure speaker voices
speaker_voice_map = {
    "Lisa": "Rachel",
    "Ali": "NYy9s57OPECPcDJavL3T"  # Replace with the ID of your cloned voice
}

# System prompt for the podcast script
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, news-oriented conversation between Ali and Lisa. 
Include relevant statistics, facts, and insights based on the summaries. The conversation should feel conversational and engaging.
"""

# Function to scrape images and text from a URL
def scrape_images_and_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract images
        images = [urljoin(url, img["src"]) for img in soup.find_all("img", src=True)]
        
        # Extract text
        text = soup.get_text(separator=" ", strip=True)
        return images, text[:5000]
    except Exception as e:
        st.warning(f"Error scraping content from {url}: {e}")
        return [], ""

# Summarize content using OpenAI
def summarize_content(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following content into meaningful insights."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Error summarizing content: {e}")
        return ""

# Generate podcast script with retry logic
def generate_script(enriched_text, duration_seconds, retries=3):
    for attempt in range(retries):
        try:
            response = openai.chat.ompletions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"{system_prompt} The script should fit within {duration_seconds} seconds."},
                    {"role": "user", "content": enriched_text}
                ]
            )
            raw_content = response.choices[0].message.content.strip()
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                st.error("The API response is not valid JSON. Response content: " + raw_content)
                return []
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait before retrying
    st.error("All retries failed.")
    return []

# Synthesize speech using ElevenLabs
def synthesize_cloned_voice(text, speaker):
    try:
        audio_generator = elevenlabs_client.generate(
            text=text,
            voice=speaker_voice_map[speaker],
            model="eleven_multilingual_v2"
        )
        audio_content = b"".join(audio_generator)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio_content)
            temp_audio_path = temp_audio_file.name
        return AudioSegment.from_file(temp_audio_path, format="mp3")
    except Exception as e:
        st.error(f"Error synthesizing speech for {speaker}: {e}")
        return None

# Create video using images and captions
def create_video(images, script, duration_seconds):
    clips = []
    segment_duration = duration_seconds / len(images)
    
    for i, image in enumerate(images):
        img_clip = ImageClip(image).set_duration(segment_duration)
        if i < len(script):
            text_clip = TextClip(script[i]["text"], fontsize=24, color='white', bg_color='black', size=img_clip.size)
            text_clip = text_clip.set_duration(segment_duration).set_position('bottom')
            composite_clip = CompositeVideoClip([img_clip, text_clip])
        else:
            composite_clip = img_clip
        clips.append(composite_clip)
    
    final_video = concatenate_videoclips(clips)
    video_file = "video_short.mp4"
    final_video.write_videofile(video_file, codec="libx264", fps=24)
    return video_file

# Streamlit app interface
st.title("CX College Profile Video & Podcast Creator")
st.write("Enter a CX Profile URL to generate a podcast and video short.")

parent_url = st.text_input("Enter the URL of the page:")
duration = st.radio("Select Duration", [15, 30, 45, 60], index=0)

if st.button("Generate Content"):
    if parent_url.strip():
        st.write("Scraping content from the URL...")
        
        images, scraped_text = scrape_images_and_text(parent_url.strip())
        if scraped_text:
            st.write("Summarizing content...")
            summary = summarize_content(scraped_text)
            
            if summary:
                st.write("Generating podcast script...")
                conversation_script = generate_script(summary, duration)
                
                if conversation_script:
                    st.write("Generating podcast audio...")
                    audio_segments = []
                    for part in conversation_script:
                        audio = synthesize_cloned_voice(part["text"], part["speaker"])
                        if audio:
                            audio_segments.append(audio)
                    
                    if audio_segments:
                        combined_audio = sum(audio_segments, AudioSegment.empty())
                        podcast_file = "podcast.mp3"
                        combined_audio.export(podcast_file, format="mp3")
                        st.success("Podcast generated successfully!")
                        st.audio(podcast_file)
                        st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")
                    
                    if images:
                        st.write("Creating video short...")
                        video_file = create_video(images, conversation_script, duration)
                        st.success("Video short created successfully!")
                        st.video(video_file)
                        st.download_button("Download Video", open(video_file, "rb"), file_name="video_short.mp4")
                    else:
                        st.error("No images found to create video short.")
                else:
                    st.error("Failed to generate the podcast script.")
            else:
                st.error("Failed to summarize the content.")
        else:
            st.error("Failed to scrape content from the URL.")
    else:
        st.error("Please enter a valid URL.")
