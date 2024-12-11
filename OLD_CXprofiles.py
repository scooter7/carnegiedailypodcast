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
Include relevant statistics, facts, and insights based on the summaries. 
The conversation should feel conversational and engaging, with natural pauses, fillers like 'um,' and occasional 'you know.'

Format the response **strictly** as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text, explanations, or formatting.
"""

# Function to extract featured profile links
from urllib.parse import urljoin

def fetch_featured_profiles(parent_url):
    try:
        response = requests.get(parent_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.select("a[href*='college/profile/']")  # Adjust the selector as needed
        profile_links = [urljoin(parent_url, link.get("href")) for link in links if "college/profile/" in link.get("href", "")]
        return list(set(profile_links))  # Remove duplicates
    except Exception as e:
        st.warning(f"Error fetching featured profiles: {e}")
        return []

# Function to scrape content from a URL
def scrape_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]  # Limit text size for OpenAI API
    except Exception as e:
        st.warning(f"Error scraping content from {url}: {e}")
        return ""

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

# Generate podcast script
def generate_script(enriched_text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enriched_text}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating script: {e}")
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

# Combine audio into a podcast
def combine_audio(audio_segments):
    combined_audio = sum(audio_segments, AudioSegment.empty())
    podcast_file = "podcast.mp3"
    combined_audio.export(podcast_file, format="mp3")
    return podcast_file

# Save script to text file
def save_script_to_file(conversation_script, filename="podcast_script.txt"):
    with open(filename, "w") as f:
        for part in conversation_script:
            f.write(f"{part['speaker']}: {part['text']}\n\n")
    return filename

# Streamlit app interface
st.title("CX College Profile Cast")
st.write("Enter a CX Profile URL to generate a podcast that includes links to featured profiles when available.")

parent_url = st.text_input("Enter the URL of the page:")

if st.button("Generate Podcast"):
    if parent_url.strip():
        st.write("Fetching featured profile links...")
        
        # Include the main URL for scraping
        profile_links = fetch_featured_profiles(parent_url.strip())
        profile_links.insert(0, parent_url.strip())  # Add the main URL to the list
        
        if profile_links:
            all_summaries = []
            st.write("Scraping and summarizing content...")
            
            # Iterate over all links (including the main URL)
            for link in profile_links:
                scraped_content = scrape_content(link)
                if scraped_content:
                    summary = summarize_content(scraped_content)
                    if summary:
                        all_summaries.append(summary)
            
            # Generate podcast from all summaries
            enriched_text = " ".join(all_summaries)
            st.write("Generating podcast script...")
            conversation_script = generate_script(enriched_text)
            
            if conversation_script:
                st.write("Generating podcast audio...")
                audio_segments = []
                for idx, part in enumerate(conversation_script):
                    audio = synthesize_cloned_voice(part["text"], part["speaker"])
                    if audio:
                        audio_segments.append(audio)
                    else:
                        st.warning(f"Failed to synthesize audio for part {idx + 1}: {part['speaker']} says: {part['text']}")
                
                if audio_segments:
                    podcast_file = combine_audio(audio_segments)
                    st.success("Podcast generated successfully!")
                    st.audio(podcast_file)
                    st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")
                    script_file = save_script_to_file(conversation_script)
                    st.download_button("Download Script", open(script_file, "rb"), file_name="podcast_script.txt")
                else:
                    st.error("Failed to generate any audio for the podcast.")
            else:
                st.error("Failed to generate the podcast script.")
        else:
            st.error("No featured profiles found.")
    else:
        st.error("Please enter a valid URL.")
