# Ensure pysqlite3-binary is used instead of sqlite3
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Standard Python and library imports
import os
import streamlit as st
from dotenv import load_dotenv
import openai
import json
import requests
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Set API keys (Streamlit secrets or local .env)
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY") or st.secrets["SERPER_API_KEY"]

# Configure speaker voices
speaker_voice_map = {
    "Lisa": "alloy",
    "Ali": "onyx"
}

# System prompt for generating a podcast script
system_prompt = """
You are a podcast host for 'CX Overview.' Generate an engaging, relaxed conversation between Ali and Lisa.
The conversation should feel casual, with natural pauses, fillers like 'um,' and occasional 'you know' to sound conversational. 
Avoid mentioning any tonal instructions directly in the conversation text.

Format the response **strictly** as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text, explanations, or formatting.
"""

# Function to fetch mentions based on the user's query
def fetch_mentions(query):
    try:
        # API URL for Serper
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": os.environ["SERPER_API_KEY"],
            "Content-Type": "application/json"
        }
        payload = {"q": query}

        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad HTTP responses

        # Return the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching mentions: {e}")
        return None

# Function to parse tool output
def parse_tool_output(api_response):
    """Parses the API response to extract mentions."""
    if not api_response or "organic" not in api_response:
        return []
    
    entries = api_response["organic"]
    return [
        {
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "snippet": entry.get("snippet", "")
        }
        for entry in entries
    ]

# Generate podcast script with Ali and Lisa
def generate_script(input_text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return []

# Synthesize speech for both speakers
def synthesize_speech(text, speaker, index):
    audio_dir = "audio-files"
    os.makedirs(audio_dir, exist_ok=True)
    file_path = os.path.join(audio_dir, f"{index:03d}_{speaker}.mp3")
    response = openai.audio.speech.create(
        model="tts-1",
        voice=speaker_voice_map[speaker],
        input=text
    )
    response.stream_to_file(file_path)
    return AudioSegment.from_file(file_path)

# Combine audio segments into a podcast
def combine_audio(audio_segments):
    combined_audio = sum(audio_segments, AudioSegment.empty())
    podcast_file = "podcast.mp3"
    combined_audio.export(podcast_file, format="mp3")
    return podcast_file

# Streamlit app interface
st.title("CX Overview Podcast Generator")
st.write("Enter a topic to generate a podcast conversation between Ali and Lisa.")

query = st.text_area("Enter the topic or discussion point for the podcast:")

if st.button("Generate Podcast"):
    if query.strip():
        st.write("Generating podcast script...")
        conversation_script = generate_script(query.strip())
        if conversation_script:
            st.write("Generating podcast audio...")
            audio_segments = [
                synthesize_speech(part["text"], part["speaker"], idx)
                for idx, part in enumerate(conversation_script)
            ]
            podcast_file = combine_audio(audio_segments)
            st.success("Podcast generated successfully!")
            st.audio(podcast_file)
            st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")
        else:
            st.error("Failed to generate the podcast script.")
    else:
        st.error("Please enter a topic to proceed.")
