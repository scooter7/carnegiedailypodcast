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
You are a podcast host for 'The Carnegie Daily' Generate an engaging, relaxed conversation between Ali and Lisa.
The conversation should feel casual, with natural pauses, fillers like 'um,' and occasional 'you know' to sound conversational. 
Avoid mentioning any tonal instructions directly in the conversation text.

Format the response **strictly** as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text, explanations, or formatting.
"""

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

# Save script to text file
def save_script_to_file(conversation_script, filename="podcast_script.txt"):
    with open(filename, "w") as f:
        for part in conversation_script:
            f.write(f"{part['speaker']}: {part['text']}\n\n")
    return filename

# Streamlit app interface
st.title("The Carnegie Daily")
st.write("Enter a topic to generate a podcast conversation between Ali and Lisa.")

query = st.text_area("Enter the topic or discussion point for the podcast:")

if st.button("Generate Podcast"):
    if query.strip():
        st.write("Generating podcast script...")
        conversation_script = generate_script(query.strip())
        if conversation_script:
            # Save script to file
            script_filename = save_script_to_file(conversation_script)

            # Display script
            st.write("Generated Script:")
            for part in conversation_script:
                st.write(f"**{part['speaker']}**: {part['text']}")

            # Generate podcast audio
            st.write("Generating podcast audio...")
            audio_segments = [
                synthesize_speech(part["text"], part["speaker"], idx)
                for idx, part in enumerate(conversation_script)
            ]
            podcast_file = combine_audio(audio_segments)
            st.success("Podcast generated successfully!")

            # Display audio and download buttons
            st.audio(podcast_file)
            st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")
            st.download_button("Download Script", open(script_filename, "rb"), file_name="podcast_script.txt")
        else:
            st.error("Failed to generate the podcast script.")
    else:
        st.error("Please enter a topic to proceed.")
