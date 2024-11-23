# Ensure pysqlite3-binary is used instead of sqlite3
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# Standard Python and library imports
import os
import time
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import openai
import json
import re
import requests
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Set API keys (Streamlit secrets or local .env)
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY") or st.secrets["SERPER_API_KEY"]

# Initialize SerperDevTool with API key (if available)
try:
    search_tool = SerperDevTool(api_key=os.environ["SERPER_API_KEY"])
except Exception as e:
    st.error(f"Error initializing SerperDevTool: {e}")

# Configure speaker voices for podcast
speaker_voice_map = {
    "Lisa": "alloy",
    "Ali": "onyx"
}

# System prompt for generating a podcast script
system_prompt = """
You are a podcast host for 'Higher Ed Marketing Insights.' Generate an engaging, conversational script between Ali and Lisa, summarizing key insights on higher education marketing news. 
The conversation should feel casual and informative, with natural pauses and fillers like 'you know' to sound conversational.

Format the response **strictly** as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text or explanations.
"""

# Function to fetch marketing news mentions
def fetch_mentions(query):
    try:
        # API URL for Serper
        url = "https://serper.dev/api/search"
        headers = {
            "Authorization": f"Bearer {os.environ['SERPER_API_KEY']}"
        }
        params = {"q": query}  # Send query in URL parameters for GET request

        # Make the GET request
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an error for bad HTTP responses

        # Return the JSON response
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching mentions: {e}")
        return None

# Function to parse tool output
def parse_tool_output(api_response):
    if not api_response or "organic" not in api_response:
        return []
    
    # Extract relevant fields from the Serper response
    entries = api_response["organic"]
    return [
        {"title": entry.get("title", ""), 
         "link": entry.get("link", ""), 
         "snippet": entry.get("snippet", "")}
        for entry in entries
    ]

# Summarize extracted mentions for script generation
def summarize_mentions(parsed_mentions):
    snippets = [mention["snippet"] for mention in parsed_mentions]
    summarized_text = " ".join(snippets)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following news snippets for a podcast discussion."},
            {"role": "user", "content": summarized_text}
        ]
    )
    return response.choices[0].message.content.strip()

# Generate podcast script
def generate_script(summarized_text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": summarized_text}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Error generating script: {e}")
        return []

# Synthesize speech from text for each speaker
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
st.title("Higher Education Marketing Podcast Generator")
st.write("Fetch the latest marketing news and create a podcast discussing the insights.")

query = st.text_input("Enter your query (e.g., 'higher education marketing news')")

if st.button("Generate Podcast"):
    if query:
        st.write("Fetching and summarizing news...")
        raw_mentions = fetch_mentions(query)
        parsed_mentions = parse_tool_output(raw_mentions)
        if parsed_mentions:
            summarized_text = summarize_mentions(parsed_mentions)
            conversation_script = generate_script(summarized_text)
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
            st.error("No news mentions found for the given query.")
    else:
        st.error("Please enter a query to proceed.")
