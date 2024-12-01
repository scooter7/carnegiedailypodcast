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
from elevenlabs import play
from elevenlabs.client import ElevenLabs
import tempfile

# Load environment variables
load_dotenv()

# Set API keys (Streamlit secrets or local .env)
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
serper_api_key = os.getenv("SERPER_API_KEY") or st.secrets["SERPER_API_KEY"]

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(
    api_key=os.getenv("ELEVENLABS_API_KEY") or st.secrets["ELEVENLABS_API_KEY"]
)

# Configure speaker voices
speaker_voice_map = {
    "Lisa": "Rachel",  # Replace with appropriate voice name
    "Ali": "NYy9s57OPECPcDJavL3T"  # Replace with the ID of your cloned voice
}

# Updated system prompt for a news-oriented conversation
system_prompt = """
You are a podcast host for 'CX Overview.' Generate a robust, fact-based, news-oriented conversation between Ali and Lisa. 
Include relevant statistics, facts, and references to current events when available. 
The conversation should still feel conversational and engaging, with natural pauses, fillers like 'um,' and occasional 'you know.'

Format the response **strictly** as a JSON array of objects, each with 'speaker' and 'text' keys. 
Only return JSON without additional text, explanations, or formatting.
"""

# Function to fetch news articles using Serper
def fetch_news_mentions(query):
    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query, "type": "news"}  # Specify the "news" type

        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching news mentions: {e}")
        return None

# Function to parse tool output
def parse_tool_output(api_response):
    if not api_response or "news" not in api_response:
        return []
    entries = api_response["news"]
    return [
        {
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "snippet": entry.get("snippet", "")
        }
        for entry in entries
    ]

# Summarize and enrich the data with OpenAI
def enrich_data_with_facts(parsed_mentions):
    snippets = [mention["snippet"] for mention in parsed_mentions]
    detailed_text = " ".join(snippets)
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following snippets into meaningful insights, including any relevant facts, statistics, and references."},
            {"role": "user", "content": detailed_text}
        ]
    )
    return response.choices[0].message.content.strip()

# Generate podcast script with Ali and Lisa
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

# Synthesize speech using ElevenLabs client
def synthesize_cloned_voice(text, speaker):
    """
    Synthesizes speech using ElevenLabs Multilingual v2 model.
    Args:
        text: The text to synthesize.
        speaker: The speaker's voice ID or name.
    Returns:
        AudioSegment: The generated audio file as an AudioSegment object.
    """
    try:
        # Generate audio using ElevenLabs
        audio = elevenlabs_client.generate(
            text=text,
            voice=speaker_voice_map[speaker],
            model="eleven_multilingual_v2"
        )
        
        # Write the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            temp_audio_file.write(audio)
            temp_audio_path = temp_audio_file.name
        
        # Load the audio as an AudioSegment object
        return AudioSegment.from_file(temp_audio_path, format="mp3")
    except Exception as e:
        st.error(f"Error synthesizing speech with ElevenLabs for {speaker}: {e}")
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
st.title("CX Overview Podcast Generator")
st.write("Generate a fact-based, news-oriented podcast conversation between Ali and Lisa.")

query = st.text_area("Enter the topic or discussion point for the podcast:")

if st.button("Generate Podcast"):
    if query.strip():
        st.write("Fetching news articles...")
        raw_mentions = fetch_news_mentions(query.strip())
        parsed_mentions = parse_tool_output(raw_mentions)
        if parsed_mentions:
            st.write("Enriching content with facts and insights...")
            enriched_text = enrich_data_with_facts(parsed_mentions)
            st.write("Generating podcast script...")
            conversation_script = generate_script(enriched_text)
            if conversation_script:
                st.write("Generating podcast audio...")
                audio_segments = [
                    synthesize_cloned_voice(part["text"], part["speaker"])
                    for part in conversation_script
                ]
                podcast_file = combine_audio(audio_segments)
                st.success("Podcast generated successfully!")

                # Display audio and download buttons
                st.audio(podcast_file)
                st.download_button("Download Podcast", open(podcast_file, "rb"), file_name="podcast.mp3")
                script_file = save_script_to_file(conversation_script)
                st.download_button("Download Script", open(script_file, "rb"), file_name="podcast_script.txt")
            else:
                st.error("Failed to generate the podcast script.")
        else:
            st.error("No relevant news articles found for the given query.")
    else:
        st.error("Please enter a topic to proceed.")
