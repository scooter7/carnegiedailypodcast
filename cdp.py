import os
import json
import openai
import streamlit as st
from pydub import AudioSegment
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
import re

# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

# Initialize the SerperDevTool for news search
search_tool = SerperDevTool()

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
        result = search_tool.search(query)
        return result if result else ""
    except Exception as e:
        st.warning(f"Error fetching mentions: {e}")
        return ""

# Parse tool output to extract structured mentions
def parse_tool_output(tool_output):
    entries = re.findall(r"Title: (.+?)\nLink: (.+?)\nSnippet: (.+?)(?=\n---|\Z)", tool_output, re.DOTALL)
    return [{"title": title.strip(), "link": link.strip(), "snippet": snippet.strip()} for title, link, snippet in entries]

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
