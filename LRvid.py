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
from PyPDF2 import PdfReader
import docx

logging.basicConfig(level=logging.INFO)

# Approximate words-per-minute rate for narration
WORDS_PER_MINUTE = 150

# Function to extract text from uploaded documents
def extract_text_from_document(file):
    try:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            return " ".join([page.extract_text() for page in pdf_reader.pages])
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            return " ".join([paragraph.text for paragraph in doc.paragraphs])
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        else:
            st.error("Unsupported file type.")
            return ""
    except Exception as e:
        logging.error(f"Error extracting text from document: {e}")
        return ""

# Function to summarize text with OpenAI
def summarize_text(text, detail_level="Concise"):
    summary_lengths = {"Concise": 100, "Medium": 250, "Comprehensive": 500}
    max_words = summary_lengths.get(detail_level, 100)
    system_prompt = f"Summarize the following text in up to {max_words} words. Focus on key points and maintain clarity."
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return ""

# Function to extract keywords using OpenAI
def extract_keywords(text):
    prompt = "Extract a list of keywords from the following text:"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        keywords = response.choices[0].message.content.strip()
        return keywords.split(", ")
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return []

# Function to generate illustrations based on keywords
def generate_illustrations(keywords):
    illustration_paths = []
    for keyword in keywords:
        try:
            response = replicate.Client(api_token=st.secrets["replicate"]["api_key"]).run(
                "catacolabs/pencil-sketch", input={"prompt": keyword}
            )
            image_path = tempfile.mktemp(suffix=".jpg")
            with open(image_path, "wb") as f:
                f.write(requests.get(response).content)
            illustration_paths.append(image_path)
        except Exception as e:
            logging.error(f"Error generating illustration for keyword '{keyword}': {e}")
    return illustration_paths

# Function to generate audio from text using OpenAI
def generate_audio(script, voice="shimmer"):
    try:
        response = openai.audio.speech.create(model="tts-1", voice=voice, input=script)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

# Function to create a video from illustrations and audio
def create_video(illustrations, audio_path, transition="None", duration_per_image=5):
    try:
        clips = [
            ImageClip(img).set_duration(duration_per_image)
            for img in illustrations
        ]
        
        if transition == "Fade":
            clips = [clip.crossfadein(1) for clip in clips]
        elif transition == "Swipe":
            clips = [clip.slidein(1) for clip in clips]

        combined_clip = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(audio_path)
        combined_clip = combined_clip.set_audio(audio)

        video_path = tempfile.mktemp(suffix=".mp4")
        combined_clip.write_videofile(video_path, codec="libx264", audio_codec="aac")
        return video_path
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        return None

# Streamlit UI
st.title("Document-to-Video Generator")

uploaded_file = st.file_uploader("Upload a document (PDF, Word, txt):", type=["pdf", "docx", "txt"])

if uploaded_file:
    text = extract_text_from_document(uploaded_file)
    if text:
        detail_level = st.selectbox("Select Summary Detail Level:", ["Concise", "Medium", "Comprehensive"])
        summary = summarize_text(text, detail_level)
        st.text_area("Generated Summary:", summary, height=150)

        keywords = extract_keywords(summary)
        selected_keywords = st.multiselect("Select Keywords for Illustrations:", keywords)

        if selected_keywords:
            illustrations = generate_illustrations(selected_keywords)

            if illustrations:
                st.image(illustrations, caption=selected_keywords, use_column_width=True)

                transition = st.selectbox("Select Video Transition:", ["None", "Fade", "Swipe"])
                duration_per_image = st.number_input(
                    "Duration per Image (seconds):", min_value=1, max_value=10, value=5
                )

                script = f"Here are the key highlights: {', '.join(selected_keywords)}. {summary}"
                audio_path = generate_audio(script)

                if audio_path:
                    video_path = create_video(illustrations, audio_path, transition, duration_per_image)
                    if video_path:
                        st.video(video_path)
                        st.download_button("Download Video", open(video_path, "rb"), "video.mp4")
