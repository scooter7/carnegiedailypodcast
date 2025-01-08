import streamlit as st
import requests
import openai
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import logging
import os
from PyPDF2 import PdfReader
import docx
import textwrap

logging.basicConfig(level=logging.INFO)

WORDS_PER_MINUTE = 150
FONT_URL = "https://github.com/scooter7/vidshorts/blob/main/Arial.ttf"
FONT_PATH = "Arial.ttf"

def download_font(font_url, local_path):
    if not os.path.exists(local_path):
        response = requests.get(font_url)
        response.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(response.content)

download_font(FONT_URL, FONT_PATH)

def generate_placeholder_image(keyword, style, output_path):
    img = Image.new("RGB", (512, 512), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, size=30)
    text = f"{keyword}\n({style})"
    wrapped_text = textwrap.fill(text, width=20)

    # Calculate text dimensions using textbbox
    text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2

    draw.text((x, y), wrapped_text, font=font, fill="black")
    img.save(output_path)

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

def summarize_text(text, detail_level="Concise"):
    summary_lengths = {"Concise": 100, "Medium": 250, "Comprehensive": 500}
    max_words = summary_lengths.get(detail_level, 100)
    system_prompt = f"Summarize the following text in up to {max_words} words. Focus on key points and maintain clarity."
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return ""

def extract_keywords(text):
    prompt = "Extract a list of concise, individual keywords (comma-separated) from the following text:"
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        keywords = response.choices[0].message.content.strip()
        return [kw.strip() for kw in keywords.split(",") if kw.strip()]
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return []

def generate_illustrations_with_placeholders(keywords, style="pencil sketch"):
    illustration_paths = []
    for keyword in keywords:
        try:
            output_path = tempfile.mktemp(suffix=".jpg")
            generate_placeholder_image(keyword, style, output_path)
            if os.path.exists(output_path):
                illustration_paths.append(output_path)
        except Exception as e:
            logging.error(f"Error generating illustration for keyword '{keyword}': {e}")
    return illustration_paths

def generate_audio(script, voice="shimmer"):
    try:
        response = openai.Audio.create(model="tts-1", voice=voice, input=script)
        audio_path = tempfile.mktemp(suffix=".mp3")
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return audio_path
    except Exception as e:
        logging.error(f"Error generating audio: {e}")
        return None

def create_video(illustrations, audio_path, transition="None", duration_per_image=5):
    try:
        clips = [ImageClip(img).set_duration(duration_per_image) for img in illustrations]
        combined_clip = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(audio_path)
        combined_clip = combined_clip.set_audio(audio)
        video_path = tempfile.mktemp(suffix=".mp4")
        combined_clip.write_videofile(video_path, codec="libx264", audio_codec="aac")
        return video_path
    except Exception as e:
        logging.error(f"Error creating video: {e}")
        return None

st.title("Document-to-Video Generator")
uploaded_file = st.file_uploader("Upload a document (PDF, Word, txt):", type=["pdf", "docx", "txt"])

if uploaded_file:
    text = extract_text_from_document(uploaded_file)
    if text:
        detail_level = st.selectbox("Select Summary Detail Level:", ["Concise", "Medium", "Comprehensive"])
        summary = summarize_text(text, detail_level)
        st.text_area("Generated Summary:", summary, height=150)
        if "keywords" not in st.session_state:
            st.session_state.keywords = extract_keywords(summary)
            st.session_state.selected_keywords = []
        st.subheader("Select Keywords for Illustrations:")
        selected_keywords = []
        for i, keyword in enumerate(st.session_state.keywords):
            is_selected = st.checkbox(keyword, key=f"keyword_{i}")
            if is_selected:
                selected_keywords.append(keyword)
        st.session_state.selected_keywords = selected_keywords
        st.subheader("Add Additional Keywords:")
        additional_keywords = st.text_input("Enter additional keywords separated by commas:")
        if st.button("Process Keywords"):
            if additional_keywords:
                new_keywords = [kw.strip() for kw in additional_keywords.split(",") if kw.strip()]
                st.session_state.selected_keywords.extend(new_keywords)
                st.session_state.selected_keywords = list(set(st.session_state.selected_keywords))
            st.success(f"Selected Keywords: {', '.join(st.session_state.selected_keywords)}")
        if st.session_state.selected_keywords:
            st.subheader("Generated Illustrations:")
            illustrations = generate_illustrations_with_placeholders(
                st.session_state.selected_keywords,
                style="pencil sketch"
        )
        valid_illustrations = [img for img in illustrations if os.path.exists(img)]
        if valid_illustrations:
            st.image(valid_illustrations, caption=st.session_state.selected_keywords, use_container_width=True)
        else:
            st.warning("No illustrations could be generated. Ensure your keywords are valid.")

