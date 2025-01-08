import streamlit as st
from bs4 import BeautifulSoup
import requests
import openai
import tempfile
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from PIL import Image
import logging
import os
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

# Function to extract keywords using OpenAI
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

# Function to generate illustrations using DALL-E 3
import json

# Function to generate illustrations using DALL-E 3
def generate_illustrations_with_dalle(keywords, style="pencil sketch"):
    """
    Generates illustrations for a list of keywords using DALL-E 3 via the chat completions API.
    Returns a list of file paths to the generated images.
    """
    illustration_paths = []

    for keyword in keywords:
        try:
            # Construct the descriptive prompt
            prompt = f"Create a {style} of {keyword}."

            # Use OpenAI chat completions API with function calling
            response = openai.chat.completions.create(
                model="gpt-4o",  # Ensure the model supports function calling
                messages=[
                    {"role": "system", "content": "You are an image generation assistant."},
                    {"role": "user", "content": prompt},
                ],
                functions=[
                    {
                        "name": "generate_image",
                        "description": "Generate an image based on a given description.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Text description of the image to generate."},
                                "size": {"type": "string", "enum": ["256x256", "512x512", "1024x1024"]},
                            },
                            "required": ["prompt", "size"]
                        },
                    }
                ],
                function_call={"name": "generate_image"}  # Explicitly request the function
            )

            # Parse the function call arguments
            function_call_args = json.loads(response.choices[0].message.function_call.arguments)

            # Generate the image using the parsed arguments
            image_response = openai.Image.create(
                prompt=function_call_args["prompt"],
                size=function_call_args["size"],
                n=1
            )

            # Retrieve the image URL
            image_url = image_response["data"][0]["url"]

            # Download the image and save it locally
            image_path = tempfile.mktemp(suffix=".jpg")
            image_data = requests.get(image_url).content
            with open(image_path, "wb") as f:
                f.write(image_data)

            # Append the local image path to the list
            illustration_paths.append(image_path)

        except json.JSONDecodeError as json_error:
            logging.error(f"JSON parsing error for keyword '{keyword}': {json_error}")
            st.warning(f"Failed to generate image for '{keyword}'. Invalid function call arguments.")
        except Exception as e:
            logging.error(f"Error generating illustration for keyword '{keyword}': {e}")
            st.warning(f"Failed to generate image for '{keyword}'. Skipping.")

    return illustration_paths

# Function to generate audio from text using OpenAI
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

# Function to create a video from illustrations and audio
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

# Streamlit UI
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
            illustrations = generate_illustrations_with_dalle(
                st.session_state.selected_keywords,
                style="pencil sketch"  # You can change the style here
        )
        if illustrations:
            st.image(illustrations, caption=st.session_state.selected_keywords, use_column_width=True)
        else:
            st.warning("No illustrations could be generated. Try different keywords or styles.")
