import streamlit as st
import os
import requests
from pathlib import Path

# Validate the API key from Streamlit secrets
try:
    REPLICATE_API_TOKEN = st.secrets["replicate"]["api_key"]
    if not REPLICATE_API_TOKEN:
        raise ValueError("Replicate API Key is missing")
except KeyError:
    st.error("Replicate API Key not found in secrets. Please configure it.")
    st.stop()

# Ensure a temporary directory exists for serving files
TEMP_DIR = Path("./tmp")
TEMP_DIR.mkdir(exist_ok=True)

# File uploader for video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Dropdown for effects
effects = {
    "Claymation": "cjwbw/videocrafter2",
    "Dreamix (Stylized)": "google-research/dreamix",
    "Smooth Motion (Frame Interpolation)": "google-research/frame-interpolation",
    "Cartoonify": "catacolabs/cartoonify",
}
selected_effect = st.selectbox("Choose an effect to apply:", list(effects.keys()))

# Process video
if uploaded_file is not None:
    # Save the uploaded file to the temporary directory
    file_path = TEMP_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Create a temporary URL to serve the file
    public_url = f"http://{st.server.server_address}:{st.server.server_port}/{file_path}"

    if st.button("Apply Effect"):
        model_name = effects[selected_effect]
        st.text(f"Processing with {selected_effect} effect. This may take some time...")

        try:
            # Call the Replicate API with the public URL
            url = "https://api.replicate.com/v1/predictions"
            headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
            payload = {
                "version": model_name,
                "input": {
                    "video": public_url
                }
            }
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code == 200:
                output_url = response.json()["output"]
                st.video(output_url)
            else:
                st.error(f"Error: {response.status_code}, Details: {response.json()}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Clean up the temporary file after use
    os.remove(file_path)
