import streamlit as st
import requests
import tempfile

# Validate the API key from Streamlit secrets
try:
    REPLICATE_API_TOKEN = st.secrets["replicate"]["api_key"]
    if not REPLICATE_API_TOKEN:
        raise ValueError("Replicate API Key is missing")
except KeyError:
    st.error("Replicate API Key not found in secrets. Please configure it.")
    st.stop()

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
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_file.read())
    temp_input.close()

    if st.button("Apply Effect"):
        model_name = effects[selected_effect]
        st.text(f"Processing with {selected_effect} effect. This may take some time...")

        try:
            # Direct HTTP call to Replicate API
            url = f"https://api.replicate.com/v1/predictions"
            headers = {"Authorization": f"Token {REPLICATE_API_TOKEN}"}
            data = {
                "version": model_name,
            }
            # Send video as a file
            with open(temp_input.name, "rb") as video_file:
                files = {"video": video_file}
                response = requests.post(url, headers=headers, data=data, files=files)

            if response.status_code == 200:
                output_url = response.json()["output"]
                st.video(output_url)
            else:
                st.error(f"Error: {response.status_code}, Details: {response.json()}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
