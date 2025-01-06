import streamlit as st
import replicate
import openai
import tempfile
import requests

# Access API keys from Streamlit secrets
REPLICATE_API_TOKEN = st.secrets["replicate"]["api_key"]
OPENAI_API_TOKEN = st.secrets["openai"]["api_key"]

# Initialize clients
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
openai.api_key = OPENAI_API_TOKEN

# Title of the app
st.title("AI Video Effects with Replicate and OpenAI")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

# Dropdown for effects
effects = {
    "Anime": "stabilityai/stable-diffusion-anime",
    "Pencil Drawing": "replicate/canny-edge-detection",
    "Pixar": "openai/video-style-transfer",
    "Claymation": "replicate/video-claymation",
    "Line Art": "replicate/video-line-art"
}
selected_effect = st.selectbox("Choose an effect to apply:", list(effects.keys()))

# Process video
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_file.read())
    temp_input.close()

    if st.button("Apply Effect"):
        # Handle Pixar effect separately for OpenAI
        if selected_effect == "Pixar":
            st.text(f"Processing with OpenAI for {selected_effect} effect...")
            try:
                # Hypothetical OpenAI video style transfer API
                response = openai.Video.create(
                    file=open(temp_input.name, "rb"),
                    effect="pixar"
                )
                output_url = response["url"]

                # Save the output video
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                output_video = requests.get(output_url).content
                with open(temp_output.name, "wb") as out_file:
                    out_file.write(output_video)

                # Display and download video
                st.video(temp_output.name)
                with open(temp_output.name, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"An error occurred with OpenAI: {e}")

        else:
            st.text(f"Processing with Replicate for {selected_effect} effect...")
            try:
                # Process with Replicate
                model = effects[selected_effect]
                output_url = replicate_client.models.get(model).predict(
                    video=open(temp_input.name, "rb")
                )

                # Save the output video
                temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                response = requests.get(output_url, stream=True)
                with open(temp_output.name, "wb") as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)

                # Display and download video
                st.video(temp_output.name)
                with open(temp_output.name, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

            except Exception as e:
                st.error(f"An error occurred with Replicate: {e}")

        # Cleanup temporary files
        temp_input.close()
