import streamlit as st
import replicate
import tempfile
import requests

# Validate the API key from Streamlit secrets
try:
    REPLICATE_API_TOKEN = st.secrets["replicate"]["api_key"]
    if not REPLICATE_API_TOKEN:
        raise ValueError("Replicate API Key is missing")
except KeyError:
    st.error("Replicate API Key not found in secrets. Please configure it.")
    st.stop()

# Initialize Replicate client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

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
            # Run the model with replicate.run()
            output_url = replicate.run(
                model_name, 
                input={"video": open(temp_input.name, "rb")}  # Adjust based on model's input API
            )

            # Save the processed video
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            response = requests.get(output_url, stream=True)
            with open(temp_output.name, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)

            # Display the processed video
            st.video(temp_output.name)
            with open(temp_output.name, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Cleanup temporary files
        temp_input.close()
