import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from feature_extractor import extract_combined_features, extract_keyframes

# ⚙️ Must be first Streamlit command
st.set_page_config(page_title="Image & Video Captioning", layout="centered")

# Constants
MAX_LENGTH = 30

# Load models & tokenizers
@st.cache_resource
def load_models_and_tokenizers():
    image_model = tf.keras.models.load_model("model.keras")
    with open("tokenizer.pkl", "rb") as f:
        image_tokenizer = pickle.load(f)

    video_model = tf.keras.models.load_model("model_video.keras")
    with open("tokenizer_video.pkl", "rb") as f:
        video_tokenizer = pickle.load(f)

    return image_model, image_tokenizer, video_model, video_tokenizer

image_model, image_tokenizer, video_model, video_tokenizer = load_models_and_tokenizers()

# 🧠 Caption generator
def generate_caption(feature, model, tokenizer):
    in_text = 'startseq'
    for _ in range(MAX_LENGTH):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=MAX_LENGTH, padding='post')
        yhat = model.predict([feature.reshape(1, -1), sequence], verbose=0)
        next_word_id = np.argmax(yhat)
        word = tokenizer.index_word.get(next_word_id, None)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    final_caption = in_text.replace('startseq', '').strip()
    return final_caption

# 📷 Feature extractor for video (averaged keyframe features)
def extract_video_feature(video_path):
    keyframes = extract_keyframes(video_path, Path(tempfile.mkdtemp()))
    keyframe_features = []

    for frame_path in keyframes:
        features = extract_combined_features(frame_path)
        if features is not None:
            keyframe_features.append(features)

    if keyframe_features:
        return np.mean(keyframe_features, axis=0)
    else:
        return None

# UI layout
st.title("🧠 Image & Video Captioning Dashboard")
st.write("Upload an image or video to generate a caption using deep learning.")

option = st.radio("Choose input type:", ["Image", "Video"])

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            feature = extract_combined_features(tmp.name)

        if feature is not None:
            caption = generate_caption(feature, image_model, image_tokenizer)
            st.success("📝 Caption:")
            st.write(caption)
        else:
            st.error("Failed to extract features from image.")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(uploaded_video.read())
            st.video(tfile.name)
            feature = extract_video_feature(tfile.name)

        if feature is not None:
            caption = generate_caption(feature, video_model, video_tokenizer)
            st.success("📝 Caption:")
            st.write(caption)
        else:
            st.error("Failed to extract features from video.")
