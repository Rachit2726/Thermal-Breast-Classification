"""
Streamlit Web App for Thermal Breast Cancer Classification.
Run: streamlit run app/app.py
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import yaml
import os

st.set_page_config(page_title="Thermal Breast Cancer Classifier", page_icon="🩺", layout="centered")

@st.cache_resource
def load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_models(cfg):
    models = {}
    for key in cfg["models"]:
        path = cfg["models"][key]["save_path"]
        if os.path.exists(path):
            models[cfg["models"][key]["name"]] = tf.keras.models.load_model(path)
    return models

cfg = load_cfg()
models = load_models(cfg)

st.title("🩺 Thermal Breast Cancer Classification")
st.markdown("Upload a **thermal breast image** to classify it as **Benign** or **Malignant** using an ensemble of deep learning models.")
st.divider()

uploaded = st.file_uploader("Upload Thermal Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize(tuple(cfg["image"]["size"]))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    threshold = cfg["training"]["threshold"]

    st.subheader("Model Predictions")
    probs = []
    for name, model in models.items():
        p = model.predict(img_array, verbose=0).ravel()[0]
        probs.append(p)
        label = "🔴 Malignant" if p > threshold else "🟢 Benign"
        st.metric(label=name, value=f"{p:.4f}", delta=label)

    st.divider()
    avg = np.mean(probs)
    final = "🔴 Malignant" if avg > threshold else "🟢 Benign"
    st.subheader("Ensemble Result")
    col1, col2 = st.columns(2)
    col1.metric("Average Probability", f"{avg:.4f}")
    col2.metric("Final Prediction", final)

    st.progress(min(avg, 1.0))
    st.caption(f"Decision threshold: {threshold}")
else:
    st.info("👆 Upload a thermal breast image to get started.")

st.divider()
st.caption("Final Year Project — Thermal Breast Cancer Classification using Ensemble CNNs with Synthetic Data Augmentation")
