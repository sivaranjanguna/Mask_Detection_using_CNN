import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# --------------------------------------------------
# Safe Flipkart-Style CSS (NO BUGS)
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #f1f3f6;
}
.main {
    background-color: #f1f3f6;
}

/* Header */
.header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    margin-top: 10px;
}

.logo {
    width: 46px;
    height: 46px;
    background-color: #2874F0;
    border-radius: 50%;
}

.title {
    font-size: 38px;
    font-weight: 700;
    color: #2874F0;
}

.subtitle {
    text-align: center;
    color: #6b7280;
    margin-top: 8px;
    margin-bottom: 24px;
    font-size: 16px;
}

/* Card */
.card {
    background-color: #ffffff;
    padding: 26px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}

/* Results */
.result-mask {
    background-color: #e6f4ea;
    color: #137333;
    padding: 14px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 18px;
    text-align: center;
}

.result-nomask {
    background-color: #fdecea;
    color: #b91c1c;
    padding: 14px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 18px;
    text-align: center;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 26px;
    color: #6b7280;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Download & Load Model (CACHED & SAFE)
# --------------------------------------------------
MODEL_URL = (
    "https://huggingface.co/spaces/SIVARANJAN03/"
    "Face_Mask_Detection_MLfile/resolve/main/"
    "Face_Mask_detection_Model.keras"
)

MODEL_PATH = "Face_Mask_detection_Model.keras"


@st.cache_resource
def load_face_mask_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    return tf.keras.models.load_model(MODEL_PATH)


model = load_face_mask_model()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("""
<div class="header">
    <div class="logo"></div>
    <div class="title">Face Mask Detection</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Upload an image to check whether a person is wearing a face mask</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Card Section
# --------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image (JPG only)",
    type=["jpg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((128, 128))

    st.image(image_resized, use_container_width=True)

    img = np.array(image_resized) / 255.0
    img = img.reshape(1, 128, 128, 3)

    prediction = model.predict(img)
    confidence = np.max(prediction) * 100
    label = np.argmax(prediction)

    st.markdown("<br>", unsafe_allow_html=True)

    # Class mapping
    # 0 -> No Mask | 1 -> Mask
    if label == 1:
        st.markdown(
            f"<div class='result-mask'>✔ Wearing Face Mask</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-nomask'>✖ Not Wearing Face Mask</div>",
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    "<div class='footer'>AI-powered Face Mask Detection | Streamlit & TensorFlow</div>",
    unsafe_allow_html=True
)
