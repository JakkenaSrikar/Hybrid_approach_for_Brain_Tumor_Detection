import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained tumor model
model = tf.keras.models.load_model("models/cnn_model.h5")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 160


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


st.title("Brain Tumor Detection System")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = preprocess_image(image)

    prediction = model.predict(img_array)[0]
    confidence = np.max(prediction)
    predicted_class = class_names[np.argmax(prediction)]

    # Replace "notumor" with clean display text
    if predicted_class == "notumor":
        st.success("No tumor detected.")
    else:
        st.success(f"Tumor Detected: {predicted_class}")
    
    st.write(f"Confidence: {confidence:.2f}")
