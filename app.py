import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import requests
import gdown


# Google Drive model file ID
url = "https://drive.google.com/uc?id=1NorO6ZiEh09_pcn-3m91Q6dJ-K2NWSoY"
model_path = "Bestcnn_model.h5"

# Download if not exists
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the trained model
model = tf.keras.models.load_model('Bestcnn_model.h5')

# Define class labels
labels = {
    0: 'Benign',
    1: '[Malignant] Pre-B',
    2: '[Malignant] Pro-B',
    3: '[Malignant] early Pre-B'
}

# Function to predict image
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return labels[predicted_class]

# ---- Background Image Setup ----
bg_image_url = "https://www.shutterstock.com/image-illustration/red-blood-science-3d-illustration-600nw-2381840831.jpg"  # Replace with your URL

page_bg_img = f"""
<style>
.stApp {{
background-image: url("{bg_image_url}");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ---- Streamlit UI ----
st.title("🧬 Image Classifier: Blood Cell Cancer Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        prediction = predict_image(image)
        st.success(f"🩺 Predicted Class: **{prediction}**")


