import streamlit as st
import tensorflow as tf
import numpy as np
import os 
from PIL import Image
import base64


# Load Model
model = tf.keras.models.load_model("X-ray.h5")

# Function to Load & Process Image
def load_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.set_page_config(page_title="X-ray Pneumonia Detection", layout="centered")

# Custom CSS for Background & Hover Effects
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    bg_image = f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Apply Background Image (Ensure "x.jpg" is in the same directory)
set_background("x.jpg")

st.title("X-ray Pneumonia Detection")
st.write("Upload an X-ray image to check for Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray Image", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        img_array = load_image(image)
        prediction = model.predict(img_array)
        result = "Pneumonia Detected" if prediction[0][0] > 0.5 else "Normal"
        
        st.subheader("Prediction Result:")
        st.write(result)
