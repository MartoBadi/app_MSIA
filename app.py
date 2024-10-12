import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('model.tflite')

# Define class names
class_names = ['burj_khalifa', 'chichen_itza', 'christ the reedemer', 'eiffel_tower', 'great_wall_of_china', 'machu_pichu', 'pyramids_of_giza', 'roman_colosseum', 'statue_of_liberty', 'stonehenge', 'taj_mahal', 'venezuela_angel_falls']

# Preprocess the uploaded image
def preprocess_image(image):
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Image Classification with CNN - Made with love by Hardik")
st.write("Simply upload a picture, and our advanced AI model will tell you if it's a building, forest, glacier, mountain, sea, or street.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = preprocess_image(image)

    # Predict the class
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")

