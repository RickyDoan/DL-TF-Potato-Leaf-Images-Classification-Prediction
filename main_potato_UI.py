# prompt: write the code on streamlit to predict the potato leaf with my model has save below by joblib.

import streamlit as st
import joblib
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = joblib.load('model.joblib')

# Define class names (replace with your actual class names)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']


# Set up the Streamlit app
st.title("Potato Leaf Disease Prediction")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((256, 256)) # Resize the image to match the model's input size
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]

    # Display the results
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
