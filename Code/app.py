import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Loading the trained models
model = tf.keras.models.load_model('Models/multiclass_model.h5')
binary_model = tf.keras.models.load_model('Models/binary_model.h5')

class_names = ['ewaste', 'food_waste', 'leaf_waste', 'metal_cans', 'paper_waste', 'plastic_bags', 'plastic_bottles',
               'wood_waste']
class_names_binary = ['biodegradable', 'non_biodegradable']  # Update based on your binary model classes


st.title("Compostable Material Classifier")


uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Converts the file to an image
    img = Image.open(uploaded_file)

    # Displays the image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocesess the image for the model
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    batch_prediction = model.predict(img_array)
    predicted_label = np.argmax(batch_prediction[0])
    st.write(f"Predicted class: {class_names[predicted_label]}")

    batch_prediction_binary = binary_model.predict(img_array)

    # For binary classification (using a threshold of 0.5)
    predicted_binary_label = 0 if batch_prediction_binary[0] < 0.5 else 1
    st.write(f"Binary classification: {class_names_binary[predicted_binary_label]}")


