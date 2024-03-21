import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Load the trained SVC model
svc_model = joblib.load('svc_model.pkl')

# Define function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize image to match model input size (if necessary)
    resized_image = cv2.resize(gray_image, (28, 28))
    # Flatten image to a 1D array
    flattened_image = resized_image.flatten()
    # Normalize pixel values
    normalized_image = flattened_image / 255.0
    return normalized_image

# Define the Streamlit app
def main():
    st.title('Digit Recognition with SVC Model')
    st.write('Upload an image of a handwritten digit to predict the number')

    # Add file uploader for user to upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    # Make predictions based on uploaded image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert image to numpy array
        img_array = np.array(image)
        # Preprocess the image
        processed_image = preprocess_image(img_array)

        # Make prediction
        prediction = svc_model.predict([processed_image])[0]

        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
