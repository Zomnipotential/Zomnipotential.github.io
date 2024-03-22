# Use model.joblib in streamlit to predict the number in that image that is uploaded by the user
# Use the following code in streamlit
import streamlit as st
from PIL import Image
import numpy as np
import joblib

model = joblib.load('model.joblib')

st.title('Number Predictor')
st.write('This app predicts the number in the image uploaded by the user')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    image = image.resize((28, 28))
    image = np.array(image)
    image = 255 - image
    image = image.reshape(1, 28*28)
    prediction = model.predict(image)
    st.write(f"The number in the image is {prediction[0]}, by 95% chance")
    st.write("")
    st.write("The probability of each number is:")
#    st.write(model.predict_proba(image)[0])
    st.write("")
    st.write(model.classes_[np.argsort(model.predict_proba(image)[0])[-2]], "with 90% chance")
    st.write(model.classes_[np.argsort(model.predict_proba(image)[0])[-3]], "with 80% chance")