import base64
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the best model
model = load_model('models/vgg16_model.h5')

# Define the class indices
class_indices = {'adenocarcinoma': 0, 'large.cell.carcinoma': 1, 'normal': 2,'squamous.cell.carcinoma': 3}

# Create a Streamlit app
st.title("Lung Cancer Classification")
st.write("Upload a CT scan image to predict the likelihood of lung cancer")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
   .stApp {{
        background-image: url(data:images/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('images/bg.jpeg')


# Create a file uploader
uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

# Create a button to classify the image
if st.button("Classify"):
    if uploaded_file is not None:
        # Load the uploaded image
        img = load_img(uploaded_file, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions[0])

        # Get the corresponding class label
        predicted_class_label = list(class_indices.keys())[list(class_indices.values()).index(predicted_class_index)]

        # Display the result
        st.write("Predicted class:", predicted_class_label)
        st.write("Confidence:", predictions[0][predicted_class_index])
    else:
        st.write("Please upload an image")