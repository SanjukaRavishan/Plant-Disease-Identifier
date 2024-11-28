import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.layers import TFSMLayer


def model_prediction(test_image):
    model = TFSMLayer("plant_disease_model_directory",
                      call_endpoint="serving_default")

    image = Image.open(test_image).resize(
        (128, 128))
    input_arr = np.array(image)

    if len(input_arr.shape) == 3:
        input_arr = np.expand_dims(input_arr, axis=0)

    # convert to tensor
    input_tensor = tf.convert_to_tensor(input_arr, dtype=tf.float32)
    predictions_dict = model(input_tensor)
    predictions = list(predictions_dict.values())[0]

    predictions_numpy = predictions.numpy()

    return np.argmax(predictions_numpy)


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page", ["Disease Recognition"]
)

if app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")

    if test_image:
        img = Image.open(test_image)
        st.image(img, width=300, caption="Uploaded Image")

    if st.button("Predict"):
        if test_image:
            st.write("Analyzing the image...")
            result_index = model_prediction(test_image)

            # Class Labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            st.success(f"Model is Predicting: **{class_name[result_index]}**")
        else:
            st.error("Please upload an image before predicting.")
