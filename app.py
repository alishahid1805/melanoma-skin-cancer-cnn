
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Settings
st.set_page_config(
    page_title="Melanoma Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

# Title
st.title("Melanoma Skin Cancer Detection")
st.write("CNN Based Image Classification")

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("melanoma_cnn_model.keras")

model = load_model()

# Upload Image
uploaded_file = st.file_uploader(
    "Upload Skin Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    # Preprocessing
    img = image.resize((64, 64))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    if st.button("Predict"):

        prediction = model.predict(img_array)

        result = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        classes = ["Benign", "Malignant"]

        st.subheader("Prediction Result")

        st.write("Result:", classes[result])
        st.write("Confidence:", round(confidence, 2), "%")

        st.warning(
            "This is an educational AI demonstration, "
            "not a medical diagnosis. Consult a dermatologist."
        )
