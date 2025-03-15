import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load the trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Function to preprocess a single image
def preprocess_single_image(pil_img):
    """
    Preprocesses a Pillow image for model inference.

    Args:
        pil_img (PIL.Image.Image): A Pillow image object.

    Returns:
        preprocessed_img (tf.Tensor): Preprocessed image tensor.
    """
    img = pil_img.convert("RGB")  # Convert to RGB
    img = img.resize((224, 224))  # Resize
    img = np.array(img)  # Convert to NumPy array
    img = tf.keras.applications.efficientnet.preprocess_input(img)  # Apply EfficientNet preprocessing
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Load class labels
CLASS_NAMES = json.load(open("class.json", "r"))

st.title("🃏 Card Classification with CNN")
st.write("Upload an image to classify and visualize the top predictions.")

# Upload image
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Preprocess image
    img = preprocess_single_image(image)

    # Predict
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)  # Get highest probability index
    predicted_class = CLASS_NAMES[str(predicted_class_index)]  # Get class label

    # Display predictions
    st.write(f"Predictions Card : { predicted_class }")
