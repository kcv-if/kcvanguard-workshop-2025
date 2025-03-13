import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json

# Load the trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# Load class labels
CLASS_NAMES = json.load(open("class.json", "r"))

st.title("🃏 Card Classification with CNN")
st.write("Upload an image to classify and visualize the top predictions.")

# Upload image
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.convert("RGB").resize((224, 224))  # Resize and convert
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)[0]  # Get 1D array
    top_3_indices = np.argsort(predictions)[-3:][::-1]  # Get top 3 indices
    top_3_classes = [CLASS_NAMES[str(i)] for i in top_3_indices]
    top_3_confidences = [predictions[i] for i in top_3_indices]

    # Display predictions
    st.write("### 🏆 Top 3 Predictions:")
    for i in range(3):
        st.write(f"**{top_3_classes[i]}** - Confidence: {top_3_confidences[i]:.2f}")

    # Create a DataFrame for Streamlit bar chart
    chart_data = pd.DataFrame({"Class": top_3_classes, "Confidence": top_3_confidences})
    
    # Visualize using built-in Streamlit bar chart
    st.bar_chart(chart_data.set_index("Class"))
