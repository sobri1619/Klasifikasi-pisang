import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ======================
# LOAD MODEL
# ======================
model = tf.keras.models.load_model("mobilenet_banana.h5")

class_names = ["Overripe", "Ripe", "Rotten", "Unripe"]

st.title("🍌 Klasifikasi Kematangan Pisang")

option = st.radio("Pilih input", ["Upload", "Kamera"])

uploaded_file = None
camera_file = None

if option == "Upload":
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

if option == "Kamera":
    camera_file = st.camera_input("Ambil gambar")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)

elif camera_file is not None:
    image = Image.open(camera_file)

if image is not None:
    st.image(image, use_container_width=True)

    # ======================
    # PREPROCESSING
    # ======================
    img = image.convert("RGB").resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ======================
    # PREDIKSI
    # ======================
    prediction = model.predict(img_array)
    hasil = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # ======================
    # OUTPUT
    # ======================
    st.success(f"Hasil: {hasil}")
    st.info(f"Confidence: {confidence:.2f}%")
