import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ======================
# LOAD TFLITE MODEL
# ======================
interpreter = tflite.Interpreter(model_path="banana_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Overripe", "Ripe", "Rotten", "Unripe"]

st.title("🍌 Klasifikasi Kematangan Pisang (TFLite)")

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
    # PREPROCESS
    # ======================
    img = image.convert("RGB").resize((224,224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # ======================
    # INFERENSI TFLITE
    # ======================
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    hasil = class_names[np.argmax(output)]
    confidence = np.max(output) * 100

    st.success(f"Hasil: {hasil}")
    st.info(f"Confidence: {confidence:.2f}%")
