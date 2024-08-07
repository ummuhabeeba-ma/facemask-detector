import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

model = YOLO("best.pt")

st.set_page_config(page_title="Mask Detector", page_icon="😷")
st.title("🦠 Real-Time Face Mask Detection")
st.write("Welcome to the Mask Detection Application. Choose an option to detect masks:")

option = st.selectbox("Select an option: ", ("Upload Image", "Capture from Webcam"))

if option == "Upload Image":
    image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
else:
    image = st.camera_input("Capture image")

if image is not None:
    st.write("## Image")
    
    if option == "Upload Image":
        img = Image.open(image)
    else:
        img = Image.open(image)

    st.image(img, caption='Uploaded/Captured Image', use_column_width=True)

    img_array = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = model(img_bgr)

    if results:
        result = results[0]
        res_plot = result.plot(show=False)
        res_plot = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)

        res_plot = Image.fromarray(res_plot)
        st.image(res_plot, caption="Processed Image", use_column_width=True)
    else:
        st.write("No results from the model. Please check the model and input image.")
else:
    st.info("Please upload an image or capture a picture.")
