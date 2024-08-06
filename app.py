import streamlit as st
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


model = YOLO("best.pt")

st.set_page_config(page_title="Mask Detector", page_icon="😷")
st.title("🦠 Real-Time Face Mask Detection")
st.write("Welcome to the Mask Detection Application. Choose an option to detect masks:")

option = st.selectbox("Select an option:", ("Webcam Detection", "Upload Image"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result = model(image)
        res_plot = result[0].plot(show=False)
        res_plot = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)

        res_plot = Image.fromarray(res_plot)

        st.image(res_plot, caption="Processed Image", use_column_width=True)
        st.success("Detection complete!")
    else:
        st.info("Please upload an image file.")

elif option == "Webcam Detection":
    st.write("Click the 'Start Webcam' button to begin detection.")

    if 'webcam_started' not in st.session_state:
        st.session_state.webcam_started = False

    if st.button("Start Webcam"):
        st.session_state.webcam_started = True
        st.session_state.stop = False

    stop_button = st.button("Stop Webcam", disabled=not st.session_state.webcam_started)

    if stop_button:
        st.session_state.webcam_started = False
        st.session_state.stop = True

    if st.session_state.webcam_started:
        stframe = st.empty()
        vid = cv2.VideoCapture(0)

        while vid.isOpened() and not st.session_state.stop:
            ret, frame = vid.read()
            if not ret:
                st.write("Failed to capture video")
                break

            result = model(frame)

            res_plot = result[0].plot(show=False)
            res_plot = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)
            res_plot = Image.fromarray(res_plot)
            stframe.image(res_plot, caption="Webcam Feed", use_column_width=True)

        vid.release()
        cv2.destroyAllWindows()
        st.write("Webcam stopped.")
