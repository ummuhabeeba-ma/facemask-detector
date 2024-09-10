import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best.pt")

# Streamlit page setup
st.set_page_config(page_title="Mask Detector", page_icon="😷")
st.title("🦠 Real-Time Face Mask Detection")
st.write("Welcome to the Mask Detection Application. Choose an option to detect masks:")

# Selection box to choose image source
option = st.selectbox("How would you like to provide an image for detection?", ("Upload Image", "Capture from Webcam"))

# Handle image input based on option
if option == "Upload Image":
    image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
else:
    image = st.camera_input("Capture image")

# Process the image
if image is not None:
    st.write("## Image")
    
    # Open the image using PIL
    img = Image.open(image)
    
    # Display the uploaded or captured image
    st.image(img, caption='Uploaded/Captured Image', use_column_width=True)

    # Convert image to numpy array for YOLO model
    img_array = np.array(img.convert("RGB"))

    # Run YOLO model on the image
    results = model(img_array)

    if results:
        # Plot the results
        result = results[0]
        res_plot = result.plot(show=False)

        # Convert result to Image object for display
        res_plot = Image.fromarray(res_plot)
        
        # Display processed image with results
        st.image(res_plot, caption="Processed Image", use_column_width=True)
    else:
        st.write("No results from the model. Please check the model and input image.")
else:
    st.info("Please upload an image or capture a picture.")
