import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🔍", layout="wide")

st.title("🔍 YOLOv8 Object Detection")
st.markdown("**Built by Pradeep Singh** | Custom Trained YOLOv8 Model")
st.markdown("---")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

st.sidebar.header("Input Type")
input_type = st.sidebar.radio("Select Input", ["Image", "Video", "Webcam"])

if input_type == "Image":
    st.header("Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with st.spinner("Detecting..."):
            results = model.predict(source=image, conf=confidence)
            result_image = results[0].plot()
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        with col2:
            st.subheader("Detection Result")
            st.image(result_image, use_container_width=True)
        st.markdown("---")
        st.subheader("Detection Details")
        boxes = results[0].boxes
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                st.write(f"**{i+1}.** {name} - Confidence: {conf:.2f}")
        else:
            st.write("No objects detected.")

elif input_type == "Video":
    st.header("Video Detection")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=confidence, verbose=False)
            result_frame = results[0].plot()
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            stframe.image(result_frame, use_container_width=True)
        cap.release()
        st.success("Video processing complete!")

elif input_type == "Webcam":
    st.header("Webcam Detection")
    start = st.button("Start Webcam")
    stop = st.button("Stop Webcam")
    if start:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break
            results = model.predict(source=frame, conf=confidence, verbose=False)
            result_frame = results[0].plot()
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            stframe.image(result_frame, use_container_width=True)
        cap.release()

st.markdown("---")
st.markdown("Made with love by **Pradeep Singh** | Powered by **YOLOv8 + Streamlit**")