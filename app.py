# app.py
import streamlit as st
import os
from yolov8_utils.counter import VehicleCounter
import time
from pathlib import Path

st.set_page_config(page_title="Vehicle Detection and Counting", layout="wide")
st.title("🚗 Vehicle Detection and Counting using YOLOv8")

# Ensure folders exist
Path("uploads").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file
    input_path = os.path.join("uploads", uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"Uploaded: {uploaded_file.name}")
    
    # Select model
    model_type = st.selectbox("Select YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
    
    if st.button("▶️ Start Detection and Counting"):
        with st.spinner("Processing video... This may take a while."):
            # Output path
            output_filename = f"output_{uploaded_file.name}"
            output_path = os.path.join("outputs", output_filename)

            # Initialize and run counter
            counter = VehicleCounter(model_path=model_type)
            success = counter.process_video(input_path, output_path)

        if success:
            st.success("Processing completed!")
            st.video(output_path)
            st.download_button("📥 Download Processed Video", data=open(output_path, "rb").read(), file_name=output_filename)
        else:
            st.error("Failed to process video. Please check logs.")
