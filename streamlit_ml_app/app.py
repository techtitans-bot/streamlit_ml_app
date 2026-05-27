import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import cv2
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_detections")
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

st.set_page_config(page_title="Live Object Detection & Tracing", layout="wide")
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

st.sidebar.title("⚙️ Detection Settings")
model_choice = st.sidebar.selectbox("Model Version", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
model = load_model(model_choice)

class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_capture_time = 0
        self.countdown_start = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results = model.track(img, persist=True, conf=conf_threshold, verbose=False)
        
        annotated_frame = results[0].plot()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            class_ids = results[0].boxes.cls.int().tolist()
            non_person_objects = [cls for cls in class_ids if cls != 0]
            
            if len(non_person_objects) > 0:
                if self.countdown_start is None:
                    self.countdown_start = time.time()
                
                elapsed = time.time() - self.countdown_start
                remaining = max(0, 3 - int(elapsed))

                cv2.putText(annotated_frame, f"OBJECT DETECTED! SAVING IN {remaining}", (50, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                if remaining == 0 and (time.time() - self.last_capture_time > 10):
                    filename = os.path.join(SAVE_DIR, f"cap_{int(time.time())}.jpg")
                    success = cv2.imwrite(filename, annotated_frame)
                    
                    if success:
                        print(f"DEBUG SUCCESS: Saved image with boxes to {filename}")
                    
                    self.last_capture_time = time.time()
                    self.countdown_start = None 
            else:
                self.countdown_start = None 

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

st.title("🚨Alert System")
st.write(f"Saving images to: {SAVE_DIR}")

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    webrtc_streamer(
        key="object-detection",
        video_transformer_factory=ObjectDetectionTransformer,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )