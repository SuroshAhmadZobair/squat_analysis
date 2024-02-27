import cv2
import numpy as np
import streamlit as st

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner

st.title('AI Fitness Trainer: Squats Analysis')

thresholds = get_thresholds_beginner()
live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)
pose = get_mediapipe_pose()

# Define the video capture device
cap = cv2.VideoCapture(0)

def video_frame_callback(frame):
    ret, frame = cap.read()
    if not ret:
        return None
    frame, _ = live_process_frame.process(frame, pose)
    return frame

# Display the processed video stream
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    processed_frame, _ = live_process_frame.process(frame, pose)
    cv2.imshow('Processed Frame', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
