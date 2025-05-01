# scripts/test_pose_roi_webcam.py
# Tests PoseDetector and RoiCalculator together using a live webcam feed.
# Displays the video feed with pose landmarks and calculated ROI overlay.
# ASSUMES this script lives in a 'scripts' subdirectory of the project root.

import cv2
import mediapipe as mp
import time
import numpy as np
import os
import sys
import traceback

# --- Add src directory to Python path ---
# Get the directory where this script lives (e.g., .../scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go one level up to the project root directory (e.g., ...)
project_root = os.path.dirname(script_dir)
# Construct the path to the 'src' directory (e.g., .../src/)
src_dir = os.path.join(project_root, 'src')

# Add src directory to the path if it's not already there
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import necessary classes from src ---
try:
    # Import the classes we need to test
    from pose_detector import PoseDetector
    from coarse_roi_calculator import RoiCalculator
    # Import config loader to potentially load settings if needed, or use defaults
    from config_loader import load_config
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure this script is run from the project root directory (e.g., using 'python scripts/test_pose_roi_webcam.py')")
    print("or that the 'src' directory exists relative to the project root.")
    sys.exit(1)

print("Initializing Webcam, PoseDetector, and RoiCalculator...")

# --- Configuration ---
# For this test, we'll use default settings by passing an empty dict
# You could load a profile here if needed, adjusting the path:
# config_path = os.path.join(project_root, "profiles", "your_profile.json")
# config = load_config(config_path)
config = {}

# --- Initialize Components ---
try:
    pose_detector = PoseDetector(config=config)
    if not pose_detector.initialized:
        raise RuntimeError("PoseDetector failed to initialize.")

    roi_calculator = RoiCalculator(config=config)

except Exception as e:
    print(f"Error during component initialization: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- MediaPipe Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose # Need this for POSE_CONNECTIONS

# --- OpenCV Video Capture Initialization ---
cap = cv2.VideoCapture(0) # Use camera index 0 (default webcam)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    if pose_detector: pose_detector.close() # Attempt cleanup
    sys.exit()

print("Webcam opened. Starting video stream...")
print("Press 'q' to quit.")

# --- Frame Processing Loop ---
prev_time = 0
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_count += 1
    frame_height, frame_width = frame.shape[:2]

    # --- Performance calculation (FPS) ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # --- MediaPipe Processing ---
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False # Performance hint

    # Process for landmarks
    landmarks = pose_detector.process_frame(image_rgb)

    # Prepare image for drawing (use original BGR frame)
    image_bgr = frame
    image_bgr.flags.writeable = True

    # --- ROI Calculation & Drawing ---
    calculated_rois = []
    if landmarks:
        # Calculate ROI(s) based on detected landmarks
        calculated_rois = roi_calculator.calculate_rois(landmarks, (frame_height, frame_width))

        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            image_bgr,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Draw the calculated ROI(s)
        if calculated_rois:
            for (x, y, w, h) in calculated_rois:
                cv2.rectangle(image_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green ROI box
                cv2.putText(image_bgr, "ROI", (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Indicate if ROI calculation failed despite landmarks being present
             cv2.putText(image_bgr, "ROI Calc Failed", (frame_width - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        # Indicate if no landmarks were found
        cv2.putText(image_bgr, "No Pose Detected", (frame_width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # --- Display FPS ---
    cv2.putText(image_bgr, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Display the resulting frame ---
    cv2.imshow('Pose Detector + ROI Calculator Test', image_bgr)

    # --- Exit Condition ---
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Exit key pressed.")
        break

# --- Cleanup ---
print("Releasing resources...")
if pose_detector: pose_detector.close() # Close the pose model
cap.release()
cv2.destroyAllWindows()
print("Finished.")
