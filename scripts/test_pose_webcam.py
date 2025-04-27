import cv2
import mediapipe as mp
import time
import numpy as np

print("Initializing MediaPipe Pose and Webcam...")

# --- MediaPipe Pose Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Pose model (using 'with' ensures resources are cleaned up)
# Using default parameters for simplicity in this test
try:
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    print("MediaPipe Pose initialized.")
except Exception as e:
    print(f"Error initializing MediaPipe Pose: {e}")
    print("Please ensure MediaPipe is installed correctly.")
    exit()

# --- OpenCV Video Capture Initialization ---
# Use camera index 0 (default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

print("Webcam opened. Starting video stream...")
print("Press 'q' to quit.")

# --- Frame Processing Loop ---
prev_time = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

    # --- Performance calculation (FPS) ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # --- MediaPipe Processing ---
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and find poses.
    results = pose.process(image_rgb)
    # Convert the image back to BGR.
    frame.flags.writeable = True
    image_bgr = frame # Use the original frame for drawing

    # --- Draw the pose annotation on the image ---
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # --- Display FPS ---
    cv2.putText(image_bgr, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Display the resulting frame ---
    cv2.imshow('MediaPipe Pose Test', image_bgr)

    # --- Exit Condition ---
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Exit key pressed.")
        break

# --- Cleanup ---
print("Releasing resources...")
pose.close() # Close the pose model
cap.release()
cv2.destroyAllWindows()
print("Finished.")
