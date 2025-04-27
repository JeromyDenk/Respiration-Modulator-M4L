# scripts/test_pipeline_feature_overlay.py
# Tests the full pipeline via PipelineManager using a live webcam feed.
# Displays the video feed with pose landmarks, calculated ROI, and tracked feature points overlay.
# MODIFIED: Uses VideoInput class and Nth frame pose detection.
# ASSUMES this script lives in a 'scripts' subdirectory of the project root.

import cv2
import mediapipe as mp
import time
import numpy as np
import os
import sys
import traceback
import json # To load profile

# --- Add src directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import necessary classes from src ---
try:
    from video_input import VideoInput # Import VideoInput
    from pipeline_manager import PipelineManager
    # Config loader might be needed if not using hardcoded config below
    # from config_loader import load_config
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure required modules exist (VideoInput, PipelineManager, etc.).")
    sys.exit(1)

print("Initializing Webcam (via VideoInput) and PipelineManager...")

# --- Load Configuration ---
config_path = os.path.join(project_root, "profiles", "test_profile.json")
config = {}
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {config_path}")
except FileNotFoundError:
    print(f"Warning: Configuration file not found at {config_path}. Using default empty config.")
    config = {} # Ensure config is an empty dict if file not found
except json.JSONDecodeError:
     print(f"Warning: Could not parse configuration file {config_path}. Using default empty config.")
     config = {} # Ensure config is an empty dict if parse error
except Exception as e:
     print(f"Warning: Error loading configuration: {e}. Using default empty config.")
     config = {} # Ensure config is an empty dict on other errors

# Extract specific config sections or use defaults
video_config = config.get("video_input", {})
# PipelineManager needs the full config to pass down
pipeline_config = config

DEFAULT_SAMPLING_RATE = 30.0 # Fallback sampling rate

# --- Initialize Video Input ---
video_input = None
try:
    video_input = VideoInput(config=video_config)
    if not video_input.initialized:
        raise RuntimeError("VideoInput failed to initialize.")
    # Use actual sampling rate if available, otherwise default
    actual_fps = video_input.get_fps()
    sampling_rate = actual_fps if actual_fps > 0 else DEFAULT_SAMPLING_RATE
    print(f"Using sampling rate for PipelineManager: {sampling_rate:.2f} Hz")
except Exception as e:
    print(f"Error initializing VideoInput: {e}")
    if video_input: video_input.release()
    sys.exit(1)


# --- Initialize Pipeline Manager ---
pipeline = None
try:
    # Pass the full config and the determined sampling rate
    pipeline = PipelineManager(config=pipeline_config, sampling_rate=sampling_rate)

except Exception as e:
    print(f"Error during PipelineManager initialization: {e}")
    traceback.print_exc()
    video_input.release() # Release video input if pipeline fails
    sys.exit(1)

# --- MediaPipe Drawing Utilities (for pose) ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose # Need this for POSE_CONNECTIONS

print("Webcam opened. Starting video stream...")
# Get pose interval from config for display message
pose_interval = pipeline.pose_detection_interval # Access interval from pipeline instance
print(f"Pose/ROI detection will run every {pose_interval} frame(s).")
print("Press 'q' to quit.")
print("Press 'r' to force recalibration.")

# --- Frame Processing Loop ---
prev_time = time.time() # Use for FPS calculation

while True: # Loop until user quits or error
    # --- Read Frame using VideoInput ---
    success, frame = video_input.get_frame()
    if not success or frame is None:
        print("End of video source or cannot read frame. Exiting.")
        break # Exit loop if frame read fails

    # Get actual resolution from VideoInput object
    frame_width, frame_height = video_input.get_resolution()
    if frame_width == 0 or frame_height == 0: # Check if resolution is valid
        print("Error: Invalid frame dimensions received from VideoInput. Exiting.")
        break

    # Get the pipeline's internal frame count *before* processing the current frame
    current_pipeline_frame_num = pipeline.frame_count + 1

    # --- Performance calculation (FPS) ---
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # --- Process Frame through Pipeline ---
    pipeline_results = pipeline.process_frame(frame)

    # --- Prepare image for drawing (use original BGR frame) ---
    display_frame = frame.copy()

    # --- Draw Overlays ---
    pose_run_attempted_this_frame = False # Default
    if pipeline_results:
        landmarks = pipeline_results.get('landmarks')
        current_rois = pipeline_results.get('current_rois', [])
        bpm = pipeline_results.get('bpm', 0.0)
        bpm_valid = pipeline_results.get('bpm_valid', False)
        phase = pipeline_results.get('phase', 0)
        pose_run_attempted_this_frame = pipeline_results.get('pose_run_attempted', False)

        # 1. Draw Pose Landmarks (if available)
        if landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # Indicate if pose hasn't run successfully yet or failed last time
        elif not pipeline.last_landmarks and current_pipeline_frame_num > 1:
             cv2.putText(display_frame, "No Pose Yet/Failed", (frame_width - 250, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 2. Draw Calculated ROI(s)
        if current_rois:
            # Use different color if ROI was just updated this frame
            roi_color = (0, 255, 0) # Default Green
            if pipeline_results.get('recalibrated_this_frame', False):
                 roi_color = (255, 100, 0) # Blue if just updated

            for i, (x, y, w, h) in enumerate(current_rois):
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), roi_color, 2)
                # Display ROI type (assuming single ROI for now)
                roi_label = f"ROI {i}"
                if pipeline.evm_enabled and pipeline_results.get('recalibrated_this_frame', False):
                     roi_label += " (Refined)"
                elif pipeline.evm_enabled:
                     roi_label += " (Refined - Old)"
                else:
                     roi_label += " (Coarse)"

                cv2.putText(display_frame, roi_label, (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)
        # Indicate ROI failure only if pose was attempted but ROI failed
        elif pose_run_attempted_this_frame and not current_rois:
            cv2.putText(display_frame, "ROI Calc Failed", (frame_width - 200, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # Indicate if no ROI because pose hasn't run/succeeded yet
        elif not current_rois and current_pipeline_frame_num > 1:
             cv2.putText(display_frame, "No ROI Yet", (frame_width - 150, 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange text


        # 3. Draw Tracked Feature Points
        # Access the feature tracker's state via the pipeline manager instance
        feature_points_per_roi = pipeline.feature_tracker.prev_features_per_roi
        point_color = (0, 0, 255) # Red for feature points
        feature_count = 0
        for roi_index, points in feature_points_per_roi.items():
            if points is not None:
                feature_count += len(points)
                for point in points:
                    try:
                        if point.shape == (1, 2):
                            x, y = int(point[0, 0]), int(point[0, 1])
                            cv2.circle(display_frame, (x, y), 3, point_color, -1) # Draw small filled circle
                    except IndexError: pass
                    except Exception as draw_err: print(f"ERROR drawing point {point}: {draw_err}")

        # Display feature count
        cv2.putText(display_frame, f"Features: {feature_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) # Blue text


        # 4. Display BPM and Phase
        bpm_text = f"BPM: {bpm:.1f}" if bpm_valid else "BPM: ---"
        cv2.putText(display_frame, bpm_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow text

        phase_map = {pipeline.signal_processor.PHASE_INHALE: "Inhale",
                     pipeline.signal_processor.PHASE_EXHALE: "Exhale",
                     pipeline.signal_processor.PHASE_UNKNOWN: "---"}
        phase_text = f"Phase: {phase_map.get(phase, 'Error')}"
        cv2.putText(display_frame, phase_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        # Indicate if pipeline processing failed entirely
        cv2.putText(display_frame, "Pipeline Error", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # --- Display FPS & Frame Count ---
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Display pipeline's internal frame count
    cv2.putText(display_frame, f"Frame: {current_pipeline_frame_num}", (frame_width - 150, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Indicate when Pose/ROI is running
    if pose_run_attempted_this_frame:
         cv2.putText(display_frame, "POSE/ROI RUNNING", (frame_width // 2 - 100, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)


    # --- Display the resulting frame ---
    # No resizing here, show at native (or requested) resolution
    cv2.imshow('Pipeline Test - Feature Overlay', display_frame)

    # --- User Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        print("Exit key pressed.")
        break
    elif key == ord('r'):
        print("Recalibration key pressed.")
        pipeline.trigger_recalibration() # Tell manager to run pose on next frame


# --- Cleanup ---
print("Releasing resources...")
if pipeline: pipeline.close() # Close the pipeline manager and its components
if video_input: video_input.release() # Release VideoInput
cv2.destroyAllWindows()
print("Finished.")
