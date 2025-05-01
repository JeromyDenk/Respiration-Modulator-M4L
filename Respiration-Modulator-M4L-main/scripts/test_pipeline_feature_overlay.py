# scripts/test_pipeline_feature_overlay.py
# Tests the full pipeline via PipelineManager using a live webcam feed.
# Displays the video feed with pose, ROI, features, and plots for raw and filtered signals.
# MODIFIED: Added raw signal plot above the filtered signal plot. Added debug print for raw signal history.
# ASSUMES this script lives in a 'scripts' subdirectory of the project root.

import cv2
import mediapipe as mp
import time
import numpy as np
import os
import sys
import traceback

# --- Add src directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import necessary classes from src ---
try:
    from pipeline_manager import PipelineManager
    # from config_loader import load_config # Uncomment if using config files
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure this script is run from the project root directory")
    print("(e.g., using 'python scripts/test_pipeline_feature_overlay.py')")
    print("or that the 'src' directory exists relative to the project root.")
    sys.exit(1)

print("Initializing Webcam and PipelineManager...")

# --- Configuration ---
POSE_RUN_INTERVAL = 30
PLOT_HEIGHT = 120 # Height for each signal plot area
PLOT_BG_COLOR = (240, 240, 240) # Light gray background
PLOT_RAW_LINE_COLOR = (100, 100, 100) # Gray for raw signal
PLOT_FILT_LINE_COLOR = (0, 0, 200) # Blue for filtered signal
PLOT_PEAK_COLOR = (0, 0, 255) # Red for peaks

config = {
    'POSE_DETECTION_FRAME_INTERVAL': POSE_RUN_INTERVAL,
    'POSE_MODEL_COMPLEXITY': 0,
    'SIGNAL_FILTER_METHOD': 'lfilter',
    'SIGNAL_BUFFER_SECONDS': 10.0,
    'BPM_AVERAGING_SECONDS': 3.0,
    'FEATURE_REDETECT_THRESHOLD': 20,
    'OPTICAL_FLOW_PARAMS': {
         'feature_params': {'maxCorners': 80, 'qualityLevel': 0.3, 'minDistance': 7},
         'lk_params': {'winSize': (15, 15), 'maxLevel': 2}
     },
     'SIGNAL_FILTER_LOW_HZ': 0.1,
     'SIGNAL_FILTER_HIGH_HZ': 2.0,
     'PEAK_DETECT_MIN_DISTANCE_SEC': 0.5,
     'PEAK_DETECT_PROMINENCE': None, # Start with None, tune visually
     'PHASE_SLOPE_WINDOW_MS': 100,
}

DEFAULT_SAMPLING_RATE = 30.0

# --- Plotting Helper Function (Updated) ---
def draw_signal_plot(title, signal_buffer, peak_indices, plot_width, plot_height, bg_color, line_color, peak_color):
    """Draws the signal buffer and peaks onto a NumPy array."""
    plot_img = np.full((plot_height, plot_width, 3), bg_color, dtype=np.uint8)
    buffer_len = len(signal_buffer)

    # Add title to the plot
    cv2.putText(plot_img, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if buffer_len < 2: # Need at least 2 points to draw a line
        cv2.putText(plot_img, "Waiting for buffer...", (10, plot_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return plot_img

    signal_np = np.array(signal_buffer)

    # Normalize signal to fit plot height (with some padding)
    min_val = np.min(signal_np)
    max_val = np.max(signal_np)
    range_val = max_val - min_val
    padding = 0.1 * plot_height # 10% padding top and bottom

    # Avoid division by zero if signal is flat
    if range_val < 1e-6:
        # If flat, draw a line in the middle
        center_y_flat = plot_height // 2
        cv2.line(plot_img, (0, center_y_flat), (plot_width -1, center_y_flat), line_color, 1)
        normalized_signal = np.full(buffer_len, center_y_flat) # Needed for peak drawing if any
    else:
        scale = (plot_height - 2 * padding) / range_val
        normalized_signal = (signal_np - min_val) * scale + padding
        # Flip vertically because OpenCV origin (0,0) is top-left
        normalized_signal = plot_height - normalized_signal

        # Create points for polyline
        points = np.zeros((buffer_len, 1, 2), dtype=np.int32)
        points[:, 0, 0] = np.linspace(0, plot_width - 1, buffer_len, dtype=np.int32)
        points[:, 0, 1] = normalized_signal.astype(np.int32)
        # Draw the signal line
        cv2.polylines(plot_img, [points], isClosed=False, color=line_color, thickness=1, lineType=cv2.LINE_AA)


    # Draw peaks (only if peak_indices are provided and not None)
    if peak_indices is not None and len(peak_indices) > 0:
        # Scale peak indices to plot width
        # Ensure buffer_len > 1 to avoid division by zero if buffer just filled
        if buffer_len > 1:
            peak_x = (peak_indices / (buffer_len - 1) * (plot_width - 1)).astype(np.int32)
            # Get corresponding y values from normalized signal
            valid_peak_indices = np.clip(peak_indices, 0, buffer_len - 1)
            peak_y = normalized_signal[valid_peak_indices].astype(np.int32)

            for px, py in zip(peak_x, peak_y):
                 # Clamp coordinates just in case
                 px_clamped = max(0, min(plot_width - 1, px))
                 py_clamped = max(0, min(plot_height - 1, py))
                 cv2.circle(plot_img, (px_clamped, py_clamped), 4, peak_color, -1) # Draw filled circles for peaks

    # Optional: Draw simple baseline/center line
    center_y = plot_height // 2
    cv2.line(plot_img, (0, center_y), (plot_width - 1, center_y), (200, 200, 200), 1)

    return plot_img

# --- Initialize Components ---
pipeline = None
try:
    pipeline = PipelineManager(config=config, sampling_rate=DEFAULT_SAMPLING_RATE)
except Exception as e:
    print(f"Error during PipelineManager initialization: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- MediaPipe Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# --- OpenCV Video Capture Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    if pipeline: pipeline.close()
    sys.exit()

print(f"Webcam opened. Pose/ROI detection will run every {POSE_RUN_INTERVAL} frames.")
print("Press 'q' to quit.")
print("Press 'r' to force recalibration.")

# --- Frame Processing Loop ---
prev_time = time.time()
frame_num_for_debug_print = 0 # Separate counter for less frequent printing

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    current_pipeline_frame_num = pipeline.frame_count + 1 # Get count before processing
    frame_height, frame_width = frame.shape[:2]
    frame_num_for_debug_print += 1

    # --- Performance calculation (FPS) ---
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # --- Process Frame through Pipeline ---
    pipeline_results = pipeline.process_frame(frame)

    # --- Prepare image for drawing ---
    display_frame = frame.copy() # Webcam feed part

    # --- Initialize plot images ---
    raw_plot_image = np.full((PLOT_HEIGHT, frame_width, 3), PLOT_BG_COLOR, dtype=np.uint8)
    cv2.putText(raw_plot_image, "Raw Signal - Initializing...", (10, PLOT_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    filtered_plot_image = np.full((PLOT_HEIGHT, frame_width, 3), PLOT_BG_COLOR, dtype=np.uint8)
    cv2.putText(filtered_plot_image, "Filtered Signal - Initializing...", (10, PLOT_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)


    # --- Draw Overlays on Webcam Frame & Generate Plots ---
    pose_run_attempted_this_frame = False
    if pipeline_results:
        landmarks = pipeline_results.get('landmarks')
        current_rois = pipeline_results.get('current_rois', [])
        bpm = pipeline_results.get('bpm', 0.0)
        bpm_valid = pipeline_results.get('bpm_valid', False)
        phase = pipeline_results.get('phase', 0)
        pose_run_attempted_this_frame = pipeline_results.get('pose_run_attempted', False)
        raw_signal = pipeline_results.get('raw_signal_history', [])
        filtered_signal = pipeline_results.get('filtered_signal_history', [])
        peak_indices = pipeline_results.get('peak_indices', [])

        # --- *** DEBUG PRINT *** ---
        # Print summary of raw signal buffer periodically
        if frame_num_for_debug_print % 60 == 0: # Print every ~2 seconds at 30fps
             if raw_signal:
                 print(f"[Debug Frame {current_pipeline_frame_num}] Raw Signal Stats: "
                       f"Len={len(raw_signal)}, Min={np.min(raw_signal):.4f}, "
                       f"Max={np.max(raw_signal):.4f}, Std={np.std(raw_signal):.4f}")
             else:
                 print(f"[Debug Frame {current_pipeline_frame_num}] Raw Signal: Empty")
        # --- *** END DEBUG PRINT *** ---


        # --- Generate Plots ---
        raw_plot_image = draw_signal_plot(
            "Raw Signal (Fused)", raw_signal, None, # No peaks on raw plot
            frame_width, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_RAW_LINE_COLOR, PLOT_PEAK_COLOR
        )
        filtered_plot_image = draw_signal_plot(
            "Filtered Signal & Peaks", filtered_signal, peak_indices,
            frame_width, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_FILT_LINE_COLOR, PLOT_PEAK_COLOR
        )

        # --- Draw on Webcam Frame ---
        # (Drawing code for pose, roi, features, bpm, phase remains the same)
        # 1. Draw Pose Landmarks
        if landmarks:
            mp_drawing.draw_landmarks(
                display_frame, landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        elif not pipeline.last_landmarks and current_pipeline_frame_num > 1:
             cv2.putText(display_frame, "No Pose Yet/Failed", (frame_width - 250, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 2. Draw Calculated ROI(s)
        if current_rois:
            roi_color = (0, 255, 0) if not pose_run_attempted_this_frame else (255, 100, 0)
            for i, (x, y, w, h) in enumerate(current_rois):
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), roi_color, 2)
                cv2.putText(display_frame, f"ROI {i}", (x + 5, y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)
        elif pose_run_attempted_this_frame and not current_rois:
            cv2.putText(display_frame, "ROI Calc Failed", (frame_width - 200, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif not current_rois and current_pipeline_frame_num > 1:
             cv2.putText(display_frame, "No ROI Yet", (frame_width - 150, 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # 3. Draw Tracked Feature Points
        feature_points_per_roi = pipeline.feature_tracker.prev_features_per_roi
        point_color = (0, 0, 255)
        feature_count = 0
        for roi_index, points in feature_points_per_roi.items():
            if points is not None:
                feature_count += len(points)
                for point in points:
                    try:
                        x, y = int(point[0][0]), int(point[0][1])
                        cv2.circle(display_frame, (x, y), 3, point_color, -1)
                    except IndexError: pass
        cv2.putText(display_frame, f"Features: {feature_count}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 4. Display BPM and Phase
        bpm_text = f"BPM: {bpm:.1f}" if bpm_valid else "BPM: ---"
        cv2.putText(display_frame, bpm_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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


    # --- Display FPS & Frame Count on Webcam Frame ---
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Frame: {current_pipeline_frame_num}", (frame_width - 150, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Indicate when Pose/ROI is running on Webcam Frame
    if pose_run_attempted_this_frame:
         cv2.putText(display_frame, "POSE/ROI RUNNING", (frame_width // 2 - 100, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)


    # --- Combine Webcam Frame and Plots ---
    # Stack: Webcam feed -> Raw Plot -> Filtered Plot
    combined_display = np.vstack((display_frame, raw_plot_image, filtered_plot_image))

    # --- Display the resulting combined frame ---
    cv2.imshow('Pipeline Test - Overlays & Dual Plots', combined_display) # Updated window title

    # --- User Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        print("Exit key pressed.")
        break
    elif key == ord('r'):
        print("Recalibration key pressed.")
        pipeline.trigger_recalibration()


# --- Cleanup ---
print("Releasing resources...")
if pipeline: pipeline.close()
cap.release()
cv2.destroyAllWindows()
print("Finished.")
