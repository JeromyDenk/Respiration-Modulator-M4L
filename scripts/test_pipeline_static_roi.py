# scripts/test_pipeline_static_roi.py
# Tests pipeline components with a fixed ROI determined during an initial calibration phase.
# Displays the video feed with the static ROI, tracked features, and signal plots.
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
    from pose_detector import PoseDetector
    from roi_calculator import RoiCalculator
    from feature_tracker import FeatureTracker
    from signal_generator import SignalGenerator
    from signal_processor import SignalProcessor
    # from config_loader import load_config # Uncomment if using config files
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure this script is run from the project root directory")
    print("(e.g., using 'python scripts/test_pipeline_static_roi.py')")
    print("or that the 'src' directory exists relative to the project root.")
    sys.exit(1)

print("Initializing Webcam and Pipeline Components...")

# --- Configuration ---
PLOT_HEIGHT = 120 # Height for each signal plot area
PLOT_BG_COLOR = (240, 240, 240) # Light gray background
PLOT_RAW_LINE_COLOR = (100, 100, 100) # Gray for raw signal
PLOT_FILT_LINE_COLOR = (0, 0, 200) # Blue for filtered signal
PLOT_PEAK_COLOR = (0, 0, 255) # Red for peaks
CALIBRATION_DURATION_SEC = 5 # How long to try finding ROI at the start

config = {
    # --- Component Settings ---
    'POSE_MODEL_COMPLEXITY': 0,
    'POSE_MIN_DETECTION_CONFIDENCE': 0.5,
    'POSE_MIN_TRACKING_CONFIDENCE': 0.5,
    'POSE_MIN_LANDMARK_VISIBILITY': 0.6,
    'ROI_PADDING_FACTOR': 1.05,
    'ROI_SHOULDER_ONLY_ASPECT_RATIO': 1.8,
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
     'SIGNAL_MIN_FEATURES_FOR_PCA': 3,
}

DEFAULT_SAMPLING_RATE = 30.0 # Estimate, can be refined

# --- Plotting Helper Function (Copied from previous script) ---
def draw_signal_plot(title, signal_buffer, peak_indices, plot_width, plot_height, bg_color, line_color, peak_color):
    """Draws the signal buffer and peaks onto a NumPy array."""
    plot_img = np.full((plot_height, plot_width, 3), bg_color, dtype=np.uint8)
    buffer_len = len(signal_buffer)
    cv2.putText(plot_img, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    if buffer_len < 2:
        cv2.putText(plot_img, "Waiting for buffer...", (10, plot_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return plot_img
    signal_np = np.array(signal_buffer)
    min_val, max_val = np.min(signal_np), np.max(signal_np)
    range_val = max_val - min_val
    padding = 0.1 * plot_height
    if range_val < 1e-6:
        center_y_flat = plot_height // 2
        cv2.line(plot_img, (0, center_y_flat), (plot_width -1, center_y_flat), line_color, 1)
        normalized_signal = np.full(buffer_len, center_y_flat)
    else:
        scale = (plot_height - 2 * padding) / range_val
        normalized_signal = (signal_np - min_val) * scale + padding
        normalized_signal = plot_height - normalized_signal
        points = np.zeros((buffer_len, 1, 2), dtype=np.int32)
        points[:, 0, 0] = np.linspace(0, plot_width - 1, buffer_len, dtype=np.int32)
        points[:, 0, 1] = normalized_signal.astype(np.int32)
        cv2.polylines(plot_img, [points], isClosed=False, color=line_color, thickness=1, lineType=cv2.LINE_AA)
    if peak_indices is not None and len(peak_indices) > 0 and buffer_len > 1:
        peak_x = (peak_indices / (buffer_len - 1) * (plot_width - 1)).astype(np.int32)
        valid_peak_indices = np.clip(peak_indices, 0, buffer_len - 1)
        peak_y = normalized_signal[valid_peak_indices].astype(np.int32)
        for px, py in zip(peak_x, peak_y):
             px_clamped, py_clamped = max(0, min(plot_width - 1, px)), max(0, min(plot_height - 1, py))
             cv2.circle(plot_img, (px_clamped, py_clamped), 4, peak_color, -1)
    center_y = plot_height // 2
    cv2.line(plot_img, (0, center_y), (plot_width - 1, center_y), (200, 200, 200), 1)
    return plot_img

# --- Initialize Components Manually ---
pose_detector = None
roi_calculator = None
feature_tracker = None
signal_generator = None
signal_processor = None
try:
    pose_detector = PoseDetector(config=config)
    roi_calculator = RoiCalculator(config=config)
    feature_tracker = FeatureTracker(config=config)
    signal_generator = SignalGenerator(config=config)
    signal_processor = SignalProcessor(config=config, sampling_rate=DEFAULT_SAMPLING_RATE)
except Exception as e:
    print(f"Error during component initialization: {e}")
    traceback.print_exc()
    if pose_detector: pose_detector.close()
    sys.exit(1)

# --- MediaPipe Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# --- OpenCV Video Capture Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    if pose_detector: pose_detector.close()
    sys.exit()

print("Webcam opened. Starting calibration phase...")
print(f"Will calibrate for ~{CALIBRATION_DURATION_SEC} seconds to find ROI.")
print("Press 'q' to quit.")

# --- Calibration Phase ---
calibration_start_time = time.time()
static_roi_list = [] # Store the fixed ROI here (as a list)
calibration_successful = False

while time.time() - calibration_start_time < CALIBRATION_DURATION_SEC:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame during calibration.")
        time.sleep(0.01) # Avoid busy-looping on empty frames
        continue

    frame_height, frame_width = frame.shape[:2]
    display_frame = frame.copy()

    # Run pose and ROI detection
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    landmarks = pose_detector.process_frame(image_rgb)
    image_rgb.flags.writeable = True

    current_rois = []
    if landmarks:
        current_rois = roi_calculator.calculate_rois(landmarks, (frame_height, frame_width))
        # Draw pose during calibration
        mp_drawing.draw_landmarks(
            display_frame, landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # If ROI found, store it and mark calibration success
    if current_rois:
        static_roi_list = current_rois # Store the first valid ROI found
        calibration_successful = True
        print(f"Calibration successful! Found ROI: {static_roi_list}")
        # Draw the found ROI during calibration
        for (x, y, w, h) in static_roi_list:
             cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # Cyan ROI box
        cv2.putText(display_frame, "ROI FOUND!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Show the frame briefly with the found ROI
        cv2.putText(display_frame, "CALIBRATING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Static ROI Test - Calibration', display_frame)
        cv2.waitKey(500) # Pause briefly to show found ROI
        break # Exit calibration loop
    else:
        # Display "Calibrating..."
        cv2.putText(display_frame, "CALIBRATING... Looking for ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Static ROI Test - Calibration', display_frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("Exit key pressed during calibration.")
        cap.release()
        cv2.destroyAllWindows()
        if pose_detector: pose_detector.close()
        sys.exit()

cv2.destroyWindow('Static ROI Test - Calibration') # Close calibration window

if not calibration_successful:
    print("Calibration failed: Could not find a stable ROI within the time limit.")
    cap.release()
    if pose_detector: pose_detector.close()
    sys.exit()

print("Calibration finished. Starting main processing loop with static ROI.")

# --- Main Processing Loop (Post-Calibration) ---
prev_time = time.time()
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
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # --- Manual Pipeline Execution with Static ROI ---
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Feature Tracking (using static_roi_list)
    tracked_data = []
    if static_roi_list: # Should always be true after successful calibration
        try:
            tracked_data = feature_tracker.process_frame(image_gray, static_roi_list)
        except Exception as e_track:
            print(f"Error during feature tracking: {e_track}")
            traceback.print_exc()
            tracked_data = [(None, None)] * len(static_roi_list)

    # 2. Signal Generation
    raw_signals = []
    if tracked_data:
        try:
            raw_signals = signal_generator.process_tracked_features(tracked_data)
        except Exception as e_siggen:
            print(f"Error during signal generation: {e_siggen}")
            traceback.print_exc()
            raw_signals = [0.0] * len(tracked_data)
    else:
        raw_signals = [0.0] * len(static_roi_list)

    # 3. Signal Processing
    try:
        signal_processor.process_signal_values(raw_signals)
    except Exception as e_sigproc:
        print(f"Error during signal processing: {e_sigproc}")
        traceback.print_exc()

    # --- Gather Results ---
    bpm, bpm_valid = signal_processor.get_bpm()
    phase = signal_processor.get_phase()
    raw_signal_history = signal_processor.get_raw_signal_buffer()
    filtered_signal_history = signal_processor.get_filtered_signal_buffer()
    peak_indices = signal_processor.get_last_peak_indices()
    feature_points_per_roi = feature_tracker.prev_features_per_roi # Get current points

    # --- Prepare image for drawing ---
    display_frame = frame.copy()

    # --- Generate Plots ---
    raw_plot_image = draw_signal_plot(
        "Raw Signal (Fused)", raw_signal_history, None,
        frame_width, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_RAW_LINE_COLOR, PLOT_PEAK_COLOR
    )
    filtered_plot_image = draw_signal_plot(
        "Filtered Signal & Peaks", filtered_signal_history, peak_indices,
        frame_width, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_FILT_LINE_COLOR, PLOT_PEAK_COLOR
    )

    # --- Draw Overlays on Webcam Frame ---
    # 1. Draw STATIC ROI
    if static_roi_list:
        roi_color = (255, 0, 0) # Blue for static ROI
        for i, (x, y, w, h) in enumerate(static_roi_list):
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), roi_color, 2)
            cv2.putText(display_frame, f"Static ROI {i}", (x + 5, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

    # 2. Draw Tracked Feature Points
    point_color = (0, 0, 255) # Red
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

    # 3. Display BPM and Phase
    bpm_text = f"BPM: {bpm:.1f}" if bpm_valid else "BPM: ---"
    cv2.putText(display_frame, bpm_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow

    phase_map = {signal_processor.PHASE_INHALE: "Inhale",
                 signal_processor.PHASE_EXHALE: "Exhale",
                 signal_processor.PHASE_UNKNOWN: "---"}
    phase_text = f"Phase: {phase_map.get(phase, 'Error')}"
    cv2.putText(display_frame, phase_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Display FPS & Frame Count ---
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Frame: {frame_count}", (frame_width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # --- Combine Webcam Frame and Plots ---
    combined_display = np.vstack((display_frame, raw_plot_image, filtered_plot_image))

    # --- Display the resulting combined frame ---
    cv2.imshow('Static ROI Test - Overlays & Plots', combined_display)

    # --- User Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        print("Exit key pressed.")
        break
    # No 'r' key needed as ROI is static

# --- Cleanup ---
print("Releasing resources...")
if pose_detector: pose_detector.close()
# Add close methods for other components if they implement them
cap.release()
cv2.destroyAllWindows()
print("Finished.")