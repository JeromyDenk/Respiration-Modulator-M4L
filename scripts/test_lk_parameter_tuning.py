# scripts/test_lk_parameter_tuning.py
# Helps tune FeatureTracker parameters by cycling through predefined sets
# and visualizing the tracked points and resulting signals (time and frequency).
# MODIFIED: Corrected SyntaxError in feature point drawing loop.

import cv2
import mediapipe as mp
import time
import numpy as np
import os
import sys
import traceback
import copy # To deep copy parameter dictionaries
import collections # For signal processor buffers
from scipy.fft import fft, fftfreq # For frequency plot

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
    # IMPORTANT: Assumes roi_calculator.py has been renamed to coarse_roi_calculator.py
    from coarse_roi_calculator import CoarseRoiCalculator
    from feature_tracker import FeatureTracker # The module we are tuning
    from signal_generator import SignalGenerator # Added
    from signal_processor import SignalProcessor # Added
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure required modules exist and RoiCalculator has been renamed.")
    sys.exit(1)

print("Initializing Webcam and Pose/ROI Components for Calibration...")

# --- Configuration ---
# Basic config for Pose/ROI
pose_config = {'POSE_MODEL_COMPLEXITY': 0}
coarse_roi_config = {} # Use defaults
# --- Increased Calibration Duration ---
CALIBRATION_DURATION_SEC = 10 # Increased to 10 seconds

# Signal Processing Config (can be adjusted or loaded from profile)
signal_config = {
    'SIGNAL_FILTER_METHOD': 'lfilter',
    'SIGNAL_BUFFER_SECONDS': 10.0,
    'BPM_AVERAGING_SECONDS': 3.0,
    'SIGNAL_FILTER_LOW_HZ': 0.1,
    'SIGNAL_FILTER_HIGH_HZ': 2.0,
    'PEAK_DETECT_MIN_DISTANCE_SEC': 0.5,
    'PEAK_DETECT_PROMINENCE': None, # Tune this based on observed filtered signal
    'PHASE_SLOPE_WINDOW_MS': 100,
    'SIGNAL_MIN_FEATURES_FOR_PCA': 3,
}
DEFAULT_SAMPLING_RATE = 30.0 # Estimate

# Plotting Config
PLOT_HEIGHT = 100 # Reduced height for each plot to fit more
PLOT_BG_COLOR = (240, 240, 240) # Light gray background for plot
PLOT_RAW_LINE_COLOR = (100, 100, 100) # Gray for raw signal
PLOT_FILT_LINE_COLOR = (0, 0, 200) # Blue for filtered signal
PLOT_FREQ_LINE_COLOR = (0, 150, 0) # Green for frequency plot
PLOT_PEAK_COLOR = (0, 0, 255) # Red for peaks
FREQ_PLOT_MAX_HZ = 5.0 # Max frequency to display on the FFT plot


# === PARAMETER SETS TO TEST ===
# (Parameter sets list remains the same as previous version)
lk_criteria_default = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
parameter_sets = [
    { # Set 0: Default-ish
        'name': 'Default-ish',
        'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
        'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': lk_criteria_default}
    },
    { # Set 1: More Corners, Lower Quality
        'name': 'More/LowerQ Feat',
        'feature_params': {'maxCorners': 150, 'qualityLevel': 0.1, 'minDistance': 5, 'blockSize': 7},
        'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': lk_criteria_default}
    },
    { # Set 2: Fewer Corners, Higher Quality
        'name': 'Fewer/HigherQ Feat',
        'feature_params': {'maxCorners': 50, 'qualityLevel': 0.5, 'minDistance': 10, 'blockSize': 7},
        'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': lk_criteria_default}
    },
    { # Set 3: Larger LK Window
        'name': 'Larger LK Win',
        'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
        'lk_params': {'winSize': (25, 25), 'maxLevel': 2, 'criteria': lk_criteria_default}
    },
    { # Set 4: Smaller LK Window
        'name': 'Smaller LK Win',
        'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
        'lk_params': {'winSize': (9, 9), 'maxLevel': 2, 'criteria': lk_criteria_default}
    },
    { # Set 5: More Pyramid Levels
        'name': 'More LK Levels',
        'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
        'lk_params': {'winSize': (15, 15), 'maxLevel': 4, 'criteria': lk_criteria_default}
    },
    { # Set 6: Stricter LK Criteria (more iterations)
        'name': 'Stricter LK Criteria',
        'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
        'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)}
    },
]

# --- Plotting Helper Functions ---
# (draw_signal_plot and draw_frequency_plot remain unchanged)
def draw_signal_plot(title, signal_buffer, peak_indices, plot_width, plot_height, bg_color, line_color, peak_color=None):
    """Draws the signal buffer (time domain) and optional peaks onto a NumPy array."""
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

    if peak_indices is not None and peak_color is not None and len(peak_indices) > 0 and buffer_len > 1:
        peak_x = (peak_indices / (buffer_len - 1) * (plot_width - 1)).astype(np.int32)
        valid_peak_indices = np.clip(peak_indices, 0, buffer_len - 1)
        peak_y = normalized_signal[valid_peak_indices].astype(np.int32)
        for px, py in zip(peak_x, peak_y):
             px_clamped, py_clamped = max(0, min(plot_width - 1, px)), max(0, min(plot_height - 1, py))
             cv2.circle(plot_img, (px_clamped, py_clamped), 4, peak_color, -1)

    center_y = plot_height // 2
    cv2.line(plot_img, (0, center_y), (plot_width - 1, center_y), (200, 200, 200), 1)
    return plot_img

def draw_frequency_plot(title, signal_buffer, sampling_rate, plot_width, plot_height, bg_color, line_color, max_freq_to_show):
    """Calculates FFT and draws the frequency spectrum onto a NumPy array."""
    plot_img = np.full((plot_height, plot_width, 3), bg_color, dtype=np.uint8)
    buffer_len = len(signal_buffer)
    cv2.putText(plot_img, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if buffer_len < 10 or sampling_rate <= 0: # Need sufficient buffer for meaningful FFT
        cv2.putText(plot_img, "Waiting for buffer/valid rate...", (10, plot_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return plot_img

    try:
        signal_np = np.array(signal_buffer)
        window = np.hanning(buffer_len)
        signal_windowed = signal_np * window
        yf = fft(signal_windowed)
        xf = fftfreq(buffer_len, 1 / sampling_rate)
        positive_mask = xf >= 0
        xf_pos = xf[positive_mask]
        yf_mag = np.abs(yf[positive_mask]) * 2 / buffer_len # Normalize magnitude
        freq_range_mask = xf_pos <= max_freq_to_show
        xf_plot = xf_pos[freq_range_mask]
        yf_plot = yf_mag[freq_range_mask]

        if len(xf_plot) < 2: # Need points to plot
             cv2.putText(plot_img, "Not enough FFT data in range", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
             return plot_img

        min_val, max_val = 0, np.max(yf_plot) # Start y-axis at 0
        range_val = max_val - min_val
        padding = 0.1 * plot_height
        if range_val < 1e-6:
            normalized_fft = np.full(len(yf_plot), plot_height - padding) # Draw at bottom if flat
        else:
            scale = (plot_height - 2 * padding) / range_val
            normalized_fft = (yf_plot - min_val) * scale + padding
            normalized_fft = plot_height - normalized_fft # Flip vertically
        points = np.zeros((len(xf_plot), 1, 2), dtype=np.int32)
        points[:, 0, 0] = (xf_plot / max_freq_to_show * (plot_width - 1)).astype(np.int32)
        points[:, 0, 1] = normalized_fft.astype(np.int32)
        cv2.polylines(plot_img, [points], isClosed=False, color=line_color, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(plot_img, "0Hz", (5, plot_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        cv2.putText(plot_img, f"{max_freq_to_show:.1f}Hz", (plot_width - 45, plot_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    except Exception as e_fft:
        print(f"Error during FFT calculation/plotting: {e_fft}")
        cv2.putText(plot_img, "FFT Error", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return plot_img


# --- Initialize Components for Calibration ---
# (Initialization code remains the same)
pose_detector = None; coarse_roi_calculator = None
try:
    pose_detector = PoseDetector(config=pose_config)
    coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_config)
except Exception as e: print(f"Error during component initialization: {e}"); traceback.print_exc(); sys.exit(1)

# --- MediaPipe Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles; mp_pose = mp.solutions.pose

# --- OpenCV Video Capture Initialization ---
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Cannot open webcam."); sys.exit(1)

print("Webcam opened. Starting calibration phase...")
print(f"Will calibrate for ~{CALIBRATION_DURATION_SEC} seconds to find ROI.") # Reference updated duration
print("Press 'q' to quit during calibration.")

# --- Calibration Phase ---
# (Calibration loop remains the same)
calibration_start_time = time.time(); static_roi_list = []; calibration_successful = False; calib_frame_count = 0
landmarks_detected_in_calib = False # Flag to track if landmarks were seen
while time.time() - calibration_start_time < CALIBRATION_DURATION_SEC:
    success, frame = cap.read();
    if not success: continue
    calib_frame_count += 1; frame_height, frame_width = frame.shape[:2]; display_frame = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
    landmarks = pose_detector.process_frame(image_rgb); current_rois = []
    if landmarks:
        landmarks_detected_in_calib = True # Mark that we saw landmarks
        current_rois = coarse_roi_calculator.calculate_coarse_roi(landmarks, (frame_height, frame_width))
        mp_drawing.draw_landmarks(display_frame, landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # print(f"Calibration Frame {calib_frame_count}: Landmarks DETECTED. Coarse ROI Result: {current_rois}") # Optional debug
    # else: # Optional debug
    #     if calib_frame_count % 30 == 0: print(f"Calibration Frame {calib_frame_count}: Landmarks NOT detected.")
    if current_rois: # Found a valid ROI
        static_roi_list = current_rois; calibration_successful = True
        for (x, y, w, h) in static_roi_list: cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # Cyan
        cv2.putText(display_frame, "ROI FOUND!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "CALIBRATING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('LK Parameter Tuning - Calibration', display_frame); cv2.waitKey(500); break
    else: # No ROI found yet this frame
        cv2.putText(display_frame, "CALIBRATING... Looking for ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if landmarks_detected_in_calib: # Add message if pose was seen but ROI failed
             cv2.putText(display_frame, "(Pose OK, ROI Calc Failed)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.imshow('LK Parameter Tuning - Calibration', display_frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): print("Exit key pressed during calibration."); calibration_successful = False; break # Exit calibration loop
# --- Cleanup AFTER Calibration Loop ---
if pose_detector: pose_detector.close(); print("[PoseDetector] Closed after calibration.")
if cv2.getWindowProperty('LK Parameter Tuning - Calibration', cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow('LK Parameter Tuning - Calibration')
# --- Check Calibration Outcome ---
if not calibration_successful:
    print(f"Calibration failed or was quit.")
    if not landmarks_detected_in_calib and calib_frame_count > 0: print("   -> Reason: Pose landmarks were not consistently detected.")
    elif landmarks_detected_in_calib and not static_roi_list: print("   -> Reason: Pose landmarks detected, but CoarseRoiCalculator failed (check landmark visibility/positioning).")
    print("Exiting."); cap.release(); cv2.destroyAllWindows(); sys.exit()

# --- Rest of the script (Tuning Loop, Cleanup) remains the same ---
print(f"Calibration finished. Using static ROI: {static_roi_list}")
print("Starting LK parameter tuning loop.")
print("Press 'n' for next parameter set, 'p' for previous, 'q' to quit.")

# Initialize Signal Components
try:
    signal_generator = SignalGenerator(config=signal_config)
    signal_processor = SignalProcessor(config=signal_config, sampling_rate=DEFAULT_SAMPLING_RATE)
except Exception as e_sig_init: print(f"ERROR initializing signal components: {e_sig_init}"); cap.release(); sys.exit(1)

# Tuning Loop
prev_time = time.time(); frame_count = 0; current_param_index = 0; feature_tracker = None
while cap.isOpened():
    # --- Get current parameter set ---
    current_params_config = copy.deepcopy(parameter_sets[current_param_index])
    param_set_name = current_params_config.get('name', f'Set {current_param_index}')

    # --- Initialize Feature Tracker if needed ---
    if feature_tracker is None:
        print(f"\n--- Initializing Tracker for: {param_set_name} ---")
        try:
            tracker_config = { # Construct config for FeatureTracker
                'OPTICAL_FLOW_PARAMS': {
                    'feature_params': current_params_config['feature_params'],
                    'lk_params': current_params_config['lk_params']
                },
                'FEATURE_REDETECT_THRESHOLD': int(current_params_config['feature_params'].get('maxCorners', 100) * 0.7)
            }
            feature_tracker = FeatureTracker(config=tracker_config)
            # Reset signal processor buffers when tracker changes
            signal_processor = SignalProcessor(config=signal_config, sampling_rate=DEFAULT_SAMPLING_RATE)
            print("   (Signal processor buffers reset)")
        except Exception as e_init:
             print(f"ERROR initializing FeatureTracker for set {current_param_index}: {e_init}")
             current_param_index = (current_param_index + 1) % len(parameter_sets)
             feature_tracker = None; time.sleep(0.5); continue

    # --- Read Frame ---
    success, frame = cap.read()
    if not success: print("Ignoring empty camera frame."); continue
    frame_count += 1; frame_height, frame_width = frame.shape[:2]

    # --- Performance calculation (FPS) ---
    curr_time = time.time(); fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0; prev_time = curr_time

    # --- Run Tracking and Signal Pipeline ---
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_data = []; raw_signals = []
    try:
        tracked_data = feature_tracker.process_frame(image_gray, static_roi_list)
        if tracked_data: raw_signals = signal_generator.process_tracked_features(tracked_data)
        else: raw_signals = [0.0] * len(static_roi_list)
        signal_processor.process_signal_values(raw_signals)
    except Exception as e_pipe: print(f"Error during tracking/signal processing: {e_pipe}"); traceback.print_exc()

    # --- Gather Results for Display ---
    bpm, bpm_valid = signal_processor.get_bpm()
    phase = signal_processor.get_phase()
    raw_signal = signal_processor.get_raw_signal_buffer() # Get raw buffer
    filtered_signal = signal_processor.get_filtered_signal_buffer()
    peak_indices = signal_processor.get_last_peak_indices()
    tracked_points_current = feature_tracker.prev_features_per_roi.get(0) # Assume ROI 0

    # --- Prepare images for drawing ---
    display_frame = frame.copy()
    # Initialize plot images to default background
    raw_time_plot = np.full((PLOT_HEIGHT, frame_width, 3), PLOT_BG_COLOR, dtype=np.uint8)
    filtered_time_plot = np.full((PLOT_HEIGHT, frame_width, 3), PLOT_BG_COLOR, dtype=np.uint8)
    frequency_plot = np.full((PLOT_HEIGHT, frame_width, 3), PLOT_BG_COLOR, dtype=np.uint8)

    # Generate the plots if data is available
    if frame_width > 0:
        raw_time_plot = draw_signal_plot(f"Raw Signal (Set {current_param_index})", raw_signal, None, frame_width, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_RAW_LINE_COLOR)
        filtered_time_plot = draw_signal_plot(f"Filtered Signal (Set {current_param_index})", filtered_signal, peak_indices, frame_width, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_FILT_LINE_COLOR, PLOT_PEAK_COLOR)
        frequency_plot = draw_frequency_plot(f"Raw Signal Spectrum (Set {current_param_index})", raw_signal, DEFAULT_SAMPLING_RATE, frame_width, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_FREQ_LINE_COLOR, FREQ_PLOT_MAX_HZ)

    # --- Draw Overlays on Webcam Frame ---
    roi_color = (255, 0, 0); point_color = (0, 0, 255); feature_count = 0
    for i, (x, y, w, h) in enumerate(static_roi_list): cv2.rectangle(display_frame, (x, y), (x + w, y + h), roi_color, 2)
    if tracked_points_current is not None:
        feature_count = len(tracked_points_current)
        # --- FIXED SYNTAX FOR DRAWING POINTS ---
        for point in tracked_points_current:
            try:
                # Ensure point has the expected structure (1, 2)
                if point.shape == (1, 2):
                    x, y = int(point[0, 0]), int(point[0, 1]) # Access elements correctly
                    cv2.circle(display_frame, (x, y), 3, point_color, -1)
                # else: print(f"DEBUG: Skipping malformed point shape: {point.shape}") # Optional debug
            except IndexError:
                 # print(f"DEBUG: IndexError accessing point: {point}") # Optional debug
                 pass # Ignore points that cause index errors
            except Exception as draw_err:
                 print(f"ERROR drawing point {point}: {draw_err}") # Log other errors
        # --- END FIXED SYNTAX ---
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Param Set {current_param_index}: {param_set_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    fp = current_params_config['feature_params']; lk = current_params_config['lk_params']
    param_text1 = f"Feat: N={fp['maxCorners']} Q={fp['qualityLevel']} D={fp['minDistance']}"; param_text2 = f"LK: Win={lk['winSize']} Lvl={lk['maxLevel']} Crit={lk['criteria'][1]},{lk['criteria'][2]}"
    cv2.putText(display_frame, param_text1, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(display_frame, param_text2, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Tracked Features: {feature_count}", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    bpm_text = f"BPM: {bpm:.1f}" if bpm_valid else "BPM: ---"; cv2.putText(display_frame, bpm_text, (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    phase_map = {signal_processor.PHASE_INHALE: "In", signal_processor.PHASE_EXHALE: "Ex", signal_processor.PHASE_UNKNOWN: "--"}; phase_text = f"P: {phase_map.get(phase, 'Err')}"
    cv2.putText(display_frame, phase_text, (frame_width - 150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # --- Combine Webcam Frame and Plots ---
    if raw_time_plot.shape[1] != display_frame.shape[1]: raw_time_plot = cv2.resize(raw_time_plot, (display_frame.shape[1], PLOT_HEIGHT))
    if filtered_time_plot.shape[1] != display_frame.shape[1]: filtered_time_plot = cv2.resize(filtered_time_plot, (display_frame.shape[1], PLOT_HEIGHT))
    if frequency_plot.shape[1] != display_frame.shape[1]: frequency_plot = cv2.resize(frequency_plot, (display_frame.shape[1], PLOT_HEIGHT))
    combined_display = np.vstack((display_frame, raw_time_plot, filtered_time_plot, frequency_plot))

    # --- Display the frame ---
    cv2.imshow('LK Parameter Tuning (with Plots)', combined_display) # Updated title

    # --- User Input to Cycle Parameters ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): print("Exit key pressed."); break
    elif key == ord('n'): current_param_index = (current_param_index + 1) % len(parameter_sets); print(f"\nSwitching to parameter set {current_param_index}"); feature_tracker = None
    elif key == ord('p'): current_param_index = (current_param_index - 1 + len(parameter_sets)) % len(parameter_sets); print(f"\nSwitching to parameter set {current_param_index}"); feature_tracker = None

# --- Cleanup ---
print("Releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Finished.")
