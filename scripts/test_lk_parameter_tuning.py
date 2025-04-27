# scripts/test_lk_parameter_tuning.py
# Helps tune FeatureTracker parameters using interactive OpenCV sliders (trackbars)
# and visualizing tracked points, raw signal, and mean displacement envelope.
# MODIFIED: Added Mean Displacement Envelope plot, removed motion viz window.

import cv2
import mediapipe as mp
import time
import numpy as np
import os
import sys
import traceback
import copy # To deep copy parameter dictionaries
import collections # For signal processor buffers and trace history
from scipy.fft import fft, fftfreq # For frequency plot (function kept, plot removed)
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
    from video_input import VideoInput
    from pose_detector import PoseDetector
    from coarse_roi_calculator import CoarseRoiCalculator
    from feature_tracker import FeatureTracker
    from signal_generator import SignalGenerator
    from signal_processor import SignalProcessor
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure required modules exist (VideoInput, CoarseRoiCalculator, etc.).")
    sys.exit(1)

print("Initializing Webcam (via VideoInput) and Pose/ROI Components for Calibration...")

# --- Load Configuration ---
PROFILE_FILENAME = "test_profile.json"
config_path = os.path.join(project_root, "profiles", PROFILE_FILENAME)
config = {}
try:
    with open(config_path, 'r') as f: config = json.load(f)
    print(f"Loaded configuration from: {config_path}")
except Exception as e: print(f"Warning: Failed to load config '{config_path}': {e}. Using defaults."); config = {}

# Extract specific config sections or use defaults
video_config = config.get("video_input", {})
pose_config = config.get("pose_detector", {})
coarse_roi_config = config.get("coarse_roi_calculator", {})
signal_config = config.get("signal_processor", {})
signal_config.update(config.get("signal_generator", {}))
initial_ft_config = config.get("feature_tracker", {})
initial_of_params = initial_ft_config.get("OPTICAL_FLOW_PARAMS", {})
initial_feat_params = initial_of_params.get("feature_params", {})
initial_lk_params = initial_of_params.get("lk_params", {})

CALIBRATION_DURATION_SEC = 5

# Plotting Config
PLOT_HEIGHT = 120 # Slightly increased plot height
PLOT_BG_COLOR = (240, 240, 240)
PLOT_RAW_LINE_COLOR = (100, 100, 100)
PLOT_MEAN_LINE_COLOR = (200, 0, 0) # Color for mean displacement line
PLOT_STD_FILL_COLOR = (200, 150, 150) # Color for std dev fill
# Font Sizes & Thickness
FONT_SCALE_INFO = 0.7
FONT_SCALE_PARAMS = 0.5
FONT_SCALE_PLOT_TITLE = 0.5
FONT_SCALE_PLOT_AXIS = 0.4 # Still used in freq plot function if called
FONT_THICKNESS = 2

# Window Names & Display Size
WINDOW_NAME = 'LK Parameter Tuning (Plots)' # Updated main window name
WEBCAM_DISPLAY_WIDTH = 960
# Removed Motion Viz constants

DEFAULT_SAMPLING_RATE = 30.0 # Estimate
MEAN_DISP_HISTORY_SECONDS = 10.0 # Match signal processor buffer for plot length


# === PARAMETER SETS TO TEST ===
# (Parameter sets list remains the same but is not used with sliders)
lk_criteria_default = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
parameter_sets = [ # Kept for reference if needed later
    { 'name': 'Default-ish', 'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}, 'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': lk_criteria_default}},
    # ... (other sets omitted for brevity but should be kept) ...
    { 'name': 'Stricter LK Criteria', 'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7}, 'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)}}
]


# --- Plotting Helper Functions ---
# draw_signal_plot remains unchanged
def draw_signal_plot(title, signal_buffer, peak_indices, plot_width, plot_height, bg_color, line_color, peak_color=None):
    plot_img = np.full((plot_height, plot_width, 3), bg_color, dtype=np.uint8)
    buffer_len = len(signal_buffer); cv2.putText(plot_img, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_PLOT_TITLE, (0, 0, 0), 1)
    if buffer_len < 2: cv2.putText(plot_img, "Waiting for buffer...", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1); return plot_img
    signal_np = np.array(signal_buffer); min_val, max_val = np.min(signal_np), np.max(signal_np); range_val = max_val - min_val; padding = 0.1 * plot_height
    if range_val < 1e-6: center_y_flat = plot_height // 2; cv2.line(plot_img, (0, center_y_flat), (plot_width -1, center_y_flat), line_color, 1); normalized_signal = np.full(buffer_len, center_y_flat)
    else:
        scale = (plot_height - 2 * padding) / range_val; normalized_signal = (signal_np - min_val) * scale + padding; normalized_signal = plot_height - normalized_signal
        points = np.zeros((buffer_len, 1, 2), dtype=np.int32); points[:, 0, 0] = np.linspace(0, plot_width - 1, buffer_len, dtype=np.int32); points[:, 0, 1] = normalized_signal.astype(np.int32)
        cv2.polylines(plot_img, [points], isClosed=False, color=line_color, thickness=1, lineType=cv2.LINE_AA)
    if peak_indices is not None and peak_color is not None and len(peak_indices) > 0 and buffer_len > 1:
        peak_x = (peak_indices / (buffer_len - 1) * (plot_width - 1)).astype(np.int32); valid_peak_indices = np.clip(peak_indices, 0, buffer_len - 1); peak_y = normalized_signal[valid_peak_indices].astype(np.int32)
        for px, py in zip(peak_x, peak_y): px_clamped, py_clamped = max(0, min(plot_width - 1, px)), max(0, min(plot_height - 1, py)); cv2.circle(plot_img, (px_clamped, py_clamped), 4, peak_color, -1)
    center_y = plot_height // 2; cv2.line(plot_img, (0, center_y), (plot_width - 1, center_y), (200, 200, 200), 1); return plot_img

# draw_frequency_plot is no longer called, but kept here in case needed later
# def draw_frequency_plot(...): ...

# --- New Function: Draw Mean Displacement Envelope ---
def draw_mean_displacement_envelope(title, mean_dy_hist, std_dy_hist, plot_width, plot_height, bg_color, mean_color, std_color):
    """Draws the mean vertical displacement and its standard deviation envelope."""
    plot_img = np.full((plot_height, plot_width, 3), bg_color, dtype=np.uint8)
    buffer_len = len(mean_dy_hist)
    cv2.putText(plot_img, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_PLOT_TITLE, (0, 0, 0), 1)

    if buffer_len < 2 or len(std_dy_hist) != buffer_len:
        cv2.putText(plot_img, "Waiting for buffer...", (10, plot_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return plot_img

    mean_dy = np.array(mean_dy_hist)
    std_dy = np.array(std_dy_hist)

    # Calculate upper and lower bounds for the envelope
    upper_bound = mean_dy + std_dy
    lower_bound = mean_dy - std_dy

    # Find overall min/max across mean and bounds for normalization
    # Avoid errors if bounds arrays are empty (shouldn't happen if buffer_len >= 2)
    if lower_bound.size == 0 or upper_bound.size == 0:
         cv2.putText(plot_img, "Calc Error", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1); return plot_img
    min_val = np.min(lower_bound)
    max_val = np.max(upper_bound)
    range_val = max_val - min_val
    padding = 0.1 * plot_height

    # Function to normalize y-values
    def normalize_y(y_values):
        if range_val < 1e-6:
            # If range is tiny, plot everything near the middle
            return np.full(len(y_values), plot_height / 2)
        else:
            scale = (plot_height - 2 * padding) / range_val
            normalized = (y_values - min_val) * scale + padding
            return plot_height - normalized # Flip vertically

    # Normalize mean, upper, and lower bounds
    norm_mean = normalize_y(mean_dy).astype(np.int32)
    norm_upper = normalize_y(upper_bound).astype(np.int32)
    norm_lower = normalize_y(lower_bound).astype(np.int32)

    # Generate x coordinates
    x_coords = np.linspace(0, plot_width - 1, buffer_len, dtype=np.int32)

    # Create points for the fill polygon (upper bound then reversed lower bound)
    # Ensure x_coords matches length of bounds after potential issues
    if len(x_coords) != len(norm_upper) or len(x_coords) != len(norm_lower):
         cv2.putText(plot_img, "Coord Mismatch", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1); return plot_img

    fill_pts = np.vstack((np.column_stack((x_coords, norm_upper)),
                          np.column_stack((x_coords[::-1], norm_lower[::-1]))))

    # Draw the filled standard deviation envelope
    cv2.fillPoly(plot_img, [fill_pts], std_color, lineType=cv2.LINE_AA)

    # Create points for the mean line
    mean_pts = np.column_stack((x_coords, norm_mean)).reshape(-1, 1, 2)

    # Draw the mean line
    cv2.polylines(plot_img, [mean_pts], isClosed=False, color=mean_color, thickness=1, lineType=cv2.LINE_AA)

    # Optional: Draw center line (representing zero displacement)
    center_y_norm = normalize_y(np.array([0.0]))[0] # Normalize zero displacement
    cv2.line(plot_img, (0, int(center_y_norm)), (plot_width - 1, int(center_y_norm)), (150, 150, 150), 1)

    return plot_img


# --- Initialize Video Input ---
# (Initialization code remains the same)
video_input = None
try:
    video_input = VideoInput(config=video_config)
    if not video_input.initialized: raise RuntimeError("VideoInput failed to initialize.")
    actual_fps = video_input.get_fps(); sampling_rate = actual_fps if actual_fps > 0 else DEFAULT_SAMPLING_RATE
    print(f"Using sampling rate for signal processing: {sampling_rate:.2f} Hz")
except Exception as e: print(f"Error initializing VideoInput: {e}"); sys.exit(1)

# --- Initialize Components for Calibration ---
# (Initialization code remains the same)
pose_detector = None; coarse_roi_calculator = None
try:
    pose_detector = PoseDetector(config=pose_config)
    coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_config)
except Exception as e: print(f"Error during component initialization: {e}"); traceback.print_exc(); video_input.release(); sys.exit(1)

# --- MediaPipe Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles; mp_pose = mp.solutions.pose

print("Starting calibration phase...")
print(f"Will calibrate for ~{CALIBRATION_DURATION_SEC} seconds to find ROI.")
print("Press 'q' to quit during calibration.")

# --- Calibration Phase ---
# (Calibration loop remains the same)
calibration_start_time = time.time(); static_roi_list = []; calibration_successful = False; calib_frame_count = 0
landmarks_detected_in_calib = False
while time.time() - calibration_start_time < CALIBRATION_DURATION_SEC:
    success, frame = video_input.get_frame();
    if not success: print("Warning: Failed to get frame during calibration."); time.sleep(0.1); continue
    calib_frame_count += 1;
    frame_width, frame_height = video_input.get_resolution()
    if frame_width == 0 or frame_height == 0: print("Warning: Invalid frame dimensions during calibration."); continue
    display_frame = frame.copy()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); image_rgb.flags.writeable = False
    landmarks = pose_detector.process_frame(image_rgb); current_rois = []
    if landmarks:
        landmarks_detected_in_calib = True
        current_rois = coarse_roi_calculator.calculate_coarse_roi(landmarks, (frame_height, frame_width))
        mp_drawing.draw_landmarks(display_frame, landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if current_rois:
        static_roi_list = current_rois; calibration_successful = True
        for (x, y, w, h) in static_roi_list: cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2) # Cyan
        cv2.putText(display_frame, "ROI FOUND!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, "CALIBRATING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('LK Parameter Tuning - Calibration', display_frame); cv2.waitKey(500); break
    else:
        cv2.putText(display_frame, "CALIBRATING... Looking for ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if landmarks_detected_in_calib: cv2.putText(display_frame, "(Pose OK, ROI Calc Failed)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.imshow('LK Parameter Tuning - Calibration', display_frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): print("Exit key pressed during calibration."); calibration_successful = False; break
# --- Cleanup AFTER Calibration Loop ---
if pose_detector: pose_detector.close(); print("[PoseDetector] Closed after calibration.")
if cv2.getWindowProperty('LK Parameter Tuning - Calibration', cv2.WND_PROP_VISIBLE) >= 1: cv2.destroyWindow('LK Parameter Tuning - Calibration')
# --- Check Calibration Outcome ---
if not calibration_successful:
    print(f"Calibration failed or was quit.")
    if not landmarks_detected_in_calib and calib_frame_count > 0: print("   -> Reason: Pose landmarks were not consistently detected.")
    elif landmarks_detected_in_calib and not static_roi_list: print("   -> Reason: Pose landmarks detected, but CoarseRoiCalculator failed (check landmark visibility/positioning).")
    print("Exiting."); video_input.release(); cv2.destroyAllWindows(); sys.exit()

print(f"Calibration finished. Using static ROI: {static_roi_list}")
print("Starting LK parameter tuning loop.")
print("Adjust sliders to tune parameters. Press 's' to SAVE to profile, 'q' to quit.")

# --- Initialize Signal Components ---
try:
    signal_generator = SignalGenerator(config=signal_config)
    signal_processor = SignalProcessor(config=signal_config, sampling_rate=sampling_rate)
except Exception as e_sig_init: print(f"ERROR initializing signal components: {e_sig_init}"); video_input.release(); sys.exit(1)

# --- Create Window and Trackbars ---
cv2.namedWindow(WINDOW_NAME)
def nothing(x): pass # Dummy callback
cv2.createTrackbar('maxCorners', WINDOW_NAME, initial_feat_params.get('maxCorners', 100), 300, nothing)
cv2.createTrackbar('qualityLevel', WINDOW_NAME, int(initial_feat_params.get('qualityLevel', 0.3) * 100), 99, nothing)
cv2.createTrackbar('minDistance', WINDOW_NAME, initial_feat_params.get('minDistance', 7), 50, nothing)
win_size_init = initial_lk_params.get('winSize', [15, 15]); win_size_val = win_size_init[0] if isinstance(win_size_init, (list, tuple)) and len(win_size_init) > 0 else 15
cv2.createTrackbar('winSize', WINDOW_NAME, win_size_val, 51, nothing)
cv2.createTrackbar('maxLevel', WINDOW_NAME, initial_lk_params.get('maxLevel', 2), 8, nothing)
# Removed Exaggeration and Trace Decay sliders

# --- Tuning Loop ---
prev_time = time.time(); frame_count = 0; feature_tracker = None
prev_params = {} # Store previous slider values to detect changes
last_saved_params = {} # Store last saved params to avoid redundant saves
latest_tracked_old = None # Store points for calculation
latest_tracked_new = None # Store points for calculation
# --- Deques for Mean Displacement History ---
mean_disp_history_len = int(MEAN_DISP_HISTORY_SECONDS * sampling_rate)
mean_dy_history = collections.deque(maxlen=mean_disp_history_len)
std_dy_history = collections.deque(maxlen=mean_disp_history_len)

while True: # Loop until user quits or error
    # --- Read Frame using VideoInput ---
    success, frame = video_input.get_frame()
    if not success or frame is None: print("End of video source or cannot read frame. Exiting."); break
    frame_count += 1;
    frame_width, frame_height = video_input.get_resolution()
    if frame_width == 0 or frame_height == 0: print("Error: Invalid frame dimensions received from VideoInput. Exiting."); break

    # --- Read Trackbar Values ---
    current_params = {
        'maxCorners': max(10, cv2.getTrackbarPos('maxCorners', WINDOW_NAME)),
        'qualityLevel': max(1, cv2.getTrackbarPos('qualityLevel', WINDOW_NAME)) / 100.0,
        'minDistance': max(1, cv2.getTrackbarPos('minDistance', WINDOW_NAME)),
        'winSize_raw': cv2.getTrackbarPos('winSize', WINDOW_NAME),
        'maxLevel': cv2.getTrackbarPos('maxLevel', WINDOW_NAME),
        # Exaggeration removed
    }
    win_size = max(3, current_params['winSize_raw'] // 2 * 2 + 1)
    current_params['winSize'] = (win_size, win_size) # Store as tuple for internal use

    # --- Check if LK parameters changed ---
    params_changed = False
    compare_keys = ['maxCorners', 'qualityLevel', 'minDistance', 'winSize_raw', 'maxLevel']
    if not prev_params: params_changed = True
    else:
        for key in compare_keys:
            if current_params[key] != prev_params.get(key): params_changed = True; break

    # --- Initialize/Re-initialize Feature Tracker if needed ---
    if feature_tracker is None or params_changed:
        if params_changed: print(f"\n--- Parameters Changed! Re-initializing Tracker ---")
        else: print(f"\n--- Initializing Tracker ---")
        try:
            # (Tracker config construction remains the same)
            feature_params_dict = {'maxCorners': current_params['maxCorners'],'qualityLevel': current_params['qualityLevel'],'minDistance': current_params['minDistance'],'blockSize': 7}
            lk_params_dict = {'winSize': current_params['winSize'],'maxLevel': current_params['maxLevel'],'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
            tracker_config = {'OPTICAL_FLOW_PARAMS': {'feature_params': feature_params_dict,'lk_params': lk_params_dict},'FEATURE_REDETECT_THRESHOLD': int(feature_params_dict.get('maxCorners', 100) * 0.7)}
            feature_tracker = FeatureTracker(config=tracker_config)
            # Reset signal processor buffers AND mean displacement history
            signal_processor = SignalProcessor(config=signal_config, sampling_rate=sampling_rate)
            mean_dy_history.clear()
            std_dy_history.clear()
            print("   (Signal processor & displacement history buffers reset)")
            # Store only the relevant params for change detection
            prev_params = {k: current_params[k] for k in compare_keys}
            prev_params['winSize'] = current_params['winSize'] # Store processed tuple
            latest_tracked_old = None
            latest_tracked_new = None

        except Exception as e_init:
             print(f"ERROR initializing FeatureTracker: {e_init}"); feature_tracker = None; time.sleep(0.5); continue

    # --- Performance calculation (FPS) ---
    curr_time = time.time(); fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0; prev_time = curr_time

    # --- Run Tracking and Signal Pipeline ---
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_data = []; raw_signals = []
    latest_tracked_old = None # Reset for this frame
    latest_tracked_new = None
    current_mean_dy = 0.0 # Default value
    current_std_dy = 0.0  # Default value

    if feature_tracker:
        try:
            tracked_data = feature_tracker.process_frame(image_gray, static_roi_list)
            # Store points for displacement calculation
            if tracked_data and tracked_data[0] is not None and tracked_data[0][0] is not None:
                 latest_tracked_old = tracked_data[0][0].copy()
                 latest_tracked_new = tracked_data[0][1].copy()

                 # --- Calculate Mean/Std Vertical Displacement ---
                 if len(latest_tracked_old) > 0:
                      # Reshape just in case points are (N, 1, 2)
                      old_pts_flat = latest_tracked_old.reshape(-1, 2)
                      new_pts_flat = latest_tracked_new.reshape(-1, 2)
                      displacements = new_pts_flat - old_pts_flat
                      dy_values = displacements[:, 1] # Get vertical components
                      current_mean_dy = np.mean(dy_values)
                      current_std_dy = np.std(dy_values)
                 # --- End Calculation ---

            if tracked_data: raw_signals = signal_generator.process_tracked_features(tracked_data)
            else: raw_signals = [0.0] * len(static_roi_list)
            signal_processor.process_signal_values(raw_signals)
        except Exception as e_pipe: print(f"Error during tracking/signal processing: {e_pipe}"); traceback.print_exc()
    else: signal_processor.process_signal_values([0.0] * len(static_roi_list))

    # Append current mean/std to history
    mean_dy_history.append(current_mean_dy)
    std_dy_history.append(current_std_dy)

    # --- Gather Results for Display ---
    bpm, bpm_valid = signal_processor.get_bpm()
    phase = signal_processor.get_phase()
    raw_signal_history = signal_processor.get_raw_signal_buffer() # Renamed for clarity
    tracked_points_current = feature_tracker.features_to_track_per_roi.get(0) if feature_tracker else None

    # --- Prepare images for drawing ---
    # --- Resize webcam feed FIRST ---
    try:
        aspect_ratio = frame_height / frame_width if frame_width > 0 else 1
        webcam_display_height = int(WEBCAM_DISPLAY_WIDTH * aspect_ratio)
        display_frame = cv2.resize(frame, (WEBCAM_DISPLAY_WIDTH, webcam_display_height), interpolation=cv2.INTER_AREA)
        display_w = display_frame.shape[1]
    except Exception as e_resize: print(f"Error resizing display frame: {e_resize}"); display_frame = frame.copy(); display_w = frame_width

    # Initialize plot images
    raw_time_plot = np.full((PLOT_HEIGHT, display_w, 3), PLOT_BG_COLOR, dtype=np.uint8)
    mean_disp_plot = np.full((PLOT_HEIGHT, display_w, 3), PLOT_BG_COLOR, dtype=np.uint8)

    # Generate the plots
    if display_w > 0:
        raw_time_plot = draw_signal_plot(f"Raw Signal (PCA)", raw_signal_history, None, display_w, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_RAW_LINE_COLOR)
        mean_disp_plot = draw_mean_displacement_envelope(
            "Mean Vertical Displacement +/- StdDev", mean_dy_history, std_dy_history,
            display_w, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_MEAN_LINE_COLOR, PLOT_STD_FILL_COLOR
        )

    # --- Draw Overlays on the RESIZED Webcam Frame ---
    # (Drawing code remains the same, uses scaled coordinates)
    scale_x = display_w / frame_width if frame_width > 0 else 1; scale_y = display_frame.shape[0] / frame_height if frame_height > 0 else 1
    roi_color = (255, 0, 0); point_color = (0, 0, 255); feature_count = 0
    for i, (x, y, w, h) in enumerate(static_roi_list): sx, sy = int(x * scale_x), int(y * scale_y); sw, sh = int(w * scale_x), int(h * scale_y); cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh), roi_color, 2)
    if tracked_points_current is not None:
        feature_count = len(tracked_points_current)
        for point in tracked_points_current:
            try:
                if point.shape == (1, 2): x, y = int(point[0, 0] * scale_x), int(point[0, 1] * scale_y); cv2.circle(display_frame, (x, y), 3, point_color, -1)
            except IndexError: pass
            except Exception as draw_err: print(f"ERROR drawing point {point}: {draw_err}")
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (0, 255, 0), FONT_THICKNESS)
    param_text1 = f"Feat: N={current_params['maxCorners']} Q={current_params['qualityLevel']:.2f} D={current_params['minDistance']}"
    param_text2 = f"LK: Win={current_params['winSize']} Lvl={current_params['maxLevel']}"
    # Removed Exaggeration text
    cv2.putText(display_frame, param_text1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_PARAMS, (255, 255, 255), 1)
    cv2.putText(display_frame, param_text2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_PARAMS, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Tracked Features: {feature_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (255, 0, 0), FONT_THICKNESS) # Adjusted y-pos
    bpm_text = f"BPM: {bpm:.1f}" if bpm_valid else "BPM: ---"; cv2.putText(display_frame, bpm_text, (display_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (0, 255, 255), FONT_THICKNESS)
    phase_map = {signal_processor.PHASE_INHALE: "In", signal_processor.PHASE_EXHALE: "Ex", signal_processor.PHASE_UNKNOWN: "--"}; phase_text = f"P: {phase_map.get(phase, 'Err')}"
    cv2.putText(display_frame, phase_text, (display_w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (0, 255, 255), FONT_THICKNESS)

    # --- Combine Resized Webcam Frame and Plots ---
    if raw_time_plot.shape[1] != display_w: raw_time_plot = cv2.resize(raw_time_plot, (display_w, PLOT_HEIGHT))
    if mean_disp_plot.shape[1] != display_w: mean_disp_plot = cv2.resize(mean_disp_plot, (display_w, PLOT_HEIGHT))
    # Stack: Webcam, Raw Signal (PCA), Mean Displacement Envelope
    combined_display = np.vstack((display_frame, raw_time_plot, mean_disp_plot))

    # --- Display the main frame ---
    cv2.imshow(WINDOW_NAME, combined_display)

    # --- Remove Motion Vector Visualization Window ---
    # Check if the window exists and destroy it if it does
    # Use the old name here just in case it was left open from a previous run
    if cv2.getWindowProperty('Velocity Endpoint Trace', cv2.WND_PROP_VISIBLE) >= 1:
        try: cv2.destroyWindow('Velocity Endpoint Trace')
        except: pass # Ignore error if window already closed
    if cv2.getWindowProperty('Displacement Vector Traces', cv2.WND_PROP_VISIBLE) >= 1:
         try: cv2.destroyWindow('Displacement Vector Traces')
         except: pass


    # --- User Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): print("Exit key pressed."); break
    elif key == ord('s'): # Save parameters
        # (Save logic remains the same)
        print("\n--- Saving current parameters ---")
        try:
            feat_params_to_save = {'maxCorners': current_params['maxCorners'],'qualityLevel': current_params['qualityLevel'],'minDistance': current_params['minDistance'],'blockSize': 7}
            lk_params_to_save = {'winSize': list(current_params['winSize']),'maxLevel': current_params['maxLevel'],'criteria': [3, 10, 0.03]}
            redetect_thresh = int(feat_params_to_save['maxCorners'] * 0.7)
            current_full_config = {};
            try:
                with open(config_path, 'r') as f_read: current_full_config = json.load(f_read)
            except FileNotFoundError: print(f"Warning: Profile '{PROFILE_FILENAME}' not found for loading, creating new structure.")
            except Exception as e_load: print(f"Warning: Error loading profile '{PROFILE_FILENAME}' for saving: {e_load}. Saving may overwrite structure.")
            if "feature_tracker" not in current_full_config: current_full_config["feature_tracker"] = {}
            if "OPTICAL_FLOW_PARAMS" not in current_full_config["feature_tracker"]: current_full_config["feature_tracker"]["OPTICAL_FLOW_PARAMS"] = {}
            current_full_config["feature_tracker"]["OPTICAL_FLOW_PARAMS"]["feature_params"] = feat_params_to_save
            current_full_config["feature_tracker"]["OPTICAL_FLOW_PARAMS"]["lk_params"] = lk_params_to_save
            current_full_config["feature_tracker"]["FEATURE_REDETECT_THRESHOLD"] = redetect_thresh
            with open(config_path, 'w') as f_write: json.dump(current_full_config, f_write, indent=2)
            print(f"Parameters saved to: {config_path}")
            last_saved_params = current_params.copy()
        except Exception as e_save: print(f"ERROR saving parameters to {config_path}: {e_save}"); traceback.print_exc()

# --- Cleanup ---
print("Releasing resources...")
if video_input: video_input.release() # Use VideoInput release method
cv2.destroyAllWindows()
print("Finished.")
