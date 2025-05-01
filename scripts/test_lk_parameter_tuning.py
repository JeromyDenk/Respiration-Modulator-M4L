# scripts/test_lk_parameter_tuning.py
# Helps tune FeatureTracker and SignalProcessor parameters using interactive OpenCV sliders
# and visualizing tracked points, raw/filtered signals, and mean displacement envelope.
# MODIFIED: Separated Signal Processor trackbars into their own window to prevent cramping.
# FIXED: NameError when reading signal trackbar values.
# FIXED: AttributeError by using the correct output from feature_tracker.process_frame for display.

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
    # Assuming VideoInput class exists in src/video_input.py
    # If not, replace with direct cv2.VideoCapture or adjust path
    try:
        from video_input import VideoInput
    except ImportError:
        print("Warning: video_input.py not found. Using direct cv2.VideoCapture.")
        VideoInput = None # Fallback indicator

    from pose_detector import PoseDetector
    from coarse_roi_calculator import CoarseRoiCalculator # Make sure class name matches file
    from feature_tracker import FeatureTracker
    from signal_generator import SignalGenerator
    from signal_processor import SignalProcessor
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure required modules exist (VideoInput, CoarseRoiCalculator, etc.).")
    sys.exit(1)

print("Initializing Webcam and Pose/ROI Components for Calibration...")

# --- Load Configuration ---
PROFILE_FILENAME = "test_profile.json" # Make sure this exists
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
signal_config_base = config.get("signal_processor", {})
signal_config_base.update(config.get("signal_generator", {})) # Merge generator config if needed
initial_ft_config = config.get("feature_tracker", {})
initial_of_params = initial_ft_config.get("OPTICAL_FLOW_PARAMS", {})
initial_feat_params = initial_of_params.get("feature_params", {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7}) # Add defaults
initial_lk_params = initial_of_params.get("lk_params", {'winSize': [15, 15], 'maxLevel': 2}) # Add defaults

CALIBRATION_DURATION_SEC = 5

# Plotting Config
PLOT_HEIGHT = 100
PLOT_BG_COLOR = (240, 240, 240)
PLOT_RAW_LINE_COLOR = (100, 100, 100)
PLOT_FILT_LINE_COLOR = (0, 0, 200)
PLOT_PEAK_COLOR = (0, 0, 255)
PLOT_MEAN_LINE_COLOR = (200, 0, 0)
PLOT_STD_FILL_COLOR = (200, 150, 150)
FONT_SCALE_INFO = 0.7
FONT_SCALE_PARAMS = 0.5
FONT_SCALE_PLOT_TITLE = 0.5
FONT_SCALE_PLOT_AXIS = 0.4
FONT_THICKNESS = 1 # Use thickness 1 for smaller fonts

# --- Window Names & Display Size ---
LK_WINDOW_NAME = 'LK Tuning (Video & LK Params)' # Main window for LK
SIGNAL_WINDOW_NAME = 'Signal Tuning (Params & Plots)' # Separate window for signal params/plots
WEBCAM_DISPLAY_WIDTH = 960

DEFAULT_SAMPLING_RATE = 30.0 # Estimate
MEAN_DISP_HISTORY_SECONDS = 10.0


# --- Plotting Helper Functions ---
# (draw_signal_plot and draw_mean_displacement_envelope functions remain the same)
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

def draw_mean_displacement_envelope(title, mean_dy_hist, std_dy_hist, plot_width, plot_height, bg_color, mean_color, std_color):
    plot_img = np.full((plot_height, plot_width, 3), bg_color, dtype=np.uint8)
    buffer_len = len(mean_dy_hist); cv2.putText(plot_img, title, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_PLOT_TITLE, (0, 0, 0), 1)
    if buffer_len < 2 or len(std_dy_hist) != buffer_len: cv2.putText(plot_img, "Waiting for buffer...", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1); return plot_img
    mean_dy = np.array(mean_dy_hist); std_dy = np.array(std_dy_hist)
    upper_bound = mean_dy + std_dy; lower_bound = mean_dy - std_dy
    if lower_bound.size == 0 or upper_bound.size == 0: cv2.putText(plot_img, "Calc Error", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1); return plot_img
    min_val = np.min(lower_bound); max_val = np.max(upper_bound); range_val = max_val - min_val; padding = 0.1 * plot_height
    def normalize_y(y_values):
        if range_val < 1e-6: return np.full(len(y_values), plot_height / 2)
        else: scale = (plot_height - 2 * padding) / range_val; normalized = (y_values - min_val) * scale + padding; return plot_height - normalized
    norm_mean = normalize_y(mean_dy).astype(np.int32); norm_upper = normalize_y(upper_bound).astype(np.int32); norm_lower = normalize_y(lower_bound).astype(np.int32)
    x_coords = np.linspace(0, plot_width - 1, buffer_len, dtype=np.int32)
    if len(x_coords) != len(norm_upper) or len(x_coords) != len(norm_lower): cv2.putText(plot_img, "Coord Mismatch", (10, plot_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1); return plot_img
    fill_pts = np.vstack((np.column_stack((x_coords, norm_upper)), np.column_stack((x_coords[::-1], norm_lower[::-1]))))
    cv2.fillPoly(plot_img, [fill_pts], std_color, lineType=cv2.LINE_AA)
    mean_pts = np.column_stack((x_coords, norm_mean)).reshape(-1, 1, 2)
    cv2.polylines(plot_img, [mean_pts], isClosed=False, color=mean_color, thickness=1, lineType=cv2.LINE_AA)
    center_y_norm = normalize_y(np.array([0.0]))[0]
    cv2.line(plot_img, (0, int(center_y_norm)), (plot_width - 1, int(center_y_norm)), (150, 150, 150), 1); return plot_img


# --- Initialize Video Input ---
video_input = None
try:
    if VideoInput: # Use VideoInput class if imported successfully
        video_input = VideoInput(config=video_config)
        if not video_input.initialized: raise RuntimeError("VideoInput failed to initialize.")
    else: # Fallback to direct OpenCV capture
        print("Using direct cv2.VideoCapture(0)")
        video_input = cv2.VideoCapture(0)
        if not video_input.isOpened(): raise RuntimeError("Cannot open webcam with cv2.VideoCapture.")
        # Dummy methods for compatibility if needed later
        video_input.get_frame = lambda: video_input.read()
        video_input.get_fps = lambda: video_input.get(cv2.CAP_PROP_FPS)
        video_input.get_resolution = lambda: (int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_input.release = lambda: video_input.release() # Ensure release method exists

    if VideoInput: # Get FPS only if using the class
        actual_fps = video_input.get_fps()
        sampling_rate = actual_fps if actual_fps > 0 else DEFAULT_SAMPLING_RATE
    else: # Estimate FPS if using direct capture
        sampling_rate = DEFAULT_SAMPLING_RATE
    print(f"Using sampling rate for signal processing: {sampling_rate:.2f} Hz")

except Exception as e: print(f"Error initializing Video Input: {e}"); sys.exit(1)


# --- Initialize Components for Calibration ---
pose_detector = None; coarse_roi_calculator = None
try:
    pose_detector = PoseDetector(config=pose_config)
    coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_config)
except Exception as e: print(f"Error during component initialization: {e}"); traceback.print_exc(); video_input.release(); sys.exit(1)

# --- MediaPipe Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils; mp_drawing_styles = mp.solutions.drawing_styles; mp_pose = mp.solutions.pose

print("Starting calibration phase...")
# --- Calibration Phase ---
# (Calibration loop remains the same)
calibration_start_time = time.time(); static_roi_list = []; calibration_successful = False; calib_frame_count = 0
landmarks_detected_in_calib = False
while time.time() - calibration_start_time < CALIBRATION_DURATION_SEC:
    success, frame = video_input.get_frame();
    if not success or frame is None: print("Warning: Failed to get frame during calibration."); time.sleep(0.1); continue
    calib_frame_count += 1;
    frame_height, frame_width = frame.shape[:2] # Get dims directly from frame
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
    signal_generator = SignalGenerator(config=signal_config_base)
    signal_processor = SignalProcessor(config=signal_config_base, sampling_rate=sampling_rate)
except Exception as e_sig_init: print(f"ERROR initializing signal components: {e_sig_init}"); video_input.release(); sys.exit(1)

# --- Create Windows and Trackbars ---
cv2.namedWindow(LK_WINDOW_NAME)
cv2.namedWindow(SIGNAL_WINDOW_NAME) # Create the second window

def nothing(x): pass # Dummy callback

# LK Trackbars (Attached to LK_WINDOW_NAME)
cv2.createTrackbar('maxCorners', LK_WINDOW_NAME, initial_feat_params.get('maxCorners', 100), 300, nothing)
cv2.createTrackbar('qualityLevel', LK_WINDOW_NAME, int(initial_feat_params.get('qualityLevel', 0.3) * 100), 99, nothing)
cv2.createTrackbar('minDistance', LK_WINDOW_NAME, initial_feat_params.get('minDistance', 7), 50, nothing)
win_size_init = initial_lk_params.get('winSize', [15, 15]); win_size_val = win_size_init[0] if isinstance(win_size_init, (list, tuple)) and len(win_size_init) > 0 else 15
cv2.createTrackbar('winSize', LK_WINDOW_NAME, win_size_val, 51, nothing) # Keep winSize odd
cv2.createTrackbar('maxLevel', LK_WINDOW_NAME, initial_lk_params.get('maxLevel', 2), 8, nothing)

# Signal Processing Trackbars (Attached to SIGNAL_WINDOW_NAME)
cv2.createTrackbar('Filt Low Hz*100', SIGNAL_WINDOW_NAME, int(signal_config_base.get('SIGNAL_FILTER_LOW_HZ', 0.1) * 100), 100, nothing)
cv2.createTrackbar('Filt High Hz*100', SIGNAL_WINDOW_NAME, int(signal_config_base.get('SIGNAL_FILTER_HIGH_HZ', 2.0) * 100), 500, nothing)
initial_prominence = signal_config_base.get('PEAK_DETECT_PROMINENCE')
if initial_prominence is None: initial_prominence = 0.15 # Default if not set
# Ensure initial prominence value is within trackbar range (0-4000)
initial_prominence_tb = min(4000, max(0, int(initial_prominence * 1000)))
cv2.createTrackbar('Peak Prom*1000', SIGNAL_WINDOW_NAME, initial_prominence_tb, 4000, nothing)


# --- Tuning Loop ---
prev_time = time.time(); frame_count = 0; feature_tracker = None
prev_lk_params = {} # Store previous LK slider values
prev_signal_params = {} # Store previous Signal slider values
last_saved_params = {}
mean_disp_history_len = int(MEAN_DISP_HISTORY_SECONDS * sampling_rate)
mean_dy_history = collections.deque(maxlen=mean_disp_history_len)
std_dy_history = collections.deque(maxlen=mean_disp_history_len)

while True:
    # --- Read Frame using VideoInput ---
    success, frame = video_input.get_frame()
    if not success or frame is None: print("End of video source or cannot read frame. Exiting."); break
    frame_count += 1;
    frame_height, frame_width = frame.shape[:2] # Get dims directly from frame
    if frame_width == 0 or frame_height == 0: print("Error: Invalid frame dimensions received. Exiting."); break

    # --- Read Trackbar Values ---
    # Read LK params from LK_WINDOW_NAME
    current_lk_params = {
        'maxCorners': max(10, cv2.getTrackbarPos('maxCorners', LK_WINDOW_NAME)),
        'qualityLevel': max(1, cv2.getTrackbarPos('qualityLevel', LK_WINDOW_NAME)) / 100.0,
        'minDistance': max(1, cv2.getTrackbarPos('minDistance', LK_WINDOW_NAME)),
        'winSize_raw': cv2.getTrackbarPos('winSize', LK_WINDOW_NAME),
        'maxLevel': cv2.getTrackbarPos('maxLevel', LK_WINDOW_NAME),
    }
    win_size = max(3, current_lk_params['winSize_raw'] // 2 * 2 + 1) # Ensure odd winSize >= 3
    current_lk_params['winSize'] = (win_size, win_size)

    # --- Read Signal params from SIGNAL_WINDOW_NAME (FIXED ORDER) ---
    filt_low_val = max(5, cv2.getTrackbarPos('Filt Low Hz*100', SIGNAL_WINDOW_NAME)) / 100.0
    filt_high_val = max(filt_low_val + 0.05, cv2.getTrackbarPos('Filt High Hz*100', SIGNAL_WINDOW_NAME) / 100.0) # Ensure high > low + small margin
    prominence_val = cv2.getTrackbarPos('Peak Prom*1000', SIGNAL_WINDOW_NAME) / 1000.0

    current_signal_params = {
        'filtLow': filt_low_val,
        'filtHigh': filt_high_val,
        'prominence': prominence_val,
    }
    if current_signal_params['filtLow'] >= current_signal_params['filtHigh']:
        current_signal_params['filtLow'] = max(0.01, current_signal_params['filtHigh'] - 0.05)


    # --- Check if parameters changed ---
    lk_params_changed = False
    compare_lk_keys = ['maxCorners', 'qualityLevel', 'minDistance', 'winSize_raw', 'maxLevel']
    if not prev_lk_params: lk_params_changed = True
    else:
        for key in compare_lk_keys:
            if current_lk_params[key] != prev_lk_params.get(key): lk_params_changed = True; break

    signal_params_changed = False
    compare_signal_keys = ['filtLow', 'filtHigh', 'prominence']
    if not prev_signal_params: signal_params_changed = True
    else:
         for key in compare_signal_keys:
             if not np.isclose(current_signal_params[key], prev_signal_params.get(key, -1.0), atol=1e-4):
                 signal_params_changed = True; break

    # --- Initialize/Re-initialize Components if needed ---
    if feature_tracker is None or lk_params_changed:
        if lk_params_changed: print(f"\n--- LK Parameters Changed! Re-initializing Tracker & Signal Processor ---")
        else: print(f"\n--- Initializing Tracker & Signal Processor ---")
        try:
            feature_params_dict = {'maxCorners': current_lk_params['maxCorners'],'qualityLevel': current_lk_params['qualityLevel'],'minDistance': current_lk_params['minDistance'],'blockSize': 7}
            lk_params_dict = {'winSize': current_lk_params['winSize'],'maxLevel': current_lk_params['maxLevel'],'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
            tracker_config = {'OPTICAL_FLOW_PARAMS': {'feature_params': feature_params_dict,'lk_params': lk_params_dict},'FEATURE_REDETECT_THRESHOLD': int(feature_params_dict.get('maxCorners', 100) * 0.7)}
            feature_tracker = FeatureTracker(config=tracker_config)

            current_sp_config = copy.deepcopy(signal_config_base)
            current_sp_config['SIGNAL_FILTER_LOW_HZ'] = current_signal_params['filtLow']
            current_sp_config['SIGNAL_FILTER_HIGH_HZ'] = current_signal_params['filtHigh']
            current_sp_config['PEAK_DETECT_PROMINENCE'] = current_signal_params['prominence'] if current_signal_params['prominence'] > 1e-6 else None
            signal_processor = SignalProcessor(config=current_sp_config, sampling_rate=sampling_rate)

            mean_dy_history.clear(); std_dy_history.clear()
            print("   (Signal processor & displacement history buffers reset)")
            prev_lk_params = {k: current_lk_params[k] for k in compare_lk_keys}; prev_lk_params['winSize'] = current_lk_params['winSize']
            prev_signal_params = current_signal_params.copy()
            latest_tracked_old = None; latest_tracked_new = None

        except Exception as e_init:
             print(f"ERROR initializing components: {e_init}"); feature_tracker = None; signal_processor = None; time.sleep(0.5); continue

    elif signal_params_changed:
         print(f"\n--- Signal Parameters Changed! Re-initializing Signal Processor ---")
         try:
            current_sp_config = copy.deepcopy(signal_config_base)
            current_sp_config['SIGNAL_FILTER_LOW_HZ'] = current_signal_params['filtLow']
            current_sp_config['SIGNAL_FILTER_HIGH_HZ'] = current_signal_params['filtHigh']
            current_sp_config['PEAK_DETECT_PROMINENCE'] = current_signal_params['prominence'] if current_signal_params['prominence'] > 1e-6 else None
            signal_processor = SignalProcessor(config=current_sp_config, sampling_rate=sampling_rate)
            print("   (Signal processor re-initialized, history kept)")
            prev_signal_params = current_signal_params.copy()

         except Exception as e_init_sp:
             print(f"ERROR re-initializing SignalProcessor: {e_init_sp}"); signal_processor = None; time.sleep(0.5); continue


    # --- Performance calculation (FPS) ---
    curr_time = time.time(); fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0; prev_time = curr_time

    # --- Run Tracking and Signal Pipeline ---
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracked_data = []; raw_signals = []
    latest_tracked_old = None; latest_tracked_new = None
    points_for_display = None # <<< Variable to store points for drawing

    current_mean_dy = 0.0; current_std_dy = 0.0

    if feature_tracker and signal_processor:
        try:
            # Process frame returns list: [(old0, new0), (old1, new1), ...]
            tracked_data = feature_tracker.process_frame(image_gray, static_roi_list)

            # --- Get points for displacement calculation AND display ---
            if tracked_data and tracked_data[0] is not None: # Assuming single ROI (index 0)
                old_pts, new_pts = tracked_data[0]
                # Check if both old and new points are valid and not empty
                if old_pts is not None and new_pts is not None and len(old_pts) > 0 and len(new_pts) > 0:
                    latest_tracked_old = old_pts.copy() # For displacement calc
                    latest_tracked_new = new_pts.copy() # For displacement calc
                    points_for_display = new_pts.copy() # <<< STORE the NEW points for drawing

                    # Calculate mean/std dy (using latest_tracked_old/new)
                    old_pts_flat = latest_tracked_old.reshape(-1, 2); new_pts_flat = latest_tracked_new.reshape(-1, 2)
                    displacements = new_pts_flat - old_pts_flat; dy_values = displacements[:, 1]
                    if dy_values.size > 0:
                        current_mean_dy = np.mean(dy_values); current_std_dy = np.std(dy_values)
                    else:
                        current_mean_dy = 0.0; current_std_dy = 0.0

            # --- Generate Signal ---
            if tracked_data: raw_signals = signal_generator.process_tracked_features(tracked_data)
            else: raw_signals = [0.0] * len(static_roi_list)
            signal_processor.process_signal_values(raw_signals)

        except Exception as e_pipe: print(f"Error during tracking/signal processing: {e_pipe}"); traceback.print_exc()
    elif signal_processor: # Still process signal even if tracker failed init
        signal_processor.process_signal_values([0.0] * len(static_roi_list))

    mean_dy_history.append(current_mean_dy)
    std_dy_history.append(current_std_dy)

    # --- Gather Results for Display ---
    bpm, bpm_valid = (0.0, False) if not signal_processor else signal_processor.get_bpm()
    phase = 0 if not signal_processor else signal_processor.get_phase()
    raw_signal_history = [] if not signal_processor else signal_processor.get_raw_signal_buffer()
    filtered_signal_history = [] if not signal_processor else signal_processor.get_filtered_signal_buffer()
    peak_indices = [] if not signal_processor else signal_processor.get_last_peak_indices()
    # Use the new variable holding the successfully tracked points from the current frame
    tracked_points_current = points_for_display # <<< USE the correct variable


    # --- Prepare images for drawing ---
    # --- Resize webcam feed FIRST ---
    try:
        aspect_ratio = frame_height / frame_width if frame_width > 0 else 1
        webcam_display_height = int(WEBCAM_DISPLAY_WIDTH * aspect_ratio)
        display_frame = cv2.resize(frame, (WEBCAM_DISPLAY_WIDTH, webcam_display_height), interpolation=cv2.INTER_AREA)
        display_w = display_frame.shape[1]
    except Exception as e_resize: print(f"Error resizing display frame: {e_resize}"); display_frame = frame.copy(); display_w = frame_width

    # Initialize plot images for the SIGNAL window
    plot_canvas_height = PLOT_HEIGHT * 3 # Height for 3 plots
    plot_canvas = np.full((plot_canvas_height, display_w, 3), (200, 200, 200), dtype=np.uint8) # Gray background

    # Generate the plots
    if display_w > 0:
        agg_method = getattr(signal_generator, 'aggregation_method', 'median').capitalize()
        raw_time_plot = draw_signal_plot(f"Raw Signal ({agg_method})", raw_signal_history, None, display_w, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_RAW_LINE_COLOR)
        filtered_time_plot = draw_signal_plot(
            f"Filtered Signal (Prom={current_signal_params['prominence']:.3f})",
            filtered_signal_history, peak_indices, display_w, PLOT_HEIGHT,
            PLOT_BG_COLOR, PLOT_FILT_LINE_COLOR, PLOT_PEAK_COLOR
        )
        mean_disp_plot = draw_mean_displacement_envelope(
            "Mean Vertical Displacement +/- StdDev", mean_dy_history, std_dy_history,
            display_w, PLOT_HEIGHT, PLOT_BG_COLOR, PLOT_MEAN_LINE_COLOR, PLOT_STD_FILL_COLOR
        )
        # Combine plots onto the canvas
        plot_canvas[0:PLOT_HEIGHT, :] = raw_time_plot
        plot_canvas[PLOT_HEIGHT:PLOT_HEIGHT*2, :] = filtered_time_plot
        plot_canvas[PLOT_HEIGHT*2:PLOT_HEIGHT*3, :] = mean_disp_plot


    # --- Draw Overlays on the RESIZED Webcam Frame (LK Window) ---
    scale_x = display_w / frame_width if frame_width > 0 else 1; scale_y = display_frame.shape[0] / frame_height if frame_height > 0 else 1
    roi_color = (255, 0, 0); point_color = (0, 0, 255); feature_count = 0
    for i, (x, y, w, h) in enumerate(static_roi_list): sx, sy = int(x * scale_x), int(y * scale_y); sw, sh = int(w * scale_x), int(h * scale_y); cv2.rectangle(display_frame, (sx, sy), (sx + sw, sy + sh), roi_color, 2)

    # Use tracked_points_current (which holds the *new* points) for drawing
    if tracked_points_current is not None: # <<< Check the correct variable
        feature_count = len(tracked_points_current)
        for point in tracked_points_current: # <<< Iterate over the correct variable
            try:
                # Check the shape of the point array
                if point.shape == (1, 2):
                    x_pt, y_pt = int(point[0, 0] * scale_x), int(point[0, 1] * scale_y)
                    cv2.circle(display_frame, (x_pt, y_pt), 3, point_color, -1)
                elif point.shape == (2,): # Handle potential (2,) shape
                     x_pt, y_pt = int(point[0] * scale_x), int(point[1] * scale_y)
                     cv2.circle(display_frame, (x_pt, y_pt), 3, point_color, -1)
            except IndexError: pass
            except Exception as draw_err: print(f"ERROR drawing point {point} with shape {point.shape}: {draw_err}")

    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (0, 255, 0), FONT_THICKNESS)
    param_text1 = f"Feat: N={current_lk_params['maxCorners']} Q={current_lk_params['qualityLevel']:.2f} D={current_lk_params['minDistance']}"
    param_text2 = f"LK: Win={current_lk_params['winSize']} Lvl={current_lk_params['maxLevel']}"
    cv2.putText(display_frame, param_text1, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_PARAMS, (255, 255, 255), 1)
    cv2.putText(display_frame, param_text2, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_PARAMS, (255, 255, 255), 1)
    cv2.putText(display_frame, f"Tracked Features: {feature_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (255, 0, 0), FONT_THICKNESS)
    bpm_text = f"BPM: {bpm:.1f}" if bpm_valid else "BPM: ---"; cv2.putText(display_frame, bpm_text, (display_w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (0, 255, 255), FONT_THICKNESS)
    phase_map = {signal_processor.PHASE_INHALE: "In", signal_processor.PHASE_EXHALE: "Ex", signal_processor.PHASE_UNKNOWN: "--"}; phase_text = f"P: {phase_map.get(phase, 'Err')}"
    cv2.putText(display_frame, phase_text, (display_w - 200, 60), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_INFO, (0, 255, 255), FONT_THICKNESS)

    # --- Display the windows ---
    cv2.imshow(LK_WINDOW_NAME, display_frame) # Show LK window with video
    cv2.imshow(SIGNAL_WINDOW_NAME, plot_canvas) # Show Signal window with plots

    # --- User Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): print("Exit key pressed."); break
    elif key == ord('s'): # Save parameters
        # (Save logic remains the same)
        print("\n--- Saving current parameters ---")
        try:
            feat_params_to_save = {'maxCorners': current_lk_params['maxCorners'],'qualityLevel': current_lk_params['qualityLevel'],'minDistance': current_lk_params['minDistance'],'blockSize': 7}
            lk_params_to_save = {'winSize': list(current_lk_params['winSize']),'maxLevel': current_lk_params['maxLevel'],'criteria': [3, 10, 0.03]} # Use default criteria for saving
            redetect_thresh = int(feat_params_to_save['maxCorners'] * 0.7)
            signal_params_to_save = {
                'SIGNAL_FILTER_LOW_HZ': current_signal_params['filtLow'],
                'SIGNAL_FILTER_HIGH_HZ': current_signal_params['filtHigh'],
                'PEAK_DETECT_PROMINENCE': current_signal_params['prominence'] if current_signal_params['prominence'] > 1e-6 else None
            }
            generator_params_to_save = {
                'SIGNAL_AGGREGATION_METHOD': getattr(signal_generator, 'aggregation_method', 'median')
            }

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

            if "signal_processor" not in current_full_config: current_full_config["signal_processor"] = {}
            current_full_config["signal_processor"].update(signal_params_to_save)

            if "signal_generator" not in current_full_config: current_full_config["signal_generator"] = {}
            current_full_config["signal_generator"].update(generator_params_to_save)


            with open(config_path, 'w') as f_write: json.dump(current_full_config, f_write, indent=2)
            print(f"Parameters saved to: {config_path}")
            last_saved_params = {**current_lk_params, **current_signal_params}

        except Exception as e_save: print(f"ERROR saving parameters to {config_path}: {e_save}"); traceback.print_exc()

# --- Cleanup ---
print("Releasing resources...")
if video_input: video_input.release()
cv2.destroyAllWindows()
print("Finished.")