# scripts/test_roi_refinement_overlay.py
# Tests and visualizes the ROI refinement process: Pose -> Coarse ROI -> EVM Refined ROI.
# Displays the webcam feed with pose landmarks and both ROI boundaries.
# Uses VideoInput class and displays EVM variance map.
# ASSUMES this script lives in a 'scripts' subdirectory of the project root.

import cv2
import mediapipe as mp
import time
import numpy as np
import os
import sys
import traceback
import json # To load the profile
import collections

# --- Add src directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import necessary classes from src ---
try:
    from video_input import VideoInput # Use VideoInput
    from pose_detector import PoseDetector
    # IMPORTANT: Assumes roi_calculator.py has been renamed to coarse_roi_calculator.py
    from coarse_roi_calculator import CoarseRoiCalculator
    from evm_processor import EvmProcessor
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure required modules exist and RoiCalculator has been renamed.")
    sys.exit(1)

print("Initializing Webcam (via VideoInput) and ROI Components...")

# --- Load Configuration ---
config_path = os.path.join(project_root, "profiles", "test_profile.json")
config = {}
try:
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {config_path}")
    # Ensure EVM is enabled in the loaded config for this test
    if not config.get("evm_processor", {}).get("EVM_ENABLED", False):
         print("Error: EVM_ENABLED must be true in the profile for this test script.")
         sys.exit(1) # Exit if EVM is not enabled, as this script requires it
except FileNotFoundError:
    print(f"Error: Configuration file not found at {config_path}. Exiting.")
    sys.exit(1)
except json.JSONDecodeError:
     print(f"Error: Could not parse configuration file {config_path}. Exiting.")
     sys.exit(1)
except Exception as e:
     print(f"Error loading configuration: {e}. Exiting.")
     sys.exit(1)

# Extract specific config sections or use defaults
video_config = config.get("video_input", {}) # Get video config
pose_config = config.get("pose_detector", {})
coarse_roi_config = config.get("coarse_roi_calculator", {})
evm_config = config.get("evm_processor", {})
evm_enabled = evm_config.get("EVM_ENABLED", True) # Should be true based on check above
evm_buffer_seconds = evm_config.get("EVM_BUFFER_SECONDS", 2.0)

DEFAULT_SAMPLING_RATE = 30.0 # Fallback sampling rate
EVM_VIZ_WINDOW_NAME = "EVM Variance Map (2x)"
MAIN_VIZ_WINDOW_NAME = 'ROI Refinement Test (Pose -> Coarse -> Refined)'
DISPLAY_WIDTH = 960 # Target width for the main display window
EVM_VIZ_SCALE_FACTOR = 2.0 # Factor to scale the EVM map window

# --- Plotting Helper Function ---
# (draw_signal_plot function is not needed for this script, can be removed or kept)
# def draw_signal_plot(...): ...

# --- Initialize Video Input ---
video_input = None
try:
    video_input = VideoInput(config=video_config)
    if not video_input.initialized:
        raise RuntimeError("VideoInput failed to initialize.")
    # Use actual sampling rate if available, otherwise default
    actual_fps = video_input.get_fps()
    sampling_rate = actual_fps if actual_fps > 0 else DEFAULT_SAMPLING_RATE
    print(f"Using sampling rate for EVM buffer calculation: {sampling_rate:.2f} Hz")
except Exception as e:
    print(f"Error initializing VideoInput: {e}")
    if video_input: video_input.release()
    sys.exit(1)


# --- Initialize Components ---
pose_detector = None
coarse_roi_calculator = None
evm_processor = None
grayscale_frame_buffer = None
try:
    pose_detector = PoseDetector(config=pose_config)
    coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_config)
    if evm_enabled: # Should always be true here
        # Pass the determined sampling rate to EvmProcessor
        evm_processor = EvmProcessor(config=evm_config, sampling_rate=sampling_rate)
        evm_buffer_size = int(evm_buffer_seconds * sampling_rate) # Calculate based on actual/estimated rate
        grayscale_frame_buffer = collections.deque(maxlen=evm_buffer_size)
        print(f"EVM Enabled. Buffer Size: {evm_buffer_size}")
    else:
         # This case should not be hit due to check during config load
         print("Error: EVM Processor could not be enabled/initialized. Exiting.")
         video_input.release()
         sys.exit(1)

except Exception as e:
    print(f"Error during component initialization: {e}")
    traceback.print_exc()
    if pose_detector: pose_detector.close()
    video_input.release() # Release video input on error
    sys.exit(1)

# --- MediaPipe Drawing Utilities ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

print("Webcam opened. Displaying Pose, Coarse ROI, and Refined ROI.")
print("EVM Variance Map will be shown in a separate window.")
print("Press 'q' to quit.")

# --- Frame Processing Loop ---
prev_time = time.time()
frame_count = 0
latest_variance_map = None
latest_refined_roi_local = None

while True: # Loop until user quits or error
    # --- Read Frame using VideoInput ---
    success, frame = video_input.get_frame()
    if not success or frame is None:
        print("End of video source or cannot read frame. Exiting.")
        break # Exit loop if frame read fails

    frame_count += 1
    # Get actual resolution from VideoInput object
    frame_width, frame_height = video_input.get_resolution()
    if frame_width == 0 or frame_height == 0: # Check if resolution is valid
        print("Error: Invalid frame dimensions received from VideoInput. Exiting.")
        break

    # --- Performance calculation (FPS) ---
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # --- Process Frame ---
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_rgb.flags.writeable = False

    if grayscale_frame_buffer is not None:
        grayscale_frame_buffer.append(image_gray.copy())

    landmarks = pose_detector.process_frame(image_rgb)

    display_frame = frame.copy() # Start with original frame for drawing
    coarse_roi_list = []
    refined_roi_list = []
    current_variance_map = None

    if landmarks:
        coarse_roi_list = coarse_roi_calculator.calculate_coarse_roi(landmarks, (frame_height, frame_width))

        if evm_processor and coarse_roi_list and grayscale_frame_buffer is not None:
            coarse_roi_coords = coarse_roi_list[0]
            if len(grayscale_frame_buffer) >= evm_processor.min_buffer_frames:
                 x_c, y_c, w_c, h_c = coarse_roi_coords
                 cropped_frames_deque = collections.deque(maxlen=grayscale_frame_buffer.maxlen)
                 buffer_valid_for_evm = True
                 # --- Simplified Buffer Cropping ---
                 try:
                     for gray_frame_buffer in grayscale_frame_buffer:
                         fh, fw = gray_frame_buffer.shape
                         x1, y1 = max(0, x_c), max(0, y_c)
                         x2, y2 = min(fw, x_c + w_c), min(fh, y_c + h_c)
                         if x1 >= x2 or y1 >= y2: buffer_valid_for_evm = False; break
                         cropped_frames_deque.append(gray_frame_buffer[y1:y2, x1:x2])
                 except Exception as crop_err:
                      print(f"Error during buffer cropping: {crop_err}")
                      buffer_valid_for_evm = False

                 if buffer_valid_for_evm and len(cropped_frames_deque) == len(grayscale_frame_buffer): # Ensure all frames were cropped
                     refined_roi_list, current_variance_map = evm_processor.find_optimal_roi(
                         cropped_frames_deque, coarse_roi_coords
                     )
                     if current_variance_map is not None:
                         latest_variance_map = current_variance_map.copy()
                         if refined_roi_list:
                              rx_orig, ry_orig, rw, rh = refined_roi_list[0]
                              rx_local = rx_orig - x_c
                              ry_local = ry_orig - y_c
                              latest_refined_roi_local = (rx_local, ry_local, rw, rh)
                         else: latest_refined_roi_local = None


    # --- Draw Overlays ---
    # 1. Draw Pose Landmarks
    if landmarks:
        mp_drawing.draw_landmarks(
            display_frame, landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    else:
        cv2.putText(display_frame, "No Pose", (frame_width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 2. Draw Coarse ROI (if calculated)
    if coarse_roi_list:
        for i, (x, y, w, h) in enumerate(coarse_roi_list):
             cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # Yellow Coarse ROI
             cv2.putText(display_frame, f"Coarse ROI {i}", (x + 5, y + 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif landmarks: # Pose found but coarse ROI failed
        cv2.putText(display_frame, "Coarse ROI Fail", (frame_width - 200, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # 3. Draw Refined ROI (if calculated)
    if refined_roi_list:
        try:
            for i, roi_tuple in enumerate(refined_roi_list):
                if isinstance(roi_tuple, tuple) and len(roi_tuple) == 4:
                    x, y, w, h = roi_tuple
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green Refined ROI
                    cv2.putText(display_frame, f"Refined ROI {i}", (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print(f"ERROR: Item in refined_roi_list is not a 4-element tuple: Index={i}, Value={roi_tuple}, Type={type(roi_tuple)}")
                    if isinstance(roi_tuple, tuple) and len(roi_tuple) >= 2:
                        try:
                           err_x, err_y = int(roi_tuple[0]), int(roi_tuple[1])
                           cv2.circle(display_frame, (err_x, err_y), 10, (0,0,255), 2)
                        except: pass
        except Exception as loop_err:
             print(f"ERROR during refined ROI drawing loop: {loop_err}")
             traceback.print_exc()

    elif evm_enabled and coarse_roi_list and grayscale_frame_buffer is not None and len(grayscale_frame_buffer) < evm_processor.min_buffer_frames:
         cv2.putText(display_frame, "EVM Buffer Filling...", (frame_width - 250, 80),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2) # Orange text
    elif evm_enabled and coarse_roi_list and not refined_roi_list:
         status_text = "Refined ROI Fail" if latest_variance_map is not None else "EVM Step Fail"
         cv2.putText(display_frame, status_text, (frame_width - 200, 80),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


    # --- Display FPS & Frame Count ---
    cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # --- Resize the Main Display Frame ---
    try:
        aspect_ratio = frame_height / frame_width if frame_width > 0 else 1
        display_height = int(DISPLAY_WIDTH * aspect_ratio)
        resized_display_frame = cv2.resize(display_frame, (DISPLAY_WIDTH, display_height), interpolation=cv2.INTER_AREA)
    except Exception as e_resize:
        print(f"Error resizing display frame: {e_resize}")
        resized_display_frame = display_frame # Fallback


    # --- Display the main frame ---
    cv2.imshow(MAIN_VIZ_WINDOW_NAME, resized_display_frame) # Use the defined constant

    # --- Prepare and Display EVM Variance Map Visualization ---
    if latest_variance_map is not None:
        try:
            norm_map = cv2.normalize(latest_variance_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            if latest_refined_roi_local:
                 lx, ly, lw, lh = latest_refined_roi_local
                 lx1, ly1 = max(0, lx), max(0, ly)
                 lx2, ly2 = min(color_map.shape[1], lx + lw), min(color_map.shape[0], ly + lh)
                 cv2.rectangle(color_map, (lx1, ly1), (lx2, ly2), (255, 255, 255), 1) # White box

            # --- Resize the EVM map ---
            evm_map_h, evm_map_w = color_map.shape[:2]
            if evm_map_h > 0 and evm_map_w > 0:
                new_evm_w = int(evm_map_w * EVM_VIZ_SCALE_FACTOR)
                new_evm_h = int(evm_map_h * EVM_VIZ_SCALE_FACTOR)
                # Use INTER_NEAREST for blocky scaling suitable for heatmaps
                resized_color_map = cv2.resize(color_map, (new_evm_w, new_evm_h), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(EVM_VIZ_WINDOW_NAME, resized_color_map) # Show resized map
            else:
                cv2.imshow(EVM_VIZ_WINDOW_NAME, color_map) # Show original if dimensions invalid

        except Exception as e_viz:
            print(f"Error creating EVM visualization: {e_viz}")
            if cv2.getWindowProperty(EVM_VIZ_WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
                 try: cv2.destroyWindow(EVM_VIZ_WINDOW_NAME)
                 except: pass # Ignore error if window already closed

    # --- User Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        print("Exit key pressed.")
        break

# --- Cleanup ---
print("Releasing resources...")
if pose_detector: pose_detector.close()
if video_input: video_input.release() # Release VideoInput
cv2.destroyAllWindows()
print("Finished.")

