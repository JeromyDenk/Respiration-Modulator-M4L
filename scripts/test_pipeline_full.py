# scripts/test_pipeline_full.py
# Runs the full pipeline using PipelineManager and displays webcam feed
# along with a separate breathing visualization window (expanding/contracting ring).

import cv2
import numpy as np
import time
import os
import sys
import traceback
import json
import collections # For deque if needed for smoothing

# --- Add src directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import necessary classes from src ---
try:
    # Use VideoInput if available, otherwise fallback
    try:
        from video_input import VideoInput
    except ImportError:
        print("Warning: video_input.py not found. Using direct cv2.VideoCapture.")
        VideoInput = None

    # Import the pipeline manager (ensure it's the version without EVM)
    from pipeline_manager import PipelineManager
    # Import SignalProcessor constants if needed for phase checking
    from signal_processor import SignalProcessor

except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure PipelineManager and other required modules exist in 'src'.")
    sys.exit(1)

print("Initializing Webcam and PipelineManager...")

# --- Configuration ---
PROFILE_FILENAME = "test_profile.json" # Ensure this profile has your tuned params
config_path = os.path.join(project_root, "profiles", PROFILE_FILENAME)
config = {}
try:
    with open(config_path, 'r') as f: config = json.load(f)
    print(f"Loaded configuration from: {config_path}")
except Exception as e:
    print(f"Warning: Failed to load config '{config_path}': {e}. Using defaults/empty config.")
    config = {} # Use empty config which might lead to default values in components

# --- Visualization Parameters ---
WEBCAM_WINDOW_NAME = 'Webcam Feed'
BREATHING_VIZ_WINDOW_NAME = 'Breathing Visualization'
VIZ_WINDOW_SIZE = 600 # Size of the square visualization window
MIN_RING_RADIUS = 50 # Minimum radius of the ring
MAX_RING_RADIUS = VIZ_WINDOW_SIZE // 2 - 30 # Max radius, leave some padding
RING_THICKNESS = 10 # Thickness of the ring
RING_COLOR = (255, 255, 255) # White ring
BG_COLOR = (0, 0, 0) # Black background
RADIUS_CHANGE_RATE = 0.05 # Controls how fast the ring size changes (fraction per frame)

DEFAULT_SAMPLING_RATE = 30.0 # Estimate

# --- Initialize Video Input ---
video_input = None
sampling_rate = DEFAULT_SAMPLING_RATE
try:
    if VideoInput:
        video_input = VideoInput(config=config.get("video_input", {}))
        if not video_input.initialized: raise RuntimeError("VideoInput failed to initialize.")
        actual_fps = video_input.get_fps()
        sampling_rate = actual_fps if actual_fps > 0 else DEFAULT_SAMPLING_RATE
    else:
        print("Using direct cv2.VideoCapture(0)")
        video_input = cv2.VideoCapture(0)
        if not video_input.isOpened(): raise RuntimeError("Cannot open webcam with cv2.VideoCapture.")
        # Dummy methods for compatibility
        video_input.get_frame = lambda: video_input.read()
        video_input.get_resolution = lambda: (int(video_input.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_input.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_input.release = lambda: video_input.release()

    print(f"Using sampling rate for signal processing: {sampling_rate:.2f} Hz")

except Exception as e:
    print(f"Error initializing Video Input: {e}")
    sys.exit(1)

# --- Initialize Pipeline Manager ---
pipeline = None
try:
    # Pass the determined sampling rate
    pipeline = PipelineManager(config=config, sampling_rate=sampling_rate)
except Exception as e:
    print(f"Error during PipelineManager initialization: {e}")
    traceback.print_exc()
    if video_input: video_input.release()
    sys.exit(1)

# --- Create Windows ---
cv2.namedWindow(WEBCAM_WINDOW_NAME)
cv2.namedWindow(BREATHING_VIZ_WINDOW_NAME)
cv2.resizeWindow(BREATHING_VIZ_WINDOW_NAME, VIZ_WINDOW_SIZE, VIZ_WINDOW_SIZE)

print("Starting main loop...")
print("Press 'q' to quit.")
print("Press 'r' to force recalibration.")

# --- Main Loop ---
prev_time = time.time()
current_radius = (MIN_RING_RADIUS + MAX_RING_RADIUS) / 2 # Start in the middle

while True:
    # --- Read Frame ---
    success, frame = video_input.get_frame()
    if not success or frame is None:
        print("End of video source or cannot read frame. Exiting.")
        break
    frame_count = pipeline.frame_count + 1 # Get count before processing
    frame_height, frame_width = frame.shape[:2]
    if frame_width == 0 or frame_height == 0:
        print("Error: Invalid frame dimensions received. Exiting.")
        break

    # --- Performance calculation (FPS) ---
    curr_time = time.time(); fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0; prev_time = curr_time

    # --- Process Frame through Pipeline ---
    pipeline_results = None
    try:
        pipeline_results = pipeline.process_frame(frame)
    except Exception as e_proc:
        print(f"Error during pipeline processing: {e_proc}")
        traceback.print_exc()
        # Continue loop, results will be None

    # --- Extract Results ---
    bpm = 0.0
    bpm_valid = False
    phase = SignalProcessor.PHASE_UNKNOWN # Default to unknown
    if pipeline_results:
        bpm = pipeline_results.get('bpm', 0.0)
        bpm_valid = pipeline_results.get('bpm_valid', False)
        phase = pipeline_results.get('phase', SignalProcessor.PHASE_UNKNOWN)

    # --- Prepare Webcam Display ---
    webcam_display = frame.copy()
    # Add FPS overlay
    cv2.putText(webcam_display, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Add BPM overlay
    bpm_text = f"BPM: {bpm:.1f}" if bpm_valid else "BPM: ---"
    bpm_color = (0, 255, 255) if bpm_valid else (0, 0, 255)
    cv2.putText(webcam_display, bpm_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bpm_color, 2)
    # Add Phase overlay
    phase_map = {SignalProcessor.PHASE_INHALE: "Inhale",
                 SignalProcessor.PHASE_EXHALE: "Exhale",
                 SignalProcessor.PHASE_UNKNOWN: "---"}
    phase_text = f"Phase: {phase_map.get(phase, 'Error')}"
    cv2.putText(webcam_display, phase_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # Add Recalibration status
    if pipeline.needs_recalibration:
         cv2.putText(webcam_display, "RECALIBRATING", (frame_width - 200, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif pipeline_results and pipeline_results.get('recalibration_run_this_frame', False):
         status = "OK" if pipeline_results.get('recalibration_succeeded', False) else "FAIL"
         cv2.putText(webcam_display, f"RECAL: {status}", (frame_width - 150, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)


    # --- Prepare Breathing Visualization ---
    viz_image = np.full((VIZ_WINDOW_SIZE, VIZ_WINDOW_SIZE, 3), BG_COLOR, dtype=np.uint8)
    center = (VIZ_WINDOW_SIZE // 2, VIZ_WINDOW_SIZE // 2)

    # Smoothly animate radius based on phase
    target_radius = current_radius # Default to current
    if phase == SignalProcessor.PHASE_INHALE:
        target_radius = MAX_RING_RADIUS
    elif phase == SignalProcessor.PHASE_EXHALE:
        target_radius = MIN_RING_RADIUS
    # If phase is unknown, keep moving towards the middle or stay put
    elif phase == SignalProcessor.PHASE_UNKNOWN:
         target_radius = (MIN_RING_RADIUS + MAX_RING_RADIUS) / 2 # Option: move to middle
         # target_radius = current_radius # Option: stay put

    # Move current radius towards target radius
    radius_diff = target_radius - current_radius
    current_radius += radius_diff * RADIUS_CHANGE_RATE
    # Clamp radius within bounds
    current_radius = max(MIN_RING_RADIUS, min(MAX_RING_RADIUS, current_radius))

    # Draw the ring
    cv2.circle(viz_image, center, int(current_radius), RING_COLOR, RING_THICKNESS, lineType=cv2.LINE_AA)

    # --- Display Windows ---
    cv2.imshow(WEBCAM_WINDOW_NAME, webcam_display)
    cv2.imshow(BREATHING_VIZ_WINDOW_NAME, viz_image)

    # --- User Input ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        print("Exit key pressed.")
        break
    elif key == ord('r'):
        print("Recalibration key pressed.")
        if pipeline:
            pipeline.trigger_recalibration()

# --- Cleanup ---
print("Releasing resources...")
if video_input: video_input.release()
if pipeline: pipeline.close()
cv2.destroyAllWindows()
print("Finished.")
