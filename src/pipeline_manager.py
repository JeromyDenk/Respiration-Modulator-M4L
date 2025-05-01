# src/pipeline_manager.py
# Orchestrates the tracking -> signal generation -> signal processing phases.
# MODIFIED: Return tracked points in results.

import cv2
import numpy as np
import time
import traceback
import collections

# Import necessary modules from src
try:
    # from pose_detector import PoseDetector # Not needed here
    # from coarse_roi_calculator import CoarseRoiCalculator # Not needed here
    from feature_tracker import FeatureTracker
    from signal_generator import SignalGenerator
    from signal_processor import SignalProcessor
except ImportError as e:
    print(f"PipelineManager Error: Failed to import sub-modules: {e}")
    raise

class PipelineManager:
    """
    Orchestrates the respiration monitoring pipeline stages *after* an ROI
    has been determined and locked: Feature Tracking -> Signal Generation -> Signal Processing.
    """
    def __init__(self, config, sampling_rate):
        """
        Initializes the pipeline manager and its processing components.

        Args:
            config (dict): The main configuration dictionary for the application.
            sampling_rate (float): The processing rate (e.g., video FPS).
        """
        print("[PipelineManager] Initializing (Tracking Mode)...")
        self.config = config
        self.sampling_rate = sampling_rate
        if self.sampling_rate <= 0:
            raise ValueError("PipelineManager requires a positive sampling_rate.")

        # --- Configuration Extraction ---
        feature_cfg = config.get('feature_tracker', {})
        signal_gen_cfg = config.get('signal_generator', {})
        signal_proc_cfg = config.get('signal_processor', {})

        # --- Initialize Processing Modules ---
        try:
            self.feature_tracker = FeatureTracker(config=feature_cfg)
            self.signal_generator = SignalGenerator(config=signal_gen_cfg)
            self.signal_processor = SignalProcessor(config=signal_proc_cfg, sampling_rate=self.sampling_rate)
            print("[PipelineManager] Tracking components initialized.")
        except Exception as e:
            print(f"[PipelineManager] FATAL ERROR initializing processing modules: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize one or more processing modules.") from e

        # --- Internal State ---
        self.current_rois = []
        self.frame_count = 0

    def set_tracking_roi(self, roi_list):
        """Sets the ROI to be used for tracking."""
        print(f"[PipelineManager] Setting tracking ROI: {roi_list}")
        self.current_rois = roi_list
        if self.feature_tracker:
             print("[PipelineManager] Resetting FeatureTracker for new ROI.")
             self.feature_tracker = FeatureTracker(config=self.config.get('feature_tracker', {}))


    def process_frame(self, frame):
        """
        Processes a single video frame through the tracking pipeline
        (Track -> Generate Signal -> Process Signal) using the externally set ROI.

        Args:
            frame (np.ndarray): The input video frame (BGR format).

        Returns:
            dict: A dictionary containing the results (BPM, phase, signals, tracked_points, etc.),
                  or None if processing fails critically.
        """
        if frame is None: print("[PipelineManager] Error: Received None frame."); return None
        if not self.current_rois:
             print("[PipelineManager] Error: Tracking ROI not set. Cannot process frame.")
             return { # Return default structure
                'bpm': 0.0, 'bpm_valid': False, 'phase': SignalProcessor.PHASE_UNKNOWN,
                'current_rois': [], 'landmarks': None,
                'filtered_signal_history': [], 'raw_signal_history': [],
                'peak_indices': [], 'processing_time': 0,
                'recalibration_run_this_frame': False, 'recalibration_succeeded': False,
                'frame_count': self.frame_count, 'tracked_points': None # Add tracked_points key
             }

        start_time = time.time()
        self.frame_count += 1

        # --- 1. Input Conversion ---
        try:
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e_cvt: print(f"[PipelineManager] Error converting frame colorspace: {e_cvt}"); return None

        # --- 2. Feature Tracking ---
        tracked_data = []
        current_tracked_points = None # Store the points tracked in *this* frame
        try:
            tracked_data = self.feature_tracker.process_frame(image_gray, self.current_rois)
            # --- Extract points tracked in this frame for returning ---
            if tracked_data and tracked_data[0] is not None:
                 # tracked_data is list of tuples: [(old0, new0), ...]
                 # Get the new points from the first ROI's tuple
                 _, new_points = tracked_data[0]
                 if new_points is not None:
                      current_tracked_points = new_points # Store for results dict
            # ---
        except Exception as e_track:
             print(f"[PipelineManager] Error during feature tracking: {e_track}"); traceback.print_exc()
             tracked_data = [(None, None)] * len(self.current_rois)

        # --- 3. Signal Generation ---
        raw_signals = []
        if tracked_data:
            try: raw_signals = self.signal_generator.process_tracked_features(tracked_data)
            except Exception as e_siggen: print(f"[PipelineManager] Error during signal generation: {e_siggen}"); traceback.print_exc(); raw_signals = [0.0] * len(tracked_data)

        # --- 4. Signal Processing ---
        try: self.signal_processor.process_signal_values(raw_signals)
        except Exception as e_sigproc: print(f"[PipelineManager] Error during signal processing: {e_sigproc}"); traceback.print_exc()

        # --- 5. Gather Results ---
        bpm, bpm_valid = self.signal_processor.get_bpm()
        phase = self.signal_processor.get_phase()
        filtered_signal_history = self.signal_processor.get_filtered_signal_buffer()
        raw_signal_history = self.signal_processor.get_raw_signal_buffer()
        peak_indices = self.signal_processor.get_last_peak_indices()

        end_time = time.time()
        processing_time = end_time - start_time

        results = {
            'bpm': bpm, 'bpm_valid': bpm_valid, 'phase': phase,
            'current_rois': self.current_rois, 'landmarks': None,
            'filtered_signal_history': filtered_signal_history,
            'raw_signal_history': raw_signal_history,
            'peak_indices': peak_indices,
            'processing_time': processing_time,
            'recalibration_run_this_frame': False, 'recalibration_succeeded': False,
            'frame_count': self.frame_count,
            'tracked_points': current_tracked_points # <<< ADDED tracked points
        }
        return results

    def close(self):
        # (Close logic remains the same)
        print("[PipelineManager] Closing pipeline...")
        print("[PipelineManager] Pipeline closed.")

