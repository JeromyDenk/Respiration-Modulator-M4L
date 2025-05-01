# src/pipeline_manager.py
# Orchestrates the tracking -> signal generation -> signal processing phases.
# MODIFIED: Removed internal recalibration logic. Assumes ROI is set externally.

import cv2
import numpy as np
import time
import traceback
import collections

# Import necessary modules from src
try:
    from pose_detector import PoseDetector # Keep for potential future use? Or remove if truly unused here
    from coarse_roi_calculator import CoarseRoiCalculator # Keep for potential future use? Or remove
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
        # Extract configs needed for sub-modules
        feature_cfg = config.get('feature_tracker', {})
        signal_gen_cfg = config.get('signal_generator', {})
        signal_proc_cfg = config.get('signal_processor', {})
        # Pose/ROI configs might not be needed here anymore unless used for reset?
        # pose_cfg = config.get('pose_detector', {})
        # coarse_roi_cfg = config.get('coarse_roi_calculator', {})

        # --- Initialize Processing Modules ---
        try:
            # Initialize components needed for the tracking pipeline
            self.feature_tracker = FeatureTracker(config=feature_cfg)
            self.signal_generator = SignalGenerator(config=signal_gen_cfg)
            self.signal_processor = SignalProcessor(config=signal_proc_cfg, sampling_rate=self.sampling_rate)
            # PoseDetector and CoarseRoiCalculator are no longer directly managed here
            # self.pose_detector = PoseDetector(config=pose_cfg)
            # self.coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_cfg)

            print("[PipelineManager] Tracking components initialized.")
        except Exception as e:
            print(f"[PipelineManager] FATAL ERROR initializing processing modules: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize one or more processing modules.") from e

        # --- Internal State ---
        # This ROI list will be set externally before process_frame is called in tracking mode
        self.current_rois = []
        self.frame_count = 0 # Still useful for debugging maybe
        # Removed state related to recalibration
        # self.last_landmarks = None
        # self.needs_recalibration = True
        # self.last_recalibration_time = 0

    def set_tracking_roi(self, roi_list):
        """Sets the ROI to be used for tracking."""
        print(f"[PipelineManager] Setting tracking ROI: {roi_list}")
        self.current_rois = roi_list
        # Reset tracker state when ROI is set to ensure clean start
        if self.feature_tracker:
             print("[PipelineManager] Resetting FeatureTracker for new ROI.")
             self.feature_tracker = FeatureTracker(config=self.config.get('feature_tracker', {}))


    # Removed _should_run_recalibration, trigger_recalibration, _rois_are_different

    def process_frame(self, frame):
        """
        Processes a single video frame through the tracking pipeline
        (Track -> Generate Signal -> Process Signal) using the externally set ROI.

        Args:
            frame (np.ndarray): The input video frame (BGR format).

        Returns:
            dict: A dictionary containing the results (BPM, phase, signals, etc.),
                  or None if processing fails critically.
        """
        if frame is None:
            print("[PipelineManager] Error: Received None frame.")
            return None
        if not self.current_rois:
             print("[PipelineManager] Error: Tracking ROI not set. Cannot process frame.")
             # Return default values or raise error? Returning defaults for now.
             return {
                'bpm': 0.0, 'bpm_valid': False, 'phase': SignalProcessor.PHASE_UNKNOWN,
                'current_rois': [], 'landmarks': None, # No landmarks processed here
                'filtered_signal_history': [], 'raw_signal_history': [],
                'peak_indices': [], 'processing_time': 0,
                'recalibration_run_this_frame': False, 'recalibration_succeeded': False,
                'frame_count': self.frame_count
             }


        start_time = time.time()
        self.frame_count += 1
        # frame_height, frame_width = frame.shape[:2] # Not strictly needed here anymore

        # --- 1. Input Conversion ---
        try:
            # Only need grayscale for feature tracking
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e_cvt:
             print(f"[PipelineManager] Error converting frame colorspace: {e_cvt}")
             return None # Critical error

        # --- 2. Feature Tracking ---
        # Uses self.current_rois which was set externally
        tracked_data = []
        try:
            # FeatureTracker expects a list of ROIs
            tracked_data = self.feature_tracker.process_frame(image_gray, self.current_rois)
        except Exception as e_track:
             print(f"[PipelineManager] Error during feature tracking: {e_track}")
             traceback.print_exc()
             tracked_data = [(None, None)] * len(self.current_rois)

        # --- 3. Signal Generation ---
        raw_signals = []
        if tracked_data:
            try:
                 raw_signals = self.signal_generator.process_tracked_features(tracked_data)
            except Exception as e_siggen:
                 print(f"[PipelineManager] Error during signal generation: {e_siggen}")
                 traceback.print_exc()
                 raw_signals = [0.0] * len(tracked_data)

        # --- 4. Signal Processing ---
        try:
            self.signal_processor.process_signal_values(raw_signals)
        except Exception as e_sigproc:
             print(f"[PipelineManager] Error during signal processing: {e_sigproc}")
             traceback.print_exc()

        # --- 5. Gather Results ---
        bpm, bpm_valid = self.signal_processor.get_bpm()
        phase = self.signal_processor.get_phase()
        filtered_signal_history = self.signal_processor.get_filtered_signal_buffer()
        raw_signal_history = self.signal_processor.get_raw_signal_buffer()
        peak_indices = self.signal_processor.get_last_peak_indices()

        end_time = time.time()
        processing_time = end_time - start_time

        # Note: Landmarks are not processed or returned by this simplified manager
        results = {
            'bpm': bpm,
            'bpm_valid': bpm_valid,
            'phase': phase,
            'current_rois': self.current_rois, # Return the locked ROI used
            'landmarks': None, # No landmarks processed in this loop
            'filtered_signal_history': filtered_signal_history,
            'raw_signal_history': raw_signal_history,
            'peak_indices': peak_indices,
            'processing_time': processing_time,
            # Recalibration flags are no longer relevant here
            'recalibration_run_this_frame': False,
            'recalibration_succeeded': False,
            'frame_count': self.frame_count
        }

        return results

    def close(self):
        """Cleans up resources used by the pipeline components."""
        print("[PipelineManager] Closing pipeline...")
        # Close components managed by this pipeline
        # if hasattr(self.feature_tracker, 'close'): self.feature_tracker.close()
        # if hasattr(self.signal_generator, 'close'): self.signal_generator.close()
        # if hasattr(self.signal_processor, 'close'): self.signal_processor.close()
        print("[PipelineManager] Pipeline closed.")

