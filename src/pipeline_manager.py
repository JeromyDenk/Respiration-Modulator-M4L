# src/pipeline_manager.py
# Orchestrates the different processing phases.
# Includes logic for periodic recalibration using Pose and Coarse ROI.
# REMOVED: All EVM-related functionality.

import cv2
import numpy as np
import time
import traceback
import collections # For deque (still potentially useful for other buffering if needed)

# Import necessary modules from src
try:
    from pose_detector import PoseDetector
    from coarse_roi_calculator import CoarseRoiCalculator # Use the renamed class
    from feature_tracker import FeatureTracker
    from signal_generator import SignalGenerator
    from signal_processor import SignalProcessor
    # from evm_processor import EvmProcessor # Removed EVM import
except ImportError as e:
    print(f"PipelineManager Error: Failed to import sub-modules: {e}")
    raise

class PipelineManager:
    """
    Orchestrates the respiration monitoring pipeline, managing state and
    coordinating the processing modules. Includes logic to run expensive
    pose/ROI detection periodically. EVM functionality has been removed.
    """
    def __init__(self, config, sampling_rate):
        """
        Initializes the pipeline manager and all its components.

        Args:
            config (dict): The main configuration dictionary for the application.
                           Expects nested sections like "pose_detector", etc.
            sampling_rate (float): The processing rate (e.g., video FPS).
        """
        print("[PipelineManager] Initializing...")
        self.config = config
        self.sampling_rate = sampling_rate
        if self.sampling_rate <= 0:
            raise ValueError("PipelineManager requires a positive sampling_rate.")

        # --- Configuration Extraction ---
        pipeline_cfg = config.get('pipeline_manager', {})
        pose_cfg = config.get('pose_detector', {})
        coarse_roi_cfg = config.get('coarse_roi_calculator', {})
        feature_cfg = config.get('feature_tracker', {})
        signal_gen_cfg = config.get('signal_generator', {})
        signal_proc_cfg = config.get('signal_processor', {})
        # evm_cfg = config.get('evm_processor', {}) # Removed EVM config extraction

        self.pose_detection_interval = max(1, pipeline_cfg.get('POSE_DETECTION_FRAME_INTERVAL', 1))
        self.recalibration_interval_sec = pipeline_cfg.get('PIPELINE_RECALIBRATION_INTERVAL_SEC', 300) # 5 minutes default
        # Removed EVM related config variables
        # self.evm_enabled = evm_cfg.get('EVM_ENABLED', False)
        # self.evm_buffer_seconds = evm_cfg.get('EVM_BUFFER_SECONDS', 2.0)
        # self.evm_buffer_size = int(self.evm_buffer_seconds * self.sampling_rate) if self.sampling_rate > 0 else 0

        # --- Initialize Processing Modules ---
        try:
            self.pose_detector = PoseDetector(config=pose_cfg)
            self.coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_cfg)
            self.feature_tracker = FeatureTracker(config=feature_cfg)
            self.signal_generator = SignalGenerator(config=signal_gen_cfg)
            self.signal_processor = SignalProcessor(config=signal_proc_cfg, sampling_rate=self.sampling_rate)
            # self.evm_processor = None # Removed EVM processor instance
            print("[PipelineManager] Core processing modules initialized (EVM Disabled).")
        except Exception as e:
            print(f"[PipelineManager] FATAL ERROR initializing processing modules: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize one or more processing modules.") from e

        # --- Internal State ---
        self.current_rois = [] # Stores the active ROI(s) from CoarseRoiCalculator
        self.last_landmarks = None
        self.needs_recalibration = True
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_recalibration_time = 0
        # self.grayscale_frame_buffer = None # Removed EVM frame buffer

        print(f"[PipelineManager] Pose detection interval: {self.pose_detection_interval} frame(s).")
        print(f"[PipelineManager] Recalibration time interval: {self.recalibration_interval_sec} sec.")


    def trigger_recalibration(self):
        """Sets a flag to force pose/ROI recalculation on the next frame."""
        print("[PipelineManager] Recalibration triggered externally.")
        self.needs_recalibration = True

    def _should_run_recalibration(self):
        """Determines if pose/ROI recalculation should run on the current frame."""
        if self.needs_recalibration: return True
        if not self.current_rois: return True
        if self.pose_detection_interval > 1 and self.frame_count % self.pose_detection_interval == 0: return True
        if self.recalibration_interval_sec > 0 and \
           time.time() - self.last_recalibration_time > self.recalibration_interval_sec:
            print("[PipelineManager] Recalibration interval reached.")
            return True
        return False

    # Removed _get_buffered_coarse_roi_frames helper function

    def _rois_are_different(self, rois1, rois2, threshold=5):
        """Helper to check if two lists of ROIs are significantly different."""
        if len(rois1) != len(rois2):
            return True
        if not rois1: # Both are empty
            return False
        r1 = np.array(rois1[0])
        r2 = np.array(rois2[0])
        if np.any(np.abs(r1 - r2) > threshold):
            return True
        return False

    def process_frame(self, frame):
        """
        Processes a single video frame through the entire pipeline (Pose -> ROI -> Track -> Signal).

        Args:
            frame (np.ndarray): The input video frame (BGR format).

        Returns:
            dict: A dictionary containing the results, or None if processing fails critically.
        """
        if frame is None:
            print("[PipelineManager] Error: Received None frame.")
            return None

        start_time = time.time()
        self.frame_count += 1
        frame_height, frame_width = frame.shape[:2]

        # --- 1. Input Conversion ---
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_rgb.flags.writeable = False # Performance hint for MediaPipe
            # No EVM buffer needed
        except cv2.error as e_cvt:
             print(f"[PipelineManager] Error converting frame colorspace: {e_cvt}")
             return None # Critical error

        # --- 2. Recalibration Logic (Pose -> Coarse ROI) ---
        run_recalibration = self._should_run_recalibration()
        recalibration_succeeded = False # Track if ROI was successfully updated this frame
        pose_succeeded = False
        reset_tracker_needed = False

        was_forced_recalibration = self.needs_recalibration
        roi_was_missing = not self.current_rois
        new_roi_for_this_cycle = []

        if run_recalibration:
            print(f"[PipelineManager] Running Recalibration (Frame {self.frame_count})...")
            landmarks = self.pose_detector.process_frame(image_rgb)
            if landmarks:
                pose_succeeded = True
                self.last_landmarks = landmarks
                # Calculate the coarse ROI directly
                coarse_roi_list = self.coarse_roi_calculator.calculate_coarse_roi(landmarks, (frame_height, frame_width))

                if coarse_roi_list:
                    print(f"[PipelineManager]   Coarse ROI calculated: {coarse_roi_list}")
                    new_roi_for_this_cycle = coarse_roi_list
                    recalibration_succeeded = True
                else: # Coarse ROI calculation failed
                    print("[PipelineManager]   Coarse ROI calculation failed.")
                    # Keep recalibration_succeeded as False
            else: # Pose detection failed
                print("[PipelineManager]   Pose detection failed.")
                self.last_landmarks = None
                # Keep recalibration_succeeded as False

            # --- Update State based on Recalibration Outcome ---
            if recalibration_succeeded:
                if self._rois_are_different(new_roi_for_this_cycle, self.current_rois):
                     print(f"[PipelineManager]   ROI changed. Resetting tracker.")
                     reset_tracker_needed = True
                     self.current_rois = new_roi_for_this_cycle
                self.needs_recalibration = False
                self.last_recalibration_time = time.time()
            else: # Recalibration failed
                if was_forced_recalibration or roi_was_missing:
                    if self.current_rois: reset_tracker_needed = True
                    self.current_rois = []
                    self.needs_recalibration = True
                # else: Periodic check failed, keep old ROI

            # Reset tracker if flagged
            if reset_tracker_needed:
                print("[PipelineManager] Re-initializing FeatureTracker state.")
                self.feature_tracker = FeatureTracker(config=self.config.get('feature_tracker', {}))

        # --- 3. Feature Tracking ---
        tracked_data = []
        if self.current_rois:
             try:
                 tracked_data = self.feature_tracker.process_frame(image_gray, self.current_rois)
             except Exception as e_track:
                  print(f"[PipelineManager] Error during feature tracking: {e_track}")
                  traceback.print_exc()
                  tracked_data = [(None, None)] * len(self.current_rois)

        # --- 4. Signal Generation ---
        raw_signals = []
        if tracked_data:
            try:
                 raw_signals = self.signal_generator.process_tracked_features(tracked_data)
            except Exception as e_siggen:
                 print(f"[PipelineManager] Error during signal generation: {e_siggen}")
                 traceback.print_exc()
                 raw_signals = [0.0] * len(tracked_data)

        # --- 5. Signal Processing ---
        try:
            self.signal_processor.process_signal_values(raw_signals)
        except Exception as e_sigproc:
             print(f"[PipelineManager] Error during signal processing: {e_sigproc}")
             traceback.print_exc()

        # --- 6. Gather Results ---
        bpm, bpm_valid = self.signal_processor.get_bpm()
        phase = self.signal_processor.get_phase()
        filtered_signal_history = self.signal_processor.get_filtered_signal_buffer()
        raw_signal_history = self.signal_processor.get_raw_signal_buffer()
        peak_indices = self.signal_processor.get_last_peak_indices()

        end_time = time.time()
        processing_time = end_time - start_time

        results = {
            'bpm': bpm,
            'bpm_valid': bpm_valid,
            'phase': phase,
            'current_rois': self.current_rois, # ROI from CoarseRoiCalculator
            'landmarks': self.last_landmarks,
            'filtered_signal_history': filtered_signal_history,
            'raw_signal_history': raw_signal_history,
            'peak_indices': peak_indices,
            'processing_time': processing_time,
            'recalibration_run_this_frame': run_recalibration,
            'recalibration_succeeded': recalibration_succeeded,
            # 'evm_refined_roi': False, # Removed EVM flag
            'frame_count': self.frame_count
        }

        return results

    def close(self):
        """Cleans up resources used by the pipeline components."""
        print("[PipelineManager] Closing pipeline...")
        if hasattr(self, 'pose_detector') and self.pose_detector:
            self.pose_detector.close()
        print("[PipelineManager] Pipeline closed.")

