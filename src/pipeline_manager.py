# src/pipeline_manager.py
# Orchestrates the different processing phases (pose, roi, tracking, signal gen, signal proc).
# Manages state like current ROI and recalibration triggers.
# CORRECTED: Reset FeatureTracker ONLY on manual trigger or when ROI was previously lost/invalid.

import cv2
import numpy as np
import time
import traceback

# Import necessary modules from src
try:
    from pose_detector import PoseDetector
    from roi_calculator import RoiCalculator
    from feature_tracker import FeatureTracker
    from signal_generator import SignalGenerator
    from signal_processor import SignalProcessor
except ImportError as e:
    print(f"PipelineManager Error: Failed to import sub-modules. Ensure they are in the 'src' directory: {e}")
    raise

class PipelineManager:
    """
    Orchestrates the respiration monitoring pipeline, managing state and
    coordinating the processing modules. Includes logic to run expensive
    pose/ROI detection periodically and resets tracking only when necessary.
    """
    def __init__(self, config, sampling_rate):
        """
        Initializes the pipeline manager and all its components.

        Args:
            config (dict): The main configuration dictionary for the application.
                           Expects 'POSE_DETECTION_FRAME_INTERVAL' (int, default 1).
            sampling_rate (float): The processing rate (e.g., video FPS), crucial for SignalProcessor.
        """
        print("[PipelineManager] Initializing...")
        self.config = config
        self.sampling_rate = sampling_rate
        if self.sampling_rate <= 0:
            raise ValueError("PipelineManager requires a positive sampling_rate.")

        # --- Initialize Processing Modules ---
        try:
            self.pose_detector = PoseDetector(config=self.config)
            self.roi_calculator = RoiCalculator(config=self.config)
            self.feature_tracker = FeatureTracker(config=self.config)
            self.signal_generator = SignalGenerator(config=self.config)
            self.signal_processor = SignalProcessor(config=self.config, sampling_rate=self.sampling_rate)
            print("[PipelineManager] All processing modules initialized.")
        except Exception as e:
            print(f"[PipelineManager] FATAL ERROR initializing processing modules: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize one or more processing modules.") from e

        # --- Internal State ---
        self.current_rois = []
        self.last_landmarks = None
        self.needs_recalibration = True # Start by needing calibration/ROI detection
        self.last_frame_time = time.time()
        self.frame_count = 0 # Internal frame counter

        # --- Configuration for Periodic Pose Detection ---
        self.pose_detection_interval = max(1, config.get('POSE_DETECTION_FRAME_INTERVAL', 1))
        self.recalibration_interval_sec = config.get('PIPELINE_RECALIBRATION_INTERVAL_SEC', 300)
        self.last_recalibration_time = 0

        print(f"[PipelineManager] Pose detection interval set to every {self.pose_detection_interval} frame(s).")


    def trigger_recalibration(self):
        """Sets a flag to force pose/ROI recalculation on the next frame."""
        print("[PipelineManager] Recalibration triggered externally.")
        self.needs_recalibration = True

    def _should_run_pose_detection(self):
        """Determines if pose/ROI detection should run on the current frame."""
        if self.needs_recalibration: return True
        if not self.current_rois: return True # Always run if no ROI
        if self.pose_detection_interval > 1 and self.frame_count % self.pose_detection_interval == 0: return True
        if self.recalibration_interval_sec > 0 and \
           time.time() - self.last_recalibration_time > self.recalibration_interval_sec:
            print("[PipelineManager] Recalibration interval reached.")
            return True
        return False

    def _rois_are_different(self, rois1, rois2, tolerance=1):
        """Helper to compare two lists of ROI tuples with a tolerance."""
        if len(rois1) != len(rois2):
            return True
        for r1, r2 in zip(rois1, rois2):
            if abs(r1[0] - r2[0]) > tolerance or \
               abs(r1[1] - r2[1]) > tolerance or \
               abs(r1[2] - r2[2]) > tolerance or \
               abs(r1[3] - r2[3]) > tolerance:
                return True
        return False

    def process_frame(self, frame):
        """
        Processes a single video frame through the entire pipeline.
        Pose/ROI detection may be skipped based on configuration.
        Feature tracker is reset only if ROI changes significantly OR on forced recalibration.

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
            image_rgb.flags.writeable = False
        except cv2.error as e_cvt:
             print(f"[PipelineManager] Error converting frame colorspace: {e_cvt}")
             return None

        # --- 2. Pose/ROI Detection (Conditional) ---
        run_pose_this_frame = self._should_run_pose_detection()
        recalibrated_this_frame = False # Track if ROI was successfully updated
        pose_succeeded = False
        reset_tracker_needed = False # Flag to indicate tracker reset

        # Store the state *before* running pose detection
        was_forced_recalibration = self.needs_recalibration
        roi_was_missing = not self.current_rois

        if run_pose_this_frame:
            landmarks = self.pose_detector.process_frame(image_rgb)
            if landmarks:
                pose_succeeded = True
                self.last_landmarks = landmarks
                new_rois = self.roi_calculator.calculate_rois(landmarks, (frame_height, frame_width))
                if new_rois:
                    # Check if tracker needs reset:
                    # 1. If it was a forced recalibration (manual trigger)
                    # 2. If the ROI was previously missing
                    # 3. (Optional) If the ROI changed drastically (can add this later if needed)
                    if was_forced_recalibration or roi_was_missing:
                         print(f"[PipelineManager] Forced recalibration/ROI recovery successful. Resetting tracker.")
                         reset_tracker_needed = True
                    # --- We no longer reset tracker just because ROI differs slightly on periodic checks ---
                    # elif self._rois_are_different(new_rois, self.current_rois):
                    #     print(f"[PipelineManager] ROI changed significantly. Resetting tracker.") # Keep if large change detection is added
                    #     reset_tracker_needed = True

                    self.current_rois = new_rois # Update ROI regardless of reset
                    self.needs_recalibration = False # Reset trigger ONLY on full success
                    self.last_recalibration_time = time.time()
                    recalibrated_this_frame = True
                else: # ROI calculation failed
                    print("[PipelineManager] Pose detected, but ROI calculation failed.")
                    # If failure happened during forced/initial attempt, clear ROI and force retry
                    if was_forced_recalibration or roi_was_missing:
                        print("[PipelineManager]   -> Forcing retry (clearing ROI).")
                        if self.current_rois: reset_tracker_needed = True # Reset if clearing existing ROI
                        self.current_rois = []
                        self.needs_recalibration = True
                    else: # Failure during periodic check, keep old ROI, don't force retry
                         print("[PipelineManager]   -> Keeping previous ROI, will retry on next interval.")
                         self.needs_recalibration = False
            else: # Pose detection failed
                print("[PipelineManager] Pose detection failed.")
                self.last_landmarks = None
                if was_forced_recalibration or roi_was_missing:
                    print("[PipelineManager]   -> Forcing retry (clearing ROI).")
                    if self.current_rois: reset_tracker_needed = True # Reset if clearing existing ROI
                    self.current_rois = []
                    self.needs_recalibration = True
                else: # Failure during periodic check, keep old ROI, don't force retry
                    print("[PipelineManager]   -> Keeping previous ROI, will retry on next interval.")
                    self.needs_recalibration = False

            # Reset tracker if flagged above
            if reset_tracker_needed:
                print("[PipelineManager] Re-initializing FeatureTracker state.")
                self.feature_tracker = FeatureTracker(config=self.config)

        # --- 3. Feature Tracking ---
        tracked_data = []
        if self.current_rois:
             try:
                 # If tracker was just reset, this detects features. Otherwise, it tracks.
                 tracked_data = self.feature_tracker.process_frame(image_gray, self.current_rois)
             except Exception as e_track:
                  print(f"[PipelineManager] Error during feature tracking: {e_track}")
                  traceback.print_exc()
                  tracked_data = [(None, None)] * len(self.current_rois)
        # else: No ROIs, tracked_data remains empty

        # --- 4. Signal Generation ---
        raw_signals = []
        if tracked_data:
            try:
                 raw_signals = self.signal_generator.process_tracked_features(tracked_data)
            except Exception as e_siggen:
                 print(f"[PipelineManager] Error during signal generation: {e_siggen}")
                 traceback.print_exc()
                 raw_signals = [0.0] * len(tracked_data)
        else:
             raw_signals = [0.0] * len(self.current_rois)


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
            'current_rois': self.current_rois,
            'landmarks': self.last_landmarks,
            'filtered_signal_history': filtered_signal_history,
            'raw_signal_history': raw_signal_history,
            'peak_indices': peak_indices,
            'processing_time': processing_time,
            'recalibrated_this_frame': recalibrated_this_frame, # ROI successfully calculated/checked
            'pose_run_attempted': run_pose_this_frame,
            'frame_count': self.frame_count
        }

        return results

    def close(self):
        """Cleans up resources used by the pipeline components."""
        print("[PipelineManager] Closing pipeline...")
        if hasattr(self, 'pose_detector') and self.pose_detector:
            self.pose_detector.close()
        print("[PipelineManager] Pipeline closed.")

# (Example usage block remains the same conceptually)

