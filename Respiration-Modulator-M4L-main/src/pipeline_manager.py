# src/pipeline_manager.py
# Orchestrates the different processing phases.
# MODIFIED: Integrates EvmProcessor for ROI refinement.

import cv2
import numpy as np
import time
import traceback
import collections # Added for deque

# Import necessary modules from src
try:
    from pose_detector import PoseDetector
    # IMPORTANT: Assumes roi_calculator.py has been renamed to coarse_roi_calculator.py
    # and the class inside renamed to CoarseRoiCalculator
    from coarse_roi_calculator import CoarseRoiCalculator
    from feature_tracker import FeatureTracker
    from signal_generator import SignalGenerator
    from signal_processor import SignalProcessor
    from evm_processor import EvmProcessor # Import the new EVM processor
except ImportError as e:
    print(f"PipelineManager Error: Failed to import sub-modules. "
          f"Ensure they exist and RoiCalculator has been renamed: {e}")
    raise

class PipelineManager:
    """
    Orchestrates the respiration monitoring pipeline, managing state and
    coordinating the processing modules. Includes logic to run expensive
    pose/ROI detection periodically and uses EvmProcessor for ROI refinement.
    """
    def __init__(self, config, sampling_rate):
        """
        Initializes the pipeline manager and all its components.

        Args:
            config (dict): The main configuration dictionary for the application.
                           Expects EVM_* parameters if EVM is enabled.
            sampling_rate (float): The processing rate (e.g., video FPS).
        """
        print("[PipelineManager] Initializing...")
        self.config = config
        self.sampling_rate = sampling_rate
        if self.sampling_rate <= 0:
            raise ValueError("PipelineManager requires a positive sampling_rate.")

        # --- Configuration ---
        self.pose_detection_interval = max(1, config.get('POSE_DETECTION_FRAME_INTERVAL', 1))
        self.recalibration_interval_sec = config.get('PIPELINE_RECALIBRATION_INTERVAL_SEC', 300)
        self.evm_enabled = config.get('EVM_ENABLED', False) # Default EVM off
        self.evm_buffer_seconds = config.get('EVM_BUFFER_SECONDS', 2.0) # How much history for EVM analysis
        self.evm_buffer_size = int(self.evm_buffer_seconds * self.sampling_rate)

        # --- Initialize Processing Modules ---
        try:
            self.pose_detector = PoseDetector(config=self.config)
            # Use the renamed class
            self.coarse_roi_calculator = CoarseRoiCalculator(config=self.config)
            self.feature_tracker = FeatureTracker(config=self.config)
            self.signal_generator = SignalGenerator(config=self.config)
            self.signal_processor = SignalProcessor(config=self.config, sampling_rate=self.sampling_rate)
            # Initialize EVM processor only if enabled
            self.evm_processor = EvmProcessor(config=self.config, sampling_rate=self.sampling_rate) if self.evm_enabled else None
            if self.evm_enabled and self.evm_processor:
                 print("[PipelineManager] EVM Processor Enabled and Initialized.")
            elif self.evm_enabled and not self.evm_processor:
                 print("[PipelineManager] Warning: EVM_ENABLED is true but EvmProcessor failed to initialize.")
                 self.evm_enabled = False # Disable if init failed
            else:
                 print("[PipelineManager] EVM Processor Disabled.")

            print("[PipelineManager] Core processing modules initialized.")
        except Exception as e:
            print(f"[PipelineManager] FATAL ERROR initializing processing modules: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to initialize one or more processing modules.") from e

        # --- Internal State ---
        self.current_rois = [] # This will store the *refined* ROI if EVM is used, otherwise the coarse one
        self.last_landmarks = None
        self.needs_recalibration = True
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_recalibration_time = 0
        # Frame buffer for EVM (stores full grayscale frames)
        self.grayscale_frame_buffer = collections.deque(maxlen=self.evm_buffer_size) if self.evm_enabled else None

        print(f"[PipelineManager] Pose detection interval: {self.pose_detection_interval} frame(s).")
        if self.evm_enabled:
            print(f"[PipelineManager] EVM frame buffer size: {self.evm_buffer_size} frames (~{self.evm_buffer_seconds:.1f}s).")


    def trigger_recalibration(self):
        """Sets a flag to force pose/ROI/EVM recalculation on the next frame."""
        print("[PipelineManager] Recalibration triggered externally.")
        self.needs_recalibration = True

    def _should_run_recalibration(self): # Renamed for clarity
        """Determines if pose/ROI/EVM recalculation should run on the current frame."""
        if self.needs_recalibration: return True
        if not self.current_rois: return True # Always run if no ROI
        if self.pose_detection_interval > 1 and self.frame_count % self.pose_detection_interval == 0: return True
        if self.recalibration_interval_sec > 0 and \
           time.time() - self.last_recalibration_time > self.recalibration_interval_sec:
            print("[PipelineManager] Recalibration interval reached.")
            return True
        return False

    def _get_buffered_coarse_roi_frames(self, coarse_roi):
        """Extracts and crops frames from the buffer for the given coarse ROI."""
        if not self.grayscale_frame_buffer or len(self.grayscale_frame_buffer) < self.evm_processor.min_buffer_frames:
            # print("[PipelineManager] Debug: EVM frame buffer too short.") # Debug noise
            return None

        x, y, w, h = coarse_roi
        cropped_frames = collections.deque(maxlen=self.evm_buffer_size)
        try:
            for frame in self.grayscale_frame_buffer:
                # Ensure ROI coordinates are valid for the frame shape
                frame_h, frame_w = frame.shape
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
                if x1 >= x2 or y1 >= y2:
                    print(f"[PipelineManager] Warning: Coarse ROI {coarse_roi} invalid for frame shape {frame.shape}")
                    return None # Cannot crop
                cropped_frames.append(frame[y1:y2, x1:x2])
            return cropped_frames
        except Exception as e:
            print(f"[PipelineManager] Error cropping frames for EVM buffer: {e}")
            return None


    def process_frame(self, frame):
        """
        Processes a single video frame through the entire pipeline.
        Uses EVM for ROI refinement if enabled.

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

        # --- 1. Input Conversion & Buffering ---
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_rgb.flags.writeable = False
            # Add current frame to EVM buffer if enabled
            if self.evm_enabled and self.grayscale_frame_buffer is not None:
                self.grayscale_frame_buffer.append(image_gray.copy()) # Store copy
        except cv2.error as e_cvt:
             print(f"[PipelineManager] Error converting frame colorspace: {e_cvt}")
             return None
        except Exception as e_buf:
            print(f"[PipelineManager] Error adding frame to buffer: {e_buf}")
            # Continue processing if possible, EVM might fail later

        # --- 2. Recalibration Logic (Pose -> Coarse ROI -> Optional EVM Refinement) ---
        run_recalibration = self._should_run_recalibration()
        recalibration_succeeded = False # Track if ROI was successfully updated (refined or coarse)
        pose_succeeded = False
        reset_tracker_needed = False

        was_forced_recalibration = self.needs_recalibration
        roi_was_missing = not self.current_rois
        new_roi_for_this_cycle = [] # Store the ROI determined in this cycle

        if run_recalibration:
            print(f"[PipelineManager] Running Recalibration (Frame {self.frame_count})...")
            landmarks = self.pose_detector.process_frame(image_rgb)
            if landmarks:
                pose_succeeded = True
                self.last_landmarks = landmarks
                # Calculate the coarse ROI first
                coarse_roi_list = self.coarse_roi_calculator.calculate_coarse_roi(landmarks, (frame_height, frame_width))

                if coarse_roi_list:
                    coarse_roi = coarse_roi_list[0] # Assume single coarse ROI for now
                    print(f"[PipelineManager]   Coarse ROI calculated: {coarse_roi}")

                    # --- EVM Refinement Step (if enabled) ---
                    if self.evm_enabled and self.evm_processor:
                        print("[PipelineManager]   Attempting EVM ROI refinement...")
                        # Get buffered frames cropped to the coarse ROI
                        evm_buffer = self._get_buffered_coarse_roi_frames(coarse_roi)
                        if evm_buffer:
                            refined_roi_list = self.evm_processor.find_optimal_roi(evm_buffer, coarse_roi)
                            if refined_roi_list:
                                print(f"[PipelineManager]   EVM Refinement successful: {refined_roi_list}")
                                new_roi_for_this_cycle = refined_roi_list
                                recalibration_succeeded = True
                            else:
                                print("[PipelineManager]   EVM Refinement failed. Falling back to coarse ROI.")
                                new_roi_for_this_cycle = coarse_roi_list # Fallback
                                recalibration_succeeded = True # Still succeeded in getting *a* ROI
                        else:
                             print("[PipelineManager]   EVM buffer not ready or cropping failed. Falling back to coarse ROI.")
                             new_roi_for_this_cycle = coarse_roi_list # Fallback
                             recalibration_succeeded = True # Still succeeded in getting *a* ROI
                    else:
                        # EVM not enabled, use coarse ROI directly
                        new_roi_for_this_cycle = coarse_roi_list
                        recalibration_succeeded = True

                else: # Coarse ROI calculation failed
                    print("[PipelineManager]   Coarse ROI calculation failed.")
                    # Failure handling depends on why recalibration was triggered
                    if was_forced_recalibration or roi_was_missing:
                        if self.current_rois: reset_tracker_needed = True # Reset if clearing existing ROI
                        self.current_rois = []
                        self.needs_recalibration = True
                    else: # Periodic failure
                         self.needs_recalibration = False # Keep old ROI, wait for next interval

            else: # Pose detection failed
                print("[PipelineManager]   Pose detection failed.")
                self.last_landmarks = None
                if was_forced_recalibration or roi_was_missing:
                    if self.current_rois: reset_tracker_needed = True # Reset if clearing existing ROI
                    self.current_rois = []
                    self.needs_recalibration = True
                else: # Periodic failure
                    self.needs_recalibration = False # Keep old ROI, wait for next interval

            # --- Update State and Check for Tracker Reset ---
            if recalibration_succeeded:
                # Check if tracker needs reset ONLY if forced or recovering from missing ROI
                if was_forced_recalibration or roi_was_missing:
                    # Only reset if the new ROI is actually different from the (potentially empty) old one
                    if self._rois_are_different(new_roi_for_this_cycle, self.current_rois):
                         print(f"[PipelineManager]   Forced recalibration/recovery successful. ROI changed. Resetting tracker.")
                         reset_tracker_needed = True
                    # else: # Forced/recovery ran, but ROI ended up being the same, no reset needed
                    #     print("[PipelineManager] Debug: Forced recalc ran, ROI unchanged.")

                self.current_rois = new_roi_for_this_cycle # Update the current ROI
                self.needs_recalibration = False # Reset trigger
                self.last_recalibration_time = time.time()

            # Reset tracker if flagged
            if reset_tracker_needed:
                print("[PipelineManager] Re-initializing FeatureTracker state.")
                self.feature_tracker = FeatureTracker(config=self.config)


        # --- 3. Feature Tracking ---
        # Uses self.current_rois, which is now the refined ROI if EVM ran, or coarse/old otherwise
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
            'current_rois': self.current_rois, # This is the refined ROI if EVM used
            'landmarks': self.last_landmarks,
            'filtered_signal_history': filtered_signal_history,
            'raw_signal_history': raw_signal_history,
            'peak_indices': peak_indices,
            'processing_time': processing_time,
            'recalibrated_this_frame': recalibration_succeeded, # ROI successfully calculated/checked
            'pose_run_attempted': run_recalibration, # If pose step was entered
            'frame_count': self.frame_count
        }

        return results

    def close(self):
        """Cleans up resources used by the pipeline components."""
        print("[PipelineManager] Closing pipeline...")
        if hasattr(self, 'pose_detector') and self.pose_detector:
            self.pose_detector.close()
        # Add close methods for other components if needed
        print("[PipelineManager] Pipeline closed.")

# (Example usage block would need updating to potentially enable EVM in config)
