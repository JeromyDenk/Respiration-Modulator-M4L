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
    from signal_generator import SignalGenerator # Assuming this import is correct for the reverted version
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
                # Assuming PHASE_UNKNOWN is available or handle import
                'bpm': 0.0, 'bpm_valid': False, 'phase': 0, # Use 0 if PHASE_UNKNOWN import fails
                'current_rois': [], 'landmarks': None,
                'filtered_signal_history': [], 'raw_signal_history': [],
                'peak_indices': [], 'processing_time': 0,
                # Add default timing structure
                'timing_ms': {
                    'feature_tracker': 0.0, 'signal_generator': 0.0,
                    'signal_processor': 0.0, 'total_pipeline': 0.0
                },
                'recalibration_run_this_frame': False, 'recalibration_succeeded': False,
                'frame_count': self.frame_count, 'tracked_points': None # Add tracked_points key
             }

        start_time = time.time()
        self.frame_count += 1

        # --- Initialize timing variables ---
        t_start_total = time.perf_counter()
        t_feature_tracker = 0.0
        t_signal_gen = 0.0
        t_signal_proc = 0.0
        # --- 1. Input Conversion ---
        try:
            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e_cvt: print(f"[PipelineManager] Error converting frame colorspace: {e_cvt}"); return None

        # --- 2. Feature Tracking ---
        tracked_data = []
        current_tracked_points = None # Store the points tracked in *this* frame
        old_tracked_points_for_signal = None # Store old points for signal generation
        feature_weights_from_tracker = None # To store weights from feature tracker
        status_from_tracker = None      # To store status from feature tracker
        try:
            t_start_ft = time.perf_counter()
            tracked_data = self.feature_tracker.process_frame(image_gray, self.current_rois)
            t_end_ft = time.perf_counter()
            # --- Extract points tracked in this frame for returning ---
            # Assuming tracked_data is a list like [(status, old_points, new_points, feature_weights), ...]
            # We'll process the first item, assuming single ROI focus for this part.
            if tracked_data and tracked_data[0] is not None: # Check if tracker returned data for the first ROI
                 # --- MODIFIED: Make print less verbose ---
                 status_dbg, old_pts_dbg, new_pts_dbg, weights_dbg = tracked_data[0]
                 old_shape_dbg = old_pts_dbg.shape if hasattr(old_pts_dbg, 'shape') else 'None'
                 new_shape_dbg = new_pts_dbg.shape if hasattr(new_pts_dbg, 'shape') else 'None'
                 weights_shape_dbg = weights_dbg.shape if hasattr(weights_dbg, 'shape') else 'None'
                 print(f"[PipelineManager Debug] Feature tracker returned: status='{status_dbg}', "
                       f"old_pts_shape={old_shape_dbg}, new_pts_shape={new_shape_dbg}, "
                       f"weights_shape={weights_shape_dbg}")
                 # --- END MODIFICATION ---
                 # Unpack status, old points, new points, and feature weights
                 status_from_tracker, temp_old_points, temp_new_points, feature_weights_from_tracker = tracked_data[0]
                 if temp_new_points is not None:
                      current_tracked_points = temp_new_points # These are the points for results['tracked_points']
                 if temp_old_points is not None:
                      old_tracked_points_for_signal = temp_old_points
            else:
                print(f"[PipelineManager Debug] Feature tracker did NOT return valid data for ROI 0. tracked_data: {tracked_data}")
            # ---
            t_feature_tracker = (t_end_ft - t_start_ft) * 1000 # Duration in ms
        except Exception as e_track:
             print(f"[PipelineManager] Error during feature tracking: {e_track}"); traceback.print_exc()
             # Adjust default if feature_tracker is expected to return 3 items per ROI
             tracked_data = [(None, None, None, None)] * len(self.current_rois) # Now 4 items

        # --- 3. Signal Generation ---
        raw_signal_value = None # Will store the single raw signal value
        tracked_points_for_signal_gen = None # Points actually used by signal generator

        # Check if we have points and weights from the feature tracker
        print(f"[PipelineManager Debug] Before signal gen: old_points is None? {old_tracked_points_for_signal is None}, new_points (current_tracked_points) is None? {current_tracked_points is None}, feature_weights is None? {feature_weights_from_tracker is None}")
        if old_tracked_points_for_signal is not None and current_tracked_points is not None: # Weights can be None
            print(f"[PipelineManager Debug] Attempting signal generation with old_points_shape {old_tracked_points_for_signal.shape if hasattr(old_tracked_points_for_signal, 'shape') else 'N/A'}, new_points_shape {current_tracked_points.shape if hasattr(current_tracked_points, 'shape') else 'N/A'}, and weights_shape {feature_weights_from_tracker.shape if hasattr(feature_weights_from_tracker, 'shape') else 'N/A'}.")
            try:
                t_start_sg = time.perf_counter()
                # Call the SignalGenerator method that accepts points, ROI, and weights.
                # Assuming generate_signal is the target method and it handles one ROI's data.
                # It should return the raw signal value and the points it effectively used.
                current_roi_for_signal = self.current_rois[0] if self.current_rois else None
                
                raw_signal_value, tracked_points_for_signal_gen = self.signal_generator.generate_signal(
                    old_tracked_points_for_signal, # Pass old points
                    current_tracked_points,
                    current_roi_for_signal,
                    feature_weights_from_tracker
                )
                print(f"[PipelineManager Debug] Signal generator returned: raw_signal_value={raw_signal_value}, tracked_points_for_signal_gen exists? {tracked_points_for_signal_gen is not None}")

                # --- Invert the raw signal ---
                if raw_signal_value is not None:
                    raw_signal_value = -raw_signal_value # Invert the single signal value
                    print(f"[PipelineManager Debug] Inverted raw_signal_value: {raw_signal_value}")

                t_end_sg = time.perf_counter()
                t_signal_gen = (t_end_sg - t_start_sg) * 1000 # Duration in ms
            except Exception as e_siggen: print(f"[PipelineManager] Error during signal generation: {e_siggen}"); traceback.print_exc(); raw_signal_value = None; print("[PipelineManager Debug] raw_signal_value set to None due to exception in signal_generator.")
        else:
            print("[PipelineManager Debug] Skipped signal generation because old_tracked_points_for_signal or current_tracked_points was None.")

        # --- 4. Signal Processing ---
        # SignalProcessor expects a list of raw signal values.
        raw_signals_for_processor = [raw_signal_value] if raw_signal_value is not None else []
        try:
            t_start_sp = time.perf_counter()
            self.signal_processor.process_signal_values(raw_signals_for_processor)
            t_end_sp = time.perf_counter()
            t_signal_proc = (t_end_sp - t_start_sp) * 1000 # Duration in ms
        except Exception as e_sigproc: print(f"[PipelineManager] Error during signal processing: {e_sigproc}"); traceback.print_exc()

        # --- 5. Gather Results ---
        bpm, bpm_valid = self.signal_processor.get_bpm()
        phase = self.signal_processor.get_phase()
        filtered_signal_history = self.signal_processor.get_filtered_signal_buffer()
        raw_signal_history = self.signal_processor.get_raw_signal_buffer()
        peak_indices = self.signal_processor.get_last_peak_indices()
        latest_filtered_value = self.signal_processor.get_latest_filtered_value() # <<< Get latest value

        end_time = time.time()
        processing_time = end_time - start_time
        t_end_total = time.perf_counter()
        t_total = (t_end_total - t_start_total) * 1000 # Total duration in ms

        results = {
            'bpm': bpm, 'bpm_valid': bpm_valid, 'phase': phase,
            'current_rois': self.current_rois, 'landmarks': None,
            'filtered_signal_history': filtered_signal_history,
            'raw_signal_history': raw_signal_history,
            'peak_indices': peak_indices,
            'processing_time': processing_time,
            'latest_filtered_value': latest_filtered_value, # <<< Add value to results

            # --- Add timing results (in milliseconds) ---
            'timing_ms': {
                'feature_tracker': t_feature_tracker,
                'signal_generator': t_signal_gen,
                'signal_processor': t_signal_proc,
                'total_pipeline': t_total
            },
            'recalibration_run_this_frame': False, 'recalibration_succeeded': False,
            'frame_count': self.frame_count,
            'tracked_points': current_tracked_points,    # Points from feature tracker
            'raw_signal': raw_signal_value,              # The (potentially inverted) raw signal value
            'feature_tracker_status': status_from_tracker, # Status from feature tracker
            'feature_weights': feature_weights_from_tracker, # Weights from feature tracker
            'tracked_points_for_signal': tracked_points_for_signal_gen # Points used by signal_generator
        }
        return results

    def close(self):
        # (Close logic remains the same)
        print("[PipelineManager] Closing pipeline...")
        print("[PipelineManager] Pipeline closed.")
