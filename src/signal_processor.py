# src/signal_processor.py
# Phase 4 & 5: Handles signal fusion, filtering (filtfilt/lfilter/ema), peak detection, BPM & phase calculation.
# MODIFIED: Includes processing for an absolute level signal (drift removal, normalization, adaptive bounds).

import numpy as np
# Make sure scipy is installed: pip install scipy
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, find_peaks
import collections
import time # Potentially useful for timestamping peaks if needed
import math # For inf
import statistics # For median if needed
import traceback

# --- EMAFilter Utility ---
# Placed at module level for use by SignalProcessor
class EMAFilter:
    """Simple Exponential Moving Average Filter."""
    def __init__(self, alpha: float, initial_value: float = None):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha
        self.last_ema = initial_value

    def update(self, value: float) -> float:
        if value is None: # Handle None input if necessary
            return self.last_ema
        if self.last_ema is None:
            self.last_ema = value
        else:
            self.last_ema = self.alpha * value + (1 - self.alpha) * self.last_ema
        return self.last_ema

    def reset(self, initial_value: float = None):
        self.last_ema = initial_value

# --- End EMAFilter Utility ---

class SignalProcessor:
    """
    Processes raw motion signal(s) to calculate respiratory BPM and phase (inhale/exhale).
    Includes steps for signal fusion, filtering (filtfilt, lfilter, or ema),
    peak detection, BPM calculation, and phase estimation.
    Can also process a raw absolute level signal for drift removal and normalization.
    """
    # Define phase constants
    PHASE_UNKNOWN = 0
    PHASE_INHALE = 1
    PHASE_EXHALE = -1

    # New constants for hold phases (derived from level signal slope)
    PHASE_HOLD_INHALE = 2 # Holding after inhale
    PHASE_HOLD_EXHALE = -2 # Holding after exhale
    PHASE_HOLD_FLAT = 3 # Generic hold (e.g., at start or after unknown phase)

    def __init__(self, config=None, sampling_rate=30.0):
        """
        Initializes the SignalProcessor.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'SIGNAL_BUFFER_SECONDS' (float): Duration of signal history for BPM analysis from level signal.
                'PEAK_DETECT_MIN_DISTANCE_SEC' (float): Min time separation between peaks.
                'BPM_AVERAGING_SECONDS' (float): Duration for BPM moving average.
                'PHASE_SLOPE_WINDOW_MS' (int): Window size in milliseconds for phase slope calculation.
                --- New for Level Signal Processing ---
                'PROCESS_LEVEL_SIGNAL_ENABLED' (bool): Enable processing of the level signal.
                'LEVEL_SIGNAL_DRIFT_CORRECTION_ENABLED' (bool): Enable baseline subtraction for drift.
                'LEVEL_SIGNAL_LIGHT_EMA_ALPHA' (float, optional): Alpha for initial light smoothing. None to disable.
                'LEVEL_SIGNAL_BASELINE_EMA_ALPHA' (float): Alpha for the slow EMA to track drift.
                'LEVEL_SIGNAL_NORMALIZATION_WINDOW_SECONDS' (float): Duration for dynamic min/max normalization.
                'LEVEL_SIGNAL_NORMALIZE_TO_MINUS_ONE_ONE' (bool): True for [-1, 1], False for [0, 1].
                'LEVEL_SIGNAL_NORMALIZATION_EPSILON' (float): Small value for safe division.
                'LEVEL_SIGNAL_ADAPTIVE_NORMALIZATION_ENABLED' (bool): Enable adaptive bounds based on clipping.
                'LEVEL_SIGNAL_ADAPTIVE_HEADROOM_FACTOR' (float): Factor to expand bounds on clip.
                'LEVEL_SIGNAL_ADAPTIVE_DECAY_FACTOR' (float): Decay factor for adaptive bounds towards window bounds.
                Defaults are used if config is None or keys are missing.
                --- New for Level Signal BPM/Phase ---
                'PHASE_BPM_LEVEL_SMOOTHING_ENABLED' (bool): Enable EMA smoothing on raw level for BPM/Phase.
                'PHASE_BPM_LEVEL_EMA_ALPHA' (float): Alpha for raw level EMA smoothing for BPM/Phase.
                'PHASE_INHALE_SLOPE_THRESHOLD' (float): Positive slope threshold on raw level derivative for inhale.
                'PHASE_EXHALE_SLOPE_THRESHOLD' (float): Negative slope threshold on raw level derivative for exhale.
                'LEVEL_SIGNAL_BPM_PEAK_PROMINENCE' (float/None): Prominence for peak detection on raw level signal for BPM.
            sampling_rate (float): The rate at which signal values are generated (samples/sec, e.g., video FPS).
        """
        if config is None:
            config = {}

        self.sampling_rate = float(sampling_rate)
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        # --- Configuration Loading ---
        self.buffer_seconds = config.get('SIGNAL_BUFFER_SECONDS', 15.0)

        # Peak detection parameters (now solely for level signal)
        self.peak_min_distance_sec = config.get('PEAK_DETECT_MIN_DISTANCE_SEC', 0.5)
        # Note: PEAK_DETECT_PROMINENCE (old general one) is removed.
        # LEVEL_SIGNAL_BPM_PEAK_PROMINENCE is used for level signal BPM.


        self.bpm_avg_seconds = config.get('BPM_AVERAGING_SECONDS', 4.0) # Default from 5.0 to 4.0
        self.phase_slope_window_ms = config.get('PHASE_SLOPE_WINDOW_MS', 100) # Default 100ms

        # --- Level Signal Processing Configuration & State (for OSC output) ---
        self.process_level_signal_enabled = config.get('PROCESS_LEVEL_SIGNAL_ENABLED', False)
        self.level_light_ema_filter = None
        self.level_baseline_ema_filter = None
        self.level_history_deque = None
        self.processed_level_signal_value = 0.0 # Default value
        self.level_drift_correction_enabled = config.get('LEVEL_SIGNAL_DRIFT_CORRECTION_ENABLED', True) # Default to True if processing level signal

        if self.process_level_signal_enabled: # This check itself might be redundant if 'PROCESS_LEVEL_SIGNAL_ENABLED' is always true
            level_light_ema_alpha = config.get('LEVEL_SIGNAL_LIGHT_EMA_ALPHA', 0.75) # MODIFIED: Default from None to 0.75
            if level_light_ema_alpha is not None and 0.0 < level_light_ema_alpha < 1.0:
                self.level_light_ema_filter = EMAFilter(alpha=level_light_ema_alpha)

            level_baseline_ema_alpha = config.get('LEVEL_SIGNAL_BASELINE_EMA_ALPHA', 0.002)
            self.level_baseline_ema_filter = EMAFilter(alpha=level_baseline_ema_alpha)
            level_norm_window_seconds = config.get('LEVEL_SIGNAL_NORMALIZATION_WINDOW_SECONDS', 25.0) # MODIFIED: Default from 20.0 to 25.0
            self.level_norm_window_size = 1
            if self.sampling_rate > 0: # Ensure sampling_rate is valid
                self.level_norm_window_size = max(1, int(level_norm_window_seconds * self.sampling_rate))
            self.level_history_deque = collections.deque(maxlen=self.level_norm_window_size)
        
        # --- Adaptive Normalization Parameters ---
        self.adaptive_norm_enabled = config.get('LEVEL_SIGNAL_ADAPTIVE_NORMALIZATION_ENABLED', False)
        self.adaptive_headroom_factor = config.get('LEVEL_SIGNAL_ADAPTIVE_HEADROOM_FACTOR', 1.05)
        self.adaptive_decay_factor = config.get('LEVEL_SIGNAL_ADAPTIVE_DECAY_FACTOR', 0.999)

        # --- State Variables for Adaptive Normalization ---
        self.adaptive_raw_max_level = -math.inf # Use math.inf for cleaner initialization
        self.adaptive_raw_min_level = math.inf
        self.no_max_clip_in_previous_cycle = True # Flag for decay logic
        self.no_min_clip_in_previous_cycle = True # Flag for decay logic
        self.level_normalize_to_minus_one_one = config.get('LEVEL_SIGNAL_NORMALIZE_TO_MINUS_ONE_ONE', True)
        self.level_normalization_epsilon = config.get('LEVEL_SIGNAL_NORMALIZATION_EPSILON', 0.01)

        # --- New: Raw Level Signal Processing Configuration & State (for BPM/Phase) ---
        # These settings are now the *only* source for BPM/Phase
        self.phase_bpm_level_smoothing_enabled = config.get('PHASE_BPM_LEVEL_SMOOTHING_ENABLED', False)
        self.phase_bpm_level_ema_filter = None
        if self.phase_bpm_level_smoothing_enabled:
             phase_bpm_level_ema_alpha = config.get('PHASE_BPM_LEVEL_EMA_ALPHA', 0.5)
             if 0.0 < phase_bpm_level_ema_alpha < 1.0:
                 self.phase_bpm_level_ema_filter = EMAFilter(alpha=phase_bpm_level_ema_alpha)
             else:
                 print(f"[SignalProcessor] Warning: Invalid PHASE_BPM_LEVEL_EMA_ALPHA {phase_bpm_level_ema_alpha}. Disabling smoothing for BPM/Phase.")
                 self.phase_bpm_level_smoothing_enabled = False # Disable if alpha is bad

        self.phase_inhale_slope_threshold = config.get('PHASE_INHALE_SLOPE_THRESHOLD', 0.01)
        self.phase_exhale_slope_threshold = config.get('PHASE_EXHALE_SLOPE_THRESHOLD', -0.01)
        self.level_signal_bpm_peak_prominence = config.get('LEVEL_SIGNAL_BPM_PEAK_PROMINENCE', 10.0)
        self.phase_slope_samples_level = max(1, int(self.phase_slope_window_ms / 1000.0 * self.sampling_rate)) # Samples for slope calculation on level
        self._last_movement_phase = self.PHASE_UNKNOWN # State for hold logic

        # --- Calculate derived parameters ---
        self.buffer_size = int(self.buffer_seconds * self.sampling_rate)
        self.peak_distance_samples = int(self.peak_min_distance_sec * self.sampling_rate)
        self.bpm_buffer_size = max(1, int(self.bpm_avg_seconds * self.sampling_rate)) # For averaging calculated BPMs
        # self.phase_slope_samples removed (was for differential signal phase)

        # --- Initialize Buffers ---
        # self.raw_signal_buffer and self.filtered_signal_buffer removed (were for differential signal)
        self.raw_level_signal_for_bpm_buffer = collections.deque(maxlen=self.buffer_size) # For peak detection on level signal
        # Buffers for Raw Level BPM/Phase
        self.raw_level_signal_for_slope_buffer = collections.deque(maxlen=max(2, self.phase_slope_samples_level + 1)) # Need at least 2 for slope
        self.instant_bpm_buffer = collections.deque(maxlen=self.bpm_buffer_size)

        # --- State Variables ---
        self.current_bpm = 0.0
        self.last_peak_indices = [] # Stores peak indices from level signal
        self.bpm_valid = False
        # self._filter_zi removed (was for lfilter on differential)
        self.current_phase = self.PHASE_UNKNOWN

        # --- Filter Design for differential signal removed ---
        # self.filter_b, self.filter_a = None, None

        print(f"[SignalProcessor] Initialized. Level Signal Analysis Buffer for BPM: {self.buffer_size} samples ({self.buffer_seconds}s)")
        # ... (rest of print statements for other params)
        # Removed print for Filtfilt Padding as it's no longer used.
        print(f"  Peak Detection: Min Distance: {self.peak_min_distance_sec}s ({self.peak_distance_samples} samples)")
        print(f"  BPM Averaging Window: {self.bpm_avg_seconds}s ({self.bpm_buffer_size} BPM values)")
        print(f"  Phase Slope Window (Level Signal): {self.phase_slope_window_ms}ms ({self.phase_slope_samples_level} samples)")

        if self.process_level_signal_enabled:
            light_ema_alpha_disp = self.level_light_ema_filter.alpha if self.level_light_ema_filter else "Disabled"
            baseline_ema_alpha_disp = self.level_baseline_ema_filter.alpha if self.level_baseline_ema_filter else "N/A" # Should exist if enabled
            norm_win_sec_disp = self.level_norm_window_size / self.sampling_rate if self.sampling_rate > 0 and self.level_norm_window_size is not None else "N/A"
            drift_corr_status = "Enabled" if self.level_drift_correction_enabled else "Disabled"
            print(f"  Level Signal Processing: Enabled (DriftCorr: {drift_corr_status}, LightEMA: {light_ema_alpha_disp}, BaselineEMA: {baseline_ema_alpha_disp}, NormWin: {norm_win_sec_disp:.1f}s)")
            if self.adaptive_norm_enabled:
                 print(f"    Adaptive Norm: Enabled (Headroom: {self.adaptive_headroom_factor:.2f}, Decay: {self.adaptive_decay_factor:.4f})")


    def process_signal_values(self, raw_level_signal_value=None):
        """
        Processes an optional raw level signal value.
        Updates BPM, phase, and the processed level signal (for OSC output), all derived from the level signal.

        Args:
            raw_level_signal_value (float, optional): The raw absolute level signal value.
        """
        # --- Differential Signal Processing Removed ---

        # --- Start: Raw Level Signal Processing (for BPM, Phase, and OSC output) ---
        # This section now handles both the OSC-bound level signal AND the BPM/Phase calculation

        # Initialize BPM/Phase for this cycle
        current_bpm_value_for_detection = self.current_bpm # Carry over last valid BPM
        current_bpm_valid_for_detection = self.bpm_valid # Carry over last validity
        current_phase_value_for_detection = self.current_phase # Carry over last phase

        if raw_level_signal_value is not None and np.isfinite(raw_level_signal_value):
            # --- 1. Processing for OSC Output (Drift Correction, Normalization, Adaptive Bounds) ---
            # This logic is largely the same as before, using raw_level_signal_value
            
            if self.process_level_signal_enabled:
                current_level_value_for_osc = raw_level_signal_value

                # 1a. Optional Light Smoothing (for OSC path)
                if self.level_light_ema_filter:
                    current_level_value_for_osc = self.level_light_ema_filter.update(current_level_value_for_osc)
                
                # Ensure baseline_ema_filter exists (it should if process_level_signal_enabled is true)
                if self.level_drift_correction_enabled and self.level_baseline_ema_filter is None:
                    print("[SignalProcessor] Error: level_baseline_ema_filter is None despite being enabled.")
                    self.processed_level_signal_value = 0.0 # Fallback
                else:
                    # 1b. Baseline Tracking and Drift Removal (for OSC path)
                    detrended_signal = 0.0
                    if self.level_drift_correction_enabled:
                        current_baseline = self.level_baseline_ema_filter.update(current_level_value_for_osc)
                        if current_baseline is not None:
                            detrended_signal = current_level_value_for_osc - current_baseline
                        # else detrended_signal remains 0.0 if baseline EMA not initialized
                    else:
                        detrended_signal = current_level_value_for_osc # Use the (possibly smoothed) signal directly

                    # 1c. Update History for Dynamic Normalization (for OSC path)
                    # Ensure level_history_deque exists
                    if self.level_history_deque is None: # Should not happen
                         print("[SignalProcessor] Error: level_history_deque is None.")
                         self.processed_level_signal_value = 0.0 # Fallback
                    else:
                        self.level_history_deque.append(detrended_signal)

                        # --- Adaptive Normalization Logic (for OSC path) ---
                        effective_raw_min = 0.0
                        effective_raw_max = 0.0
                        
                        # Get min/max from the rolling window (baseline)
                        if len(self.level_history_deque) > 0: # Should be true after append
                            window_min = np.min(self.level_history_deque)
                            window_max = np.max(self.level_history_deque)
                        else: # Fallback if somehow empty
                            window_min = detrended_signal
                            window_max = detrended_signal

                        if self.adaptive_norm_enabled:
                            # Decay Adaptive Bounds towards Window Bounds (if no clip in previous cycle)
                            # This check happens *before* processing the current sample for clipping
                            if self.no_max_clip_in_previous_cycle and self.adaptive_raw_max_level > window_max:
                                self.adaptive_raw_max_level = window_max + (self.adaptive_raw_max_level - window_max) * self.adaptive_decay_factor
                                self.adaptive_raw_max_level = max(self.adaptive_raw_max_level, window_max) # Ensure it doesn't decay below window_max

                            if self.no_min_clip_in_previous_cycle and self.adaptive_raw_min_level < window_min:
                                self.adaptive_raw_min_level = window_min - (window_min - self.adaptive_raw_min_level) * self.adaptive_decay_factor
                                self.adaptive_raw_min_level = min(self.adaptive_raw_min_level, window_min) # Ensure it doesn't decay above window_min

                            # Determine Effective Normalization Boundaries (wider of adaptive or window)
                            effective_raw_max = max(self.adaptive_raw_max_level, window_max)
                            effective_raw_min = min(self.adaptive_raw_min_level, window_min)
                        else: # Adaptive norm disabled, just use window bounds
                            effective_raw_max = window_max
                            effective_raw_min = window_min

                        # 1d. Dynamic Normalization (for OSC path)
                        # Use the effective_raw_min and effective_raw_max calculated by the adaptive logic
                        if len(self.level_history_deque) > 0: # Ensure deque is not empty (though it should have been appended to)
                            value_range = effective_raw_max - effective_raw_min

                            if value_range < self.level_normalization_epsilon:
                                # Range too small, output neutral value
                                self.processed_level_signal_value = 0.0 if self.level_normalize_to_minus_one_one else 0.5
                            else:
                                normalized_0_1 = (detrended_signal - effective_raw_min) / value_range

                                if self.level_normalize_to_minus_one_one:
                                    normalized_val = 2.0 * normalized_0_1 - 1.0
                                    upper_clip_bound = 1.0
                                    lower_clip_bound = -1.0
                                else:
                                    normalized_val = normalized_0_1
                                    upper_clip_bound = 1.0
                                    lower_clip_bound = 0.0

                                # Detect Clipping and Expand Adaptive Bounds (if enabled)
                                # This check happens *after* normalization using the effective bounds
                                if self.adaptive_norm_enabled:
                                    if normalized_val > upper_clip_bound:
                                        # Expand adaptive max based on the detrended value that caused the clip
                                        self.adaptive_raw_max_level = max(self.adaptive_raw_max_level, detrended_signal * self.adaptive_headroom_factor)
                                        self.no_max_clip_in_previous_cycle = False # A clip occurred on the max side
                                    elif normalized_val < lower_clip_bound:
                                        # Expand adaptive min based on the detrended value that caused the clip
                                        min_candidate = detrended_signal - abs(detrended_signal * (self.adaptive_headroom_factor - 1.0)) # Apply footroom
                                        self.adaptive_raw_min_level = min(self.adaptive_raw_min_level, min_candidate)
                                        self.no_min_clip_in_previous_cycle = False # A clip occurred on the min side

                                # Apply hard clip to the final output value for this frame
                                self.processed_level_signal_value = np.clip(normalized_val, lower_clip_bound, upper_clip_bound)
                        else:
                            # Should not happen if level_history_deque is appended to
                            self.processed_level_signal_value = 0.0 if self.level_normalize_to_minus_one_one else 0.5
            # else: raw_level_signal_value is None, so self.processed_level_signal_value holds its last value
            # This means if level signal input stops, the OSC output for level will freeze at the last valid processed value.
            # --- End: Level Signal Processing (for OSC output) ---

            # --- 2. Processing for BPM and Phase (from Raw Level Signal) ---
            current_raw_level_for_phase_bpm = raw_level_signal_value

            # 2a. Optional Smoothing (dedicated for BPM/Phase path)
            if self.phase_bpm_level_smoothing_enabled and self.phase_bpm_level_ema_filter:
                current_raw_level_for_phase_bpm = self.phase_bpm_level_ema_filter.update(current_raw_level_for_phase_bpm)

            # 2b. Buffer for slope calculation (for Phase)
            self.raw_level_signal_for_slope_buffer.append(current_raw_level_for_phase_bpm)            
            # 2c. Buffer for BPM peak detection
            # This buffer needs to be long enough for BPM calculation (self.buffer_size)
            # We need a separate buffer for BPM calculation on the raw level signal
            # Let's assume a new buffer self.raw_level_signal_for_bpm_buffer exists
            # self.raw_level_signal_for_bpm_buffer.append(current_raw_level_for_phase_bpm) # Assuming this buffer exists and is managed

            # --- Phase Calculation from Raw Level Slope ---
            self.raw_level_signal_for_bpm_buffer.append(current_raw_level_for_phase_bpm) # Populate BPM buffer
            # This logic replaces the old _calculate_phase method
            required_samples_for_phase = self.phase_slope_samples_level + 1
            if len(self.raw_level_signal_for_slope_buffer) >= required_samples_for_phase:
                try:
                    slope_val_current = self.raw_level_signal_for_slope_buffer[-1]
                    slope_val_past = self.raw_level_signal_for_slope_buffer[-required_samples_for_phase]
                    slope = slope_val_current - slope_val_past

                    inh_thresh = self.phase_inhale_slope_threshold
                    exh_thresh = self.phase_exhale_slope_threshold
                    
                    prev_phase_for_hold_logic = current_phase_value_for_detection # Use the phase from the start of this cycle

                    if slope > inh_thresh:
                        current_phase_value_for_detection = self.PHASE_INHALE
                        self._last_movement_phase = self.PHASE_INHALE # Update last movement phase
                    elif slope < exh_thresh:
                        current_phase_value_for_detection = self.PHASE_EXHALE
                        self._last_movement_phase = self.PHASE_EXHALE # Update last movement phase
                    else: # Slope is within hold thresholds
                        # Refine hold phase based on previous movement phase
                        if self._last_movement_phase == self.PHASE_INHALE:
                             current_phase_value_for_detection = self.PHASE_HOLD_INHALE
                        elif self._last_movement_phase == self.PHASE_EXHALE:
                             current_phase_value_for_detection = self.PHASE_HOLD_EXHALE
                        else: # If last movement phase is unknown (e.g., at start)
                             current_phase_value_for_detection = self.PHASE_HOLD_FLAT # Generic hold
                except IndexError:
                     print("[SignalProcessor] Warning: IndexError during raw level phase calculation sample access.")
                     current_phase_value_for_detection = self.PHASE_UNKNOWN # Fallback
            else:
                current_phase_value_for_detection = self.PHASE_UNKNOWN # Not enough data yet

            # --- BPM Calculation from Raw Level Peaks ---
            # This logic replaces the old BPM calculation based on differential signal
            # Need a buffer for raw level signal for BPM calculation
            # Let's assume self.raw_level_signal_for_bpm_buffer exists and is populated above
            
            # Ensure buffer is full enough for reliable peak detection
            if len(self.raw_level_signal_for_bpm_buffer) < self.buffer_size:
                current_bpm_valid_for_detection = False
                self.last_peak_indices = [] # Clear peaks if buffer not full
            else: # Buffer is full
                level_buffer_np = np.array(self.raw_level_signal_for_bpm_buffer)
                try:
                    # Check for near-zero standard deviation to avoid issues with find_peaks on flat signals
                    if np.std(level_buffer_np) < 1e-9: # Threshold for "flat"
                        peaks = np.array([], dtype=int) # No peaks on a flat signal
                    else:
                        # Use the dedicated prominence parameter for level signal BPM
                        level_peak_prominence = self.level_signal_bpm_peak_prominence
                        peaks, properties = find_peaks(
                            level_buffer_np,
                            # height=..., # May not need height if prominence is good
                            distance=self.peak_distance_samples, # Reuse or define new distance for level
                            prominence=level_peak_prominence if level_peak_prominence is not None and level_peak_prominence > 1e-6 else None # Use None if 0 or None
                        )
                    self.last_peak_indices = peaks # Store all found peaks in the current buffer (relative to buffer start)

                    if len(self.last_peak_indices) >= 2:
                        peak_intervals_samples = np.diff(self.last_peak_indices)
                        peak_intervals_sec = peak_intervals_samples / self.sampling_rate
                        
                        # Define reasonable min/max intervals for breathing (can reuse or define new ones)
                        # These might need tuning based on the raw level signal characteristics
                        min_interval = self.peak_min_distance_sec * 0.8 # Example lower bound
                        max_interval = 15.0 # Example upper bound (4 BPM)

                        valid_intervals = peak_intervals_sec[
                            (peak_intervals_sec >= min_interval) & (peak_intervals_sec <= max_interval)
                        ]

                        if len(valid_intervals) > 0:
                            avg_interval_sec = np.mean(valid_intervals)
                            if avg_interval_sec > 0: # Avoid division by zero
                                instant_bpm = 60.0 / avg_interval_sec
                                self.instant_bpm_buffer.append(instant_bpm)
                                if len(self.instant_bpm_buffer) > 0: # Should always be true after append
                                     current_bpm_value_for_detection = np.mean(list(self.instant_bpm_buffer))
                                     current_bpm_valid_for_detection = True
                        else:
                             current_bpm_valid_for_detection = False # No valid intervals found
                except Exception as e:
                    print(f"[SignalProcessor] Error during raw level peak detection/BPM: {e}")
                    traceback.print_exc()
                    self.last_peak_indices = [] # Clear peaks on error
                    current_bpm_valid_for_detection = False # Mark BPM as invalid on error
            # --- End: BPM Calculation from Raw Level Peaks ---

        else: # raw_level_signal_value is None or not finite
            # If raw level signal is missing, BPM/Phase cannot be calculated
            self.bpm_valid = False
            current_bpm_valid_for_detection = False
            # Phase remains its last value or UNKNOWN if at start

        # --- Update master state variables ---
        self.current_phase = current_phase_value_for_detection
        self.current_bpm = current_bpm_value_for_detection
        self.bpm_valid = current_bpm_valid_for_detection

        # --- End: Raw Level Signal Processing (for BPM, Phase, and OSC output) ---

        # --- Duplicated Level Signal Processing block removed ---


    def get_bpm(self):
        # BPM is valid if the calculation was successful AND the level signal BPM buffer is full
        is_currently_valid = self.bpm_valid and len(self.raw_level_signal_for_bpm_buffer) == self.buffer_size
        return self.current_bpm, is_currently_valid

    def get_phase(self):
        # Phase calculation requires a minimum number of samples in the raw_level_signal_for_slope_buffer
        if len(self.raw_level_signal_for_slope_buffer) < self.phase_slope_samples_level + 1:
             return self.PHASE_UNKNOWN
        return self.current_phase # Return the phase calculated in process_signal_values

    def get_last_peak_indices(self):
        # Returns peak indices relative to the start of the current raw_level_signal_for_bpm_buffer
        return self.last_peak_indices


    def get_processed_level_signal(self):
        """Returns the latest processed (drift-removed, normalized) level signal."""
        return self.processed_level_signal_value
    
    def reset(self):
        """Resets all buffers and state variables."""
        # self.raw_signal_buffer.clear() # Removed
        # self.filtered_signal_buffer.clear() # Removed
        self.instant_bpm_buffer.clear()
        self.current_bpm = 0.0
        self.last_peak_indices = [] # Clear peaks
        self.raw_level_signal_for_slope_buffer.clear() # Clear level slope buffer
        if hasattr(self, 'raw_level_signal_for_bpm_buffer'): # Check if initialized
            self.raw_level_signal_for_bpm_buffer.clear() # Clear level BPM buffer
        self.bpm_valid = False
        self.current_phase = self.PHASE_UNKNOWN
        
        # Reset level signal processing state
        if self.process_level_signal_enabled:
            if self.level_light_ema_filter: self.level_light_ema_filter.reset()
            if self.level_baseline_ema_filter: self.level_baseline_ema_filter.reset() # Should exist if enabled
            # No need to reset self.level_drift_correction_enabled as it's a config param
            # Reset adaptive normalization state
            self.adaptive_raw_max_level = -math.inf
            self.adaptive_raw_min_level = math.inf
            self.no_min_clip_in_previous_cycle = True # Reset min clip flag too

            # Reset dedicated EMA filter for raw level BPM/Phase
            if self.phase_bpm_level_ema_filter: self.phase_bpm_level_ema_filter.reset()
            if self.level_history_deque: self.level_history_deque.clear() # Should exist if enabled
            if self.level_history_deque: self.level_history_deque.clear() # Should exist if enabled
            self.processed_level_signal_value = 0.0
        print("[SignalProcessor] State reset.")


# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("\nTesting SignalProcessor module (Level Signal Processing Only)...")

    test_sampling_rate = 50.0
    base_config = {
        'SIGNAL_BUFFER_SECONDS': 10.0,
        'PEAK_DETECT_MIN_DISTANCE_SEC': 0.5, # Min 0.5s between breaths (120 BPM max)
        'BPM_AVERAGING_SECONDS': 4.0,
        'PHASE_SLOPE_WINDOW_MS': 100,
        # --- Config for Level Signal Processing (Example) ---
        'PROCESS_LEVEL_SIGNAL_ENABLED': True,
        'LEVEL_SIGNAL_DRIFT_CORRECTION_ENABLED': True,
        'LEVEL_SIGNAL_LIGHT_EMA_ALPHA': 0.6, 
        'LEVEL_SIGNAL_BASELINE_EMA_ALPHA': 0.005, # Slower baseline for more drift removal
        'LEVEL_SIGNAL_NORMALIZATION_WINDOW_SECONDS': 15.0,
        'LEVEL_SIGNAL_NORMALIZE_TO_MINUS_ONE_ONE': True,
        'LEVEL_SIGNAL_NORMALIZATION_EPSILON': 0.01, # Epsilon for normalization range
        'LEVEL_SIGNAL_ADAPTIVE_NORMALIZATION_ENABLED': True, # Enable adaptive norm for testing (for OSC output)
        'LEVEL_SIGNAL_ADAPTIVE_HEADROOM_FACTOR': 1.1, # Test with 10% headroom
        'LEVEL_SIGNAL_ADAPTIVE_DECAY_FACTOR': 0.995, # Test with a slightly faster decay
        # --- Config for Raw Level BPM/Phase (Example) ---
        'PHASE_BPM_LEVEL_SMOOTHING_ENABLED': True,
        'PHASE_BPM_LEVEL_EMA_ALPHA': 0.5,
        'PHASE_INHALE_SLOPE_THRESHOLD': 0.05, # Example threshold (will need tuning)
        'PHASE_EXHALE_SLOPE_THRESHOLD': -0.05, # Example threshold
        'LEVEL_SIGNAL_ADAPTIVE_HEADROOM_FACTOR': 1.1, # Test with 10% headroom
        'LEVEL_SIGNAL_ADAPTIVE_DECAY_FACTOR': 0.995 # Test with a slightly faster decay
    }

    # --- Generate Mock Signal ---
    duration = 20 # Longer duration for testing drift and normalization
    num_samples = int(duration * test_sampling_rate)
    time_vector = np.linspace(0, duration, num_samples, endpoint=False)
    breathing_freq = 0.25 # Slower breathing for clearer visualization (15 BPM)

    # Mock level signal with drift and different characteristics
    mock_level_signal_raw = 75 + 15 * np.sin(2 * np.pi * breathing_freq * time_vector) + \
                              0.3 * time_vector**1.2 + 2.0 * np.random.randn(num_samples) # Base + Breath + Non-linear Drift + Noise

    print(f"\n--- Testing with Level Signal Processing ---")
    processor = SignalProcessor(config=base_config, sampling_rate=test_sampling_rate)
    
    bpm_history = []
    validity_history = []
    phase_history = []
    processed_level_history = []

    for i in range(num_samples):
        processor.process_signal_values(mock_level_signal_raw[i])
        bpm, is_valid = processor.get_bpm()
        phase = processor.get_phase()
        processed_level_history.append(processor.get_processed_level_signal())
        bpm_history.append(bpm if is_valid else np.nan)
        validity_history.append(is_valid)
        phase_history.append(phase)

    first_valid_idx = next((i for i, valid in enumerate(validity_history) if valid), None)
    valid_bpms = [b for b in bpm_history if not np.isnan(b)]
    avg_valid_bpm = np.mean(valid_bpms) if valid_bpms else 0
    expected_bpm = breathing_freq * 60 # Expected BPM from mock signal

    print(f"  BPM valid around sample: {first_valid_idx} (~{first_valid_idx / test_sampling_rate:.1f}s)")
    print(f"  Avg BPM (when valid): {avg_valid_bpm:.2f} (Expected: {expected_bpm:.2f})")
    
    results = {
        'bpm_history': bpm_history,
        'phase_history': phase_history,
        'processed_level_signal': processed_level_history,
        'avg_bpm': avg_valid_bpm,
        'first_valid_idx': first_valid_idx
    }

    if first_valid_idx is not None: # Only assert if BPM became valid
        if first_valid_idx is not None: # Only assert if BPM became valid
            assert abs(avg_valid_bpm - expected_bpm) < 3.0, f"Avg BPM ({avg_valid_bpm:.2f}) too far from expected ({expected_bpm:.2f})"
        else:
            print(f"  BPM did not become valid during the test.")

    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        print("\nPlotting results (close plot to finish)...")
        # Add more subplots for raw level and processed level signals
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14, 12)) 

        axs[0].plot(time_vector, mock_level_signal_raw, label='Input Raw Level Signal (with drift)', alpha=0.7, color='purple')
        axs[0].set_title("Input Raw Level Signal (with drift)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Processed Level Signal and BPM
        axs[1].plot(time_vector, results['processed_level_signal'], label=f'Processed Level Sig', linestyle='-', color='orange', alpha=0.9)
        axs[1].set_title(f"Processed Level Signal & BPM")
        axs[1].set_ylabel("Processed Level", color='orange')
        axs[1].tick_params(axis='y', labelcolor='orange')
        axs[1].set_ylim(-1.1, 1.1) # Assuming -1 to 1 normalization
        axs[1].legend()
        axs[1].grid(True)
        
        ax_bpm_on_1 = axs[1].twinx()
        ax_bpm_on_1.plot(time_vector, results['bpm_history'], label=f'BPM (Level Source)', marker='.', linestyle=':', color='red', alpha=0.6)
        ax_bpm_on_1.set_ylabel("BPM", color='red')
        ax_bpm_on_1.tick_params(axis='y', labelcolor='red')
        ax_bpm_on_1.set_ylim(0, expected_bpm * 2.5 if expected_bpm > 0 else 60)
        
        # Consolidate legends for axs[1]
        lines1, labels1 = axs[1].get_legend_handles_labels()
        lines_bpm1, labels_bpm1 = ax_bpm_on_1.get_legend_handles_labels()
        axs[1].legend(lines1 + lines_bpm1, labels1 + labels_bpm1, loc='upper left')

        # Plot all phases on a dedicated subplot
        ax_phase = axs[2] 
        ax_phase.plot(time_vector, results['phase_history'], label=f'Phase (Level Source)', marker='.', linestyle='-', drawstyle='steps-post', alpha=0.7)
        ax_phase.set_title("Phase Calculation Over Time")
        ax_phase.set_xlabel("Time (s)")
        ax_phase.set_ylabel("Phase (1=In, -1=Ex)")
        ax_phase.set_yticks([-1, 0, 1])
        ax_phase.set_yticklabels(['Exhale', 'Unknown', 'Inhale'])
        ax_phase.legend(loc='upper right')
        ax_phase.grid(True)

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plots.")
    except Exception as plot_err:
         print(f"\nError during plotting: {plot_err}")
         traceback.print_exc()

    print("\nSignalProcessor module test finished.")
