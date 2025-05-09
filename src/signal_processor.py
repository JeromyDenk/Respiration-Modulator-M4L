# src/signal_processor.py
# Phase 4 & 5: Handles signal fusion, filtering (filtfilt/lfilter/ema), peak detection, BPM & phase calculation.
# MODIFIED: Includes processing for an absolute level signal (drift removal, normalization, adaptive bounds).

import numpy as np
# Make sure scipy is installed: pip install scipy
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, find_peaks
import collections
import time # Potentially useful for timestamping peaks if needed
import math # For inf
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

    def __init__(self, config=None, sampling_rate=30.0):
        """
        Initializes the SignalProcessor.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'SIGNAL_BUFFER_SECONDS' (float): Duration of signal history for analysis.
                'SIGNAL_FUSION_STRATEGY' (str): 'first', 'average' (default 'first').
                'SIGNAL_FILTER_METHOD' (str): 'filtfilt', 'lfilter', or 'ema'. Default 'filtfilt'.
                'SIGNAL_FILTER_TYPE' (str): 'butterworth' (default). Only for lfilter/filtfilt.
                'SIGNAL_FILTER_ORDER' (int): Order of the filter. Only for lfilter/filtfilt.
                'SIGNAL_FILTER_LOW_HZ' (float): Lower cutoff frequency. Only for lfilter/filtfilt.
                'SIGNAL_FILTER_HIGH_HZ' (float): Upper cutoff frequency. Only for lfilter/filtfilt.
                'EMA_ALPHA' (float): Smoothing factor for EMA (0 < alpha <= 1). Only for 'ema'.
                'PAD_TYPE' (str): Padding type for filtfilt ('odd', 'even', 'constant', 'gust').
                'PAD_LEN' (int): Padding length for filtfilt (0 for default).
                'PEAK_DETECT_MIN_HEIGHT' (float): Min height for find_peaks.
                'PEAK_DETECT_MIN_DISTANCE_SEC' (float): Min time separation between peaks.
                'PEAK_DETECT_PROMINENCE' (float/None): Min prominence for find_peaks.
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
            sampling_rate (float): The rate at which signal values are generated (samples/sec, e.g., video FPS).
        """
        if config is None:
            config = {}

        self.sampling_rate = float(sampling_rate)
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive.")

        # --- Configuration Loading ---
        self.buffer_seconds = config.get('SIGNAL_BUFFER_SECONDS', 15.0)
        self.fusion_strategy = config.get('SIGNAL_FUSION_STRATEGY', 'first')
        self.filter_method = config.get('SIGNAL_FILTER_METHOD', 'filtfilt').lower()
        
        # Butterworth filter parameters (used if method is 'lfilter' or 'filtfilt')
        self.filter_type = config.get('SIGNAL_FILTER_TYPE', 'butterworth')
        self.filter_order = config.get('SIGNAL_FILTER_ORDER', 2)
        self.filter_low_hz = config.get('SIGNAL_FILTER_LOW_HZ', 0.1)
        self.filter_high_hz = config.get('SIGNAL_FILTER_HIGH_HZ', 2.0)

        # EMA specific parameter
        self.ema_alpha = config.get('EMA_ALPHA', 0.1)
        if not (0 < self.ema_alpha <= 1.0):
            print(f"[SignalProcessor] Warning: Invalid EMA_ALPHA {self.ema_alpha}. Defaulting to 0.1.")
            self.ema_alpha = 0.1
        self._last_ema_value = 0.0 # State for EMA filter
        self._ema_initialized = False # Flag to initialize EMA with the first value

        # Filtfilt specific padding parameters
        self.pad_type = config.get('PAD_TYPE', 'gust')
        self.configured_pad_len = config.get('PAD_LEN', 0)

        self.peak_min_height = config.get('PEAK_DETECT_MIN_HEIGHT', 0.0)
        self.peak_min_distance_sec = config.get('PEAK_DETECT_MIN_DISTANCE_SEC', 0.5)
        self.peak_prominence = config.get('PEAK_DETECT_PROMINENCE', None)
        self.bpm_avg_seconds = config.get('BPM_AVERAGING_SECONDS', 5.0)
        self.phase_slope_window_ms = config.get('PHASE_SLOPE_WINDOW_MS', 100)

        # --- New: Level Signal Processing Configuration & State ---
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


        # --- Calculate derived parameters ---
        self.buffer_size = int(self.buffer_seconds * self.sampling_rate)
        self.peak_distance_samples = int(self.peak_min_distance_sec * self.sampling_rate)
        self.bpm_buffer_size = int(self.bpm_avg_seconds * self.sampling_rate)
        self.phase_slope_samples = max(1, int(self.phase_slope_window_ms / 1000.0 * self.sampling_rate))

        # --- Initialize Buffers ---
        self.raw_signal_buffer = collections.deque(maxlen=self.buffer_size)
        self.filtered_signal_buffer = collections.deque(maxlen=self.buffer_size)
        self.peak_indices_buffer = collections.deque(maxlen=self.buffer_size)
        self.instant_bpm_buffer = collections.deque(maxlen=self.bpm_buffer_size)

        # --- State Variables ---
        self.current_bpm = 0.0
        self.last_peak_indices = []
        self.bpm_valid = False
        self._filter_zi = None # Initial state for lfilter
        self.current_phase = self.PHASE_UNKNOWN

        # --- Filter Design ---
        self.filter_b, self.filter_a = None, None # Initialize to None
        if self.filter_method in ['lfilter', 'filtfilt']:
            nyquist = 0.5 * self.sampling_rate
            low = self.filter_low_hz / nyquist
            high = self.filter_high_hz / nyquist
            low = max(0.001, low) # Ensure low > 0
            high = min(0.999, high) # Ensure high < 1

            if low >= high:
                 print(f"[SignalProcessor] Warning: Filter low cutoff ({self.filter_low_hz} Hz) not less than high cutoff ({self.filter_high_hz} Hz). Using passthrough for lfilter/filtfilt.")
                 self.filter_b, self.filter_a = np.array([1.0]), np.array([1.0]) # Passthrough
            else:
                try:
                    self.filter_b, self.filter_a = butter(self.filter_order, [low, high], btype='bandpass')
                    print(f"[SignalProcessor] Designed {self.filter_order}-order Butterworth bandpass filter ({self.filter_low_hz:.2f} - {self.filter_high_hz:.2f} Hz).")
                    if self.filter_method == 'lfilter':
                        self._filter_zi = lfilter_zi(self.filter_b, self.filter_a) # Calculate initial state for lfilter
                        print(f"[SignalProcessor] Using 'lfilter' method (causal). Initial state zi calculated.")
                    elif self.filter_method == 'filtfilt':
                         print(f"[SignalProcessor] Using 'filtfilt' method (zero-phase).")
                except Exception as e:
                     print(f"[SignalProcessor] ERROR designing Butterworth filter: {e}")
                     traceback.print_exc()
                     # Fallback to passthrough if filter design fails
                     self.filter_b, self.filter_a = np.array([1.0]), np.array([1.0])
        elif self.filter_method == 'ema':
            print(f"[SignalProcessor] Using 'ema' method with alpha={self.ema_alpha:.3f}.")
        else:
            print(f"[SignalProcessor] Warning: Unknown filter method '{self.filter_method}'. No filtering will be applied (passthrough).")
            self.filter_method = 'none' # Explicitly 'none'
            self.filter_b, self.filter_a = np.array([1.0]), np.array([1.0]) # Passthrough

        print(f"[SignalProcessor] Initialized. Analysis Buffer: {self.buffer_size} samples ({self.buffer_seconds}s)")
        # ... (rest of print statements for other params)
        if self.filter_method == 'filtfilt' and self.filter_b is not None: # Check filter_b to ensure design was attempted
            print(f"  Filtfilt Padding: type='{self.pad_type}', configured_len={self.configured_pad_len}")
        if self.process_level_signal_enabled:
            light_ema_alpha_disp = self.level_light_ema_filter.alpha if self.level_light_ema_filter else "Disabled"
            baseline_ema_alpha_disp = self.level_baseline_ema_filter.alpha if self.level_baseline_ema_filter else "N/A" # Should exist if enabled
            norm_win_sec_disp = self.level_norm_window_size / self.sampling_rate if self.sampling_rate > 0 and self.level_norm_window_size is not None else "N/A"
            drift_corr_status = "Enabled" if self.level_drift_correction_enabled else "Disabled"
            print(f"  Level Signal Processing: Enabled (DriftCorr: {drift_corr_status}, LightEMA: {light_ema_alpha_disp}, BaselineEMA: {baseline_ema_alpha_disp}, NormWin: {norm_win_sec_disp:.1f}s)")
            if self.adaptive_norm_enabled:
                 print(f"    Adaptive Norm: Enabled (Headroom: {self.adaptive_headroom_factor:.2f}, Decay: {self.adaptive_decay_factor:.4f})")


    def _calculate_phase(self):
        """Estimates inhale/exhale phase based on recent filtered signal slope."""
        required_samples_for_phase = self.phase_slope_samples + 1
        if len(self.filtered_signal_buffer) < required_samples_for_phase:
            # Not enough data in the *differential* signal buffer for phase calculation
            return self.PHASE_UNKNOWN

        # Phase calculation is based on the filtered *differential* signal
        try:
            current_sample = self.filtered_signal_buffer[-1]
            past_sample = self.filtered_signal_buffer[-required_samples_for_phase]
        except IndexError:
             # This can happen if buffer size changes dynamically or during reset
             print("[SignalProcessor] Warning: IndexError during phase calculation sample access.")
             return self.PHASE_UNKNOWN

        slope = current_sample - past_sample
        # Use a small tolerance to avoid classifying tiny noise as inhale/exhale
        slope_tolerance = 1e-7 # Adjust if needed based on signal magnitude
        if slope > slope_tolerance:
            return self.PHASE_INHALE
        elif slope < -slope_tolerance:
            return self.PHASE_EXHALE
        else:
            return self.PHASE_UNKNOWN


    def process_signal_values(self, raw_signals_list, raw_level_signal_value=None):
        """
        Processes a list of raw differential signal values and an optional raw level signal value.
        Updates BPM and phase (from differential signal) and the processed level signal.

        Args:
            raw_signals_list (list): List of raw differential signal values.
            raw_level_signal_value (float, optional): The raw absolute level signal value.
        """
        # --- Start: Differential Signal Processing (for BPM, Phase) ---
        if not raw_signals_list:
            fused_signal_value = 0.0 # Or handle as error/skip
        else:
            if self.fusion_strategy == 'average':
                 fused_signal_value = np.mean(raw_signals_list)
            else: # Default to 'first'
                 fused_signal_value = raw_signals_list[0]

        self.raw_signal_buffer.append(fused_signal_value)

        # --- Filtering for Differential Signal ---
        filtered_value = 0.0
        filter_ran_successfully = False # For BPM calculation logic

        if self.filter_method == 'filtfilt':
            if self.filter_b is None or self.filter_a is None: # Filter design failed
                filtered_value = fused_signal_value # Passthrough
            else:
                # Determine effective padlen for the check, considering filtfilt's default
                effective_padlen_for_check = self.configured_pad_len
                if self.configured_pad_len == 0: # filtfilt default padlen is 3 * max(len(a), len(b))
                    effective_padlen_for_check = 3 * (max(len(self.filter_b), len(self.filter_a)) - 1)
                    if effective_padlen_for_check < 0: effective_padlen_for_check = 0 # Ensure non-negative
                
                if len(self.raw_signal_buffer) > effective_padlen_for_check:
                    try:
                        raw_buffer_np = np.array(self.raw_signal_buffer)
                        filtered_signal_full = filtfilt(self.filter_b, self.filter_a, raw_buffer_np,
                                                        padtype=self.pad_type, padlen=self.configured_pad_len)
                        filtered_value = filtered_signal_full[-1]
                        filter_ran_successfully = True
                    except ValueError as ve: # Catch specific filtfilt padding errors
                        print(f"[SignalProcessor] ValueError during filtfilt (padtype='{self.pad_type}', padlen={self.configured_pad_len}, "
                              f"segment_len={len(self.raw_signal_buffer)}, effective_check_padlen={effective_padlen_for_check}): {ve}")
                        print("[SignalProcessor] Falling back to lfilter for this segment due to filtfilt ValueError.")
                        # Fallback to lfilter
                        if self._filter_zi is None: self._filter_zi = lfilter_zi(self.filter_b, self.filter_a) * fused_signal_value # Initialize zi
                        filtered_value, self._filter_zi = lfilter(self.filter_b, self.filter_a, [fused_signal_value], zi=self._filter_zi)
                        filtered_value = filtered_value[0]
                        filter_ran_successfully = True # lfilter ran
                    except Exception as e:
                        print(f"[SignalProcessor] Error during filtfilt: {e}. Using raw value.")
                        filtered_value = fused_signal_value # Fallback to raw
                else: # Not enough data for filtfilt, use lfilter as a fallback if possible
                    if self._filter_zi is None: self._filter_zi = lfilter_zi(self.filter_b, self.filter_a) * fused_signal_value # Initialize zi
                    filtered_value, self._filter_zi = lfilter(self.filter_b, self.filter_a, [fused_signal_value], zi=self._filter_zi)
                    filtered_value = filtered_value[0]
                    filter_ran_successfully = True # lfilter ran

        elif self.filter_method == 'lfilter':
            if self.filter_b is None or self.filter_a is None: # Filter design failed
                filtered_value = fused_signal_value # Passthrough
            else:
                try:
                    # Initialize zi state if it's None (first run or after reset)
                    if self._filter_zi is None:
                         self._filter_zi = lfilter_zi(self.filter_b, self.filter_a)
                         # Scale initial state by the first input value to avoid large transient
                         self._filter_zi = self._filter_zi * fused_signal_value
                    filtered_value, self._filter_zi = lfilter(self.filter_b, self.filter_a, [fused_signal_value], zi=self._filter_zi)
                    filtered_value = filtered_value[0] # lfilter returns an array
                    filter_ran_successfully = True
                except Exception as e:
                    print(f"[SignalProcessor] Error during lfilter: {e}")
                    filtered_value = fused_signal_value # Fallback to raw
        
        elif self.filter_method == 'ema':
            if not self._ema_initialized or not self.filtered_signal_buffer: # If first point or buffer was cleared
                self._last_ema_value = fused_signal_value
                self._ema_initialized = True
            else:
                # Use the previous EMA value for calculation
                self._last_ema_value = self.ema_alpha * fused_signal_value + (1 - self.ema_alpha) * self._last_ema_value
            filtered_value = self._last_ema_value
            filter_ran_successfully = True
        
        else: # 'none' or unknown method
            filtered_value = fused_signal_value # Passthrough
            filter_ran_successfully = True # Considered "successful" as it's intentional

        self.filtered_signal_buffer.append(filtered_value)
        self.current_phase = self._calculate_phase() # Phase based on filtered differential signal

        # --- BPM Calculation (based on filtered differential signal) ---
        if len(self.filtered_signal_buffer) < self.buffer_size: # Need full buffer for reliable peak detection
            self.bpm_valid = False
            self.last_peak_indices = [] # Clear peaks if buffer not full
            # return # Don't return early, still need to process level signal if enabled
        else: # Buffer is full
            current_calculation_valid = False # Assume invalid until proven
            if filter_ran_successfully: # Only proceed if filtering was successful
                filtered_buffer_np = np.array(self.filtered_signal_buffer)
                try:
                    # Check for near-zero standard deviation to avoid issues with find_peaks on flat signals
                    if np.std(filtered_buffer_np) < 1e-9: # Threshold for "flat"
                        peaks = np.array([], dtype=int) # No peaks on a flat signal
                    else:
                        peaks, properties = find_peaks(
                            filtered_buffer_np,
                            height=self.peak_min_height if self.peak_min_height > 0 else None, # Use None if min_height is 0 or less
                            distance=self.peak_distance_samples,
                            prominence=self.peak_prominence if self.peak_prominence is not None else None # Use None if not set
                        )
                    self.last_peak_indices = peaks # Store all found peaks in the current buffer

                    if len(self.last_peak_indices) >= 2:
                        peak_intervals_samples = np.diff(self.last_peak_indices)
                        peak_intervals_sec = peak_intervals_samples / self.sampling_rate
                        
                        # Define reasonable min/max intervals for breathing
                        min_interval_filt = (1.0 / self.filter_high_hz) * 0.8 if self.filter_method in ['lfilter', 'filtfilt'] and self.filter_high_hz > 0 else self.peak_min_distance_sec
                        min_interval = max(min_interval_filt, self.peak_min_distance_sec * 0.8) # Robust min
                        max_interval_filt = (1.0 / self.filter_low_hz) * 1.2 if self.filter_method in ['lfilter', 'filtfilt'] and self.filter_low_hz > 0 else 15.0 # e.g., 4 BPM
                        max_interval = min(max_interval_filt, 15.0) # Robust max (4 BPM)


                        valid_intervals = peak_intervals_sec[
                            (peak_intervals_sec >= min_interval) & (peak_intervals_sec <= max_interval)
                        ]

                        if len(valid_intervals) > 0:
                            avg_interval_sec = np.mean(valid_intervals)
                            if avg_interval_sec > 0: # Avoid division by zero
                                instant_bpm = 60.0 / avg_interval_sec
                                self.instant_bpm_buffer.append(instant_bpm)
                                if len(self.instant_bpm_buffer) > 0: # Should always be true after append
                                     self.current_bpm = np.mean(list(self.instant_bpm_buffer))
                                     current_calculation_valid = True
                except Exception as e:
                    print(f"[SignalProcessor] Error during peak detection/BPM: {e}")
                    traceback.print_exc()
                    self.last_peak_indices = [] # Clear peaks on error
            self.bpm_valid = current_calculation_valid
        # --- End: Differential Signal Processing ---

        # --- Start: Level Signal Processing (for direct OSC output) ---
        if self.process_level_signal_enabled:
            if raw_level_signal_value is not None:
                current_level_value = raw_level_signal_value

                # 1. Optional Light Smoothing
                if self.level_light_ema_filter:
                    current_level_value = self.level_light_ema_filter.update(current_level_value)
                
                if current_level_value is None: # Should only happen if raw_level_signal_value was None and no EMA
                    # self.processed_level_signal_value remains unchanged (holds last value)
                    return # Skip further level processing for this frame
                
                # Ensure baseline_ema_filter exists (it should if process_level_signal_enabled is true)
                if self.level_drift_correction_enabled and self.level_baseline_ema_filter is None:
                    print("[SignalProcessor] Error: level_baseline_ema_filter is None despite being enabled.")
                    self.processed_level_signal_value = 0.0 # Fallback
                    return

                # 2. Baseline Tracking and Drift Removal
                detrended_signal = 0.0
                if self.level_drift_correction_enabled:
                    current_baseline = self.level_baseline_ema_filter.update(current_level_value)
                    if current_baseline is not None:
                        detrended_signal = current_level_value - current_baseline
                    # else detrended_signal remains 0.0 if baseline EMA not initialized
                else:
                    detrended_signal = current_level_value # Use the (possibly smoothed) signal directly

                # 3. Update History for Dynamic Normalization
                # Ensure level_history_deque exists
                if self.level_history_deque is None: # Should not happen
                     print("[SignalProcessor] Error: level_history_deque is None.")
                     self.processed_level_signal_value = 0.0 # Fallback
                     return
                self.level_history_deque.append(detrended_signal)

                # --- Adaptive Normalization Logic ---
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

                # 4. Dynamic Normalization
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
        # --- End: Level Signal Processing ---


    def get_bpm(self):
        # BPM is valid if the calculation was successful AND the buffer is full (implying stable processing)
        is_currently_valid = self.bpm_valid and len(self.filtered_signal_buffer) == self.buffer_size
        return self.current_bpm, is_currently_valid

    def get_phase(self):
        # Phase calculation requires a minimum number of samples in the filtered_signal_buffer
        if len(self.filtered_signal_buffer) < self.phase_slope_samples + 1:
             return self.PHASE_UNKNOWN
        return self.current_phase

    def get_filtered_signal_buffer(self):
        return list(self.filtered_signal_buffer)

    def get_raw_signal_buffer(self):
        return list(self.raw_signal_buffer)

    def get_last_peak_indices(self):
        # Returns peak indices relative to the start of the current filtered_signal_buffer
        return self.last_peak_indices

    def get_latest_filtered_value(self):
            # Returns the most recent value from the filtered_signal_buffer (differential signal path)
            if self.filtered_signal_buffer:
                return self.filtered_signal_buffer[-1]
            else:
                return 0.0

    def get_processed_level_signal(self):
        """Returns the latest processed (drift-removed, normalized) level signal."""
        return self.processed_level_signal_value
    
    def reset(self):
        """Resets all buffers and state variables."""
        self.raw_signal_buffer.clear()
        self.filtered_signal_buffer.clear()
        self.peak_indices_buffer.clear() # Though not directly used for output, good to clear
        self.instant_bpm_buffer.clear()
        self.current_bpm = 0.0
        self.bpm_valid = False
        self.current_phase = self.PHASE_UNKNOWN
        self.last_peak_indices = []
        if self.filter_method == 'lfilter' and self.filter_b is not None and self.filter_a is not None:
            # Re-initialize zi state for lfilter
            self._filter_zi = lfilter_zi(self.filter_b, self.filter_a)
            # Note: zi state will be scaled by the first signal value upon next processing
        elif self.filter_method == 'ema':
            self._last_ema_value = 0.0 # Reset EMA state
            self._ema_initialized = False
        
        # Reset level signal processing state
        if self.process_level_signal_enabled:
            if self.level_light_ema_filter: self.level_light_ema_filter.reset()
            if self.level_baseline_ema_filter: self.level_baseline_ema_filter.reset() # Should exist if enabled
            # No need to reset self.level_drift_correction_enabled as it's a config param
            # Reset adaptive normalization state
            self.adaptive_raw_max_level = -math.inf
            self.adaptive_raw_min_level = math.inf
            self.no_max_clip_in_previous_cycle = True
            if self.level_history_deque: self.level_history_deque.clear() # Should exist if enabled
            if self.level_history_deque: self.level_history_deque.clear() # Should exist if enabled
            self.processed_level_signal_value = 0.0
        print("[SignalProcessor] State reset.")


# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("\nTesting SignalProcessor module (with Phase, EMA, and Level Signal Processing)...")

    test_sampling_rate = 50.0
    base_config = {
        'SIGNAL_BUFFER_SECONDS': 10.0,
        'SIGNAL_FILTER_LOW_HZ': 0.1,
        'SIGNAL_FILTER_HIGH_HZ': 2.0, # Adjusted for typical breathing
        'SIGNAL_FILTER_ORDER': 2,
        'PEAK_DETECT_MIN_DISTANCE_SEC': 0.5, # Min 0.5s between breaths (120 BPM max)
        'PEAK_DETECT_PROMINENCE': 0.1, # Adjusted prominence
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
        'LEVEL_SIGNAL_ADAPTIVE_NORMALIZATION_ENABLED': True, # Enable adaptive norm for testing
        'LEVEL_SIGNAL_ADAPTIVE_HEADROOM_FACTOR': 1.1, # Test with 10% headroom
        'LEVEL_SIGNAL_ADAPTIVE_DECAY_FACTOR': 0.995 # Test with a slightly faster decay
    }

    # --- Generate Mock Signal ---
    duration = 20 # Longer duration for testing drift and normalization
    num_samples = int(duration * test_sampling_rate)
    time_vector = np.linspace(0, duration, num_samples, endpoint=False)
    breathing_freq = 0.25 # Slower breathing for clearer visualization (15 BPM)
    mock_signal_clean = 0.8 * np.sin(2 * np.pi * breathing_freq * time_vector) # Differential signal
    noise_diff = 0.1 * np.random.randn(num_samples)
    mock_signal_noisy_diff = mock_signal_clean + noise_diff + 0.5 # Add DC offset to differential

    # Mock level signal with drift and different characteristics
    mock_level_signal_raw = 75 + 15 * np.sin(2 * np.pi * breathing_freq * time_vector) + \
                              0.3 * time_vector**1.2 + 2.0 * np.random.randn(num_samples) # Base + Breath + Non-linear Drift + Noise

    filter_methods_to_test = ['filtfilt', 'lfilter', 'ema']
    all_results = {}

    for method in filter_methods_to_test:
        print(f"\n--- Testing with Filter Method: {method.upper()} ---")
        current_config = {**base_config, 'SIGNAL_FILTER_METHOD': method}
        if method == 'ema':
            current_config['EMA_ALPHA'] = 0.08 # Test with a specific alpha for EMA
            # For EMA, bandpass parameters are not used, but peak detection might still be affected by smoothing
            # current_config['PEAK_DETECT_PROMINENCE'] = 0.05 # Example: potentially lower prominence for EMA

        processor = SignalProcessor(config=current_config, sampling_rate=test_sampling_rate)
        
        bpm_history = []
        validity_history = []
        phase_history = []
        filtered_output_history = []
        processed_level_history = []

        for i in range(num_samples):
            # Pass both differential and level signals
            processor.process_signal_values([mock_signal_noisy_diff[i]], mock_level_signal_raw[i])
            bpm, is_valid = processor.get_bpm()
            phase = processor.get_phase()
            filtered_output_history.append(processor.get_latest_filtered_value())
            processed_level_history.append(processor.get_processed_level_signal())
            bpm_history.append(bpm if is_valid else np.nan)
            validity_history.append(is_valid)
            phase_history.append(phase)

        first_valid_idx = next((i for i, valid in enumerate(validity_history) if valid), None)
        valid_bpms = [b for b in bpm_history if not np.isnan(b)]
        avg_valid_bpm = np.mean(valid_bpms) if valid_bpms else 0
        expected_bpm = breathing_freq * 60

        print(f"  BPM valid around sample: {first_valid_idx} (~{first_valid_idx / test_sampling_rate:.1f}s)")
        print(f"  Avg BPM (when valid): {avg_valid_bpm:.2f} (Expected: {expected_bpm:.2f})")
        
        all_results[method] = {
            'bpm_history': bpm_history,
            'phase_history': phase_history,
            'filtered_signal': filtered_output_history, # Store the full filtered output
            'processed_level_signal': processed_level_history,
            'avg_bpm': avg_valid_bpm,
            'first_valid_idx': first_valid_idx
        }
        if method != 'ema': # EMA doesn't rely on traditional peak finding for its primary output
             if first_valid_idx is not None: # Only assert if BPM became valid
                assert abs(avg_valid_bpm - expected_bpm) < 3.0, f"{method}: Avg BPM ({avg_valid_bpm:.2f}) too far from expected ({expected_bpm:.2f})"
             else:
                print(f"  {method}: BPM did not become valid during the test.")
        else: 
            pass 

    # --- Specific Test for EMA Step Response ---
    print("\n--- Testing EMA Step Response (Differential Path) ---")
    ema_config_step = {**base_config, 
                       'SIGNAL_FILTER_METHOD': 'ema', 
                       'EMA_ALPHA': 0.1,
                       'PROCESS_LEVEL_SIGNAL_ENABLED': False} # Disable level for this specific test
    processor_ema_step = SignalProcessor(config=ema_config_step, sampling_rate=test_sampling_rate)
    ema_step_output_history = []
    step_input_signal = np.concatenate([np.zeros(50), np.ones(50), np.zeros(50) + 0.5])
    for val in step_input_signal: # Only testing differential path here for EMA step
        processor_ema_step.process_signal_values([val]) # No level signal passed
        ema_step_output_history.append(processor_ema_step.get_latest_filtered_value())
    
    print(f"EMA Step Test: Last raw input: {step_input_signal[-1]}, Last EMA output: {ema_step_output_history[-1]:.3f}")
    assert np.abs(ema_step_output_history[-1] - step_input_signal[-1]) < np.abs(ema_step_output_history[99] - step_input_signal[-1]), \
        "EMA should be closer to the final step value at the end than after the previous step."
    print("EMA Step Response test passed basic check.")


    # --- Plotting ---
    try:
        import matplotlib.pyplot as plt
        print("\nPlotting results (close plot to finish)...")
        num_methods = len(filter_methods_to_test)
        # Add more subplots for raw level and processed level signals
        fig, axs = plt.subplots(num_methods + 3, 1, sharex=True, figsize=(14, 6 + num_methods * 3.5)) 

        axs[0].plot(time_vector, mock_signal_noisy_diff, label='Input Differential Signal (Noisy)', alpha=0.7, color='gray')
        axs[0].set_title("Input Differential Signal (for BPM/Phase)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(time_vector, mock_level_signal_raw, label='Input Raw Level Signal (with drift)', alpha=0.7, color='purple')
        axs[1].set_title("Input Raw Level Signal (with drift)")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend()
        axs[1].grid(True)

        for i, method in enumerate(filter_methods_to_test):
            ax_idx = i + 2 # Offset by 2 due to the two input plots
            results = all_results[method]
            
            # Plot Filtered Differential Signal
            axs[ax_idx].plot(time_vector, results['filtered_signal'], label=f'Filtered Diff Sig ({method.upper()})', color='blue')
            axs[ax_idx].set_title(f"Diff Signal ({method.upper()}) & BPM | Level Signal (Processed)")
            axs[ax_idx].set_ylabel("Diff Sig Amp", color='blue')
            axs[ax_idx].tick_params(axis='y', labelcolor='blue')
            axs[ax_idx].grid(True)
            
            # Plot BPM on a twin axis for the differential signal plot
            ax_bpm = axs[ax_idx].twinx()
            ax_bpm.plot(time_vector, results['bpm_history'], label=f'BPM ({method})', marker='.', linestyle=':', color='red', alpha=0.6)
            ax_bpm.set_ylabel("BPM", color='red')
            ax_bpm.tick_params(axis='y', labelcolor='red')
            ax_bpm.set_ylim(0, expected_bpm * 2.5 if expected_bpm > 0 else 60) # Handle expected_bpm=0
            
            # Plot Processed Level Signal on another twin axis
            ax_level_processed = axs[ax_idx].twinx()
            ax_level_processed.spines["right"].set_position(("outward", 60)) # Offset the new y-axis
            ax_level_processed.plot(time_vector, results['processed_level_signal'], label=f'Processed Level Sig', linestyle='--', color='orange', alpha=0.9)
            ax_level_processed.set_ylabel("Processed Level", color='orange')
            ax_level_processed.tick_params(axis='y', labelcolor='orange')
            ax_level_processed.set_ylim(-1.1, 1.1) # Assuming -1 to 1 normalization for level signal

            # Consolidate legends
            lines, labels = axs[ax_idx].get_legend_handles_labels()
            lines_bpm, labels_bpm = ax_bpm.get_legend_handles_labels()
            lines_level, labels_level = ax_level_processed.get_legend_handles_labels()
            axs[ax_idx].legend(lines + lines_bpm + lines_level, labels + labels_bpm + labels_level, loc='upper left')


        # Plot all phases on a dedicated subplot
        ax_phase = axs[-1] 
        for method in filter_methods_to_test:
            results = all_results[method]
            ax_phase.plot(time_vector, results['phase_history'], label=f'Phase ({method})', marker='.', linestyle='-', drawstyle='steps-post', alpha=0.7)
        
        ax_phase.set_title("Phase Calculation Over Time (All Methods)")
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
