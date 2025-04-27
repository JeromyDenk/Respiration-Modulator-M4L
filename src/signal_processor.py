# src/signal_processor.py
# Phase 4 & 5: Handles signal fusion, filtering (filtfilt/lfilter), peak detection, BPM & phase calculation.

import numpy as np
# Make sure scipy is installed: pip install scipy
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, find_peaks
import collections
import time # Potentially useful for timestamping peaks if needed
import traceback

class SignalProcessor:
    """
    Processes raw motion signal(s) to calculate respiratory BPM and phase (inhale/exhale).
    Includes steps for signal fusion, filtering (filtfilt or lfilter),
    peak detection, BPM calculation, and phase estimation.
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
                'SIGNAL_FILTER_METHOD' (str): 'filtfilt' (zero-phase) or 'lfilter' (causal). Default 'filtfilt'.
                'SIGNAL_FILTER_TYPE' (str): 'butterworth' (default).
                'SIGNAL_FILTER_ORDER' (int): Order of the filter.
                'SIGNAL_FILTER_LOW_HZ' (float): Lower cutoff frequency (e.g., 0.1 Hz for 6 BPM).
                'SIGNAL_FILTER_HIGH_HZ' (float): Upper cutoff frequency (e.g., 2.0 Hz for 120 BPM). <-- Updated Default
                'PEAK_DETECT_MIN_HEIGHT' (float): Min height for find_peaks.
                'PEAK_DETECT_MIN_DISTANCE_SEC' (float): Min time separation between peaks. <-- Updated Default
                'PEAK_DETECT_PROMINENCE' (float/None): Min prominence for find_peaks. Crucial for noise rejection.
                'BPM_AVERAGING_SECONDS' (float): Duration for BPM moving average.
                'PHASE_SLOPE_WINDOW_MS' (int): Window size in milliseconds for phase slope calculation.
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
        self.filter_type = config.get('SIGNAL_FILTER_TYPE', 'butterworth')
        self.filter_order = config.get('SIGNAL_FILTER_ORDER', 2)
        self.filter_low_hz = config.get('SIGNAL_FILTER_LOW_HZ', 0.1) # ~6 BPM
        self.filter_high_hz = config.get('SIGNAL_FILTER_HIGH_HZ', 2.0) # ~120 BPM (UPDATED DEFAULT)
        self.peak_min_height = config.get('PEAK_DETECT_MIN_HEIGHT', 0.0) # Tune based on filtered signal amplitude
        # Min distance: 0.5s allows up to 120 BPM (matches 2Hz upper filter limit)
        self.peak_min_distance_sec = config.get('PEAK_DETECT_MIN_DISTANCE_SEC', 0.5) # (UPDATED DEFAULT)
        # Prominence: VERY IMPORTANT parameter. Default to None (off).
        # Needs tuning based on signal noise level. Start small (e.g., 0.1 * typical peak height)
        # and increase until noise peaks are rejected. Requires visual inspection of filtered signal + peaks.
        self.peak_prominence = config.get('PEAK_DETECT_PROMINENCE', None)
        self.bpm_avg_seconds = config.get('BPM_AVERAGING_SECONDS', 5.0) # Average BPM over last 5s
        # Phase calculation window (e.g., 100ms)
        self.phase_slope_window_ms = config.get('PHASE_SLOPE_WINDOW_MS', 100)

        # --- Calculate derived parameters ---
        self.buffer_size = int(self.buffer_seconds * self.sampling_rate)
        self.peak_distance_samples = int(self.peak_min_distance_sec * self.sampling_rate)
        self.bpm_buffer_size = int(self.bpm_avg_seconds * self.sampling_rate) # Store recent instant BPMs
        self.phase_slope_samples = max(1, int(self.phase_slope_window_ms / 1000.0 * self.sampling_rate)) # Ensure at least 1 sample window

        # --- Initialize Buffers ---
        self.raw_signal_buffer = collections.deque(maxlen=self.buffer_size)
        self.filtered_signal_buffer = collections.deque(maxlen=self.buffer_size)
        self.peak_indices_buffer = collections.deque(maxlen=self.buffer_size) # Store indices relative to buffer start
        self.instant_bpm_buffer = collections.deque(maxlen=self.bpm_buffer_size)

        # --- State Variables ---
        self.current_bpm = 0.0 # Stores the last calculated valid BPM
        self.last_peak_indices = [] # Store indices found in the most recent valid processing step
        self.bpm_valid = False # Reflects if the *last* calculation attempt was successful
        self._filter_zi = None # Initial state for lfilter
        self.current_phase = self.PHASE_UNKNOWN # Initialize phase state

        # --- Filter Design ---
        nyquist = 0.5 * self.sampling_rate
        low = self.filter_low_hz / nyquist
        high = self.filter_high_hz / nyquist
        low = max(0.001, low) # Avoid zero frequency
        high = min(0.999, high) # Avoid Nyquist frequency

        if low >= high:
             raise ValueError(f"Filter low cutoff ({self.filter_low_hz} Hz) must be less than high cutoff ({self.filter_high_hz} Hz)")

        try:
            self.filter_b, self.filter_a = butter(self.filter_order, [low, high], btype='bandpass')
            print(f"[SignalProcessor] Designed {self.filter_order}-order Butterworth bandpass filter ({self.filter_low_hz:.2f} - {self.filter_high_hz:.2f} Hz).")

            # Initialize filter state if using lfilter
            if self.filter_method == 'lfilter':
                self._filter_zi = lfilter_zi(self.filter_b, self.filter_a)
                print(f"[SignalProcessor] Using 'lfilter' method (causal). Initial state zi calculated.")
            elif self.filter_method == 'filtfilt':
                 print(f"[SignalProcessor] Using 'filtfilt' method (zero-phase).")
            else:
                 print(f"[SignalProcessor] Warning: Unknown filter method '{self.filter_method}'. Defaulting to 'filtfilt'.")
                 self.filter_method = 'filtfilt'

        except Exception as e:
             print(f"[SignalProcessor] FATAL ERROR designing filter: {e}")
             traceback.print_exc()
             self.filter_b, self.filter_a = np.array([1]), np.array([1]) # Dummy filter
             self.filter_method = 'none' # Indicate filter failure


        print(f"[SignalProcessor] Initialized. Analysis Buffer: {self.buffer_size} samples ({self.buffer_seconds}s)")
        print(f"  Peak Min Distance: {self.peak_distance_samples} samples ({self.peak_min_distance_sec}s)")
        print(f"  Peak Prominence: {self.peak_prominence} (Tune this!)")
        print(f"  BPM Avg Window: {self.bpm_buffer_size} samples (~{self.bpm_avg_seconds}s)")
        print(f"  Phase Slope Window: {self.phase_slope_samples} samples (~{self.phase_slope_window_ms}ms)")


    def _calculate_phase(self):
        """Estimates inhale/exhale phase based on recent filtered signal slope."""
        # Need at least window_size + 1 samples to compare current vs past
        required_samples_for_phase = self.phase_slope_samples + 1
        if len(self.filtered_signal_buffer) < required_samples_for_phase:
            return self.PHASE_UNKNOWN # Not enough data

        # Get the required recent samples using negative indexing from the deque
        # Sample at index -1 is the most recent
        # Sample at index -(required_samples_for_phase) is the oldest needed
        try:
            current_sample = self.filtered_signal_buffer[-1]
            past_sample = self.filtered_signal_buffer[-required_samples_for_phase]
        except IndexError:
             # Should not happen if length check passed, but safety first
             print("[SignalProcessor] Warning: IndexError during phase calculation sample access.")
             return self.PHASE_UNKNOWN

        # Calculate the difference (slope approximation over the window)
        slope = current_sample - past_sample

        # Determine phase based on slope sign (add a small tolerance for near-zero slope)
        # Tolerance should be small relative to typical signal changes during inhale/exhale
        slope_tolerance = 1e-7 # Adjust if needed based on signal scale/noise
        if slope > slope_tolerance:
            return self.PHASE_INHALE
        elif slope < -slope_tolerance:
            return self.PHASE_EXHALE
        else:
            # Slope is very close to zero, consider it transition/unknown
            return self.PHASE_UNKNOWN


    def process_signal_values(self, raw_signals_list):
        """
        Processes a list of raw signal values for the current time step.
        Updates BPM and phase.
        """
        # --- 1. Signal Fusion ---
        if not raw_signals_list:
            fused_signal_value = 0.0
        else:
            # Simple fusion for now
            if self.fusion_strategy == 'average':
                 fused_signal_value = np.mean(raw_signals_list)
            elif self.fusion_strategy == 'first':
                 fused_signal_value = raw_signals_list[0]
            else: # Default to 'first'
                 fused_signal_value = raw_signals_list[0]

        self.raw_signal_buffer.append(fused_signal_value)

        # --- 2. Filtering ---
        filtered_value = 0.0 # Default value if filtering fails or not ready
        filter_ran_successfully = False
        if self.filter_method == 'filtfilt':
            # filtfilt needs enough data in the buffer
            min_len_filtfilt = 3 * max(len(self.filter_a), len(self.filter_b))
            if len(self.raw_signal_buffer) >= min_len_filtfilt:
                try:
                    raw_buffer_np = np.array(self.raw_signal_buffer)
                    filtered_signal_full = filtfilt(self.filter_b, self.filter_a, raw_buffer_np)
                    filtered_value = filtered_signal_full[-1] # Get the latest filtered value
                    filter_ran_successfully = True
                except Exception as e:
                    print(f"[SignalProcessor] Error during filtfilt: {e}")
                    # Keep filtered_value as 0.0
            # else: Not enough data yet for filtfilt

        elif self.filter_method == 'lfilter':
            # lfilter processes sample by sample
            try:
                # Ensure _filter_zi is initialized
                if self._filter_zi is None:
                     self._filter_zi = lfilter_zi(self.filter_b, self.filter_a)
                     # Initialize with steady state for the first sample's value
                     self._filter_zi = self._filter_zi * fused_signal_value

                filtered_value, self._filter_zi = lfilter(self.filter_b, self.filter_a, [fused_signal_value], zi=self._filter_zi)
                filtered_value = filtered_value[0] # lfilter returns an array
                filter_ran_successfully = True
            except Exception as e:
                print(f"[SignalProcessor] Error during lfilter: {e}")
                # Reset zi state on error? Or just skip? Skipping for now.
                # Keep filtered_value as 0.0
        # else: Filter method is 'none' or invalid

        self.filtered_signal_buffer.append(filtered_value)

        # --- Phase Calculation ---
        # Calculate phase based on the latest filtered signal history
        # Needs to happen *after* appending the latest filtered value
        self.current_phase = self._calculate_phase()


        # --- Check if buffer is full enough for peak detection/BPM ---
        # We need the filtered buffer to be reasonably full for reliable analysis
        if len(self.filtered_signal_buffer) < self.buffer_size:
            self.bpm_valid = False # Cannot calculate BPM yet
            self.last_peak_indices = []
            return # Exit early if buffer not full for BPM analysis

        # --- 3. Peak Detection & 4. BPM Calculation ---
        current_calculation_valid = False # Reset validity for this specific calculation attempt
        if filter_ran_successfully: # Only proceed if filter ran
            filtered_buffer_np = np.array(self.filtered_signal_buffer)
            try:
                # Check for flat signal before peak detection
                # Use a small threshold for standard deviation
                if np.std(filtered_buffer_np) < 1e-9:
                    # print("[SignalProcessor] Debug: Filtered signal is flat, skipping peak detection.") # Debug noise
                    peaks = np.array([], dtype=int)
                else:
                    peaks, properties = find_peaks(
                        filtered_buffer_np,
                        height=self.peak_min_height if self.peak_min_height > 0 else None,
                        distance=self.peak_distance_samples,
                        prominence=self.peak_prominence if self.peak_prominence is not None else None
                    )
                self.last_peak_indices = peaks # Store indices relative to buffer start

                # --- BPM Calculation ---
                if len(self.last_peak_indices) >= 2:
                    peak_intervals_samples = np.diff(self.last_peak_indices)
                    peak_intervals_sec = peak_intervals_samples / self.sampling_rate

                    # Filter intervals based on expected physiological range derived from filter settings
                    # Allow some margin around filter cutoffs
                    min_interval = (1.0 / self.filter_high_hz) * 0.8 if self.filter_high_hz > 0 else self.peak_min_distance_sec
                    min_interval = max(min_interval, self.peak_min_distance_sec * 0.8) # Ensure it respects peak distance
                    max_interval = (1.0 / self.filter_low_hz) * 1.2 if self.filter_low_hz > 0 else 15.0
                    max_interval = min(max_interval, 15.0) # Absolute max interval (e.g., 4 BPM)

                    valid_intervals = peak_intervals_sec[
                        (peak_intervals_sec >= min_interval) & (peak_intervals_sec <= max_interval)
                    ]

                    if len(valid_intervals) > 0:
                        avg_interval_sec = np.mean(valid_intervals)
                        if avg_interval_sec > 0:
                            instant_bpm = 60.0 / avg_interval_sec
                            self.instant_bpm_buffer.append(instant_bpm)

                            # Update smoothed BPM (using the buffer of recent *valid* instant BPMs)
                            if len(self.instant_bpm_buffer) > 0:
                                 # --- BPM Update ---
                                 self.current_bpm = np.mean(list(self.instant_bpm_buffer))
                                 current_calculation_valid = True # Mark this calculation as successful
                            # else: No valid instant BPMs in buffer yet

                # else: Not enough peaks found in this window

            except Exception as e:
                print(f"[SignalProcessor] Error during peak detection/BPM: {e}")
                traceback.print_exc() # Print full traceback for easier debugging
                self.last_peak_indices = []
                # current_calculation_valid remains False

        # Update overall validity status based on this attempt
        self.bpm_valid = current_calculation_valid
        # Note: self.current_bpm retains its previous value if current_calculation_valid is False


    def get_bpm(self):
        """Returns the current smoothed BPM and its validity status."""
        # Validity here means the *last calculation attempt* was successful
        # AND the buffer is full. The BPM value itself might be from an earlier
        # successful calculation if the latest failed (it freezes).
        is_currently_valid = self.bpm_valid and len(self.filtered_signal_buffer) == self.buffer_size
        return self.current_bpm, is_currently_valid

    def get_phase(self):
        """
        Returns the estimated respiratory phase.

        Returns:
            int: PHASE_INHALE (1), PHASE_EXHALE (-1), or PHASE_UNKNOWN (0).
        """
        # Phase calculation validity depends only on having enough samples in the filtered buffer
        # for the slope calculation, not necessarily on BPM validity.
        if len(self.filtered_signal_buffer) < self.phase_slope_samples + 1:
             return self.PHASE_UNKNOWN
        return self.current_phase

    def get_filtered_signal_buffer(self):
        """Returns the current buffer of filtered signal values."""
        return list(self.filtered_signal_buffer)

    def get_raw_signal_buffer(self):
        """Returns the current buffer of raw (fused) signal values."""
        return list(self.raw_signal_buffer)

    def get_last_peak_indices(self):
        """Returns the indices of peaks found in the last processed buffer."""
        # Return indices relative to the start of the buffer
        return self.last_peak_indices


# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("\nTesting SignalProcessor module (with Phase)...")

    # --- Test Parameters ---
    test_sampling_rate = 50.0 # 50 Hz
    test_config = {
        'SIGNAL_BUFFER_SECONDS': 10.0,
        'SIGNAL_FILTER_METHOD': 'filtfilt', # Test with filtfilt first
        'SIGNAL_FILTER_LOW_HZ': 0.1,  # 6 BPM
        'SIGNAL_FILTER_HIGH_HZ': 2.0, # 120 BPM (Updated upper limit)
        'SIGNAL_FILTER_ORDER': 2,
        'PEAK_DETECT_MIN_DISTANCE_SEC': 0.5, # Min 0.5s separation (Updated)
        'PEAK_DETECT_PROMINENCE': 0.15, # Example prominence value
        'BPM_AVERAGING_SECONDS': 4.0,
        'PHASE_SLOPE_WINDOW_MS': 100, # 100ms window for phase slope
    }
    processor = SignalProcessor(config=test_config, sampling_rate=test_sampling_rate)

    # --- Generate Mock Signal ---
    duration = 15 # Generate 15 seconds of data
    num_samples = int(duration * test_sampling_rate)
    time_vector = np.linspace(0, duration, num_samples, endpoint=False)
    # Simulate breathing at 18 BPM (0.3 Hz)
    breathing_freq = 0.3
    mock_signal = 0.6 * np.sin(2 * np.pi * breathing_freq * time_vector) # Slightly stronger signal
    # Add some noise
    noise = 0.08 * np.random.randn(num_samples) # Slightly less noise
    mock_signal += noise
    # Add a baseline offset
    mock_signal += 1.0

    # --- Process the signal step-by-step ---
    print(f"\nProcessing {duration}s of mock signal ({num_samples} samples)...")
    start_time = time.time()
    bpm_history = []
    validity_history = []
    phase_history = []
    for i in range(num_samples):
        # Simulate receiving one value at a time
        processor.process_signal_values([mock_signal[i]])
        bpm, is_valid = processor.get_bpm()
        phase = processor.get_phase() # Get the calculated phase
        bpm_history.append(bpm if is_valid else np.nan) # Store NaN if not valid
        validity_history.append(is_valid)
        phase_history.append(phase) # Store phase value

    end_time = time.time()
    print(f"Processing finished in {end_time - start_time:.3f} seconds.")

    # --- Analyze Results ---
    # Find the point where BPM becomes valid
    first_valid_idx = next((i for i, valid in enumerate(validity_history) if valid), None)

    print(f"\nBPM calculation became valid around sample index: {first_valid_idx} (after ~{first_valid_idx / test_sampling_rate:.1f}s)")

    # Calculate average BPM during valid period
    valid_bpms = [b for b in bpm_history if not np.isnan(b)]
    avg_valid_bpm = np.mean(valid_bpms) if valid_bpms else 0
    expected_bpm = breathing_freq * 60
    print(f"Average calculated BPM (when valid): {avg_valid_bpm:.2f}")
    print(f"Expected BPM: {expected_bpm:.2f}")

    # Basic Assertions
    assert first_valid_idx is not None, "BPM should become valid"
    assert abs(avg_valid_bpm - expected_bpm) < 3.0, f"Average BPM ({avg_valid_bpm:.2f}) should be close to expected ({expected_bpm:.2f})"

    # Check phase transitions (expecting mostly 1 and -1 after buffer fills)
    # Phase calculation needs fewer samples than BPM, check after phase window fills
    first_phase_calc_idx = processor.phase_slope_samples + 1
    valid_phases = phase_history[first_phase_calc_idx:] if len(phase_history) > first_phase_calc_idx else []
    inhale_count = sum(1 for p in valid_phases if p == SignalProcessor.PHASE_INHALE)
    exhale_count = sum(1 for p in valid_phases if p == SignalProcessor.PHASE_EXHALE)
    unknown_count = sum(1 for p in valid_phases if p == SignalProcessor.PHASE_UNKNOWN)
    print(f"Phase counts (after {first_phase_calc_idx} samples): Inhale={inhale_count}, Exhale={exhale_count}, Unknown={unknown_count}")
    assert inhale_count > 0 and exhale_count > 0, "Should detect both inhale and exhale phases"
    # Expect relatively few unknowns in a clean sine wave
    assert unknown_count < (inhale_count + exhale_count) * 0.2, "Should have relatively few unknown phases in mock signal"


    print("\n--- Checking buffer contents ---")
    raw_buf = processor.get_raw_signal_buffer()
    filtered_buf = processor.get_filtered_signal_buffer()
    peaks = processor.get_last_peak_indices()
    print(f"Raw buffer length: {len(raw_buf)}")
    print(f"Filtered buffer length: {len(filtered_buf)}")
    print(f"Number of peaks found in last buffer: {len(peaks)}")
    # print(f"Peak indices (relative to buffer start): {peaks}") # Can be long

    assert len(raw_buf) == processor.buffer_size, "Raw buffer should be full"
    assert len(filtered_buf) == processor.buffer_size, "Filtered buffer should be full"
    assert len(peaks) > 0, "Should find peaks in the final buffer"


    # --- Optional: Plotting (requires matplotlib) ---
    try:
        import matplotlib.pyplot as plt
        print("\nPlotting results (close plot to finish)...")
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 10)) # Added subplot for phase

        # Plot Filtered Signal (last buffer)
        buffer_time = np.arange(processor.buffer_size) / test_sampling_rate
        axs[0].plot(buffer_time, filtered_buf, label='Filtered Signal')
        # Plot detected peaks on the filtered signal
        if len(peaks) > 0 and max(peaks) < len(filtered_buf): # Ensure peaks are within bounds
             axs[0].plot(buffer_time[peaks], np.array(filtered_buf)[peaks], "x", label="Detected Peaks", color='red', markersize=8)
        else:
             print("[Plotting Warning] Peak indices out of bounds for filtered buffer.")

        axs[0].set_title("Filtered Signal Buffer (Last Window)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[0].grid(True)

        # Plot Full Mock Signal
        axs[1].plot(time_vector, mock_signal, label='Original Mock Signal + Noise', alpha=0.7)
        axs[1].set_title("Full Input Signal")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend()
        axs[1].grid(True)

        # Plot BPM History
        axs[2].plot(time_vector, bpm_history, label='Calculated BPM', marker='.', linestyle='-')
        axs[2].axhline(expected_bpm, color='r', linestyle='--', label=f'Expected BPM ({expected_bpm:.1f})')
        axs[2].set_title("BPM Calculation Over Time")
        axs[2].set_ylabel("BPM")
        axs[2].set_ylim(0, expected_bpm * 2.5) # Adjust Y limits for 120 BPM range
        axs[2].legend()
        axs[2].grid(True)

        # Plot Phase History
        axs[3].plot(time_vector, phase_history, label='Calculated Phase', marker='.', linestyle='-', drawstyle='steps-post')
        axs[3].set_title("Phase Calculation Over Time")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_ylabel("Phase (1=In, -1=Ex)")
        axs[3].set_yticks([-1, 0, 1])
        axs[3].set_yticklabels(['Exhale', 'Unknown', 'Inhale'])
        axs[3].legend()
        axs[3].grid(True)


        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\nMatplotlib not found. Skipping plots.")
        print("Install it with: pip install matplotlib")
    except Exception as plot_err:
         print(f"\nError during plotting: {plot_err}")
         traceback.print_exc()


    print("\nSignalProcessor module test finished.")

