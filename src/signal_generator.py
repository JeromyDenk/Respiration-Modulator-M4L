# src/signal_generator.py
# Phase 3: Calculates motion signal(s) from tracked feature flow vectors.
# MODIFIED: Calculates signal based on mean OR median vertical displacement,
#           configurable via 'SIGNAL_AGGREGATION_METHOD'. Defaults to median.

import numpy as np
import traceback
# Optional: PCA related imports removed or commented out
# from sklearn.decomposition import PCA

class SignalGenerator:
    """
    Calculates a raw motion signal for each ROI based on the displacement
    of tracked features, using either the mean or median vertical displacement.
    """
    def __init__(self, config=None):
        """
        Initializes the SignalGenerator.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'SIGNAL_MIN_FEATURES_FOR_SIGNAL' (int): Min features for calculation.
                'SIGNAL_AGGREGATION_METHOD' (str): 'mean' or 'median' (default 'median').
                Defaults are used if config is None or keys are missing.
        """
        if config is None:
            config = {}

        # Minimum number of valid displacement vectors required for calculation
        self.min_features_for_signal = config.get('SIGNAL_MIN_FEATURES_FOR_SIGNAL', 3)

        # Configuration for aggregation method
        self.aggregation_method = config.get('SIGNAL_AGGREGATION_METHOD', 'median').lower()
        if self.aggregation_method not in ['mean', 'median']:
            print(f"[SignalGenerator] Warning: Invalid SIGNAL_AGGREGATION_METHOD '{self.aggregation_method}'. Defaulting to 'median'.")
            self.aggregation_method = 'median'

        # --- NEW: Configuration for IQR Outlier Filtering ---
        self.iqr_filter_enabled = config.get('IQR_FILTER_ENABLED', False)
        self.iqr_k_factor = config.get('IQR_K_FACTOR', 1.5)
        # Ensure k factor is reasonable
        if not isinstance(self.iqr_k_factor, (int, float)) or self.iqr_k_factor <= 0:
            print(f"[SignalGenerator] Warning: Invalid IQR_K_FACTOR '{self.iqr_k_factor}'. Defaulting to 1.5.")
            self.iqr_k_factor = 1.5
        # --- END IQR Config ---

        print(f"[SignalGenerator] Initialized (Using {self.aggregation_method.capitalize()} Vertical Displacement).")
        print(f"  Min Features for Signal Calc: {self.min_features_for_signal}")
        print(f"  Aggregation Method: {self.aggregation_method}")


    # --- PCA calculation method is no longer directly used by default ---
    # Kept here for reference or if you want to add a switch later
    def _calculate_pca_signal_numpy(self, displacements):
        """Calculates signal using PCA manually with NumPy."""
        num_displacements = displacements.shape[0]
        if num_displacements < self.min_features_for_signal:
            return 0.0, "Too few points for PCA"
        try:
            mean_disp = np.mean(displacements, axis=0)
            centered_data = displacements - mean_disp
            if np.allclose(centered_data, 0, atol=1e-6): return 0.0, "No relative motion"
            cov_matrix = np.cov(centered_data, rowvar=False)
            if np.allclose(cov_matrix, 0, atol=1e-6): return 0.0, "Zero covariance"
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            if np.any(np.isnan(eigenvalues)) or np.any(np.isnan(eigenvectors)):
                 print("[SignalGenerator] Warning: NaN encountered in eigenvalues/vectors during PCA.")
                 return 0.0, "NaN in PCA result"
            principal_component = eigenvectors[:, np.argmax(eigenvalues)]
            projection_magnitudes = np.dot(centered_data, principal_component)
            signal_value = np.mean(np.abs(projection_magnitudes))
            return signal_value, f"PCA OK ({num_displacements} pts)"
        except np.linalg.LinAlgError as e_linalg:
            print(f"[SignalGenerator] NumPy PCA Error: {e_linalg}. Displacements shape: {displacements.shape}")
            return 0.0, "LinAlgError"
        except Exception as e:
            print(f"[SignalGenerator] Error during NumPy PCA calculation: {e}")
            traceback.print_exc()
            return 0.0, "Exception in PCA"

    def process_tracked_features(self, tracked_data_all_rois):
        """
        Calculates raw motion signals from tracked feature points for multiple ROIs
        using the configured aggregation method (mean or median) on vertical displacement.

        Args:
            tracked_data_all_rois (list): A list where each element is a tuple
                (good_old_points, good_new_points) from FeatureTracker.
                Points are NumPy arrays of shape (N, 2) or (N, 1, 2).

        Returns:
            list: A list of raw motion signal float values, one for each ROI processed.
                  Returns 0.0 for ROIs where calculation was not possible.
        """
        raw_signals = []
        # processing_summary = [] # Use for less noisy debug output if needed

        for i, (old_points, new_points) in enumerate(tracked_data_all_rois):
            signal_value = 0.0 # Default signal
            reason = "Input None/Empty" # Default reason for zero signal

            # Check if we have valid data for this ROI
            if old_points is not None and new_points is not None and \
               len(old_points) > 0 and len(old_points) == len(new_points):

                # Ensure points are in (N, 2) format
                if old_points.ndim == 3 and old_points.shape[1] == 1:
                    old_points = old_points.reshape(-1, 2)
                if new_points.ndim == 3 and new_points.shape[1] == 1:
                    new_points = new_points.reshape(-1, 2)

                # Check shape again after potential reshape
                if old_points.shape[1] == 2 and new_points.shape[1] == 2:
                    num_points = old_points.shape[0]
                    # Use the potentially renamed config key here
                    if num_points >= self.min_features_for_signal:
                        try:
                            # Calculate displacement vectors (dx, dy)
                            displacements = new_points - old_points # Shape (N, 2)

                            # Calculate vertical displacements (dy)
                            vertical_displacements = displacements[:, 1] # Select only the y-component (index 1)

                            # --- *** NEW: Apply IQR Filter if enabled *** ---
                            filtered_displacements = vertical_displacements # Start with original
                            num_original = len(filtered_displacements)
                            num_filtered = num_original # Initialize in case filter doesn't run
                            if self.iqr_filter_enabled and num_original >= 4: # Need at least 4 points for quartiles
                                try:
                                    q1, q3 = np.percentile(filtered_displacements, [25, 75])
                                    iqr = q3 - q1
                                    # Only filter if IQR is positive to avoid issues with constant data
                                    if iqr > 1e-9: # Use a small epsilon
                                        lower_bound = q1 - self.iqr_k_factor * iqr
                                        upper_bound = q3 + self.iqr_k_factor * iqr
                                        mask = (filtered_displacements >= lower_bound) & (filtered_displacements <= upper_bound)
                                        filtered_displacements = filtered_displacements[mask]
                                        num_filtered = len(filtered_displacements)
                                        # --- Add More Debugging ---
                                        if num_filtered < num_original:
                                             print(f"[IQR Debug Frame {i}] Filtered {num_original - num_filtered} outliers ({num_original} -> {num_filtered}). IQR={iqr:.3f}, Bounds=[{lower_bound:.3f}, {upper_bound:.3f}]")
                                        elif num_filtered == 0:
                                             print(f"[IQR Debug Frame {i}] Filtered ALL points ({num_original} -> 0). IQR={iqr:.3f}, Bounds=[{lower_bound:.3f}, {upper_bound:.3f}]")
                                    # else: # Optional: Print if IQR was too small to filter
                                    #     print(f"[IQR Debug Frame {i}] IQR <= 1e-9 ({iqr:.3f}), not filtering.")

                                except Exception as e_iqr:
                                    print(f"[SignalGenerator] Warning: IQR filter failed for ROI {i}: {e_iqr}")
                                    # Fallback to original displacements on error
                                    filtered_displacements = vertical_displacements
                                    num_filtered = len(filtered_displacements) # Update count

                            # --- *** CORRECTED: Aggregation happens *after* potential filtering *** ---
                            # Check if any points remain *after* filtering
                            if num_filtered > 0:
                                if self.aggregation_method == 'mean':
                                    signal_value = np.mean(filtered_displacements)
                                    reason = f"Mean Vertical Disp ({num_filtered}/{num_original} pts)"
                                else: # Default/Median
                                    signal_value = np.median(filtered_displacements)
                                    reason = f"Median Vertical Disp ({num_filtered}/{num_original} pts)"
                            else:
                                signal_value = 0.0 # No points left after filtering
                                reason = f"No points left after IQR filter ({num_original} -> 0)"

                            # Handle potential NaN if filtering removed all points or input was NaN
                            if np.isnan(signal_value):
                                signal_value = 0.0
                                reason += " (NaN result -> 0.0)"


                        except Exception as e_calc:
                             print(f"[SignalGenerator] Error during {self.aggregation_method.capitalize()} Vertical Displacement calc for ROI {i}: {e_calc}")
                             traceback.print_exc()
                             signal_value = 0.0
                             reason = f"Exception in {self.aggregation_method.capitalize()} Vert Disp"

                    else: # Handle cases where points exist but not enough for calculation
                        reason = f"Too few points ({num_points}/{self.min_features_for_signal})"
                else: # Handle case where reshape failed or shape is wrong
                     reason = "Invalid point shape"
            # else: Handled by default reason "Input None/Empty"

            # --- More Verbose Debugging ---
            if signal_value == 0.0 and reason != "Input None/Empty":
                 print(f"[SignalGenerator Frame Debug] ROI{i}: Signal=0.0, Reason='{reason}'")
            # elif np.random.rand() < 0.02:
            #      print(f"[SignalGenerator Frame Debug] ROI{i}: Signal={signal_value:.4f}, Reason='{reason}'")
            # --- End Debugging ---

            raw_signals.append(signal_value)
            # processing_summary.append(f"ROI{i}:{signal_value:.3f}({reason})")

        # Optional: Print summary less frequently
        # if np.random.rand() < 0.05:
        #    print(f"[SignalGenerator] Debug Summary: {', '.join(processing_summary)}")

        return raw_signals

# Example usage (for testing this module directly)
# NOTE: Assertions updated for the MEDIAN calculation (default)
if __name__ == '__main__':
    print("Testing SignalGenerator module (Mean/Median Vertical Displacement Mode)...")

    # --- Mock Setup ---
    mock_config_median = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'median' # Explicitly median (default)
    }
    mock_config_mean = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'mean' # Explicitly mean
    }
    mock_config_median_iqr = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'median',
        'IQR_FILTER_ENABLED': True, 'IQR_K_FACTOR': 1.5 # Enable IQR
    }
    generator_median = SignalGenerator(config=mock_config_median)
    generator_mean = SignalGenerator(config=mock_config_mean)


    # --- Mock Tracked Data (Same as before) ---
    old1 = np.array([[10, 10], [15, 11], [12, 9], [18, 10]], dtype=np.float32)
    new1 = np.array([[10, 12], [15, 13], [12, 11], [18, 12]], dtype=np.float32) # dy = [2, 2, 2, 2] -> mean=2, median=2
    old2 = np.array([[50, 50], [55, 51], [52, 49], [58, 50], [51, 52]], dtype=np.float32)
    new2 = np.array([[52, 50], [57, 51.5], [54, 49.5], [60, 50], [53, 51.8]], dtype=np.float32) # dy = [0, 0.5, 0.5, 0, -0.2] -> mean=0.16, median=0.0
    old3 = np.array([[100, 100], [105, 101]], dtype=np.float32)
    new3 = np.array([[100, 101], [105, 102]], dtype=np.float32) # dy = [1, 1] -> mean=1, median=1 (but too few points)
    old4, new4 = None, None # -> 0.0
    old5 = np.array([[200, 200], [205, 201], [202, 199]], dtype=np.float32)
    new5 = old5.copy() # dy = [0, 0, 0] -> mean=0, median=0
    old6 = np.array([[300, 300], [305, 301], [302, 299]], dtype=np.float32)
    new6 = old6 + np.array([1, 2], dtype=np.float32) # dy = [2, 2, 2] -> mean=2, median=2
    # --- Data with outliers for IQR test ---
    old7 = np.array([[10, 10], [15, 11], [12, 9], [18, 10], [5, 50], [25, -40]], dtype=np.float32)
    new7 = np.array([[10, 12], [15, 13], [12, 11], [18, 12], [5, 55], [25, -35]], dtype=np.float32) # dy = [2, 2, 2, 2, 5, 5] -> Outliers 5, 5. Median should be 2.

    tracked_data_list = [
        (old1, new1), (old2, new2), (old3, new3), (old4, new4), (old5, new5), (old6, new6),
        (old7, new7) # Add outlier data
    ]

    # --- Test Processing (Median) ---
    print("\n--- Processing Mock Data (Median Aggregation) ---")
    signals_median = generator_median.process_tracked_features(tracked_data_list)
    print(f"Calculated Signals (Median): {signals_median}")
    # Expected Median: [2.0, 0.0, 0.0 (too few points), 0.0, 0.0, 2.0]

    # --- Basic Assertions (Median) ---
    print("\n--- Running Assertions (Median) ---")
    assert len(signals_median) == len(tracked_data_list), "Median: Number of signals should match number of ROIs"
    assert np.isclose(signals_median[0], 2.0), f"Median ROI 1 expected ~2.0, got {signals_median[0]}"
    assert np.isclose(signals_median[1], 0.0), f"Median ROI 2 expected ~0.0, got {signals_median[1]}" # Median is 0.0
    assert signals_median[2] == 0.0, "Median ROI 3 (not enough points) should produce zero signal"
    assert signals_median[3] == 0.0, "Median ROI 4 (no points) should produce zero signal"
    assert signals_median[4] == 0.0, "Median ROI 5 (constant points) should produce zero signal"
    assert np.isclose(signals_median[5], 2.0), f"Median ROI 6 (identical displacements) should produce signal 2.0, got {signals_median[5]}"
    print("--- Median Assertions Passed ---")

    # --- Test Processing (Mean) ---
    print("\n--- Processing Mock Data (Mean Aggregation) ---")
    signals_mean = generator_mean.process_tracked_features(tracked_data_list)
    print(f"Calculated Signals (Mean): {signals_mean}")
    # Expected Mean: [2.0, 0.16, 0.0 (too few points), 0.0, 0.0, 2.0]

    # --- Basic Assertions (Mean) ---
    print("\n--- Running Assertions (Mean) ---")
    assert len(signals_mean) == len(tracked_data_list), "Mean: Number of signals should match number of ROIs"
    assert np.isclose(signals_mean[0], 2.0), f"Mean ROI 1 expected ~2.0, got {signals_mean[0]}"
    assert np.isclose(signals_mean[1], 0.16), f"Mean ROI 2 expected ~0.16, got {signals_mean[1]}" # Mean is 0.16
    assert signals_mean[2] == 0.0, "Mean ROI 3 (not enough points) should produce zero signal"
    assert signals_mean[3] == 0.0, "Mean ROI 4 (no points) should produce zero signal"
    assert signals_mean[4] == 0.0, "Mean ROI 5 (constant points) should produce zero signal"
    assert np.isclose(signals_mean[5], 2.0), f"Mean ROI 6 (identical displacements) should produce signal 2.0, got {signals_mean[5]}"
    print("--- Mean Assertions Passed ---")

    # --- Test Processing (Median with IQR) ---
    print("\n--- Processing Mock Data (Median Aggregation with IQR Filter) ---")
    generator_median_iqr = SignalGenerator(config=mock_config_median_iqr)
    signals_median_iqr = generator_median_iqr.process_tracked_features(tracked_data_list)
    print(f"Calculated Signals (Median w/ IQR): {signals_median_iqr}")
    # Expected Median w/ IQR: [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0 (outliers 5, 5 removed)]
    print("\n--- Running Assertions (Median w/ IQR) ---")
    assert np.isclose(signals_median_iqr[6], 2.0), f"Median ROI 7 (IQR) expected ~2.0 (outliers removed), got {signals_median_iqr[6]}"
    print("--- Median w/ IQR Assertions Passed ---")

    print("\nSignalGenerator module test finished.")
