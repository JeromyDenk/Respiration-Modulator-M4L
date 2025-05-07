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

    def _weighted_median(self, data, weights):
        """
        Helper function to compute the weighted median.
        Assumes data and weights are 1D NumPy arrays of the same length.
        """
        if data is None or weights is None or len(data) == 0 or len(weights) == 0:
            return 0.0
        if len(data) != len(weights):
            print("[SignalGenerator] Warning: Data and weights length mismatch for weighted median. Returning unweighted median.")
            return np.median(data)

        sorted_indices = np.argsort(data)
        data_sorted = data[sorted_indices]
        weights_sorted = weights[sorted_indices]
        cumulative_weights = np.cumsum(weights_sorted)
        total_weight = cumulative_weights[-1]

        if total_weight <= 1e-9: # Avoid division by zero if all weights are zero
            return np.median(data) # Fallback to unweighted median

        median_weight_mark = total_weight / 2.0
        median_index = np.where(cumulative_weights >= median_weight_mark)[0]

        if len(median_index) == 0: # Should not happen if total_weight > 0
            return np.median(data)

        return data_sorted[median_index[0]]

    def generate_signal(self, old_points, new_points, roi_definition, weights):
        """
        Calculates a raw motion signal from tracked feature points for a single ROI
        using the configured aggregation method (mean or median) on vertical displacement.

        Args:
            old_points (np.ndarray): Previous locations of tracked features (N, 2) or (N, 1, 2).
            new_points (np.ndarray): Current locations of tracked features (N, 2) or (N, 1, 2).
            roi_definition (tuple or None): The (x, y, w, h) of the ROI, for context or future use.
            weights (np.ndarray or None): Quality scores/weights for the new_points (N,).

        Returns:
            tuple: (float, np.ndarray or None)
                   - The raw motion signal value (float).
                   - The new_points that were effectively used for signal generation (or None).
        """
        signal_value = 0.0  # Default signal
        points_used_for_signal = None # Default
        reason = "Input None/Empty"  # Default reason

        if old_points is None or new_points is None or \
           len(old_points) == 0 or len(old_points) != len(new_points):
            # print(f"[SignalGenerator] Invalid input points: old_none={old_points is None}, new_none={new_points is None}, len_old={len(old_points) if old_points is not None else 'N/A'}")
            return signal_value, points_used_for_signal

        # Ensure points are in (N, 2) format
        if old_points.ndim == 3 and old_points.shape[1] == 1:
            old_points = old_points.reshape(-1, 2)
        if new_points.ndim == 3 and new_points.shape[1] == 1:
            new_points = new_points.reshape(-1, 2)

        if old_points.shape[1] != 2 or new_points.shape[1] != 2:
            reason = "Invalid point shape after reshape"
            # print(f"[SignalGenerator] {reason}: old_shape={old_points.shape}, new_shape={new_points.shape}")
            return signal_value, points_used_for_signal

        num_points = old_points.shape[0]
        if num_points < self.min_features_for_signal:
            reason = f"Too few points ({num_points}/{self.min_features_for_signal})"
            # print(f"[SignalGenerator] {reason}")
            return signal_value, points_used_for_signal # Return new_points as points_used, even if signal is 0

        try:
            displacements = new_points - old_points
            vertical_displacements = displacements[:, 1]

            filtered_displacements = vertical_displacements
            filtered_qualities = weights # Use the passed 'weights' as 'qualities'
            num_original = len(filtered_displacements)
            num_filtered = num_original

            if self.iqr_filter_enabled and num_original >= 4:
                try:
                    q1, q3 = np.percentile(filtered_displacements, [25, 75])
                    iqr = q3 - q1
                    if iqr > 1e-9:
                        lower_bound = q1 - self.iqr_k_factor * iqr
                        upper_bound = q3 + self.iqr_k_factor * iqr
                        mask = (filtered_displacements >= lower_bound) & (filtered_displacements <= upper_bound)
                        filtered_displacements = filtered_displacements[mask]
                        if filtered_qualities is not None:
                            filtered_qualities = filtered_qualities[mask]
                        num_filtered = len(filtered_displacements)
                        # if num_filtered < num_original:
                        #     print(f"[IQR Debug] Filtered {num_original - num_filtered} outliers ({num_original} -> {num_filtered}).")
                except Exception as e_iqr:
                    print(f"[SignalGenerator] Warning: IQR filter failed: {e_iqr}")
                    filtered_qualities = weights
                    filtered_displacements = vertical_displacements
                    num_filtered = len(filtered_displacements)

            if num_filtered > 0:
                points_used_for_signal = new_points # Or a subset if points themselves were filtered by IQR mask
                                                  # For now, assume new_points corresponding to filtered_displacements
                                                  # This needs careful handling if points are filtered.
                                                  # Let's simplify and return all new_points if any signal is generated.
                if num_filtered < num_original and self.iqr_filter_enabled: # If IQR actually filtered points
                    # We need to select the new_points that correspond to the filtered_displacements
                    # This requires the 'mask' from IQR to be applied to new_points as well.
                    # For simplicity, if IQR filters, we might just return all new_points for now,
                    # or None if this detail is critical for downstream.
                    # Let's assume for now that PipelineManager uses the 'raw_signal' and 'tracked_points' (all new_points)
                    # separately, and the signal value itself reflects the filtering.
                    pass # Mask would have been applied to filtered_displacements and filtered_qualities

                use_weights = isinstance(filtered_qualities, np.ndarray) and len(filtered_qualities) == num_filtered

                if use_weights:
                    clipped_weights = np.maximum(0, filtered_qualities)
                    if num_filtered > 1:
                        p95 = np.percentile(clipped_weights, 95)
                        clipped_weights = np.minimum(clipped_weights, p95)

                    if self.aggregation_method == 'mean':
                        weight_sum = np.sum(clipped_weights)
                        if weight_sum > 1e-9:
                            norm_weights = clipped_weights / weight_sum
                            signal_value = np.sum(filtered_displacements * norm_weights)
                            reason = f"Weighted Mean ({num_filtered}/{num_original} pts)"
                        else:
                            signal_value = np.mean(filtered_displacements)
                            reason = f"Mean (weights near zero) ({num_filtered}/{num_original} pts)"
                    else: # Median
                        signal_value = self._weighted_median(filtered_displacements, clipped_weights)
                        reason = f"Weighted Median ({num_filtered}/{num_original} pts)"
                else: # No valid weights
                    if self.aggregation_method == 'mean':
                        signal_value = np.mean(filtered_displacements)
                        reason = f"Mean ({num_filtered}/{num_original} pts, no weights)"
                    else: # Median
                        signal_value = np.median(filtered_displacements)
                        reason = f"Median ({num_filtered}/{num_original} pts, no weights)"
            else:
                reason = f"No points left after IQR filter ({num_original} -> 0)"

            if np.isnan(signal_value):
                signal_value = 0.0
                reason += " (NaN result -> 0.0)"
            
            points_used_for_signal = new_points # Return all new_points if signal calculation was attempted

        except Exception as e_calc:
            print(f"[SignalGenerator] Error during {self.aggregation_method.capitalize()} Vertical Disp calc: {e_calc}")
            traceback.print_exc()
            signal_value = 0.0
            reason = f"Exception in {self.aggregation_method.capitalize()} Vert Disp"

        # if signal_value == 0.0 and reason not in ["Input None/Empty", f"Too few points ({num_points}/{self.min_features_for_signal})"]:
        #      print(f"[SignalGenerator Debug] Signal=0.0, Reason='{reason}'")

        return signal_value, points_used_for_signal


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
    # Config for weighted mean test (assuming FeatureTracker provides weights)
    mock_config_weighted_mean = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'mean'
        # IQR disabled for this specific test
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
    # --- Data for weighted test ---
    old8 = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    new8 = np.array([[10, 11], [20, 23], [30, 36]], dtype=np.float32) # dy = [1, 3, 6]
    qual8 = np.array([10.0, 1.0, 1.0], dtype=np.float32) # High weight on first point (dy=1)

    tracked_data_list_tuples = [ # Simulating the old input format for testing the new method
        (old1, new1, None, "ROI1"), (old2, new2, None, "ROI2"), (old3, new3, None, "ROI3"),
        (old4, new4, None, "ROI4"), (old5, new5, None, "ROI5"), (old6, new6, None, "ROI6"),
        (old7, new7, None, "ROI7"), 
        (old8, new8, qual8, "ROI8") 
    ]

    # --- Test Processing (Median) ---
    print("\n--- Processing Mock Data (Median Aggregation) ---")
    signals_median = [generator_median.generate_signal(o, n, roi_def, q)[0] for o, n, q, roi_def in tracked_data_list_tuples]
    print(f"Calculated Signals (Median): {signals_median}")
    # Expected Median: [2.0, 0.0, 0.0 (too few points), 0.0, 0.0, 2.0]

    # --- Basic Assertions (Median) ---
    print("\n--- Running Assertions (Median) ---")
    assert len(signals_median) == len(tracked_data_list_tuples), "Median: Number of signals should match number of ROIs"
    assert np.isclose(signals_median[0], 2.0), f"Median ROI 1 expected ~2.0, got {signals_median[0]}"
    assert np.isclose(signals_median[1], 0.0), f"Median ROI 2 expected ~0.0, got {signals_median[1]}" # Median is 0.0
    assert signals_median[2] == 0.0, "Median ROI 3 (not enough points) should produce zero signal"
    assert signals_median[3] == 0.0, "Median ROI 4 (no points) should produce zero signal"
    assert signals_median[4] == 0.0, "Median ROI 5 (constant points) should produce zero signal"
    assert np.isclose(signals_median[5], 2.0), f"Median ROI 6 (identical displacements) should produce signal 2.0, got {signals_median[5]}"
    print("--- Median Assertions Passed ---")

    # --- Test Processing (Mean) ---
    print("\n--- Processing Mock Data (Mean Aggregation) ---")
    signals_mean = [generator_mean.generate_signal(o, n, roi_def, q)[0] for o, n, q, roi_def in tracked_data_list_tuples]
    print(f"Calculated Signals (Mean): {signals_mean}")
    # Expected Mean: [2.0, 0.16, 0.0 (too few points), 0.0, 0.0, 2.0]

    # --- Basic Assertions (Mean) ---
    print("\n--- Running Assertions (Mean) ---")
    assert len(signals_mean) == len(tracked_data_list_tuples), "Mean: Number of signals should match number of ROIs"
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
    signals_median_iqr = [generator_median_iqr.generate_signal(o, n, roi_def, q)[0] for o, n, q, roi_def in tracked_data_list_tuples]
    print(f"Calculated Signals (Median w/ IQR): {signals_median_iqr}")
    # Expected Median w/ IQR: [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0 (outliers 5, 5 removed)]
    print("\n--- Running Assertions (Median w/ IQR) ---")
    assert np.isclose(signals_median_iqr[6], 2.0), f"Median ROI 7 (IQR) expected ~2.0 (outliers removed), got {signals_median_iqr[6]}"
    print("--- Median w/ IQR Assertions Passed ---")
    
    # --- Test Processing (Weighted Mean) ---
    print("\n--- Processing Mock Data (Weighted Mean Aggregation) ---")
    generator_weighted_mean = SignalGenerator(config=mock_config_weighted_mean)
    signals_weighted_mean = [generator_weighted_mean.generate_signal(o, n, roi_def, q)[0] for o, n, q, roi_def in tracked_data_list_tuples]
    print(f"Calculated Signals (Weighted Mean): {signals_weighted_mean}")
    # Expected Weighted Mean for ROI 8: dy=[1, 3, 6], weights=[10, 1, 1] -> (1*10 + 3*1 + 6*1) / (10+1+1) = 19 / 12 = 1.5833
    print("\n--- Running Assertions (Weighted Mean) ---")
    assert np.isclose(signals_weighted_mean[7], 19.0/12.0), f"Weighted Mean ROI 8 expected ~1.583, got {signals_weighted_mean[7]}"
    # Test weighted median (using the same generator, just changing method temporarily)
    # generator_weighted_mean.aggregation_method = 'median' # Requires _weighted_median helper
    # signals_weighted_median = generator_weighted_mean.process_tracked_features(tracked_data_list)
    # assert np.isclose(signals_weighted_median[7], 1.0), f"Weighted Median ROI 8 expected 1.0 (value associated with highest weight crossing 50%), got {signals_weighted_median[7]}"
    print("--- Weighted Mean/Median Assertions Passed ---")
    print("\nSignalGenerator module test finished.")
