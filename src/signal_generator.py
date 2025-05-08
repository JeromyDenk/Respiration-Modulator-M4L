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
                'CALCULATE_LEVEL_SIGNAL' (bool): Whether to calculate absolute level signal.
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
        
        # --- NEW: Configuration for Level Signal ---
        self.calculate_level_signal = config.get('CALCULATE_LEVEL_SIGNAL', False)
        # --- END Level Signal Config ---

        print(f"[SignalGenerator] Initialized.")
        print(f"  Config: MinFeat={self.min_features_for_signal}, AggMethod={self.aggregation_method}, IQR={self.iqr_filter_enabled}, K={self.iqr_k_factor}, CalcLevel={self.calculate_level_signal}")


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
            tuple: (float, float or None, np.ndarray or None)
                   - raw_differential_signal (float): The traditional displacement-based signal.
                   - raw_level_signal (float or None): The absolute vertical level signal, or None if not calculated or error.
                   - points_output (np.ndarray or None): The new_points array if processing was attempted, else None.
        """
        raw_differential_signal = 0.0
        raw_level_signal = None
        points_output = None # Will be new_points if processing is attempted, else None
        # reason = "Input None/Empty" # For debugging, not returned

        if old_points is None or new_points is None or \
           len(old_points) == 0 or len(old_points) != len(new_points):
            # print(f"[SignalGenerator] Invalid input points: old_none={old_points is None}, new_none={new_points is None}, len_old={len(old_points) if old_points is not None else 'N/A'}")
            return raw_differential_signal, raw_level_signal, points_output

        # Ensure points are in (N, 2) format
        if old_points.ndim == 3 and old_points.shape[1] == 1:
            old_points = old_points.reshape(-1, 2)
        if new_points.ndim == 3 and new_points.shape[1] == 1:
            new_points = new_points.reshape(-1, 2)
        
        points_output = new_points # If we got this far, these are the points we're working with

        if old_points.shape[1] != 2 or new_points.shape[1] != 2:
            # reason = "Invalid point shape after reshape"
            # print(f"[SignalGenerator] {reason}: old_shape={old_points.shape}, new_shape={new_points.shape}")
            return raw_differential_signal, raw_level_signal, points_output

        num_initial_points = old_points.shape[0]
        if num_initial_points < self.min_features_for_signal:
            # reason = f"Too few initial points ({num_initial_points}/{self.min_features_for_signal})"
            # print(f"[SignalGenerator] {reason}")
            return raw_differential_signal, raw_level_signal, points_output

        try:
            displacements = new_points - old_points
            vertical_displacements = displacements[:, 1]

            # This mask will be relative to the original num_initial_points
            final_selection_mask = np.ones(num_initial_points, dtype=bool)

            if self.iqr_filter_enabled and num_initial_points >= 4: # Min points for sensible IQR
                try:
                    # Calculate IQR on the original vertical_displacements
                    q1, q3 = np.percentile(vertical_displacements, [25, 75])
                    iqr_val = q3 - q1
                    if iqr_val > 1e-9: # Only apply if IQR is meaningful
                        lower_bound = q1 - self.iqr_k_factor * iqr_val
                        upper_bound = q3 + self.iqr_k_factor * iqr_val
                        # This mask is on the original vertical_displacements
                        final_selection_mask = (vertical_displacements >= lower_bound) & (vertical_displacements <= upper_bound)
                    # else: no filtering if IQR is zero (all displacements are same or very close)
                except Exception as e_iqr:
                    print(f"[SignalGenerator] Warning: IQR filter failed: {e_iqr}")
                    # final_selection_mask remains all True, so no filtering due to error

            effective_vertical_displacements = vertical_displacements[final_selection_mask]
            effective_new_points = new_points[final_selection_mask]
            effective_qualities = None
            if weights is not None:
                # Ensure weights align with the selected points
                if len(weights) == num_initial_points:
                    effective_qualities = weights[final_selection_mask]
                else:
                    print(f"[SignalGenerator] Warning: Mismatch in length of weights ({len(weights)}) and points ({num_initial_points}). Ignoring weights.")
            
            num_effective_points = len(effective_vertical_displacements)

            if num_effective_points < self.min_features_for_signal:
                # reason = f"Too few effective points ({num_effective_points}/{self.min_features_for_signal} from {num_initial_points})"
                # raw_differential_signal remains 0.0
                # raw_level_signal remains None
                # points_output is already new_points
                pass # Fall through to return current values
            else:
                # --- Calculate Raw Differential Signal ---
                use_weights = isinstance(effective_qualities, np.ndarray) and len(effective_qualities) == num_effective_points

                if use_weights:
                    clipped_weights = np.maximum(0, effective_qualities)
                    if num_effective_points > 1: # Percentile needs at least 2 points
                        p95 = np.percentile(clipped_weights, 95)
                        clipped_weights = np.minimum(clipped_weights, p95)

                    if self.aggregation_method == 'mean':
                        weight_sum = np.sum(clipped_weights)
                        if weight_sum > 1e-9:
                            norm_weights = clipped_weights / weight_sum
                            raw_differential_signal = np.sum(effective_vertical_displacements * norm_weights)
                            # reason = f"Weighted Mean ({num_effective_points}/{num_initial_points} pts)"
                        else:
                            raw_differential_signal = np.mean(effective_vertical_displacements)
                            # reason = f"Mean (weights near zero) ({num_effective_points}/{num_initial_points} pts)"
                    else: # Median
                        raw_differential_signal = self._weighted_median(effective_vertical_displacements, clipped_weights)
                        # reason = f"Weighted Median ({num_effective_points}/{num_initial_points} pts)"
                else: # No valid weights
                    if self.aggregation_method == 'mean':
                        raw_differential_signal = np.mean(effective_vertical_displacements)
                        # reason = f"Mean ({num_effective_points}/{num_initial_points} pts, no weights)"
                    else: # Median
                        raw_differential_signal = np.median(effective_vertical_displacements)
                        # reason = f"Median ({num_effective_points}/{num_initial_points} pts, no weights)"

                # --- Calculate Raw Level Signal (if enabled) ---
                if self.calculate_level_signal:
                    # Use the same set of points that contributed to the differential signal
                    # (i.e., those that survived the displacement-based IQR filter)
                    if num_effective_points > 0: # Should be true if we are in this block
                        y_coordinates = effective_new_points[:, 1]
                        # Invert: lower Y (up on screen) = higher signal (fuller lungs)
                        raw_level_signal = -np.mean(y_coordinates)
                    # else raw_level_signal remains None
            
            if np.isnan(raw_differential_signal):
                raw_differential_signal = 0.0
                # reason += " (NaN diff result -> 0.0)"
            if raw_level_signal is not None and np.isnan(raw_level_signal):
                raw_level_signal = None # Or 0.0 if preferred for missing data after attempt
                # reason += " (NaN level result -> None)"

        except Exception as e_calc:
            print(f"[SignalGenerator] Error during signal calculation: {e_calc}")
            traceback.print_exc()
            # raw_differential_signal, raw_level_signal remain at their defaults (0.0, None)
            # points_output is already new_points

        return raw_differential_signal, raw_level_signal, points_output


# Example usage (for testing this module directly)
# NOTE: Assertions updated for the MEDIAN calculation (default)
if __name__ == '__main__':
    print("Testing SignalGenerator module (Mean/Median Vertical Displacement Mode)...")

    # --- Mock Setup ---
    mock_config_median = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'median', # Explicitly median (default)
        'CALCULATE_LEVEL_SIGNAL': True # Enable for testing
    }
    mock_config_mean = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'mean', # Explicitly mean
        'CALCULATE_LEVEL_SIGNAL': False # Test with it off
    }
    mock_config_median_iqr = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'median',
        'IQR_FILTER_ENABLED': True, 'IQR_K_FACTOR': 1.5, # Enable IQR
        'CALCULATE_LEVEL_SIGNAL': True
    }
    # Config for weighted mean test (assuming FeatureTracker provides weights)
    mock_config_weighted_mean = {
        'SIGNAL_MIN_FEATURES_FOR_SIGNAL': 3,
        'SIGNAL_AGGREGATION_METHOD': 'mean',
        # IQR disabled for this specific test
        'CALCULATE_LEVEL_SIGNAL': True
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
    results_median = [generator_median.generate_signal(o, n, roi_def, q) for o, n, q, roi_def in tracked_data_list_tuples]
    diff_signals_median = [r[0] for r in results_median]
    level_signals_median = [r[1] for r in results_median]
    print(f"Calculated Diff Signals (Median): {diff_signals_median}")
    print(f"Calculated Level Signals (Median): {level_signals_median}")
    # Expected Median Diff: [2.0, 0.0, 0.0 (too few points), 0.0, 0.0, 2.0, 2.0, 1.0 (weighted median)]
    # Expected Level for new1 (median): -np.mean([12,13,11,12]) = -12.0
    # Expected Level for new2 (median): -np.mean([50, 51.5, 49.5, 50, 51.8]) = -50.56
    # Expected Level for new8 (median, weighted): -np.mean([11,23,36]) = -23.333... (IQR not active, all points used for level)

    # --- Basic Assertions (Median) ---
    print("\n--- Running Assertions (Median) ---")
    assert len(diff_signals_median) == len(tracked_data_list_tuples), "Median: Number of diff signals should match"
    assert np.isclose(diff_signals_median[0], 2.0), f"Median Diff ROI 1 expected ~2.0, got {diff_signals_median[0]}"
    assert np.isclose(level_signals_median[0], -12.0), f"Median Level ROI 1 expected ~-12.0, got {level_signals_median[0]}"
    assert np.isclose(diff_signals_median[1], 0.0), f"Median Diff ROI 2 expected ~0.0, got {diff_signals_median[1]}"
    assert np.isclose(level_signals_median[1], -np.mean(new2[:,1])), f"Median Level ROI 2 expected ~{-np.mean(new2[:,1]):.3f}, got {level_signals_median[1]}"
    assert diff_signals_median[2] == 0.0, "Median Diff ROI 3 (not enough points) should produce zero signal"
    assert level_signals_median[2] is None, "Median Level ROI 3 (not enough points) should be None"
    assert diff_signals_median[3] == 0.0, "Median Diff ROI 4 (no points) should produce zero signal"
    assert level_signals_median[3] is None, "Median Level ROI 4 (no points) should be None"
    assert diff_signals_median[4] == 0.0, "Median Diff ROI 5 (constant points) should produce zero signal"
    assert np.isclose(level_signals_median[4], -np.mean(new5[:,1])), f"Median Level ROI 5 expected ~{-np.mean(new5[:,1]):.3f}, got {level_signals_median[4]}"
    assert np.isclose(diff_signals_median[5], 2.0), f"Median Diff ROI 6 (identical displacements) should produce signal 2.0, got {diff_signals_median[5]}"
    # Weighted median for ROI8 with generator_median (aggregation_method='median')
    # dy = [1, 3, 6], weights = [10, 1, 1]. Sorted data: [1,3,6], sorted weights: [10,1,1]. Cum_weights: [10,11,12]. Total=12. Median mark=6. First index where cum_weight >= 6 is 0. Data at index 0 is 1.
    assert np.isclose(diff_signals_median[7], 1.0), f"Median Diff ROI 8 (weighted) expected 1.0, got {diff_signals_median[7]}"
    assert np.isclose(level_signals_median[7], -np.mean(new8[:,1])), f"Median Level ROI 8 expected ~{-np.mean(new8[:,1]):.3f}, got {level_signals_median[7]}"
    print("--- Median Assertions Passed ---")

    # --- Test Processing (Mean) ---
    print("\n--- Processing Mock Data (Mean Aggregation) ---")
    results_mean = [generator_mean.generate_signal(o, n, roi_def, q) for o, n, q, roi_def in tracked_data_list_tuples]
    diff_signals_mean = [r[0] for r in results_mean]
    level_signals_mean = [r[1] for r in results_mean] # CALCULATE_LEVEL_SIGNAL is False for generator_mean
    print(f"Calculated Diff Signals (Mean): {diff_signals_mean}")
    print(f"Calculated Level Signals (Mean - should be None): {level_signals_mean}")
    # Expected Mean Diff: [2.0, 0.16, 0.0 (too few points), 0.0, 0.0, 2.0, 3.0, 1.583]

    # --- Basic Assertions (Mean) ---
    print("\n--- Running Assertions (Mean) ---")
    assert len(diff_signals_mean) == len(tracked_data_list_tuples), "Mean: Number of diff signals should match"
    assert all(ls is None for ls in level_signals_mean), "Mean: All level signals should be None as CALCULATE_LEVEL_SIGNAL is False"
    assert np.isclose(diff_signals_mean[0], 2.0), f"Mean Diff ROI 1 expected ~2.0, got {diff_signals_mean[0]}"
    assert np.isclose(diff_signals_mean[1], 0.16), f"Mean Diff ROI 2 expected ~0.16, got {diff_signals_mean[1]}"
    assert diff_signals_mean[2] == 0.0, "Mean Diff ROI 3 (not enough points) should produce zero signal"
    assert diff_signals_mean[3] == 0.0, "Mean Diff ROI 4 (no points) should produce zero signal"
    assert diff_signals_mean[4] == 0.0, "Mean Diff ROI 5 (constant points) should produce zero signal"
    assert np.isclose(diff_signals_mean[5], 2.0), f"Mean Diff ROI 6 (identical displacements) should produce signal 2.0, got {diff_signals_mean[5]}"
    print("--- Mean Assertions Passed ---")

    # --- Test Processing (Median with IQR) ---
    print("\n--- Processing Mock Data (Median Aggregation with IQR Filter) ---")
    generator_median_iqr = SignalGenerator(config=mock_config_median_iqr)
    results_median_iqr = [generator_median_iqr.generate_signal(o, n, roi_def, q) for o, n, q, roi_def in tracked_data_list_tuples]
    diff_signals_median_iqr = [r[0] for r in results_median_iqr]
    level_signals_median_iqr = [r[1] for r in results_median_iqr]
    print(f"Calculated Diff Signals (Median w/ IQR): {diff_signals_median_iqr}")
    print(f"Calculated Level Signals (Median w/ IQR): {level_signals_median_iqr}")
    # Expected Median w/ IQR: [2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0 (outliers 5, 5 removed)]
    # For ROI7, dy = [2,2,2,2,5,5]. IQR: Q1=2, Q3=3.5, IQR=1.5. Lower=2-1.5*1.5 = -0.25. Upper=3.5+1.5*1.5 = 5.75.
    # Mask keeps all points. Median of [2,2,2,2,5,5] is (2+2)/2 = 2.
    # Points for level: all new7 points. Level = -mean(new7[:,1])
    print("\n--- Running Assertions (Median w/ IQR) ---")
    assert np.isclose(diff_signals_median_iqr[6], 2.0), f"Median Diff ROI 7 (IQR) expected ~2.0, got {diff_signals_median_iqr[6]}"
    # new7 y-coords: [12, 13, 11, 12, 55, -35]. Mean = (12+13+11+12+55-35)/6 = 68/6 = 11.333...
    # Level signal = -11.333...
    assert np.isclose(level_signals_median_iqr[6], -np.mean(new7[:,1])), f"Median Level ROI 7 (IQR) expected ~{-np.mean(new7[:,1]):.3f}, got {level_signals_median_iqr[6]}"
    print("--- Median w/ IQR Assertions Passed ---")
    
    # --- Test Processing (Weighted Mean) ---
    print("\n--- Processing Mock Data (Weighted Mean Aggregation) ---")
    generator_weighted_mean = SignalGenerator(config=mock_config_weighted_mean)
    results_weighted_mean = [generator_weighted_mean.generate_signal(o, n, roi_def, q) for o, n, q, roi_def in tracked_data_list_tuples]
    diff_signals_weighted_mean = [r[0] for r in results_weighted_mean]
    level_signals_weighted_mean = [r[1] for r in results_weighted_mean]
    print(f"Calculated Diff Signals (Weighted Mean): {diff_signals_weighted_mean}")
    print(f"Calculated Level Signals (Weighted Mean): {level_signals_weighted_mean}")
    # Expected Weighted Mean for ROI 8: dy=[1, 3, 6], weights=[10, 1, 1] -> (1*10 + 3*1 + 6*1) / (10+1+1) = 19 / 12 = 1.5833
    # Expected Level for ROI 8 (new8): -np.mean([11, 23, 36]) = -(70/3) = -23.333...
    print("\n--- Running Assertions (Weighted Mean) ---")
    assert np.isclose(diff_signals_weighted_mean[7], 19.0/12.0), f"Weighted Mean Diff ROI 8 expected ~1.583, got {diff_signals_weighted_mean[7]}"
    assert np.isclose(level_signals_weighted_mean[7], -np.mean(new8[:,1])), f"Weighted Mean Level ROI 8 expected ~{-np.mean(new8[:,1]):.3f}, got {level_signals_weighted_mean[7]}"
    # Test weighted median (using the same generator, just changing method temporarily)
    # generator_weighted_mean.aggregation_method = 'median' # Requires _weighted_median helper
    # signals_weighted_median = generator_weighted_mean.process_tracked_features(tracked_data_list)
    # assert np.isclose(signals_weighted_median[7], 1.0), f"Weighted Median ROI 8 expected 1.0 (value associated with highest weight crossing 50%), got {signals_weighted_median[7]}"
    print("--- Weighted Mean/Median Assertions Passed ---")
    print("\nSignalGenerator module test finished.")
