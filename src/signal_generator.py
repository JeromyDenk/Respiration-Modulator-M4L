# src/signal_generator.py
# Phase 3: Calculates motion signal(s) from tracked feature flow vectors using PCA.
# ADDED: More specific debug prints for zero signal cases.

import numpy as np
import traceback
# Optional: Use scikit-learn for PCA if available and preferred.
# from sklearn.decomposition import PCA
# If using sklearn, add it to requirements.txt

class SignalGenerator:
    """
    Calculates a raw motion signal for each ROI based on the displacement
    of tracked features, using PCA to find the principal axis of motion.
    """
    def __init__(self, config=None):
        """
        Initializes the SignalGenerator.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'SIGNAL_MIN_FEATURES_FOR_PCA' (int): Minimum tracked features needed for PCA.
                'SIGNAL_PCA_METHOD' (str): 'numpy' or 'sklearn' (if sklearn is used).
                Defaults are used if config is None or keys are missing.
        """
        if config is None:
            config = {}

        # Minimum number of valid displacement vectors required to run PCA
        self.min_features_for_pca = config.get('SIGNAL_MIN_FEATURES_FOR_PCA', 3) # Need at least 2 for variance, 3 is safer
        self.pca_method = config.get('SIGNAL_PCA_METHOD', 'numpy') # Default to numpy implementation

        # Optional: Initialize sklearn PCA if chosen
        # if self.pca_method == 'sklearn':
        #     self.pca = PCA(n_components=1) # We only need the first principal component

        print("[SignalGenerator] Initialized.")
        print(f"  Min Features for PCA: {self.min_features_for_pca}")
        print(f"  PCA Method: {self.pca_method}")


    def _calculate_pca_signal_numpy(self, displacements):
        """Calculates signal using PCA manually with NumPy."""
        num_displacements = displacements.shape[0]
        if num_displacements < self.min_features_for_pca:
            # This case is handled before calling, but double-check
            # print(f"[SignalGenerator] Debug PCA: Not enough features ({num_displacements})") # Noisy
            return 0.0, "Too few points for PCA"

        try:
            # Center the data (subtract the mean)
            mean_disp = np.mean(displacements, axis=0)
            centered_data = displacements - mean_disp

            # Check if data is constant (all displacements the same -> zero centered data)
            if np.allclose(centered_data, 0, atol=1e-6): # Added tolerance
                 # print("[SignalGenerator] Debug PCA: Centered data is all zero (no relative motion)") # Noisy
                 return 0.0, "No relative motion"

            # Calculate the covariance matrix
            cov_matrix = np.cov(centered_data, rowvar=False)

            # Check if covariance matrix is valid
            if np.allclose(cov_matrix, 0, atol=1e-6): # Added tolerance
                 # print("[SignalGenerator] Debug PCA: Covariance matrix is zero") # Noisy
                 return 0.0, "Zero covariance"


            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) # Use eigh for symmetric matrix

            # Check for valid eigenvalues/vectors (e.g., NaN)
            if np.any(np.isnan(eigenvalues)) or np.any(np.isnan(eigenvectors)):
                 print("[SignalGenerator] Warning: NaN encountered in eigenvalues/vectors during PCA.")
                 return 0.0, "NaN in PCA result"


            # Find the eigenvector corresponding to the largest eigenvalue (the first principal component)
            principal_component = eigenvectors[:, np.argmax(eigenvalues)] # Shape (2,)

            # Project the centered displacements onto the principal component
            projection_magnitudes = np.dot(centered_data, principal_component)

            # Calculate the signal: mean of the *absolute* projections
            signal_value = np.mean(np.abs(projection_magnitudes))

            # print(f"[SignalGenerator] Debug: PCA success. Signal={signal_value:.4f}") # Debug noise
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
        Calculates raw motion signals from tracked feature points for multiple ROIs.

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
                    if num_points >= self.min_features_for_pca:
                        # Calculate displacement vectors (dx, dy)
                        displacements = new_points - old_points # Shape (N, 2)

                        # Calculate signal using the chosen PCA method
                        if self.pca_method == 'numpy':
                            signal_value, reason = self._calculate_pca_signal_numpy(displacements)
                        # elif self.pca_method == 'sklearn':
                        #     signal_value, reason = self._calculate_pca_signal_sklearn(displacements)
                        else:
                             print(f"[SignalGenerator] Warning: Unknown PCA method '{self.pca_method}'. Defaulting to 0 signal.")
                             reason = "Unknown PCA method"
                    else: # Handle cases where points exist but not enough for PCA
                        reason = f"Too few points ({num_points}/{self.min_features_for_pca})"
                else: # Handle case where reshape failed or shape is wrong
                     reason = "Invalid point shape"
            # else: Handled by default reason "Input None/Empty"

            # --- More Verbose Debugging ---
            # Print status for every frame if signal is zero, otherwise print less often
            if signal_value == 0.0 and reason != "PCA OK (0 pts)": # Don't print if PCA was ok but signal was zero
                 print(f"[SignalGenerator Frame Debug] ROI{i}: Signal=0.0, Reason='{reason}'")
            # elif np.random.rand() < 0.05: # Print non-zero signals occasionally
            #      print(f"[SignalGenerator Frame Debug] ROI{i}: Signal={signal_value:.4f}, Reason='{reason}'")
            # --- End Debugging ---


            raw_signals.append(signal_value)
            # processing_summary.append(f"ROI{i}:{signal_value:.3f}({reason})") # Less noisy summary

        # Optional: Print summary less frequently than every frame
        # if np.random.rand() < 0.05: # Print ~5% of the time
        #    print(f"[SignalGenerator] Debug Summary: {', '.join(processing_summary)}")

        return raw_signals

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing SignalGenerator module...")

    # --- Mock Setup ---
    mock_config = {
        'SIGNAL_MIN_FEATURES_FOR_PCA': 3,
        'SIGNAL_PCA_METHOD': 'numpy'
    }
    generator = SignalGenerator(config=mock_config)

    # --- Mock Tracked Data ---
    # ROI 1: Simulate vertical motion (respiration-like) + some noise
    old1 = np.array([[10, 10], [15, 11], [12, 9], [18, 10]], dtype=np.float32)
    new1 = np.array([[10, 12], [15, 13], [12, 11], [18, 12]], dtype=np.float32) # Moved ~2 units in Y

    # ROI 2: Simulate horizontal motion + noise (less respiration-like)
    old2 = np.array([[50, 50], [55, 51], [52, 49], [58, 50], [51, 52]], dtype=np.float32)
    new2 = np.array([[52, 50], [57, 51.5], [54, 49.5], [60, 50], [53, 51.8]], dtype=np.float32) # Moved ~2 units in X

    # ROI 3: Not enough points
    old3 = np.array([[100, 100], [105, 101]], dtype=np.float32)
    new3 = np.array([[100, 101], [105, 102]], dtype=np.float32)

    # ROI 4: No points tracked
    old4, new4 = None, None

    # ROI 5: Constant points (zero displacement)
    old5 = np.array([[200, 200], [205, 201], [202, 199]], dtype=np.float32)
    new5 = old5.copy()

    # ROI 6: Identical displacements (relative motion is zero)
    old6 = np.array([[300, 300], [305, 301], [302, 299]], dtype=np.float32)
    new6 = old6 + np.array([1, 2], dtype=np.float32) # All points moved by (1, 2)


    tracked_data_list = [
        (old1, new1), (old2, new2), (old3, new3), (old4, new4), (old5, new5), (old6, new6)
    ]

    # --- Test Processing ---
    print("\n--- Processing Mock Data ---")
    signals = generator.process_tracked_features(tracked_data_list)

    print(f"\nCalculated Signals: {signals}")

    # --- Basic Assertions ---
    assert len(signals) == len(tracked_data_list), "Number of signals should match number of ROIs"
    assert signals[0] > 0, "ROI 1 (vertical motion) should produce a positive signal"
    assert signals[1] > 0, "ROI 2 (horizontal motion) should produce a positive signal"
    assert signals[2] == 0.0, "ROI 3 (not enough points) should produce zero signal"
    assert signals[3] == 0.0, "ROI 4 (no points) should produce zero signal"
    assert signals[4] == 0.0, "ROI 5 (constant points) should produce zero signal (Reason: No relative motion)"
    assert signals[5] == 0.0, "ROI 6 (identical displacements) should produce zero signal (Reason: No relative motion)"


    print("\nSignalGenerator module test finished.")
