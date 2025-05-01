# src/feature_tracker.py
# Detects and tracks features within specified ROIs using Optical Flow.

import cv2
import numpy as np
import traceback

class FeatureTracker:
    """
    Detects features (corners) within ROI(s) and tracks them using
    Lucas-Kanade optical flow.
    """
    def __init__(self, config=None):
        """
        Initializes the FeatureTracker.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'OPTICAL_FLOW_PARAMS': {
                    'feature_params': Dict for cv2.goodFeaturesToTrack,
                    'lk_params': Dict for cv2.calcOpticalFlowPyrLK
                },
                'FEATURE_REDETECT_THRESHOLD': Min number of features before redetection.
                Defaults are used if config is None or keys are missing.
        """
        if config is None:
            config = {}

        # --- Load Optical Flow Parameters ---
        # Use nested .get() with defaults for safety
        default_of_params = {
            'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
            'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        }
        of_params = config.get('OPTICAL_FLOW_PARAMS', default_of_params)
        # Merge defaults with loaded params carefully
        self.feature_params = {**default_of_params['feature_params'], **of_params.get('feature_params', {})}
        self.lk_params = {**default_of_params['lk_params'], **of_params.get('lk_params', {})}

        # Ensure lk_params values have correct types (tuples) if loaded from JSON list
        if isinstance(self.lk_params.get('winSize'), list):
             self.lk_params['winSize'] = tuple(self.lk_params['winSize'])
        if isinstance(self.lk_params.get('criteria'), list) and len(self.lk_params['criteria']) == 3:
             crit_list = self.lk_params['criteria']
             self.lk_params['criteria'] = (int(crit_list[0]), int(crit_list[1]), float(crit_list[2]))
        # Ensure criteria is a tuple if loaded incorrectly
        elif not isinstance(self.lk_params.get('criteria'), tuple):
             self.lk_params['criteria'] = default_of_params['lk_params']['criteria']


        # Threshold for re-detecting features
        self.redetect_threshold = config.get('FEATURE_REDETECT_THRESHOLD', int(self.feature_params.get('maxCorners', 100) * 0.7))

        # --- Internal State ---
        self.prev_gray_frame = None
        # Store features per ROI. Key: ROI index, Value: np.array of points
        self.prev_features_per_roi = {}

        print("[FeatureTracker] Initialized.")
        print(f"  Feature Params: {self.feature_params}")
        print(f"  LK Params: {self.lk_params}")
        print(f"  Redetect Threshold: {self.redetect_threshold}")

    def _detect_features(self, gray_frame, roi):
        """
        Detects good features to track within a specific ROI.

        Args:
            gray_frame (np.ndarray): Grayscale image to detect features in.
            roi (tuple): The (x, y, w, h) region of interest.

        Returns:
            np.ndarray or None: Detected feature points (shape (N, 1, 2)) or None if detection fails.
        """
        x, y, w, h = roi
        if w <= 0 or h <= 0: return None # Skip invalid ROIs

        # Create a mask for the ROI
        mask = np.zeros_like(gray_frame)
        roi_y_end = min(gray_frame.shape[0], y + h)
        roi_x_end = min(gray_frame.shape[1], x + w)
        # Ensure start coords are also valid
        roi_y_start = max(0, y)
        roi_x_start = max(0, x)

        if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end:
            # print(f"[FeatureTracker] Warning: ROI {roi} results in empty mask slice.")
            return None # ROI is outside frame or invalid

        mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = 255

        try:
            features = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **self.feature_params)
            # print(f"[FeatureTracker] Detected {len(features) if features is not None else 0} features in ROI {roi}") # Debug
            return features
        except cv2.error as e_gftt:
            print(f"[FeatureTracker] OpenCV Error detecting features in ROI {roi}: {e_gftt}")
            return None
        except Exception as e:
            print(f"[FeatureTracker] Error detecting features in ROI {roi}: {e}")
            traceback.print_exc()
            return None

    def process_frame(self, current_gray_frame, current_rois):
        """
        Processes a grayscale frame to track features within the given ROIs.

        Args:
            current_gray_frame (np.ndarray): The current grayscale frame.
            current_rois (list): A list of ROI tuples [(x, y, w, h), ...].

        Returns:
            list: A list where each element corresponds to an ROI. Each element is a
                  tuple (good_old_points, good_new_points) containing NumPy arrays
                  of successfully tracked points. Returns an empty list or tuple
                  of (None, None) for ROIs where tracking failed or wasn't possible.
                  Example: [ (arr_old_roi0, arr_new_roi0), (arr_old_roi1, arr_new_roi1), ... ]
        """
        if current_gray_frame is None:
            print("[FeatureTracker] Error: Received None frame.")
            return [(None, None)] * len(current_rois) # Return list of Nones matching ROI count

        tracked_data_all_rois = []
        new_features_per_roi = {} # Store features found in *this* cycle

        for i, roi in enumerate(current_rois):
            good_old_points, good_new_points = None, None # Defaults for this ROI
            prev_features = self.prev_features_per_roi.get(i) # Get features tracked from *last* cycle for this ROI index

            # --- Feature Detection Logic ---
            # Detect features if first frame, no previous features, or too few features left
            needs_redetection = (
                self.prev_gray_frame is None or
                prev_features is None or
                len(prev_features) < self.redetect_threshold
            )

            if needs_redetection:
                # print(f"[FeatureTracker] Redetecting features for ROI {i}") # Debug
                # Use previous frame if available (more stable), else current
                detection_frame = self.prev_gray_frame if self.prev_gray_frame is not None else current_gray_frame
                prev_features = self._detect_features(detection_frame, roi)
                # Store newly detected features for the *next* frame's tracking
                new_features_per_roi[i] = prev_features
                # Cannot track on the frame features were just detected on
                tracked_data_all_rois.append((None, None))
                continue # Skip tracking for this ROI on this frame

            # --- Feature Tracking Logic ---
            if self.prev_gray_frame is not None and prev_features is not None and len(prev_features) > 0:
                try:
                    # Ensure prev_features is float32
                    if prev_features.dtype != np.float32:
                         prev_features = prev_features.astype(np.float32)

                    next_points, status, error = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray_frame, current_gray_frame, prev_features, None, **self.lk_params
                    )

                    if next_points is not None and status is not None:
                        # Filter points based on status == 1
                        status_flat = status.flatten()
                        valid_mask = (status_flat == 1)
                        good_new_points = next_points[valid_mask]
                        good_old_points = prev_features[valid_mask] # Corresponding old points

                        # Update features for the next frame
                        # Reshape to (N, 1, 2) format expected by LK flow
                        new_features_per_roi[i] = good_new_points.reshape(-1, 1, 2) if len(good_new_points) > 0 else None

                    else: # Tracking failed (returned None)
                         new_features_per_roi[i] = None

                except cv2.error as e_lk:
                    print(f"[FeatureTracker] OpenCV Error tracking features for ROI {i}: {e_lk}")
                    new_features_per_roi[i] = None # Reset features on error
                except Exception as e:
                    print(f"[FeatureTracker] Error tracking features for ROI {i}: {e}")
                    traceback.print_exc()
                    new_features_per_roi[i] = None # Reset features on error

            else: # No previous frame or no features to track
                 new_features_per_roi[i] = None

            # Append the tracked points (or None if failed/redetected) for this ROI
            tracked_data_all_rois.append((good_old_points, good_new_points))

        # --- Update State for Next Frame ---
        self.prev_gray_frame = current_gray_frame.copy() # Store copy for next iteration
        self.prev_features_per_roi = new_features_per_roi # Update features for next cycle

        return tracked_data_all_rois

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing FeatureTracker module...")

    # --- Mock Setup ---
    frame_shape = (480, 640)
    mock_roi = (200, 150, 100, 80) # Example ROI (x, y, w, h)
    mock_config = {
        'OPTICAL_FLOW_PARAMS': {
            'feature_params': {'maxCorners': 50, 'qualityLevel': 0.2, 'minDistance': 5},
            'lk_params': {'winSize': (21, 21), 'maxLevel': 3} # Use defaults for criteria
        },
        'FEATURE_REDETECT_THRESHOLD': 10
    }

    tracker = FeatureTracker(config=mock_config)

    # Create two dummy frames (e.g., noise with slight shift)
    frame1_gray = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
    frame2_gray = np.roll(frame1_gray, shift=(2, 3), axis=(0, 1)) # Shift down 2, right 3

    # --- Test Cycle 1 (Feature Detection) ---
    print("\n--- Test Cycle 1 (Detection) ---")
    tracked_data1 = tracker.process_frame(frame1_gray, [mock_roi]) # Pass ROI in a list
    print(f"Tracked Data 1: Old points: {tracked_data1[0][0]}, New points: {tracked_data1[0][1]}")
    print(f"Internal prev_features count: {len(tracker.prev_features_per_roi.get(0, [])) if tracker.prev_features_per_roi.get(0) is not None else 0}")
    assert tracked_data1[0][0] is None, "Should not have old points on first frame"
    assert tracked_data1[0][1] is None, "Should not have new points on first frame (detection frame)"
    assert tracker.prev_features_per_roi.get(0) is not None, "Features should have been detected and stored"

    # --- Test Cycle 2 (Tracking) ---
    print("\n--- Test Cycle 2 (Tracking) ---")
    tracked_data2 = tracker.process_frame(frame2_gray, [mock_roi])
    print(f"Tracked Data 2: Old points shape: {tracked_data2[0][0].shape if tracked_data2[0][0] is not None else 'None'}, New points shape: {tracked_data2[0][1].shape if tracked_data2[0][1] is not None else 'None'}")
    print(f"Internal prev_features count: {len(tracker.prev_features_per_roi.get(0, [])) if tracker.prev_features_per_roi.get(0) is not None else 0}")
    assert tracked_data2[0][0] is not None, "Should have old points on second frame"
    assert tracked_data2[0][1] is not None, "Should have new points on second frame"
    assert len(tracked_data2[0][0]) == len(tracked_data2[0][1]), "Old and new points count should match"
    assert tracker.prev_features_per_roi.get(0) is not None, "Tracked features should be stored for next frame"

    # --- Test Cycle 3 (Potential Redetection if points lost) ---
    print("\n--- Test Cycle 3 (Simulate Lost Points -> Redetection) ---")
    # Manually reduce tracked points below threshold
    tracker.prev_features_per_roi[0] = tracker.prev_features_per_roi[0][:5] # Keep only 5 points
    print(f"Manually reduced prev_features count to: {len(tracker.prev_features_per_roi.get(0, []))}")
    frame3_gray = np.roll(frame2_gray, shift=(1, 1), axis=(0, 1))
    tracked_data3 = tracker.process_frame(frame3_gray, [mock_roi])
    print(f"Tracked Data 3: Old points: {tracked_data3[0][0]}, New points: {tracked_data3[0][1]}")
    print(f"Internal prev_features count: {len(tracker.prev_features_per_roi.get(0, [])) if tracker.prev_features_per_roi.get(0) is not None else 0}")
    assert tracked_data3[0][0] is None, "Should not have old points after redetection"
    assert tracked_data3[0][1] is None, "Should not have new points after redetection"
    assert tracker.prev_features_per_roi.get(0) is not None, "New features should have been detected"

    print("\nFeatureTracker module test finished.")

