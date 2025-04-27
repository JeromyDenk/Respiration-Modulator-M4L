# src/feature_tracker.py
# Detects and tracks features within specified ROIs using Optical Flow.
# CORRECTED: Refined redetection logic to avoid loop after first frame.

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
                'FEATURE_REDETECT_THRESHOLD': Min number of successfully tracked features
                                            before redetection is triggered.
                Defaults are used if config is None or keys are missing.
        """
        if config is None:
            config = {}

        # --- Load Optical Flow Parameters ---
        default_of_params = {
            'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
            'lk_params': {'winSize': (15, 15), 'maxLevel': 2, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}
        }
        of_params = config.get('OPTICAL_FLOW_PARAMS', default_of_params)
        self.feature_params = {**default_of_params['feature_params'], **of_params.get('feature_params', {})}
        self.lk_params = {**default_of_params['lk_params'], **of_params.get('lk_params', {})}

        # Ensure lk_params values have correct types
        if isinstance(self.lk_params.get('winSize'), list):
             self.lk_params['winSize'] = tuple(self.lk_params['winSize'])
        if isinstance(self.lk_params.get('criteria'), list) and len(self.lk_params['criteria']) == 3:
             crit_list = self.lk_params['criteria']
             self.lk_params['criteria'] = (int(crit_list[0]), int(crit_list[1]), float(crit_list[2]))
        elif not isinstance(self.lk_params.get('criteria'), tuple):
             self.lk_params['criteria'] = default_of_params['lk_params']['criteria']

        # Threshold for re-detecting features - based on *tracked* points count
        self.redetect_threshold = config.get('FEATURE_REDETECT_THRESHOLD', int(self.feature_params.get('maxCorners', 100) * 0.7))

        # --- Internal State ---
        self.prev_gray_frame = None
        # Stores the features from the PREVIOUS frame that will be USED for tracking in the CURRENT frame
        self.features_to_track_per_roi = {}
        # Stores the count of features successfully tracked IN the previous frame
        self.last_tracked_count_per_roi = {}
        # Flag to indicate if the *previous* frame performed detection (True) or tracking (False)
        self.did_detect_last_frame_per_roi = {}

        # Debug counter
        self._frame_counter = 0

        print("[FeatureTracker] Initialized.")
        print(f"  Feature Params: {self.feature_params}")
        print(f"  LK Params: {self.lk_params}")
        print(f"  Redetect Threshold (applied to tracked count): {self.redetect_threshold}")

    def _detect_features(self, gray_frame, roi):
        """
        Detects good features to track within a specific ROI.
        (Function body remains the same)
        """
        x, y, w, h = roi
        if w <= 0 or h <= 0: return None
        mask = np.zeros_like(gray_frame); roi_y_end = min(gray_frame.shape[0], y + h); roi_x_end = min(gray_frame.shape[1], x + w)
        roi_y_start = max(0, y); roi_x_start = max(0, x)
        if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end: return None
        mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = 255
        try:
            features = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **self.feature_params)
            num_detected = len(features) if features is not None else 0
            # print(f"[FeatureTracker DETECT] Detected {num_detected} features in ROI {roi}") # Noisy
            return features
        except cv2.error as e_gftt: print(f"[FeatureTracker DETECT] OpenCV Error detecting features in ROI {roi}: {e_gftt}"); return None
        except Exception as e: print(f"[FeatureTracker DETECT] Error detecting features in ROI {roi}: {e}"); traceback.print_exc(); return None

    def process_frame(self, current_gray_frame, current_rois):
        """
        Processes a grayscale frame to track features within the given ROIs.
        Redetection is triggered if the number of *successfully tracked* points
        in the previous frame drops below the threshold (and tracking was attempted).
        """
        self._frame_counter += 1
        if current_gray_frame is None:
            print(f"[FeatureTracker Frame {self._frame_counter}] Error: Received None frame.")
            return [(None, None)] * len(current_rois)

        tracked_data_all_rois = []
        next_features_to_track = {} # Features for next tracking attempt
        current_tracked_count = {} # Count from THIS frame's tracking attempt
        did_detect_this_frame = {} # Track if detection happened THIS frame

        for i, roi in enumerate(current_rois):
            good_old_points, good_new_points = None, None # Defaults for this ROI's output this frame
            features_for_current_tracking = self.features_to_track_per_roi.get(i)
            num_features_for_current = len(features_for_current_tracking) if features_for_current_tracking is not None else 0
            last_tracked_count = self.last_tracked_count_per_roi.get(i, 0)
            was_detection_last_frame = self.did_detect_last_frame_per_roi.get(i, True) # Default to True if no history

            # --- Determine if Redetection is Needed ---
            needs_redetection = False
            redetection_reason = ""
            if self.prev_gray_frame is None:
                needs_redetection = True
                redetection_reason = "First frame"
            elif features_for_current_tracking is None or num_features_for_current == 0:
                 needs_redetection = True
                 redetection_reason = "No features available from previous cycle"
            # --- *** CORRECTED LOGIC V3 *** ---
            # Check threshold ONLY IF the last action was TRACKING (not detection)
            elif not was_detection_last_frame:
                 if last_tracked_count < self.redetect_threshold:
                     needs_redetection = True
                     redetection_reason = f"Last tracked count below threshold ({last_tracked_count}/{self.redetect_threshold}) after tracking attempt"
            # --- *** END CORRECTION V3 *** ---

            # --- Feature Detection Logic ---
            if needs_redetection:
                print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] Redetecting features. Reason: {redetection_reason}")
                detection_frame = self.prev_gray_frame if self.prev_gray_frame is not None else current_gray_frame
                detected_features = self._detect_features(detection_frame, roi)
                next_features_to_track[i] = detected_features
                current_tracked_count[i] = 0 # No points tracked this frame
                did_detect_this_frame[i] = True # Mark that detection happened
                tracked_data_all_rois.append((None, None))
                # print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] Output: (None, None) due to redetection.") # Noisy
                continue # Skip tracking

            # --- Feature Tracking Logic ---
            # This block runs ONLY if needs_redetection is False
            num_tracked_ok = 0 # Reset count for this frame's tracking attempt
            did_detect_this_frame[i] = False # Mark that tracking was attempted
            try:
                if features_for_current_tracking.dtype != np.float32:
                     features_for_current_tracking = features_for_current_tracking.astype(np.float32)

                next_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray_frame, current_gray_frame, features_for_current_tracking, None, **self.lk_params
                )

                if next_points is not None and status is not None:
                    status_flat = status.flatten(); valid_mask = (status_flat == 1)
                    num_tracked_ok = np.sum(valid_mask)
                    # print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] LK Tracking: {num_tracked_ok}/{num_features_for_current} points tracked successfully.") # Noisy

                    if num_tracked_ok > 0:
                        good_new_points = next_points[valid_mask]
                        good_old_points = features_for_current_tracking[valid_mask]
                        next_features_to_track[i] = good_new_points.reshape(-1, 1, 2)
                    else: # Tracking ran but lost all points
                         next_features_to_track[i] = None
                         # Output remains (None, None)
                else: # Tracking failed (returned None)
                     next_features_to_track[i] = None
                     # Output remains (None, None)

            except cv2.error as e_lk:
                print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] OpenCV Error tracking features: {e_lk}")
                next_features_to_track[i] = None; num_tracked_ok = 0
            except Exception as e:
                print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] Error tracking features: {e}")
                traceback.print_exc(); next_features_to_track[i] = None; num_tracked_ok = 0

            # Store the number of points successfully tracked *this frame*
            current_tracked_count[i] = num_tracked_ok
            # Append the tracked points (or None if failed) for this ROI
            tracked_data_all_rois.append((good_old_points, good_new_points))

        # --- Update State for Next Frame ---
        self.prev_gray_frame = current_gray_frame.copy()
        self.features_to_track_per_roi = next_features_to_track # Features for next tracking attempt
        self.last_tracked_count_per_roi = current_tracked_count # Count from THIS frame's tracking attempt
        self.did_detect_last_frame_per_roi = did_detect_this_frame # Store action for next frame's check

        return tracked_data_all_rois

# Example usage (for testing this module directly)
# (Test block remains the same)
if __name__ == '__main__':
    print("Testing FeatureTracker module...")
    # ... (rest of the test code is unchanged) ...
