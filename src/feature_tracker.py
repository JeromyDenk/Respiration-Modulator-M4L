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
                'FEATURE_QUALITY_WEIGHTING_ENABLED' (bool): Enable Shi-Tomasi quality weighting.
                'FEATURE_MIN_QUALITY_SCORE' (float): Minimum Shi-Tomasi score to keep a feature.
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

        # --- NEW: Quality Weighting Settings ---
        self.quality_weighting_enabled = config.get('FEATURE_QUALITY_WEIGHTING_ENABLED', False)
        self.min_quality_score = config.get('FEATURE_MIN_QUALITY_SCORE', 0.001) # Default to a small positive value
        # --- END Quality Weighting Settings ---

        # --- Internal State ---
        self.prev_gray_frame = None
        # Stores the features from the PREVIOUS frame that will be USED for tracking in the CURRENT frame
        self.features_to_track_per_roi = {}
        # Stores the count of features successfully tracked IN the previous frame
        self.last_tracked_count_per_roi = {}
        # Flag to indicate if the *previous* frame performed detection (True) or tracking (False)
        # --- NEW: Store qualities corresponding to features_to_track_per_roi ---
        self.feature_qualities_per_roi = {}
        self.did_detect_last_frame_per_roi = {}

        # Debug counter
        self._frame_counter = 0

        print("[FeatureTracker] Initialized.")
        print(f"  Feature Params: {self.feature_params}")
        print(f"  LK Params: {self.lk_params}")
        print(f"  Redetect Threshold (applied to tracked count): {self.redetect_threshold}")
        print(f"  Quality Weighting Enabled: {self.quality_weighting_enabled}")
        if self.quality_weighting_enabled:
            print(f"  Min Quality Score: {self.min_quality_score}")

    def _detect_features(self, gray_frame, roi):
        """
        Detects good features to track within a specific ROI.
        """
        x, y, w, h = roi
        if w <= 0 or h <= 0: return None
        mask = np.zeros_like(gray_frame); roi_y_end = min(gray_frame.shape[0], y + h); roi_x_end = min(gray_frame.shape[1], x + w)
        roi_y_start = max(0, y); roi_x_start = max(0, x)
        if roi_y_start >= roi_y_end or roi_x_start >= roi_x_end: return None
        mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = 255
        try:
            # --- Detect Features ---
            features = cv2.goodFeaturesToTrack(gray_frame, mask=mask, **self.feature_params)
            num_detected = len(features) if features is not None else 0
            if num_detected == 0:
                return None, None # No features detected

            # --- Calculate Quality Scores (if enabled) ---
            qualities = None
            if self.quality_weighting_enabled:
                # Calculate eigenvalue map - use same block size as goodFeaturesToTrack
                block_size = self.feature_params.get('blockSize', 7)
                eigen_map = cv2.cornerMinEigenVal(gray_frame, blockSize=block_size)

                qualities_list = []
                valid_features_list = []
                for pt in features.reshape(-1, 2): # Iterate through (x, y) pairs
                    # --- Use getRectSubPix for bilinear interpolation ---
                    quality_val = cv2.getRectSubPix(eigen_map, (1, 1), (pt[0], pt[1]))[0, 0]

                    # --- Filter by minimum quality score ---
                    if quality_val >= self.min_quality_score:
                        qualities_list.append(quality_val)
                        valid_features_list.append(pt) # Keep the corresponding feature

                if not valid_features_list: # Check if all features were filtered out
                    return None, None

                # Convert back to the required shape (N, 1, 2) for features
                features = np.array(valid_features_list, dtype=np.float32).reshape(-1, 1, 2)
                qualities = np.array(qualities_list, dtype=np.float32)
                # print(f"[FeatureTracker DETECT] ROI {roi}: Detected {num_detected}, kept {len(features)} after quality filter ({self.min_quality_score=}).")

            return features, qualities # Return both features and their qualities (or None)
        except cv2.error as e_gftt: print(f"[FeatureTracker DETECT] OpenCV Error detecting features in ROI {roi}: {e_gftt}"); return None
        except Exception as e: print(f"[FeatureTracker DETECT] Error detecting features in ROI {roi}: {e}"); traceback.print_exc(); return None, None

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
        next_feature_qualities = {} # Corresponding qualities for next attempt
        current_tracked_count = {} # Count from THIS frame's tracking attempt
        did_detect_this_frame = {} # Track if detection happened THIS frame

        for i, roi in enumerate(current_rois):
            # Initialize output for this ROI for the current frame
            current_frame_status = "UNKNOWN_STATUS" # Default status
            current_frame_old_points = None         # Old points used for LK, corresponding to new_points
            current_frame_new_points = None         # Points tracked/detected in current_gray_frame
            current_frame_qualities = None          # Qualities of current_frame_new_points

            features_for_current_tracking = self.features_to_track_per_roi.get(i)
            qualities_for_current_tracking = self.feature_qualities_per_roi.get(i) # Get corresponding qualities
            last_tracked_count = self.last_tracked_count_per_roi.get(i, 0)
            was_detection_last_frame = self.did_detect_last_frame_per_roi.get(i, True) # Default to True if no history

            # --- Determine if Redetection is Needed ---
            needs_redetection = False
            redetection_reason = ""
            if self.prev_gray_frame is None:
                needs_redetection = True
                redetection_reason = "First frame"
            elif features_for_current_tracking is None or len(features_for_current_tracking) == 0:
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
                detected_features, detected_qualities = self._detect_features(detection_frame, roi) # Get qualities too

                if detected_features is not None:
                    current_frame_status = "DETECT_SUCCESS"
                else:
                    current_frame_status = "DETECT_NO_FEATURES"

                next_features_to_track[i] = detected_features
                next_feature_qualities[i] = detected_qualities # Store detected qualities
                current_tracked_count[i] = 0 # No points tracked this frame
                did_detect_this_frame[i] = True # Mark that detection happened
                # print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] Output: (None, None) due to redetection.") # Noisy
                tracked_data_all_rois.append((current_frame_status, None, None, None)) # No old/new_points or qualities from tracking this frame
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

                    if num_tracked_ok > 0: # If some points were tracked successfully
                        current_frame_status = "TRACK_OK"
                        current_frame_old_points = features_for_current_tracking[valid_mask] # These are the old points
                        current_frame_new_points = next_points[valid_mask]
                        next_features_to_track[i] = current_frame_new_points.reshape(-1, 1, 2)

                        # --- Keep corresponding qualities ---
                        if self.quality_weighting_enabled and qualities_for_current_tracking is not None:
                            current_frame_qualities = qualities_for_current_tracking[valid_mask]
                            next_feature_qualities[i] = current_frame_qualities # Store qualities for next frame
                        else:
                            # current_frame_qualities remains None
                            next_feature_qualities[i] = None # No qualities if weighting disabled or missing
                    else: # Tracking ran but lost all points
                         current_frame_status = "TRACK_LOST_ALL"                         
                         # old_points, new_points, qualities remain None
                         next_features_to_track[i] = None
                         next_feature_qualities[i] = None
                else: # Tracking failed (returned None)
                     current_frame_status = "TRACK_LK_FAILED" # e.g. calcOpticalFlowPyrLK returned None                     
                     # old_points, new_points, qualities remain None                     
                     next_features_to_track[i] = None
                     next_feature_qualities[i] = None

            except cv2.error as e_lk: # Handle OpenCV specific errors
                print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] OpenCV Error tracking features: {e_lk}")
                current_frame_status = "TRACK_ERROR_CV"
                next_features_to_track[i] = None; num_tracked_ok = 0
                next_feature_qualities[i] = None # Ensure qualities are reset on error
            except Exception as e:
                print(f"[FeatureTracker Frame {self._frame_counter} ROI {i}] Error tracking features: {e}")
                traceback.print_exc()
                current_frame_status = "TRACK_ERROR_GENERIC"
                # old_points, new_points, qualities remain None
                next_features_to_track[i] = None; num_tracked_ok = 0
                next_feature_qualities[i] = None # Ensure qualities are reset on error


            # Store the number of points successfully tracked *this frame*
            current_tracked_count[i] = num_tracked_ok
            # Append the status, old points, new points, and qualities for this ROI
            tracked_data_all_rois.append((current_frame_status, current_frame_old_points, current_frame_new_points, current_frame_qualities))

        # --- Update State for Next Frame ---
        self.prev_gray_frame = current_gray_frame.copy()
        self.features_to_track_per_roi = next_features_to_track # Features for next tracking attempt
        self.feature_qualities_per_roi = next_feature_qualities # Store qualities for next attempt
        self.last_tracked_count_per_roi = current_tracked_count # Count from THIS frame's tracking attempt
        self.did_detect_last_frame_per_roi = did_detect_this_frame # Store action for next frame's check

        return tracked_data_all_rois

# Example usage (for testing this module directly)
# (Test block remains the same)
if __name__ == '__main__':
    print("Testing FeatureTracker module...")
    # ... (rest of the test code is unchanged) ...
