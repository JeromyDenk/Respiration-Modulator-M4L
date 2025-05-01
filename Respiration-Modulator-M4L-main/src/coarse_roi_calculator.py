# src/coarse_roi_calculator.py
# Calculates a COARSE ROI based on detected pose landmarks.
# REFACTORED from roi_calculator.py

import mediapipe as mp
import numpy as np
import traceback # For debugging potential errors

# Define constants for easier landmark access
mp_pose = mp.solutions.pose

# Renamed class
class CoarseRoiCalculator:
    """
    Calculates a coarse Region of Interest (ROI) based on MediaPipe Pose landmarks.
    This ROI is intended as input for further refinement (e.g., by EVM).
    Handles cases with full torso visibility (shoulders + hips) and
    degraded visibility (shoulders only).
    """
    def __init__(self, config=None):
        """
        Initializes the CoarseRoiCalculator.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'ROI_STRATEGY' (str): 'single_chest_abdomen', etc. (Currently only supports single)
                'ROI_LANDMARKS_SHOULDERS' (list): Indices/names for shoulders.
                'ROI_LANDMARKS_HIPS' (list): Indices/names for hips.
                'ROI_PADDING_FACTOR' (float): Multiplier for padding the ROI.
                'POSE_MIN_LANDMARK_VISIBILITY' (float): Min visibility score (0.0-1.0).
                'ROI_SHOULDER_ONLY_ASPECT_RATIO' (float): Estimated H/W ratio when only shoulders visible.
                Defaults are used if config is None or keys are missing.
        """
        if config is None:
            config = {}

        # Config keys might still refer to 'ROI_' for backward compatibility in profile files,
        # or you can update profile files to use 'COARSE_ROI_' prefixes.
        self.strategy = config.get('ROI_STRATEGY', 'single_chest_abdomen')
        # Define the required landmarks for each case
        self.shoulder_landmarks = config.get('ROI_LANDMARKS_SHOULDERS', [
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ])
        self.hip_landmarks = config.get('ROI_LANDMARKS_HIPS', [
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP
        ])
        self.padding_factor = config.get('ROI_PADDING_FACTOR', 1.05) # Default 5% padding
        self.min_visibility = config.get('POSE_MIN_LANDMARK_VISIBILITY', 0.6) # Default visibility threshold
        self.shoulder_aspect_ratio = config.get('ROI_SHOULDER_ONLY_ASPECT_RATIO', 1.8) # Estimated H/W ratio for shoulder-only case

        print(f"[CoarseRoiCalculator] Initialized with Strategy: {self.strategy}") # Class name updated in log
        print(f"  Shoulder Landmarks: {[lm.name for lm in self.shoulder_landmarks]}")
        print(f"  Hip Landmarks: {[lm.name for lm in self.hip_landmarks]}")
        print(f"  Padding Factor: {self.padding_factor}, Min Visibility: {self.min_visibility}")
        print(f"  Shoulder-Only Aspect Ratio: {self.shoulder_aspect_ratio}")


    def _get_visible_landmark_coords(self, landmarks, landmark_indices, frame_shape):
        """Helper to get pixel coordinates for visible landmarks."""
        visible_coords = {} # Use dict to store {landmark_index: (px, py)}
        all_visible = True
        frame_height, frame_width = frame_shape

        for index in landmark_indices:
            try:
                landmark = landmarks.landmark[index.value] # Access landmark by index value
                if landmark.visibility >= self.min_visibility:
                    px = int(landmark.x * frame_width)
                    py = int(landmark.y * frame_height)
                    visible_coords[index] = (px, py)
                else:
                    # print(f"[CoarseRoiCalculator] Debug: Landmark {index.name} visibility {landmark.visibility:.2f} < {self.min_visibility}") # Debug noise
                    all_visible = False
            except IndexError:
                 print(f"[CoarseRoiCalculator] Error: Landmark index {index.value} out of bounds.")
                 all_visible = False
            except Exception as e:
                 print(f"[CoarseRoiCalculator] Error accessing landmark {index.name}: {e}")
                 all_visible = False

        return visible_coords, all_visible


    # Renamed method
    def calculate_coarse_roi(self, landmarks, frame_shape):
        """
        Calculates a coarse ROI based on the detected landmarks and configured strategy.

        Args:
            landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                The detected pose landmarks from PoseDetector.
            frame_shape (tuple): The shape of the frame (height, width).

        Returns:
            list: A list of ROI tuples [(x, y, w, h), ...]. Returns an empty list
                  if the required landmarks are not sufficiently visible or calculation fails.
                  Currently only returns a single ROI or empty list.
        """
        if landmarks is None: return []
        if not hasattr(landmarks, 'landmark') or not landmarks.landmark:
            # print("[CoarseRoiCalculator] Invalid or empty landmarks object.") # Debug noise
            return []

        frame_height, frame_width = frame_shape
        calculated_rois = []

        # --- Check Shoulder Visibility ---
        shoulder_coords, shoulders_visible = self._get_visible_landmark_coords(
            landmarks, self.shoulder_landmarks, frame_shape
        )
        if len(shoulder_coords) != len(self.shoulder_landmarks): # Need both shoulders
            # print("[CoarseRoiCalculator] Not all required shoulder landmarks are sufficiently visible.") # Debug noise
            return []

        # --- Check Hip Visibility ---
        hip_coords, hips_visible = self._get_visible_landmark_coords(
            landmarks, self.hip_landmarks, frame_shape
        )
        hips_sufficiently_visible = len(hip_coords) == len(self.hip_landmarks)

        # --- Calculate ROI based on strategy and visibility ---
        if self.strategy == 'single_chest_abdomen':
            try:
                # Get shoulder coordinates
                left_shoulder = shoulder_coords.get(mp_pose.PoseLandmark.LEFT_SHOULDER)
                right_shoulder = shoulder_coords.get(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                if not left_shoulder or not right_shoulder: return [] # Should have these

                min_x, max_x, min_y, max_y = 0, 0, 0, 0

                # --- Determine ROI bounds based on visibility ---
                if hips_sufficiently_visible:
                    # print("[CoarseRoiCalculator] Debug: Using Full Case (Shoulders + Hips)") # Debug noise
                    left_hip = hip_coords.get(mp_pose.PoseLandmark.LEFT_HIP)
                    right_hip = hip_coords.get(mp_pose.PoseLandmark.RIGHT_HIP)
                    if not left_hip or not right_hip: return [] # Should have hips

                    all_points = [left_shoulder, right_shoulder, left_hip, right_hip]
                    coords_array = np.array(all_points)
                    min_x = np.min(coords_array[:, 0])
                    max_x = np.max(coords_array[:, 0])
                    min_y = np.min(coords_array[0:2, 1]) # Shoulders define top
                    max_y = np.max(coords_array[2:4, 1]) # Hips define bottom

                else: # Degraded Case: Only Shoulders visible
                    # print("[CoarseRoiCalculator] Debug: Using Degraded Case (Shoulders Only)") # Debug noise
                    shoulder_points = [left_shoulder, right_shoulder]
                    coords_array = np.array(shoulder_points)
                    min_x = np.min(coords_array[:, 0])
                    max_x = np.max(coords_array[:, 0])
                    min_y = np.min(coords_array[:, 1]) # Top edge is shoulder line

                    shoulder_width = max_x - min_x
                    if shoulder_width <= 0: return []
                    estimated_height = int(shoulder_width * self.shoulder_aspect_ratio)
                    max_y = min_y + estimated_height # Bottom edge estimated

                # --- Calculate Padded ROI from bounds ---
                roi_w = max_x - min_x
                roi_h = max_y - min_y

                if roi_w <= 0 or roi_h <= 0:
                     # print(f"[CoarseRoiCalculator] Warning: Calculated ROI has zero/negative dims before padding (w={roi_w}, h={roi_h}).") # Debug noise
                     return []

                center_x = min_x + roi_w / 2
                center_y = min_y + roi_h / 2
                padded_w = int(roi_w * self.padding_factor)
                padded_h = int(roi_h * self.padding_factor)

                padded_x = int(center_x - padded_w / 2)
                padded_y = int(center_y - padded_h / 2)

                # Clamp coordinates to frame boundaries
                final_x = max(0, padded_x)
                final_y = max(0, padded_y)
                final_w = min(padded_w, frame_width - final_x)
                final_h = min(padded_h, frame_height - final_y)

                if final_w > 0 and final_h > 0:
                    calculated_rois.append((final_x, final_y, final_w, final_h))
                # else:
                     # print("[CoarseRoiCalculator] Warning: ROI dimensions became invalid after padding/clamping.") # Debug noise

            except Exception as e:
                print(f"[CoarseRoiCalculator] Error during '{self.strategy}' calculation: {e}")
                traceback.print_exc()
                return []

        # --- Placeholder for potential future multi-ROI strategies ---
        # elif self.strategy == 'multi_chest_abdomen':
        #      print("[CoarseRoiCalculator] Warning: 'multi_chest_abdomen' strategy not yet implemented.")
        #      pass
        else:
            print(f"[CoarseRoiCalculator] Error: Unknown ROI strategy '{self.strategy}'")
            return []

        return calculated_rois


# Example usage block (updated class/method names)
if __name__ == '__main__':
    print("Testing CoarseRoiCalculator module...") # Updated name

    # --- Mock Landmarks ---
    # (Mock landmark setup remains the same as original RoiCalculator test)
    mock_landmarks = mp_pose.PoseLandmark
    landmarks_data_full = {
        mock_landmarks.LEFT_SHOULDER: {'x': 0.3, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.RIGHT_SHOULDER: {'x': 0.7, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.LEFT_HIP: {'x': 0.35, 'y': 0.6, 'visibility': 0.8},
        mock_landmarks.RIGHT_HIP: {'x': 0.65, 'y': 0.6, 'visibility': 0.8},
        mock_landmarks.NOSE: {'x': 0.5, 'y': 0.1, 'visibility': 0.95},
    }
    landmarks_data_shoulders_only = {
        mock_landmarks.LEFT_SHOULDER: {'x': 0.3, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.RIGHT_SHOULDER: {'x': 0.7, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.LEFT_HIP: {'x': 0.35, 'y': 0.6, 'visibility': 0.1}, # Low visibility
        mock_landmarks.RIGHT_HIP: {'x': 0.65, 'y': 0.6, 'visibility': 0.1}, # Low visibility
        mock_landmarks.NOSE: {'x': 0.5, 'y': 0.1, 'visibility': 0.95},
    }
    landmarks_data_no_shoulders = {
        mock_landmarks.LEFT_SHOULDER: {'x': 0.3, 'y': 0.2, 'visibility': 0.1}, # Low visibility
        mock_landmarks.RIGHT_SHOULDER: {'x': 0.7, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.LEFT_HIP: {'x': 0.35, 'y': 0.6, 'visibility': 0.8},
        mock_landmarks.RIGHT_HIP: {'x': 0.65, 'y': 0.6, 'visibility': 0.8},
        mock_landmarks.NOSE: {'x': 0.5, 'y': 0.1, 'visibility': 0.95},
    }

    class MockLandmark:
        def __init__(self, x, y, visibility): self.x, self.y, self.visibility = x, y, visibility
    class MockLandmarkList:
        def __init__(self):
            # Use max value + 1 for list size
            max_idx = max(lm.value for lm in mp_pose.PoseLandmark)
            self.landmark = [None] * (max_idx + 1)
        def add(self, index, x, y, visibility):
             if 0 <= index.value < len(self.landmark): self.landmark[index.value] = MockLandmark(x, y, visibility)

    def create_mock_list(data):
        lm_list = MockLandmarkList()
        for index, d in data.items(): lm_list.add(index, d['x'], d['y'], d['visibility'])
        return lm_list

    mock_pose_landmarks_full = create_mock_list(landmarks_data_full)
    mock_pose_landmarks_shoulders = create_mock_list(landmarks_data_shoulders_only)
    mock_pose_landmarks_fail = create_mock_list(landmarks_data_no_shoulders)

    frame_shape_test = (480, 640) # height, width

    # --- Test Cases ---
    # Use updated class name
    calculator = CoarseRoiCalculator(config={'POSE_MIN_LANDMARK_VISIBILITY': 0.5})

    print("\n--- Test 1: Full Case (Shoulders + Hips Visible) ---")
    # Use updated method name
    rois_full = calculator.calculate_coarse_roi(mock_pose_landmarks_full, frame_shape_test)
    print(f"Calculated ROIs (full): {rois_full}")
    assert len(rois_full) == 1, "Should calculate one ROI for full case"
    assert rois_full[0][2] > 0 and rois_full[0][3] > 0, "ROI dims should be positive"

    print("\n--- Test 2: Degraded Case (Shoulders Only Visible) ---")
    # Use updated method name
    rois_shoulders = calculator.calculate_coarse_roi(mock_pose_landmarks_shoulders, frame_shape_test)
    print(f"Calculated ROIs (shoulders only): {rois_shoulders}")
    assert len(rois_shoulders) == 1, "Should calculate one ROI for shoulders-only case"
    assert rois_shoulders[0][2] > 0 and rois_shoulders[0][3] > 0, "ROI dims should be positive"
    roi_x, roi_y, roi_w, roi_h = rois_shoulders[0]
    expected_ratio = calculator.shoulder_aspect_ratio
    actual_ratio = roi_h / roi_w if roi_w > 0 else 0
    print(f"  Actual H/W Ratio: {actual_ratio:.2f} (Expected ~{expected_ratio})")

    print("\n--- Test 3: Failure Case (Shoulders Not Visible) ---")
    # Use updated method name
    rois_fail = calculator.calculate_coarse_roi(mock_pose_landmarks_fail, frame_shape_test)
    print(f"Calculated ROIs (no shoulders): {rois_fail}")
    assert len(rois_fail) == 0, "Should calculate zero ROIs when shoulders are not visible"

    print("\n--- Test 4: No Landmarks Input ---")
    # Use updated method name
    rois_none = calculator.calculate_coarse_roi(None, frame_shape_test)
    print(f"Calculated ROIs (None input): {rois_none}")
    assert len(rois_none) == 0, "Should calculate zero ROIs for None input"

    print("\nCoarseRoiCalculator module test finished.") # Updated name
