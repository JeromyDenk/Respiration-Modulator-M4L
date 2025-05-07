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
                'ROI_HORIZONTAL_SCALE_FACTOR' (float): Multiplier for scaling ROI width.
                'ROI_TOP_PADDING_RATIO' (float): Ratio of initial height to add as top padding.
                'ROI_BOTTOM_PADDING_RATIO' (float): Ratio of initial height to add as bottom padding.
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
        
        # New padding parameters
        self.horizontal_scale_factor = config.get('ROI_HORIZONTAL_SCALE_FACTOR', 1.0)
        self.top_padding_ratio = config.get('ROI_TOP_PADDING_RATIO', 0.0)
        self.bottom_padding_ratio = config.get('ROI_BOTTOM_PADDING_RATIO', 0.0)

        self.min_visibility = config.get('POSE_MIN_LANDMARK_VISIBILITY', 0.6) # Default visibility threshold
        self.shoulder_aspect_ratio = config.get('ROI_SHOULDER_ONLY_ASPECT_RATIO', 1.8) # Estimated H/W ratio for shoulder-only case

        print(f"[CoarseRoiCalculator] Initialized with Strategy: {self.strategy}") # Class name updated in log
        print(f"  Shoulder Landmarks: {[lm.name for lm in self.shoulder_landmarks]}")
        print(f"  Hip Landmarks: {[lm.name for lm in self.hip_landmarks]}")
        print(f"  Horizontal Scale Factor: {self.horizontal_scale_factor}, Top Padding Ratio: {self.top_padding_ratio}, Bottom Padding Ratio: {self.bottom_padding_ratio}")
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

    def _apply_padding_and_clamping(self, initial_x, initial_y, initial_w, initial_h, frame_shape):
        """Applies configured padding and clamps ROI to frame boundaries."""
        frame_height, frame_width = frame_shape
        
        if initial_w <= 0 or initial_h <= 0:
            return 0, 0, 0, 0 # Invalid initial ROI

        # 1. Apply horizontal scaling
        padded_w = initial_w * self.horizontal_scale_factor
        # Adjust x to keep the ROI centered horizontally after width scaling
        padded_x = initial_x - (padded_w - initial_w) / 2

        # 2. Apply vertical padding
        padding_top_pixels = initial_h * self.top_padding_ratio
        padding_bottom_pixels = initial_h * self.bottom_padding_ratio

        # Adjust y for top padding
        padded_y = initial_y - padding_top_pixels
        # Adjust height for both top and bottom padding
        padded_h = initial_h + padding_top_pixels + padding_bottom_pixels
        
        # Convert to int before clamping
        padded_x = int(padded_x)
        padded_y = int(padded_y)
        padded_w = int(padded_w)
        padded_h = int(padded_h)

        # Clamp to frame boundaries
        final_x = max(0, padded_x)
        final_y = max(0, padded_y)
        # Ensure width and height don't extend beyond frame, considering the new x, y
        final_w = min(padded_w, frame_width - final_x)
        final_h = min(padded_h, frame_height - final_y)

        # Ensure width and height are not negative after clamping
        final_w = max(0, final_w)
        final_h = max(0, final_h)

        return final_x, final_y, final_w, final_h

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
        shoulder_coords, _ = self._get_visible_landmark_coords(
            landmarks, self.shoulder_landmarks, frame_shape
        )
        if len(shoulder_coords) != len(self.shoulder_landmarks): # Need both shoulders
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

                initial_roi_x, initial_roi_y, initial_roi_w, initial_roi_h = 0, 0, 0, 0

                # --- Determine ROI bounds based on visibility ---
                if hips_sufficiently_visible:
                    left_hip = hip_coords.get(mp_pose.PoseLandmark.LEFT_HIP)
                    right_hip = hip_coords.get(mp_pose.PoseLandmark.RIGHT_HIP)
                    if not left_hip or not right_hip: return [] # Should have hips

                    all_points = [left_shoulder, right_shoulder, left_hip, right_hip]
                    coords_array = np.array(all_points)
                    initial_roi_x = np.min(coords_array[:, 0])
                    max_x = np.max(coords_array[:, 0])
                    initial_roi_y = np.min(coords_array[0:2, 1]) # Shoulders define top
                    max_y = np.max(coords_array[2:4, 1]) # Hips define bottom
                    initial_roi_w = max_x - initial_roi_x
                    initial_roi_h = max_y - initial_roi_y
                else: # Degraded Case: Only Shoulders visible
                    shoulder_points_arr = np.array([left_shoulder, right_shoulder])
                    initial_roi_x = np.min(shoulder_points_arr[:, 0])
                    max_x = np.max(shoulder_points_arr[:, 0])
                    initial_roi_y = np.min(shoulder_points_arr[:, 1])
                    initial_roi_w = max_x - initial_roi_x
                    if initial_roi_w <= 0: return []
                    initial_roi_h = int(initial_roi_w * self.shoulder_aspect_ratio)
                
                if initial_roi_w <= 0 or initial_roi_h <= 0:
                     return []

                final_x, final_y, final_w, final_h = self._apply_padding_and_clamping(
                    initial_roi_x, initial_roi_y, initial_roi_w, initial_roi_h, frame_shape
                )

                if final_w > 0 and final_h > 0:
                    calculated_rois.append((final_x, final_y, final_w, final_h))

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
        mock_landmarks.LEFT_SHOULDER: {'x': 0.3, 'y': 0.2, 'visibility': 0.9}, # x=192, y=96
        mock_landmarks.RIGHT_SHOULDER: {'x': 0.7, 'y': 0.2, 'visibility': 0.9},# x=448, y=96
        mock_landmarks.LEFT_HIP: {'x': 0.35, 'y': 0.6, 'visibility': 0.8},    # x=224, y=288
        mock_landmarks.RIGHT_HIP: {'x': 0.65, 'y': 0.6, 'visibility': 0.8},   # x=416, y=288
    }
    landmarks_data_shoulders_only = {
        mock_landmarks.LEFT_SHOULDER: {'x': 0.3, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.RIGHT_SHOULDER: {'x': 0.7, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.LEFT_HIP: {'x': 0.35, 'y': 0.6, 'visibility': 0.1}, # Low visibility
        mock_landmarks.RIGHT_HIP: {'x': 0.65, 'y': 0.6, 'visibility': 0.1}, # Low visibility
    }
    landmarks_data_no_shoulders = {
        mock_landmarks.LEFT_SHOULDER: {'x': 0.3, 'y': 0.2, 'visibility': 0.1}, # Low visibility
        mock_landmarks.RIGHT_SHOULDER: {'x': 0.7, 'y': 0.2, 'visibility': 0.9},
        mock_landmarks.LEFT_HIP: {'x': 0.35, 'y': 0.6, 'visibility': 0.8},
        mock_landmarks.RIGHT_HIP: {'x': 0.65, 'y': 0.6, 'visibility': 0.8},
    }

    class MockLandmarkProto: # Renamed to avoid conflict if mediapipe.framework.formats.landmark_pb2.NormalizedLandmark is imported
        def __init__(self, x, y, visibility): self.x, self.y, self.visibility = x, y, visibility
    class MockLandmarkListProto: # Renamed
        def __init__(self):
            # Use max value + 1 for list size
            max_idx = max(lm.value for lm in mp_pose.PoseLandmark)
            self.landmark = [MockLandmarkProto(0,0,0)] * (max_idx + 1) # Initialize with dummy landmarks
        def add(self, index, x, y, visibility):
             if 0 <= index.value < len(self.landmark): self.landmark[index.value] = MockLandmarkProto(x, y, visibility)

    def create_mock_list_proto(data): # Renamed
        lm_list = MockLandmarkListProto()
        for index, d_val in data.items(): lm_list.add(index, d_val['x'], d_val['y'], d_val['visibility'])
        return lm_list

    mock_pose_landmarks_full = create_mock_list_proto(landmarks_data_full)
    mock_pose_landmarks_shoulders = create_mock_list_proto(landmarks_data_shoulders_only)
    mock_pose_landmarks_fail = create_mock_list_proto(landmarks_data_no_shoulders)

    frame_shape_test = (480, 640) # height, width

    # --- Test Cases ---

    print("\n--- Test 1: Full Case (Default Padding: 1.0, 0.0, 0.0) ---")
    calculator_default = CoarseRoiCalculator(config={
        'POSE_MIN_LANDMARK_VISIBILITY': 0.5,
        # ROI_HORIZONTAL_SCALE_FACTOR defaults to 1.0
        # ROI_TOP_PADDING_RATIO defaults to 0.0
        # ROI_BOTTOM_PADDING_RATIO defaults to 0.0
    })
    rois_default = calculator_default.calculate_coarse_roi(mock_pose_landmarks_full, frame_shape_test)
    print(f"Calculated ROIs (default padding): {rois_default}")
    # Expected initial: x=0.3*640=192, y_sh=0.2*480=96, x_max_sh=0.7*640=448 -> w_sh = 256
    #                   y_hip=0.6*480=288. So, initial_x=192, initial_y=96, initial_w=256, initial_h=288-96=192
    # With default padding (no change): [(192, 96, 256, 192)]
    if rois_default:
        assert rois_default[0] == (192, 96, 256, 192), f"Test 1 Failed. Got {rois_default[0]}"
    else:
        assert False, "Test 1 Failed. No ROI returned."
    print("Test 1 Passed.")

    print("\n--- Test 2: Full Case (Custom Padding) ---")
    custom_config = {
        'POSE_MIN_LANDMARK_VISIBILITY': 0.5,
        'ROI_HORIZONTAL_SCALE_FACTOR': 1.1,
        'ROI_TOP_PADDING_RATIO': 0.1,
        'ROI_BOTTOM_PADDING_RATIO': 0.05
    }
    calculator_custom = CoarseRoiCalculator(config=custom_config)
    rois_custom = calculator_custom.calculate_coarse_roi(mock_pose_landmarks_full, frame_shape_test)
    print(f"Calculated ROIs (custom padding): {rois_custom}")
    # Initial: x=192, y=96, w=256, h=192
    # Padded w = 256 * 1.1 = 281.6 -> 281
    # Padded x = 192 - (281.6 - 256)/2 = 192 - 12.8 = 179.2 -> 179
    # Top pad = 192 * 0.1 = 19.2
    # Bottom pad = 192 * 0.05 = 9.6
    # Padded y = 96 - 19.2 = 76.8 -> 76
    # Padded h = 192 + 19.2 + 9.6 = 220.8 -> 220
    # Expected: (179, 76, 281, 220)
    if rois_custom:
        assert rois_custom[0] == (179, 76, 281, 220), f"Test 2 Failed. Got {rois_custom[0]}"
    else:
        assert False, "Test 2 Failed. No ROI returned."
    print("Test 2 Passed.")

    print("\n--- Test 3: Degraded Case (Shoulders Only, Custom Padding) ---")
    calculator_shoulders = CoarseRoiCalculator(config=custom_config) # Use same custom padding
    rois_shoulders = calculator_shoulders.calculate_coarse_roi(mock_pose_landmarks_shoulders, frame_shape_test)
    print(f"Calculated ROIs (shoulders only, custom padding): {rois_shoulders}")
    # Initial shoulder: x=192, y_sh=96, w_sh=256.
    # Estimated h = 256 * 1.8 (default aspect_ratio) = 460.8
    # Padded w = 256 * 1.1 = 281.6 -> 281
    # Padded x = 179.2 -> 179
    # Top pad = 460.8 * 0.1 = 46.08
    # Bottom pad = 460.8 * 0.05 = 23.04
    # Padded y = 96 - 46.08 = 49.92 -> 49
    # Padded h = 460.8 + 46.08 + 23.04 = 529.92 -> 529
    # Clamped h = min(529, 480 - 49) = min(529, 431) = 431
    # Expected: (179, 49, 281, 431)
    if rois_shoulders:
        assert rois_shoulders[0] == (179, 49, 281, 431), f"Test 3 Failed. Got {rois_shoulders[0]}"
    else:
        assert False, "Test 3 Failed. No ROI returned."
    print("Test 3 Passed.")

    print("\n--- Test 3: Failure Case (Shoulders Not Visible) ---")
    # Use an existing calculator instance, e.g., calculator_default
    rois_fail = calculator_default.calculate_coarse_roi(mock_pose_landmarks_fail, frame_shape_test)
    print(f"Calculated ROIs (no shoulders): {rois_fail}")
    assert len(rois_fail) == 0, "Should calculate zero ROIs when shoulders are not visible"

    print("\n--- Test 4: No Landmarks Input ---")
    # Use an existing calculator instance
    rois_none = calculator_default.calculate_coarse_roi(None, frame_shape_test)
    print(f"Calculated ROIs (None input): {rois_none}")
    assert len(rois_none) == 0, "Should calculate zero ROIs for None input"

    print("\nCoarseRoiCalculator module test finished.") # Updated name
