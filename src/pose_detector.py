# src/pose_detector.py
# Encapsulates MediaPipe Pose detection.

import cv2
import mediapipe as mp
import numpy as np
import traceback

# Define constants for MediaPipe Pose components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseDetector:
    """
    Handles loading the MediaPipe Pose model and detecting landmarks in frames.
    """
    def __init__(self, config=None):
        """
        Initializes the MediaPipe Pose model.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'POSE_MODEL_COMPLEXITY' (0, 1, or 2)
                'POSE_STATIC_IMAGE_MODE' (bool)
                'POSE_MIN_DETECTION_CONFIDENCE' (float 0.0-1.0)
                'POSE_MIN_TRACKING_CONFIDENCE' (float 0.0-1.0)
                Defaults are used if config is None or keys are missing.
        """
        if config is None:
            config = {} # Use defaults if no config provided

        model_complexity = config.get('POSE_MODEL_COMPLEXITY', 1) # Default to 1 (balance)
        static_image_mode = config.get('POSE_STATIC_IMAGE_MODE', False) # Default to False for video
        min_detection_confidence = config.get('POSE_MIN_DETECTION_CONFIDENCE', 0.5)
        min_tracking_confidence = config.get('POSE_MIN_TRACKING_CONFIDENCE', 0.5)

        print(f"[PoseDetector] Initializing MediaPipe Pose...")
        print(f"  Config: Complexity={model_complexity}, StaticMode={static_image_mode}, MinDetectConf={min_detection_confidence}, MinTrackConf={min_tracking_confidence}")

        try:
            # --- Initialize MediaPipe Pose ---
            # Using 'with' ensures resources are managed properly
            # Note: We create the object here, but processing happens in process_frame
            self.pose_processor = mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                enable_segmentation=False, # Segmentation not needed for landmarks
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)
            print("[PoseDetector] MediaPipe Pose initialized successfully.")
            self.initialized = True
        except Exception as e:
            print(f"[PoseDetector] FATAL ERROR initializing MediaPipe Pose: {e}")
            traceback.print_exc()
            self.pose_processor = None
            self.initialized = False

    def process_frame(self, frame_rgb):
        """
        Processes a single frame to detect pose landmarks.

        Args:
            frame_rgb (np.ndarray): The input frame in RGB format.

        Returns:
            mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList or None:
            Detected pose landmarks, or None if detection failed or not initialized.
        """
        if not self.initialized or self.pose_processor is None:
            # print("[PoseDetector] Error: Not initialized.") # Can be noisy
            return None
        if frame_rgb is None:
            # print("[PoseDetector] Error: Input frame is None.") # Can be noisy
            return None

        try:
            # Process the frame with MediaPipe Pose
            # Make image non-writeable to pass input by reference (performance hint)
            frame_rgb.flags.writeable = False
            results = self.pose_processor.process(frame_rgb)
            frame_rgb.flags.writeable = True # Make it writeable again if needed later

            # Return only the landmarks object
            return results.pose_landmarks

        except Exception as e:
            print(f"[PoseDetector] Error during pose processing: {e}")
            # traceback.print_exc() # Uncomment for detailed debugging
            return None

    def close(self):
        """Releases MediaPipe Pose resources."""
        if hasattr(self, 'pose_processor') and self.pose_processor:
            print("[PoseDetector] Closing MediaPipe Pose...")
            # The 'with' statement in __init__ might handle closing,
            # but explicit closing can be added if issues arise.
            # self.pose_processor.close() # Usually not needed if used via 'with' or if it manages its own lifecycle
            self.pose_processor = None
            self.initialized = False
            print("[PoseDetector] Closed.")

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing PoseDetector module...")

    # --- Create a dummy black image ---
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # --- Test with default config ---
    print("\n--- Test 1: Default Config ---")
    detector_default = PoseDetector()
    if detector_default.initialized:
        landmarks_default = detector_default.process_frame(dummy_frame)
        print(f"Detected landmarks (default): {'Yes' if landmarks_default else 'No'}")
        detector_default.close()
    else:
        print("Initialization failed (default).")

    # --- Test with specific config ---
    print("\n--- Test 2: Specific Config ---")
    test_config = {
        'POSE_MODEL_COMPLEXITY': 0, # Lite model
        'POSE_STATIC_IMAGE_MODE': True,
        'POSE_MIN_DETECTION_CONFIDENCE': 0.6
    }
    detector_specific = PoseDetector(config=test_config)
    if detector_specific.initialized:
        landmarks_specific = detector_specific.process_frame(dummy_frame)
        print(f"Detected landmarks (specific): {'Yes' if landmarks_specific else 'No'}")
        detector_specific.close()
    else:
        print("Initialization failed (specific).")

    # --- Test with invalid input ---
    print("\n--- Test 3: Invalid Input ---")
    detector_invalid = PoseDetector()
    if detector_invalid.initialized:
        landmarks_invalid = detector_invalid.process_frame(None) # Pass None
        print(f"Detected landmarks (invalid input): {'Yes' if landmarks_invalid else 'No'}")
        detector_invalid.close()

    print("\nPoseDetector module test finished.")