# tests/test_pose_detector.py
# Unit tests for the PoseDetector class.

import unittest
import os
import sys
import cv2
import numpy as np
import mediapipe as mp # Import mediapipe to check results type

# --- Add src directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src') # Go up one level from tests/ to project root, then into src/
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import the class to test ---
try:
    from pose_detector import PoseDetector
except ImportError as e:
    print(f"Error importing PoseDetector from 'src': {e}")
    print("Please ensure this script is run from the project root directory using 'python -m unittest tests.test_pose_detector'")
    print("or that the 'src' directory is correctly added to the Python path.")
    sys.exit(1)

# --- Test Data ---
# Create a simple dummy image (e.g., black) for basic processing tests
# A real image with a person would be better for testing landmark presence.
DUMMY_IMAGE_HEIGHT = 480
DUMMY_IMAGE_WIDTH = 640
# Create RGB image as MediaPipe prefers RGB
dummy_rgb_image = np.zeros((DUMMY_IMAGE_HEIGHT, DUMMY_IMAGE_WIDTH, 3), dtype=np.uint8)

# --- Test Class ---
class TestPoseDetector(unittest.TestCase):
    """Test suite for the PoseDetector class."""

    pose_detector_default = None # Store instance for reuse

    @classmethod
    def setUpClass(cls):
        """Initialize the detector once for all tests (using default config)."""
        print("\n--- Initializing PoseDetector for Tests (Default Config) ---")
        cls.pose_detector_default = PoseDetector(config={}) # Use empty dict for defaults
        if not cls.pose_detector_default.initialized:
             raise RuntimeError("FATAL: Failed to initialize default PoseDetector for testing.")
        print("--- PoseDetector Initialized ---")

    @classmethod
    def tearDownClass(cls):
        """Clean up the detector once after all tests."""
        if cls.pose_detector_default:
            cls.pose_detector_default.close()
        print("\n--- PoseDetector Cleaned Up ---")

    def test_01_initialization(self):
        """Test if the PoseDetector initializes correctly."""
        print("\n--- Running Test: test_01_initialization ---")
        # The setUpClass already tried to initialize it.
        # We just need to check if it succeeded.
        self.assertIsNotNone(self.pose_detector_default, "PoseDetector instance should not be None.")
        self.assertTrue(self.pose_detector_default.initialized, "PoseDetector should be marked as initialized.")
        self.assertIsNotNone(self.pose_detector_default.pose_processor, "Internal MediaPipe Pose processor should be initialized.")
        print("--- Finished Test: test_01_initialization ---")

    def test_02_process_valid_frame(self):
        """Test processing a valid dummy frame."""
        print("\n--- Running Test: test_02_process_valid_frame ---")
        self.assertTrue(self.pose_detector_default.initialized, "Pre-condition failed: Detector not initialized.")
        landmarks = self.pose_detector_default.process_frame(dummy_rgb_image.copy()) # Pass a copy

        # On a dummy black image, it likely won't find landmarks, but the *process* should run without error
        # and return either None or a LandmarkList object.
        print(f"  Result type: {type(landmarks)}")
        is_landmark_list = isinstance(landmarks, mp.framework.formats.landmark_pb2.NormalizedLandmarkList)
        is_none = landmarks is None
        self.assertTrue(is_landmark_list or is_none, "Result should be None or MediaPipe PoseLandmarks")
        print("--- Finished Test: test_02_process_valid_frame ---")

    def test_03_process_none_frame(self):
        """Test processing a None frame."""
        print("\n--- Running Test: test_03_process_none_frame ---")
        self.assertTrue(self.pose_detector_default.initialized, "Pre-condition failed: Detector not initialized.")
        landmarks = self.pose_detector_default.process_frame(None)
        self.assertIsNone(landmarks, "Processing a None frame should return None.")
        print("--- Finished Test: test_03_process_none_frame ---")

    # --- Optional: Add a test with a real image ---
    # This requires having an image file available.
    # def test_04_process_real_image(self):
    #     """Test processing an image known to contain a person."""
    #     print("\n--- Running Test: test_04_process_real_image ---")
    #     image_path = "path/to/your/test_image_with_person.jpg" # !!! REPLACE WITH ACTUAL PATH !!!
    #     if not os.path.exists(image_path):
    #         self.skipTest(f"Test image not found at {image_path}") # Skip if image doesn't exist
    #
    #     frame = cv2.imread(image_path)
    #     self.assertIsNotNone(frame, f"Failed to load test image: {image_path}")
    #
    #     # Convert BGR to RGB
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #
    #     self.assertTrue(self.pose_detector_default.initialized, "Pre-condition failed: Detector not initialized.")
    #     landmarks = self.pose_detector_default.process_frame(frame_rgb)
    #
    #     self.assertIsNotNone(landmarks, "Landmarks should be detected in the test image.")
    #     self.assertIsInstance(landmarks, mp.framework.formats.landmark_pb2.NormalizedLandmarkList, "Result should be MediaPipe PoseLandmarks")
    #     # Check if specific landmarks exist (e.g., nose)
    #     self.assertTrue(hasattr(landmarks, 'landmark'))
    #     self.assertGreater(len(landmarks.landmark), 0, "Should detect more than 0 landmarks.")
    #     # Example: Check visibility of nose landmark (index 0)
    #     # nose_landmark = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    #     # self.assertGreater(nose_landmark.visibility, 0.5, "Nose should be reasonably visible.")
    #     print(f"  Detected {len(landmarks.landmark)} landmarks in test image.")
    #     print("--- Finished Test: test_04_process_real_image ---")


# --- Run Tests ---
if __name__ == '__main__':
    print("=======================================")
    print("     Running PoseDetector Test Suite    ")
    print("=======================================")
    # Run using unittest's discovery mechanism if run directly,
    # or use standard runner. Standard runner is simpler here.
    unittest.main(verbosity=2) # Increase verbosity for more detail