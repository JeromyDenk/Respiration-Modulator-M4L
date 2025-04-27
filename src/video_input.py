# src/video_input.py
# Handles camera/video file input and resolution setting.

import cv2
import time
import sys

class VideoInput:
    """
    Manages video input from a webcam or file, including attempting
    to set the desired resolution.
    """
    def __init__(self, config=None):
        """
        Initializes the video source.

        Args:
            config (dict, optional): Configuration dictionary. Expected keys:
                'VIDEO_SOURCE' (int or str): Camera index (e.g., 0) or video file path. Default 0.
                'VIDEO_WIDTH' (int, optional): Desired frame width (e.g., 1280, 640).
                'VIDEO_HEIGHT' (int, optional): Desired frame height (e.g., 720, 480).
                'VIDEO_FPS' (int, optional): Desired FPS (support varies greatly by camera).
                Defaults are used if config is None or keys are missing.
        """
        if config is None:
            config = {}

        self.source = config.get('VIDEO_SOURCE', 0) # Default to camera 0
        self.desired_width = config.get('VIDEO_WIDTH', None)
        self.desired_height = config.get('VIDEO_HEIGHT', None)
        self.desired_fps = config.get('VIDEO_FPS', None)

        print(f"[VideoInput] Initializing video source: {self.source}")
        if self.desired_width and self.desired_height:
            print(f"  Attempting resolution: {self.desired_width}x{self.desired_height}")
        if self.desired_fps:
             print(f"  Attempting FPS: {self.desired_fps}")


        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video source: {self.source}")

            # --- Attempt to set resolution and FPS ---
            if self.desired_width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
            if self.desired_height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
            if self.desired_fps is not None:
                 self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)

            # --- Verify actual resolution and FPS ---
            # It's important to wait briefly for settings to apply on some systems
            time.sleep(0.5)
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) # Can be unreliable

            print(f"  Actual resolution: {self.actual_width}x{self.actual_height}")
            print(f"  Actual FPS reported by camera: {self.actual_fps:.2f}")

            if self.desired_width and self.desired_width != self.actual_width:
                 print(f"  Warning: Actual width ({self.actual_width}) differs from desired ({self.desired_width}).")
            if self.desired_height and self.desired_height != self.actual_height:
                 print(f"  Warning: Actual height ({self.actual_height}) differs from desired ({self.desired_height}).")
            if self.desired_fps and abs(self.desired_fps - self.actual_fps) > 1: # Allow some tolerance for FPS
                 print(f"  Warning: Actual FPS ({self.actual_fps:.2f}) differs significantly from desired ({self.desired_fps}).")

            self.initialized = True

        except Exception as e:
            print(f"[VideoInput] FATAL ERROR initializing video source: {e}")
            self.cap = None
            self.initialized = False
            self.actual_width = 0
            self.actual_height = 0
            self.actual_fps = 0

    def get_frame(self):
        """Reads and returns the next frame from the video source."""
        if not self.initialized or not self.cap or not self.cap.isOpened():
            # print("[VideoInput] Error: Video source not ready.") # Can be noisy
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def get_resolution(self):
        """Returns the actual width and height of the video source."""
        return self.actual_width, self.actual_height

    def get_fps(self):
        """Returns the FPS reported by the camera (may not be accurate)."""
        return self.actual_fps

    def release(self):
        """Releases the video capture object."""
        if self.cap and self.cap.isOpened():
            print("[VideoInput] Releasing video source...")
            self.cap.release()
            print("[VideoInput] Released.")
        self.initialized = False

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing VideoInput module...")

    # --- Test Case 1: Default (Camera 0, default resolution) ---
    print("\n--- Test 1: Default Camera ---")
    vid_default = VideoInput()
    if vid_default.initialized:
        ret, frame = vid_default.get_frame()
        if ret:
            print(f"  Successfully read frame. Shape: {frame.shape}")
            cv2.imshow("Default Test", frame)
            cv2.waitKey(1000) # Display for 1 second
            cv2.destroyWindow("Default Test")
        else:
            print("  Failed to read frame from default camera.")
        vid_default.release()
    else:
        print("  Failed to initialize default camera.")


    # --- Test Case 2: Specific Resolution (e.g., 640x480) ---
    print("\n--- Test 2: Requesting 640x480 ---")
    config_640 = {
        'VIDEO_SOURCE': 0,
        'VIDEO_WIDTH': 640,
        'VIDEO_HEIGHT': 480
    }
    vid_640 = VideoInput(config=config_640)
    if vid_640.initialized:
        ret, frame = vid_640.get_frame()
        if ret:
            print(f"  Successfully read frame. Shape: {frame.shape}")
            cv2.imshow("640x480 Test", frame)
            cv2.waitKey(1000)
            cv2.destroyWindow("640x480 Test")
        else:
            print("  Failed to read frame (640x480 request).")
        vid_640.release()
    else:
        print("  Failed to initialize camera (640x480 request).")

    # --- Test Case 3: Invalid Source ---
    print("\n--- Test 3: Invalid Source ---")
    config_invalid = {'VIDEO_SOURCE': 99} # Assuming camera 99 doesn't exist
    vid_invalid = VideoInput(config=config_invalid)
    if not vid_invalid.initialized:
        print("  Correctly failed to initialize invalid source.")
    else:
        print("  Error: Initialized an invalid source?")
        vid_invalid.release()


    print("\nVideoInput module test finished.")
    cv2.destroyAllWindows() # Ensure all windows closed
