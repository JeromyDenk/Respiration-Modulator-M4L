# src/video_input.py
# Handles camera/video file input and resolution setting.

import cv2
import time

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
                'VIDEO_API_PREFERENCE' (int, optional): OpenCV API preference (e.g., cv2.CAP_DSHOW, cv2.CAP_MSMF).
                'VIDEO_MANUAL_EXPOSURE' (bool, optional): If True, attempts to set manual exposure. Default False.
                'VIDEO_EXPOSURE_VALUE' (float/int, optional): Exposure value if manual exposure is enabled.
                'VIDEO_ISO_VALUE' (int, optional): Desired ISO level (e.g., 100, 200, 400). Support is camera-dependent.
                'VIDEO_ZOOM_VALUE' (int, optional): Desired zoom level.
                Defaults are used if config is None or keys are missing.
        """
        init_start_time = time.perf_counter()

        if config is None:
            config = {}

        self.source = config.get('VIDEO_SOURCE', 0) # Default to camera 0
        self.desired_width = config.get('VIDEO_WIDTH', None)
        self.desired_height = config.get('VIDEO_HEIGHT', None)
        self.desired_fps = config.get('VIDEO_FPS', None)
        self.api_preference = config.get('VIDEO_API_PREFERENCE', None)
        # New camera control settings
        self.manual_exposure = config.get('VIDEO_MANUAL_EXPOSURE', False)
        self.desired_exposure_value = config.get('VIDEO_EXPOSURE_VALUE', None)
        self.desired_iso_value = config.get('VIDEO_ISO_VALUE', None)
        self.desired_zoom_value = config.get('VIDEO_ZOOM_VALUE', None)

        print(f"[VideoInput] Initializing video source: {self.source}...")
        if self.desired_width and self.desired_height:
            print(f"  Desired resolution: {self.desired_width}x{self.desired_height}")
        if self.desired_fps:
             print(f"  Desired FPS: {self.desired_fps}")

        if self.manual_exposure:
            print(f"  Attempting manual exposure. Desired exposure (shutter speed related): {self.desired_exposure_value if self.desired_exposure_value is not None else 'Not set'}")
            if self.desired_iso_value is not None:
                print(f"  Attempting manual ISO: {self.desired_iso_value}")
        else:
            print(f"  Using auto exposure.")
        if self.desired_zoom_value is not None: # Zoom can typically be set independently
            print(f"  Attempting zoom: {self.desired_zoom_value}")
        capture_source = self.source
        if self.api_preference is not None and isinstance(self.source, int):
            print(f"  Using API Preference: {self.api_preference}")
            capture_source = self.source + self.api_preference
        else:
            print(f"  Using default API backend.")

        try:
            t0 = time.perf_counter()
            self.cap = cv2.VideoCapture(capture_source)
            t1 = time.perf_counter()
            print(f"  [TIMING] cv2.VideoCapture() took: {t1 - t0:.4f}s")

            if not self.cap.isOpened():
                raise IOError(f"Cannot open video source: {self.source}")

            # --- Attempt to set resolution and FPS ---
            if self.desired_width is not None:
                t0 = time.perf_counter()
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
                t1 = time.perf_counter()
                print(f"  [TIMING] set(CAP_PROP_FRAME_WIDTH, {self.desired_width}) took: {t1 - t0:.4f}s")
            if self.desired_height is not None:
                t0 = time.perf_counter()
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)
                t1 = time.perf_counter()
                print(f"  [TIMING] set(CAP_PROP_FRAME_HEIGHT, {self.desired_height}) took: {t1 - t0:.4f}s")
            if self.desired_fps is not None:
                 t0 = time.perf_counter()
                 self.cap.set(cv2.CAP_PROP_FPS, self.desired_fps)
                 t1 = time.perf_counter()
                 print(f"  [TIMING] set(CAP_PROP_FPS, {self.desired_fps}) took: {t1 - t0:.4f}s")

            # --- Attempt to set exposure and zoom ---
            if self.manual_exposure:
                # Attempt to disable auto exposure.
                # Common values: 0.25 for MSMF, 1 for DSHOW/V4L for 'manual'.
                # This might need adjustment based on the backend in use.
                auto_exposure_manual_val = 0.25 if self.api_preference == cv2.CAP_MSMF else 1
                print(f"  Setting CAP_PROP_AUTO_EXPOSURE to {auto_exposure_manual_val} (manual attempt)")
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_manual_val)
                if self.desired_exposure_value is not None:
                    t0 = time.perf_counter()
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, self.desired_exposure_value)
                    t1 = time.perf_counter()
                    print(f"  [TIMING] set(CAP_PROP_EXPOSURE, {self.desired_exposure_value}) took: {t1 - t0:.4f}s")
                if self.desired_iso_value is not None:
                    t0 = time.perf_counter()
                    self.cap.set(cv2.CAP_PROP_ISO_SPEED, self.desired_iso_value)
                    t1 = time.perf_counter()
                    print(f"  [TIMING] set(CAP_PROP_ISO_SPEED, {self.desired_iso_value}) took: {t1 - t0:.4f}s")
            else:
                # Explicitly set to auto if manual is false (some cameras might default to manual)
                auto_exposure_auto_val = 0.75 if self.api_preference == cv2.CAP_MSMF else 3
                print(f"  Setting CAP_PROP_AUTO_EXPOSURE to {auto_exposure_auto_val} (auto attempt)")
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_auto_val)

            if self.desired_zoom_value is not None:
                t0 = time.perf_counter()
                self.cap.set(cv2.CAP_PROP_ZOOM, self.desired_zoom_value)
                t1 = time.perf_counter()
                print(f"  [TIMING] set(CAP_PROP_ZOOM, {self.desired_zoom_value}) took: {t1 - t0:.4f}s")

            # --- Verify actual resolution and FPS ---
            # It's important to wait briefly for settings to apply on some systems
            print(f"  [TIMING] Waiting for 0.5s for settings to apply...")
            t0 = time.perf_counter()
            time.sleep(0.5)
            t1 = time.perf_counter()
            print(f"  [TIMING] time.sleep(0.5) actually took: {t1 - t0:.4f}s")

            t0_get = time.perf_counter()
            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) # Can be unreliable
            # Get actual exposure and zoom settings
            self.actual_auto_exposure = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            self.actual_exposure_value = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            self.actual_iso_value = self.cap.get(cv2.CAP_PROP_ISO_SPEED)
            self.actual_zoom_value = self.cap.get(cv2.CAP_PROP_ZOOM)
            t1_get = time.perf_counter()
            print(f"  [TIMING] Getting actual camera parameters took: {t1_get - t0_get:.4f}s")

            print(f"  Actual resolution: {self.actual_width}x{self.actual_height}")
            print(f"  Actual FPS reported by camera: {self.actual_fps:.2f}")
            print(f"  Actual Auto Exposure mode: {self.actual_auto_exposure}")
            print(f"  Actual Exposure value (shutter speed related): {self.actual_exposure_value}")
            print(f"  Actual ISO value: {self.actual_iso_value}")
            print(f"  Actual Zoom value: {self.actual_zoom_value}")

            if self.desired_width and self.desired_width != self.actual_width:
                 print(f"  [VideoInput] Warning: Actual width ({self.actual_width}) differs from desired ({self.desired_width}).")
            if self.desired_height and self.desired_height != self.actual_height:
                 print(f"  [VideoInput] Warning: Actual height ({self.actual_height}) differs from desired ({self.desired_height}).")
            if self.desired_fps and abs(self.desired_fps - self.actual_fps) > 1: # Allow some tolerance for FPS
                 print(f"  [VideoInput] Warning: Actual FPS ({self.actual_fps:.2f}) differs significantly from desired ({self.desired_fps}).")

            # Perform a dummy read or two to "warm up" some camera backends (like MSMF)
            # that might have a slow first frame grab.
            print(f"  [VideoInput] Performing dummy frame read(s) to initialize stream...")
            for _ in range(2): # Read two frames; sometimes one isn't enough
                t_dummy_read_start = time.perf_counter()
                ret_dummy, _ = self.cap.read()
                t_dummy_read_end = time.perf_counter()
                print(f"    Dummy read success: {ret_dummy}, time: {(t_dummy_read_end - t_dummy_read_start)*1000:.1f}ms")
                if not ret_dummy: break # Stop if a dummy read fails
            self.initialized = True

        except Exception as e:
            print(f"[VideoInput] FATAL ERROR initializing video source: {e}", flush=True) # Ensure this prints immediately
            self.cap = None
            self.initialized = False
            self.actual_width = 0
            self.actual_height = 0
            self.actual_fps = 0
            self.actual_auto_exposure = -1 # Indicate unset/error
            self.actual_exposure_value = -1
            self.actual_iso_value = -1
            self.actual_zoom_value = -1
        finally:
            init_end_time = time.perf_counter()
            print(f"[VideoInput] Total initialization time: {init_end_time - init_start_time:.4f}s")

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
        'VIDEO_HEIGHT': 480,
        # Example: Try to set manual exposure and zoom
        # 'VIDEO_MANUAL_EXPOSURE': True,
        # 'VIDEO_EXPOSURE_VALUE': -6, # Example exposure value, adjust for your camera
        # 'VIDEO_ISO_VALUE': 200,     # Example ISO value, adjust for your camera
        # 'VIDEO_ZOOM_VALUE': 200      # Example value, adjust for your camera
        # 'VIDEO_API_PREFERENCE': cv2.CAP_DSHOW # Example: Try DirectShow
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

    # --- Test Case 2b: Specific Resolution with DSHOW ---
    print("\n--- Test 2b: Requesting 640x480 with CAP_DSHOW ---")
    config_640_dshow = {
        'VIDEO_SOURCE': 0,
        'VIDEO_WIDTH': 640,
        'VIDEO_HEIGHT': 480,
        'VIDEO_API_PREFERENCE': cv2.CAP_DSHOW
    }
    vid_640_dshow = VideoInput(config=config_640_dshow)
    if vid_640_dshow.initialized:
        ret, frame = vid_640_dshow.get_frame()
        if ret:
            print(f"  Successfully read frame. Shape: {frame.shape}")
            cv2.imshow("640x480 DSHOW Test", frame)
            cv2.waitKey(1000)
            cv2.destroyWindow("640x480 DSHOW Test")
        else:
            print("  Failed to read frame (640x480 DSHOW request).")
        vid_640_dshow.release()
    else:
        print("  Failed to initialize camera (640x480 DSHOW request).")


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
