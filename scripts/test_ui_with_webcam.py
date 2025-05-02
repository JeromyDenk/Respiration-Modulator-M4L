# scripts/test_ui_with_webcam.py
# Runs the PyQt UI (MainWindow) and displays a live webcam feed using VideoInput,
# but without initializing or running the backend processing pipeline.
# Useful for testing UI layout, responsiveness, and webcam display.

import sys
import os
import traceback
import time
import cv2
import numpy as np

# --- PyQt6 Imports ---
try:
    from PyQt6.QtWidgets import QApplication, QMessageBox
    from PyQt6.QtCore import QTimer, Qt
    print("Using PyQt6")
except ImportError:
    print("Fatal Error: PyQt6 not found. Please install it (e.g., pip install PyQt6)")
    sys.exit(1)

# --- pyqtgraph Import ---
try:
    import pyqtgraph as pg
except ImportError:
     print("Fatal Error: pyqtgraph not found. Please install it (e.g., pip install pyqtgraph)")
     sys.exit(1)


# --- Add src directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import UI and Video Components ---
try:
    from ui.main_window import MainWindow # Import the UI class
    from video_input import VideoInput # Import VideoInput directly
except ImportError as e:
    print(f"Fatal Error: Failed to import MainWindow or VideoInput: {e}")
    traceback.print_exc(); sys.exit(1)
except Exception as e_general:
     print(f"An unexpected error occurred during imports: {e_general}")
     traceback.print_exc(); sys.exit(1)

# --- Constants ---
PROFILES_DIR = os.path.join(project_root, "profiles")
TARGET_FPS = 30 # Target FPS for webcam reading loop
TIMER_INTERVAL_MS = int(1000 / TARGET_FPS) # Calculate timer interval


# --- Main Application Setup ---
if __name__ == "__main__":
    print("Starting UI Test Script with Webcam...")

    app = QApplication(sys.argv)

    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOptions(antialias=True)

    if not os.path.exists(PROFILES_DIR):
         print(f"Warning: Profiles directory does not exist: '{PROFILES_DIR}'")

    # --- Create UI ---
    try:
        main_window = MainWindow(profiles_dir=PROFILES_DIR)
        print("MainWindow instance created.")
    except Exception as e_create_ui:
         print(f"Fatal Error creating MainWindow: {e_create_ui}")
         traceback.print_exc()
         sys.exit(1)

    # --- Initialize Video Input ---
    video_input = None
    try:
        print("Initializing VideoInput...") # Example: Use camera 0, try for 640x480 resolution
        # Use a default config or specify one if needed
        video_config = {
            'VIDEO_SOURCE': 0,
            'VIDEO_WIDTH': 640,
            'VIDEO_HEIGHT': 480
        }
        video_input = VideoInput(config=video_config)
        if not video_input.initialized:
            raise RuntimeError(f"VideoInput failed to initialize for source: {video_config.get('VIDEO_SOURCE', 'default')}")
        print("VideoInput initialized successfully.")
    except Exception as e_video:
        error_msg = f"Fatal Error initializing VideoInput: {e_video}"
        print(error_msg)
        traceback.print_exc()
        QMessageBox.critical(None, "Video Error", error_msg)
        sys.exit(1) # Exit if video cannot be opened

    # --- Setup QTimer for Frame Grabbing ---
    frame_timer = QTimer()

    def update_frame():
        """Gets a frame from VideoInput and updates the UI."""
        try:
            success, frame = video_input.get_frame()
            # --- FIX: Correct logic ---
            if success and frame is not None:
                # Call the UI update method when a frame is successfully read
                main_window.update_webcam_feed(frame)
            elif not success:
                # Handle failure (e.g., end of video file, camera disconnected)
                print("Failed to grab frame or end of source reached.")
                frame_timer.stop() # Stop timer if reading fails
                main_window.show_error_message("Webcam feed stopped or failed.")
            # --- END FIX ---
        except Exception as e:
            # Catch any other errors during frame grabbing or UI update
            print(f"Error in update_frame: {e}")
            traceback.print_exc()
            frame_timer.stop() # Stop the timer on error
            main_window.show_error_message(f"Error reading/displaying frame: {e}")

    frame_timer.timeout.connect(update_frame)

    # --- Manually Enable UI Controls for Testing ---
    print("Manually enabling UI controls for testing...")
    # Need to enable after setup might have disabled them
    # Use a short timer to enable after UI is shown and potentially initialized
    def enable_ui():
        # Enable the controls
        main_window.track_button.setEnabled(True)
        main_window.load_button.setEnabled(True)
        main_window.save_button.setEnabled(True)
        main_window.save_as_button.setEnabled(True)
        main_window.profile_combo.setEnabled(True)
        if hasattr(main_window, 'settings_toggle_button'):
            main_window.settings_toggle_button.setEnabled(True)
        main_window._update_ui_state() # Update dependent states like apply button
        main_window.statusBar.showMessage("UI Test Mode - VideoInput Feed.")
        print("UI Controls Enabled.") # This will now print only once

        # Start the timer *after* UI is enabled and shown
        # This is the correct place to start the timer
        frame_timer.start(TIMER_INTERVAL_MS)
        print(f"Frame timer started with interval: {TIMER_INTERVAL_MS} ms") # This will now print only once

    # Schedule enable_ui to run once after the event loop starts
    QTimer.singleShot(100, enable_ui) # Enable after 100ms

    # --- Start Application ---
    main_window.show()
    print("MainWindow shown.")

    # Ensure cleanup on exit
    def cleanup():
        print("Application quitting. Releasing resources...")
        frame_timer.stop() # Make sure timer is stopped
        if video_input:
            video_input.release() # Release the camera
        print("Cleanup finished.")
    app.aboutToQuit.connect(cleanup)

    print("Starting application event loop...")
    sys.exit(app.exec())
