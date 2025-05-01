# main.py
# Main entry point for the Respiration Modulator application.
# Sets up the PyQt6 application, creates the UI window, and manages the backend worker thread.
# FIXED: Removed problematic High DPI attribute settings.

import sys
import os
import time # For sleep if needed
import traceback
import json # Needed by worker
import numpy as np # Needed for PipelineWorker signals
import cv2 # Needed for VideoCapture fallback

# --- PyQt6 Imports ---
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QThread, QObject, pyqtSignal, QTimer, Qt # Added Qt
    print("Using PyQt6")
except ImportError:
    print("Fatal Error: PyQt6 not found. Please install it (e.g., pip install PyQt6)")
    sys.exit(1)


# --- Add src directory to Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import UI and Backend Components ---
try:
    from ui.main_window import MainWindow # Import the UI class
    # Import pipeline components needed for the worker
    try:
        from video_input import VideoInput
    except ImportError:
        print("Warning: video_input.py not found. Using direct cv2.VideoCapture fallback in worker.")
        VideoInput = None
    from pipeline_manager import PipelineManager
    from signal_processor import SignalProcessor # Import for constants in worker
    from feature_tracker import FeatureTracker # Import for reset in worker

except ImportError as e:
    print(f"Fatal Error: Failed to import necessary modules from 'src' or 'src/ui': {e}")
    print("Please ensure 'src' directory and required files (main_window.py, pipeline_manager.py etc.) exist.")
    traceback.print_exc() # Print detailed import error
    sys.exit(1)
except Exception as e_general:
     print(f"An unexpected error occurred during imports: {e_general}")
     traceback.print_exc()
     sys.exit(1)


# --- Constants ---
DEFAULT_PROFILE = "test_profile.json"
# Define PROFILES_DIR relative to main.py (which is in the root)
PROFILES_DIR = os.path.join(script_dir, "profiles")

# --- Backend Worker Thread Definition ---
class PipelineWorker(QObject):
    """
    Runs the pipeline processing in a separate thread to avoid freezing the GUI.
    """
    # Signals to emit results back to the main GUI thread
    new_frame_ready = pyqtSignal(np.ndarray) # Emits the frame for display
    new_plot_data = pyqtSignal(list)         # Emits the filtered signal history
    new_status = pyqtSignal(float, bool, int)# Emits BPM, validity, phase
    processing_error = pyqtSignal(str)       # Emits error messages
    finished = pyqtSignal()                  # Signal when the run loop finishes

    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.pipeline = None
        self.video_input = None
        self._running = False # Use underscore for internal control flag
        self.tracking_active = False # Controls if full pipeline runs
        self.sampling_rate = 30.0 # Default, will be updated
        self.current_config = {} # Store loaded config

    def _load_config(self):
        """Loads or reloads configuration."""
        config = {}
        try:
            # Ensure config_path is valid before trying to open
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(f"[Worker] Loaded config from: {self.config_path}")
                self.current_config = config # Store loaded config
            else:
                print(f"[Worker] Warning: Config file not found or path invalid: '{self.config_path}'. Using empty config.")
                self.current_config = {}
        except json.JSONDecodeError as json_err:
             print(f"[Worker] Error decoding JSON from '{self.config_path}': {json_err}")
             self.processing_error.emit(f"Config JSON Error: {json_err}")
             self.current_config = {}
        except Exception as e:
            print(f"[Worker] Error loading config '{self.config_path}': {e}")
            self.processing_error.emit(f"Config Load Error: {e}")
            self.current_config = {} # Reset config on error
        return self.current_config # Return the stored config

    def setup(self):
        """Initializes video input and pipeline manager. Returns True on success."""
        print("[Worker] Setting up...")
        try:
            config = self._load_config() # Load initial config
            video_config = config.get("video_input", {})

            # Release previous video input if exists
            if self.video_input:
                 if hasattr(self.video_input, 'release'):
                     self.video_input.release()
                 self.video_input = None

            # Initialize Video Input
            if VideoInput:
                self.video_input = VideoInput(config=video_config)
                if not self.video_input.initialized: raise RuntimeError("VideoInput failed.")
                fps = self.video_input.get_fps()
                self.sampling_rate = fps if fps > 0 else 30.0
            else: # Fallback
                self.video_input = cv2.VideoCapture(0) # Try default camera
                if not self.video_input.isOpened():
                    # Try next camera index if default fails (simple fallback)
                    print("[Worker] Camera 0 failed, trying Camera 1...")
                    self.video_input = cv2.VideoCapture(1)
                    if not self.video_input.isOpened():
                        raise RuntimeError("cv2.VideoCapture failed for index 0 and 1.")
                # Define dummy methods for compatibility
                self.video_input.get_frame = lambda: self.video_input.read() # type: ignore
                self.video_input.release = lambda: self.video_input.release() # type: ignore
                self.sampling_rate = 30.0 # Estimate

            print(f"[Worker] Using sampling rate: {self.sampling_rate:.2f} Hz")

            # Initialize Pipeline Manager with currently loaded config
            # Ensure pipeline is closed if it exists before re-creating
            if self.pipeline:
                self.pipeline.close()
            self.pipeline = PipelineManager(config=config, sampling_rate=self.sampling_rate)
            print("[Worker] Setup successful.")
            return True
        except Exception as e:
            error_msg = f"Error setting up worker: {e}"
            print(error_msg); traceback.print_exc()
            self.processing_error.emit(error_msg)
            if hasattr(self.video_input, 'release'): self.video_input.release()
            self.pipeline = None; self.video_input = None
            return False

    def run(self):
        """The main processing loop that runs in the thread."""
        if not self.pipeline or not self.video_input:
            print("[Worker] Error: Not set up correctly. Stopping run.")
            self.processing_error.emit("Worker not initialized.")
            self._running = False; self.finished.emit(); return

        self._running = True
        print("[Worker] Starting processing loop...")

        while self._running:
            start_loop_time = time.perf_counter()
            try:
                success, frame = self.video_input.get_frame()
                if not success or frame is None:
                    print("[Worker] End of video source or cannot read frame.")
                    self._running = False; break # Stop if video ends or fails

                processed_frame = frame.copy() # Start with original frame for display
                results = None

                # --- Core Processing ---
                if self.tracking_active:
                    results = self.pipeline.process_frame(frame)
                    # TODO: Add overlay drawing logic here based on results and UI checkboxes
                    # This should ideally happen here or be triggered from here
                    # based on signals/state from the UI thread to keep drawing logic
                    # close to the data source.
                    # Example:
                    # if ui_state.show_pose and results.get('landmarks'):
                    #    mp_drawing.draw_landmarks(processed_frame, results['landmarks'], ...)
                    # if ui_state.show_roi and results.get('current_rois'):
                    #    for x,y,w,h in results['current_rois']: cv2.rectangle(...)
                    # if ui_state.show_features and self.pipeline.feature_tracker: ... draw points ...
                else:
                    # Initial State: Maybe run only pose/roi for display?
                    # For now, just show raw frame
                    pass

                # --- Emit Signals ---
                self.new_frame_ready.emit(processed_frame) # Emit frame (potentially with overlays)

                if results:
                    # Emit plot data and status only if full pipeline ran
                    plot_data = results.get('filtered_signal_history', [])
                    bpm = results.get('bpm', 0.0)
                    valid = results.get('bpm_valid', False)
                    phase = results.get('phase', SignalProcessor.PHASE_UNKNOWN)
                    self.new_plot_data.emit(plot_data)
                    self.new_status.emit(bpm, valid, phase)
                else:
                    # Emit default/empty data when not tracking
                    self.new_plot_data.emit([])
                    self.new_status.emit(0.0, False, SignalProcessor.PHASE_UNKNOWN)

                # --- Loop Delay ---
                # Aim for a target frame interval, considering processing time
                loop_duration = time.perf_counter() - start_loop_time
                target_interval = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.033
                sleep_time = max(0.001, target_interval - loop_duration)
                time.sleep(sleep_time)


            except Exception as e:
                error_msg = f"Error in worker loop: {e}"
                print(error_msg); traceback.print_exc()
                self.processing_error.emit(error_msg)
                time.sleep(0.1) # Avoid spamming errors if loop fails continuously

        # --- Cleanup ---
        print("[Worker] Stopping processing loop...")
        if hasattr(self.video_input, 'release'): self.video_input.release()
        if self.pipeline: self.pipeline.close()
        print("[Worker] Resources released.")
        self.finished.emit()

    def stop(self):
        """Sets the flag to stop the processing loop."""
        print("[Worker] Stop requested.")
        self._running = False

    def set_tracking_active(self, active: bool):
        """Activates/deactivates the full pipeline processing."""
        print(f"[Worker] Setting tracking active: {active}")
        self.tracking_active = active
        if not active and self.pipeline:
             pass # Optionally reset parts of pipeline state when stopping
        elif active and self.pipeline:
             # Reset pipeline state when starting tracking
             print("[Worker] Resetting pipeline state for tracking start.")
             try:
                 # Use the currently loaded config for re-initialization
                 self.pipeline.signal_processor = SignalProcessor(
                     config=self.current_config.get('signal_processor', {}),
                     sampling_rate=self.sampling_rate
                 )
                 self.pipeline.feature_tracker = FeatureTracker(
                     config=self.current_config.get('feature_tracker', {})
                 )
                 # Ensure pipeline knows it needs to recalibrate ROI/features
                 self.pipeline.trigger_recalibration()
                 # Force feature redetection on next frame
                 if self.pipeline.feature_tracker:
                     self.pipeline.feature_tracker.prev_gray_frame = None
                     self.pipeline.feature_tracker.prev_features_per_roi = {}

             except Exception as e_reset:
                 error_msg = f"Error resetting pipeline components: {e_reset}"
                 print(error_msg); traceback.print_exc()
                 self.processing_error.emit(error_msg)

    def reload_profile(self, new_config_path):
        """Stops current processing, reloads config, re-setups pipeline."""
        print(f"[Worker] Reload profile requested: {new_config_path}")
        was_tracking = self.tracking_active
        self.set_tracking_active(False) # Ensure tracking is off
        # Note: Ideally wait for the loop to fully stop if it was running.
        # QThread management can be complex. This simple version assumes
        # setting _running=False is sufficient for quick stop.

        self.config_path = new_config_path
        # Re-run setup which loads the new config and re-initializes pipeline
        setup_ok = self.setup()
        if setup_ok:
            self.processing_error.emit(f"Profile '{os.path.basename(new_config_path)}' loaded. Restart tracking if needed.")
        else:
             print("[Worker] Setup failed after profile reload.")
             self.processing_error.emit("Setup failed after profile reload.")


# --- Main Application Setup ---
if __name__ == "__main__":
    # --- REMOVED High DPI Attribute settings ---
    # QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling) # Causes AttributeError
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps) # Causes AttributeError

    app = QApplication(sys.argv)

    # --- Determine Config Path ---
    config_file = os.path.join(PROFILES_DIR, DEFAULT_PROFILE)
    if not os.path.exists(PROFILES_DIR):
         print(f"Warning: Profiles directory does not exist: '{PROFILES_DIR}'")
         # Optionally create it?
         # os.makedirs(PROFILES_DIR, exist_ok=True)
    if not os.path.exists(config_file):
        print(f"Warning: Default profile '{DEFAULT_PROFILE}' not found in '{PROFILES_DIR}'.")
        config_file = "" # Worker will handle empty path

    # --- Create UI and Worker ---
    # Pass the profiles directory path to MainWindow
    main_window = MainWindow(config_file=config_file, profiles_dir=PROFILES_DIR)

    worker = PipelineWorker(config_path=config_file)
    worker_thread = QThread()
    worker.moveToThread(worker_thread)

    # --- Connect Signals and Slots ---
    # Worker -> UI
    worker.new_frame_ready.connect(main_window.update_webcam_feed)
    worker.new_plot_data.connect(main_window.update_plot)
    worker.new_status.connect(main_window.update_status_labels)
    worker.processing_error.connect(main_window.show_error_message)
    worker.finished.connect(worker_thread.quit)
    worker.finished.connect(worker.deleteLater)

    # UI -> Worker
    main_window.start_tracking_signal.connect(lambda: worker.set_tracking_active(True))
    main_window.stop_tracking_signal.connect(lambda: worker.set_tracking_active(False))
    main_window.load_profile_signal.connect(worker.reload_profile)
    # Connect save signal (worker currently doesn't handle saving, UI just requests path)
    # main_window.save_profile_signal.connect(worker.save_current_config_as) # If worker handled saving

    # Connect thread start/stop
    # Use a QTimer to delay setup/run until event loop starts
    QTimer.singleShot(100, worker_thread.start) # Start thread shortly after UI shows
    worker_thread.started.connect(worker.setup)
    # Connect run to start after setup is confirmed successful (more robust)
    # This requires setup to signal its completion, or run checks if setup failed.
    # For now, connect directly, setup failure is handled in run() start check.
    worker_thread.started.connect(worker.run)

    # Ensure thread quits when the app is closing
    app.aboutToQuit.connect(worker.stop) # Tell worker loop to stop
    # Connect worker thread finish signal to app quit
    worker_thread.finished.connect(app.quit)


    # --- Start Application ---
    main_window.show()
    sys.exit(app.exec())
