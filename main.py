# main.py
# Main entry point for the Respiration Modulator application.
# Sets up the PyQt6 application, creates the UI window, and manages the backend worker thread.
# MODIFIED: Set _running flag earlier in setup to allow incremental init to proceed.

import sys
import os
import time # For sleep if needed
import traceback
import json # Needed by worker
import numpy as np # Needed for PipelineWorker signals
import cv2 # Needed for VideoCapture fallback
import mediapipe as mp # Needed for drawing landmarks
import copy # For deepcopy

# --- PyQt6 Imports ---
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QThread, QObject, pyqtSignal, QTimer, Qt
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
    try:
        from video_input import VideoInput
    except ImportError:
        print("Warning: video_input.py not found. Using direct cv2.VideoCapture fallback in worker.")
        VideoInput = None
    # Import all components needed by the worker now
    from pipeline_manager import PipelineManager
    from pose_detector import PoseDetector
    from coarse_roi_calculator import CoarseRoiCalculator
    from signal_processor import SignalProcessor # Import for constants
    from feature_tracker import FeatureTracker # Import for reset

except ImportError as e:
    print(f"Fatal Error: Failed to import necessary modules from 'src' or 'src/ui': {e}")
    traceback.print_exc(); sys.exit(1)
except Exception as e_general:
     print(f"An unexpected error occurred during imports: {e_general}")
     traceback.print_exc(); sys.exit(1)


# --- Constants ---
DEFAULT_PROFILE = "test_profile.json"
PROFILES_DIR = os.path.join(script_dir, "profiles")

# --- MediaPipe Drawing Utilities (for worker) ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# --- Worker State Enum ---
class WorkerState:
    INITIALIZING = 0
    PREVIEWING = 1 # Running Pose/ROI only
    TRACKING = 2   # Running full pipeline with locked ROI

# --- Backend Worker Thread Definition ---
class PipelineWorker(QObject):
    """
    Runs the pipeline processing in a separate thread to avoid freezing the GUI.
    Manages PREVIEWING (Pose/ROI) and TRACKING (Full Pipeline) states.
    Initializes heavy components incrementally.
    """
    new_frame_ready = pyqtSignal(np.ndarray)
    new_plot_data = pyqtSignal(list)
    new_status = pyqtSignal(float, bool, int)
    processing_error = pyqtSignal(str)
    finished = pyqtSignal()
    setup_finished = pyqtSignal(bool, str)
    component_initialized = pyqtSignal(str, bool, str)

    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.current_config = {}
        self.video_input = None
        self.sampling_rate = 30.0
        self.pose_detector = None
        self.coarse_roi_calculator = None
        self.pipeline_manager = None
        self._running = False # Flag to control the main run loop AND initialization steps
        self.state = WorkerState.INITIALIZING
        self.latest_preview_roi = []
        self.locked_roi = []
        self.latest_landmarks = None
        self._components_initialized = False
        self._init_step = 0

    def _load_config(self):
        # (Load config logic remains the same)
        config = {}
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f: config = json.load(f)
                print(f"[Worker] Loaded config from: {self.config_path}")
                self.current_config = config
            else:
                print(f"[Worker] Warning: Config file '{self.config_path}' not found. Using empty config.")
                self.current_config = {}
        except Exception as e:
            print(f"[Worker] Error loading config '{self.config_path}': {e}")
            self.current_config = {}
            raise
        return self.current_config

    def setup(self):
        """Initializes ONLY Video input and loads config. Emits setup_finished."""
        print("[Worker] Running initial setup (Video & Config)...")
        error_msg = ""
        setup_success = False
        try:
            config = self._load_config()
            video_config = config.get("video_input", {})

            if self.video_input: # Release previous
                 if hasattr(self.video_input, 'release'): self.video_input.release()
                 self.video_input = None
            if VideoInput:
                self.video_input = VideoInput(config=video_config)
                if not self.video_input.initialized: raise RuntimeError("VideoInput failed.")
                fps = self.video_input.get_fps(); self.sampling_rate = fps if fps > 0 else 30.0
            else: # Fallback
                self.video_input = cv2.VideoCapture(0)
                if not self.video_input.isOpened(): self.video_input = cv2.VideoCapture(1)
                if not self.video_input.isOpened(): raise RuntimeError("cv2.VideoCapture failed.")
                self.video_input.get_frame = lambda: self.video_input.read() # type: ignore
                self.video_input.release = lambda: self.video_input.release() # type: ignore
                self.sampling_rate = 30.0
            print(f"[Worker] Using sampling rate: {self.sampling_rate:.2f} Hz")
            print("[Worker] Initial setup successful.")
            setup_success = True
            error_msg = "Video setup successful."

        except Exception as e:
            error_msg = f"Error during initial setup: {e}"
            print(error_msg); traceback.print_exc()
            if hasattr(self.video_input, 'release'): self.video_input.release()
            self.video_input = None
            setup_success = False
        finally:
            self.setup_finished.emit(setup_success, error_msg)
            if setup_success:
                # --- SET _running = True HERE ---
                # Indicate that the worker intends to proceed with initialization and running
                self._running = True
                # --- Schedule component initialization ---
                QTimer.singleShot(50, self._run_next_init_step)
            return setup_success

    def _run_next_init_step(self):
        """Runs the next step of the component initialization process."""
        # --- REMOVED check for self._running here ---
        # The flag is now set before this is called.
        # stop() method will set it to False if needed during init.
        if not self._running: # Still check just in case stop() was called right between setup and this
             print("[Worker] Stop called just before init step.")
             return

        if self._init_step == 0:
            self._initialize_pose_detector()
        elif self._init_step == 1:
            self._initialize_roi_calculator()
        elif self._init_step == 2:
            self._initialize_pipeline_manager()

    def _initialize_pose_detector(self):
        """Initializes PoseDetector."""
        if not self._running: return # Check flag before potentially blocking call
        print("[Worker] Initializing PoseDetector...")
        success = False; msg = ""
        try:
            pose_config = self.current_config.get("pose_detector", {})
            if self.pose_detector: self.pose_detector.close()
            self.pose_detector = PoseDetector(config=pose_config)
            if not self.pose_detector.initialized: raise RuntimeError("PoseDetector internal initialization failed.")
            print("[Worker] PoseDetector Initialized.")
            success = True; msg = "PoseDetector OK."
            self._init_step += 1
            if self._running: QTimer.singleShot(10, self._run_next_init_step) # Schedule next only if still running
        except Exception as e:
            msg = f"Error initializing PoseDetector: {e}"; print(f"[Worker] {msg}"); traceback.print_exc()
            self.pose_detector = None; success = False
        finally:
            self.component_initialized.emit("PoseDetector", success, msg)
            if not success: self._initialization_failed()

    def _initialize_roi_calculator(self):
        """Initializes CoarseRoiCalculator."""
        if not self._running: return
        print("[Worker] Initializing CoarseRoiCalculator...")
        success = False; msg = ""
        try:
            coarse_roi_config = self.current_config.get("coarse_roi_calculator", {})
            self.coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_config)
            print("[Worker] CoarseRoiCalculator Initialized.")
            success = True; msg = "RoiCalculator OK."
            self._init_step += 1
            if self._running: QTimer.singleShot(10, self._run_next_init_step)
        except Exception as e:
            msg = f"Error initializing CoarseRoiCalculator: {e}"; print(f"[Worker] {msg}"); traceback.print_exc()
            self.coarse_roi_calculator = None; success = False
        finally:
            self.component_initialized.emit("RoiCalculator", success, msg)
            if not success: self._initialization_failed()

    def _initialize_pipeline_manager(self):
        """Initializes PipelineManager."""
        if not self._running: return
        print("[Worker] Initializing PipelineManager...")
        success = False; msg = ""
        try:
            if self.pipeline_manager: self.pipeline_manager.close()
            self.pipeline_manager = PipelineManager(config=self.current_config, sampling_rate=self.sampling_rate)
            print("[Worker] PipelineManager Initialized.")
            success = True; msg = "PipelineManager OK."
            self._init_step += 1
            self._components_initialized = True
            self.state = WorkerState.PREVIEWING # Ready for preview
            print("[Worker] All components initialized successfully.")
            # --- Start the run loop now ---
            if self._running: QTimer.singleShot(0, self.run) # Schedule run to start immediately
        except Exception as e:
            msg = f"Error initializing PipelineManager: {e}"; print(f"[Worker] {msg}"); traceback.print_exc()
            self.pipeline_manager = None; success = False
        finally:
            self.component_initialized.emit("PipelineManager", success, msg)
            if not success: self._initialization_failed()

    def _initialization_failed(self):
        """Handle overall component initialization failure."""
        print("[Worker] Component initialization failed. Worker will not run.")
        self._components_initialized = False
        self.state = WorkerState.INITIALIZING
        self._running = False # Ensure running flag is false if init fails
        self.finished.emit() # Signal that this worker path is done


    def run(self):
        """Main processing loop. Starts AFTER successful initialization."""
        if not self._running: # Check if stop was called before run could start
            print("[Worker Run] Stop called before run loop started.")
            self.finished.emit(); return
        if self.state != WorkerState.PREVIEWING: # Should be previewing after successful init
            print(f"[Worker Run] Unexpected state {self.state} at run start.")
            self.processing_error.emit(f"Worker started in unexpected state {self.state}")
            self._running = False; self.finished.emit(); return

        print(f"[Worker Run Start] Starting processing loop in state: {self.state}")

        loop_count = 0
        while self._running:
            start_loop_time = time.perf_counter()
            loop_count += 1
            try:
                # print(f"[Worker Loop {loop_count}] Getting frame...") # Noisy diagnostic
                success, frame = self.video_input.get_frame()
                if not success or frame is None:
                    print("[Worker] End of video source or cannot read frame.")
                    self._running = False; break

                # Component check should not be needed here if init logic is correct
                # if not self._components_initialized: ...

                processed_frame = frame.copy()
                results = None

                # --- State-Dependent Processing ---
                if self.state == WorkerState.PREVIEWING:
                    # print(f"[Worker Loop {loop_count}] In PREVIEWING state.") # Noisy diagnostic
                    if not self.pose_detector or not self.coarse_roi_calculator:
                         self.processing_error.emit("Preview components not ready."); time.sleep(0.1); continue

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    self.latest_landmarks = self.pose_detector.process_frame(image_rgb)
                    image_rgb.flags.writeable = True

                    if self.latest_landmarks:
                        frame_h, frame_w = frame.shape[:2]
                        self.latest_preview_roi = self.coarse_roi_calculator.calculate_coarse_roi(self.latest_landmarks, (frame_h, frame_w))
                    else: self.latest_preview_roi = []

                    if self.latest_landmarks: mp_drawing.draw_landmarks(processed_frame, self.latest_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    if self.latest_preview_roi:
                        for (x, y, w, h) in self.latest_preview_roi: cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # Cyan ROI

                    # print(f"[Worker Loop {loop_count}] Emitting frame...") # Noisy diagnostic
                    self.new_frame_ready.emit(processed_frame)
                    self.new_plot_data.emit([])
                    self.new_status.emit(0.0, False, SignalProcessor.PHASE_UNKNOWN)

                elif self.state == WorkerState.TRACKING:
                    # (Tracking logic remains the same)
                    if not self.locked_roi: print("[Worker] Error: Tracking active but no ROI locked."); self.processing_error.emit("Tracking started without ROI."); self.set_tracking_active(False); continue
                    if not self.pipeline_manager: print("[Worker] Error: PipelineManager not initialized for tracking."); self.processing_error.emit("PipelineManager Error."); self.set_tracking_active(False); continue
                    results = self.pipeline_manager.process_frame(frame)
                    for (x, y, w, h) in self.locked_roi: cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 3) # Green Locked ROI
                    # TODO: Draw features if enabled in UI
                    self.new_frame_ready.emit(processed_frame)
                    if results:
                        plot_data = results.get('filtered_signal_history', []); bpm = results.get('bpm', 0.0); valid = results.get('bpm_valid', False); phase = results.get('phase', SignalProcessor.PHASE_UNKNOWN)
                        self.new_plot_data.emit(plot_data); self.new_status.emit(bpm, valid, phase)
                    else:
                        self.new_plot_data.emit([]); self.new_status.emit(0.0, False, SignalProcessor.PHASE_UNKNOWN)

                # --- Loop Delay ---
                loop_duration = time.perf_counter() - start_loop_time
                target_interval = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.033
                sleep_time = max(0.001, target_interval - loop_duration)
                time.sleep(sleep_time)

            except Exception as e:
                error_msg = f"Error in worker loop (State: {self.state}): {e}"
                print(error_msg); traceback.print_exc()
                self.processing_error.emit(error_msg)
                time.sleep(0.1)

        print("[Worker] Stopping processing loop...")
        if hasattr(self.video_input, 'release'): self.video_input.release()
        if self.pipeline_manager: self.pipeline_manager.close()
        if self.pose_detector: self.pose_detector.close()
        print("[Worker] Resources released.")
        self.finished.emit()

    def stop(self):
        print("[Worker] Stop requested.")
        self._running = False # This flag controls both init steps and run loop

    def set_tracking_active(self, active: bool):
        # (Logic remains the same)
        if active and not self._components_initialized: print("[Worker] Cannot start tracking: Components not yet initialized."); self.processing_error.emit("Components still initializing, please wait."); return
        if active and self.state == WorkerState.PREVIEWING:
            if not self.latest_preview_roi: print("[Worker] Cannot start tracking: No valid ROI found in preview."); self.processing_error.emit("Cannot start tracking: Position yourself for ROI detection."); return
            print(f"[Worker] Locking ROI {self.latest_preview_roi} and starting tracking."); self.locked_roi = copy.deepcopy(self.latest_preview_roi); self.state = WorkerState.TRACKING
            if self.pipeline_manager:
                self.pipeline_manager.set_tracking_roi(self.locked_roi)
                try: self.pipeline_manager.signal_processor = SignalProcessor(config=self.current_config.get('signal_processor', {}), sampling_rate=self.sampling_rate); print("[Worker] SignalProcessor reset.")
                except Exception as e_sp_reset: print(f"[Worker] Error resetting SignalProcessor: {e_sp_reset}"); self.processing_error.emit("Error resetting signal processor.")
            else: print("[Worker] Error: PipelineManager not available to set ROI/reset."); self.processing_error.emit("PipelineManager error on tracking start."); self.state = WorkerState.PREVIEWING; self.locked_roi = []
        elif not active and self.state == WorkerState.TRACKING: print("[Worker] Stopping tracking and returning to preview mode."); self.state = WorkerState.PREVIEWING; self.locked_roi = []
        else: print(f"[Worker] set_tracking_active called in state {self.state} with active={active}. No change.")

    def reload_profile(self, new_config_path):
        # (Logic remains the same)
        print(f"[Worker] Reload profile requested: {new_config_path}"); self.state = WorkerState.INITIALIZING; self.set_tracking_active(False); self._components_initialized = False; self._init_step = 0; time.sleep(0.2)
        self.config_path = new_config_path; self.setup()


# --- Main Application Setup ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    config_file = os.path.join(PROFILES_DIR, DEFAULT_PROFILE)
    if not os.path.exists(PROFILES_DIR): print(f"Warning: Profiles directory does not exist: '{PROFILES_DIR}'")
    if not os.path.exists(config_file): print(f"Warning: Default profile '{DEFAULT_PROFILE}' not found. Using empty config."); config_file = ""

    main_window = MainWindow(config_file=config_file, profiles_dir=PROFILES_DIR)
    worker = PipelineWorker(config_path=config_file)
    worker_thread = QThread()
    worker.moveToThread(worker_thread)

    # --- Connect Signals and Slots ---
    worker.new_frame_ready.connect(main_window.update_webcam_feed)
    worker.new_plot_data.connect(main_window.update_plot)
    worker.new_status.connect(main_window.update_status_labels)
    worker.processing_error.connect(main_window.show_error_message)
    worker.setup_finished.connect(main_window.handle_worker_setup_finished)
    worker.component_initialized.connect(main_window.handle_component_initialized)
    worker.finished.connect(worker_thread.quit)
    worker.finished.connect(worker.deleteLater)

    main_window.start_tracking_signal.connect(lambda: worker.set_tracking_active(True))
    main_window.stop_tracking_signal.connect(lambda: worker.set_tracking_active(False))
    main_window.load_profile_signal.connect(worker.reload_profile)

    # --- Connect thread start/stop ---
    worker_thread.started.connect(worker.setup) # Trigger setup when thread starts
    # Run loop is triggered internally after successful initialization via QTimer

    QTimer.singleShot(100, worker_thread.start) # Start the thread after event loop begins

    app.aboutToQuit.connect(worker.stop) # Tell worker loop to stop
    worker_thread.finished.connect(app.quit) # Quit app when thread finishes

    main_window.show()
    sys.exit(app.exec())
