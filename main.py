# c:\Software Development\Respiration Modulator M4L\main.py
# main.py
# Main entry point for the Respiration Modulator application.
# Sets up the PyQt6 application, creates the UI window, and manages the backend worker thread.
# MODIFIED: Added settings application logic (apply_new_settings slot, signals).
# FIXED: Added missing pyqtSlot import.
# MODIFIED: Refactored worker run loop to use QTimer for responsiveness.

import sys
import os
import time # For sleep if needed
import traceback
import json # Needed by worker
import numpy as np # Needed for PipelineWorker signals
import cv2 # Needed for VideoCapture fallback
import mediapipe as mp # Needed for drawing landmarks
import copy # For deepcopy
import threading # <<< IMPORT THREADING
import multiprocessing as mp_proc # Use alias to avoid confusion with mediapipe
from datetime import datetime # For recording directory names

# --- PyQt6 Imports ---
try:
    from PyQt6.QtWidgets import QApplication
    # Import Qt explicitly for ConnectionType and pyqtSlot
    from PyQt6.QtCore import QThread, QObject, pyqtSignal, QTimer, Qt, pyqtSlot # <<< FIXED: Added pyqtSlot
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
    from ui.visualizer import SignalVisualizer # <<< Import the new visualizer

    try:
        from video_input import VideoInput
    except ImportError:
        print("Warning: video_input.py not found. Using direct cv2.VideoCapture fallback in worker.")
        VideoInput = None
    # Import all components needed by the worker now
    from pipeline_manager import PipelineManager
    from pose_detector import PoseDetector
    from coarse_roi_calculator import CoarseRoiCalculator
    from signal_processor import SignalProcessor # Import for constants and re-init
    from feature_tracker import FeatureTracker # Import for re-init
    from signal_generator import SignalGenerator # Import for re-init

    from osc_manager import OSCManager # <<< Import OSCManager
    from raw_data_recorder import DataRecorder # Import the recorder class
except ImportError as e:
    print(f"Fatal Error: Failed to import necessary modules from 'src' or 'src/ui': {e}")
    traceback.print_exc(); sys.exit(1)
except Exception as e_general:
     print(f"An unexpected error occurred during imports: {e_general}")
     traceback.print_exc(); sys.exit(1)


# --- Constants ---
DEFAULT_PROFILE = "test_profile 2.json"
PROFILES_DIR = os.path.join(script_dir, "profiles")
RECORDINGS_DIR = os.path.join(script_dir, "recordings") # Base dir for recordings

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
    Handles dynamic settings updates.
    """
    # --- Signals ---
    new_frame_ready = pyqtSignal(np.ndarray)
    # new_plot_data = pyqtSignal(list) # Redundant, covered by new_pipeline_results
    new_status = pyqtSignal(float, bool, int)
    new_filtered_signal_value = pyqtSignal(float) # <<< Signal for the visualizer
    processing_error = pyqtSignal(str)
    finished = pyqtSignal()
    setup_finished = pyqtSignal(bool, str)
    component_initialized = pyqtSignal(str, bool, str)
    current_settings_signal = pyqtSignal(dict) # Emit initial/current settings to UI
    new_pipeline_results = pyqtSignal(dict) # <<< ADD THIS SIGNAL DEFINITION
    profile_saved_signal = pyqtSignal(str, bool, str) # file_path, success, message

    # --- ADDED: Timer for scheduling run loop ---
    def __init__(self, config_path, parent=None):
        super().__init__(parent)
        self.config_path = config_path
        self.current_config = {}
        self.video_input = None
        self.sampling_rate = 30.0
        self.pose_detector = None
        self.coarse_roi_calculator = None
        self.pipeline_manager = None
        self._running = False
        self.state = WorkerState.INITIALIZING
        self.latest_preview_roi = []
        self.locked_roi = []
        self.latest_landmarks = None
        self._components_initialized = False
        self._init_step = 0
        # --- Overlay states ---
        self.show_pose = True # Default based on UI
        self.show_roi = True  # Default based on UI
        self.show_features = False # Default based on UI
        # --- Timer ---
        self.run_timer = QTimer(self) # Timer to drive the run loop
        # --- Revert to DirectConnection for potentially lower latency ---
        self.run_timer.timeout.connect(self.run, Qt.ConnectionType.DirectConnection)
        self._is_transitioning_tracking = False # Add flag
        self.run_timer.setTimerType(Qt.TimerType.PreciseTimer) # Or AccurateTimer
        self._loop_start_time = 0 # To calculate processing time

        # --- Recording Attributes ---
        self.recording_enabled = False
        self.recorder_process = None
        self.recorder_queue = None
        self.recorder_stop_event = None

        # --- OSC Manager ---
        self.osc_manager = None

        # --- Breath Phase Mapping ---
        self.phase_map = {
            SignalProcessor.PHASE_INHALE: "inhaling",
            SignalProcessor.PHASE_EXHALE: "exhaling",
            SignalProcessor.PHASE_UNKNOWN: "neutral"
        }
    def _load_config(self):
        """Loads the configuration from the specified file path."""
        config = {}
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(f"[Worker] Loaded config from: {self.config_path}") # Verbose for tracking
                self.current_config = config
                # --- Read Recording Setting ---
                self.recording_enabled = config.get("enable_raw_data_recording", False)
                print(f"[Worker] Raw data recording enabled: {self.recording_enabled}")
            else:
                print(f"[Worker] Warning: Config file '{self.config_path}' not found. Using empty config.")
                self.current_config = {}
                # Optionally create a default config here if needed
        except Exception as e:
            print(f"[Worker] Error loading config '{self.config_path}': {e}")
            self.current_config = {} # Fallback to empty on error
            raise # Re-raise to signal the failure upwards
        return self.current_config

    def _initialize_osc_manager(self):
        """Initializes the OSCManager based on the current configuration."""
        if self.osc_manager:
            print("[Worker] Stopping existing OSC manager before re-initializing.")
            self.osc_manager.stop_server()
            self.osc_manager = None

        osc_config = self.current_config.get("osc", {})
        if osc_config.get("enabled", False):
            try:
                send_ip = osc_config["send_ip"]
                send_port = int(osc_config["send_port"])
                receive_ip = osc_config["receive_ip"]
                receive_port = int(osc_config["receive_port"])

                self.osc_manager = OSCManager(
                    send_ip=send_ip,
                    send_port=send_port,
                    receive_ip=receive_ip,
                    receive_port=receive_port
                )
                self.osc_manager.start_server()
                print("[Worker] OSCManager initialized and server started.")
            except KeyError as e:
                print(f"[Worker] OSC configuration missing key: {e}. OSC features will be disabled.")
            except Exception as e:
                print(f"[Worker] Failed to initialize OSCManager: {e}. OSC features will be disabled.")
                self.osc_manager = None
    def _load_config_from_path(self, file_path):
        """Loads a configuration dictionary from a given file path."""
        config = {}
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config = json.load(f)
                print(f"[Worker] Successfully loaded config from path: {file_path}")
            else:
                print(f"[Worker] Warning: Config file '{file_path}' not found for direct load. Returning empty config.")
        except Exception as e:
            print(f"[Worker] Error loading config from path '{file_path}': {e}")
        return config

    def setup(self):
        """Initial setup: Load config and initialize video source."""
        print("[Worker] Running initial setup (Video & Config)...")
        error_msg = ""
        setup_success = False
        try:
            # Load configuration first
            config = self._load_config()
            self._initialize_osc_manager() # Initialize OSC after config is loaded
            video_config = config.get("video_input", {})

            # Release existing video input if any
            if self.video_input:
                 if hasattr(self.video_input, 'release'): self.video_input.release()
                 self.video_input = None

            # Initialize VideoInput or fallback to cv2.VideoCapture
            if VideoInput:
                self.video_input = VideoInput(config=video_config)
                if not self.video_input.initialized:
                    raise RuntimeError("VideoInput failed to initialize.")
                fps = self.video_input.get_fps()
                self.sampling_rate = fps if fps > 0 else 30.0
            else: # Fallback
                # Try common camera indices
                capture_source = -1
                for i in range(4): # Try indices 0, 1, 2, 3
                    cap_test = cv2.VideoCapture(i)
                    if cap_test.isOpened():
                        print(f"[Worker] Found working camera at index {i}.")
                        capture_source = i
                        cap_test.release()
                        break
                    cap_test.release()

                if capture_source == -1:
                    raise RuntimeError("cv2.VideoCapture failed for indices 0-3.")

                self.video_input = cv2.VideoCapture(capture_source)
                if not self.video_input.isOpened(): # Double check
                     raise RuntimeError(f"cv2.VideoCapture failed for index {capture_source} even after check.")

                # Add compatible methods for the fallback
                self.video_input.get_frame = lambda: self.video_input.read() # type: ignore
                self.video_input.release = lambda: self.video_input.release() # type: ignore
                self.sampling_rate = 30.0 # Assume default FPS for fallback

            print(f"[Worker] Using sampling rate: {self.sampling_rate:.2f} Hz")
            print("[Worker] Initial setup successful.")
            setup_success = True
            error_msg = "Video setup successful."

        except Exception as e:
            error_msg = f"Error during initial setup: {e}"
            print(error_msg)
            traceback.print_exc()
            if hasattr(self.video_input, 'release'):
                self.video_input.release()
            self.video_input = None
            setup_success = False

        finally:
            # Emit signal about setup completion status
            self.setup_finished.emit(setup_success, error_msg)
            # If setup succeeded, start running the component initialization steps
            if setup_success:
                self._running = True
                # Use QTimer.singleShot to schedule the next step on the event loop
                QTimer.singleShot(50, self._run_next_init_step)
            return setup_success

    def _run_next_init_step(self):
        """Sequentially initializes pipeline components."""
        if not self._running:
            print("[Worker] Stop called just before init step.")
            return

        # Execute initialization steps one by one
        if self._init_step == 0:
            self._initialize_pose_detector()
        elif self._init_step == 1:
            self._initialize_pipeline_manager()

    def _initialize_pose_detector(self):
        """Initializes the PoseDetector component."""
        # if not self._running: return False # Removed: Allow init even if main loop isn't "running"
        print("[Worker] Initializing PoseDetector..."); success = False; msg = ""
        try:
            pose_config = self.current_config.get("pose_detector", {});
            # Ensure any previous instance is closed
            if self.pose_detector: self.pose_detector.close()
            self.pose_detector = PoseDetector(config=pose_config)
            if not self.pose_detector.initialized:
                raise RuntimeError("PoseDetector internal initialization failed.")
            print("[Worker] PoseDetector Initialized.")
            success = True; msg = "PoseDetector OK."
            # Orchestration of next step is handled by _run_next_init_step or reload_profile
            # self._init_step += 1 # Moved to _run_next_init_step
            # if self._running: QTimer.singleShot(10, self._run_next_init_step) # Moved
        except Exception as e:
            msg = f"Error initializing PoseDetector: {e}"
            print(f"[Worker] {msg}"); traceback.print_exc()
            self.pose_detector = None; success = False
        finally:
            # Emit signal regardless of success
            self.component_initialized.emit("PoseDetector", success, msg);
            # If failed, stop the initialization sequence
            if not success: self._initialization_failed() # This will stop further init steps
        return success # Return success status

    def _initialize_roi_calculator(self):
        """Initializes the CoarseRoiCalculator component."""
        # if not self._running: return False # Removed: Allow init even if main loop isn't "running"
        print("[Worker] Initializing CoarseRoiCalculator..."); success = False; msg = ""
        try:
            coarse_roi_config = self.current_config.get("coarse_roi_calculator", {})
            # No close method needed for RoiCalculator currently
            self.coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_config)
            print("[Worker] CoarseRoiCalculator Initialized.")
            success = True; msg = "RoiCalculator OK."
            # Orchestration of next step is handled by _run_next_init_step or reload_profile
            # self._init_step += 1 # Moved
            # if self._running: QTimer.singleShot(10, self._run_next_init_step) # Moved
        except Exception as e:
            msg = f"Error initializing CoarseRoiCalculator: {e}"
            print(f"[Worker] {msg}"); traceback.print_exc()
            self.coarse_roi_calculator = None; success = False
        finally:
            # Emit signal regardless of success
            self.component_initialized.emit("RoiCalculator", success, msg);
            # If failed, stop the initialization sequence
            if not success: self._initialization_failed() # This will stop further init steps
        return success # Return success status

    def _initialize_pipeline_manager(self):
        """Initializes the PipelineManager and emits initial settings."""
        # if not self._running: return False # Removed: Allow init even if main loop isn't "running"
        print("[Worker] Initializing PipelineManager..."); success = False; msg = ""
        try:
            # Close existing manager if any
            if self.pipeline_manager: self.pipeline_manager.close()
            # Initialize with current config and sampling rate
            self.pipeline_manager = PipelineManager(config=self.current_config, sampling_rate=self.sampling_rate)
            print("[Worker] PipelineManager Initialized.")
            success = True; msg = "PipelineManager OK."
            # Final setup steps (state, timer start, settings emit) are handled by the orchestrator
            # (_run_next_init_step or reload_profile)

        except Exception as e:
            msg = f"Error initializing PipelineManager: {e}"
            print(f"[Worker] {msg}"); traceback.print_exc()
            self.pipeline_manager = None
            success = False
        finally:
            # Emit signal regardless of success
            self.component_initialized.emit("PipelineManager", success, msg)
            # If failed, stop the initialization sequence
            if not success: self._initialization_failed() # This will stop further init steps
        return success # Return success status

    def _run_next_init_step(self):
        """Sequentially initializes pipeline components during initial setup."""
        if not self._running:
            print("[Worker] Stop called during initial setup sequence.")
            return

        success = False
        if self._init_step == 0:
            if self._initialize_pose_detector(): # Call the refactored method
                self._init_step += 1
                if self._running: QTimer.singleShot(10, self._run_next_init_step)
        elif self._init_step == 1:
            if self._initialize_roi_calculator(): # Call the refactored method
                self._init_step += 1
                if self._running: QTimer.singleShot(10, self._run_next_init_step)
        elif self._init_step == 2:
            if self._initialize_pipeline_manager(): # Call the refactored method
                # This is the last step of initial setup
                self._components_initialized = True
                self.state = WorkerState.PREVIEWING
                print("[Worker] All components initialized successfully (initial setup).")
                initial_settings = self._extract_relevant_settings(self.current_config)
                self.current_settings_signal.emit(initial_settings)
                if self._running:
                    print(f"[Worker Run Start] Starting processing loop via timer in state: {self.state}")
                    self._loop_start_time = time.perf_counter()
                    self.run_timer.start(1)
            # No further steps after PipelineManager in initial setup
        # If any step failed, _initialization_failed() would have been called by the
        # respective _initialize_X method, and _running would be false, stopping this chain.


    def _extract_relevant_settings(self, config_dict):
        """Extracts settings relevant for UI population from the full config."""
        # Define default structures for each component's settings
        default_ft = {
            'OPTICAL_FLOW_PARAMS': {
                'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7},
                'lk_params': {'winSize': [15, 15], 'maxLevel': 2}
            }
        }
        default_sg = {'SIGNAL_AGGREGATION_METHOD': 'median'}
        default_sp = {'SIGNAL_FILTER_LOW_HZ': 0.1, 'SIGNAL_FILTER_HIGH_HZ': 2.0, 'PEAK_DETECT_PROMINENCE': None}

        # Get settings, falling back to defaults if the section is missing
        ft_settings = config_dict.get('feature_tracker', default_ft)
        sg_settings = config_dict.get('signal_generator', default_sg)
        sp_settings = config_dict.get('signal_processor', default_sp)

        # Ensure nested dictionaries exist in feature_tracker settings
        if 'OPTICAL_FLOW_PARAMS' not in ft_settings: ft_settings['OPTICAL_FLOW_PARAMS'] = default_ft['OPTICAL_FLOW_PARAMS']
        if 'feature_params' not in ft_settings['OPTICAL_FLOW_PARAMS']: ft_settings['OPTICAL_FLOW_PARAMS']['feature_params'] = default_ft['OPTICAL_FLOW_PARAMS']['feature_params']
        if 'lk_params' not in ft_settings['OPTICAL_FLOW_PARAMS']: ft_settings['OPTICAL_FLOW_PARAMS']['lk_params'] = default_ft['OPTICAL_FLOW_PARAMS']['lk_params']

        relevant_settings = {
            'feature_tracker': ft_settings,
            'signal_generator': sg_settings,
            'signal_processor': sp_settings
        }
        return relevant_settings


    def _initialization_failed(self):
        """Handles component initialization failure."""
        print("[Worker] Component initialization failed. Worker will not run.")
        self._components_initialized = False
        self.state = WorkerState.INITIALIZING
        self._running = False # Ensure worker stops trying to run
        # Optionally emit finished signal here if appropriate
        # self.finished.emit()

    # --- Recorder Process Management ---
    def _start_recorder_process(self):
        """Starts the DataRecorder in a separate process."""
        # print("[Worker Debug] _start_recorder_process: Recorder already running, skipping.")
        if self.recorder_process and self.recorder_process.is_alive():
            print("[Worker] Recorder process already running.")
            # print("[Worker Debug] Recorder already running, skipping start.") # DEBUG
            return

        try:
            print("[Worker] Starting raw data recorder process...")
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            run_dir_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.recorder_queue = mp_proc.Queue()
            self.recorder_stop_event = mp_proc.Event()

            # Create an instance of the recorder class
            recorder_instance = DataRecorder(
                data_queue=self.recorder_queue,
                stop_event=self.recorder_stop_event,
                run_dir_name=run_dir_name
            )

            # Start the process, targeting the recorder's run method
            # print("[Worker Debug] _start_recorder_process: About to start DataRecorder process.")
            self.recorder_process = mp_proc.Process(target=recorder_instance.run, daemon=True)
            self.recorder_process.start()
            # print("[Worker Debug] Recorder process initiated.") # DEBUG
            print(f"[Worker] Recorder process started (PID: {self.recorder_process.pid}). Saving to {run_dir_name}")
        except Exception as e:
            print(f"[Worker] Failed to start recorder process: {e}")
            # print("[Worker Debug] Exception during recorder start.") # DEBUG
            traceback.print_exc()
    # --- NEW SLOT TO HANDLE SAVE REQUESTS ---
    @pyqtSlot(str)
    def handle_save_profile(self, file_path):
        """Saves the current configuration to the specified file path."""
        print(f"[Worker Slot] handle_save_profile called for path: {file_path}")
        try:
            # Ensure the directory exists
            save_dir = os.path.dirname(file_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"[Worker] Created directory: {save_dir}")

            # Use the worker's current_config which should be up-to-date
            settings_to_save = self.current_config

            # Write the settings to the file
            with open(file_path, 'w') as f:
                json.dump(settings_to_save, f, indent=4)

            print(f"[Worker] Profile saved successfully: {file_path}")
            # Emit success signal *after* saving
            self.profile_saved_signal.emit(file_path, True, f"Profile '{os.path.basename(file_path)}' saved.")

        except Exception as e:
            error_msg = f"Failed to save profile '{os.path.basename(file_path)}': {e}"
            print(f"[Worker] {error_msg}")
            traceback.print_exc()
            # Emit failure signal
            self.profile_saved_signal.emit(file_path, False, error_msg)
        finally:
             pass # Keep finally block structure if needed, or remove if empty

    # --- ADDED: Wrapper slots for direct connection ---
    @pyqtSlot()
    def set_tracking_active_true(self):
        """Slot to specifically call set_tracking_active(True)."""
        self.set_tracking_active(True)

    @pyqtSlot()
    def set_tracking_active_false(self):
        """Slot to specifically call set_tracking_active(False)."""
        self.set_tracking_active(False)
    # --- END WRAPPERS ---
    # --- MODIFIED: run method processes ONE frame and reschedules ---
    # @pyqtSlot() # No longer needs to be a slot if only called by timer
    def run(self):
        """Processes a single frame and schedules the next iteration."""
        # --- IMMEDIATE CHECK: Exit if stop has been requested ---
        if not self._running:
            # Ensure timer is stopped if this is reached unexpectedly
            # if self.run_timer.isActive(): # This check might be redundant if stop() handles it
                # print("[Worker Run Debug] Stopping timer from top check.")
            self.run_timer.stop() # Stop timer if active
            self._cleanup_resources() # Perform cleanup
            self.finished.emit()
            return

        # --- Check if components are initialized ---
        if not self._components_initialized:
             print("[Worker Run] Components not initialized. Stopping timer.")
             if self.run_timer.isActive():
                 self.run_timer.stop()
             self.processing_error.emit("Initialization failed. Cannot run.")
             self._running = False
             self._cleanup_resources()
             self.finished.emit()
             return

        # --- Process one frame ---
        try:
            # --- Expanded Timing Start ---
            t_cycle_start = time.perf_counter()

            # Calculate time since last loop start for FPS control
            frame_start_time = time.perf_counter()
            time_since_last_loop = frame_start_time - self._loop_start_time
            self._loop_start_time = frame_start_time # Reset for next iteration

            t_grab_start = time.perf_counter()
            success, frame = self.video_input.get_frame()
            t_grab_end = time.perf_counter()
            t_frame_grab_ms = (t_grab_end - t_grab_start) * 1000

            if not success or frame is None:
                print("[Worker] End of video source or cannot read frame. Stopping.")
                self._running = False
                if self.run_timer.isActive():
                    self.run_timer.stop()
                self._cleanup_resources()
                self.finished.emit()
                return

            t_draw_overlays_ms = 0.0 # Initialize overlay drawing time
            processed_frame = frame.copy()
            results = None
            current_tracked_points = None

            # --- State-Dependent Processing & Drawing ---
            if self.state == WorkerState.PREVIEWING:
                # Preview logic: Run pose detection and ROI calculation
                if not self.pose_detector or not self.coarse_roi_calculator:
                    self.processing_error.emit("Preview components not ready.")
                    # Schedule next run slightly later to avoid busy loop
                    if self._running: self.run_timer.start(50)
                    return

                t_preview_proc_start = time.perf_counter()
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                self.latest_landmarks = self.pose_detector.process_frame(image_rgb)
                image_rgb.flags.writeable = True

                if self.latest_landmarks:
                    frame_h, frame_w = frame.shape[:2]
                    self.latest_preview_roi = self.coarse_roi_calculator.calculate_coarse_roi(self.latest_landmarks, (frame_h, frame_w))
                else:
                    self.latest_preview_roi = []

                t_preview_proc_end = time.perf_counter()
                t_preview_overhead_ms = (t_preview_proc_end - t_preview_proc_start) * 1000

                # Draw overlays for preview
                t_overlay_start = time.perf_counter()
                if self.show_pose and self.latest_landmarks:
                    mp_drawing.draw_landmarks(processed_frame, self.latest_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                if self.show_roi and self.latest_preview_roi:
                    for (x, y, w, h) in self.latest_preview_roi:
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # Cyan ROI
                t_overlay_end = time.perf_counter()
                t_draw_overlays_ms = (t_overlay_end - t_overlay_start) * 1000

                # --- Emit preview results ---
                # --- Expanded Timing Calculation for Preview ---
                t_cycle_end = time.perf_counter()
                t_worker_cycle_ms = (t_cycle_end - t_cycle_start) * 1000
                # print(f"[Timing Preview (ms)] Grab: {t_frame_grab_ms:.1f}, "
                #       f"Pose/ROI: {t_preview_overhead_ms:.1f}, "
                #       f"Draw: {t_draw_overlays_ms:.1f}, "
                #       f"CycleTotal: {t_worker_cycle_ms:.1f}")

                self.new_frame_ready.emit(processed_frame)
                # self.new_plot_data.emit([]) # No plot data in preview
                self.new_status.emit(0.0, False, SignalProcessor.PHASE_UNKNOWN) # No status in preview

            elif self.state == WorkerState.TRACKING:
                # Tracking logic: Run the full pipeline via PipelineManager
                if not self.locked_roi:
                    print("[Worker] Error: Tracking active but no ROI locked.")
                    self.processing_error.emit("Tracking error: ROI/Pipeline not ready.")
                    self.set_tracking_active(False) # Revert to preview
                    # Schedule next run
                    if self._running: self.run_timer.start(1)
                    return
                if not self.pipeline_manager:
                    print("[Worker] Error: PipelineManager not initialized for tracking.")
                    self.processing_error.emit("Tracking error: ROI/Pipeline not ready.")
                    self.set_tracking_active(False) # Revert to preview
                    # Schedule next run
                    if self._running: self.run_timer.start(1)
                    return

                # --- REMOVED: Redundant set_tracking_roi call ---
                # The ROI is set during the state transition in set_tracking_active

                results = self.pipeline_manager.process_frame(frame)

                # --- Send Data to Recorder (if enabled) ---
                # --- Add more detailed debug for recorder data sending ---
                can_send_to_recorder = self.recording_enabled and self.recorder_queue is not None and results is not None
                # print(f"[Worker Debug Rec] Pre-send check: recording_enabled={self.recording_enabled}, recorder_queue_exists={self.recorder_queue is not None}, results_exist={results is not None}")

                if can_send_to_recorder:
                    raw_signal_from_results = results.get('raw_signal')
                    tracked_points_from_results = results.get('tracked_points')

                    # --- NEW DETAILED DEBUG ---
                    print(f"[Worker Debug Rec Data] Check before queueing: "
                          f"recording_enabled={self.recording_enabled}, "
                          f"recorder_queue_valid={self.recorder_queue is not None}, "
                          f"results_valid={results is not None}, "
                          f"raw_signal_valid={raw_signal_from_results is not None}, "
                          f"tracked_points_valid={tracked_points_from_results is not None}")
                    if raw_signal_from_results is not None:
                        print(f"[Worker Debug Rec Data] raw_signal_value_to_queue={raw_signal_from_results:.4f}")
                    if tracked_points_from_results is not None:
                        print(f"[Worker Debug Rec Data] tracked_points_to_queue_shape={tracked_points_from_results.shape if hasattr(tracked_points_from_results, 'shape') else 'N/A'}")
                    # --- END NEW DETAILED DEBUG ---

                    try:
                        # *** IMPORTANT: Ensure these keys exist in your 'results' dict ***
                        # --- COMMENTED OUT DEBUG PRINT ---
                        # print(f"[Worker Debug] Results keys: {results.keys() if results else 'None'}")

                        if raw_signal_from_results is not None and tracked_points_from_results is not None:
                            timestamp = time.time()
                            # Put data onto the queue for the recorder process
                            # This is the key debug print for successful queuing attempt:
                            # print(f"[Worker Debug] Putting data onto recorder queue: ts={timestamp:.2f}, signal={raw_signal_from_results:.4f}, points_shape={tracked_points_from_results.shape if hasattr(tracked_points_from_results, 'shape') else 'N/A'}") # DEBUG
                            self.recorder_queue.put_nowait((timestamp, raw_signal_from_results, tracked_points_from_results))
                        else:
                            print(f"[Worker Debug Rec Skip] Did NOT queue data: raw_signal_from_results is None ({raw_signal_from_results is None}) or tracked_points_from_results is None ({tracked_points_from_results is None}).")
                    except mp_proc.queues.Full:
                        print("[Worker] Warning: Recorder queue is full. Data point dropped.")

                # --- Get tracked points from results ---
                if results:
                    current_tracked_points = results.get('tracked_points') # Adjust key if needed

                # --- Draw Tracking Overlays ---
                t_overlay_start = time.perf_counter()
                # Draw locked ROI
                if self.show_roi and self.locked_roi:
                    for (x, y, w, h) in self.locked_roi:
                        cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 3) # Green Locked ROI
                # Optionally draw pose if available and enabled
                if self.show_pose and self.latest_landmarks:
                    mp_drawing.draw_landmarks(processed_frame, self.latest_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # --- Feature Drawing Logic ---
                if self.show_features:
                    if current_tracked_points is not None and isinstance(current_tracked_points, np.ndarray):
                        point_color = (0, 0, 255); points_drawn = 0
                        try:
                            # Ensure points are float32 and correct shape (N, 1, 2) or (N, 2)
                            points_to_draw_float = current_tracked_points.astype(np.float32)
                            if points_to_draw_float.ndim >= 2 and points_to_draw_float.shape[-1] == 2:
                                # Reshape to (N, 2) for easier iteration
                                points_to_draw = points_to_draw_float.reshape(-1, 2)
                                for i, point in enumerate(points_to_draw):
                                    if point is not None and point.shape == (2,):
                                        x_pt, y_pt = int(point[0]), int(point[1])
                                        # Check bounds before drawing
                                        frame_h_draw, frame_w_draw = processed_frame.shape[:2]
                                        if 0 <= x_pt < frame_w_draw and 0 <= y_pt < frame_h_draw:
                                            cv2.circle(processed_frame, (x_pt, y_pt), 3, point_color, -1)
                                            points_drawn += 1
                            else:
                                 pass # Avoid excessive logging
                        except Exception as draw_err:
                            print(f"!!! ERROR during feature drawing loop: {draw_err}")
                            traceback.print_exc() # Print full traceback for drawing errors
                t_overlay_end = time.perf_counter()
                t_draw_overlays_ms = (t_overlay_end - t_overlay_start) * 1000

                # --- Emit Tracking Results ---
                # --- Expanded Timing Calculation for Tracking ---
                t_cycle_end = time.perf_counter()
                t_worker_cycle_ms = (t_cycle_end - t_cycle_start) * 1000

                self.new_frame_ready.emit(processed_frame)
                if results:
                    # --- ADD PRINT STATEMENT FOR TIMING ---
                    # if 'timing_ms' in results:
                    #     pipeline_timings = results['timing_ms'] # type: ignore
                    #     # Print expanded timings
                    #     print(f"[Timing (ms)] Grab: {t_frame_grab_ms:.1f}, "
                    #           f"FT: {pipeline_timings.get('feature_tracker', 0):.1f}, "
                    #           f"SG: {pipeline_timings.get('signal_generator', 0):.1f}, "
                    #           f"SP: {pipeline_timings.get('signal_processor', 0):.1f}, "
                    #           f"Draw: {t_draw_overlays_ms:.1f}, "
                    #           f"PipeTotal: {pipeline_timings.get('total_pipeline', 0):.1f}, "
                    #           f"CycleTotal: {t_worker_cycle_ms:.1f}")
                    # --- END PRINT STATEMENT ---

                    # --- Emit latest filtered value for visualizer ---
                    latest_val = results.get('latest_filtered_value', 0.0)
                    self.new_filtered_signal_value.emit(latest_val)
                    # ---

                    # Emit plot data and status (adjust based on your reverted SignalProcessor interface)
                    # Example using get methods from the reverted pipeline_manager context:
                    bpm, bpm_valid = self.pipeline_manager.signal_processor.get_bpm()
                    phase = self.pipeline_manager.signal_processor.get_phase()
                    plot_data = self.pipeline_manager.signal_processor.get_filtered_signal_buffer()
                    # Or if results dict contains them directly (less likely in reverted code):
                    # plot_data = results.get('filtered_signal_history', [])
                    # bpm = results.get('bpm', 0.0)
                    # valid = results.get('bpm_valid', False)
                    # phase = results.get('phase', SignalProcessor.PHASE_UNKNOWN)

                    # --- Emit the full results dictionary ---
                    if self.osc_manager and results:
                        # Assuming 'latest_filtered_value' is the "filtered differential signal"
                        # And also using it as a placeholder for "processed level signal"
                        # You might need to adjust the keys based on what `pipeline_manager` actually returns.
                        filtered_diff_signal = results.get('latest_filtered_value', 0.0)
                        # If you have a distinct "processed level signal" in results, use its key here:
                        # e.g., processed_lvl_signal = results.get('processed_level_signal_key', 0.0)
                        processed_lvl_signal = results.get('processed_level_signal', 0.0) # Use the actual key

                        current_phase_int = self.pipeline_manager.signal_processor.get_phase()
                        breath_phase_str = self.phase_map.get(current_phase_int, "unknown")

                        self.osc_manager.send_filtered_differential_signal(filtered_diff_signal)
                        self.osc_manager.send_processed_level_signal(processed_lvl_signal)
                        self.osc_manager.send_breath_phase(breath_phase_str)

                        # You can log M4L connection status if needed:
                        # print(f"M4L Connected: {self.osc_manager.get_m4l_connection_status()}")

                    self.new_pipeline_results.emit(results) # <<< EMIT THE FULL RESULTS

                    # self.new_plot_data.emit(plot_data) # Redundant, UI uses new_pipeline_results
                    self.new_status.emit(bpm, bpm_valid, phase)
                else: # Handle case where pipeline_manager.process_frame returned None
                    # self.new_plot_data.emit([]) # Redundant
                    self.new_status.emit(0.0, False, SignalProcessor.PHASE_UNKNOWN)

            # --- Calculate next schedule time ---
            processing_duration = time.perf_counter() - frame_start_time
            target_interval = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.033
            # Calculate delay needed to approximate target interval
            delay_ms = max(1, int((target_interval - processing_duration) * 1000))

            # --- Reschedule the next run ---
            if self._running:
                self.run_timer.start(delay_ms)

        except Exception as e:
            error_msg = f"Error in worker processing cycle (State: {self.state}): {e}"
            print(error_msg); traceback.print_exc()
            self.processing_error.emit(error_msg)
            # Reschedule even after error to avoid stopping completely
            if self._running:
                self.run_timer.start(100) # Schedule after a short delay on error

    # --- ADDED: Centralized cleanup ---
    def _cleanup_resources(self):
        """Releases resources like video capture and pipeline components."""
        print("[Worker] Cleaning up resources...") # Removed "(excluding timer stop)" for clarity
        # DO NOT stop self.run_timer here. It must be stopped from the worker thread itself (handled in run()).
        # if self.run_timer.isActive(): self.run_timer.stop() # <<< REMOVED/COMMENTED
        if hasattr(self.video_input, 'release'):
            self.video_input.release()
            print("  Video input released.")
        if self.pipeline_manager:
            self.pipeline_manager.close()
            print("  PipelineManager closed.")
        if self.pose_detector:
            self.pose_detector.close()
            print("  PoseDetector closed.")
        if self.osc_manager:
            self.osc_manager.stop_server()
            print("  OSCManager stopped.")
        print("[Worker] Resource cleanup finished.")
        # --- Stop Recorder Process ---
        self._stop_recorder_process()

    def _stop_recorder_process(self):
        """Signals the recorder process to stop and waits for it."""
        if self.recorder_process and self.recorder_process.is_alive():
            # print("[Worker Debug] Attempting to stop recorder process...") # DEBUG
            print("[Worker] Stopping recorder process...")
            if self.recorder_stop_event:
                self.recorder_stop_event.set() # Signal the recorder loop to exit
            if self.recorder_queue:
                 self.recorder_queue.put("STOP") # Also send STOP command via queue
            self.recorder_process.join(timeout=5.0) # Wait for graceful exit
            if self.recorder_process.is_alive():
                print("[Worker] Recorder process did not exit gracefully, terminating.")
                self.recorder_process.terminate() # Force terminate if needed
            # print("[Worker Debug] Recorder process stop sequence complete.") # DEBUG
            print("[Worker] Recorder process stopped.")
        self.recorder_process = None
        self.recorder_queue = None
        self.recorder_stop_event = None

    def stop(self):
        """Requests the worker loop to stop."""
        # This method is called via QueuedConnection from app.aboutToQuit or directly
        # if the worker thread is stopping itself.
        print("[Worker] stop() method executing in worker thread.")
        self._running = False # Signal the run loop to stop processing on its next check
        # Since we are (likely) in the worker thread, we can safely stop the timer directly here.
        if self.run_timer.isActive():
            # print("[Worker Debug] Stopping timer directly within stop().")
            self.run_timer.stop()

        # Removed QApplication.processEvents(), _cleanup_resources(), and finished.emit()
        # Rely on the run() method's check for self._running == False to trigger
        # the final cleanup and finished signal emission.
        # print("[Worker Debug] stop() method finished setting flag and stopping timer.")

    def set_tracking_active(self, active: bool):
        """Slot to start or stop the tracking state."""
        print(f"[Worker Slot] set_tracking_active called with: {active}")

        # Prevent state changes if components aren't ready
        if not self._components_initialized and active:
            print("[Worker] Cannot start tracking: Components not yet initialized.")
            self.processing_error.emit("Components still initializing, please wait.")
            return

        # --- Add re-entry guard for starting tracking ---
        if active and self._is_transitioning_tracking:
            print("[Worker] Warning: Ignoring set_tracking_active(True) call while already transitioning.")
            return
        # --- End guard ---

        if active and self.state == WorkerState.PREVIEWING:
            # Check if a valid ROI was found during preview
            if not self.latest_preview_roi:
                print("[Worker] Cannot start tracking: No valid ROI found in preview.")
                self.processing_error.emit("Cannot start tracking: Position yourself for ROI detection.")
                return

            # --- Set transitioning flag ---
            self._is_transitioning_tracking = True
            try:
                print(f"[Worker] Locking ROI {self.latest_preview_roi} and starting tracking.")
                self.locked_roi = copy.deepcopy(self.latest_preview_roi)
                self.state = WorkerState.TRACKING # State changes here
                # Ensure the pipeline manager uses the locked ROI and reset signal processor
                if self.pipeline_manager:
                    if hasattr(self.pipeline_manager, 'set_tracking_roi'):
                        self.pipeline_manager.set_tracking_roi(self.locked_roi)
                    else:
                         print("[Worker] Warning: PipelineManager lacks set_tracking_roi method.")

                    # Reset signal processor when tracking starts to clear old buffers
                    try:
                        print("[Worker] Resetting SignalProcessor for new tracking session.")
                        # Re-initialize SignalProcessor using the current config
                        self.pipeline_manager.signal_processor = SignalProcessor(
                            config=self.current_config.get('signal_processor', {}),
                            sampling_rate=self.sampling_rate
                        )
                    except Exception as e_sp_reset:
                        print(f"[Worker] Error resetting SignalProcessor on tracking start: {e_sp_reset}")
                        self.processing_error.emit("Error resetting signal processor.")
                        # If SP reset fails, revert state
                        self.state = WorkerState.PREVIEWING
                        self.locked_roi = []
                    # --- MODIFIED: Add 3-second delay for recorder start ---
                    if self.recording_enabled:
                        print("[Worker] Scheduling recorder start in 3 seconds...")
                        QTimer.singleShot(3000, self._delayed_start_recorder) # Delay start
                else:
                    print("[Worker] Error: PipelineManager not available to set ROI/reset.")
                    self.processing_error.emit("PipelineManager error on tracking start.")
                    # Revert state if manager is missing
                    self.state = WorkerState.PREVIEWING
                    self.locked_roi = []
            finally:
                # --- Clear transitioning flag ---
                self._is_transitioning_tracking = False

        elif not active and self.state == WorkerState.TRACKING:
            print("[Worker] Stopping tracking and returning to preview mode.")
            # --- RE-ADD RECORDER STOP HERE ---
            if self.recorder_process and self.recorder_process.is_alive():
                self._stop_recorder_process() # Stop and save recording now
            self.state = WorkerState.PREVIEWING
            self.locked_roi = [] # Clear the locked ROI
        else:
            # Log if no state change occurs
            print(f"[Worker] set_tracking_active called in state {self.state} with active={active}. No change.")

    # --- ADDED: Method to handle delayed recorder start ---
    def _delayed_start_recorder(self):
        """Starts the recorder process if still in TRACKING state."""
        # print(f"[Worker Debug] _delayed_start_recorder called. State: {self.state}, Running: {self._running}, RecEnabled: {self.recording_enabled}")
        if self._running and self.state == WorkerState.TRACKING and self.recording_enabled:
            # print("[Worker Debug] Conditions met for delayed recorder start.") # DEBUG
            print("[Worker] 3-second delay complete. Starting recorder now.")
            self._start_recorder_process()

    def reload_profile(self, new_config_path):
        """Loads a new profile, stops processing, and re-initializes."""
        print(f"[Worker Slot] reload_profile called with path: {new_config_path}")

        # 1. Stop current processing
        self._running = False
        if self.run_timer.isActive():
            self.run_timer.stop()
        print("[Worker] Processing loop stopped for profile reload.")

        # 2. Load new configuration and compare video settings
        old_video_config = copy.deepcopy(self.current_config.get("video_input", {}))
        new_config = self._load_config_from_path(new_config_path)
        if not new_config: # If loading failed
            self.processing_error.emit(f"Failed to load profile: {new_config_path}")
            # Attempt to restart with old config if possible, or signal critical error
            if self._components_initialized and self.video_input: # Check if we can restart
                self._running = True
                self.run_timer.start(1)
            return

        new_video_config = new_config.get("video_input", {})
        video_settings_changed = (old_video_config != new_video_config)

        # 3. Update internal config path and current_config
        self.config_path = new_config_path # Store the new path
        self.current_config = new_config   # Adopt the new config
        self.recording_enabled = self.current_config.get("enable_raw_data_recording", False) # Update recording flag

        # Re-initialize OSC manager with new config
        self._initialize_osc_manager()

        if video_settings_changed:
            print("[Worker] Video settings changed. Full re-initialization required.")
            self._cleanup_resources() # Releases everything, including old VideoInput
            # Reset state for full setup
            self.state = WorkerState.INITIALIZING
            self._components_initialized = False
            self._init_step = 0
            self.locked_roi = []
            self.latest_preview_roi = []
            self.latest_landmarks = None
            self.setup() # This will re-init VideoInput and then other components
        else:
            print("[Worker] Video settings unchanged. Reconfiguring processing components.")
            # VideoInput is preserved. Re-initialize other components.
            # Stop recorder if it was running, as other params might affect data
            self._stop_recorder_process()

            try:
                # Re-initialize PoseDetector
                if not self._initialize_pose_detector(): # Call refactored
                    raise RuntimeError("PoseDetector re-init failed.")

                # Re-initialize CoarseRoiCalculator
                if not self._initialize_roi_calculator(): # Call refactored
                    raise RuntimeError("CoarseRoiCalculator re-init failed.")

                # Re-initialize PipelineManager
                if not self._initialize_pipeline_manager(): # Call refactored
                    raise RuntimeError("PipelineManager re-init failed.")

                # If all re-initializations are successful
                self.state = WorkerState.PREVIEWING # Default to preview after profile change
                self.locked_roi = []
                self._components_initialized = True # Mark as ready
                self._running = True
                self._loop_start_time = time.perf_counter()
                # Emit the (potentially new) settings to the UI after successful re-init
                current_ui_settings = self._extract_relevant_settings(self.current_config)
                self.current_settings_signal.emit(current_ui_settings)
                self.run_timer.start(1)
                print("[Worker] Processing components reconfigured. Resuming in preview mode.")

            except Exception as e:
                error_msg = f"Error reconfiguring components after profile reload: {e}"
                print(f"[Worker] {error_msg}")
                traceback.print_exc()
                self.processing_error.emit(error_msg)
                self._initialization_failed() # Go to a safe, non-running state

    # --- SLOT TO UPDATE OVERLAY STATES ---
    @pyqtSlot(bool, bool, bool) # Decorate slot for type safety
    def update_overlay_settings(self, show_pose, show_roi, show_features):
        """Slot to update the overlay visibility flags."""
        current_thread_id = threading.get_ident() # Get current thread ID for debugging
        self.show_pose = show_pose
        self.show_roi = show_roi
        self.show_features = show_features

    # --- NEW SLOT TO APPLY SETTINGS ---
    @pyqtSlot(dict)
    def apply_new_settings(self, settings_dict):
        """Applies new settings received from the UI."""
        print(f"[Worker Slot] apply_new_settings called with: {settings_dict}")
        if self.state == WorkerState.TRACKING:
            print("[Worker] Stopping tracking before applying new settings.")
            self.set_tracking_active(False)
            # Give a moment for the state change to potentially take effect
            time.sleep(0.1)

        # --- Merge Settings ---
        print("[Worker] Merging new settings into current config...")
        try:
            # Deep merge is safer for nested dictionaries like OPTICAL_FLOW_PARAMS
            def merge_dicts(base, update):
                 for key, value in update.items():
                     if isinstance(value, dict) and key in base and isinstance(base.get(key), dict):
                         # Recursively merge nested dictionaries
                         merge_dicts(base[key], value)
                     else:
                         # Overwrite or add new key/value
                         base[key] = value
                 return base

            # Use deepcopy to avoid modifying the original config if merge fails midway
            merged_config = merge_dicts(copy.deepcopy(self.current_config), settings_dict)
            self.current_config = merged_config # Update the worker's config
            print("[Worker] Settings merged successfully.")
            # print("Updated Config:", json.dumps(self.current_config, indent=2)) # Optional: Print merged config for verification
        except Exception as e_merge:
             print(f"[Worker] Error merging settings: {e_merge}")
             self.processing_error.emit("Error applying settings (merge failed).")
             return # Stop if merging fails

        # --- Re-initialize Components ---
        # Only re-initialize if components were initialized in the first place
        if not self._components_initialized or not self.pipeline_manager:
             print("[Worker] Cannot re-initialize components: Initial setup not complete or PipelineManager missing.")
             self.processing_error.emit("Cannot apply settings: Components not ready.")
             return

        print("[Worker] Re-initializing components with new settings...")
        reinit_success = True
        try:
            # Re-initialize components within the PipelineManager instance
            # Assumes PipelineManager allows direct replacement or has re-init methods
            print("  Re-initializing FeatureTracker...")
            # Pass the relevant sub-dictionary from the *updated* config
            ft_config = self.current_config.get('feature_tracker', {})
            self.pipeline_manager.feature_tracker = FeatureTracker(config=ft_config)

            print("  Re-initializing SignalGenerator...")
            sg_config = self.current_config.get('signal_generator', {})
            self.pipeline_manager.signal_generator = SignalGenerator(config=sg_config)

            print("  Re-initializing SignalProcessor...")
            sp_config = self.current_config.get('signal_processor', {})
            self.pipeline_manager.signal_processor = SignalProcessor(
                config=sp_config,
                sampling_rate=self.sampling_rate # Pass the current sampling rate
            )
            print("[Worker] Components re-initialized successfully.")

        except Exception as e_reinit:
            print(f"[Worker] Error re-initializing components after settings change: {e_reinit}")
            traceback.print_exc()
            self.processing_error.emit("Error applying settings (re-init failed).")
            reinit_success = False
            # Consider trying to revert to old config or stopping worker? For now, just report error.

        if reinit_success:
            # Inform UI that settings were applied and tracking needs restart
            self.processing_error.emit("Settings applied. Please restart tracking if needed.")


# --- Main Application Setup ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # --- Required for multiprocessing on some platforms (e.g., Windows) ---
    # Should be called early, especially if freezing the app later
    # mp_proc.freeze_support() # Uncomment if needed, typically for frozen executables
    # Set start method if needed (e.g., 'spawn' might be more stable on macOS/Windows)
    # mp_proc.set_start_method('spawn', force=True) # Uncomment/adjust if experiencing issues

    # --- Determine Config Path ---
    config_file = os.path.join(PROFILES_DIR, DEFAULT_PROFILE)
    if not os.path.exists(PROFILES_DIR):
        print(f"Warning: Profiles directory does not exist: '{PROFILES_DIR}'")
        # Optionally create it: os.makedirs(PROFILES_DIR, exist_ok=True)
    if not os.path.exists(config_file):
        print(f"Warning: Default profile '{DEFAULT_PROFILE}' not found in '{PROFILES_DIR}'. Using empty config.")
        config_file = "" # Use empty path to signal worker to use defaults

    # --- Create UI and Worker ---
    main_window = MainWindow(config_file=config_file, profiles_dir=PROFILES_DIR)
    visualizer_window = SignalVisualizer() # <<< Create visualizer instance
    worker = PipelineWorker(config_path=config_file) # Pass initial config path
    worker_thread = QThread()
    worker.moveToThread(worker_thread)

    # --- Connect Signals and Slots ---
    # Worker -> UI
    worker.new_frame_ready.connect(main_window.update_webcam_feed)
    # worker.new_plot_data.connect(main_window.update_plot) # OBSOLETE: update_plot removed from MainWindow
    worker.new_status.connect(main_window.update_status_labels)
    worker.new_filtered_signal_value.connect(visualizer_window.update_signal) # <<< Connect visualizer
    worker.processing_error.connect(main_window.show_error_message)
    worker.setup_finished.connect(main_window.handle_worker_setup_finished)
    worker.component_initialized.connect(main_window.handle_component_initialized)
    worker.current_settings_signal.connect(main_window.populate_settings_widgets) # New connection
    worker.profile_saved_signal.connect(main_window.handle_profile_saved)
    # --- Ensure this connection is robust ---
    worker.new_pipeline_results.connect(main_window.handle_worker_output, Qt.ConnectionType.QueuedConnection)
    print("[Main Setup] Connected worker.new_pipeline_results to main_window.handle_worker_output")

    # UI -> Worker
    # --- MODIFIED: Connect directly to wrapper slots ---
    main_window.start_tracking_signal.connect(worker.set_tracking_active_true)
    main_window.stop_tracking_signal.connect(worker.set_tracking_active_false)
    # --- END MODIFICATION ---
    main_window.load_profile_signal.connect(worker.reload_profile, Qt.ConnectionType.QueuedConnection)
    # --- MODIFIED: Explicitly queue save and apply settings ---
    main_window.save_profile_signal.connect(worker.handle_save_profile, Qt.ConnectionType.QueuedConnection)
    main_window.overlay_settings_changed.connect(worker.update_overlay_settings) # AutoConnection should be fine
    main_window.apply_settings_signal.connect(worker.apply_new_settings, Qt.ConnectionType.QueuedConnection)

    # Connect Reset Signal (handle potential missing manager)
    def safe_reset_tracker():
        if worker.pipeline_manager and hasattr(worker.pipeline_manager, 'reset_tracker'):
             worker.pipeline_manager.reset_tracker()
        else:
             print("[Main] Error: Cannot reset tracker, PipelineManager not ready or lacks method.")
             main_window.show_error_message("Reset failed: Pipeline not ready.")
    main_window.reset_tracking_signal.connect(safe_reset_tracker) # Direct connection likely okay


    # --- Thread Management ---
    # --- MODIFIED: Explicitly queue the setup connection ---
    worker_thread.started.connect(worker.setup, Qt.ConnectionType.QueuedConnection)
    worker.finished.connect(worker_thread.quit) # Quit thread when worker finishes
    worker.finished.connect(worker.deleteLater) # Schedule worker deletion
    worker_thread.finished.connect(worker_thread.deleteLater) # Schedule thread deletion

    # Graceful shutdown
    # --- MODIFIED: Explicitly queue the stop connection ---
    app.aboutToQuit.connect(worker.stop, Qt.ConnectionType.QueuedConnection)

    # --- Start Application ---
    main_window.show()
    # visualizer_window.show() # <<< Show the visualizer window
    # Start the worker thread's event loop
    # Use QTimer for a slightly delayed start to ensure the main event loop is running
    QTimer.singleShot(100, worker_thread.start)

    sys.exit(app.exec())
