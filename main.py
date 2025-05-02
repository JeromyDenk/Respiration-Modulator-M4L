# main.py
# Main entry point for the Respiration Modulator application.
# Sets up the PyQt6 application, creates the UI window, and manages the backend worker thread.
# MODIFIED: Added settings application logic (apply_new_settings slot, signals).
# FIXED: Added missing pyqtSlot import.

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
    Handles dynamic settings updates.
    """
    # --- Signals ---
    new_frame_ready = pyqtSignal(np.ndarray)
    new_plot_data = pyqtSignal(list)
    new_status = pyqtSignal(float, bool, int)
    processing_error = pyqtSignal(str)
    finished = pyqtSignal()
    setup_finished = pyqtSignal(bool, str)
    component_initialized = pyqtSignal(str, bool, str)
    current_settings_signal = pyqtSignal(dict) # Emit initial/current settings to UI

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

    def _load_config(self):
        """Loads the configuration from the specified file path."""
        config = {}
        try:
            if self.config_path and os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                print(f"[Worker] Loaded config from: {self.config_path}") # Verbose for tracking
                self.current_config = config
            else:
                print(f"[Worker] Warning: Config file '{self.config_path}' not found. Using empty config.")
                self.current_config = {}
                # Optionally create a default config here if needed
        except Exception as e:
            print(f"[Worker] Error loading config '{self.config_path}': {e}")
            self.current_config = {} # Fallback to empty on error
            raise # Re-raise to signal the failure upwards
        return self.current_config

    def setup(self):
        """Initial setup: Load config and initialize video source."""
        print("[Worker] Running initial setup (Video & Config)...")
        error_msg = ""
        setup_success = False
        try:
            # Load configuration first
            config = self._load_config()
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
            self._initialize_roi_calculator()
        elif self._init_step == 2:
            self._initialize_pipeline_manager()
        # Add more steps here if needed

    def _initialize_pose_detector(self):
        """Initializes the PoseDetector component."""
        if not self._running: return
        print("[Worker] Initializing PoseDetector..."); success = False; msg = ""
        try:
            pose_config = self.current_config.get("pose_detector", {});
            # Ensure any previous instance is closed
            if self.pose_detector: self.pose_detector.close()
            self.pose_detector = PoseDetector(config=pose_config)
            if not self.pose_detector.initialized:
                raise RuntimeError("PoseDetector internal initialization failed.")
            print("[Worker] PoseDetector Initialized."); success = True; msg = "PoseDetector OK."
            self._init_step += 1
            # Schedule the next step if still running
            if self._running: QTimer.singleShot(10, self._run_next_init_step)
        except Exception as e:
            msg = f"Error initializing PoseDetector: {e}"
            print(f"[Worker] {msg}"); traceback.print_exc()
            self.pose_detector = None; success = False
        finally:
            # Emit signal regardless of success
            self.component_initialized.emit("PoseDetector", success, msg);
            # If failed, stop the initialization sequence
            if not success: self._initialization_failed()

    def _initialize_roi_calculator(self):
        """Initializes the CoarseRoiCalculator component."""
        if not self._running: return
        print("[Worker] Initializing CoarseRoiCalculator..."); success = False; msg = ""
        try:
            coarse_roi_config = self.current_config.get("coarse_roi_calculator", {})
            # No close method needed for RoiCalculator currently
            self.coarse_roi_calculator = CoarseRoiCalculator(config=coarse_roi_config)
            print("[Worker] CoarseRoiCalculator Initialized."); success = True; msg = "RoiCalculator OK."
            self._init_step += 1
            # Schedule the next step if still running
            if self._running: QTimer.singleShot(10, self._run_next_init_step)
        except Exception as e:
            msg = f"Error initializing CoarseRoiCalculator: {e}"
            print(f"[Worker] {msg}"); traceback.print_exc()
            self.coarse_roi_calculator = None; success = False
        finally:
            # Emit signal regardless of success
            self.component_initialized.emit("RoiCalculator", success, msg);
            # If failed, stop the initialization sequence
            if not success: self._initialization_failed()

    def _initialize_pipeline_manager(self):
        """Initializes the PipelineManager and emits initial settings."""
        if not self._running: return
        print("[Worker] Initializing PipelineManager..."); success = False; msg = ""
        try:
            # Close existing manager if any
            if self.pipeline_manager: self.pipeline_manager.close()
            # Initialize with current config and sampling rate
            self.pipeline_manager = PipelineManager(config=self.current_config, sampling_rate=self.sampling_rate)
            print("[Worker] PipelineManager Initialized."); success = True; msg = "PipelineManager OK."
            self._init_step += 1
            self._components_initialized = True
            self.state = WorkerState.PREVIEWING # Ready for preview
            print("[Worker] All components initialized successfully.")

            # --- Emit Initial Settings ---
            # Extract relevant settings from current_config to send to UI
            initial_settings = self._extract_relevant_settings(self.current_config)
            print(f"[Worker] Emitting initial settings: {initial_settings}")
            self.current_settings_signal.emit(initial_settings)
            # --- End Emit Initial Settings ---

            # Schedule the main run loop if still running
            if self._running:
                # Use QTimer.singleShot with 0ms delay to schedule run() on the event loop
                QTimer.singleShot(0, self.run)

        except Exception as e:
            msg = f"Error initializing PipelineManager: {e}"
            print(f"[Worker] {msg}"); traceback.print_exc()
            self.pipeline_manager = None
            success = False
        finally:
            # Emit signal regardless of success
            self.component_initialized.emit("PipelineManager", success, msg)
            # If failed, stop the initialization sequence
            if not success:
                self._initialization_failed()

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


    def run(self):
        """Main processing loop managing PREVIEWING and TRACKING states."""
        if not self._running:
            print("[Worker Run] Stop called before run loop started.")
            self.finished.emit()
            return
        # Check if components are initialized before starting the loop
        if not self._components_initialized:
             print("[Worker Run] Components not initialized. Cannot start run loop.")
             self.processing_error.emit("Initialization failed. Cannot run.")
             self._running = False
             self.finished.emit()
             return
        # Ensure state is PREVIEWING after successful initialization
        if self.state != WorkerState.PREVIEWING:
            print(f"[Worker Run] Warning: Starting run loop in unexpected state {self.state}. Resetting to PREVIEWING.")
            self.state = WorkerState.PREVIEWING

        print(f"[Worker Run Start] Starting processing loop in state: {self.state}")
        loop_count = 0
        while self._running:
            start_loop_time = time.perf_counter()
            loop_count += 1
            try:
                success, frame = self.video_input.get_frame()
                if not success or frame is None:
                    print("[Worker] End of video source or cannot read frame.")
                    self._running = False
                    break

                processed_frame = frame.copy() # Frame for display with overlays
                results = None
                current_tracked_points = None # Store points from this frame's tracking result

                # --- State-Dependent Processing & Drawing ---
                if self.state == WorkerState.PREVIEWING:
                    # Preview logic: Run pose detection and ROI calculation
                    if not self.pose_detector or not self.coarse_roi_calculator:
                        self.processing_error.emit("Preview components not ready.")
                        time.sleep(0.1) # Avoid busy-looping
                        continue

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False
                    self.latest_landmarks = self.pose_detector.process_frame(image_rgb)
                    image_rgb.flags.writeable = True

                    if self.latest_landmarks:
                        frame_h, frame_w = frame.shape[:2]
                        self.latest_preview_roi = self.coarse_roi_calculator.calculate_coarse_roi(self.latest_landmarks, (frame_h, frame_w))
                    else:
                        self.latest_preview_roi = []

                    # Draw overlays for preview
                    if self.show_pose and self.latest_landmarks:
                        mp_drawing.draw_landmarks(processed_frame, self.latest_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    if self.show_roi and self.latest_preview_roi:
                        for (x, y, w, h) in self.latest_preview_roi:
                            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 255), 2) # Cyan ROI

                    # Emit preview results
                    self.new_frame_ready.emit(processed_frame)
                    self.new_plot_data.emit([]) # No plot data in preview
                    self.new_status.emit(0.0, False, SignalProcessor.PHASE_UNKNOWN) # No status in preview

                elif self.state == WorkerState.TRACKING:
                    # Tracking logic: Run the full pipeline via PipelineManager
                    if not self.locked_roi:
                        print("[Worker] Error: Tracking active but no ROI locked.")
                        self.processing_error.emit("Tracking started without ROI.")
                        self.set_tracking_active(False) # Revert to preview
                        continue
                    if not self.pipeline_manager:
                        print("[Worker] Error: PipelineManager not initialized for tracking.")
                        self.processing_error.emit("PipelineManager Error.")
                        self.set_tracking_active(False) # Revert to preview
                        continue

                    # Set the locked ROI in the manager (might be redundant if set at transition)
                    # Ensure PipelineManager has a set_tracking_roi method
                    if hasattr(self.pipeline_manager, 'set_tracking_roi'):
                         self.pipeline_manager.set_tracking_roi(self.locked_roi)
                    else:
                         print("[Worker] Warning: PipelineManager does not have set_tracking_roi method.")


                    results = self.pipeline_manager.process_frame(frame)

                    # --- Get tracked points from results ---
                    if results:
                        current_tracked_points = results.get('tracked_points') # Adjust key if needed

                    # --- Draw Tracking Overlays ---
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
                                     # Log if shape is unexpected after conversion
                                     if loop_count % 60 == 0: # Log occasionally
                                          print(f"[Worker Debug Frame {loop_count}] current_tracked_points has unexpected shape/ndim after float conversion: {points_to_draw_float.shape}")
                            except Exception as draw_err:
                                print(f"!!! ERROR during feature drawing loop: {draw_err}")
                                traceback.print_exc() # Print full traceback for drawing errors

                    # --- Emit Tracking Results ---
                    self.new_frame_ready.emit(processed_frame)
                    if results:
                        plot_data = results.get('filtered_signal_history', [])
                        bpm = results.get('bpm', 0.0)
                        valid = results.get('bpm_valid', False)
                        phase = results.get('phase', SignalProcessor.PHASE_UNKNOWN)
                        self.new_plot_data.emit(plot_data)
                        self.new_status.emit(bpm, valid, phase)
                    else: # Handle case where pipeline_manager.process_frame returned None
                        self.new_plot_data.emit([])
                        self.new_status.emit(0.0, False, SignalProcessor.PHASE_UNKNOWN)

                # --- Loop Delay ---
                loop_duration = time.perf_counter() - start_loop_time
                target_interval = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.033
                sleep_time = max(0.001, target_interval - loop_duration)
                time.sleep(sleep_time) # Prevent excessive CPU usage

            except Exception as e:
                error_msg = f"Error in worker loop (State: {self.state}): {e}"
                print(error_msg); traceback.print_exc()
                self.processing_error.emit(error_msg)
                time.sleep(0.1) # Pause briefly after an error

        # --- Post-Loop Cleanup ---
        print("[Worker] Stopping processing loop...")
        if hasattr(self.video_input, 'release'):
            self.video_input.release()
        if self.pipeline_manager:
            self.pipeline_manager.close()
        if self.pose_detector:
            self.pose_detector.close()
        print("[Worker] Resources released.")
        self.finished.emit() # Signal that the worker has finished


    def stop(self):
        """Requests the worker loop to stop."""
        print("[Worker] Stop requested.")
        self._running = False # Set the flag to false

    def set_tracking_active(self, active: bool):
        """Slot to start or stop the tracking state."""
        print(f"[Worker Slot] set_tracking_active called with: {active}")
        # Prevent state changes if components aren't ready
        if not self._components_initialized and active:
            print("[Worker] Cannot start tracking: Components not yet initialized.")
            self.processing_error.emit("Components still initializing, please wait.")
            return

        if active and self.state == WorkerState.PREVIEWING:
            # Check if a valid ROI was found during preview
            if not self.latest_preview_roi:
                print("[Worker] Cannot start tracking: No valid ROI found in preview.")
                self.processing_error.emit("Cannot start tracking: Position yourself for ROI detection.")
                return

            print(f"[Worker] Locking ROI {self.latest_preview_roi} and starting tracking.")
            self.locked_roi = copy.deepcopy(self.latest_preview_roi)
            self.state = WorkerState.TRACKING
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
            else:
                print("[Worker] Error: PipelineManager not available to set ROI/reset.")
                self.processing_error.emit("PipelineManager error on tracking start.")
                # Revert state if manager is missing
                self.state = WorkerState.PREVIEWING
                self.locked_roi = []

        elif not active and self.state == WorkerState.TRACKING:
            print("[Worker] Stopping tracking and returning to preview mode.")
            self.state = WorkerState.PREVIEWING
            self.locked_roi = [] # Clear the locked ROI
        else:
            # Log if no state change occurs
            print(f"[Worker] set_tracking_active called in state {self.state} with active={active}. No change.")

    def reload_profile(self, new_config_path):
        """Loads a new profile, stops processing, and re-initializes."""
        print(f"[Worker Slot] reload_profile called with path: {new_config_path}")
        # Stop current processing and reset state
        self.stop() # Request the loop to stop
        # Wait briefly for the loop to potentially finish its current iteration
        # Note: A more robust approach might involve QThread.wait() or signals
        time.sleep(0.2)

        # Reset internal state variables
        self.state = WorkerState.INITIALIZING
        self._components_initialized = False
        self._init_step = 0
        self.locked_roi = []
        self.latest_preview_roi = []
        self.latest_landmarks = None

        # Load new config and re-run setup (setup will set _running=True if successful)
        self.config_path = new_config_path
        self.setup() # This will load config and start component init if successful


    # --- SLOT TO UPDATE OVERLAY STATES ---
    @pyqtSlot(bool, bool, bool) # Decorate slot for type safety
    def update_overlay_settings(self, show_pose, show_roi, show_features):
        """Slot to update the overlay visibility flags."""
        current_thread_id = threading.get_ident() # Get current thread ID for debugging
        print("*"*20)
        print(f"[Worker Slot EXECUTION CONFIRMED - Thread: {current_thread_id}] Received: Pose={show_pose}, ROI={show_roi}, Features={show_features}")
        # --- *** ---
        self.show_pose = show_pose
        self.show_roi = show_roi
        self.show_features = show_features
        print(f"[Worker Slot Post-Set] self.show_features is now: {self.show_features}")
        print("*"*20)

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
    worker = PipelineWorker(config_path=config_file) # Pass initial config path
    worker_thread = QThread()
    worker.moveToThread(worker_thread)

    # --- Connect Signals and Slots ---
    # Worker -> UI
    worker.new_frame_ready.connect(main_window.update_webcam_feed)
    worker.new_plot_data.connect(main_window.update_plot)
    worker.new_status.connect(main_window.update_status_labels)
    worker.processing_error.connect(main_window.show_error_message)
    worker.setup_finished.connect(main_window.handle_worker_setup_finished)
    worker.component_initialized.connect(main_window.handle_component_initialized)
    worker.current_settings_signal.connect(main_window.populate_settings_widgets) # New connection

    # UI -> Worker
    main_window.start_tracking_signal.connect(lambda: worker.set_tracking_active(True))
    main_window.stop_tracking_signal.connect(lambda: worker.set_tracking_active(False))
    main_window.load_profile_signal.connect(worker.reload_profile)
    main_window.overlay_settings_changed.connect(worker.update_overlay_settings)
    main_window.apply_settings_signal.connect(worker.apply_new_settings) # New connection

    # Connect Reset Signal (handle potential missing manager)
    def safe_reset_tracker():
        if worker.pipeline_manager and hasattr(worker.pipeline_manager, 'reset_tracker'):
             worker.pipeline_manager.reset_tracker()
        else:
             print("[Main] Error: Cannot reset tracker, PipelineManager not ready or lacks method.")
             main_window.show_error_message("Reset failed: Pipeline not ready.")
    main_window.reset_tracking_signal.connect(safe_reset_tracker)


    # --- Thread Management ---
    worker_thread.started.connect(worker.setup) # Start setup when thread starts
    worker.finished.connect(worker_thread.quit) # Quit thread when worker finishes
    worker.finished.connect(worker.deleteLater) # Schedule worker deletion
    worker_thread.finished.connect(worker_thread.deleteLater) # Schedule thread deletion

    # Graceful shutdown
    app.aboutToQuit.connect(worker.stop) # Tell worker to stop on app quit
    # Removed worker_thread.finished.connect(app.quit) to prevent potential race conditions on exit

    # --- Start Application ---
    main_window.show()
    # Start the worker thread's event loop
    # Use QTimer for a slightly delayed start to ensure the main event loop is running
    QTimer.singleShot(100, worker_thread.start)

    sys.exit(app.exec())
