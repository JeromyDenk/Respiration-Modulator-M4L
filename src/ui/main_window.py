# src/ui/main_window.py
# Defines the main application window using PyQt6.
# Updated to handle two-stage worker initialization signals.

import sys
import os
import numpy as np
# --- PyQt6 Imports ---
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
        QGridLayout, QGroupBox, QComboBox, QCheckBox, QFileDialog, QMessageBox,
        QSizePolicy, QStatusBar
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QSize
    from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
except ImportError:
    print("Fatal Error: PyQt6 not found. Please install it (e.g., pip install PyQt6)")
    sys.exit(1)

import pyqtgraph as pg # For plotting

# --- Constants ---
PLOT_BUFFER_SIZE = 500 # Number of points to display on the plot

# --- Import needed for status label update ---
try:
    # Need to go up two levels from src/ui to reach src
    # This might be fragile depending on how the script is run.
    # Consider adding src to PYTHONPATH environment variable instead.
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from signal_processor import SignalProcessor
except ImportError:
    print("Warning: Could not import SignalProcessor for phase constants in UI.")
    # Define dummy constants if import fails
    class SignalProcessor: PHASE_INHALE = 1; PHASE_EXHALE = -1; PHASE_UNKNOWN = 0


class MainWindow(QMainWindow):
    """
    Main application window class. Handles UI layout, display updates,
    and emits signals based on user interaction.
    """
    # Signals emitted by the UI for the backend worker
    start_tracking_signal = pyqtSignal()
    stop_tracking_signal = pyqtSignal()
    load_profile_signal = pyqtSignal(str) # Emits the selected profile path
    save_profile_signal = pyqtSignal(str) # Emits the path to save to
    # Signal to update overlay status in worker (bool: show_pose, bool: show_roi, bool: show_features)
    overlay_settings_changed = pyqtSignal(bool, bool, bool)


    def __init__(self, config_file="", profiles_dir="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Respiration Modulator M4L")
        self.setGeometry(100, 100, 1200, 800) # Initial size

        self.config_file = config_file # Store initial config path
        self.profiles_dir = profiles_dir # Store profiles directory path
        self.plot_data_buffer = np.zeros(PLOT_BUFFER_SIZE)
        self.tracking_active = False # UI's understanding of tracking state

        self._init_ui()
        # Set initial state BEFORE showing
        self.webcam_label.setText("Initializing Video...") # Updated initial text
        self.statusBar.showMessage("Initializing video source...")
        self._update_ui_state() # Initial state (controls likely disabled)

    def _init_ui(self):
        """Creates and arranges all UI widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget) # Main vertical layout

        # --- Top Section: Webcam and Plot ---
        top_layout = QVBoxLayout()
        main_layout.addLayout(top_layout, 7) # Allocate 70% height

        # Webcam Feed Label
        self.webcam_label = QLabel() # Initial text set in __init__
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Set size policy to expand and ignore aspect ratio initially, scaling will handle it
        self.webcam_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        palette = self.webcam_label.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('black'))
        self.webcam_label.setAutoFillBackground(True)
        self.webcam_label.setPalette(palette)
        top_layout.addWidget(self.webcam_label, 5) # Allocate more space to webcam

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w') # White background
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Samples')
        self.plot_widget.setTitle('Filtered Signal', size='10pt')
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2)) # Blue line
        top_layout.addWidget(self.plot_widget, 2) # Allocate less space to plot

        # --- Bottom Section: Controls ---
        bottom_controls_widget = QWidget()
        bottom_controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum) # Limit height
        main_layout.addWidget(bottom_controls_widget, 3) # Allocate 30% height

        bottom_layout = QHBoxLayout(bottom_controls_widget) # Horizontal layout for controls

        # Left Side: Status & Tracking Button
        left_panel_layout = QVBoxLayout()
        bottom_layout.addLayout(left_panel_layout, 1) # Allocate 1 part width
        status_group = QGroupBox("Status")
        status_layout = QGridLayout(status_group)
        self.bpm_label = QLabel("BPM: ---")
        self.phase_label = QLabel("Phase: ---")
        status_layout.addWidget(self.bpm_label, 0, 0)
        status_layout.addWidget(self.phase_label, 1, 0)
        left_panel_layout.addWidget(status_group)
        self.track_button = QPushButton("Start Tracking")
        self.track_button.setCheckable(True) # Make it a toggle button
        self.track_button.setFixedHeight(40) # Make button taller
        self.track_button.toggled.connect(self._handle_track_button_toggle)
        self.track_button.setEnabled(False) # Initially disabled until setup completes
        left_panel_layout.addWidget(self.track_button)
        left_panel_layout.addStretch() # Push elements up

        # Middle: Overlays & Settings (Placeholders)
        middle_panel_layout = QVBoxLayout()
        bottom_layout.addLayout(middle_panel_layout, 2) # Allocate 2 parts width
        overlays_group = QGroupBox("Overlays")
        overlays_layout = QVBoxLayout(overlays_group)
        self.pose_overlay_check = QCheckBox("Show Pose")
        self.roi_overlay_check = QCheckBox("Show ROI")
        self.features_overlay_check = QCheckBox("Show Features")
        # Connect checkboxes to a handler function that emits a signal
        self.pose_overlay_check.stateChanged.connect(self._emit_overlay_settings)
        self.roi_overlay_check.stateChanged.connect(self._emit_overlay_settings)
        self.features_overlay_check.stateChanged.connect(self._emit_overlay_settings)
        overlays_layout.addWidget(self.pose_overlay_check)
        overlays_layout.addWidget(self.roi_overlay_check)
        overlays_layout.addWidget(self.features_overlay_check)
        middle_panel_layout.addWidget(overlays_group)
        settings_group = QGroupBox("Settings (Display Only - From Profile)")
        settings_layout = QVBoxLayout(settings_group)
        # Use different labels for easier updating later
        self.filter_settings_label = QLabel("Filter: ...")
        self.peak_settings_label = QLabel("Peak Det: ...")
        self.lk_settings_label = QLabel("LK Params: ...")
        settings_layout.addWidget(self.filter_settings_label)
        settings_layout.addWidget(self.peak_settings_label)
        settings_layout.addWidget(self.lk_settings_label)
        middle_panel_layout.addWidget(settings_group)
        middle_panel_layout.addStretch()

        # Right Side: Profile Management
        right_panel_layout = QVBoxLayout()
        bottom_layout.addLayout(right_panel_layout, 1) # Allocate 1 part width
        profile_group = QGroupBox("Profile Management")
        profile_layout = QVBoxLayout(profile_group)
        self.profile_combo = QComboBox()
        self._populate_profiles() # Load available profiles using self.profiles_dir
        self.profile_combo.currentIndexChanged.connect(self._profile_selected)
        profile_layout.addWidget(QLabel("Select Profile:"))
        profile_layout.addWidget(self.profile_combo)
        self.load_button = QPushButton("Reload Selected Profile") # Assign to self
        self.save_button = QPushButton("Save Current Settings As...") # Assign to self
        self.load_button.clicked.connect(self._load_profile)
        self.save_button.clicked.connect(self._save_profile_as)
        self.load_button.setEnabled(False) # Initially disabled
        self.save_button.setEnabled(False) # Initially disabled
        profile_layout.addWidget(self.load_button)
        profile_layout.addWidget(self.save_button)
        right_panel_layout.addWidget(profile_group)
        right_panel_layout.addStretch()

        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        # Initial message set in __init__

    def _populate_profiles(self):
        """Finds JSON files in the profiles directory."""
        self.profile_combo.clear()
        if not self.profiles_dir or not os.path.isdir(self.profiles_dir):
            print(f"Warning: Profiles directory not found or not set: {self.profiles_dir}")
            self.profile_combo.addItem("No Profiles Found"); self.profile_combo.setEnabled(False); return
        try:
            profiles = sorted([f for f in os.listdir(self.profiles_dir) if f.lower().endswith('.json')]) # Sort list
            if not profiles: self.profile_combo.addItem("No Profiles Found"); self.profile_combo.setEnabled(False); return

            self.profile_combo.addItems(profiles); self.profile_combo.setEnabled(True) # Enable here, might be disabled later
            base_config_file = os.path.basename(self.config_file) if self.config_file else ""
            default_profile_name = "test_profile.json"
            if base_config_file and base_config_file in profiles: self.profile_combo.setCurrentText(base_config_file)
            elif default_profile_name in profiles: self.profile_combo.setCurrentText(default_profile_name)
            else: self.profile_combo.setCurrentIndex(0) # Select first item if default/current not found
            # Manually trigger selection logic for the initially selected item
            self._profile_selected(self.profile_combo.currentIndex())
        except Exception as e:
             print(f"Error populating profiles from {self.profiles_dir}: {e}")
             self.profile_combo.addItem("Error Loading Profiles"); self.profile_combo.setEnabled(False)

    def _profile_selected(self, index):
        """Handles selection change in the profile dropdown."""
        selected_profile = self.profile_combo.itemText(index)
        if selected_profile and "No Profiles" not in selected_profile and "Error Loading" not in selected_profile:
             self.config_file = os.path.join(self.profiles_dir, selected_profile)
             print(f"Selected profile for next load/save: {self.config_file}")
             self.statusBar.showMessage(f"Profile selected: {selected_profile}. Press 'Reload' to apply.")

    def _load_profile(self):
        """Signals to reload the currently selected profile."""
        if self.tracking_active: QMessageBox.warning(self, "Tracking Active", "Stop tracking before reloading profile."); return
        if self.config_file and os.path.exists(self.config_file):
            print(f"Signaling to load profile: {self.config_file}")
            self.statusBar.showMessage(f"Reloading {os.path.basename(self.config_file)}...")
            # Disable controls during reload process
            self.load_button.setEnabled(False); self.track_button.setEnabled(False)
            self.profile_combo.setEnabled(False); self.save_button.setEnabled(False)
            self.webcam_label.setText(f"Reloading profile:\n{os.path.basename(self.config_file)}...")
            self.load_profile_signal.emit(self.config_file) # Emit signal with the path
        else: self.statusBar.showMessage("No valid profile selected to load.")

    def _save_profile_as(self):
        """Opens a dialog to save current settings."""
        # NOTE: This only saves the filename, the actual saving logic
        # needs to be implemented, likely involving getting current params from worker.
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Profile As...", self.profiles_dir, "JSON Files (*.json)")
        if filePath:
            if not filePath.lower().endswith('.json'): filePath += '.json'
            print(f"Placeholder: Save current settings to {filePath}")
            self.save_profile_signal.emit(filePath) # Signal if backend handles saving
            self.statusBar.showMessage(f"Save requested to {os.path.basename(filePath)}. (Backend saving not implemented)")
            self._populate_profiles() # Refresh list
            self.profile_combo.setCurrentText(os.path.basename(filePath)) # Select the newly saved file

    def _handle_track_button_toggle(self, checked):
        """Handles the Start/Stop Tracking button state change."""
        if checked:
            self.tracking_active = True
            self.track_button.setText("Stop Tracking")
            self.start_tracking_signal.emit() # Signal worker to start
            self.statusBar.showMessage("Tracking started...")
        else:
            self.tracking_active = False
            self.track_button.setText("Start Tracking")
            self.stop_tracking_signal.emit() # Signal worker to stop
            self.statusBar.showMessage("Previewing... Adjust position and press 'Start Tracking'.") # Updated message
        self._update_ui_state()

    def _emit_overlay_settings(self):
        """Emits the current state of overlay checkboxes."""
        # TODO: Connect this signal in main.py to a slot in the worker
        #       that stores these states for use during frame drawing.
        show_pose = self.pose_overlay_check.isChecked()
        show_roi = self.roi_overlay_check.isChecked()
        show_features = self.features_overlay_check.isChecked()
        self.overlay_settings_changed.emit(show_pose, show_roi, show_features)
        print(f"Overlay settings changed (UI only): Pose={show_pose}, ROI={show_roi}, Features={show_features}")

    def _update_ui_state(self):
        """Updates enabled/disabled state of widgets."""
        # Update based on whether components are initialized AND tracking state
        # Use track button state as proxy for readiness (enabled only after successful init)
        components_ready = self.track_button.isEnabled()
        is_previewing = not self.tracking_active

        self.profile_combo.setEnabled(is_previewing and components_ready)
        self.load_button.setEnabled(is_previewing and components_ready)
        self.save_button.setEnabled(is_previewing and components_ready)

        # Overlays are always toggleable if components are ready
        self.pose_overlay_check.setEnabled(components_ready)
        self.roi_overlay_check.setEnabled(components_ready)
        # Features only make sense when tracking
        self.features_overlay_check.setEnabled(self.tracking_active and components_ready)


    # --- Slots for Backend Signals ---

    def update_webcam_feed(self, frame):
        """Updates the webcam display label with a new frame."""
        try:
            if frame is None or frame.size == 0: return
            # Clear "Initializing/Waiting/Starting" text only if it's currently displayed
            current_text = self.webcam_label.text()
            if current_text and ("..." in current_text or "Failed" in current_text):
                self.webcam_label.setText("") # Clear text once valid frame arrives

            h, w, ch = frame.shape; bytes_per_line = ch * w
            qt_format = QImage.Format.Format_BGR888 if ch == 3 else QImage.Format.Format_Grayscale8 if ch == 1 else None
            if qt_format is None: print(f"Warning: Unexpected frame channel count: {ch}"); return
            qt_image = QImage(frame.data, w, h, bytes_per_line, qt_format)
            pixmap = QPixmap.fromImage(qt_image).scaled(self.webcam_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.webcam_label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error updating webcam feed: {e}")
            self.webcam_label.setText("Error displaying frame") # Show error on label

    def update_plot(self, plot_data):
        """Updates the plot widget with new signal data."""
        if not plot_data: self.plot_curve.setData([]); return
        data_len = len(plot_data)
        if data_len >= PLOT_BUFFER_SIZE: self.plot_data_buffer = np.array(plot_data[-PLOT_BUFFER_SIZE:])
        else: self.plot_data_buffer[:data_len] = plot_data; self.plot_data_buffer[data_len:] = 0
        self.plot_curve.setData(self.plot_data_buffer)

    def update_status_labels(self, bpm, is_valid, phase):
        """Updates the BPM and Phase status labels."""
        bpm_text = f"BPM: {bpm:.1f}" if is_valid else "BPM: ---"; self.bpm_label.setText(bpm_text)
        palette = self.bpm_label.palette(); color = QColor('green') if is_valid else QColor('red'); palette.setColor(QPalette.ColorRole.WindowText, color); self.bpm_label.setPalette(palette)
        phase_map = {SignalProcessor.PHASE_INHALE: "Inhale", SignalProcessor.PHASE_EXHALE: "Exhale", SignalProcessor.PHASE_UNKNOWN: "---"}
        phase_text = f"Phase: {phase_map.get(phase, 'Error')}"; self.phase_label.setText(phase_text)

    def show_error_message(self, message):
        """Displays an error message in the status bar."""
        print(f"UI Received Error: {message}")
        self.statusBar.showMessage(f"Error: {message}", 5000) # Show for 5 seconds

    def handle_worker_setup_finished(self, success, message):
        """Handles the signal indicating worker VIDEO/CONFIG setup is complete."""
        print(f"[UI] Worker initial setup finished. Success: {success}, Message: {message}")
        if success:
            # Update status bar to indicate next step
            self.statusBar.showMessage("Initializing components (PoseDetector)...")
            self.webcam_label.setText("Initializing components...") # Update label
        else:
            # Video/Config failed, update UI and keep controls disabled
            self.webcam_label.setText(f"Video/Config Setup Failed:\n{message}")
            self.statusBar.showMessage(f"Video/Config Setup Failed: {message}")
            self.track_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.profile_combo.setEnabled(False)

    # --- UPDATED SLOT ---
    def handle_component_initialized(self, component_name, success, message):
        """Handles the signal indicating a specific component initialization status."""
        print(f"[UI] Component Initialized: {component_name}, Success: {success}, Msg: {message}")
        if success:
            # Update status bar to show progress
            self.statusBar.showMessage(f"Initialized {component_name}...")
            # Check if this was the last component (PipelineManager)
            if component_name == "PipelineManager":
                # All components are ready, switch to preview mode message
                self.statusBar.showMessage("Previewing... Adjust position and press 'Start Tracking'.", 5000) # Show for 5 secs
                self.webcam_label.setText("Waiting for webcam feed...") # Ready for frames
                self.track_button.setEnabled(True) # NOW enable tracking button
                self.load_button.setEnabled(True) # Enable profile buttons
                self.save_button.setEnabled(True)
                self.profile_combo.setEnabled(True)
                self._update_ui_state() # Update other UI states based on readiness
        else:
            # Initialization failed for this component
            fail_msg = f"{component_name} Initialization Failed: {message}"
            self.webcam_label.setText(fail_msg)
            self.statusBar.showMessage(fail_msg)
            # Keep tracking disabled, but allow profile reload
            self.track_button.setEnabled(False)
            self.load_button.setEnabled(True)
            self.save_button.setEnabled(False)
            self.profile_combo.setEnabled(True)
            self._update_ui_state()

    def closeEvent(self, event):
        """Handles the window close event."""
        print("Close event triggered.")
        if self.tracking_active: self.stop_tracking_signal.emit()
        event.accept()

# Example of running just the UI window for testing layout (without backend)
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication

    PROFILES_DIR_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'profiles'))
    print(f"Testing UI - Profiles Dir: {PROFILES_DIR_TEST}")

    app = QApplication(sys.argv)
    pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
    main_win = MainWindow(profiles_dir=PROFILES_DIR_TEST)
    main_win.show()
    sys.exit(app.exec())
