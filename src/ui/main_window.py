# src/ui/main_window.py
# Defines the main application window using PyQt.
# MODIFIED: Accept profiles_dir during initialization and use it.
# FIXED: Added QApplication import to __main__ block for standalone testing.

import sys
import os
import numpy as np
# Import QApplication here for the main class usage is fine,
# but it also needs to be imported within the __main__ block if run standalone.
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QSizePolicy, QStatusBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
import pyqtgraph as pg # For plotting

# --- Constants ---
PLOT_BUFFER_SIZE = 500 # Number of points to display on the plot
# PROFILES_DIR is now passed in __init__

# --- Import needed for status label update ---
try:
    # Need to go up two levels from src/ui to reach src
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from signal_processor import SignalProcessor
except ImportError:
    print("Warning: Could not import SignalProcessor for phase constants in UI.")
    # Define dummy constants if import fails
    class SignalProcessor:
        PHASE_INHALE = 1
        PHASE_EXHALE = -1
        PHASE_UNKNOWN = 0


class MainWindow(QMainWindow):
    """
    Main application window class.
    """
    # Signals emitted by the UI for the backend
    start_tracking_signal = pyqtSignal()
    stop_tracking_signal = pyqtSignal()
    load_profile_signal = pyqtSignal(str) # Emits the selected profile path
    save_profile_signal = pyqtSignal(str) # Emits the path to save to
    # Add signals for overlay toggles if needed

    def __init__(self, config_file="", profiles_dir="", parent=None): # Added profiles_dir
        super().__init__(parent)
        self.setWindowTitle("Respiration Modulator M4L")
        self.setGeometry(100, 100, 1200, 800) # Initial size

        self.config_file = config_file # Store initial config path
        self.profiles_dir = profiles_dir # Store profiles directory path

        # --- Data storage ---
        self.plot_data_buffer = np.zeros(PLOT_BUFFER_SIZE)

        # --- Initialize UI Elements ---
        self._init_ui()

        # --- Initial State ---
        self.tracking_active = False
        self._update_ui_state()

    def _init_ui(self):
        """Creates and arranges all UI widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget) # Main vertical layout

        # --- Top Section: Webcam and Plot ---
        top_layout = QVBoxLayout()
        main_layout.addLayout(top_layout, 7) # Allocate 70% height

        # Webcam Feed Label
        self.webcam_label = QLabel("Waiting for webcam feed...")
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored) # Allow stretching
        palette = self.webcam_label.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('black'))
        self.webcam_label.setAutoFillBackground(True)
        self.webcam_label.setPalette(palette)
        top_layout.addWidget(self.webcam_label, 5)

        # Plot Widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Samples')
        self.plot_widget.setTitle('Filtered Signal', size='10pt')
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2))
        top_layout.addWidget(self.plot_widget, 2)

        # --- Bottom Section: Controls ---
        bottom_controls_widget = QWidget()
        bottom_controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        main_layout.addWidget(bottom_controls_widget, 3)

        bottom_layout = QHBoxLayout(bottom_controls_widget)

        # Left Side: Status & Tracking Button
        left_panel_layout = QVBoxLayout()
        bottom_layout.addLayout(left_panel_layout, 1)
        status_group = QGroupBox("Status")
        status_layout = QGridLayout(status_group)
        self.bpm_label = QLabel("BPM: ---")
        self.phase_label = QLabel("Phase: ---")
        status_layout.addWidget(self.bpm_label, 0, 0)
        status_layout.addWidget(self.phase_label, 1, 0)
        left_panel_layout.addWidget(status_group)
        self.track_button = QPushButton("Start Tracking")
        self.track_button.setCheckable(True)
        self.track_button.setFixedHeight(40)
        self.track_button.toggled.connect(self._handle_track_button_toggle)
        left_panel_layout.addWidget(self.track_button)
        left_panel_layout.addStretch()

        # Middle: Overlays & Settings
        middle_panel_layout = QVBoxLayout()
        bottom_layout.addLayout(middle_panel_layout, 2)
        overlays_group = QGroupBox("Overlays")
        overlays_layout = QVBoxLayout(overlays_group)
        self.pose_overlay_check = QCheckBox("Show Pose")
        self.roi_overlay_check = QCheckBox("Show ROI")
        self.features_overlay_check = QCheckBox("Show Features")
        overlays_layout.addWidget(self.pose_overlay_check)
        overlays_layout.addWidget(self.roi_overlay_check)
        overlays_layout.addWidget(self.features_overlay_check)
        middle_panel_layout.addWidget(overlays_group)
        settings_group = QGroupBox("Settings (Display Only)")
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
        bottom_layout.addLayout(right_panel_layout, 1)
        profile_group = QGroupBox("Profile Management")
        profile_layout = QVBoxLayout(profile_group)
        self.profile_combo = QComboBox()
        self._populate_profiles() # Load available profiles using self.profiles_dir
        self.profile_combo.currentIndexChanged.connect(self._profile_selected)
        profile_layout.addWidget(QLabel("Select Profile:"))
        profile_layout.addWidget(self.profile_combo)
        self.load_button = QPushButton("Reload Selected Profile") # Assign to self if needed elsewhere
        self.save_button = QPushButton("Save Current Settings As...") # Assign to self
        self.load_button.clicked.connect(self._load_profile)
        self.save_button.clicked.connect(self._save_profile_as)
        profile_layout.addWidget(self.load_button)
        profile_layout.addWidget(self.save_button)
        right_panel_layout.addWidget(profile_group)
        right_panel_layout.addStretch()

        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready. Press 'Start Tracking'.")

    def _populate_profiles(self):
        """Finds JSON files in the profiles directory."""
        self.profile_combo.clear()
        # Use the stored profiles directory path
        if not self.profiles_dir or not os.path.isdir(self.profiles_dir):
            print(f"Warning: Profiles directory not found or not set: {self.profiles_dir}")
            self.profile_combo.addItem("No Profiles Found")
            self.profile_combo.setEnabled(False)
            return

        try:
            profiles = [f for f in os.listdir(self.profiles_dir) if f.lower().endswith('.json')]
            if not profiles:
                self.profile_combo.addItem("No Profiles Found")
                self.profile_combo.setEnabled(False)
                return

            self.profile_combo.addItems(profiles)
            self.profile_combo.setEnabled(True)

            # Try to select the default or last used profile
            base_config_file = os.path.basename(self.config_file) if self.config_file else ""
            # Find default profile name from config if possible (adjust as needed)
            default_profile_name = "test_profile.json" # Fallback

            if base_config_file and base_config_file in profiles:
                self.profile_combo.setCurrentText(base_config_file)
            elif default_profile_name in profiles:
                 self.profile_combo.setCurrentText(default_profile_name)

        except Exception as e:
             print(f"Error populating profiles from {self.profiles_dir}: {e}")
             self.profile_combo.addItem("Error Loading Profiles")
             self.profile_combo.setEnabled(False)


    def _profile_selected(self, index):
        """Handles selection change in the profile dropdown."""
        selected_profile = self.profile_combo.itemText(index)
        if selected_profile and "No Profiles" not in selected_profile and "Error Loading" not in selected_profile:
             # Update the internal config_file path based on selection
             self.config_file = os.path.join(self.profiles_dir, selected_profile)
             print(f"Selected profile for next load/save: {self.config_file}")
             self.statusBar.showMessage(f"Profile selected: {selected_profile}. Press 'Reload' to apply.")


    def _load_profile(self):
        """Placeholder: Signals to reload the currently selected profile."""
        if self.tracking_active:
            QMessageBox.warning(self, "Tracking Active", "Stop tracking before reloading profile.")
            return
        # Use self.config_file which is updated by _profile_selected
        if self.config_file and os.path.exists(self.config_file):
            print(f"Signaling to load profile: {self.config_file}")
            self.statusBar.showMessage(f"Reload requested for {os.path.basename(self.config_file)}. (Restart app to apply fully for now)")
            self.load_profile_signal.emit(self.config_file) # Emit signal with the path
        else:
             self.statusBar.showMessage("No valid profile selected to load.")


    def _save_profile_as(self):
        """Placeholder: Opens a dialog to save current settings."""
        # Use the stored profiles directory path for the dialog
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Profile As...", self.profiles_dir, "JSON Files (*.json)")
        if filePath:
            # Ensure it ends with .json
            if not filePath.lower().endswith('.json'):
                filePath += '.json'
            print(f"Placeholder: Save current settings to {filePath}")
            self.save_profile_signal.emit(filePath) # Signal if backend handles saving
            self.statusBar.showMessage(f"Save requested to {os.path.basename(filePath)}. (Functionality not fully implemented)")
            self._populate_profiles() # Refresh list
            self.profile_combo.setCurrentText(os.path.basename(filePath)) # Select the newly saved file


    def _handle_track_button_toggle(self, checked):
        """Handles the Start/Stop Tracking button state change."""
        if checked:
            self.tracking_active = True
            self.track_button.setText("Stop Tracking")
            self.start_tracking_signal.emit()
            self.statusBar.showMessage("Tracking started...")
            # Disable profile controls while tracking
            self.profile_combo.setEnabled(False)
            self.load_button.setEnabled(False)
            self.save_button.setEnabled(False)
        else:
            self.tracking_active = False
            self.track_button.setText("Start Tracking")
            self.stop_tracking_signal.emit()
            self.statusBar.showMessage("Tracking stopped.")
            # Re-enable profile controls
            self._populate_profiles() # Re-check available profiles and re-enable combo
            self.load_button.setEnabled(True)
            self.save_button.setEnabled(True)
        self._update_ui_state()

    def _update_ui_state(self):
        """Updates enabled/disabled state of widgets based on tracking status."""
        # Example: Disable overlay checkboxes if not tracking?
        # self.pose_overlay_check.setEnabled(self.tracking_active)
        # self.roi_overlay_check.setEnabled(self.tracking_active)
        # self.features_overlay_check.setEnabled(self.tracking_active)
        pass # Add more state updates as needed

    # --- Slots for Backend Signals ---

    def update_webcam_feed(self, frame):
        """Updates the webcam display label with a new frame."""
        try:
            if frame is None or frame.size == 0: return
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.webcam_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.webcam_label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error updating webcam feed: {e}")

    def update_plot(self, plot_data):
        """Updates the plot widget with new signal data."""
        if not plot_data:
            self.plot_curve.setData([]) # Clear plot if no data
            return
        data_len = len(plot_data)
        if data_len >= PLOT_BUFFER_SIZE:
            self.plot_data_buffer = np.array(plot_data[-PLOT_BUFFER_SIZE:])
        else:
            self.plot_data_buffer[:data_len] = plot_data
            self.plot_data_buffer[data_len:] = 0 # Zero pad if needed
        self.plot_curve.setData(self.plot_data_buffer)

    def update_status_labels(self, bpm, is_valid, phase):
        """Updates the BPM and Phase status labels."""
        bpm_text = f"BPM: {bpm:.1f}" if is_valid else "BPM: ---"
        self.bpm_label.setText(bpm_text)
        palette = self.bpm_label.palette()
        color = QColor('green') if is_valid else QColor('red')
        palette.setColor(QPalette.ColorRole.WindowText, color)
        self.bpm_label.setPalette(palette)

        phase_map = {SignalProcessor.PHASE_INHALE: "Inhale",
                     SignalProcessor.PHASE_EXHALE: "Exhale",
                     SignalProcessor.PHASE_UNKNOWN: "---"}
        phase_text = f"Phase: {phase_map.get(phase, 'Error')}"
        self.phase_label.setText(phase_text)

    def show_error_message(self, message):
        """Displays an error message in the status bar and potentially a dialog."""
        print(f"UI Received Error: {message}")
        self.statusBar.showMessage(f"Error: {message}", 5000)

    def closeEvent(self, event):
        """Handles the window close event."""
        print("Close event triggered.")
        event.accept()

# Example of running just the UI window for testing layout (without backend)
if __name__ == '__main__':
    # --- *** ADDED IMPORT FOR STANDALONE TESTING *** ---
    from PyQt6.QtWidgets import QApplication
    # --- *** ---

    # Define PROFILES_DIR relative to this file's location (src/ui)
    # Go up two levels to the project root, then into 'profiles'
    PROFILES_DIR_TEST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'profiles'))
    print(f"Testing UI - Profiles Dir: {PROFILES_DIR_TEST}")

    app = QApplication(sys.argv)
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    # Pass the profiles directory path
    main_win = MainWindow(profiles_dir=PROFILES_DIR_TEST)
    main_win.show()
    sys.exit(app.exec())
