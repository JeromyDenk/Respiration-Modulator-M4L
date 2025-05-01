# src/ui/main_window.py
# Defines the main application window using PyQt6.
# MODIFIED: Revert to Preferred policy for bottom, use updateGeometry on settings group.

import sys
import os
import numpy as np
import threading # For thread ID diagnostic
import json # For loading initial config
import traceback # <<< ENSURE TRACEBACK IMPORT

# --- PyQt6 Imports ---
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
        QGridLayout, QGroupBox, QComboBox, QCheckBox, QFileDialog, QMessageBox,
        QSizePolicy, QStatusBar, QSpinBox, QDoubleSpinBox, QFormLayout # Added widgets
    )
    from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer # Added QTimer
    from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor # Added QFont
except ImportError:
    print("Fatal Error: PyQt6 not found. Please install it (e.g., pip install PyQt6)")
    sys.exit(1)

import pyqtgraph as pg # For plotting

# --- Constants ---
PLOT_BUFFER_SIZE = 500 # Number of points to display on the plot

# --- Import needed for status label update ---
try:
    module_dir = os.path.dirname(__file__)
    src_path = os.path.abspath(os.path.join(module_dir, '..'))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from signal_processor import SignalProcessor
except ImportError:
    print("Warning: Could not import SignalProcessor for phase constants in UI.")
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
    save_profile_signal = pyqtSignal(str) # Emits the path to save to (for Save As)
    save_current_profile_signal = pyqtSignal() # Signal to save to the currently selected profile
    overlay_settings_changed = pyqtSignal(bool, bool, bool)
    apply_settings_signal = pyqtSignal(dict) # Emits dict of settings to apply
    reset_tracking_signal = pyqtSignal() # Signal to reset tracking


    def __init__(self, config_file="", profiles_dir="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Respiration Modulator M4L")
        self.setGeometry(100, 100, 1200, 700) # Initial size

        self.config_file = config_file # Store initial config path
        self.profiles_dir = profiles_dir # Store profiles directory path
        self.plot_data_buffer = np.zeros(PLOT_BUFFER_SIZE)
        self.tracking_active = False # UI's understanding of tracking state

        self._init_ui()
        self._load_and_populate_initial_settings()
        self.webcam_label.setText("Initializing Video...")
        self.statusBar.showMessage("Initializing video source...")
        self._update_ui_state()
        # Ensure settings are hidden initially if toggle is unchecked
        self._toggle_settings_visibility(self.settings_toggle_button.isChecked())


    def _init_ui(self):
        """Creates and arranges all UI widgets."""

        # --- Set Slightly Larger Font ---
        default_font = QApplication.font()
        default_point_size = default_font.pointSize()
        if default_point_size < 8: default_point_size = 10
        new_point_size = int(default_point_size * 1.1) # Increase by 10%
        larger_font = QFont(default_font)
        larger_font.setPointSize(new_point_size)
        self.setFont(larger_font)
        print(f"[UI] Setting base font size to {new_point_size}pt (was {default_point_size}pt)")
        # ---

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget) # Main vertical layout
        main_layout.setContentsMargins(5,5,5,5) # Reduce margins

        # --- Top Section: Webcam and Plot ---
        self.top_widget = QWidget() # Store reference
        top_layout = QVBoxLayout(self.top_widget)
        top_layout.setContentsMargins(0,0,0,0)
        # --- SET RECOMMENDED POLICY ---
        self.top_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        main_layout.addWidget(self.top_widget, 1) # Stretch factor 1

        self.webcam_label = QLabel(); self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        self.webcam_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.webcam_label.setMinimumHeight(300)
        palette = self.webcam_label.palette(); palette.setColor(QPalette.ColorRole.Window, QColor('black')); self.webcam_label.setAutoFillBackground(True); self.webcam_label.setPalette(palette)
        top_layout.addWidget(self.webcam_label, 5)

        self.plot_widget = pg.PlotWidget(); self.plot_widget.setBackground('w'); self.plot_widget.showGrid(x=True, y=True, alpha=0.3); self.plot_widget.setLabel('left', 'Amplitude'); self.plot_widget.setLabel('bottom', 'Samples')
        self.plot_widget.setTitle('Filtered Signal', size=f'{int(new_point_size*1.05)}pt')
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2))
        top_layout.addWidget(self.plot_widget, 2)

        # --- Bottom Section: Controls ---
        self.bottom_controls_widget = QWidget();
        # --- SET RECOMMENDED POLICY ---
        self.bottom_controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred) # Use Preferred vertically
        main_layout.addWidget(self.bottom_controls_widget, 0) # Stretch factor 0

        bottom_main_layout = QVBoxLayout(self.bottom_controls_widget)
        bottom_main_layout.setContentsMargins(5, 5, 5, 5)
        bottom_main_layout.setSpacing(10)

        # --- Row 1: Status, Overlays, Profiles ---
        row1_layout = QHBoxLayout()
        bottom_main_layout.addLayout(row1_layout)
        # (Group box layouts remain the same)
        status_group = QGroupBox("Status"); status_layout = QGridLayout(status_group)
        self.bpm_label = QLabel("BPM: ---"); self.phase_label = QLabel("Phase: ---"); self.osc_status_label = QLabel("OSC Status: ---")
        status_layout.addWidget(self.bpm_label, 0, 0); status_layout.addWidget(self.phase_label, 1, 0); status_layout.addWidget(self.osc_status_label, 2, 0)
        row1_layout.addWidget(status_group, 1)
        overlays_group = QGroupBox("Overlays"); overlays_layout = QVBoxLayout(overlays_group)
        self.pose_overlay_check = QCheckBox("Show Pose"); self.roi_overlay_check = QCheckBox("Show ROI"); self.features_overlay_check = QCheckBox("Show Features")
        self.pose_overlay_check.setChecked(True); self.roi_overlay_check.setChecked(True); self.features_overlay_check.setChecked(False)
        self.pose_overlay_check.stateChanged.connect(self._emit_overlay_settings); self.roi_overlay_check.stateChanged.connect(self._emit_overlay_settings); self.features_overlay_check.stateChanged.connect(self._emit_overlay_settings)
        overlays_layout.addWidget(self.pose_overlay_check); overlays_layout.addWidget(self.roi_overlay_check); overlays_layout.addWidget(self.features_overlay_check)
        row1_layout.addWidget(overlays_group, 1)
        profile_group = QGroupBox("Profile Management"); profile_layout = QVBoxLayout(profile_group)
        self.profile_combo = QComboBox(); self.profile_combo.currentIndexChanged.connect(self._profile_selected); self._populate_profiles()
        profile_layout.addWidget(self.profile_combo)
        self.load_button = QPushButton("Reload Selected Profile")
        save_button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save"); self.save_as_button = QPushButton("Save As...")
        save_button_layout.addWidget(self.save_button); save_button_layout.addWidget(self.save_as_button)
        self.load_button.clicked.connect(self._load_profile); self.save_button.clicked.connect(self._save_profile); self.save_as_button.clicked.connect(self._save_profile_as)
        self.load_button.setEnabled(False); self.save_button.setEnabled(False); self.save_as_button.setEnabled(False)
        profile_layout.addWidget(self.load_button); profile_layout.addLayout(save_button_layout)
        row1_layout.addWidget(profile_group, 1)

        # --- Row 2: Tracking Buttons (Centered) ---
        row2_layout = QHBoxLayout()
        bottom_main_layout.addLayout(row2_layout)
        row2_layout.addStretch(1); self.track_button = QPushButton("Start Tracking"); self.track_button.setCheckable(True); self.track_button.setFixedHeight(int(40 * 1.1)); self.track_button.toggled.connect(self._handle_track_button_toggle); self.track_button.setEnabled(False); row2_layout.addWidget(self.track_button, 0)
        self.reset_button = QPushButton("Reset Tracking"); self.reset_button.setFixedHeight(int(40 * 1.1)); self.reset_button.clicked.connect(self._handle_reset_button_click); self.reset_button.setEnabled(False); row2_layout.addWidget(self.reset_button, 0); row2_layout.addStretch(1)

        # --- Row 3: Settings Toggle Button ---
        self.settings_toggle_button = QPushButton("▶ Show Settings"); self.settings_toggle_button.setCheckable(True); self.settings_toggle_button.setChecked(False)
        self.settings_toggle_button.setStyleSheet(""" QPushButton { text-align: left; padding-left: 5px; border: none; font-weight: bold; background-color: transparent; } QPushButton:checked { background-color: transparent; border: none; } QPushButton:hover { background-color: #eee; } QPushButton:pressed { background-color: #ddd; } """)
        self.settings_toggle_button.setFlat(True); self.settings_toggle_button.toggled.connect(self._toggle_settings_visibility); self.settings_toggle_button.setEnabled(False)
        bottom_main_layout.addWidget(self.settings_toggle_button)

        # --- Row 4: Settings Group Box ---
        self.settings_main_group = QGroupBox(); self.settings_main_group.setStyleSheet("QGroupBox { border: none; margin-top: 0px; padding-top: 0px; }")
        settings_main_layout = QVBoxLayout(self.settings_main_group); settings_main_layout.setContentsMargins(0, 0, 0, 0)
        settings_row_layout = QHBoxLayout(); settings_row_layout.setSpacing(10)
        # (Settings widgets layout remains the same)
        ft_group = QGroupBox("Feature Tracking (LK)"); ft_layout = QFormLayout(ft_group); ft_layout.setContentsMargins(5, 10, 5, 5)
        self.maxCorners_spin = QSpinBox(minimum=10, maximum=500); self.qualityLevel_spin = QDoubleSpinBox(minimum=0.01, maximum=0.99, singleStep=0.01, decimals=2); self.minDistance_spin = QSpinBox(minimum=1, maximum=50); self.winSize_spin = QSpinBox(minimum=3, maximum=51, singleStep=2); self.maxLevel_spin = QSpinBox(minimum=0, maximum=8)
        ft_layout.addRow("Max Corners:", self.maxCorners_spin); ft_layout.addRow("Quality Level:", self.qualityLevel_spin); ft_layout.addRow("Min Distance:", self.minDistance_spin); ft_layout.addRow("Window Size:", self.winSize_spin); ft_layout.addRow("Pyramid Levels:", self.maxLevel_spin)
        settings_row_layout.addWidget(ft_group)
        sp_group = QGroupBox("Signal Gen & Processing"); sp_layout = QFormLayout(sp_group); sp_layout.setContentsMargins(5, 10, 5, 5)
        self.aggMethod_combo = QComboBox(); self.aggMethod_combo.addItems(["median", "mean"]); self.filtLow_spin = QDoubleSpinBox(minimum=0.05, maximum=1.00, singleStep=0.01, decimals=2); self.filtHigh_spin = QDoubleSpinBox(minimum=0.10, maximum=5.00, singleStep=0.05, decimals=2); self.filtType_combo = QComboBox(); self.filtType_combo.addItems(["lfilter", "filtfilt"]); self.peakProm_spin = QDoubleSpinBox(minimum=0.0, maximum=10.0, singleStep=0.005, value=0.0, decimals=3)
        self.filtLow_spin.setToolTip("Lower cutoff frequency (Hz) for bandpass filter."); self.filtHigh_spin.setToolTip("Upper cutoff frequency (Hz) for bandpass filter."); self.filtType_combo.setToolTip("Filter method: lfilter (causal, faster) or filtfilt (zero-phase, slower)."); self.peakProm_spin.setToolTip("Minimum peak prominence (0 = disabled). Tune visually.")
        sp_layout.addRow("Aggregation:", self.aggMethod_combo); sp_layout.addRow("Filter Low Hz:", self.filtLow_spin); sp_layout.addRow("Filter High Hz:", self.filtHigh_spin); sp_layout.addRow("Filter Method:", self.filtType_combo); sp_layout.addRow("Peak Prominence:", self.peakProm_spin)
        settings_row_layout.addWidget(sp_group)
        settings_main_layout.addLayout(settings_row_layout)
        self.apply_button = QPushButton("Apply Settings"); self.apply_button.clicked.connect(self._gather_and_apply_settings); self.apply_button.setEnabled(False)
        settings_main_layout.addWidget(self.apply_button, 0, Qt.AlignmentFlag.AlignCenter)
        bottom_main_layout.addWidget(self.settings_main_group)

        # Status Bar
        self.statusBar = QStatusBar(); self.setStatusBar(self.statusBar)

        # --- Apply Larger Font to Group Box Titles explicitly ---
        title_font = QFont(larger_font)
        status_group.setFont(title_font); overlays_group.setFont(title_font); ft_group.setFont(title_font); sp_group.setFont(title_font); profile_group.setFont(title_font)

        # --- Call toggle settings AFTER all widgets are created ---
        self._toggle_settings_visibility(self.settings_toggle_button.isChecked())


    def _load_and_populate_initial_settings(self):
        # (Function remains the same)
        initial_config = {}
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f: initial_config = json.load(f)
                print(f"[UI Init] Loaded initial settings from: {self.config_file}")
            except Exception as e:
                print(f"[UI Init] Error loading initial config '{self.config_file}': {e}")
                QMessageBox.warning(self, "Config Error", f"Could not load initial profile:\n{self.config_file}\n\n{e}\n\nUsing default widget values.")
                initial_config = {}
        else: print("[UI Init] No initial config file specified or found. Using default widget values.")
        self.populate_settings_widgets(initial_config)

    def _populate_profiles(self):
        # (Function remains the same)
        current_selection = self.profile_combo.currentText()
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        found_profiles = False
        if not self.profiles_dir or not os.path.isdir(self.profiles_dir): print(f"Warning: Profiles directory not found or not set: {self.profiles_dir}")
        else:
            try:
                profiles = sorted([f for f in os.listdir(self.profiles_dir) if f.lower().endswith('.json')])
                if profiles: self.profile_combo.addItems(profiles); found_profiles = True
            except Exception as e: print(f"Error populating profiles from {self.profiles_dir}: {e}")
        if not found_profiles: self.profile_combo.addItem("No Profiles Found"); self.profile_combo.setEnabled(False); self.profile_combo.blockSignals(False); return
        target_profile = os.path.basename(self.config_file) if self.config_file else current_selection
        if not target_profile: target_profile = "test_profile.json"
        index_to_set = self.profile_combo.findText(target_profile)
        if index_to_set != -1: self.profile_combo.setCurrentIndex(index_to_set)
        elif self.profile_combo.count() > 0: self.profile_combo.setCurrentIndex(0)
        self.profile_combo.setEnabled(True); self.profile_combo.blockSignals(False)

    def _profile_selected(self, index):
        # (Function remains the same)
        selected_profile = self.profile_combo.itemText(index)
        if selected_profile and "No Profiles" not in selected_profile and "Error Loading" not in selected_profile:
             self.config_file = os.path.join(self.profiles_dir, selected_profile)
             print(f"Selected profile for next load/save: {self.config_file}")
             if hasattr(self, 'statusBar') and self.statusBar: self.statusBar.showMessage(f"Profile selected: {selected_profile}. Press 'Reload' or 'Save' to apply.")
             else: print("Warning: statusBar not available when _profile_selected was called.")

    def _load_profile(self):
        # (Function remains the same)
        if self.tracking_active: QMessageBox.warning(self, "Tracking Active", "Stop tracking before reloading profile."); return
        if self.config_file and os.path.exists(self.config_file):
            print(f"Signaling to load profile: {self.config_file}")
            self.statusBar.showMessage(f"Reloading {os.path.basename(self.config_file)}...")
            self.load_button.setEnabled(False); self.track_button.setEnabled(False); self.profile_combo.setEnabled(False); self.save_button.setEnabled(False); self.save_as_button.setEnabled(False)
            if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(False)
            self.webcam_label.setText(f"Reloading profile:\n{os.path.basename(self.config_file)}...")
            self.load_profile_signal.emit(self.config_file)
        else: self.statusBar.showMessage("No valid profile selected to load.")

    def _save_profile(self):
        # (Function remains the same)
        if self.tracking_active: QMessageBox.warning(self, "Tracking Active", "Stop tracking before saving profile."); return
        current_profile_name = self.profile_combo.currentText()
        if not current_profile_name or "No Profiles" in current_profile_name or "Error Loading" in current_profile_name: QMessageBox.warning(self, "Save Error", "No valid profile selected to save to."); return
        save_path = os.path.join(self.profiles_dir, current_profile_name)
        reply = QMessageBox.question(self, "Confirm Save", f"Overwrite profile '{current_profile_name}' with current settings?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            print(f"Signaling to save current settings to: {save_path}")
            self.save_profile_signal.emit(save_path) # Re-use Save As signal for now
            self.statusBar.showMessage(f"Save requested for {current_profile_name}. (Backend saving not implemented)")

    def _save_profile_as(self):
        # (Function remains the same)
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Profile As...", self.profiles_dir, "JSON Files (*.json)")
        if filePath:
            if not filePath.lower().endswith('.json'): filePath += '.json'
            print(f"Placeholder: Save current settings to {filePath}")
            self.save_profile_signal.emit(filePath)
            self.statusBar.showMessage(f"Save As requested to {os.path.basename(filePath)}. (Backend saving not implemented)")
            self._populate_profiles(); self.profile_combo.setCurrentText(os.path.basename(filePath))

    def _handle_track_button_toggle(self, checked):
        # (Function remains the same)
        if checked:
            self.tracking_active = True; self.track_button.setText("Stop Tracking")
            self.start_tracking_signal.emit(); self.statusBar.showMessage("Tracking started...")
        else:
            self.tracking_active = False; self.track_button.setText("Start Tracking")
            self.stop_tracking_signal.emit(); self.statusBar.showMessage("Previewing... Adjust position and press 'Start Tracking'.")
        self._update_ui_state()

    def _handle_reset_button_click(self):
        # (Function remains the same)
        if self.tracking_active:
            print("[UI] Reset Tracking button clicked.")
            self.statusBar.showMessage("Resetting tracking...")
            self.reset_tracking_signal.emit()
            self.reset_button.setEnabled(False)
            QTimer.singleShot(1000, lambda: self.reset_button.setEnabled(self.tracking_active))
        else:
            print("[UI] Reset Tracking clicked but not tracking.")

    def _emit_overlay_settings(self):
        # (Function remains the same)
        show_pose = self.pose_overlay_check.isChecked(); show_roi = self.roi_overlay_check.isChecked(); show_features = self.features_overlay_check.isChecked()
        current_thread_id = threading.get_ident(); print(f"[UI Emit - Thread: {current_thread_id}] Emitting overlay_settings_changed: Pose={show_pose}, ROI={show_roi}, Features={show_features}")
        self.overlay_settings_changed.emit(show_pose, show_roi, show_features)

    def _gather_and_apply_settings(self):
        # (Function remains the same)
        if self.tracking_active: QMessageBox.warning(self, "Tracking Active", "Stop tracking before applying new settings."); return
        settings = {
            'feature_tracker': {
                'OPTICAL_FLOW_PARAMS': {
                    'feature_params': {'maxCorners': self.maxCorners_spin.value(), 'qualityLevel': self.qualityLevel_spin.value(), 'minDistance': self.minDistance_spin.value(), 'blockSize': 7 },
                    'lk_params': {'winSize': [self.winSize_spin.value(), self.winSize_spin.value()], 'maxLevel': self.maxLevel_spin.value(), 'criteria': [3, 10, 0.03] }
                }, 'FEATURE_REDETECT_THRESHOLD': int(self.maxCorners_spin.value() * 0.7)
            },
            'signal_generator': { 'SIGNAL_AGGREGATION_METHOD': self.aggMethod_combo.currentText() },
            'signal_processor': { 'SIGNAL_FILTER_LOW_HZ': self.filtLow_spin.value(), 'SIGNAL_FILTER_HIGH_HZ': self.filtHigh_spin.value(), 'SIGNAL_FILTER_METHOD': self.filtType_combo.currentText(), 'PEAK_DETECT_PROMINENCE': self.peakProm_spin.value() if self.peakProm_spin.value() > 1e-6 else None }
        }
        print("[UI] Applying settings:", settings); self.statusBar.showMessage("Applying settings... Restart tracking if needed.")
        self.apply_settings_signal.emit(settings)
        self.apply_button.setEnabled(False); QTimer.singleShot(1000, lambda: self.apply_button.setEnabled(not self.tracking_active))

    def _update_ui_state(self):
        # (Function remains the same)
        components_ready = self.track_button.isEnabled(); is_previewing = not self.tracking_active
        settings_enabled = is_previewing and components_ready
        self.profile_combo.setEnabled(settings_enabled); self.load_button.setEnabled(settings_enabled); self.save_button.setEnabled(settings_enabled); self.save_as_button.setEnabled(settings_enabled)
        if hasattr(self, 'settings_main_group'): self.settings_main_group.setEnabled(settings_enabled)
        if hasattr(self, 'apply_button'): self.apply_button.setEnabled(settings_enabled)
        if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(components_ready)
        self.pose_overlay_check.setEnabled(components_ready); self.roi_overlay_check.setEnabled(components_ready); self.features_overlay_check.setEnabled(self.tracking_active and components_ready)
        if hasattr(self, 'reset_button'): self.reset_button.setEnabled(self.tracking_active and components_ready)

    def _toggle_settings_visibility(self, checked):
        """Shows or hides the settings group box using updateGeometry."""
        if hasattr(self, 'settings_main_group'):
            self.settings_main_group.setVisible(checked)
            if checked: self.settings_toggle_button.setText("▼ Hide Settings")
            else: self.settings_toggle_button.setText("▶ Show Settings")
            # --- Use updateGeometry on the toggled widget ---
            self.settings_main_group.updateGeometry()


    # --- Slots for Backend Signals ---
    def update_webcam_feed(self, frame):
        # (Function remains the same)
        try:
            if frame is None or frame.size == 0: return
            current_text = self.webcam_label.text()
            if current_text and ("..." in current_text or "Failed" in current_text): self.webcam_label.setText("")
            h, w, ch = frame.shape; bytes_per_line = ch * w
            qt_format = QImage.Format.Format_BGR888 if ch == 3 else QImage.Format.Format_Grayscale8 if ch == 1 else None
            if qt_format is None: print(f"Warning: Unexpected frame channel count: {ch}"); return
            qt_image = QImage(frame.data, w, h, bytes_per_line, qt_format)
            pixmap = QPixmap.fromImage(qt_image).scaled(self.webcam_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.webcam_label.setPixmap(pixmap)
        except Exception as e: print(f"Error updating webcam feed: {e}"); self.webcam_label.setText("Error displaying frame")

    def update_plot(self, plot_data):
        # (Function remains the same)
        if not plot_data: self.plot_curve.setData([]); return
        data_len = len(plot_data)
        if data_len >= PLOT_BUFFER_SIZE: self.plot_data_buffer = np.array(plot_data[-PLOT_BUFFER_SIZE:])
        else: self.plot_data_buffer[:data_len] = plot_data; self.plot_data_buffer[data_len:] = 0
        self.plot_curve.setData(self.plot_data_buffer)

    def update_status_labels(self, bpm, is_valid, phase):
        # (Function remains the same)
        bpm_text = f"BPM: {bpm:.1f}" if is_valid else "BPM: ---"; self.bpm_label.setText(bpm_text)
        palette = self.bpm_label.palette(); color = QColor('green') if is_valid else QColor('red'); palette.setColor(QPalette.ColorRole.WindowText, color); self.bpm_label.setPalette(palette)
        phase_map = {SignalProcessor.PHASE_INHALE: "Inhale", SignalProcessor.PHASE_EXHALE: "Exhale", SignalProcessor.PHASE_UNKNOWN: "---"}
        phase_text = f"Phase: {phase_map.get(phase, 'Error')}"; self.phase_label.setText(phase_text)

    def update_osc_status(self, status_text, is_error=False):
         # (Function remains the same)
         self.osc_status_label.setText(f"OSC Status: {status_text}")
         palette = self.osc_status_label.palette(); color = QColor('red') if is_error else self.bpm_label.palette().color(QPalette.ColorRole.WindowText); palette.setColor(QPalette.ColorRole.WindowText, color); self.osc_status_label.setPalette(palette)

    def show_error_message(self, message):
        # (Function remains the same)
        print(f"UI Received Error: {message}")
        self.statusBar.showMessage(f"Error: {message}", 5000)

    def handle_worker_setup_finished(self, success, message):
        # (Function remains the same)
        print(f"[UI] Worker initial setup finished. Success: {success}, Message: {message}")
        if success: self.statusBar.showMessage("Initializing components (PoseDetector)..."); self.webcam_label.setText("Initializing components...")
        else: self.webcam_label.setText(f"Video/Config Setup Failed:\n{message}"); self.statusBar.showMessage(f"Video/Config Setup Failed: {message}"); self.track_button.setEnabled(False); self.load_button.setEnabled(False); self.save_button.setEnabled(False); self.profile_combo.setEnabled(False)

    def handle_component_initialized(self, component_name, success, message):
        # (Function remains the same)
        print(f"[UI] Component Initialized: {component_name}, Success: {success}, Msg: {message}")
        if success:
            self.statusBar.showMessage(f"Initialized {component_name}...")
            if component_name == "PipelineManager": # Last component
                self.statusBar.showMessage("Previewing... Adjust position and press 'Start Tracking'.", 5000)
                self.webcam_label.setText("Waiting for webcam feed...")
                self.track_button.setEnabled(True); self.load_button.setEnabled(True); self.save_button.setEnabled(True); self.save_as_button.setEnabled(True) # Enable new save buttons
                if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(True)
                self._update_ui_state()
                self._emit_overlay_settings()
                print("[UI] TODO: Request initial settings from worker to populate widgets.")
        else:
            fail_msg = f"{component_name} Initialization Failed: {message}"
            self.webcam_label.setText(fail_msg); self.statusBar.showMessage(fail_msg)
            self.track_button.setEnabled(False); self.load_button.setEnabled(True); self.save_button.setEnabled(False); self.save_as_button.setEnabled(False) # Disable save buttons
            self.profile_combo.setEnabled(True)
            if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(False)
            self._update_ui_state()

    def populate_settings_widgets(self, settings_dict):
        # (Function remains the same)
        print("[UI] Populating settings widgets with:", settings_dict)
        try:
            ft_settings = settings_dict.get('feature_tracker', {}); of_params = ft_settings.get('OPTICAL_FLOW_PARAMS', {}); feat_params = of_params.get('feature_params', {}); lk_params = of_params.get('lk_params', {})
            self.maxCorners_spin.setValue(feat_params.get('maxCorners', 100)); self.qualityLevel_spin.setValue(feat_params.get('qualityLevel', 0.3)); self.minDistance_spin.setValue(feat_params.get('minDistance', 7))
            win_size_val = lk_params.get('winSize', [15, 15]); self.winSize_spin.setValue(win_size_val[0] if isinstance(win_size_val, (list, tuple)) and len(win_size_val) > 0 else 15)
            self.maxLevel_spin.setValue(lk_params.get('maxLevel', 2))
            sg_settings = settings_dict.get('signal_generator', {}); agg_method = sg_settings.get('SIGNAL_AGGREGATION_METHOD', 'median'); self.aggMethod_combo.setCurrentText(agg_method)
            sp_settings = settings_dict.get('signal_processor', {}); self.filtLow_spin.setValue(sp_settings.get('SIGNAL_FILTER_LOW_HZ', 0.1)); self.filtHigh_spin.setValue(sp_settings.get('SIGNAL_FILTER_HIGH_HZ', 1.0)); filt_method = sp_settings.get('SIGNAL_FILTER_METHOD', 'lfilter'); self.filtType_combo.setCurrentText(filt_method)
            prominence = sp_settings.get('PEAK_DETECT_PROMINENCE'); self.peakProm_spin.setValue(prominence if prominence is not None else 0.0)
            self.statusBar.showMessage("Settings populated from backend.", 3000)
        except Exception as e: print(f"Error populating settings widgets: {e}"); traceback.print_exc(); self.statusBar.showMessage("Error loading settings to UI.", 3000)

    def closeEvent(self, event):
        # (Function remains the same)
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
