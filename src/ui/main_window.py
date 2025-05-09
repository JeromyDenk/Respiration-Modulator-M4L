# src/ui/main_window.py
# Defines the main application window using PyQt6.
# MODIFIED: Replaced QLabel for webcam with a custom VideoWidget for robust aspect ratio handling.
# MODIFIED: Refactored settings section layout for clarity.

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
        QSizePolicy, QStatusBar, QSpinBox, QDoubleSpinBox, QFormLayout, QSpacerItem # Added widgets
    )
    # Added QPainter, QRectF
    from PyQt6.QtCore import Qt, pyqtSignal, QSize, QTimer, QRect, QPoint, QRectF, pyqtSlot # Added pyqtSlot
    # Added QPainter, pyqtSlot
    from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QPainter
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


# --- Custom Video Widget ---
class VideoWidget(QWidget):
    """
    A custom QWidget optimized for displaying video frames (QImage)
    while strictly maintaining aspect ratio and providing letter/pillarboxing.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None # The current QImage to display
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False) # We handle background in paintEvent

    def update_frame(self, image: QImage):
        """Receives a new QImage frame and triggers a repaint."""
        if image and not image.isNull():
            self._image = image.copy() # Keep a copy
            self.update() # Schedule a repaint
        else:
            self._image = None
            self.update()

    def clear_frame(self):
        """Clears the current frame."""
        self._image = None
        self.update()

    def paintEvent(self, event):
        """Handles painting the widget. Draws the current frame scaled with aspect ratio."""
        painter = QPainter(self)

        # 1. Fill background (ensures letterboxing color)
        painter.fillRect(self.rect(), QColor('black'))

        if self._image is None or self._image.isNull():
            # Optionally draw placeholder text if no image
            # painter.setPen(Qt.GlobalColor.white)
            # painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Video Feed")
            return # Nothing else to draw

        # 2. Calculate target rectangle preserving aspect ratio
        img_size = self._image.size() # QSize of the source image
        widget_rect = self.rect()    # QRect of the widget area

        # Scale the image size to fit within the widget rect, keeping aspect ratio
        scaled_size = img_size.scaled(widget_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)

        # Calculate the top-left position to center the scaled image
        x = (widget_rect.width() - scaled_size.width()) / 2
        y = (widget_rect.height() - scaled_size.height()) / 2

        target_rect = QRect(QPoint(int(x), int(y)), scaled_size) # Create the target QRect

        # 3. Draw the image
        # Use SmoothTransformation for better quality if performance allows
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target_rect, self._image)

    def sizeHint(self):
        # Provide a reasonable default size hint, though layout policy often overrides
        return QSize(640, 480)


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
        self.setGeometry(100, 100, 1200, 1050) # Initial size (height increased by 50%)

        self.config_file = config_file # Store initial config path
        self.profiles_dir = profiles_dir # Store profiles directory path
        # Initialize plot_data_buffer with NaNs for a cleaner initial plot for level signal
        self.plot_data_buffer = np.full(PLOT_BUFFER_SIZE, np.nan) 
        self._pose_overlay_state_before_tracking = True # Default to True
        self.tracking_active = False # UI's understanding of tracking state

        self._init_ui()
        self._load_and_populate_initial_settings()
        # self.webcam_label.setText("Initializing Video...") # Custom widget doesn't use setText
        self.statusBar.showMessage("Initializing video source...")
        self._update_ui_state()
        # Ensure settings are hidden initially if toggle is unchecked
        # Connect the (hypothetical) worker signal to the new slot
        # self.worker.profile_saved_signal.connect(self.handle_profile_saved) # Placeholder
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
        self.top_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding) # Allow top section to expand fully
        main_layout.addWidget(self.top_widget, 1) # Stretch factor 1

        # --- MODIFICATION START: Use Custom VideoWidget ---
        # Replace QLabel with the custom VideoWidget
        self.webcam_label = VideoWidget()
        # Size policy is set within VideoWidget, but we ensure it's Expanding here too
        self.webcam_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.webcam_label.setMinimumHeight(300) # Keep a minimum height
        # Background color is handled by VideoWidget's paintEvent
        # --- MODIFICATION END ---

        # Add webcam widget to the top layout with a higher stretch factor
        top_layout.addWidget(self.webcam_label, 5) # Give more vertical space to webcam initially

        # Plot Widget Setup
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Samples')
        self.plot_widget.setTitle('Processed Absolute Level Signal', size=f'{int(new_point_size*1.05)}pt')
        # Define pens for different plot types
        # self.plot_curve_differential_pen = pg.mkPen('b', width=2) # Blue for differential - REMOVED
        self.plot_curve_level_pen = pg.mkPen('r', width=2)      # Red for level
        self.plot_curve = self.plot_widget.plot(pen=self.plot_curve_level_pen) # Always use level pen
        # Add plot widget to the top layout with a lower stretch factor
        top_layout.addWidget(self.plot_widget, 2) # Give less vertical space to plot initially

        # --- Bottom Section: Controls ---
        self.bottom_controls_widget = QWidget()
        # Prevent bottom controls from expanding vertically
        self.bottom_controls_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        main_layout.addWidget(self.bottom_controls_widget, 0) # Stretch factor 0

        bottom_main_layout = QVBoxLayout(self.bottom_controls_widget)
        bottom_main_layout.setContentsMargins(5, 5, 5, 5)
        bottom_main_layout.setSpacing(10)

        # --- Row 1: Status, Overlays, Profiles ---
        row1_layout = QHBoxLayout()
        bottom_main_layout.addLayout(row1_layout)

        # Status Group
        status_group = QGroupBox("Status")
        status_layout = QGridLayout(status_group)
        self.bpm_label = QLabel("BPM: ---")
        self.phase_label = QLabel("Phase: ---")
        self.osc_status_label = QLabel("OSC Status: ---")
        status_layout.addWidget(self.bpm_label, 0, 0)
        status_layout.addWidget(self.phase_label, 1, 0)
        status_layout.addWidget(self.osc_status_label, 2, 0)
        row1_layout.addWidget(status_group, 1) # Equal stretch factor

        # Overlays Group
        overlays_group = QGroupBox("Overlays")
        overlays_layout = QVBoxLayout(overlays_group)
        self.pose_overlay_check = QCheckBox("Show Pose")
        self.roi_overlay_check = QCheckBox("Show ROI")
        self.features_overlay_check = QCheckBox("Show Features")
        self.pose_overlay_check.setChecked(True)
        self.roi_overlay_check.setChecked(True)
        self.features_overlay_check.setChecked(False)
        # --- MODIFIED: Disconnect signals initially ---
        # self.pose_overlay_check.stateChanged.connect(self._emit_overlay_settings)
        # self.roi_overlay_check.stateChanged.connect(self._emit_overlay_settings)
        # self.features_overlay_check.stateChanged.connect(self._emit_overlay_settings)
        overlays_layout.addWidget(self.pose_overlay_check)
        overlays_layout.addWidget(self.roi_overlay_check)
        overlays_layout.addWidget(self.features_overlay_check)
        row1_layout.addWidget(overlays_group, 1) # Equal stretch factor

        # Profile Management Group
        profile_group = QGroupBox("Profile Management")
        profile_layout = QVBoxLayout(profile_group)
        self.profile_combo = QComboBox()
        self.profile_combo.currentIndexChanged.connect(self._profile_selected)
        self._populate_profiles() # Populate before adding widgets that depend on it
        profile_layout.addWidget(self.profile_combo)

        self.load_button = QPushButton("Reload Selected Profile")
        save_button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_as_button = QPushButton("Save As...")
        save_button_layout.addWidget(self.save_button)
        save_button_layout.addWidget(self.save_as_button)

        self.load_button.clicked.connect(self._load_profile)
        self.save_button.clicked.connect(self._save_profile)
        self.save_as_button.clicked.connect(self._save_profile_as)

        # Disable buttons initially until components are ready
        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.save_as_button.setEnabled(False)

        profile_layout.addWidget(self.load_button)
        profile_layout.addLayout(save_button_layout)
        row1_layout.addWidget(profile_group, 1) # Equal stretch factor

        # --- Row 2: Tracking Buttons (Centered) ---
        row2_layout = QHBoxLayout()
        bottom_main_layout.addLayout(row2_layout)
        row2_layout.addStretch(1) # Push buttons to center
        self.track_button = QPushButton("Start Tracking")
        self.track_button.setCheckable(True)
        self.track_button.setFixedHeight(int(40 * 1.1)) # Slightly taller button
        # --- ADDED: Horizontal padding for track button ---
        self.track_button.setStyleSheet("padding-left: 15px; padding-right: 15px;")
        self.track_button.toggled.connect(self._handle_track_button_toggle)
        self.track_button.setEnabled(False) # Disabled until components ready
        row2_layout.addWidget(self.track_button, 0) # No stretch factor for button

        self.reset_button = QPushButton("Reset Tracking")
        self.reset_button.setFixedHeight(int(40 * 1.1))
        # --- ADDED: Horizontal padding for reset button ---
        self.reset_button.setStyleSheet("padding-left: 15px; padding-right: 15px;")
        self.reset_button.clicked.connect(self._handle_reset_button_click)
        self.reset_button.setEnabled(False) # Disabled until tracking starts
        row2_layout.addWidget(self.reset_button, 0) # No stretch factor for button
        row2_layout.addStretch(1) # Push buttons to center

        # --- Row 3: Settings Toggle Button ---
        self.settings_toggle_button = QPushButton("▶ Show Settings")
        self.settings_toggle_button.setCheckable(True)
        self.settings_toggle_button.setChecked(False) # Start hidden
        # Basic styling for the toggle button
        self.settings_toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding-left: 5px;
                border: none; /* Flat look */
                font-weight: bold;
                background-color: transparent; /* No background */
            }
            QPushButton:checked {
                background-color: transparent; /* Keep transparent when checked */
                border: none;
            }
            QPushButton:hover {
                background-color: #eee; /* Slight highlight on hover */
            }
            QPushButton:pressed {
                background-color: #ddd; /* Darker when pressed */
            }
        """)
        self.settings_toggle_button.setFlat(True) # Reinforce flat look
        self.settings_toggle_button.toggled.connect(self._toggle_settings_visibility)
        self.settings_toggle_button.setEnabled(False) # Disabled until components ready
        bottom_main_layout.addWidget(self.settings_toggle_button)

        # --- Row 4: Settings Group Box (Initially Hidden) ---
        self.settings_main_group = QGroupBox()
        # --- RE-ADDED: Make the main container borderless ---
        self.settings_main_group.setStyleSheet("QGroupBox { border: none; margin-top: 0px; padding-top: 0px; }")
        settings_main_layout = QVBoxLayout(self.settings_main_group)
        # Add margins around the whole settings section
        # --- MODIFIED: Remove top margin as container is borderless ---
        settings_main_layout.setContentsMargins(10, 0, 10, 10) # L, T=0, R, B
        # Spacing between the row of groups and the apply button row
        settings_main_layout.setSpacing(15)

        # --- Feature Tracking Settings Group ---
        # 1. Encapsulate in QGroupBox
        ft_group = QGroupBox("Feature Tracking (Lucas-Kanade)")

        # Use QFormLayout inside
        ft_layout = QFormLayout(ft_group)
        # --- MODIFIED: Adjust margins for default group box style ---
        ft_layout.setContentsMargins(10, 20, 10, 10) # L, T, R, B (Increased top margin for title)
        # Add vertical spacing between rows within this form
        # --- Explicitly add a border to ft_group ---
        ft_group.setStyleSheet("QGroupBox { border: 1px solid gray; margin-top: 0.5em; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        ft_layout.setVerticalSpacing(12) # Increased vertical spacing

        ft_group.setFont(larger_font) # Apply larger font to the group box itself (for title)

        self.maxCorners_spin = QSpinBox(minimum=10, maximum=500)
        self.qualityLevel_spin = QDoubleSpinBox(minimum=0.01, maximum=0.99, singleStep=0.01, decimals=2)
        self.minDistance_spin = QSpinBox(minimum=1, maximum=50)
        self.winSize_spin = QSpinBox(minimum=3, maximum=51, singleStep=2) # Must be odd
        self.maxLevel_spin = QSpinBox(minimum=0, maximum=8)

        # Moved from sp_group
        self.aggMethod_combo = QComboBox()
        self.aggMethod_combo.addItems(["median", "mean"])

        # --- ADDED: Tooltips for Feature Tracking ---
        self.maxCorners_spin.setToolTip("Maximum number of feature points to detect and track.")
        self.qualityLevel_spin.setToolTip("Minimum acceptable quality of feature points (0.01-0.99). Higher values mean stricter filtering.")
        self.minDistance_spin.setToolTip("Minimum Euclidean distance between detected feature points.")
        self.winSize_spin.setToolTip("Size of the search window (pixels) for Lucas-Kanade optical flow.")
        self.maxLevel_spin.setToolTip("Maximum level for the image pyramid used in Lucas-Kanade (0 means only original image).")
        # --- END TOOLTIPS ---
        self.aggMethod_combo.setToolTip("Method to aggregate multiple feature movements into a single signal.")

        ft_layout.addRow("Max Corners:", self.maxCorners_spin)
        ft_layout.addRow("Quality Level:", self.qualityLevel_spin)
        ft_layout.addRow("Min Distance:", self.minDistance_spin)
        ft_layout.addRow("Window Size:", self.winSize_spin)
        ft_layout.addRow("Pyramid Levels:", self.maxLevel_spin)

        # Apply larger font and fixed height to FT input widgets
        for widget in [self.maxCorners_spin, self.qualityLevel_spin, self.minDistance_spin,
                       self.winSize_spin, self.maxLevel_spin, self.aggMethod_combo]: # Added aggMethod_combo
            widget.setFont(larger_font)
            widget.setFixedHeight(28) # Optional: Adjust height

        settings_main_layout.addWidget(ft_group) # Add ft_group directly

        # --- Level Signal Processing Settings (New Group) ---
        self.lsp_group = QGroupBox("Level Signal Processing") # Store as instance member
        self.lsp_group.setFont(larger_font)
        # --- ADDED: Explicit border style to match other groups ---
        self.lsp_group.setStyleSheet("QGroupBox { border: 1px solid gray; margin-top: 0.5em; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        lsp_layout = QFormLayout(self.lsp_group)

        self.levelDriftCorrection_check = QCheckBox("Enable Drift Correction")
        self.levelDriftCorrection_check.setFont(larger_font)
        self.levelDriftCorrection_check.setToolTip("Enable subtraction of slow baseline (drift) before normalization.")
        self.levelDriftCorrection_check.setFixedHeight(28)

        self.levelLightEma_check = QCheckBox("Enable Initial Light EMA Smoothing")
        self.levelLightEma_check.setFont(larger_font)
        self.levelLightEma_check.setToolTip("Apply a light EMA filter to the raw level signal before baseline removal and normalization. Helps reduce noise.")
        self.levelLightEma_check.setFixedHeight(28)
        self.levelLightEmaAlpha_spin = QDoubleSpinBox(minimum=0.01, maximum=0.99, singleStep=0.01, value=0.75, decimals=2) # MODIFIED: Default value from 0.6 to 0.75
        self.levelLightEmaAlpha_spin.setFont(larger_font)
        self.levelLightEmaAlpha_spin.setToolTip("Alpha for initial light EMA (0.01-0.99). Higher alpha = less smoothing. Lower alpha = more smoothing.")
        self.levelLightEmaAlpha_spin.setFixedHeight(28)

        self.levelBaselineEmaAlpha_spin = QDoubleSpinBox(minimum=0.0001, maximum=0.1, singleStep=0.0001, value=0.002, decimals=4)
        self.levelBaselineEmaAlpha_spin.setFont(larger_font)
        self.levelBaselineEmaAlpha_spin.setToolTip("Alpha for slow baseline tracking EMA (0.0001-0.1). Smaller is slower.")
        self.levelBaselineEmaAlpha_spin.setFixedHeight(28)
        self.levelNormWindow_spin = QDoubleSpinBox(minimum=1.0, maximum=60.0, singleStep=1.0, value=25.0, decimals=1) # MODIFIED: Default value from 15.0 to 25.0
        self.levelNormWindow_spin.setFont(larger_font)
        self.levelNormWindow_spin.setToolTip("Normalization window duration in seconds (1-60s). Longer window provides more stable normalization over varying breath amplitudes but adapts slower to overall signal level changes.")
        self.levelNormWindow_spin.setFixedHeight(28)

        self.levelNormToMinusOne_check = QCheckBox("Normalize to [-1, 1]")
        self.levelNormToMinusOne_check.setFont(larger_font)
        self.levelNormToMinusOne_check.setChecked(True)
        self.levelNormToMinusOne_check.setToolTip("If checked, normalizes to [-1, 1]. If unchecked, normalizes to [0, 1].")
        self.levelNormToMinusOne_check.setFixedHeight(28)

        self.levelNormEpsilon_spin = QDoubleSpinBox(minimum=1e-6, maximum=1.0, singleStep=1e-3, value=0.01, decimals=6)
        self.levelNormEpsilon_spin.setFont(larger_font)
        self.levelNormEpsilon_spin.setToolTip("Epsilon for normalization range (1e-6 to 1.0).")
        self.levelNormEpsilon_spin.setFixedHeight(28)

        self.levelAdaptiveNorm_check = QCheckBox("Enable Adaptive Bounds")
        self.levelAdaptiveNorm_check.setFont(larger_font)
        self.levelAdaptiveNorm_check.setToolTip("Enable adaptive expansion and decay of normalization bounds based on signal clipping.")
        self.levelAdaptiveNorm_check.setFixedHeight(28)
        self.levelAdaptiveHeadroom_spin = QDoubleSpinBox(minimum=1.0, maximum=1.5, singleStep=0.01, value=1.05, decimals=2)
        self.levelAdaptiveHeadroom_spin.setFont(larger_font)
        self.levelAdaptiveHeadroom_spin.setToolTip("Factor to expand bounds on clip (e.g., 1.05 for 5% headroom). Active if adaptive bounds enabled.")
        self.levelAdaptiveHeadroom_spin.setFixedHeight(28)
        self.levelAdaptiveDecay_spin = QDoubleSpinBox(minimum=0.9000, maximum=0.9999, singleStep=0.0001, value=0.9990, decimals=4)
        self.levelAdaptiveDecay_spin.setFont(larger_font)
        self.levelAdaptiveDecay_spin.setToolTip("Decay factor per frame for adaptive bounds towards window bounds (closer to 1 = slower decay). Active if adaptive bounds enabled.")
        self.levelAdaptiveDecay_spin.setFixedHeight(28)

        lsp_layout.addRow(self.levelDriftCorrection_check)
        lsp_layout.addRow(self.levelLightEma_check)
        lsp_layout.addRow("Level Light EMA Alpha:", self.levelLightEmaAlpha_spin)
        lsp_layout.addRow("Level Baseline EMA Alpha:", self.levelBaselineEmaAlpha_spin)
        lsp_layout.addRow("Level Norm. Window (s):", self.levelNormWindow_spin)
        lsp_layout.addRow(self.levelNormToMinusOne_check)
        lsp_layout.addRow("Level Norm. Epsilon:", self.levelNormEpsilon_spin)
        lsp_layout.addRow(self.levelAdaptiveNorm_check)
        lsp_layout.addRow("Adaptive Headroom:", self.levelAdaptiveHeadroom_spin)
        lsp_layout.addRow("Adaptive Decay:", self.levelAdaptiveDecay_spin)
        
        # Add this new group to your main settings layout, e.g., settings_layout
        # settings_layout.addWidget(self.lsp_group) # Or wherever it fits best
        # Add to settings_main_layout (the QVBoxLayout of self.settings_main_group)
        settings_main_layout.addWidget(self.lsp_group)

        # Connect signals for enabling/disabling level processing widgets
        # self.calcLevelSignal_check removed, level signal is always processed
        self.levelDriftCorrection_check.toggled.connect(self._update_level_signal_widgets_state_main_ui)
        self.levelLightEma_check.toggled.connect(self._update_level_signal_widgets_state_main_ui)
        self.levelAdaptiveNorm_check.toggled.connect(self._update_level_signal_widgets_state_main_ui) # Connect new checkbox
        self._update_level_signal_widgets_state_main_ui() # Initial state

        # --- BPM & Phase Detection Settings Group (New) --- MOVE INITIALIZATION UP
        self.bpm_phase_group = QGroupBox("BPM & Phase Detection")
        self.bpm_phase_group.setFont(larger_font)
        self.bpm_phase_group.setStyleSheet("QGroupBox { border: 1px solid gray; margin-top: 0.5em; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        bpm_phase_layout = QFormLayout(self.bpm_phase_group)
        
        self.phaseBpmLevelSmooth_check = QCheckBox("Smooth Raw Level for BPM/Phase")
        self.phaseBpmLevelSmooth_check.setToolTip("Enable light EMA smoothing on the raw level signal before BPM/Phase detection.")
        
        self.phaseBpmLevelEmaAlpha_spin = QDoubleSpinBox(minimum=0.01, maximum=0.99, singleStep=0.01, value=0.5, decimals=2)
        self.phaseBpmLevelEmaAlpha_spin.setToolTip("EMA Alpha for smoothing raw level signal (if enabled).")
        
        self.phaseSlopeInhale_spin = QDoubleSpinBox(minimum=0.0001, maximum=1.0, singleStep=0.001, value=0.01, decimals=4)
        self.phaseSlopeInhale_spin.setToolTip("Positive slope threshold on raw level derivative for inhale detection.")
        
        self.phaseSlopeExhale_spin = QDoubleSpinBox(minimum=-1.0, maximum=-0.0001, singleStep=0.001, value=-0.01, decimals=4)
        self.phaseSlopeExhale_spin.setToolTip("Negative slope threshold on raw level derivative for exhale detection.")
        
        self.levelSignalBpmPeakProm_spin = QDoubleSpinBox(minimum=0.0, maximum=100.0, singleStep=0.1, value=10.0, decimals=2)
        self.levelSignalBpmPeakProm_spin.setToolTip("Peak prominence for BPM detection on raw level signal. Units depend on raw level signal scale.")

        bpm_phase_layout.addRow(self.phaseBpmLevelSmooth_check)
        bpm_phase_layout.addRow("Raw Level EMA Alpha:", self.phaseBpmLevelEmaAlpha_spin)
        bpm_phase_layout.addRow("Inhale Slope Thresh (Raw Level):", self.phaseSlopeInhale_spin)
        bpm_phase_layout.addRow("Exhale Slope Thresh (Raw Level):", self.phaseSlopeExhale_spin)
        bpm_phase_layout.addRow("Raw Level BPM Peak Prom.:", self.levelSignalBpmPeakProm_spin)

        for widget in [self.phaseBpmLevelSmooth_check, self.phaseBpmLevelEmaAlpha_spin,
                       self.phaseSlopeInhale_spin, self.phaseSlopeExhale_spin, self.levelSignalBpmPeakProm_spin]:
            widget.setFont(larger_font)
            widget.setFixedHeight(28)

        # Add BPM/Phase group to the main settings layout, below the lsp_group
        settings_main_layout.addWidget(self.bpm_phase_group)

        # Connect signals for enabling/disabling BPM/Phase widgets
        self.phaseBpmLevelSmooth_check.toggled.connect(self._update_bpm_phase_widgets_state_main_ui)
        self._update_bpm_phase_widgets_state_main_ui() # Initial state

        # Apply Settings Button (Centered)
        self.apply_button = QPushButton("Apply Settings")
        # --- ADDED: Padding for apply button ---
        self.apply_button.setStyleSheet("padding: 5px 10px;") # 5px top/bottom, 10px left/right
        self.apply_button.clicked.connect(self._gather_and_apply_settings)
        self.apply_button.setEnabled(False)
        # Add button to the main settings layout, centered horizontally
        apply_button_layout = QHBoxLayout()
        # --- MODIFIED: Add stretch on both sides to center ---
        apply_button_layout.addStretch(1)
        apply_button_layout.addWidget(self.apply_button)
        apply_button_layout.addStretch(1)
        settings_main_layout.addLayout(apply_button_layout) # Add below the groups row

        # --- Call toggle settings AFTER all widgets are created ---
        # This ensures the settings group exists before trying to hide/show it.
        self._toggle_settings_visibility(self.settings_toggle_button.isChecked())

        # --- Status Bar ---
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)


    def _update_main_plot_button_text(self):
        self.toggle_main_plot_button.setText("Show Filtered Differential" if self.main_plot_shows_level else "Show Processed Level Signal")

    def _update_gui_elements(self):
        """Updates GUI elements with data from the pipeline."""
        data = None
        if hasattr(self, 'pipeline_data') and self.pipeline_data is not None:
            data = self.pipeline_data
        
        # Update video feed (assuming this part is okay)
        annotated_frame = data.get("annotated_frame") if data is not None else None
        if annotated_frame is not None:
            try:
                if annotated_frame.ndim == 3 and annotated_frame.shape[2] == 3:
                    h, w, ch = annotated_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(bytes(annotated_frame.data), w, h, bytes_per_line, QImage.Format.Format_BGR888)
                elif annotated_frame.ndim == 2:
                    h, w = annotated_frame.shape
                    bytes_per_line = w
                    qt_image = QImage(bytes(annotated_frame.data), w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
                else:
                    qt_image = None
                if qt_image and not qt_image.isNull():
                    self.webcam_label.update_frame(qt_image)
                else:
                    self.webcam_label.clear_frame()
            except Exception as e_img:
                print(f"Error updating video feed: {e_img}")
                self.webcam_label.clear_frame()
        # If annotated_frame is None, the logic within the 'if annotated_frame is not None:' block
        # (specifically, qt_image being None or image.isNull()) will handle clearing the frame.
        # Plot title and pen are set in _init_ui and no longer change
        self.plot_widget.setLabel('bottom', 'Samples (Rolling Window)') # Ensure label is correct

        # For processed level signal, Y-axis is fixed based on normalization, X-axis is fixed for buffer
        self.plot_widget.getPlotItem().getViewBox().disableAutoRange(axis=pg.ViewBox.XAxis)
        self.plot_widget.setXRange(0, PLOT_BUFFER_SIZE - 1, padding=0.02)

        # Fetch PROCESSED level signal
        processed_level_signal_value = data.get("processed_level_signal") if data is not None else None

        self.plot_data_buffer = np.roll(self.plot_data_buffer, -1) # Roll buffer and ASSIGN BACK
        if processed_level_signal_value is not None and np.isfinite(processed_level_signal_value): # Ensure value is finite
            self.plot_data_buffer[-1] = float(processed_level_signal_value) # Ensure it's a float
        else:
            # If no new valid level signal value, roll in a NaN to represent missing data.
            self.plot_data_buffer[-1] = np.nan
        
        x_axis_raw_level_plot = np.arange(len(self.plot_data_buffer)) # Indices for the rolling buffer
        self.plot_curve.setData(x_axis_raw_level_plot, self.plot_data_buffer)

        # Set Y-axis range based on normalization target
        normalize_to_minus_one_one = data.get('level_normalize_to_minus_one_one', True) if data is not None else True
        if normalize_to_minus_one_one:
            self.plot_widget.setYRange(-1.1, 1.1, padding=0)
        else:
            self.plot_widget.setYRange(-0.1, 1.1, padding=0)
        self.plot_widget.getPlotItem().getViewBox().disableAutoRange(axis=pg.ViewBox.YAxis) # Ensure Y-axis auto-range is off

        # Update BPM display and status bar (only if data was available)
        if data:
            bpm = data.get("bpm", 0.0)
            bpm_is_valid = data.get("bpm_valid", False)
            phase = data.get("phase", SignalProcessor.PHASE_UNKNOWN) 
            self.update_status_labels(bpm, bpm_is_valid, phase)

            phase_str = "Inhale" if phase == SignalProcessor.PHASE_INHALE else "Exhale" if phase == SignalProcessor.PHASE_EXHALE else "---"
            current_plot_value_display = 0.0
            current_plot_value_display = self.plot_data_buffer[-1] # Always from level signal buffer now
            
            status_bar_text = f"BPM: {bpm:.1f} ({'Valid' if bpm_is_valid else 'Invalid'}) | Phase: {phase_str} | Plot Val: {current_plot_value_display:.2f}"
            if hasattr(self, 'statusBar') and self.statusBar:
                self.statusBar.showMessage(status_bar_text)
        else: # If no data, clear status or set to default
            self.update_status_labels(0.0, False, SignalProcessor.PHASE_UNKNOWN)
            if hasattr(self, 'statusBar') and self.statusBar:
                self.statusBar.showMessage("Status: N/A (No data)")

    def _load_and_populate_initial_settings(self):
        """Loads settings from the initial config file (if provided) and populates widgets."""
        initial_config = {}
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    initial_config = json.load(f)
                print(f"[UI Init] Loaded initial settings from: {self.config_file}")
            except Exception as e:
                print(f"[UI Init] Error loading initial config '{self.config_file}': {e}")
                QMessageBox.warning(self, "Config Error",
                                    f"Could not load initial profile:\n{self.config_file}\n\n{e}\n\nUsing default widget values.")
                initial_config = {} # Fallback to defaults
        else:
            print("[UI Init] No initial config file specified or found. Using default widget values.")

        # Populate widgets using the loaded or default config
        self.populate_settings_widgets(initial_config)


    def _populate_profiles(self):
        """Clears and repopulates the profile dropdown."""
        current_selection = self.profile_combo.currentText() # Remember selection if possible
        self.profile_combo.blockSignals(True) # Prevent signals during update
        self.profile_combo.clear()
        found_profiles = False

        if not self.profiles_dir or not os.path.isdir(self.profiles_dir):
            print(f"Warning: Profiles directory not found or not set: {self.profiles_dir}")
        else:
            try:
                profiles = sorted([f for f in os.listdir(self.profiles_dir) if f.lower().endswith('.json')])
                if profiles:
                    self.profile_combo.addItems(profiles)
                    found_profiles = True
                else:
                    print(f"No .json profiles found in {self.profiles_dir}")
            except Exception as e:
                print(f"Error populating profiles from {self.profiles_dir}: {e}")
                self.profile_combo.addItem("Error Loading Profiles") # Indicate error

        if not found_profiles and self.profile_combo.count() == 0:
            self.profile_combo.addItem("No Profiles Found")
            self.profile_combo.setEnabled(False) # Disable if none found/error
            self.profile_combo.blockSignals(False)
            return

        # Try to restore previous selection or select the initial config file
        target_profile = os.path.basename(self.config_file) if self.config_file else current_selection
        if not target_profile and found_profiles: # Default to first if no other target
             target_profile = self.profile_combo.itemText(0)

        index_to_set = self.profile_combo.findText(target_profile)

        if index_to_set != -1:
            self.profile_combo.setCurrentIndex(index_to_set)
        elif self.profile_combo.count() > 0:
            self.profile_combo.setCurrentIndex(0) # Fallback to first item

        self.profile_combo.setEnabled(True) # Ensure enabled if profiles were found
        self.profile_combo.blockSignals(False) # Re-enable signals


    def _profile_selected(self, index):
        """Updates the internal config_file path when a profile is selected."""
        selected_profile = self.profile_combo.itemText(index)
        # Basic check to ignore placeholder items
        if selected_profile and "No Profiles" not in selected_profile and "Error Loading" not in selected_profile:
             # Construct the full path
             self.config_file = os.path.join(self.profiles_dir, selected_profile)
             print(f"Selected profile for next load/save: {self.config_file}")
             # Update status bar (check if it exists first)
             if hasattr(self, 'statusBar') and self.statusBar:
                 self.statusBar.showMessage(f"Profile selected: {selected_profile}. Press 'Reload' or 'Save' to apply.")
             else:
                 print("Warning: statusBar not available when _profile_selected was called.")
        else:
            self.config_file = "" # Clear if an invalid item is selected
            print("Invalid profile selection.")


    def _load_profile(self):
        """Emits signal to load the currently selected profile."""
        if self.tracking_active:
            QMessageBox.warning(self, "Tracking Active", "Stop tracking before reloading profile.")
            return

        if self.config_file and os.path.exists(self.config_file):
            print(f"Signaling to load profile: {self.config_file}")
            self.statusBar.showMessage(f"Reloading {os.path.basename(self.config_file)}...")
            # Temporarily disable controls during load
            self.load_button.setEnabled(False)
            self.track_button.setEnabled(False)
            self.profile_combo.setEnabled(False)
            self.save_button.setEnabled(False)
            self.save_as_button.setEnabled(False)
            if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(False)
            # self.webcam_label.setText(f"Reloading profile:\n{os.path.basename(self.config_file)}...") # Can't setText on VideoWidget
            self.webcam_label.clear_frame() # Clear the video widget instead
            # TODO: Optionally display loading text in VideoWidget paintEvent if needed
            # Emit the signal with the full path
            self.load_profile_signal.emit(self.config_file)
        else:
            self.statusBar.showMessage("No valid profile selected to load.")
            QMessageBox.warning(self, "Load Error", "Please select a valid profile from the dropdown first.")


    def _save_profile(self):
        """Emits signal to save current settings to the selected profile."""
        if self.tracking_active:
            QMessageBox.warning(self, "Tracking Active", "Stop tracking before saving profile.")
            return

        current_profile_name = self.profile_combo.currentText()
        # Validate selection
        if not current_profile_name or "No Profiles" in current_profile_name or "Error Loading" in current_profile_name:
            QMessageBox.warning(self, "Save Error", "No valid profile selected to save to.")
            return

        save_path = os.path.join(self.profiles_dir, current_profile_name)

        # Confirmation dialog
        reply = QMessageBox.question(self, "Confirm Save",
                                     f"Overwrite profile '{current_profile_name}' with current settings?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No) # Default to No

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Signaling to save current settings to: {save_path}")
            # Emit signal with the target path (using save_profile_signal for consistency)
            self.save_profile_signal.emit(save_path)
            self.statusBar.showMessage(f"Save requested for {current_profile_name}.")
        else:
            self.statusBar.showMessage("Save cancelled.")


    def _save_profile_as(self):
        """Opens dialog to save current settings to a new profile file."""
        if self.tracking_active:
            QMessageBox.warning(self, "Tracking Active", "Stop tracking before saving profile.")
            return

        # Open 'Save As' dialog
        filePath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Profile As...",
            self.profiles_dir, # Start in the profiles directory
            "JSON Files (*.json)" # Filter for JSON files
        )

        if filePath: # Proceed only if a file path was chosen
            # Ensure it ends with .json
            if not filePath.lower().endswith('.json'):
                filePath += '.json'

            print(f"Signaling Save As to: {filePath}")
            # Emit signal with the new file path
            self.save_profile_signal.emit(filePath)
            self.statusBar.showMessage(f"Save As requested to {os.path.basename(filePath)}.")

            # --- REMOVE REFRESH LOGIC FROM HERE ---
            # self._populate_profiles()
            # new_profile_name = os.path.basename(filePath)
            # index = self.profile_combo.findText(new_profile_name)
            # if index != -1:
            #     self.profile_combo.setCurrentIndex(index)
            # else:
            #     print(f"Warning: Could not automatically select new profile '{new_profile_name}' in dropdown.")
            # --- END REMOVAL ---

    # --- ADD NEW SLOT ---
    def handle_profile_saved(self, file_path, success, message):
        """Handles the confirmation signal from the backend after a save attempt."""
        print(f"[UI] Received profile_saved confirmation: Path={file_path}, Success={success}, Msg={message}")
        if success:
            self.statusBar.showMessage(message, 3000)
            # Refresh the profile list now that we know the file exists
            self._populate_profiles()
            # Find and select the profile in the combo box
            profile_name = os.path.basename(file_path)
            index = self.profile_combo.findText(profile_name)
            if index != -1:
                self.profile_combo.setCurrentIndex(index)
                print(f"[UI] Automatically selected '{profile_name}' in dropdown.")
            else:
                # This warning is now more meaningful if it still occurs
                print(f"Warning: Could not automatically select profile '{profile_name}' in dropdown after successful save confirmation.")
        else:
            # Show error message if saving failed
            self.statusBar.showMessage(f"Save Failed: {message}", 5000)
            QMessageBox.critical(self, "Save Failed", message)


    def _handle_track_button_toggle(self, checked):
        """Handles the Start/Stop Tracking button state change."""
        if checked:
            self.tracking_active = True
            self.track_button.setText("Stop Tracking")
            # Store current pose overlay state before disabling it
            self._pose_overlay_state_before_tracking = self.pose_overlay_check.isChecked()
            # --- UI Changes on Tracking START ---
            self.pose_overlay_check.setChecked(False)
            self.pose_overlay_check.setEnabled(False) # Explicitly disable
            self.features_overlay_check.setChecked(True)
            # Emit overlay changes immediately so worker is aware
            self._emit_overlay_settings()
            # --- End UI Changes ---
            print("[UI] Start Tracking signal emitted.")
            self.start_tracking_signal.emit()
            self.statusBar.showMessage("Tracking started...")
        else:
            self.tracking_active = False
            self.track_button.setText("Start Tracking")
            # --- UI Changes on Tracking STOP ---
            self.pose_overlay_check.setEnabled(True) # Explicitly enable
            # Restore the pose overlay state from before tracking
            self.pose_overlay_check.setChecked(self._pose_overlay_state_before_tracking)
            # "Show Features" remains as is (likely checked from when tracking started)
            self._emit_overlay_settings() # Emit settings after restoring/enabling pose checkbox
            # --- End UI Changes ---
            # --- MODIFIED: Clear plot_data_buffer on stop ---
            self.plot_data_buffer.fill(np.nan) # Clear level signal buffer for next start
            self.pipeline_data = None # Clear pipeline data to signal no data available
            self.stop_tracking_signal.emit()
            # Reset plot and status when stopped
            # self.update_plot([]) # Old direct call, now handled by _update_gui_elements
            self.update_status_labels(0, False, SignalProcessor.PHASE_UNKNOWN)
            self._update_gui_elements() # Force GUI update, which will clear plot due to no pipeline_data
            self.webcam_label.clear_frame() # Clear video widget when stopping
            self.statusBar.showMessage("Previewing... Adjust position and press 'Start Tracking'.")

        # Update UI element enabled states based on tracking status
        self._update_ui_state()


    def _handle_reset_button_click(self):
        """Handles the Reset Tracking button click."""
        if self.tracking_active:
            print("[UI] Reset Tracking button clicked.")
            self.statusBar.showMessage("Resetting tracking...")
            self.reset_tracking_signal.emit()
            # Briefly disable the reset button to prevent spamming
            self.reset_button.setEnabled(False)
            QTimer.singleShot(1000, lambda: self.reset_button.setEnabled(self.tracking_active)) # Re-enable after 1 sec if still tracking
        else:
            print("[UI] Reset Tracking clicked but not tracking.")
            self.statusBar.showMessage("Reset only works while tracking is active.")


    def _emit_overlay_settings(self):
        """Gathers overlay checkbox states and emits the signal."""
        show_pose = self.pose_overlay_check.isChecked()
        show_roi = self.roi_overlay_check.isChecked()
        show_features = self.features_overlay_check.isChecked()
        current_thread_id = threading.get_ident() # For debugging threading issues
        print(f"[UI Emit - Thread: {current_thread_id}] Emitting overlay_settings_changed: Pose={show_pose}, ROI={show_roi}, Features={show_features}")
        self.overlay_settings_changed.emit(show_pose, show_roi, show_features)


    def _gather_and_apply_settings(self):
        """Gathers settings from UI widgets and emits the apply_settings_signal."""
        if self.tracking_active:
            QMessageBox.warning(self, "Tracking Active", "Stop tracking before applying new settings.")
            return

        # Construct the nested dictionary matching the expected backend structure
        settings = {
            'feature_tracker': {
                'OPTICAL_FLOW_PARAMS': {
                    'feature_params': {
                        'maxCorners': self.maxCorners_spin.value(),
                        'qualityLevel': self.qualityLevel_spin.value(),
                        'minDistance': self.minDistance_spin.value(),
                        'blockSize': 7 # Assuming fixed blockSize, or add a widget if needed
                    },
                    'lk_params': {
                        'winSize': [self.winSize_spin.value(), self.winSize_spin.value()], # Use same value for width/height
                        'maxLevel': self.maxLevel_spin.value(),
                        'criteria': [3, 10, 0.03] # Assuming fixed criteria (type, max_iter, epsilon)
                    }
                },
                # Calculate redetect threshold based on maxCorners
                'FEATURE_REDETECT_THRESHOLD': int(self.maxCorners_spin.value() * 0.7)
            },
            'signal_generator': {
                'SIGNAL_AGGREGATION_METHOD': self.aggMethod_combo.currentText(),
                'CALCULATE_LEVEL_SIGNAL': True # Always True now
            },
            'signal_processor': {
                # Differential signal filter parameters removed
                # Level signal processing settings for SignalProcessor
                'PROCESS_LEVEL_SIGNAL_ENABLED': True, # Permanently enabled
                'LEVEL_SIGNAL_DRIFT_CORRECTION_ENABLED': self.levelDriftCorrection_check.isChecked(),
                'LEVEL_SIGNAL_LIGHT_EMA_ALPHA': self.levelLightEmaAlpha_spin.value() if self.levelLightEma_check.isChecked() else None,
                'LEVEL_SIGNAL_BASELINE_EMA_ALPHA': self.levelBaselineEmaAlpha_spin.value(),
                'LEVEL_SIGNAL_NORMALIZATION_WINDOW_SECONDS': self.levelNormWindow_spin.value(),
                'LEVEL_SIGNAL_NORMALIZE_TO_MINUS_ONE_ONE': self.levelNormToMinusOne_check.isChecked(),
                'LEVEL_SIGNAL_NORMALIZATION_EPSILON': self.levelNormEpsilon_spin.value(),
                'LEVEL_SIGNAL_ADAPTIVE_NORMALIZATION_ENABLED': self.levelAdaptiveNorm_check.isChecked(),
                'LEVEL_SIGNAL_ADAPTIVE_HEADROOM_FACTOR': self.levelAdaptiveHeadroom_spin.value(),
                'LEVEL_SIGNAL_ADAPTIVE_DECAY_FACTOR': self.levelAdaptiveDecay_spin.value(),
                # New BPM/Phase settings
                # 'SIGNAL_PROCESSOR_BPM_PHASE_SOURCE' key removed, source is always Raw Level
                'PHASE_BPM_LEVEL_SMOOTHING_ENABLED': self.phaseBpmLevelSmooth_check.isChecked(),
                'PHASE_BPM_LEVEL_EMA_ALPHA': self.phaseBpmLevelEmaAlpha_spin.value(),
                'PHASE_INHALE_SLOPE_THRESHOLD': self.phaseSlopeInhale_spin.value(),
                'PHASE_EXHALE_SLOPE_THRESHOLD': self.phaseSlopeExhale_spin.value(),
                'LEVEL_SIGNAL_BPM_PEAK_PROMINENCE': self.levelSignalBpmPeakProm_spin.value()
            }
        }

        print("[UI] Applying settings:", json.dumps(settings, indent=2)) # Pretty print for readability
        self.statusBar.showMessage("Applying settings... Restart tracking if needed.")
        self.apply_settings_signal.emit(settings)

        # Briefly disable Apply button after click
        self.apply_button.setEnabled(False)
        QTimer.singleShot(1000, lambda: self.apply_button.setEnabled(not self.tracking_active))


    def _update_ui_state(self):
        """Enables/disables UI elements based on tracking state and component readiness."""
        components_ready = self.track_button.isEnabled()
        is_previewing = not self.tracking_active
        settings_enabled = is_previewing and components_ready

        self.profile_combo.setEnabled(settings_enabled)
        self.load_button.setEnabled(settings_enabled)
        self.save_button.setEnabled(settings_enabled)
        self.save_as_button.setEnabled(settings_enabled)

        if hasattr(self, 'settings_main_group'): self.settings_main_group.setEnabled(settings_enabled)
        if hasattr(self, 'apply_button'): self.apply_button.setEnabled(settings_enabled)
        if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(components_ready)

        # Show Pose is enabled when components are ready AND not actively tracking (i.e., previewing)
        # This logic remains correct.
        self.pose_overlay_check.setEnabled(is_previewing and components_ready)
        self.roi_overlay_check.setEnabled(components_ready)
        self.features_overlay_check.setEnabled(self.tracking_active and components_ready)

        if hasattr(self, 'reset_button'): self.reset_button.setEnabled(self.tracking_active and components_ready)


    def _toggle_settings_visibility(self, checked):
        """Shows or hides the settings group box using setVisible."""
        if hasattr(self, 'settings_main_group'):
            self.settings_main_group.setVisible(checked)
            if checked:
                self.settings_toggle_button.setText("▼ Hide Settings")
            else:
                self.settings_toggle_button.setText("▶ Show Settings")


    # --- Slots for Backend Signals ---

    @pyqtSlot(dict)
    def handle_worker_output(self, results_dict: dict):
        """Receives the full results dictionary from the PipelineWorker and stores it."""
        # This is the central point where data from the backend updates the UI's data store.
        # print(f"[UI Main - Slot] Received worker output. Keys: {list(results_dict.keys())}") # Optional: Verbose Debug
        self.pipeline_data = results_dict # Store the new data
        self._update_gui_elements()      # Immediately update the GUI with the new data

    def update_webcam_feed(self, frame):
        """Receives a numpy frame, converts to QImage, and updates the VideoWidget."""
        try:
            if frame is None or frame.size == 0:
                self.webcam_label.clear_frame() # Clear widget if frame is invalid
                return

            # Get frame dimensions and format
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_format = None
            if ch == 3:
                qt_format = QImage.Format.Format_BGR888 # OpenCV default is BGR
            elif ch == 1:
                qt_format = QImage.Format.Format_Grayscale8
            else:
                print(f"Warning: Unexpected frame channel count: {ch}")
                self.webcam_label.clear_frame() # Clear on error
                # Optionally display error in status bar or log
                self.statusBar.showMessage(f"Error: Bad frame format (Channels: {ch})", 3000)
                return

            # --- FIX: Convert memoryview to bytes ---
            # Convert memoryview (frame.data) to bytes to ensure a copy is made for QImage
            qt_image = QImage(bytes(frame.data), w, h, bytes_per_line, qt_format) # Corrected line
            # --- END FIX ---

            # --- MODIFICATION START: Update VideoWidget ---
            # Pass the QImage to the custom widget's update method
            self.webcam_label.update_frame(qt_image)
            # --- MODIFICATION END ---

        except Exception as e:
            print(f"Error updating webcam feed: {e}")
            traceback.print_exc()
            self.webcam_label.clear_frame() # Clear widget on exception
            self.statusBar.showMessage(f"Error displaying frame: {e}", 3000)


    # def update_plot(self, plot_data):
    #     """Updates the plot widget with new data.
    #     DEPRECATED: Plot updates are now handled by _update_gui_elements using self.pipeline_data.
    #     """
    #     if plot_data is None: plot_data = []
    #     if not isinstance(plot_data, (list, np.ndarray)):
    #         print(f"Warning (old update_plot): Received invalid data type for plot: {type(plot_data)}")
    #         self.plot_curve.setData([])
    #         return
    #     try:
    #         data_len = len(plot_data)
    #         if data_len == 0: self.plot_curve.setData([]); return
    #         if data_len >= PLOT_BUFFER_SIZE: self.plot_data_buffer = np.array(plot_data[-PLOT_BUFFER_SIZE:])
    #         else: self.plot_data_buffer[:data_len] = plot_data; self.plot_data_buffer[data_len:] = 0
    #         self.plot_curve.setData(self.plot_data_buffer)
    #     except Exception as e: print(f"Error updating plot (old update_plot): {e}"); traceback.print_exc(); self.plot_curve.setData([])


    def update_status_labels(self, bpm, is_valid, phase):
        """Updates BPM and Phase labels."""
        try:
            bpm_text = f"BPM: {bpm:.1f}" if is_valid else "BPM: ---"
            self.bpm_label.setText(bpm_text)
            palette = self.bpm_label.palette(); color = QColor('green') if is_valid else QColor('red'); palette.setColor(QPalette.ColorRole.WindowText, color); self.bpm_label.setPalette(palette)
            phase_map = {SignalProcessor.PHASE_INHALE: "Inhale", SignalProcessor.PHASE_EXHALE: "Exhale", SignalProcessor.PHASE_UNKNOWN: "---"}
            phase_text = f"Phase: {phase_map.get(phase, 'Error')}"; self.phase_label.setText(phase_text)
        except Exception as e: print(f"Error updating status labels: {e}"); self.bpm_label.setText("BPM: Error"); self.phase_label.setText("Phase: Error")


    def update_osc_status(self, status_text, is_error=False):
         """Updates the OSC status label."""
         try:
            self.osc_status_label.setText(f"OSC Status: {status_text}")
            palette = self.osc_status_label.palette()
            # Use red if error, otherwise use the current text color (usually black/white depending on theme)
            color = QColor('red') if is_error else self.osc_status_label.palette().color(QPalette.ColorRole.WindowText)
            palette.setColor(QPalette.ColorRole.WindowText, color)
            self.osc_status_label.setPalette(palette)
         except Exception as e:
            print(f"Error updating OSC status label: {e}")
            self.osc_status_label.setText("OSC Status: Error")
            # Optionally set error color here too
            try: # Nested try-except for safety during error handling
                palette = self.osc_status_label.palette()
                palette.setColor(QPalette.ColorRole.WindowText, QColor('red'))
                self.osc_status_label.setPalette(palette)
            except Exception as pe:
                print(f"Error setting error color on OSC status label: {pe}")


    def show_error_message(self, message):
        """Displays an error message in the status bar."""
        print(f"[UI Received Error]: {message}")
        self.statusBar.showMessage(f"Error: {message}", 5000)


    def handle_worker_setup_finished(self, success, message):
        """Handles the result of the initial worker setup (e.g., video source)."""
        print(f"[UI] Worker initial setup finished. Success: {success}, Message: {message}")
        if success:
            self.statusBar.showMessage("Initializing components (PoseDetector)...")
            # self.webcam_label.setText("Initializing components...") # Not applicable to VideoWidget
        else:
            fail_msg = f"Video/Config Setup Failed:\n{message}"
            # self.webcam_label.setText(fail_msg) # Not applicable
            # TODO: Could add a state to VideoWidget to display error text in paintEvent
            self.statusBar.showMessage(f"Video/Config Setup Failed: {message}")
            self.track_button.setEnabled(False); self.load_button.setEnabled(False); self.save_button.setEnabled(False); self.save_as_button.setEnabled(False); self.profile_combo.setEnabled(False)
            if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(False)
            self._update_ui_state()


    def handle_component_initialized(self, component_name, success, message):
        """Handles the initialization result of individual backend components."""
        print(f"[UI] Component Initialized: {component_name}, Success: {success}, Msg: {message}")
        if success:
            self.statusBar.showMessage(f"Initialized {component_name}...")
            if component_name == "PipelineManager": # Assuming PipelineManager is the last one
                self.statusBar.showMessage("Ready. Press 'Start Tracking'.", 5000)
                # self.webcam_label.setText("Waiting for webcam feed...") # Not applicable
                self.track_button.setEnabled(True)
                if self.profile_combo.count() > 0 and "No Profiles" not in self.profile_combo.itemText(0):
                     self.profile_combo.setEnabled(True); self.load_button.setEnabled(True); self.save_button.setEnabled(True); self.save_as_button.setEnabled(True)
                else:
                     self.profile_combo.setEnabled(False); self.load_button.setEnabled(False); self.save_button.setEnabled(False); self.save_as_button.setEnabled(False)
                if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(True)
                self._update_ui_state()

                self._emit_overlay_settings() # Emit initial state once

                # --- MODIFIED: Reconnect checkbox signals AFTER initial emit ---
                try:
                    self.pose_overlay_check.stateChanged.disconnect(self._emit_overlay_settings)
                    self.roi_overlay_check.stateChanged.disconnect(self._emit_overlay_settings)
                    self.features_overlay_check.stateChanged.disconnect(self._emit_overlay_settings)
                except TypeError: # Signals might not have been connected yet (safety)
                    pass
                self.pose_overlay_check.stateChanged.connect(self._emit_overlay_settings)
                self.roi_overlay_check.stateChanged.connect(self._emit_overlay_settings)
                self.features_overlay_check.stateChanged.connect(self._emit_overlay_settings)
                # --- END MODIFICATION ---

                print("[UI] All components initialized. UI controls enabled.")
        else:
            fail_msg = f"{component_name} Initialization Failed: {message}"
            # self.webcam_label.setText(fail_msg) # Not applicable
            self.statusBar.showMessage(fail_msg)
            self.track_button.setEnabled(False)
            self.profile_combo.setEnabled(True); self.load_button.setEnabled(True) # Allow trying other profiles
            self.save_button.setEnabled(False); self.save_as_button.setEnabled(False)
            if hasattr(self, 'settings_toggle_button'): self.settings_toggle_button.setEnabled(False)
            self._update_ui_state()


    def populate_settings_widgets(self, settings_dict):
        """Populates the settings widgets from a dictionary (e.g., loaded from JSON)."""
        print("[UI] Populating settings widgets with:", json.dumps(settings_dict, indent=2))
        try:
            # --- Default empty dicts for robustness ---
            sg_settings = settings_dict.get('signal_generator', {})
            sp_settings = settings_dict.get('signal_processor', {})
            # osc_settings = settings_dict.get('osc_sender', {})
            video_settings = settings_dict.get('video_input', {})
            feature_tracker_settings = settings_dict.get('feature_tracker', {}); optical_flow_params = feature_tracker_settings.get('OPTICAL_FLOW_PARAMS', {}); feature_params = optical_flow_params.get('feature_params', {}); lk_params = optical_flow_params.get('lk_params', {})
            signal_generator_settings = settings_dict.get('signal_generator', {}); signal_processor_settings = settings_dict.get('signal_processor', {})
            self.maxCorners_spin.setValue(feature_params.get('maxCorners', 100)); self.qualityLevel_spin.setValue(feature_params.get('qualityLevel', 0.3)); self.minDistance_spin.setValue(feature_params.get('minDistance', 7))
            win_size_val = lk_params.get('winSize', [15, 15]); self.winSize_spin.setValue(win_size_val[0] if isinstance(win_size_val, (list, tuple)) and len(win_size_val) > 0 else 15)
            self.maxLevel_spin.setValue(lk_params.get('maxLevel', 2))
            
            # Populate aggMethod_combo (moved to ft_group)
            agg_method = sg_settings.get('SIGNAL_AGGREGATION_METHOD', 'median'); self.aggMethod_combo.setCurrentText(agg_method)
            self.statusBar.showMessage("Settings populated.", 3000)

            # Populate Level Signal Processing widgets (from SignalProcessor settings)
            light_ema_alpha = sp_settings.get('LEVEL_SIGNAL_LIGHT_EMA_ALPHA', 0.75) # MODIFIED: Default fallback to 0.75
            self.levelLightEma_check.setChecked(light_ema_alpha is not None)
            self.levelLightEmaAlpha_spin.setValue(light_ema_alpha if light_ema_alpha is not None else 0.75) # MODIFIED: Default fallback to 0.75
            self.levelBaselineEmaAlpha_spin.setValue(sp_settings.get('LEVEL_SIGNAL_BASELINE_EMA_ALPHA', 0.002))
            self.levelNormWindow_spin.setValue(sp_settings.get('LEVEL_SIGNAL_NORMALIZATION_WINDOW_SECONDS', 25.0)) # MODIFIED: Default fallback to 25.0
            self.levelNormToMinusOne_check.setChecked(sp_settings.get('LEVEL_SIGNAL_NORMALIZE_TO_MINUS_ONE_ONE', True))
            self.levelNormEpsilon_spin.setValue(sp_settings.get('LEVEL_SIGNAL_NORMALIZATION_EPSILON', 0.01))
            # Populate new adaptive normalization widgets
            self.levelAdaptiveNorm_check.setChecked(sp_settings.get('LEVEL_SIGNAL_ADAPTIVE_NORMALIZATION_ENABLED', False)) # Default to False
            self.levelAdaptiveHeadroom_spin.setValue(sp_settings.get('LEVEL_SIGNAL_ADAPTIVE_HEADROOM_FACTOR', 1.05))
            self.levelAdaptiveDecay_spin.setValue(sp_settings.get('LEVEL_SIGNAL_ADAPTIVE_DECAY_FACTOR', 0.999))
            
            # Populate new BPM/Phase detection widgets
            # phaseBpmSource_combo removed
            self.phaseBpmLevelSmooth_check.setChecked(sp_settings.get('PHASE_BPM_LEVEL_SMOOTHING_ENABLED', False))
            self.phaseBpmLevelEmaAlpha_spin.setValue(sp_settings.get('PHASE_BPM_LEVEL_EMA_ALPHA', 0.5))
            self.phaseSlopeInhale_spin.setValue(sp_settings.get('PHASE_INHALE_SLOPE_THRESHOLD', 0.01))
            self.phaseSlopeExhale_spin.setValue(sp_settings.get('PHASE_EXHALE_SLOPE_THRESHOLD', -0.01))
            self.levelSignalBpmPeakProm_spin.setValue(sp_settings.get('LEVEL_SIGNAL_BPM_PEAK_PROMINENCE', 10.0))


            # Update enabled state of filter params based on loaded method
            self._update_level_signal_widgets_state_main_ui() # Update for level signal group
            self._update_bpm_phase_widgets_state_main_ui() # Update for BPM/Phase group
            # _update_filter_param_widgets_state_main_ui() removed
        except Exception as e: print(f"Error populating settings widgets: {e}"); traceback.print_exc(); self.statusBar.showMessage("Error loading settings into UI.", 3000); QMessageBox.warning(self, "Settings Error", f"Could not fully populate settings widgets:\n{e}")

    # _update_filter_param_widgets_state_main_ui removed

    def _update_level_signal_widgets_state_main_ui(self):
        """Enables/disables level signal parameter widgets in the main UI."""
        # Level signal processing is always enabled conceptually.
        group_enabled = True # self.calcLevelSignal_check.isChecked() -> True
        self.lsp_group.setEnabled(group_enabled) # Enable/disable the whole groupbox

        if group_enabled: # Only update individual widgets if group is enabled
            drift_correction_ui_enabled = self.levelDriftCorrection_check.isChecked()
            light_ema_ui_enabled = self.levelLightEma_check.isChecked()

            adaptive_norm_enabled = self.levelAdaptiveNorm_check.isChecked()

            self.levelDriftCorrection_check.setEnabled(True) # Always enabled if group is enabled
            self.levelLightEma_check.setEnabled(True)      # Always enabled if group is enabled
            self.levelLightEmaAlpha_spin.setEnabled(light_ema_ui_enabled)
            self.levelBaselineEmaAlpha_spin.setEnabled(drift_correction_ui_enabled)
            # Other widgets (norm window, norm target, epsilon) are always enabled if group is enabled
            self.levelAdaptiveNorm_check.setEnabled(True) # The checkbox itself is always enabled if group is
            self.levelAdaptiveHeadroom_spin.setEnabled(adaptive_norm_enabled)
            self.levelAdaptiveDecay_spin.setEnabled(adaptive_norm_enabled)
            self.levelNormWindow_spin.setEnabled(True)
            self.levelNormToMinusOne_check.setEnabled(True)
            self.levelNormEpsilon_spin.setEnabled(True)
        # If group_enabled is false, the setEnabled(False) on lsp_group handles children.

        # toggle_main_plot_button removed, no need to update its state.

    def _update_bpm_phase_widgets_state_main_ui(self):
        """Enables/disables BPM & Phase detection parameter widgets based on selections."""
        if not hasattr(self, 'bpm_phase_group'): # Widgets might not be initialized yet
            return

        # The whole group is enabled/disabled by settings_main_group.setEnabled()
        # phaseBpmSource_combo removed, so no logic depends on its state.

        # Enable/disable EMA alpha based on the smoothing checkbox

        self.phaseBpmLevelSmooth_check.setEnabled(True)
        smoothing_enabled_for_level = self.phaseBpmLevelSmooth_check.isChecked()
        self.phaseBpmLevelEmaAlpha_spin.setEnabled(smoothing_enabled_for_level)
        self.phaseSlopeInhale_spin.setEnabled(True); self.phaseSlopeExhale_spin.setEnabled(True); self.levelSignalBpmPeakProm_spin.setEnabled(True)

    def closeEvent(self, event):
        """Ensures tracking is stopped when the window is closed."""
        print("[UI] Close event triggered.")
        if self.tracking_active:
            print("[UI] Stopping tracking due to window close.")
            self.stop_tracking_signal.emit()
        event.accept()


# Example of running just the UI window for testing layout (without backend)
if __name__ == '__main__':
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
    script_dir = os.path.dirname(__file__); base_dir = os.path.abspath(os.path.join(script_dir, '..', '..')); PROFILES_DIR_TEST = os.path.join(base_dir, 'profiles')
    print(f"Testing UI - Profiles Dir: {PROFILES_DIR_TEST}")
    if not os.path.exists(PROFILES_DIR_TEST):
        try:
            os.makedirs(PROFILES_DIR_TEST)
            print(f"Created dummy profiles directory: {PROFILES_DIR_TEST}")
            dummy_profile_path = os.path.join(PROFILES_DIR_TEST, "test_profile.json")
            with open(dummy_profile_path, 'w') as f:
                json.dump({"info": "Dummy profile for UI testing"}, f)
            print(f"Created dummy profile: {dummy_profile_path}") # Moved to new line
        except Exception as e:
            print(f"Could not create dummy profiles directory/file: {e}")
    main_win = MainWindow(profiles_dir=PROFILES_DIR_TEST)
    main_win.handle_worker_setup_finished(True, "Simulated video source OK")
    main_win.handle_component_initialized("PoseDetector", True, "Simulated OK")
    main_win.handle_component_initialized("RoiCalculator", True, "Simulated OK"); # Added RoiCalculator init call
    main_win.handle_component_initialized("PipelineManager", True, "Simulated OK")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8); dummy_frame[:,:,1] = 100 # Greenish frame
    main_win.update_webcam_feed(dummy_frame)
    main_win.show()
    sys.exit(app.exec())
