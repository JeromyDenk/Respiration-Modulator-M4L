# scripts/test_ui.py
# Runs the PyQt UI (MainWindow) in isolation for testing and development,
# without starting the backend processing pipeline.
# MODIFIED: Manually enable buttons for UI interaction testing. Added traceback import.

import sys
import os
import traceback # <<< IMPORT TRACEBACK

# --- PyQt6 Imports ---
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt # Import Qt if needed for flags
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
# Assumes this script is in the 'scripts' directory, one level below root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) # Go up one level to root
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    print(f"Added '{src_dir}' to Python path.")

# --- Import UI Component ---
try:
    from ui.main_window import MainWindow # Import the UI class
except ImportError as e:
    print(f"Fatal Error: Failed to import MainWindow from 'src/ui/main_window.py': {e}")
    print("Please ensure the file exists and there are no import errors within it.")
    traceback.print_exc()
    sys.exit(1)
except Exception as e_general:
     print(f"An unexpected error occurred during UI import: {e_general}")
     traceback.print_exc()
     sys.exit(1)

# --- Constants ---
# Define PROFILES_DIR relative to the project root
PROFILES_DIR = os.path.join(project_root, "profiles")

# --- Main Application Setup ---
if __name__ == "__main__":
    print("Starting UI Test Script...")

    # --- Basic Application Setup ---
    app = QApplication(sys.argv)

    # --- Configure pyqtgraph (optional, but good practice) ---
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    # Enable anti-aliasing for smoother plots
    pg.setConfigOptions(antialias=True)

    # --- Check if Profiles Directory Exists ---
    if not os.path.exists(PROFILES_DIR):
         print(f"Warning: Profiles directory does not exist: '{PROFILES_DIR}'")
         # You might want to create it or handle this case more gracefully
         # os.makedirs(PROFILES_DIR, exist_ok=True)

    # --- Create and Show the Main Window ---
    try:
        # Pass the profiles directory path to the MainWindow constructor
        main_window = MainWindow(profiles_dir=PROFILES_DIR)
        print("MainWindow instance created.")

        # --- *** Manually Enable Controls for Testing *** ---
        print("Manually enabling UI controls for testing...")
        main_window.track_button.setEnabled(True)
        main_window.load_button.setEnabled(True)
        main_window.save_button.setEnabled(True)
        main_window.profile_combo.setEnabled(True)
        # Enable settings widgets by calling update_ui_state after enabling track_button
        main_window._update_ui_state()
        # Set a non-initializing status message
        main_window.statusBar.showMessage("UI Test Mode - Backend not running.")
        main_window.webcam_label.setText("UI Test Mode")
        # --- *** ---

        main_window.show()
        print("MainWindow shown.")
    except Exception as e_create:
         print(f"Fatal Error creating or enabling MainWindow: {e_create}")
         traceback.print_exc()
         sys.exit(1)


    # --- Start the Event Loop ---
    print("Starting application event loop...")
    sys.exit(app.exec())
