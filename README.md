# Respiration Modulator M4L

**Turn your breathing into a control signal for Max for Live (or other OSC-compatible software) using computer vision.**

This application uses a standard webcam to monitor the user's respiration rate and phase (inhale/exhale) by analyzing subtle body movements. It processes the video feed in real-time and outputs the breathing information as Open Sound Control (OSC) messages, suitable for modulating parameters in creative software like Ableton Live's Max for Live devices.

This project builds upon concepts and analysis inspired by the original `respmon` project and subsequent performance analysis.

## Key Features

* **Real-time Respiration Tracking:** Detects breathing rate (BPM) and phase (inhale/exhale) from webcam video.
* **Computer Vision Pipeline:**
    * **Pose Estimation:** Uses MediaPipe Pose to detect key body landmarks (shoulders, hips).
    * **Region of Interest (ROI):** Automatically calculates an ROI on the chest/abdomen based on detected landmarks.
    * **Feature Tracking:** Employs Lucas-Kanade (LK) optical flow to track subtle movements of features within the ROI.
    * **Signal Generation:** Extracts a raw motion signal representing breathing (currently using median vertical displacement of tracked features).
    * **Signal Processing:** Filters the raw signal (Butterworth bandpass), detects peaks, and calculates BPM and phase.
* **Graphical User Interface (GUI):** Built with PyQt, providing:
    * Live webcam feed display.
    * Overlays for Pose, ROI, and Tracked Features (toggleable).
    * Live plot of the filtered respiratory signal.
    * Real-time display of BPM and breathing phase.
    * Interactive controls for adjusting tracking and signal processing parameters.
    * Profile management (loading/saving settings).
* **OSC Output:** Sends BPM and phase data via OSC messages to a configurable address and port (intended for Max for Live). *(Note: OSC sending implementation is planned but might not be fully integrated in the current worker thread yet).*
* **Configurable:** Uses JSON profiles (`profiles/`) to store and manage different parameter sets.

## Dependencies

The primary dependencies are listed in `requirements.txt`. Key libraries include:

* `opencv-python` (cv2)
* `mediapipe`
* `numpy`
* `scipy`
* `pyqt6` (or `pyqt5`)
* `pyqtgraph`
* `python-osc` (for OSC output)

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd Respiration-Modulator-M4L
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure `requirements.txt` accurately reflects all needed packages, including `PyQt6` and `pyqtgraph`)*

## Usage

1.  **Activate Virtual Environment** (if you created one):
    ```bash
    # Windows
    .\.venv\Scripts\activate
    # macOS / Linux
    source .venv/bin/activate
    ```
2.  **Run the Main Application:**
    ```bash
    python main.py
    ```
3.  **Position Yourself:** Ensure your upper body (shoulders, chest/abdomen) is clearly visible in the webcam feed displayed in the UI. The application will start in "Preview" mode.
4.  **Preview Mode:**
    * Observe the webcam feed.
    * Toggle the "Show Pose" and "Show ROI" checkboxes to see the detected landmarks and the calculated Region of Interest (cyan rectangle). Adjust your camera or seating position until the ROI covers the area where your breathing motion is most visible (usually upper abdomen/lower chest).
5.  **Start Tracking:**
    * Once you have a good ROI, click the "Start Tracking" button.
    * The ROI will lock (displayed as a green rectangle).
    * The application will begin tracking features within the ROI, processing the signal, and displaying the filtered signal plot, BPM, and phase.
    * You can now toggle "Show Features" to visualize the points being tracked (red dots).
6.  **Adjust Settings (Optional):**
    * Click the "▶ Show Settings" button to expand the settings panel.
    * Modify parameters for Feature Tracking, Signal Generation, or Signal Processing as needed.
    * Click "Apply Settings". **Note:** Applying settings currently stops tracking and requires you to click "Start Tracking" again.
7.  **Stop Tracking:** Click "Stop Tracking" to return to Preview mode.
8.  **Profile Management:**
    * Select a different settings profile from the dropdown.
    * Click "Reload Selected Profile" to load and apply its settings (tracking must be stopped).
    * Click "Save" to overwrite the currently selected profile with the settings currently configured in the UI widgets.
    * Click "Save As..." to save the current UI settings to a new profile file.
9.  **Quit:** Close the application window or press 'q' (if focus is on an OpenCV window, though not typically used with the PyQt UI).

## Configuration

* Settings are managed through JSON files located in the `profiles/` directory.
* `test_profile.json` is provided as a default starting point.
* You can create multiple profiles for different lighting conditions, users, or desired responsiveness.
* Key configurable sections within the JSON include:
    * `video_input`: Camera index, desired resolution/FPS.
    * `pose_detector`: MediaPipe model complexity, confidence thresholds.
    * `coarse_roi_calculator`: Strategy for calculating ROI from landmarks.
    * `feature_tracker`: Parameters for `goodFeaturesToTrack` and `calcOpticalFlowPyrLK`.
    * `signal_generator`: Method for aggregating feature motion (`median` or `mean`).
    * `signal_processor`: Filter type, cutoff frequencies, buffer sizes, peak detection parameters.
    * *(Add `osc_sender` section when implemented: IP address, port)*

## Troubleshooting / Notes

* **Camera Access:** Ensure no other application is using the webcam. If the default camera (index 0) doesn't work, the application attempts index 1. You might need to specify a different index in the `video_input` section of your profile.
* **Performance:** Real-time performance depends heavily on CPU resources.
    * Lowering `maxCorners` or `winSize` in feature tracking can improve FPS.
    * Using `filtfilt` for signal filtering is computationally more expensive than `lfilter`.
    * Ensure MediaPipe Pose complexity is set appropriately (0 is fastest).
* **OSC Output:** Verify the target IP address and port match the receiving application (e.g., Max for Live device). Ensure no firewall is blocking the connection.
* **Layout Issues:** If UI elements appear cut off, try resizing the window slightly.

## License

*(Placeholder: Specify your chosen license here, e.g., MIT, Apache 2.0, GPL, or indicate if it's proprietary)*

## Acknowledgements

* Inspired by the original `respmon` project.
* Utilizes Google's [MediaPipe](https://developers.google.com/mediapipe) library for pose estimation.
* Built with the [OpenCV](https://opencv.org/) library.
* GUI developed using [PyQt6](https://riverbankcomputing.com/software/pyqt/).
* Plotting powered by [pyqtgraph](http://www.pyqtgraph.org/).
* Uses [SciPy](https://scipy.org/) for signal processing.
* Uses [NumPy](https://numpy.org/) for numerical operations.
* *(Add python-osc acknowledgement when implemented)*

