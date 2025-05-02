# src/raw_data_recorder.py
"""
Dedicated process for recording raw signal and feature data received via a queue.
Saves data to a timestamped directory upon receiving a 'STOP' signal or termination.
"""

import time
import os
import numpy as np
import queue # For Empty exception
import signal
import sys
from datetime import datetime

# --- Configuration ---
RECORDINGS_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recordings'))

class DataRecorder:
    def __init__(self, data_queue, stop_event, run_dir_name):
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.run_dir_path = os.path.join(RECORDINGS_BASE_DIR, run_dir_name)
        self.recorded_timestamps = []
        self.recorded_signals = []
        self.recorded_coords = []
        self._running = True
        print(f"[Recorder] Initialized. Saving to: {self.run_dir_path}")

    def run(self):
        """Main loop to receive and store data."""
        while self._running:
            try:
                # Check for stop signal without blocking indefinitely
                if self.stop_event.is_set():
                    print("[Recorder] Stop event received.")
                    self._running = False
                    break

                # Get data from the queue with a timeout
                # Format expected: (timestamp, raw_signal, feature_coords_array)
                data_item = self.data_queue.get(timeout=0.1)

                if data_item == "STOP":
                    print("[Recorder] 'STOP' command received.")
                    self._running = False
                    break
                elif isinstance(data_item, tuple) and len(data_item) == 3:
                    timestamp, signal_val, coords = data_item
                    self.recorded_timestamps.append(timestamp)
                    self.recorded_signals.append(signal_val)
                    # Ensure coords are copied if they are numpy arrays
                    self.recorded_coords.append(coords.copy() if isinstance(coords, np.ndarray) else coords)
                else:
                    print(f"[Recorder] Warning: Received unexpected data format: {type(data_item)}")

            except queue.Empty:
                # Timeout occurred, just loop again to check stop_event
                continue
            except Exception as e:
                print(f"[Recorder] Error in run loop: {e}")
                # Decide if error is fatal
                # self._running = False # Optionally stop on error

        print("[Recorder] Run loop finished.")
        self.save_data()

    def save_data(self):
        """Saves the collected data to a .npz file."""
        if not self.recorded_signals:
            print("[Recorder] No data recorded, nothing to save.")
            return

        try:
            print(f"[Recorder] Saving {len(self.recorded_signals)} data points...")
            os.makedirs(self.run_dir_path, exist_ok=True)
            output_path = os.path.join(self.run_dir_path, "raw_pipeline_data.npz")

            # Use dtype=object for coordinates as the number of points can vary
            np.savez_compressed(
                output_path,
                timestamps=np.array(self.recorded_timestamps),
                raw_signal=np.array(self.recorded_signals),
                feature_coords=np.array(self.recorded_coords, dtype=object)
            )
            print(f"[Recorder] Data saved successfully to {output_path}")
        except Exception as e:
            print(f"[Recorder] Error saving data: {e}")

# Note: This script is intended to be run as a separate process,
# likely managed by main.py using the multiprocessing module.
# The main execution block is omitted as it won't be run directly typically.