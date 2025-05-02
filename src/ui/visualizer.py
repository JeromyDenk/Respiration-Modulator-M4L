# src/ui/visualizer.py
# Provides a simple visual representation (ring and ball) of the filtered signal.

import sys
import math
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QPaintEvent
from PyQt6.QtCore import Qt, QSize, pyqtSlot, QPointF

class SignalVisualizer(QWidget):
    """
    A widget that visualizes a normalized signal value (-1 to 1) using:
    - Left: A ring whose radius changes.
    - Right: A ball whose vertical position changes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Signal Visualizer")
        self.setMinimumSize(300, 200)
        self.setStyleSheet("background-color: black;")

        # Signal value, expected to be roughly normalized (-1 to 1 is ideal)
        self._signal_value = 0.0
        # Parameters for drawing
        self.ring_color = QColor(0, 150, 255, 200) # Light blue, semi-transparent
        self.ball_color = QColor(255, 255, 255) # White
        self.base_ring_radius_factor = 0.1 # Base radius as fraction of min widget dimension
        self.ring_scale_factor = 0.3 # How much radius changes with signal (fraction of min dim)
        self.ball_radius_factor = 0.08 # Ball radius as fraction of min widget dimension

    def sizeHint(self):
        return QSize(400, 250)

    @pyqtSlot(float)
    def update_signal(self, value: float):
        """
        Slot to receive the latest signal value.
        Clamps the value to -1 to 1 for visualization stability.
        """
        # Clamp the value to prevent extreme visuals if normalization is off
        self._signal_value = max(-1.0, min(1.0, value))
        self.update() # Trigger a repaint

    def paintEvent(self, event: QPaintEvent):
        """Draws the ring and the ball based on the current signal value."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        min_dim = min(width / 2, height) # Use half-width for calculations

        # --- Left Side: Ring ---
        ring_center_x = width * 0.25
        ring_center_y = height * 0.5

        # Map signal value (0 to 1, assuming inhale is positive) to radius change
        # Use absolute value or a specific mapping if phase matters differently
        # Let's map signal from -1 to 1 -> radius factor from 0 to 1
        normalized_signal_for_radius = (self._signal_value + 1.0) / 2.0 # Map -1..1 to 0..1

        base_radius = min_dim * self.base_ring_radius_factor
        dynamic_radius = min_dim * self.ring_scale_factor * normalized_signal_for_radius
        current_radius = base_radius + dynamic_radius

        # Draw the ring (as a filled circle with a hole, or just an outline)
        # Option 1: Outline
        # pen_width = max(2, int(min_dim * 0.02))
        # painter.setPen(QPen(self.ring_color, pen_width))
        # painter.setBrush(Qt.BrushStyle.NoBrush)
        # painter.drawEllipse(QPointF(ring_center_x, ring_center_y), current_radius, current_radius)

        # Option 2: Filled Circle (simpler visual for growth)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.ring_color))
        painter.drawEllipse(QPointF(ring_center_x, ring_center_y), current_radius, current_radius)


        # --- Right Side: Ball ---
        ball_center_x = width * 0.75
        ball_radius = min_dim * self.ball_radius_factor

        # Map signal value (-1 to 1) to vertical position (top to bottom)
        # Top margin/padding = ball_radius + small buffer
        # Bottom margin/padding = ball_radius + small buffer
        top_y = ball_radius + 10
        bottom_y = height - ball_radius - 10
        drawable_height = bottom_y - top_y

        if drawable_height <= 0: # Avoid division by zero if widget is too small
            return

        # Map signal from -1..1 (exhale..inhale) to 0..1
        normalized_signal_for_y = (self._signal_value + 1.0) / 2.0
        # Map 0..1 to bottom_y..top_y (inverted Y-axis)
        ball_center_y = bottom_y - (normalized_signal_for_y * drawable_height)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(self.ball_color))
        painter.drawEllipse(QPointF(ball_center_x, ball_center_y), ball_radius, ball_radius)


# Example usage (for testing this module directly)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    visualizer = SignalVisualizer()

    # --- Simple Test Data Simulation ---
    test_timer = app.instance().thread().create_timer() # Use QTimer from app thread
    phase = 0.0
    def update_test_signal():
        global phase
        # Simulate a sine wave signal between -1 and 1
        value = math.sin(phase)
        visualizer.update_signal(value)
        phase += 0.05 # Adjust speed of change
        if phase > 2 * math.pi:
            phase -= 2 * math.pi

    test_timer.setInterval(50) # Update ~20 times per second
    test_timer.timeout.connect(update_test_signal)
    # --- End Simulation ---

    visualizer.show()
    test_timer.start() # Start simulation
    sys.exit(app.exec())