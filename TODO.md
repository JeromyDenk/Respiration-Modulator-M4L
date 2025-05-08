# TODO List for Signal Processing Tuner (`tune_signal_processing.py`)

## Bugs

-   **Zero Displacement Bug:** Investigate and fix the issue where feature displacements sometimes show as zero for all features between frames, resulting in a flat histogram and zero raw motion signal for that frame transition. This is visible in the histogram plot showing only a single bar at 0.

## Enhancements

### Histogram Plot

-   **Add Median/Mode Indicators:** Add vertical lines or other indicators to the live displacement histogram plot to show the calculated median and mode of the displacements for the current frame transition. This will help visualize the central tendency and compare it to the mean (which is implicitly related to the `mean` aggregation method).

### Raw Signal Generation (Outlier Removal)

Explore and potentially implement explicit outlier removal strategies before aggregating feature displacements into the raw motion signal. This could be added as an optional step controlled via the UI.

1.  **Statistical Filtering (IQR):**
    *   **How:** For each frame transition, calculate Q1 (25th percentile) and Q3 (75th percentile) of vertical displacements. Calculate IQR = Q3 - Q1. Define outliers as displacements outside `[Q1 - k*IQR, Q3 + k*IQR]` (e.g., k=1.5).
    *   **Why:** Robust statistical method, adapts to data spread in each frame.
    *   **Implementation:** Calculate per-frame in `_reprocess_data` before `np.mean`/`np.median`. Filter the `displacements` array. Add UI control for enabling/disabling and setting `k`.

2.  **Thresholding (Manual/Visual):**
    *   **How:** Observe histogram, set fixed upper/lower displacement thresholds (e.g., +/- 5 pixels). Remove displacements outside this range.
    *   **Why:** Simple, good if outliers have consistent magnitudes.
    *   **Implementation:** Add UI spin boxes for thresholds. Apply filtering in `_reprocess_data`.
    *   **Caution:** Not adaptive to varying motion amplitudes.

3.  **Sigma Clipping (Alternative Statistical):**
    *   **How:** Calculate mean and standard deviation (std dev) of displacements. Remove points outside `mean +/- k*std_dev` (e.g., k=2 or 3). Optionally iterate.
    *   **Why:** Standard method, works well for ~normally distributed data.
    *   **Implementation:** Calculate per-frame in `_reprocess_data`. Add UI control.
    *   **Caution:** Mean/std dev are sensitive to outliers, making it less robust than IQR initially.

## Code Quality / Refactoring

-   **(Placeholder)** Review opportunities for code simplification, better variable naming, or improved efficiency, especially within the `_reprocess_data` and plotting update methods.

## Documentation

-   **(Placeholder)** Update README or add comments explaining any new features (like outlier removal methods) or significant bug fixes.


MORE TODO

Implement PCA-based signal generation as a toggleable alternative to median vertical displacement.

Feature Quality Weighting (Shi-Tomasi Score)

Add TQDM to webcam feed when loading

Explore Kalman Filter to filter noise from raw motion signal

Option to choose either A - differential signal (current implementation) or B - Derive a Level Metric

You've hit on a very important distinction in signal processing for control versus analysis! The current raw signal (based on frame-to-frame displacement of features) is excellent for detecting the rhythm and phase of breathing, which is why it works well for BPM. However, as you've correctly observed, it doesn't represent the absolute level or sustained state of your breath (like a held inhale).

When you hold your breath, the features become relatively static from one frame to the next. Thus, new_points - old_points becomes close to zero. The EMA then faithfully smooths this near-zero input, causing its output to also return towards zero.

To get a signal that reflects the current expansion level of your chest for OSC modulation, we need to derive a different kind of raw signal from the feature points—one that represents an absolute position or extent, rather than just change.

Here's how we can change that, focusing on creating a separate, level-based signal path specifically for your OSC output, while leaving the existing displacement-based signal path intact for BPM/phase calculation:

The Strategy: Derive a Level Metric and Smooth It

Derive a "Raw Level Metric": In PipelineManager, after getting the current_tracked_points (which are the absolute coordinates of features in the current frame), we can calculate a simple metric that represents the overall vertical position of these features. A good candidate is the mean Y-coordinate of the tracked points.
When you inhale, your chest (and the features on it) tends to move upwards, so the mean Y-coordinate would decrease (assuming Y=0 is at the top of the frame).
When you hold that inhale, the mean Y-coordinate would stay at that new, lower value.
Smooth this Raw Level Metric: Apply a simple Exponential Moving Average (EMA) to this new raw_level_metric. This will smooth out jitter while still allowing the signal to track and hold sustained levels.
Output for OSC: This smoothed level signal will then be suitable for your OSC modulation.