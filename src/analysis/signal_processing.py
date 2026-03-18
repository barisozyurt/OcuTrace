"""Signal processing utilities for saccade detection pipeline.

Provides Savitzky-Golay smoothing and velocity computation from
degree-space iris positions. Pure NumPy/SciPy — no OcuTrace dependencies.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def smooth_positions(
    positions: np.ndarray,
    window: int = 5,
    polyorder: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay filter to an iris position signal.

    Parameters
    ----------
    positions : np.ndarray
        1-D array of iris positions (pixels or degrees).
    window : int, optional
        Window length for the filter (must be odd). Default is 5.
    polyorder : int, optional
        Polynomial order for the filter. Default is 2.

    Returns
    -------
    np.ndarray
        Smoothed positions, same length as *positions*.

    Notes
    -----
    If the signal is shorter than *window*, the window is reduced to the
    largest odd number <= ``len(positions)`` and *polyorder* is clamped
    accordingly so that ``polyorder < window``.
    """
    n = len(positions)
    if n < window:
        # Fall back to the largest odd window that fits
        window = n if n % 2 == 1 else n - 1
        if window < 1:
            return positions.copy()
        polyorder = min(polyorder, window - 1)

    return savgol_filter(positions, window_length=window, polyorder=polyorder)


def compute_velocity(
    positions_deg: np.ndarray,
    timestamps_ms: np.ndarray,
) -> np.ndarray:
    """Compute absolute velocity in degrees/second.

    Parameters
    ----------
    positions_deg : np.ndarray
        1-D array of iris positions in degrees of visual angle.
    timestamps_ms : np.ndarray
        1-D array of corresponding timestamps in milliseconds.

    Returns
    -------
    np.ndarray
        Absolute velocity in deg/s, same length as input.
        The first element is always 0.0 (no previous frame).

    Notes
    -----
    Zero-length time deltas are replaced with 1e-9 ms to avoid
    division-by-zero errors.
    """
    dt_ms = np.diff(timestamps_ms).astype(float)
    dt_ms[dt_ms == 0.0] = 1e-9  # guard against zero dt
    dt_s = dt_ms / 1000.0

    dp = np.abs(np.diff(positions_deg).astype(float))

    velocity = np.empty_like(positions_deg, dtype=float)
    velocity[0] = 0.0
    velocity[1:] = dp / dt_s

    return velocity
