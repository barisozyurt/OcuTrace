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


def smooth_positions_kalman(
    positions: np.ndarray,
    timestamps_ms: np.ndarray,
    process_noise: float = 0.1,
    measurement_noise: float = 1.0,
) -> np.ndarray:
    """Apply 1D Kalman filter to iris position signal.

    Uses a constant-velocity model: state = [position, velocity].
    Better than Savitzky-Golay at preserving saccade edges while
    suppressing fixation noise, because it adapts to velocity changes.

    Parameters
    ----------
    positions : np.ndarray
        1-D array of iris positions (degrees).
    timestamps_ms : np.ndarray
        1-D array of timestamps in milliseconds.
    process_noise : float
        Process noise (how much the eye is expected to move between frames).
    measurement_noise : float
        Measurement noise (iris tracking jitter in degrees).

    Returns
    -------
    np.ndarray
        Kalman-filtered positions, same length as input.
    """
    n = len(positions)
    if n < 2:
        return positions.copy()

    # State: [position, velocity]
    x = np.array([positions[0], 0.0])  # initial state
    P = np.array([[measurement_noise, 0.0],
                  [0.0, 10.0]])  # initial covariance

    R = measurement_noise  # measurement noise variance
    filtered = np.empty(n, dtype=float)
    filtered[0] = positions[0]

    for i in range(1, n):
        dt = (timestamps_ms[i] - timestamps_ms[i - 1]) / 1000.0
        if dt <= 0:
            dt = 0.033  # assume 30fps

        # State transition: position += velocity * dt
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])

        # Process noise: acceleration uncertainty
        Q = process_noise * np.array([
            [dt**4 / 4, dt**3 / 2],
            [dt**3 / 2, dt**2],
        ])

        # Predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Update with measurement
        H = np.array([1.0, 0.0])  # we observe position only
        innovation = positions[i] - H @ x_pred
        S = H @ P_pred @ H + R
        K = P_pred @ H / S  # Kalman gain

        x = x_pred + K * innovation
        P = (np.eye(2) - np.outer(K, H)) @ P_pred

        filtered[i] = x[0]

    return filtered


def upsample_positions(
    positions: np.ndarray,
    timestamps_ms: np.ndarray,
    factor: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample position signal using cubic spline interpolation.

    Increases effective temporal resolution, giving smoother velocity
    curves for better saccade onset detection at low frame rates.

    Parameters
    ----------
    positions : np.ndarray
        1-D array of iris positions (degrees).
    timestamps_ms : np.ndarray
        1-D array of timestamps in milliseconds.
    factor : int
        Upsampling factor (e.g., 3 = triple the samples).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (upsampled_positions, upsampled_timestamps)
    """
    from scipy.interpolate import CubicSpline

    n = len(positions)
    if n < 4:
        return positions.copy(), timestamps_ms.copy()

    cs = CubicSpline(timestamps_ms, positions)
    new_ts = np.linspace(timestamps_ms[0], timestamps_ms[-1], n * factor)
    new_pos = cs(new_ts)

    return new_pos, new_ts
