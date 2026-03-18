"""Calibration system for pixel-to-degree coordinate transformation.

Maps raw iris pixel coordinates to visual angle (degrees) using
an affine transformation fitted from calibration point measurements.
"""
from __future__ import annotations

import numpy as np

from src.storage.models import CalibrationPoint, CalibrationResult


def fit_pixel_to_degree_transform(
    points: list[CalibrationPoint],
) -> np.ndarray:
    """Fit an affine transform from pixel coordinates to degrees.

    Fits X and Y axes independently using grouped averaging: all
    measurements sharing the same target-X are averaged to produce
    one robust pixel-X value per target-X level (and likewise for Y).
    This eliminates cross-axis contamination from webcam parallax
    and eyelid occlusion at extreme vertical gaze angles.

    Parameters
    ----------
    points : list[CalibrationPoint]
        Calibration measurements (minimum 3 points).

    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix (off-diagonal terms are zero
        because axes are fitted independently).

    Raises
    ------
    ValueError
        If fewer than 3 points are provided.
    """
    if len(points) < 3:
        raise ValueError(
            f"Need at least 3 calibration points, got {len(points)}"
        )

    # Group by target_x and average iris_x per level
    x_groups: dict[float, list[float]] = {}
    for p in points:
        x_groups.setdefault(p.target_x_deg, []).append(p.measured_x_px)

    avg_px_x = np.array([np.mean(v) for v in x_groups.values()])
    target_x = np.array(list(x_groups.keys()))

    # Group by target_y and average iris_y per level
    y_groups: dict[float, list[float]] = {}
    for p in points:
        y_groups.setdefault(p.target_y_deg, []).append(p.measured_y_px)

    avg_px_y = np.array([np.mean(v) for v in y_groups.values()])
    target_y = np.array(list(y_groups.keys()))

    # Fit X: deg_x = ax * pixel_x + bx
    src_x = np.column_stack([avg_px_x, np.ones(len(avg_px_x))])
    coeff_x, _, _, _ = np.linalg.lstsq(src_x, target_x, rcond=None)

    # Fit Y: deg_y = ay * pixel_y + by
    src_y = np.column_stack([avg_px_y, np.ones(len(avg_px_y))])
    coeff_y, _, _, _ = np.linalg.lstsq(src_y, target_y, rcond=None)

    matrix = np.array([
        [coeff_x[0], 0.0,        coeff_x[1]],
        [0.0,        coeff_y[0], coeff_y[1]],
        [0.0,        0.0,        1.0       ],
    ])

    return matrix


def apply_transform(
    matrix: np.ndarray,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float]:
    """Apply calibration transform to convert pixels to degrees.

    Parameters
    ----------
    matrix : np.ndarray
        3x3 affine transformation matrix.
    pixel_x : float
        Iris X position in pixels.
    pixel_y : float
        Iris Y position in pixels.

    Returns
    -------
    tuple[float, float]
        (degree_x, degree_y) visual angle from center.
    """
    pixel_vec = np.array([pixel_x, pixel_y, 1.0])
    result = matrix @ pixel_vec
    return (float(result[0]), float(result[1]))


def compute_calibration_error(
    matrix: np.ndarray,
    points: list[CalibrationPoint],
) -> float:
    """Compute mean horizontal calibration error in degrees.

    Only evaluates horizontal (X) accuracy because saccade detection
    uses horizontal gaze exclusively. Vertical (Y) accuracy from
    webcam iris tracking is unreliable due to parallax and eyelid
    occlusion and should not gate calibration acceptance.

    Parameters
    ----------
    matrix : np.ndarray
        Fitted transform matrix.
    points : list[CalibrationPoint]
        Calibration points to validate against.

    Returns
    -------
    float
        Mean absolute horizontal error in degrees.
    """
    errors = []
    for p in points:
        pred_x, _ = apply_transform(matrix, p.measured_x_px, p.measured_y_px)
        errors.append(abs(pred_x - p.target_x_deg))
    return float(np.mean(errors))


def create_calibration_result(
    session_id: str,
    points: list[CalibrationPoint],
    max_error_deg: float,
) -> CalibrationResult:
    """Run full calibration pipeline: fit, compute error, decide acceptance.

    Parameters
    ----------
    session_id : str
        Session UUID.
    points : list[CalibrationPoint]
        Calibration measurements.
    max_error_deg : float
        Maximum acceptable mean error in degrees.

    Returns
    -------
    CalibrationResult
        Complete calibration result.
    """
    matrix = fit_pixel_to_degree_transform(points)
    mean_error = compute_calibration_error(matrix, points)

    return CalibrationResult(
        session_id=session_id,
        points=points,
        transform_matrix=matrix.tolist(),
        mean_error_deg=mean_error,
        accepted=mean_error <= max_error_deg,
    )
