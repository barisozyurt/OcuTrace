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

    Parameters
    ----------
    points : list[CalibrationPoint]
        Calibration measurements (minimum 3 points).

    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix.

    Raises
    ------
    ValueError
        If fewer than 3 points are provided.
    """
    if len(points) < 3:
        raise ValueError(
            f"Need at least 3 calibration points, got {len(points)}"
        )

    n = len(points)
    src = np.zeros((n, 3))
    dst = np.zeros((n, 2))
    for i, p in enumerate(points):
        src[i] = [p.measured_x_px, p.measured_y_px, 1.0]
        dst[i] = [p.target_x_deg, p.target_y_deg]

    a_t, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)

    matrix = np.eye(3)
    matrix[:2, :] = a_t.T

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
    """Compute mean calibration error in degrees.

    Parameters
    ----------
    matrix : np.ndarray
        Fitted transform matrix.
    points : list[CalibrationPoint]
        Calibration points to validate against.

    Returns
    -------
    float
        Mean Euclidean error in degrees.
    """
    errors = []
    for p in points:
        pred_x, pred_y = apply_transform(matrix, p.measured_x_px, p.measured_y_px)
        error = np.sqrt(
            (pred_x - p.target_x_deg) ** 2
            + (pred_y - p.target_y_deg) ** 2
        )
        errors.append(error)
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
