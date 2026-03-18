"""Calibration display utilities.

Generates calibration target positions. Target generation is pure
computation (no display dependency), making it testable without PsychoPy.
"""
from __future__ import annotations


def generate_calibration_targets(
    n_points: int,
    eccentricity_deg: float,
) -> list[tuple[float, float]]:
    """Generate calibration target positions in degrees.

    Parameters
    ----------
    n_points : int
        Number of calibration points (5 or 9).
    eccentricity_deg : float
        Maximum eccentricity in degrees from center.

    Returns
    -------
    list[tuple[float, float]]
        Target positions as (x_deg, y_deg) from center.

    Raises
    ------
    ValueError
        If n_points is not 5 or 9.
    """
    e = eccentricity_deg

    if n_points == 5:
        return [
            (0.0, 0.0),
            (-e, 0.0),
            (e, 0.0),
            (0.0, e),
            (0.0, -e),
        ]
    elif n_points == 9:
        return [
            (-e, e),    (-e, 0.0),   (-e, -e),
            (0.0, e),   (0.0, 0.0),  (0.0, -e),
            (e, e),     (e, 0.0),    (e, -e),
        ]
    else:
        raise ValueError(f"n_points must be 5 or 9, got {n_points}")
