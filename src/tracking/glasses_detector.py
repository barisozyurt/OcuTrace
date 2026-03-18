"""Glasses detection and tracking quality assessment.

Detects glasses presence via FaceMesh landmark geometry and assesses
iris tracking quality using jitter and detection rate metrics.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.storage.models import TrackingQuality
from src.tracking.iris_tracker import IrisCoordinates


class GlassesDetector:
    """Detect glasses using FaceMesh landmark geometry.

    Uses the ratio of distances around the eye-nose bridge area.
    Glasses frames create a distinct geometric signature in the
    landmark positions around the eyes.
    """

    _LEFT_EYE_TOP = 159
    _LEFT_EYE_BOTTOM = 145
    _RIGHT_EYE_TOP = 386
    _RIGHT_EYE_BOTTOM = 374
    _NOSE_BRIDGE_TOP = 6
    _LEFT_TEMPLE = 226
    _RIGHT_TEMPLE = 446

    def detect_from_landmarks(self, landmarks: list[Any]) -> bool:
        """Detect glasses from FaceMesh landmarks.

        Parameters
        ----------
        landmarks : list
            FaceMesh landmark list (478 landmarks).
            If empty or insufficient, returns False.

        Returns
        -------
        bool
            True if glasses are likely present.
        """
        if len(landmarks) < 478:
            return False

        left_top = landmarks[self._LEFT_EYE_TOP]
        left_bottom = landmarks[self._LEFT_EYE_BOTTOM]
        right_top = landmarks[self._RIGHT_EYE_TOP]
        right_bottom = landmarks[self._RIGHT_EYE_BOTTOM]
        left_temple = landmarks[self._LEFT_TEMPLE]
        right_temple = landmarks[self._RIGHT_TEMPLE]

        temple_dist = np.sqrt(
            (right_temple.x - left_temple.x) ** 2
            + (right_temple.y - left_temple.y) ** 2
        )

        eye_height_left = abs(left_top.y - left_bottom.y)
        eye_height_right = abs(right_top.y - right_bottom.y)
        avg_eye_height = (eye_height_left + eye_height_right) / 2.0

        if avg_eye_height < 1e-6:
            return False

        ratio = temple_dist / avg_eye_height
        return ratio > 12.0


def assess_tracking_quality(
    coords_history: list[IrisCoordinates],
    none_count: int,
    total_frames: int,
    jitter_threshold_px: float,
    min_detection_rate: float,
) -> TrackingQuality:
    """Assess tracking quality from a window of iris coordinates.

    Parameters
    ----------
    coords_history : list[IrisCoordinates]
        Successfully detected coordinates.
    none_count : int
        Number of frames where detection failed.
    total_frames : int
        Total frames in the assessment window.
    jitter_threshold_px : float
        Maximum acceptable mean jitter in pixels.
    min_detection_rate : float
        Minimum acceptable detection rate (0-1).

    Returns
    -------
    TrackingQuality
        Assessment result with metrics.
    """
    detection_rate = len(coords_history) / total_frames if total_frames > 0 else 0.0

    if len(coords_history) < 2:
        return TrackingQuality(
            detection_rate=detection_rate,
            mean_jitter_px=float("inf"),
            glasses_detected=False,
            quality_acceptable=False,
        )

    x_values = np.array([c.mean_x for c in coords_history])
    diffs = np.abs(np.diff(x_values))
    mean_jitter = float(np.mean(diffs))

    quality_ok = (
        detection_rate >= min_detection_rate
        and mean_jitter <= jitter_threshold_px
    )

    return TrackingQuality(
        detection_rate=detection_rate,
        mean_jitter_px=mean_jitter,
        glasses_detected=False,  # Set by caller after landmark check
        quality_acceptable=quality_ok,
    )
