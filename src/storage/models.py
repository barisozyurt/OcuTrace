"""Data models for OcuTrace sessions, trials, and gaze data.

All timestamps are UTC Unix milliseconds.
Sessions are identified by UUID for pseudonymization.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

VALID_TRIAL_TYPES = ("antisaccade", "prosaccade")
VALID_STIMULUS_SIDES = ("left", "right")


def _utc_now_ms() -> float:
    """Return current UTC time as Unix milliseconds."""
    return datetime.now(timezone.utc).timestamp() * 1000


def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


@dataclass
class Session:
    """A single recording session.

    Parameters
    ----------
    participant_id : str
        Pseudonymized participant identifier.
    notes : str
        Optional session notes.
    session_id : str
        Auto-generated UUID4.
    created_at : float
        UTC Unix milliseconds, auto-generated.
    glasses_detected : bool or None
        Set after quality gate check.
    tracking_quality_score : float or None
        Set after quality gate check.
    """
    participant_id: str
    notes: str = ""
    session_id: str = field(default_factory=_new_uuid)
    created_at: float = field(default_factory=_utc_now_ms)
    glasses_detected: Optional[bool] = None
    tracking_quality_score: Optional[float] = None


@dataclass
class Trial:
    """A single trial within a session.

    Parameters
    ----------
    session_id : str
        Parent session UUID.
    trial_number : int
        1-based trial index.
    trial_type : str
        'antisaccade' or 'prosaccade'.
    stimulus_side : str
        'left' or 'right'.
    stimulus_onset_ms : float
        Timestamp of stimulus flip in UTC Unix ms.
    response_correct : bool or None
        Set after saccade classification.
    saccade_latency_ms : float or None
        Time from stimulus onset to saccade onset.
    saccade_direction : str or None
        Detected saccade direction.
    """
    session_id: str
    trial_number: int
    trial_type: str
    stimulus_side: str
    stimulus_onset_ms: float
    response_correct: Optional[bool] = None
    saccade_latency_ms: Optional[float] = None
    saccade_direction: Optional[str] = None

    def __post_init__(self) -> None:
        if self.trial_type not in VALID_TRIAL_TYPES:
            raise ValueError(
                f"trial_type must be one of {VALID_TRIAL_TYPES}, "
                f"got '{self.trial_type}'"
            )
        if self.stimulus_side not in VALID_STIMULUS_SIDES:
            raise ValueError(
                f"stimulus_side must be one of {VALID_STIMULUS_SIDES}, "
                f"got '{self.stimulus_side}'"
            )


@dataclass
class GazeData:
    """A single gaze sample (one camera frame).

    Parameters
    ----------
    session_id : str
        Parent session UUID.
    trial_number : int
        Which trial this sample belongs to (0 = calibration/inter-trial).
    timestamp_ms : float
        UTC Unix milliseconds of camera frame capture.
    left_iris_x : float
        Left iris center X in pixels.
    left_iris_y : float
        Left iris center Y in pixels.
    right_iris_x : float
        Right iris center X in pixels.
    right_iris_y : float
        Right iris center Y in pixels.
    confidence : float
        MediaPipe detection confidence (0-1).
    """
    session_id: str
    trial_number: int
    timestamp_ms: float
    left_iris_x: float
    left_iris_y: float
    right_iris_x: float
    right_iris_y: float
    confidence: float


@dataclass
class TrackingQuality:
    """Result of tracking quality assessment.

    Parameters
    ----------
    detection_rate : float
        Fraction of frames where iris was detected (0-1).
    mean_jitter_px : float
        Mean frame-to-frame jitter in pixels during fixation.
    glasses_detected : bool
        Whether glasses were detected on the face.
    quality_acceptable : bool
        Whether tracking quality meets threshold for experiment.
    """
    detection_rate: float
    mean_jitter_px: float
    glasses_detected: bool
    quality_acceptable: bool


@dataclass
class CalibrationPoint:
    """A single calibration target measurement.

    Parameters
    ----------
    target_x_deg : float
        Target position X in degrees from center.
    target_y_deg : float
        Target position Y in degrees from center.
    measured_x_px : float
        Mean measured iris X in pixels at this target.
    measured_y_px : float
        Mean measured iris Y in pixels at this target.
    """
    target_x_deg: float
    target_y_deg: float
    measured_x_px: float
    measured_y_px: float


@dataclass
class CalibrationResult:
    """Result of a calibration procedure.

    Parameters
    ----------
    session_id : str
        Session this calibration belongs to.
    points : list[CalibrationPoint]
        All calibration measurements.
    transform_matrix : list[list[float]]
        3x3 affine transform matrix (pixel to degree), stored as nested list.
    mean_error_deg : float
        Mean validation error in degrees.
    accepted : bool
        Whether error is within acceptable threshold.
    """
    session_id: str
    points: list[CalibrationPoint]
    transform_matrix: list[list[float]]
    mean_error_deg: float
    accepted: bool
