"""Velocity-threshold saccade detection algorithm.

Implements the saccade detection pipeline described in CLAUDE.md:
onset when velocity exceeds threshold for N consecutive frames,
offset when velocity drops below a lower threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SaccadeEvent:
    """A single detected saccade.

    Attributes
    ----------
    onset_idx : int
        Frame index of saccade onset.
    offset_idx : int
        Frame index of saccade offset.
    onset_ms : float
        Timestamp of saccade onset in milliseconds.
    offset_ms : float
        Timestamp of saccade offset in milliseconds.
    peak_velocity : float
        Maximum absolute velocity during the saccade (deg/s).
    """

    onset_idx: int
    offset_idx: int
    onset_ms: float
    offset_ms: float
    peak_velocity: float


def detect_saccades(
    velocity: np.ndarray,
    timestamps_ms: np.ndarray,
    onset_threshold: float = 30.0,
    offset_threshold: float = 20.0,
    min_onset_frames: int = 3,
) -> list[SaccadeEvent]:
    """Detect saccades using a velocity-threshold algorithm.

    Parameters
    ----------
    velocity : np.ndarray
        1-D array of absolute eye velocity in deg/s.
    timestamps_ms : np.ndarray
        1-D array of timestamps in milliseconds, same length as *velocity*.
    onset_threshold : float, optional
        Velocity threshold for saccade onset (deg/s). Default 30.0.
    offset_threshold : float, optional
        Velocity threshold for saccade offset (deg/s). Default 20.0.
    min_onset_frames : int, optional
        Minimum consecutive frames above *onset_threshold* to qualify as
        a saccade onset. Default 3.

    Returns
    -------
    list[SaccadeEvent]
        Detected saccade events sorted by onset index.
    """
    n = len(velocity)
    events: list[SaccadeEvent] = []
    i = 0

    while i < n:
        # 1. Find a run of min_onset_frames consecutive frames >= onset_threshold
        run_start = None
        run_length = 0

        while i < n:
            if velocity[i] >= onset_threshold:
                if run_start is None:
                    run_start = i
                run_length += 1
                if run_length >= min_onset_frames:
                    break
            else:
                run_start = None
                run_length = 0
            i += 1

        if run_length < min_onset_frames:
            break  # no more onsets found

        assert run_start is not None
        onset_idx = run_start

        # 2. Continue scanning until velocity drops below offset_threshold
        i += 1
        while i < n and velocity[i] >= offset_threshold:
            i += 1

        offset_idx = i - 1 if i <= n else n - 1
        # Ensure offset is at least at the last frame of the onset run
        offset_idx = max(offset_idx, onset_idx + min_onset_frames - 1)

        # 3. Record event
        peak_vel = float(np.max(velocity[onset_idx : offset_idx + 1]))
        events.append(
            SaccadeEvent(
                onset_idx=onset_idx,
                offset_idx=offset_idx,
                onset_ms=float(timestamps_ms[onset_idx]),
                offset_ms=float(timestamps_ms[offset_idx]),
                peak_velocity=peak_vel,
            )
        )

        # 4. Continue scanning after offset
        i = offset_idx + 1

    return events


def classify_direction(
    positions_deg: np.ndarray,
    onset_idx: int,
    offset_idx: int,
) -> str:
    """Classify saccade direction based on positional displacement.

    Parameters
    ----------
    positions_deg : np.ndarray
        1-D array of eye positions in degrees of visual angle.
    onset_idx : int
        Frame index of saccade onset.
    offset_idx : int
        Frame index of saccade offset.

    Returns
    -------
    str
        ``"right"`` if the displacement is positive, ``"left"`` otherwise.
    """
    displacement = float(positions_deg[offset_idx] - positions_deg[onset_idx])
    return "right" if displacement > 0 else "left"
