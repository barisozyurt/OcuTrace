"""Velocity-threshold saccade detection algorithm.

Implements the saccade detection pipeline described in CLAUDE.md:
onset when velocity exceeds threshold for N consecutive frames,
offset when velocity drops below a lower threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

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

        if run_start is None:  # pragma: no cover — defensive guard
            break
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


def detect_saccades_displacement(
    positions_deg: np.ndarray,
    timestamps_ms: np.ndarray,
    stimulus_onset_ms: float,
    displacement_threshold: float = 2.0,
    search_window_ms: float = 800.0,
    baseline_window_ms: float = 100.0,
) -> list[SaccadeEvent]:
    """Detect saccades using position displacement from baseline.

    This is a fallback detector for low-framerate webcam data where
    velocity-based detection misses saccades due to smoothing.

    It computes a baseline position from the period just before stimulus
    onset, then scans for the first point where displacement exceeds
    the threshold.

    Parameters
    ----------
    positions_deg : np.ndarray
        1-D array of eye positions in degrees.
    timestamps_ms : np.ndarray
        1-D array of timestamps in milliseconds.
    stimulus_onset_ms : float
        Timestamp of stimulus onset.
    displacement_threshold : float
        Minimum displacement from baseline to count as saccade (degrees).
    search_window_ms : float
        How long after stimulus onset to search (ms).
    baseline_window_ms : float
        Period before stimulus onset to compute baseline position (ms).

    Returns
    -------
    list[SaccadeEvent]
        At most one saccade event (the first threshold crossing).
    """
    n = len(positions_deg)
    if n < 3:
        return []

    # Compute baseline: mean position in the window before stimulus onset
    baseline_start = stimulus_onset_ms - baseline_window_ms
    baseline_indices = [
        i for i in range(n)
        if baseline_start <= timestamps_ms[i] <= stimulus_onset_ms
    ]
    if not baseline_indices:
        # Fall back to the last few samples before stimulus
        pre_stim = [i for i in range(n) if timestamps_ms[i] <= stimulus_onset_ms]
        baseline_indices = pre_stim[-3:] if len(pre_stim) >= 3 else pre_stim

    if not baseline_indices:
        return []

    baseline_pos = float(np.mean(positions_deg[baseline_indices]))

    # Scan for displacement exceeding threshold after stimulus onset
    search_end = stimulus_onset_ms + search_window_ms
    onset_idx = None

    for i in range(n):
        if timestamps_ms[i] < stimulus_onset_ms:
            continue
        if timestamps_ms[i] > search_end:
            break

        displacement = abs(float(positions_deg[i]) - baseline_pos)
        if displacement >= displacement_threshold:
            onset_idx = i
            break

    if onset_idx is None:
        return []

    # Find offset: where displacement stabilizes or reverses
    offset_idx = onset_idx
    peak_displacement = abs(float(positions_deg[onset_idx]) - baseline_pos)
    for i in range(onset_idx + 1, n):
        if timestamps_ms[i] > search_end:
            break
        curr_displacement = abs(float(positions_deg[i]) - baseline_pos)
        if curr_displacement > peak_displacement:
            peak_displacement = curr_displacement
            offset_idx = i
        elif curr_displacement < peak_displacement * 0.7:
            break  # displacement started decreasing
        else:
            offset_idx = i

    # Estimate peak velocity from the displacement
    dt = (timestamps_ms[offset_idx] - timestamps_ms[onset_idx]) / 1000.0
    if dt <= 0:
        dt = 0.033  # assume 30fps
    peak_vel = peak_displacement / dt

    return [SaccadeEvent(
        onset_idx=onset_idx,
        offset_idx=offset_idx,
        onset_ms=float(timestamps_ms[onset_idx]),
        offset_ms=float(timestamps_ms[offset_idx]),
        peak_velocity=peak_vel,
    )]


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
    if displacement == 0.0:
        logger.warning(
            "Zero displacement saccade at onset_idx=%d, defaulting to 'right'",
            onset_idx,
        )
    return "right" if displacement >= 0 else "left"
