"""Integration tests for the full saccade analysis pipeline.

Validates: positions -> smooth -> velocity -> detect -> classify -> latency
using synthetic degree-space data. No camera or MediaPipe dependency.
"""
from __future__ import annotations

import numpy as np

from src.analysis import (
    smooth_positions,
    compute_velocity,
    detect_saccades,
    classify_direction,
    compute_saccade_latency,
    classify_response,
)


def test_full_pipeline_rightward_saccade(synthetic_saccade_degrees: dict) -> None:
    """End-to-end pipeline with a rightward saccade detects event correctly."""
    data = synthetic_saccade_degrees
    positions = data["positions_deg"]
    timestamps = data["timestamps_ms"]

    smoothed = smooth_positions(positions)
    velocity = compute_velocity(smoothed, timestamps)
    events = detect_saccades(velocity, timestamps)

    # At least one saccade detected
    assert len(events) >= 1

    evt = events[0]

    # Onset within +/- 2 frames of expected
    assert abs(evt.onset_idx - data["expected_onset_idx"]) <= 2

    # Direction is right
    direction = classify_direction(smoothed, evt.onset_idx, evt.offset_idx)
    assert direction == data["expected_direction"]

    # Latency within 100ms of expected
    latency = compute_saccade_latency(data["stimulus_onset_ms"], evt.onset_ms)
    assert abs(latency - data["expected_latency_ms"]) <= 100.0

    # Antisaccade with stimulus on left, saccade to right -> correct
    correct = classify_response("antisaccade", "left", direction)
    assert correct is True


def test_no_saccade_in_pure_fixation(rng: np.random.Generator) -> None:
    """Pure fixation data with small noise produces zero saccade events."""
    n_frames = 60
    dt_ms = 33.33
    positions = np.full(n_frames, 0.0) + rng.normal(0, 0.1, n_frames)
    timestamps = np.arange(n_frames) * dt_ms

    smoothed = smooth_positions(positions)
    velocity = compute_velocity(smoothed, timestamps)
    events = detect_saccades(velocity, timestamps)

    assert len(events) == 0


def test_prosaccade_latency_in_normal_range() -> None:
    """Prosaccade latency falls within the normal 100-300ms range."""
    fps = 60
    dt_ms = 1000.0 / fps

    # 12 frames fixation, 4 frames saccade (0 -> 10deg), 20 frames fixation
    fix1 = np.full(12, 0.0)
    saccade = np.linspace(0, 10, 4)
    fix2 = np.full(20, 10.0)
    positions = np.concatenate([fix1, saccade, fix2])
    timestamps = np.arange(len(positions)) * dt_ms

    stimulus_onset_ms = 0.0

    smoothed = smooth_positions(positions)
    velocity = compute_velocity(smoothed, timestamps)
    events = detect_saccades(velocity, timestamps)

    assert len(events) >= 1

    latency = compute_saccade_latency(stimulus_onset_ms, events[0].onset_ms)
    assert 100.0 <= latency <= 300.0
