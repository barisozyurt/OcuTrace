"""Tests for velocity-threshold saccade detection algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.saccade_detector import (
    SaccadeEvent,
    classify_direction,
    detect_saccades,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_saccade_velocity(
    n_frames: int = 60,
    saccade_start: int = 20,
    saccade_end: int = 28,
    peak_velocity: float = 80.0,
    baseline: float = 5.0,
) -> np.ndarray:
    """Create a synthetic velocity profile with one saccade-like peak."""
    vel = np.full(n_frames, baseline)
    saccade_len = saccade_end - saccade_start
    half = saccade_len // 2
    vel[saccade_start : saccade_start + half] = np.linspace(
        baseline, peak_velocity, half
    )
    vel[saccade_start + half : saccade_end] = np.linspace(
        peak_velocity, baseline, saccade_len - half
    )
    return vel


def _timestamps(n_frames: int = 60, fps: float = 60.0) -> np.ndarray:
    """Evenly spaced timestamps in ms."""
    return np.arange(n_frames) * (1000.0 / fps)


# ---------------------------------------------------------------------------
# Tests — detect_saccades
# ---------------------------------------------------------------------------


class TestDetectSingleSaccade:
    def test_detect_single_saccade(self) -> None:
        vel = _make_saccade_velocity()
        ts = _timestamps(len(vel))
        events = detect_saccades(vel, ts)

        assert len(events) == 1
        ev = events[0]
        assert ev.onset_idx >= 20
        assert ev.offset_idx <= 28
        assert ev.peak_velocity == pytest.approx(80.0, abs=5.0)


class TestNoSaccadeInFixation:
    def test_no_saccade_in_fixation(self) -> None:
        vel = np.full(60, 5.0)
        ts = _timestamps(60)
        events = detect_saccades(vel, ts)

        assert events == []


class TestMinOnsetFramesFiltersNoise:
    def test_min_onset_frames_filters_noise(self) -> None:
        vel = np.full(60, 5.0)
        # 2-frame spike — should be filtered with min_onset_frames=3
        vel[30] = 50.0
        vel[31] = 50.0
        ts = _timestamps(60)
        events = detect_saccades(vel, ts, min_onset_frames=3)

        assert events == []


class TestSaccadeEventHasTimestamps:
    def test_saccade_event_has_timestamps(self) -> None:
        vel = _make_saccade_velocity()
        ts = _timestamps(len(vel))
        events = detect_saccades(vel, ts)

        assert len(events) == 1
        ev = events[0]
        assert ev.onset_ms > 0.0
        assert ev.offset_ms > ev.onset_ms


class TestDetectTwoSaccades:
    def test_detect_two_saccades(self) -> None:
        vel1 = _make_saccade_velocity(
            n_frames=60, saccade_start=10, saccade_end=18
        )
        vel2 = _make_saccade_velocity(
            n_frames=60, saccade_start=40, saccade_end=48
        )
        # Combine: take the max at each frame so both peaks survive
        vel = np.maximum(vel1, vel2)
        ts = _timestamps(len(vel))
        events = detect_saccades(vel, ts)

        assert len(events) == 2
        assert events[0].onset_idx < events[1].onset_idx


# ---------------------------------------------------------------------------
# Tests — classify_direction
# ---------------------------------------------------------------------------


class TestRightwardSaccade:
    def test_rightward_saccade(self) -> None:
        positions = np.zeros(60)
        positions[25:] = 10.0  # rightward displacement
        direction = classify_direction(positions, onset_idx=20, offset_idx=28)
        assert direction == "right"


class TestLeftwardSaccade:
    def test_leftward_saccade(self) -> None:
        positions = np.zeros(60)
        positions[25:] = -10.0  # leftward displacement
        direction = classify_direction(positions, onset_idx=20, offset_idx=28)
        assert direction == "left"
