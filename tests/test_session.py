"""Tests for session management logic."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiment.session import analyze_trial
from src.experiment.paradigm import TrialSpec
from src.storage.models import GazeData


def _make_gaze_samples(
    session_id: str,
    trial_number: int,
    n_fixation: int = 15,
    n_saccade: int = 5,
    n_post: int = 15,
    saccade_px: float = 100.0,
    base_x: float = 320.0,
    fps: float = 30.0,
) -> list[GazeData]:
    """Create synthetic gaze samples: fixation -> rightward saccade -> fixation."""
    samples: list[GazeData] = []
    dt_ms = 1000.0 / fps
    rng = np.random.default_rng(42)

    # Fixation phase
    for i in range(n_fixation):
        x = base_x + rng.normal(0, 0.5)
        samples.append(
            GazeData(
                session_id=session_id,
                trial_number=trial_number,
                timestamp_ms=i * dt_ms,
                left_iris_x=x,
                left_iris_y=240.0,
                right_iris_x=x,
                right_iris_y=240.0,
                confidence=0.95,
            )
        )

    # Saccade phase (rightward)
    for i in range(n_saccade):
        x = base_x + saccade_px * (i + 1) / n_saccade
        t = (n_fixation + i) * dt_ms
        samples.append(
            GazeData(
                session_id=session_id,
                trial_number=trial_number,
                timestamp_ms=t,
                left_iris_x=x,
                left_iris_y=240.0,
                right_iris_x=x,
                right_iris_y=240.0,
                confidence=0.95,
            )
        )

    # Post-saccade fixation
    for i in range(n_post):
        x = base_x + saccade_px + rng.normal(0, 0.5)
        t = (n_fixation + n_saccade + i) * dt_ms
        samples.append(
            GazeData(
                session_id=session_id,
                trial_number=trial_number,
                timestamp_ms=t,
                left_iris_x=x,
                left_iris_y=240.0,
                right_iris_x=x,
                right_iris_y=240.0,
                confidence=0.95,
            )
        )

    return samples


@pytest.fixture
def identity_matrix() -> np.ndarray:
    """Identity calibration matrix — pixels pass through as degrees."""
    return np.eye(3)


@pytest.fixture
def saccade_cfg() -> dict:
    """Saccade detection configuration matching settings.yaml."""
    return {
        "smoothing_window": 5,
        "smoothing_polyorder": 2,
        "onset_velocity_threshold": 30.0,
        "offset_velocity_threshold": 20.0,
        "min_onset_frames": 3,
    }


class TestAnalyzeTrial:
    """Tests for single-trial saccade analysis."""

    def test_detects_rightward_saccade(
        self, identity_matrix: np.ndarray, saccade_cfg: dict
    ) -> None:
        samples = _make_gaze_samples("S1", 1)
        spec = TrialSpec(
            trial_number=1, trial_type="antisaccade", stimulus_side="left"
        )
        stimulus_onset_ms = 15 * 33.33  # stimulus at saccade start

        direction, latency, correct = analyze_trial(
            samples, spec, stimulus_onset_ms, identity_matrix, saccade_cfg
        )

        assert direction == "right"
        assert latency is not None
        assert abs(latency) < 200.0
        assert correct is True  # antisaccade, stim left, looked right

    def test_no_saccade_in_fixation(
        self, identity_matrix: np.ndarray, saccade_cfg: dict
    ) -> None:
        samples = _make_gaze_samples(
            "S1", 1, n_saccade=0, n_post=0, n_fixation=35
        )
        spec = TrialSpec(
            trial_number=1, trial_type="prosaccade", stimulus_side="right"
        )

        direction, latency, correct = analyze_trial(
            samples, spec, 500.0, identity_matrix, saccade_cfg
        )

        assert direction is None
        assert latency is None
        assert correct is None

    def test_too_few_samples(
        self, identity_matrix: np.ndarray, saccade_cfg: dict
    ) -> None:
        samples = _make_gaze_samples(
            "S1", 1, n_fixation=2, n_saccade=0, n_post=0
        )
        spec = TrialSpec(
            trial_number=1, trial_type="prosaccade", stimulus_side="left"
        )

        direction, latency, correct = analyze_trial(
            samples, spec, 0.0, identity_matrix, saccade_cfg
        )

        assert direction is None
        assert latency is None
        assert correct is None

    def test_prosaccade_error_detection(
        self, identity_matrix: np.ndarray, saccade_cfg: dict
    ) -> None:
        # Rightward saccade but stimulus is on the left -> prosaccade error
        samples = _make_gaze_samples("S1", 1)
        spec = TrialSpec(
            trial_number=1, trial_type="prosaccade", stimulus_side="left"
        )
        stimulus_onset_ms = 15 * 33.33

        direction, latency, correct = analyze_trial(
            samples, spec, stimulus_onset_ms, identity_matrix, saccade_cfg
        )

        assert direction == "right"
        assert correct is False  # prosaccade, stim left, looked right = error
