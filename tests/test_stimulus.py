"""Tests for pure-logic stimulus functions (no PsychoPy window required)."""
from __future__ import annotations

import numpy as np
import pytest

from src.experiment.stimulus import (
    StimulusConfig,
    compute_iti_frame_count,
    compute_trial_frame_counts,
    create_stimulus_config,
)


class TestCreateFromParadigmConfig:
    """Test create_stimulus_config maps paradigm dict to StimulusConfig."""

    def test_create_from_paradigm_config(self) -> None:
        paradigm = {
            "fixation_duration_ms": 1000,
            "gap_duration_ms": 200,
            "stimulus_duration_ms": 1500,
            "iti_min_ms": 1000,
            "iti_max_ms": 1500,
            "stimulus_eccentricity_deg": 10.0,
        }
        config = create_stimulus_config(paradigm, monitor_refresh_hz=60.0)

        assert isinstance(config, StimulusConfig)
        assert config.fixation_duration_ms == 1000
        assert config.gap_duration_ms == 200
        assert config.stimulus_duration_ms == 1500
        assert config.iti_min_ms == 1000
        assert config.iti_max_ms == 1500
        assert config.eccentricity_deg == 10.0
        assert config.refresh_hz == 60.0

    def test_config_is_frozen(self) -> None:
        paradigm = {
            "fixation_duration_ms": 1000,
            "gap_duration_ms": 200,
            "stimulus_duration_ms": 1500,
            "iti_min_ms": 1000,
            "iti_max_ms": 1500,
            "stimulus_eccentricity_deg": 10.0,
        }
        config = create_stimulus_config(paradigm)
        with pytest.raises(AttributeError):
            config.fixation_duration_ms = 500  # type: ignore[misc]


class TestComputeTrialFrameCounts:
    """Test ms-to-frame conversion at various refresh rates."""

    def test_fixation_frames_at_60hz(self) -> None:
        result = compute_trial_frame_counts(
            fixation_ms=1000, gap_ms=200, stimulus_ms=1500, refresh_hz=60.0
        )
        assert result["fixation"] == 60
        assert result["gap"] == 12
        assert result["stimulus"] == 90

    def test_frames_at_30hz(self) -> None:
        result = compute_trial_frame_counts(
            fixation_ms=1000, gap_ms=200, stimulus_ms=1500, refresh_hz=30.0
        )
        assert result["fixation"] == 30
        assert result["gap"] == 6
        assert result["stimulus"] == 45

    def test_rounding_to_nearest_frame(self) -> None:
        result = compute_trial_frame_counts(
            fixation_ms=250, gap_ms=200, stimulus_ms=1500, refresh_hz=60.0
        )
        assert result["fixation"] == 15  # 250 * 60 / 1000 = 15.0 exact


class TestComputeItiFrameCount:
    """Test random ITI frame count generation."""

    def test_iti_within_range(self) -> None:
        rng = np.random.default_rng(42)
        min_frames = round(1000 * 60.0 / 1000)  # 60
        max_frames = round(1500 * 60.0 / 1000)  # 90
        for _ in range(100):
            count = compute_iti_frame_count(
                iti_min_ms=1000, iti_max_ms=1500, refresh_hz=60.0, rng=rng
            )
            assert min_frames <= count <= max_frames

    def test_iti_deterministic_with_seed(self) -> None:
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        a = compute_iti_frame_count(1000, 1500, 60.0, rng1)
        b = compute_iti_frame_count(1000, 1500, 60.0, rng2)
        assert a == b
