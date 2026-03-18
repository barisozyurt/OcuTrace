"""Tests for saccade latency calculation and response classification."""

import pytest

from src.analysis.metrics import (
    classify_response,
    compute_saccade_latency,
    compute_session_metrics,
    SessionMetrics,
)


class TestComputeSaccadeLatency:
    """Tests for compute_saccade_latency."""

    def test_basic_latency(self) -> None:
        assert compute_saccade_latency(1000.0, 1200.0) == 200.0

    def test_negative_latency_is_anticipatory(self) -> None:
        assert compute_saccade_latency(1000.0, 950.0) == -50.0


class TestClassifyResponse:
    """Tests for classify_response."""

    def test_antisaccade_correct(self) -> None:
        assert classify_response("antisaccade", "right", "left") is True

    def test_antisaccade_error(self) -> None:
        assert classify_response("antisaccade", "right", "right") is False

    def test_prosaccade_correct(self) -> None:
        assert classify_response("prosaccade", "left", "left") is True

    def test_prosaccade_error(self) -> None:
        assert classify_response("prosaccade", "left", "right") is False


class TestComputeSessionMetrics:
    """Tests for compute_session_metrics."""

    def test_basic_session_metrics(self) -> None:
        trials = [
            {"trial_type": "antisaccade", "response_correct": True, "saccade_latency_ms": 250.0},
            {"trial_type": "antisaccade", "response_correct": False, "saccade_latency_ms": 200.0},
            {"trial_type": "antisaccade", "response_correct": True, "saccade_latency_ms": 300.0},
            {"trial_type": "prosaccade", "response_correct": True, "saccade_latency_ms": 180.0},
            {"trial_type": "prosaccade", "response_correct": False, "saccade_latency_ms": 160.0},
        ]
        m = compute_session_metrics(trials)

        assert m.n_antisaccade_trials == 3
        assert m.n_prosaccade_trials == 2
        assert m.antisaccade_error_rate == pytest.approx(1 / 3)
        assert m.prosaccade_error_rate == pytest.approx(0.5)
        assert m.mean_antisaccade_latency_ms == pytest.approx(250.0)
        assert m.mean_prosaccade_latency_ms == pytest.approx(170.0)
        assert m.median_antisaccade_latency_ms == pytest.approx(250.0)
        assert m.median_prosaccade_latency_ms == pytest.approx(170.0)

    def test_empty_trials(self) -> None:
        m = compute_session_metrics([])

        assert m.n_antisaccade_trials == 0
        assert m.n_prosaccade_trials == 0
        assert m.antisaccade_error_rate == 0.0
        assert m.prosaccade_error_rate == 0.0
        assert m.mean_antisaccade_latency_ms is None
        assert m.median_antisaccade_latency_ms is None
        assert m.mean_prosaccade_latency_ms is None
        assert m.median_prosaccade_latency_ms is None

    def test_no_saccade_detected_excluded(self) -> None:
        trials = [
            {"trial_type": "antisaccade", "response_correct": True, "saccade_latency_ms": 200.0},
            {"trial_type": "antisaccade", "response_correct": True, "saccade_latency_ms": None},
            {"trial_type": "antisaccade", "response_correct": None, "saccade_latency_ms": None},
        ]
        m = compute_session_metrics(trials)

        assert m.n_antisaccade_trials == 3
        # Only 2 classified (non-None response_correct), 0 errors
        assert m.antisaccade_error_rate == 0.0
        # Only 1 valid latency
        assert m.mean_antisaccade_latency_ms == pytest.approx(200.0)
        assert m.median_antisaccade_latency_ms == pytest.approx(200.0)
