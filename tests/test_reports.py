"""Tests for clinical report generation."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.visualization.reports import (
    plot_latency_by_trial,
    plot_latency_distribution,
    plot_error_rates,
    plot_gaze_trace,
    generate_session_report,
)
from src.analysis.metrics import SessionMetrics
from src.storage.models import GazeData, Trial


def _make_trial_dicts():
    """Create synthetic trial result dicts."""
    trials = []
    for i in range(20):
        trials.append({
            "trial_number": i + 1,
            "trial_type": "antisaccade" if i < 14 else "prosaccade",
            "stimulus_side": "left" if i % 2 == 0 else "right",
            "saccade_latency_ms": 200.0 + i * 15 if i % 3 != 0 else None,
            "response_correct": True if i % 4 != 0 else False,
            "saccade_direction": "left" if i % 2 == 0 else "right",
        })
    return trials


class TestPlotLatencyByTrial:
    def test_generates_without_error(self):
        fig, ax = plt.subplots()
        plot_latency_by_trial(_make_trial_dicts(), ax=ax)
        plt.close(fig)

    def test_creates_own_axes(self):
        fig = plot_latency_by_trial(_make_trial_dicts())
        assert fig is not None
        plt.close(fig)


class TestPlotLatencyDistribution:
    def test_generates_without_error(self):
        fig, ax = plt.subplots()
        plot_latency_distribution(_make_trial_dicts(), ax=ax)
        plt.close(fig)

    def test_creates_own_axes(self):
        fig = plot_latency_distribution(_make_trial_dicts())
        assert fig is not None
        plt.close(fig)


class TestPlotErrorRates:
    def test_generates_without_error(self):
        metrics = SessionMetrics(
            n_antisaccade_trials=14,
            n_prosaccade_trials=6,
            antisaccade_error_rate=0.25,
            prosaccade_error_rate=0.10,
            mean_antisaccade_latency_ms=300.0,
            median_antisaccade_latency_ms=280.0,
            mean_prosaccade_latency_ms=200.0,
            median_prosaccade_latency_ms=190.0,
        )
        fig, ax = plt.subplots()
        plot_error_rates(metrics, ax=ax)
        plt.close(fig)

    def test_creates_own_axes(self):
        metrics = SessionMetrics(
            n_antisaccade_trials=14,
            n_prosaccade_trials=6,
            antisaccade_error_rate=0.25,
            prosaccade_error_rate=0.10,
            mean_antisaccade_latency_ms=300.0,
            median_antisaccade_latency_ms=280.0,
            mean_prosaccade_latency_ms=200.0,
            median_prosaccade_latency_ms=190.0,
        )
        fig = plot_error_rates(metrics)
        assert fig is not None
        plt.close(fig)


class TestPlotGazeTrace:
    def test_generates_without_error(self):
        trial = Trial(
            session_id="test-session",
            trial_number=1,
            trial_type="antisaccade",
            stimulus_side="left",
            stimulus_onset_ms=1000.0,
        )
        gaze_data = []
        for i in range(60):
            gaze_data.append(GazeData(
                session_id="test-session",
                trial_number=1,
                timestamp_ms=900.0 + i * 16.67,
                left_iris_x=320.0 + i * 2.0,
                left_iris_y=240.0,
                right_iris_x=380.0 + i * 2.0,
                right_iris_y=240.0,
                confidence=0.95,
            ))
        # Identity-like calibration matrix
        cal_matrix = np.array([
            [0.1, 0.0, -32.0],
            [0.0, 0.1, -24.0],
            [0.0, 0.0, 1.0],
        ])
        fig, ax = plt.subplots()
        plot_gaze_trace(gaze_data, trial, cal_matrix, ax=ax)
        plt.close(fig)


class TestGenerateSessionReport:
    def test_generates_report_file(self, tmp_path):
        """Test full report generation with mocked repository."""
        trial_objs = [
            Trial(
                session_id="s1",
                trial_number=i + 1,
                trial_type="antisaccade" if i < 10 else "prosaccade",
                stimulus_side="left" if i % 2 == 0 else "right",
                stimulus_onset_ms=1000.0 + i * 3700.0,
                response_correct=True if i % 3 != 0 else False,
                saccade_latency_ms=200.0 + i * 10 if i % 4 != 0 else None,
            )
            for i in range(15)
        ]
        gaze_samples = [
            GazeData(
                session_id="s1",
                trial_number=1,
                timestamp_ms=900.0 + j * 16.67,
                left_iris_x=320.0 + j * 2.0,
                left_iris_y=240.0,
                right_iris_x=380.0 + j * 2.0,
                right_iris_y=240.0,
                confidence=0.95,
            )
            for j in range(60)
        ]
        cal_matrix = np.array([
            [0.1, 0.0, -32.0],
            [0.0, 0.1, -24.0],
            [0.0, 0.0, 1.0],
        ])

        repo = MagicMock()
        repo.get_trials.return_value = trial_objs
        repo.get_gaze_data.return_value = gaze_samples

        output_path = generate_session_report(
            "s1", repo, cal_matrix, output_dir=str(tmp_path)
        )
        assert output_path.exists()
        assert output_path.suffix == ".png"
