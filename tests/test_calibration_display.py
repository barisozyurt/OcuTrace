"""Tests for calibration display target generation."""
import pytest

from src.tracking.calibration_display import generate_calibration_targets


class TestGenerateCalibrationTargets:
    def test_9_point_grid(self):
        targets = generate_calibration_targets(n_points=9, eccentricity_deg=10.0)
        assert len(targets) == 9
        center = [t for t in targets if t == (0.0, 0.0)]
        assert len(center) == 1

    def test_5_point_cross(self):
        targets = generate_calibration_targets(n_points=5, eccentricity_deg=10.0)
        assert len(targets) == 5
        assert (0.0, 0.0) in targets

    def test_targets_within_eccentricity(self):
        targets = generate_calibration_targets(n_points=9, eccentricity_deg=10.0)
        for x, y in targets:
            assert abs(x) <= 10.0
            assert abs(y) <= 10.0

    def test_invalid_n_points(self):
        with pytest.raises(ValueError):
            generate_calibration_targets(n_points=3, eccentricity_deg=10.0)
