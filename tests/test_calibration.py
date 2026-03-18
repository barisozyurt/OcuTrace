"""Tests for calibration system."""
import numpy as np
import pytest

from src.storage.models import CalibrationPoint, CalibrationResult
from src.tracking.calibration import (
    fit_pixel_to_degree_transform,
    apply_transform,
    compute_calibration_error,
    create_calibration_result,
)


class TestCalibrationModels:
    def test_create_calibration_point(self):
        point = CalibrationPoint(
            target_x_deg=10.0, target_y_deg=0.0,
            measured_x_px=450.0, measured_y_px=240.0,
        )
        assert point.target_x_deg == 10.0
        assert point.measured_x_px == 450.0

    def test_create_calibration_result(self):
        points = [
            CalibrationPoint(float(x), 0.0, 320.0 + x * 13.0, 240.0)
            for x in [-10, -5, 0, 5, 10]
        ]
        result = CalibrationResult(
            session_id="test-123", points=points,
            transform_matrix=np.eye(3).tolist(),
            mean_error_deg=0.5, accepted=True,
        )
        assert result.accepted is True
        assert len(result.points) == 5


class TestPixelToDegreeTransform:
    def test_fit_identity_like_transform(self):
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                px = 320.0 + tx * 13.0
                py = 240.0 + ty * 13.0
                points.append(CalibrationPoint(tx, ty, px, py))
        matrix = fit_pixel_to_degree_transform(points)
        assert matrix.shape == (3, 3)

        deg_x, deg_y = apply_transform(matrix, 320.0, 240.0)
        assert abs(deg_x) < 0.1
        assert abs(deg_y) < 0.1

        deg_x, deg_y = apply_transform(matrix, 320.0 + 10.0 * 13.0, 240.0)
        assert abs(deg_x - 10.0) < 0.1
        assert abs(deg_y) < 0.1

    def test_fit_with_noise(self):
        rng = np.random.default_rng(42)
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                px = 320.0 + tx * 13.0 + rng.normal(0, 1.0)
                py = 240.0 + ty * 13.0 + rng.normal(0, 1.0)
                points.append(CalibrationPoint(tx, ty, px, py))
        matrix = fit_pixel_to_degree_transform(points)
        deg_x, deg_y = apply_transform(matrix, 320.0, 240.0)
        assert abs(deg_x) < 1.0
        assert abs(deg_y) < 1.0

    def test_apply_transform_returns_tuple(self):
        matrix = np.eye(3)
        result = apply_transform(matrix, 100.0, 200.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fit_requires_minimum_points(self):
        points = [
            CalibrationPoint(0.0, 0.0, 320.0, 240.0),
            CalibrationPoint(10.0, 0.0, 450.0, 240.0),
        ]
        with pytest.raises(ValueError, match="at least 3"):
            fit_pixel_to_degree_transform(points)


class TestCalibrationError:
    def test_perfect_calibration_zero_error(self):
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(tx, ty, 320.0 + tx * 13.0, 240.0 + ty * 13.0))
        matrix = fit_pixel_to_degree_transform(points)
        error = compute_calibration_error(matrix, points)
        assert error < 0.01

    def test_noisy_calibration_bounded_error(self):
        rng = np.random.default_rng(42)
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(
                    tx, ty,
                    320.0 + tx * 13.0 + rng.normal(0, 2.0),
                    240.0 + ty * 13.0 + rng.normal(0, 2.0),
                ))
        matrix = fit_pixel_to_degree_transform(points)
        error = compute_calibration_error(matrix, points)
        assert error < 2.0


class TestCreateCalibrationResult:
    def test_accepted_calibration(self):
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(tx, ty, 320.0 + tx * 13.0, 240.0 + ty * 13.0))
        result = create_calibration_result("test-123", points, max_error_deg=2.0)
        assert result.accepted is True
        assert result.mean_error_deg < 0.01
        assert len(result.transform_matrix) == 3

    def test_rejected_calibration_high_noise(self):
        rng = np.random.default_rng(42)
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(
                    tx, ty,
                    320.0 + rng.normal(0, 100.0),
                    240.0 + rng.normal(0, 100.0),
                ))
        result = create_calibration_result("test-456", points, max_error_deg=2.0)
        assert result.mean_error_deg > 2.0
        assert result.accepted is False
