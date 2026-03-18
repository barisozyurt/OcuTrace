"""Tests for glasses detection and tracking quality gate."""
import numpy as np
import pytest

from src.tracking.glasses_detector import GlassesDetector, assess_tracking_quality
from src.tracking.iris_tracker import IrisCoordinates
from src.storage.models import TrackingQuality


def _make_stable_coords(n: int, base_x: float = 320.0, jitter: float = 0.5):
    """Generate n stable iris coordinates with small jitter."""
    rng = np.random.default_rng(42)
    return [
        IrisCoordinates(
            left_x=base_x + rng.uniform(-jitter, jitter),
            left_y=240.0 + rng.uniform(-jitter, jitter),
            right_x=base_x + 1.0 + rng.uniform(-jitter, jitter),
            right_y=240.0 + rng.uniform(-jitter, jitter),
            confidence=0.95,
            timestamp_ms=float(i * 33),
        )
        for i in range(n)
    ]


def _make_jittery_coords(n: int, base_x: float = 320.0, jitter: float = 5.0):
    """Generate n coords with high jitter (simulating glasses interference)."""
    rng = np.random.default_rng(42)
    return [
        IrisCoordinates(
            left_x=base_x + rng.uniform(-jitter, jitter),
            left_y=240.0 + rng.uniform(-jitter, jitter),
            right_x=base_x + 1.0 + rng.uniform(-jitter, jitter),
            right_y=240.0 + rng.uniform(-jitter, jitter),
            confidence=0.7,
            timestamp_ms=float(i * 33),
        )
        for i in range(n)
    ]


class TestAssessTrackingQuality:
    def test_stable_tracking_is_acceptable(self):
        coords = _make_stable_coords(90)
        quality = assess_tracking_quality(
            coords_history=coords,
            none_count=0,
            total_frames=90,
            jitter_threshold_px=2.0,
            min_detection_rate=0.95,
        )
        assert quality.quality_acceptable is True
        assert quality.detection_rate > 0.95
        assert quality.mean_jitter_px < 2.0

    def test_jittery_tracking_is_unacceptable(self):
        coords = _make_jittery_coords(90)
        quality = assess_tracking_quality(
            coords_history=coords,
            none_count=0,
            total_frames=90,
            jitter_threshold_px=2.0,
            min_detection_rate=0.95,
        )
        assert quality.quality_acceptable is False
        assert quality.mean_jitter_px > 2.0

    def test_low_detection_rate_is_unacceptable(self):
        coords = _make_stable_coords(50)
        quality = assess_tracking_quality(
            coords_history=coords,
            none_count=50,
            total_frames=100,
            jitter_threshold_px=2.0,
            min_detection_rate=0.95,
        )
        assert quality.quality_acceptable is False
        assert quality.detection_rate == 0.50


class TestGlassesDetector:
    def test_no_glasses_landmark_ratio(self):
        detector = GlassesDetector()
        assert isinstance(detector.detect_from_landmarks([]), bool)

    def test_glasses_detection_returns_bool(self):
        detector = GlassesDetector()
        result = detector.detect_from_landmarks([])
        assert isinstance(result, bool)
