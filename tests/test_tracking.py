"""Tests for iris tracker.

Uses mock for MediaPipe to avoid camera dependency and model file in CI.
"""
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.tracking.iris_tracker import IrisTracker, IrisCoordinates


class TestIrisCoordinates:
    def test_create(self):
        coords = IrisCoordinates(
            left_x=320.0,
            left_y=240.0,
            right_x=321.0,
            right_y=240.5,
            confidence=0.95,
            timestamp_ms=1000.0,
        )
        assert coords.left_x == 320.0
        assert coords.confidence == 0.95

    def test_mean_x(self):
        coords = IrisCoordinates(
            left_x=320.0, left_y=240.0,
            right_x=322.0, right_y=240.0,
            confidence=0.95, timestamp_ms=0.0,
        )
        assert coords.mean_x == 321.0

    def test_mean_y(self):
        coords = IrisCoordinates(
            left_x=320.0, left_y=238.0,
            right_x=322.0, right_y=242.0,
            confidence=0.95, timestamp_ms=0.0,
        )
        assert coords.mean_y == 240.0


def _make_mock_landmark(x: float, y: float):
    """Create a mock landmark with x, y attributes."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = 0.0
    return lm


def _make_mock_result_with_face(left_x, left_y, right_x, right_y, num_landmarks=478):
    """Create mock FaceLandmarkerResult with landmarks."""
    landmarks = [_make_mock_landmark(0.5, 0.5) for _ in range(num_landmarks)]
    landmarks[468] = _make_mock_landmark(left_x, left_y)
    landmarks[473] = _make_mock_landmark(right_x, right_y)

    result = MagicMock()
    result.face_landmarks = [landmarks]
    return result


def _make_mock_result_no_face():
    """Create mock FaceLandmarkerResult with no face."""
    result = MagicMock()
    result.face_landmarks = []
    return result


class TestIrisTracker:
    def test_process_frame_returns_coordinates(self):
        tracker = IrisTracker.__new__(IrisTracker)
        tracker._landmarker = MagicMock()
        tracker._frame_width = 640
        tracker._frame_height = 480
        tracker._left_iris_idx = 468
        tracker._right_iris_idx = 473
        tracker._frame_counter = 0

        mock_result = _make_mock_result_with_face(0.5, 0.5, 0.502, 0.501)
        tracker._landmarker.detect_for_video.return_value = mock_result

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coords = tracker.process_frame(frame, timestamp_ms=100.0)

        assert coords is not None
        assert abs(coords.left_x - 320.0) < 0.1
        assert abs(coords.right_x - 321.28) < 0.1
        assert coords.timestamp_ms == 100.0

    def test_process_frame_no_face_returns_none(self):
        tracker = IrisTracker.__new__(IrisTracker)
        tracker._landmarker = MagicMock()
        tracker._frame_width = 640
        tracker._frame_height = 480
        tracker._left_iris_idx = 468
        tracker._right_iris_idx = 473
        tracker._frame_counter = 0

        tracker._landmarker.detect_for_video.return_value = _make_mock_result_no_face()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coords = tracker.process_frame(frame, timestamp_ms=100.0)
        assert coords is None

    def test_tracker_config_from_settings(self):
        config = {
            "tracking": {
                "max_num_faces": 1,
                "refine_landmarks": True,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "left_iris_index": 468,
                "right_iris_index": 473,
                "model_path": "models/face_landmarker.task",
            },
            "camera": {
                "frame_width": 640,
                "frame_height": 480,
            },
        }
        with patch("src.tracking.iris_tracker.FaceLandmarker") as mock_fl:
            mock_fl.create_from_options.return_value = MagicMock()
            tracker = IrisTracker(config)
            assert tracker._left_iris_idx == 468
            assert tracker._right_iris_idx == 473
            mock_fl.create_from_options.assert_called_once()
