"""Tests for iris tracker.

Uses mock for MediaPipe to avoid camera dependency in CI.
"""
from unittest.mock import MagicMock, patch

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
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = 0.0
    return lm


def _make_mock_results(left_x, left_y, right_x, right_y, num_landmarks=478):
    landmarks = [MagicMock() for _ in range(num_landmarks)]
    landmarks[468] = _make_mock_landmark(left_x, left_y)
    landmarks[473] = _make_mock_landmark(right_x, right_y)
    face = MagicMock()
    face.landmark = landmarks
    results = MagicMock()
    results.multi_face_landmarks = [face]
    return results


def _make_mock_results_no_face():
    results = MagicMock()
    results.multi_face_landmarks = None
    return results


class TestIrisTracker:
    def test_process_frame_returns_coordinates(self):
        tracker = IrisTracker.__new__(IrisTracker)
        tracker._face_mesh = MagicMock()
        tracker._frame_width = 640
        tracker._frame_height = 480
        tracker._left_iris_idx = 468
        tracker._right_iris_idx = 473

        mock_results = _make_mock_results(0.5, 0.5, 0.502, 0.501)
        tracker._face_mesh.process.return_value = mock_results

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coords = tracker.process_frame(frame, timestamp_ms=100.0)

        assert coords is not None
        assert abs(coords.left_x - 320.0) < 0.1
        assert abs(coords.right_x - 321.28) < 0.1
        assert coords.timestamp_ms == 100.0

    def test_process_frame_no_face_returns_none(self):
        tracker = IrisTracker.__new__(IrisTracker)
        tracker._face_mesh = MagicMock()
        tracker._frame_width = 640
        tracker._frame_height = 480
        tracker._left_iris_idx = 468
        tracker._right_iris_idx = 473

        tracker._face_mesh.process.return_value = _make_mock_results_no_face()

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
            },
            "camera": {
                "frame_width": 640,
                "frame_height": 480,
            },
        }
        with patch("src.tracking.iris_tracker.mp") as mock_mp:
            mock_mp.solutions.face_mesh.FaceMesh.return_value = MagicMock()
            tracker = IrisTracker(config)
            assert tracker._left_iris_idx == 468
            assert tracker._right_iris_idx == 473
