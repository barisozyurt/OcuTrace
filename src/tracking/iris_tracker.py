"""Real-time iris tracking using MediaPipe FaceLandmarker.

Uses the Tasks API (mediapipe.tasks) with FaceLandmarker to get
iris center landmarks (468 for left, 473 for right eye).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


@dataclass
class IrisCoordinates:
    """Iris center coordinates from a single frame.

    Parameters
    ----------
    left_x : float
        Left iris center X in pixels.
    left_y : float
        Left iris center Y in pixels.
    right_x : float
        Right iris center X in pixels.
    right_y : float
        Right iris center Y in pixels.
    confidence : float
        Detection confidence (0-1).
    timestamp_ms : float
        Frame timestamp in UTC Unix milliseconds.
    """

    left_x: float
    left_y: float
    right_x: float
    right_y: float
    confidence: float
    timestamp_ms: float

    @property
    def mean_x(self) -> float:
        """Mean X of both iris centers."""
        return (self.left_x + self.right_x) / 2.0

    @property
    def mean_y(self) -> float:
        """Mean Y of both iris centers."""
        return (self.left_y + self.right_y) / 2.0


class IrisTracker:
    """MediaPipe-based iris tracker using the Tasks API.

    Parameters
    ----------
    config : dict
        Configuration dict with 'tracking' and 'camera' sections.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        tracking_cfg = config["tracking"]
        camera_cfg = config["camera"]

        self._left_iris_idx: int = tracking_cfg["left_iris_index"]
        self._right_iris_idx: int = tracking_cfg["right_iris_index"]
        self._frame_width: int = camera_cfg["frame_width"]
        self._frame_height: int = camera_cfg["frame_height"]

        from src.paths import get_bundle_dir

        raw_model_path = tracking_cfg.get(
            "model_path", "models/face_landmarker.task"
        )
        model_path = Path(raw_model_path)
        if not model_path.is_absolute():
            model_path = get_bundle_dir() / model_path

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=tracking_cfg["max_num_faces"],
            min_face_detection_confidence=tracking_cfg[
                "min_detection_confidence"
            ],
            min_face_presence_confidence=tracking_cfg.get(
                "min_presence_confidence", 0.5
            ),
            min_tracking_confidence=tracking_cfg["min_tracking_confidence"],
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_counter = 0

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
    ) -> Optional[IrisCoordinates]:
        """Extract iris coordinates from a BGR camera frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (H, W, 3).
        timestamp_ms : float
            Timestamp of this frame in UTC Unix ms.

        Returns
        -------
        IrisCoordinates or None
            Iris positions if face detected, None otherwise.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=rgb_frame
        )

        self._frame_counter += 1
        result = self._landmarker.detect_for_video(
            mp_image, self._frame_counter
        )

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        if len(landmarks) <= max(self._left_iris_idx, self._right_iris_idx):
            return None

        left = landmarks[self._left_iris_idx]
        right = landmarks[self._right_iris_idx]

        return IrisCoordinates(
            left_x=left.x * self._frame_width,
            left_y=left.y * self._frame_height,
            right_x=right.x * self._frame_width,
            right_y=right.y * self._frame_height,
            confidence=1.0,
            timestamp_ms=timestamp_ms,
        )

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
