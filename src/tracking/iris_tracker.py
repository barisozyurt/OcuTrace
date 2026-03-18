"""Real-time iris tracking using MediaPipe FaceMesh.

Uses FaceMesh with refine_landmarks=True to get iris center
landmarks (468 for left, 473 for right eye).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np


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
    """MediaPipe-based iris tracker.

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

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=tracking_cfg["max_num_faces"],
            refine_landmarks=tracking_cfg["refine_landmarks"],
            min_detection_confidence=tracking_cfg["min_detection_confidence"],
            min_tracking_confidence=tracking_cfg["min_tracking_confidence"],
        )

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
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        left = face.landmark[self._left_iris_idx]
        right = face.landmark[self._right_iris_idx]

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
        self._face_mesh.close()
