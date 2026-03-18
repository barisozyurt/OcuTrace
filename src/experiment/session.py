"""Session lifecycle management for gap antisaccade experiments.

Orchestrates: calibration lookup → trial sequencing → gaze collection →
saccade analysis → persistence.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.storage.models import GazeData
from src.tracking.iris_tracker import IrisTracker, IrisCoordinates
from src.tracking.calibration import apply_transform
from src.experiment.paradigm import TrialSpec
from src.analysis.signal_processing import smooth_positions, compute_velocity
from src.analysis.saccade_detector import detect_saccades, classify_direction
from src.analysis.metrics import (
    compute_saccade_latency,
    classify_response,
    SessionMetrics,
)

logger = logging.getLogger(__name__)


class GazeCollector:
    """Thread-safe gaze data collector using camera + iris tracker.

    Runs camera capture in a background thread.  The ``on_frame``
    callback (called from PsychoPy's flip loop) records the latest
    iris coordinates into a sample list.

    Parameters
    ----------
    tracker : IrisTracker
        Initialized iris tracker.
    cap : cv2.VideoCapture
        Open camera capture.
    session_id : str
        Current session UUID.
    """

    def __init__(
        self,
        tracker: IrisTracker,
        cap: cv2.VideoCapture,
        session_id: str,
    ) -> None:
        self._tracker = tracker
        self._cap = cap
        self._session_id = session_id
        self._trial_number = 0
        self._samples: list[GazeData] = []
        self._lock = threading.Lock()
        self._latest_coords: Optional[IrisCoordinates] = None
        self._latest_frame_id: int = 0
        self._last_recorded_frame_id: int = -1  # deduplicate webcam frames
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background camera capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background capture."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def set_trial(self, trial_number: int) -> None:
        """Set current trial number for gaze data tagging."""
        self._trial_number = trial_number

    def on_frame(self, phase: str, frame_idx: int, timestamp_ms: float) -> None:
        """PsychoPy on_frame callback — records latest iris position.

        Uses the PsychoPy flip timestamp (not the camera timestamp) so
        that gaze data and stimulus timestamps share the same clock.

        Parameters
        ----------
        phase : str
            Trial phase name (fixation, gap, stimulus, iti).
        frame_idx : int
            Frame index within the current phase.
        timestamp_ms : float
            Flip timestamp in milliseconds (from PsychoPy clock).
        """
        with self._lock:
            coords = self._latest_coords
            frame_id = self._latest_frame_id
        if coords is None:
            return
        # Skip duplicate webcam frames (webcam ~30fps, PsychoPy ~60fps)
        if frame_id == self._last_recorded_frame_id:
            return
        self._last_recorded_frame_id = frame_id
        self._samples.append(
            GazeData(
                session_id=self._session_id,
                trial_number=self._trial_number,
                timestamp_ms=timestamp_ms,  # PsychoPy clock for sync
                left_iris_x=coords.left_x,
                left_iris_y=coords.left_y,
                right_iris_x=coords.right_x,
                right_iris_y=coords.right_y,
                confidence=coords.confidence,
            )
        )

    def get_trial_samples(self, trial_number: int) -> list[GazeData]:
        """Get collected samples for a specific trial."""
        return [s for s in self._samples if s.trial_number == trial_number]

    def get_all_samples(self) -> list[GazeData]:
        """Get all collected gaze samples."""
        return list(self._samples)

    def _capture_loop(self) -> None:
        """Background loop: read frames and update latest coordinates."""
        frame_counter = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            frame_counter += 1
            ts = time.monotonic() * 1000.0
            coords = self._tracker.process_frame(frame, ts)
            with self._lock:
                self._latest_coords = coords
                self._latest_frame_id = frame_counter


def analyze_trial(
    gaze_samples: list[GazeData],
    trial_spec: TrialSpec,
    stimulus_onset_ms: float,
    calibration_matrix: np.ndarray,
    saccade_cfg: dict,
) -> tuple[Optional[str], Optional[float], Optional[bool]]:
    """Analyze a single trial's gaze data for saccade detection.

    Parameters
    ----------
    gaze_samples : list[GazeData]
        Raw gaze data collected during this trial.
    trial_spec : TrialSpec
        Trial specification (type, side).
    stimulus_onset_ms : float
        Stimulus onset timestamp.
    calibration_matrix : np.ndarray
        3x3 pixel-to-degree affine transform.
    saccade_cfg : dict
        Saccade detection config section from settings.yaml.

    Returns
    -------
    tuple[Optional[str], Optional[float], Optional[bool]]
        ``(saccade_direction, saccade_latency_ms, response_correct)``.
        All ``None`` if no saccade detected or insufficient data.
    """
    if len(gaze_samples) < 5:
        logger.warning(
            "Trial %d: only %d gaze samples, skipping analysis",
            trial_spec.trial_number,
            len(gaze_samples),
        )
        return None, None, None

    # Convert pixels to degrees (mean of both eyes, X only for horizontal saccades)
    positions_deg: list[float] = []
    timestamps_ms: list[float] = []
    for g in gaze_samples:
        mean_x = (g.left_iris_x + g.right_iris_x) / 2.0
        mean_y = (g.left_iris_y + g.right_iris_y) / 2.0
        x_deg, _ = apply_transform(calibration_matrix, mean_x, mean_y)
        positions_deg.append(x_deg)
        timestamps_ms.append(g.timestamp_ms)

    pos_arr = np.array(positions_deg)
    ts_arr = np.array(timestamps_ms)

    # Smooth and compute velocity
    smoothed = smooth_positions(
        pos_arr,
        window=saccade_cfg["smoothing_window"],
        polyorder=saccade_cfg["smoothing_polyorder"],
    )
    velocity = compute_velocity(smoothed, ts_arr)

    # Detect saccades
    events = detect_saccades(
        velocity,
        ts_arr,
        onset_threshold=saccade_cfg["onset_velocity_threshold"],
        offset_threshold=saccade_cfg["offset_velocity_threshold"],
        min_onset_frames=saccade_cfg["min_onset_frames"],
    )

    max_vel = float(np.max(velocity)) if len(velocity) > 0 else 0.0
    logger.debug(
        "Trial %d: %d samples, max_velocity=%.1f deg/s, %d saccades detected",
        trial_spec.trial_number, len(gaze_samples), max_vel, len(events),
    )

    if not events:
        return None, None, None

    # Find first saccade near or after stimulus onset (allow 80 ms anticipatory)
    first = None
    for event in events:
        if event.onset_ms >= stimulus_onset_ms - 80.0:
            first = event
            break

    if first is None:
        return None, None, None

    direction = classify_direction(smoothed, first.onset_idx, first.offset_idx)
    latency = compute_saccade_latency(stimulus_onset_ms, first.onset_ms)
    correct = classify_response(
        trial_spec.trial_type, trial_spec.stimulus_side, direction
    )

    return direction, latency, correct


def print_session_summary(metrics: SessionMetrics) -> None:
    """Print formatted session summary to console.

    Parameters
    ----------
    metrics : SessionMetrics
        Aggregate clinical metrics for the session.
    """
    print("\n" + "=" * 60)
    print("SESSION RESULTS")
    print("=" * 60)
    print(f"Antisaccade trials:     {metrics.n_antisaccade_trials}")
    print(f"Prosaccade trials:      {metrics.n_prosaccade_trials}")
    print(f"Antisaccade error rate: {metrics.antisaccade_error_rate:.1%}")
    print(f"Prosaccade error rate:  {metrics.prosaccade_error_rate:.1%}")
    if metrics.mean_antisaccade_latency_ms is not None:
        print(
            f"Mean anti latency:      {metrics.mean_antisaccade_latency_ms:.1f} ms"
        )
    if metrics.median_antisaccade_latency_ms is not None:
        print(
            f"Median anti latency:    {metrics.median_antisaccade_latency_ms:.1f} ms"
        )
    if metrics.mean_prosaccade_latency_ms is not None:
        print(
            f"Mean pro latency:       {metrics.mean_prosaccade_latency_ms:.1f} ms"
        )
    if metrics.median_prosaccade_latency_ms is not None:
        print(
            f"Median pro latency:     {metrics.median_prosaccade_latency_ms:.1f} ms"
        )
    print("=" * 60)
