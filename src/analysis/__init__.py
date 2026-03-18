"""Saccade detection, latency computation, and clinical metrics."""
from src.analysis.signal_processing import smooth_positions, compute_velocity
from src.analysis.saccade_detector import SaccadeEvent, detect_saccades, classify_direction
from src.analysis.metrics import (
    compute_saccade_latency,
    classify_response,
    compute_session_metrics,
    SessionMetrics,
)

__all__ = [
    "smooth_positions",
    "compute_velocity",
    "SaccadeEvent",
    "detect_saccades",
    "classify_direction",
    "compute_saccade_latency",
    "classify_response",
    "compute_session_metrics",
    "SessionMetrics",
]
