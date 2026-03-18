"""Saccade latency calculation, response classification, and session metrics.

Provides the key clinical outputs for OcuTrace: antisaccade error rate,
saccade latency, and antisaccade latency.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional


def compute_saccade_latency(
    stimulus_onset_ms: float, saccade_onset_ms: float
) -> float:
    """Return saccade latency in milliseconds.

    Parameters
    ----------
    stimulus_onset_ms : float
        Timestamp of peripheral stimulus onset (ms).
    saccade_onset_ms : float
        Timestamp of saccade onset (ms).

    Returns
    -------
    float
        Latency in ms. Negative values indicate anticipatory saccades.
    """
    return saccade_onset_ms - stimulus_onset_ms


def classify_response(
    trial_type: str, stimulus_side: str, saccade_direction: str
) -> bool:
    """Classify whether a saccade response is correct.

    Parameters
    ----------
    trial_type : str
        ``"antisaccade"`` or ``"prosaccade"``.
    stimulus_side : str
        ``"left"`` or ``"right"`` — side where the stimulus appeared.
    saccade_direction : str
        ``"left"`` or ``"right"`` — direction of the first saccade.

    Returns
    -------
    bool
        ``True`` if the response is correct for the given trial type.
    """
    if trial_type == "antisaccade":
        return saccade_direction != stimulus_side
    # prosaccade
    return saccade_direction == stimulus_side


@dataclass(frozen=True)
class SessionMetrics:
    """Aggregated metrics for a session of saccade/antisaccade trials.

    Attributes
    ----------
    n_antisaccade_trials : int
        Total number of antisaccade trials.
    n_prosaccade_trials : int
        Total number of prosaccade trials.
    antisaccade_error_rate : float
        Fraction of classified antisaccade trials answered incorrectly.
    prosaccade_error_rate : float
        Fraction of classified prosaccade trials answered incorrectly.
    mean_antisaccade_latency_ms : float or None
        Mean latency across antisaccade trials with valid latency.
    median_antisaccade_latency_ms : float or None
        Median latency across antisaccade trials with valid latency.
    mean_prosaccade_latency_ms : float or None
        Mean latency across prosaccade trials with valid latency.
    median_prosaccade_latency_ms : float or None
        Median latency across prosaccade trials with valid latency.
    """

    n_antisaccade_trials: int
    n_prosaccade_trials: int
    antisaccade_error_rate: float
    prosaccade_error_rate: float
    mean_antisaccade_latency_ms: Optional[float]
    median_antisaccade_latency_ms: Optional[float]
    mean_prosaccade_latency_ms: Optional[float]
    median_prosaccade_latency_ms: Optional[float]


def compute_session_metrics(trials: list[dict]) -> SessionMetrics:
    """Compute aggregate metrics from a list of trial result dicts.

    Parameters
    ----------
    trials : list[dict]
        Each dict must contain keys ``trial_type`` (``"antisaccade"`` or
        ``"prosaccade"``), ``response_correct`` (``bool`` or ``None``),
        and ``saccade_latency_ms`` (``float`` or ``None``).

    Returns
    -------
    SessionMetrics
        Aggregated session-level metrics.
    """
    anti_trials = [t for t in trials if t["trial_type"] == "antisaccade"]
    pro_trials = [t for t in trials if t["trial_type"] == "prosaccade"]

    def _error_rate(subset: list[dict]) -> float:
        classified = [t for t in subset if t["response_correct"] is not None]
        if not classified:
            return 0.0
        errors = sum(1 for t in classified if not t["response_correct"])
        return errors / len(classified)

    def _latency_stats(
        subset: list[dict],
    ) -> tuple[Optional[float], Optional[float]]:
        latencies = [
            t["saccade_latency_ms"]
            for t in subset
            if t["saccade_latency_ms"] is not None
        ]
        if not latencies:
            return None, None
        return statistics.mean(latencies), statistics.median(latencies)

    anti_mean, anti_median = _latency_stats(anti_trials)
    pro_mean, pro_median = _latency_stats(pro_trials)

    return SessionMetrics(
        n_antisaccade_trials=len(anti_trials),
        n_prosaccade_trials=len(pro_trials),
        antisaccade_error_rate=_error_rate(anti_trials),
        prosaccade_error_rate=_error_rate(pro_trials),
        mean_antisaccade_latency_ms=anti_mean,
        median_antisaccade_latency_ms=anti_median,
        mean_prosaccade_latency_ms=pro_mean,
        median_prosaccade_latency_ms=pro_median,
    )
