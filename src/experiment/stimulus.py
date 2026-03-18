"""PsychoPy stimulus components for the gap antisaccade paradigm.

Provides frame-accurate stimulus presentation using PsychoPy's flip-based
timing. All visual timing is driven by monitor refresh counting — never
``time.sleep()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StimulusConfig:
    """Immutable configuration for stimulus presentation.

    Parameters
    ----------
    fixation_duration_ms : int
        Duration of the fixation cross phase in milliseconds.
    gap_duration_ms : int
        Duration of the gap (blank screen) phase in milliseconds.
    stimulus_duration_ms : int
        Duration of the peripheral stimulus phase in milliseconds.
    iti_min_ms : int
        Minimum inter-trial interval in milliseconds.
    iti_max_ms : int
        Maximum inter-trial interval in milliseconds.
    eccentricity_deg : float
        Horizontal eccentricity of the target in degrees of visual angle.
    refresh_hz : float
        Monitor refresh rate in Hz.
    """

    fixation_duration_ms: int
    gap_duration_ms: int
    stimulus_duration_ms: int
    iti_min_ms: int
    iti_max_ms: int
    eccentricity_deg: float
    refresh_hz: float


@dataclass
class TrialTimestamps:
    """Timestamps recorded during a single trial.

    All values are in milliseconds (UTC-epoch or clock-relative).

    Parameters
    ----------
    fixation_onset_ms : float
        Timestamp of the first fixation-phase flip.
    gap_onset_ms : float
        Timestamp of the first gap-phase flip.
    stimulus_onset_ms : float
        Timestamp of the first stimulus-phase flip.
    stimulus_offset_ms : float
        Timestamp of the last stimulus-phase flip.
    iti_onset_ms : float
        Timestamp of the first ITI-phase flip.
    """

    fixation_onset_ms: float = 0.0
    gap_onset_ms: float = 0.0
    stimulus_onset_ms: float = 0.0
    stimulus_offset_ms: float = 0.0
    iti_onset_ms: float = 0.0


# ---------------------------------------------------------------------------
# Pure-logic helpers (no PsychoPy dependency)
# ---------------------------------------------------------------------------

def create_stimulus_config(
    paradigm: dict[str, Any],
    monitor_refresh_hz: float = 60.0,
) -> StimulusConfig:
    """Map a paradigm configuration dict to a ``StimulusConfig``.

    Parameters
    ----------
    paradigm : dict[str, Any]
        Paradigm section from ``settings.yaml``. Must contain keys
        ``fixation_duration_ms``, ``gap_duration_ms``,
        ``stimulus_duration_ms``, ``iti_min_ms``, ``iti_max_ms``, and
        ``stimulus_eccentricity_deg``.
    monitor_refresh_hz : float
        Monitor refresh rate in Hz (default 60.0).

    Returns
    -------
    StimulusConfig
        Frozen dataclass with the mapped values.
    """
    return StimulusConfig(
        fixation_duration_ms=paradigm["fixation_duration_ms"],
        gap_duration_ms=paradigm["gap_duration_ms"],
        stimulus_duration_ms=paradigm["stimulus_duration_ms"],
        iti_min_ms=paradigm["iti_min_ms"],
        iti_max_ms=paradigm["iti_max_ms"],
        eccentricity_deg=paradigm["stimulus_eccentricity_deg"],
        refresh_hz=monitor_refresh_hz,
    )


def compute_trial_frame_counts(
    fixation_ms: int,
    gap_ms: int,
    stimulus_ms: int,
    refresh_hz: float,
) -> dict[str, int]:
    """Convert millisecond durations to frame counts.

    Parameters
    ----------
    fixation_ms : int
        Fixation phase duration in milliseconds.
    gap_ms : int
        Gap phase duration in milliseconds.
    stimulus_ms : int
        Stimulus phase duration in milliseconds.
    refresh_hz : float
        Monitor refresh rate in Hz.

    Returns
    -------
    dict[str, int]
        Mapping of phase name to frame count, with keys
        ``"fixation"``, ``"gap"``, and ``"stimulus"``.
    """
    return {
        "fixation": round(fixation_ms * refresh_hz / 1000),
        "gap": round(gap_ms * refresh_hz / 1000),
        "stimulus": round(stimulus_ms * refresh_hz / 1000),
    }


def compute_iti_frame_count(
    iti_min_ms: int,
    iti_max_ms: int,
    refresh_hz: float,
    rng: np.random.Generator,
) -> int:
    """Compute a random inter-trial interval in frames.

    Draws a uniform random frame count between the minimum and maximum
    ITI durations (inclusive) converted to frames.

    Parameters
    ----------
    iti_min_ms : int
        Minimum ITI duration in milliseconds.
    iti_max_ms : int
        Maximum ITI duration in milliseconds.
    refresh_hz : float
        Monitor refresh rate in Hz.
    rng : numpy.random.Generator
        Random number generator instance.

    Returns
    -------
    int
        Number of frames for this ITI.
    """
    min_frames = round(iti_min_ms * refresh_hz / 1000)
    max_frames = round(iti_max_ms * refresh_hz / 1000)
    return int(rng.integers(min_frames, max_frames + 1))


# ---------------------------------------------------------------------------
# PsychoPy visual components (lazy imports)
# ---------------------------------------------------------------------------

def create_fixation_dot(win: Any, color: str = "white") -> Any:
    """Create a circular fixation dot.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window to draw into.
    color : str
        Dot color (e.g. "white", "red", "green").

    Returns
    -------
    psychopy.visual.Circle
        Fixation dot, 0.3 deg radius.
    """
    from psychopy import visual  # lazy import

    return visual.Circle(
        win,
        radius=0.3,
        pos=(0, 0),
        fillColor=color,
        lineColor=color,
        units="deg",
    )


# Keep old name as alias for backward compatibility
def create_fixation_cross(win: Any) -> Any:
    """Create a fixation stimulus (dot).

    .. deprecated::
        Use :func:`create_fixation_dot` instead.
    """
    return create_fixation_dot(win, color="white")


def create_target(
    win: Any,
    side: str,
    eccentricity_deg: float,
) -> Any:
    """Create a peripheral target circle.

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window to draw into.
    side : str
        ``"left"`` or ``"right"``.
    eccentricity_deg : float
        Horizontal offset from center in degrees of visual angle.

    Returns
    -------
    psychopy.visual.Circle
        White circle at the specified horizontal position with radius 0.5 deg.

    Raises
    ------
    ValueError
        If *side* is not ``"left"`` or ``"right"``.
    """
    from psychopy import visual  # lazy import

    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    x_pos = -eccentricity_deg if side == "left" else eccentricity_deg

    return visual.Circle(
        win,
        radius=0.5,
        pos=(x_pos, 0),
        fillColor="white",
        lineColor="white",
        units="deg",
    )


# ---------------------------------------------------------------------------
# Trial runner (frame-accurate timing)
# ---------------------------------------------------------------------------

def run_single_trial(
    win: Any,
    clock: Any,
    fixation: Any,
    side: str,
    config: StimulusConfig,
    rng: np.random.Generator,
    on_frame: Callable[[str, int, float], None] | None = None,
) -> TrialTimestamps:
    """Run a single trial of the gap antisaccade paradigm.

    All timing is driven by ``win.flip()`` frame counting — ``time.sleep()``
    is **never** used.

    Phase sequence: fixation -> gap (blank) -> stimulus -> ITI (blank).

    Parameters
    ----------
    win : psychopy.visual.Window
        PsychoPy window.
    clock : psychopy.core.Clock
        Reserved for future use (e.g., global session clock). Currently
        timestamps come from ``win.flip()`` return values.
    fixation : psychopy.visual.ShapeStim
        Pre-created fixation cross stimulus.
    side : str
        ``"left"`` or ``"right"`` — where the target appears.
    config : StimulusConfig
        Timing and eccentricity parameters.
    rng : numpy.random.Generator
        Random number generator (used for ITI jitter).
    on_frame : callable or None
        Optional callback invoked each frame with
        ``(phase: str, frame_idx: int, timestamp_ms: float)``.

    Returns
    -------
    TrialTimestamps
        Timestamps (in ms) recorded at the first flip of each phase, plus
        ``stimulus_offset_ms`` at the last stimulus flip.
    """
    frames = compute_trial_frame_counts(
        config.fixation_duration_ms,
        config.gap_duration_ms,
        config.stimulus_duration_ms,
        config.refresh_hz,
    )
    iti_frames = compute_iti_frame_count(
        config.iti_min_ms, config.iti_max_ms, config.refresh_hz, rng
    )

    target = create_target(win, side, config.eccentricity_deg)
    ts = TrialTimestamps()

    # --- Fixation phase ---
    for i in range(frames["fixation"]):
        fixation.draw()
        flip_time = win.flip()
        timestamp_ms = flip_time * 1000.0
        if i == 0:
            ts.fixation_onset_ms = timestamp_ms
        if on_frame is not None:
            on_frame("fixation", i, timestamp_ms)

    # --- Gap phase (blank screen) ---
    for i in range(frames["gap"]):
        flip_time = win.flip()
        timestamp_ms = flip_time * 1000.0
        if i == 0:
            ts.gap_onset_ms = timestamp_ms
        if on_frame is not None:
            on_frame("gap", i, timestamp_ms)

    # --- Stimulus phase ---
    for i in range(frames["stimulus"]):
        target.draw()
        flip_time = win.flip()
        timestamp_ms = flip_time * 1000.0
        if i == 0:
            ts.stimulus_onset_ms = timestamp_ms
        if on_frame is not None:
            on_frame("stimulus", i, timestamp_ms)

    # --- ITI phase (blank screen) ---
    for i in range(iti_frames):
        flip_time = win.flip()
        timestamp_ms = flip_time * 1000.0
        if i == 0:
            ts.iti_onset_ms = timestamp_ms
            # The first ITI flip is the true stimulus offset (screen cleared)
            ts.stimulus_offset_ms = timestamp_ms
        if on_frame is not None:
            on_frame("iti", i, timestamp_ms)

    return ts
