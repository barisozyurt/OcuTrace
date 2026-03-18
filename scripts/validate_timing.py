"""Validate stimulus timing accuracy on the current hardware.

Runs a short sequence of 5 trials and checks that PsychoPy flip-based
timing matches expected durations within 1.5 frame tolerance. Exits
with code 1 if any trial fails the tolerance check.

Usage:
    python scripts/validate_timing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.experiment.paradigm import generate_trial_sequence
from src.experiment.stimulus import (
    create_stimulus_config,
    create_fixation_cross,
    run_single_trial,
)


def main() -> None:
    """Run timing validation and report results."""
    from psychopy import visual, core

    config = load_config()
    paradigm_cfg = config["paradigm"]

    # --- Create PsychoPy window ---
    win = visual.Window(
        fullscr=True,
        units="deg",
        color="black",
        allowGUI=False,
    )

    # --- Detect actual refresh rate ---
    measured_hz = win.getActualFrameRate(
        nIdentical=10, nMaxFrames=100, nWarmUpFrames=10
    )
    refresh_hz = measured_hz if measured_hz is not None else 60.0
    frame_ms = 1000.0 / refresh_hz
    tolerance_ms = frame_ms * 1.5

    print("=" * 60)
    print("OcuTrace Stimulus Timing Validation")
    print("=" * 60)
    print(f"Refresh rate:  {refresh_hz:.2f} Hz")
    print(f"Frame period:  {frame_ms:.2f} ms")
    print(f"Tolerance:     {tolerance_ms:.2f} ms (1.5 frames)")
    print("=" * 60)

    # --- Build stimulus components ---
    stim_config = create_stimulus_config(paradigm_cfg, monitor_refresh_hz=refresh_hz)
    fixation = create_fixation_cross(win)
    clock = core.Clock()
    rng = np.random.default_rng(99)

    # --- Generate 5 trials ---
    trials = generate_trial_sequence(n_antisaccade=4, n_prosaccade=2, seed=42)[:5]

    results: list[bool] = []

    for idx, trial in enumerate(trials):
        ts = run_single_trial(
            win=win,
            clock=clock,
            fixation=fixation,
            side=trial.stimulus_side,
            config=stim_config,
            rng=rng,
        )

        # Compute actual durations
        actual_fix_ms = ts.gap_onset_ms - ts.fixation_onset_ms
        actual_gap_ms = ts.stimulus_onset_ms - ts.gap_onset_ms
        actual_stim_ms = ts.stimulus_offset_ms - ts.stimulus_onset_ms

        # Expected durations
        expected_fix_ms = float(stim_config.fixation_duration_ms)
        expected_gap_ms = float(stim_config.gap_duration_ms)
        expected_stim_ms = float(stim_config.stimulus_duration_ms)

        # Check each phase
        fix_err = abs(actual_fix_ms - expected_fix_ms)
        gap_err = abs(actual_gap_ms - expected_gap_ms)
        stim_err = abs(actual_stim_ms - expected_stim_ms)

        trial_pass = (
            fix_err <= tolerance_ms
            and gap_err <= tolerance_ms
            and stim_err <= tolerance_ms
        )
        results.append(trial_pass)

        status = "PASS" if trial_pass else "FAIL"
        print(
            f"\nTrial {idx + 1}/{len(trials)} [{trial.trial_type}, "
            f"{trial.stimulus_side}] — {status}"
        )
        print(
            f"  Fixation:  expected {expected_fix_ms:.1f} ms, "
            f"actual {actual_fix_ms:.1f} ms, err {fix_err:.2f} ms"
        )
        print(
            f"  Gap:       expected {expected_gap_ms:.1f} ms, "
            f"actual {actual_gap_ms:.1f} ms, err {gap_err:.2f} ms"
        )
        print(
            f"  Stimulus:  expected {expected_stim_ms:.1f} ms, "
            f"actual {actual_stim_ms:.1f} ms, err {stim_err:.2f} ms"
        )

    # --- Summary ---
    win.close()
    core.quit()

    n_pass = sum(results)
    n_total = len(results)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass}/{n_total} trials passed timing validation")
    print("=" * 60)

    if n_pass < n_total:
        print("TIMING VALIDATION FAILED — check hardware/driver settings.")
        sys.exit(1)
    else:
        print("All trials within tolerance. Timing is frame-accurate.")


if __name__ == "__main__":
    main()
