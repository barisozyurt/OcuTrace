"""Run a complete gap antisaccade experiment session.

Orchestrates: calibration lookup -> trial paradigm -> real-time gaze
collection -> saccade analysis -> persistence -> session summary.

Usage:
    python scripts/run_session.py --participant SUBJ001
    python scripts/run_session.py --participant SUBJ001 --calibrate
    python scripts/run_session.py --participant SUBJ001 --seed 42
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.storage.models import Session, Trial
from src.storage.sqlite_repo import SQLiteRepository
from src.tracking.iris_tracker import IrisTracker
from src.experiment.paradigm import generate_trial_sequence
from src.experiment.stimulus import (
    create_stimulus_config,
    create_fixation_cross,
    run_single_trial,
)
from src.experiment.session import (
    GazeCollector,
    analyze_trial,
    print_session_summary,
)
from src.analysis.metrics import compute_session_metrics


def main() -> None:
    """Run a complete experiment session."""
    parser = argparse.ArgumentParser(description="OcuTrace Session Runner")
    parser.add_argument("--participant", required=True, help="Participant ID")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration before session",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for trial sequence",
    )
    args = parser.parse_args()

    config = load_config()
    paradigm_cfg = config["paradigm"]
    saccade_cfg = config["saccade_detection"]
    cam_cfg = config["camera"]
    storage_cfg = config["storage"]
    display_cfg = config["display"]

    # --- Initialize storage ---
    repo = SQLiteRepository(storage_cfg["sqlite"]["database_path"])
    repo.initialize()

    # --- Optionally run calibration first ---
    if args.calibrate:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).parent / "calibrate.py"),
                "--participant",
                args.participant,
            ],
            check=False,
        )
        if result.returncode != 0:
            print("Calibration failed. Cannot proceed.")
            repo.close()
            sys.exit(1)

    # --- Find latest accepted calibration for this participant ---
    sessions = repo.list_sessions()
    calibration = None
    for s in sessions:
        if s.participant_id == args.participant:
            cal = repo.get_calibration(s.session_id)
            if cal is not None and cal.accepted:
                calibration = cal
                break

    if calibration is None:
        print(
            "ERROR: No accepted calibration found for "
            f"'{args.participant}'. Run with --calibrate first."
        )
        repo.close()
        sys.exit(1)

    cal_matrix = np.array(calibration.transform_matrix)
    print(f"Using calibration (error: {calibration.mean_error_deg:.2f} deg)")

    # --- Create session record ---
    session = Session(participant_id=args.participant, notes="experiment")
    repo.save_session(session)
    print(f"Session: {session.session_id}")

    # --- Initialize tracker + camera ---
    print("Initializing iris tracker...", end=" ", flush=True)
    tracker = IrisTracker(config)
    print("OK")

    print("Opening camera...", end=" ", flush=True)
    cap = cv2.VideoCapture(cam_cfg["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    if not cap.isOpened():
        print("FAILED")
        tracker.release()
        repo.close()
        sys.exit(1)
    print("OK")

    # --- Initialize PsychoPy ---
    print("Initializing display...", end=" ", flush=True)
    from psychopy import visual, core, event, monitors

    mon = monitors.Monitor("OcuTrace")
    mon.setWidth(display_cfg["screen_width_cm"])
    mon.setDistance(display_cfg["viewing_distance_cm"])
    mon.setSizePix(display_cfg["screen_resolution"])
    mon.saveMon()

    win = visual.Window(
        fullscr=True,
        units="deg",
        color="black",
        allowGUI=False,
        monitor=mon,
    )
    print("OK")

    print("Measuring refresh rate...", end=" ", flush=True)
    actual_hz = win.getActualFrameRate(
        nIdentical=10, nMaxFrames=60, nWarmUpFrames=10
    )
    refresh_hz = actual_hz if actual_hz is not None else 60.0
    print(f"{refresh_hz:.1f} Hz")

    stim_config = create_stimulus_config(paradigm_cfg, monitor_refresh_hz=refresh_hz)
    fixation = create_fixation_cross(win)
    clock = core.Clock()
    rng = np.random.default_rng(args.seed)

    # --- Generate trial sequence ---
    trials = generate_trial_sequence(
        n_antisaccade=paradigm_cfg["n_antisaccade_trials"],
        n_prosaccade=paradigm_cfg["n_prosaccade_trials"],
        seed=args.seed,
    )
    print(f"Trials: {len(trials)} ({paradigm_cfg['n_antisaccade_trials']} anti + "
          f"{paradigm_cfg['n_prosaccade_trials']} pro)")
    print("Ready!\n")

    # --- Show instructions ---
    instruction = visual.TextStim(
        win,
        text=(
            "Antisaccade experiment\n\n"
            "Look at the center cross.\n"
            "When a dot appears on one side,\n"
            "look at the OPPOSITE side.\n\n"
            "Press SPACE to begin."
        ),
        color="white",
        height=1.0,
    )
    instruction.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    # --- Countdown before first trial ---
    countdown_text = visual.TextStim(win, color="white", height=2.0)
    for sec in range(3, 0, -1):
        countdown_text.text = str(sec)
        countdown_text.draw()
        win.flip()
        core.wait(1.0, hogCPUperiod=0)
    # Brief blank before first trial
    win.flip()
    core.wait(0.5, hogCPUperiod=0)

    # --- Start gaze collector ---
    collector = GazeCollector(tracker, cap, session.session_id)
    collector.start()

    # --- Run trials ---
    trial_results: list[dict] = []
    aborted = False

    print("-" * 60)

    try:
        for trial_spec in trials:
            collector.set_trial(trial_spec.trial_number)

            # Run stimulus presentation with gaze collection
            timestamps = run_single_trial(
                win=win,
                clock=clock,
                fixation=fixation,
                side=trial_spec.stimulus_side,
                config=stim_config,
                rng=rng,
                on_frame=collector.on_frame,
            )

            # Get gaze samples for this trial
            gaze_samples = collector.get_trial_samples(trial_spec.trial_number)

            # Save raw gaze data
            if gaze_samples:
                repo.save_gaze_data_batch(gaze_samples)

            # Analyze trial
            direction, latency, correct = analyze_trial(
                gaze_samples,
                trial_spec,
                timestamps.stimulus_onset_ms,
                cal_matrix,
                saccade_cfg,
            )

            # Save trial record
            trial = Trial(
                session_id=session.session_id,
                trial_number=trial_spec.trial_number,
                trial_type=trial_spec.trial_type,
                stimulus_side=trial_spec.stimulus_side,
                stimulus_onset_ms=timestamps.stimulus_onset_ms,
                saccade_direction=direction,
                saccade_latency_ms=latency,
                response_correct=correct,
            )
            repo.save_trial(trial)

            trial_results.append(
                {
                    "trial_type": trial_spec.trial_type,
                    "response_correct": correct,
                    "saccade_latency_ms": latency,
                }
            )

            # Progress indicator
            if correct is True:
                status = "OK"
            elif correct is False:
                status = "ERR"
            else:
                status = "---"
            lat_str = f"{latency:.0f}ms" if latency is not None else "N/A"
            print(
                f"  Trial {trial_spec.trial_number:2d}/{len(trials)} "
                f"[{trial_spec.trial_type:12s} {trial_spec.stimulus_side:5s}] "
                f"{status:3s}  lat={lat_str}"
            )

            # Check for escape key
            keys = event.getKeys(keyList=["escape"])
            if keys:
                print("\nSession aborted by user (ESC).")
                aborted = True
                break

    except KeyboardInterrupt:
        print("\nSession aborted by user (Ctrl+C).")
        aborted = True

    # --- Cleanup (always runs) ---
    collector.stop()

    if trial_results:
        metrics = compute_session_metrics(trial_results)
        print_session_summary(metrics)
        if aborted:
            print("NOTE: Session was aborted early. Metrics are partial.")
    else:
        print("\nNo trials completed.")

    win.close()
    cap.release()
    tracker.release()
    repo.close()


if __name__ == "__main__":
    main()
