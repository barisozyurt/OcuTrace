"""High-level workflow functions for OcuTrace.

Clean, importable functions for calibration, experiment, and analysis.
Used by both CLI scripts and the GUI launcher. No argparse, no sys.path
hacks, no sys.exit — just return values and raise exceptions.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import load_config
from src.paths import get_db_path
from src.storage.models import Session, Trial, CalibrationPoint
from src.storage.sqlite_repo import SQLiteRepository
from src.tracking.iris_tracker import IrisTracker, IrisCoordinates
from src.tracking.calibration import create_calibration_result
from src.tracking.calibration_display import generate_calibration_targets
from src.experiment.paradigm import generate_trial_sequence
from src.experiment.stimulus import (
    create_stimulus_config,
    create_fixation_dot,
    run_single_trial,
)
from src.experiment.session import (
    GazeCollector,
    analyze_trial,
    print_session_summary,
)
from src.analysis.metrics import compute_session_metrics


def _get_repo() -> SQLiteRepository:
    """Create and initialize a SQLiteRepository using paths.py."""
    repo = SQLiteRepository(str(get_db_path()))
    repo.initialize()
    return repo


def _find_calibration(repo: SQLiteRepository, participant_id: str):
    """Find the latest accepted calibration for a participant."""
    sessions = repo.list_sessions()
    for s in sessions:
        if s.participant_id == participant_id:
            cal = repo.get_calibration(s.session_id)
            if cal is not None and cal.accepted:
                return cal
    return None


def run_calibration(
    participant_id: str,
    on_status: Optional[callable] = None,
) -> dict:
    """Run the full calibration procedure.

    Parameters
    ----------
    participant_id : str
        Patient/participant name.
    on_status : callable, optional
        Callback for status messages: on_status(message: str).

    Returns
    -------
    dict
        Keys: session_id, accepted, mean_error_deg, n_points.

    Raises
    ------
    RuntimeError
        If camera cannot be opened or calibration fails.
    """
    def status(msg: str) -> None:
        if on_status:
            on_status(msg)
        print(msg)

    config = load_config()
    cal_cfg = config["calibration"]
    cam_cfg = config["camera"]
    display_cfg = config["display"]

    repo = _get_repo()

    session = Session(participant_id=participant_id, notes="calibration")
    repo.save_session(session)

    tracker = IrisTracker(config)

    cap = cv2.VideoCapture(cam_cfg["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    if not cap.isOpened():
        tracker.release()
        repo.close()
        raise RuntimeError("Cannot open camera")

    targets = generate_calibration_targets(
        n_points=cal_cfg["n_points"],
        eccentricity_deg=10.0,
    )

    from psychopy import visual, core, event, monitors

    mon = monitors.Monitor("OcuTrace")
    mon.setWidth(display_cfg["screen_width_cm"])
    mon.setDistance(display_cfg["viewing_distance_cm"])
    mon.setSizePix(display_cfg["screen_resolution"])
    mon.saveMon()

    win = visual.Window(
        fullscr=True,
        color="black",
        units="deg",
        monitor=mon,
    )

    target_stim = visual.Circle(
        win, radius=0.5, fillColor="white", lineColor="white"
    )
    instruction = visual.TextStim(
        win,
        text="Look at each dot as it appears.\nPress SPACE to begin.",
        color="white",
        height=1.0,
    )
    countdown_text = visual.TextStim(
        win, color="white", height=0.8, pos=(0, -2),
    )

    instruction.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    calibration_points: list[CalibrationPoint] = []
    point_duration_s = cal_cfg["point_duration_ms"] / 1000.0
    settle_time_s = 0.5
    collect_time_s = point_duration_s - settle_time_s

    for i, (tx, ty) in enumerate(targets):
        remaining = len(targets) - i
        status(f"  Target {i+1}/{len(targets)}: ({tx:.1f}, {ty:.1f}) deg")

        target_stim.pos = (tx, ty)

        countdown_secs = max(1, int(settle_time_s + 0.5))
        for sec in range(countdown_secs, 0, -1):
            countdown_text.text = f"{remaining} left"
            target_stim.draw()
            countdown_text.draw()
            win.flip()

            sec_start = time.monotonic()
            while (time.monotonic() - sec_start) < 1.0:
                ret, frame = cap.read()
                if ret:
                    tracker.process_frame(frame, time.monotonic() * 1000)

        target_stim.draw()
        win.flip()

        iris_samples: list[IrisCoordinates] = []
        collect_start = time.monotonic()
        while (time.monotonic() - collect_start) < collect_time_s:
            ret, frame = cap.read()
            if not ret:
                continue
            coords = tracker.process_frame(frame, time.monotonic() * 1000)
            if coords is not None:
                iris_samples.append(coords)

        if len(iris_samples) < 5:
            status(f"    WARNING: Only {len(iris_samples)} samples, skipping")
            continue

        mean_x = float(np.mean([c.mean_x for c in iris_samples]))
        mean_y = float(np.mean([c.mean_y for c in iris_samples]))
        std_x = float(np.std([c.mean_x for c in iris_samples]))
        std_y = float(np.std([c.mean_y for c in iris_samples]))

        status(
            f"    samples={len(iris_samples):3d}  "
            f"iris_px=({mean_x:.1f}, {mean_y:.1f})  "
            f"std=({std_x:.2f}, {std_y:.2f})"
        )

        calibration_points.append(CalibrationPoint(
            target_x_deg=tx,
            target_y_deg=ty,
            measured_x_px=mean_x,
            measured_y_px=mean_y,
        ))

    win.close()
    cap.release()
    tracker.release()

    if len(calibration_points) < 3:
        repo.close()
        raise RuntimeError(
            f"Only {len(calibration_points)} valid points. Need at least 3."
        )

    result = create_calibration_result(
        session_id=session.session_id,
        points=calibration_points,
        max_error_deg=cal_cfg["max_acceptable_error_deg"],
    )
    repo.save_calibration(result)
    repo.close()

    return {
        "session_id": session.session_id,
        "accepted": result.accepted,
        "mean_error_deg": result.mean_error_deg,
        "n_points": len(calibration_points),
    }


def run_experiment(
    participant_id: str,
    calibrate_first: bool = False,
    seed: Optional[int] = None,
    on_status: Optional[callable] = None,
) -> dict:
    """Run a complete experiment session.

    Parameters
    ----------
    participant_id : str
        Patient/participant name.
    calibrate_first : bool
        If True, run calibration before the experiment.
    seed : int, optional
        Random seed for trial sequence.
    on_status : callable, optional
        Callback for status messages.

    Returns
    -------
    dict
        Keys: session_id, n_trials, metrics (SessionMetrics), aborted.

    Raises
    ------
    RuntimeError
        If no calibration found or camera fails.
    """
    def status(msg: str) -> None:
        if on_status:
            on_status(msg)
        print(msg)

    if calibrate_first:
        cal_result = run_calibration(participant_id, on_status=on_status)
        if not cal_result["accepted"]:
            raise RuntimeError(
                f"Calibration failed (error: {cal_result['mean_error_deg']:.2f} deg)"
            )
        status(f"\nCalibration successful! (error: {cal_result['mean_error_deg']:.2f} deg)")

    config = load_config()
    paradigm_cfg = config["paradigm"]
    saccade_cfg = config["saccade_detection"]
    cam_cfg = config["camera"]
    display_cfg = config["display"]

    repo = _get_repo()

    calibration = _find_calibration(repo, participant_id)
    if calibration is None:
        repo.close()
        raise RuntimeError(
            f"No accepted calibration for '{participant_id}'. "
            "Run calibration first."
        )

    cal_matrix = np.array(calibration.transform_matrix)
    status(f"Using calibration (error: {calibration.mean_error_deg:.2f} deg)")

    session = Session(participant_id=participant_id, notes="experiment")
    repo.save_session(session)
    status(f"Session: {session.session_id}")

    status("Initializing iris tracker...")
    tracker = IrisTracker(config)

    status("Opening camera...")
    cap = cv2.VideoCapture(cam_cfg["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    if not cap.isOpened():
        tracker.release()
        repo.close()
        raise RuntimeError("Cannot open camera")

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

    actual_hz = win.getActualFrameRate(
        nIdentical=10, nMaxFrames=60, nWarmUpFrames=10
    )
    refresh_hz = actual_hz if actual_hz is not None else 60.0
    status(f"Refresh rate: {refresh_hz:.1f} Hz")

    stim_config = create_stimulus_config(paradigm_cfg, monitor_refresh_hz=refresh_hz)
    clock = core.Clock()
    rng = np.random.default_rng(seed)

    trials = generate_trial_sequence(
        n_antisaccade=paradigm_cfg["n_antisaccade_trials"],
        n_prosaccade=paradigm_cfg["n_prosaccade_trials"],
        seed=seed,
    )
    status(f"Trials: {len(trials)} ({paradigm_cfg['n_antisaccade_trials']} anti + "
           f"{paradigm_cfg['n_prosaccade_trials']} pro)")

    # Instructions
    instruction = visual.TextStim(
        win,
        text=(
            f"Patient: {participant_id}\n\n"
            "Saccade / Antisaccade Experiment\n\n"
            "Always look at the center dot first.\n"
            "The dot color tells you what to do:\n\n"
            "RED dot  =  look OPPOSITE to the target\n"
            "GREEN dot  =  look TOWARD the target\n\n"
            "Press SPACE to begin."
        ),
        color="white",
        height=0.9,
    )
    instruction.draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    # Countdown
    countdown_text = visual.TextStim(win, color="white", height=2.0)
    for sec in range(3, 0, -1):
        countdown_text.text = str(sec)
        countdown_text.draw()
        win.flip()
        core.wait(1.0, hogCPUperiod=0)
    win.flip()
    core.wait(0.5, hogCPUperiod=0)

    # Start gaze collector
    collector = GazeCollector(tracker, cap, session.session_id)
    collector.start()

    trial_counter_text = visual.TextStim(
        win, color=[0.3, 0.3, 0.3], height=0.6,
        pos=(0, 12), anchorHoriz="center",
    )

    fixation_anti = create_fixation_dot(win, color="red")
    fixation_pro = create_fixation_dot(win, color="green")

    trial_results: list[dict] = []
    aborted = False

    try:
        for trial_spec in trials:
            collector.set_trial(trial_spec.trial_number)

            if trial_spec.trial_type == "antisaccade":
                trial_fixation = fixation_anti
            else:
                trial_fixation = fixation_pro

            trial_counter_text.text = f"{trial_spec.trial_number}/{len(trials)}"
            trial_counter_text.draw()
            trial_fixation.draw()
            win.flip()
            core.wait(0.3, hogCPUperiod=0)

            timestamps = run_single_trial(
                win=win,
                clock=clock,
                fixation=trial_fixation,
                side=trial_spec.stimulus_side,
                config=stim_config,
                rng=rng,
                on_frame=collector.on_frame,
            )

            gaze_samples = collector.get_trial_samples(trial_spec.trial_number)

            if gaze_samples:
                repo.save_gaze_data_batch(gaze_samples)

            direction, latency, correct = analyze_trial(
                gaze_samples,
                trial_spec,
                timestamps.stimulus_onset_ms,
                cal_matrix,
                saccade_cfg,
            )

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

            trial_results.append({
                "trial_type": trial_spec.trial_type,
                "response_correct": correct,
                "saccade_latency_ms": latency,
            })

            if correct is True:
                result_str = "OK"
            elif correct is False:
                result_str = "ERR"
            else:
                result_str = "---"
            lat_str = f"{latency:.0f}ms" if latency is not None else "N/A"
            status(
                f"  Trial {trial_spec.trial_number:2d}/{len(trials)} "
                f"[{trial_spec.trial_type:12s} {trial_spec.stimulus_side:5s}] "
                f"{result_str:3s}  lat={lat_str:>8s}  gaze={len(gaze_samples)}"
            )

            keys = event.getKeys(keyList=["escape"])
            if keys:
                status("\nSession aborted by user (ESC).")
                aborted = True
                break

    except KeyboardInterrupt:
        status("\nSession aborted by user (Ctrl+C).")
        aborted = True

    collector.stop()

    metrics = None
    if trial_results:
        metrics = compute_session_metrics(trial_results)
        print_session_summary(metrics)
        if aborted:
            status("NOTE: Session was aborted early. Metrics are partial.")

    win.close()
    cap.release()
    tracker.release()
    repo.close()

    return {
        "session_id": session.session_id,
        "n_trials": len(trial_results),
        "metrics": metrics,
        "aborted": aborted,
    }


def generate_report(
    session_id: Optional[str] = None,
    on_status: Optional[callable] = None,
) -> Path:
    """Generate report for a session.

    Parameters
    ----------
    session_id : str, optional
        Session to analyze. If None, uses most recent experiment session.
    on_status : callable, optional
        Callback for status messages.

    Returns
    -------
    Path
        Path to the saved report PNG.

    Raises
    ------
    RuntimeError
        If session not found or has no trials.
    """
    from src.visualization.reports import generate_session_report

    def status(msg: str) -> None:
        if on_status:
            on_status(msg)
        print(msg)

    repo = _get_repo()

    if session_id is None:
        sessions = repo.list_sessions()
        experiment_sessions = [
            s for s in sessions
            if s.notes == "experiment" and repo.get_trials(s.session_id)
        ]
        if not experiment_sessions:
            repo.close()
            raise RuntimeError("No experiment sessions found.")
        session_id = experiment_sessions[0].session_id

    session = repo.get_session(session_id)
    if session is None:
        repo.close()
        raise RuntimeError(f"Session '{session_id}' not found.")

    trials = repo.get_trials(session_id)
    if not trials:
        repo.close()
        raise RuntimeError(f"No trials for session '{session_id}'.")

    calibration = _find_calibration(repo, session.participant_id)
    cal_matrix = np.array(calibration.transform_matrix) if calibration else None

    status(f"Generating report for {session.participant_id}...")

    report_path = generate_session_report(
        session_id=session_id,
        repo=repo,
        calibration_matrix=cal_matrix,
        participant_name=session.participant_id,
    )

    repo.close()
    status(f"Report saved: {report_path}")
    return report_path
