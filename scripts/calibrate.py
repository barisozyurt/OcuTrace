"""Run calibration procedure.

Displays calibration targets on screen using PsychoPy while recording
iris positions. Fits a pixel-to-degree transform and saves the result.

Usage:
    python scripts/calibrate.py [--participant SUBJ001]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.storage.models import Session, CalibrationPoint
from src.storage.sqlite_repo import SQLiteRepository
from src.tracking.iris_tracker import IrisTracker, IrisCoordinates
from src.tracking.calibration import create_calibration_result
from src.tracking.calibration_display import generate_calibration_targets


def run_calibration(participant_id: str = "UNKNOWN") -> None:
    """Run the full calibration procedure."""
    config = load_config()
    cal_cfg = config["calibration"]
    cam_cfg = config["camera"]
    storage_cfg = config["storage"]

    # Initialize storage
    repo = SQLiteRepository(storage_cfg["sqlite"]["database_path"])
    repo.initialize()

    # Create session
    session = Session(participant_id=participant_id, notes="calibration")
    repo.save_session(session)

    # Initialize tracker
    tracker = IrisTracker(config)

    # Open camera
    cap = cv2.VideoCapture(cam_cfg["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    # Generate targets
    targets = generate_calibration_targets(
        n_points=cal_cfg["n_points"],
        eccentricity_deg=10.0,
    )

    try:
        from psychopy import visual, core, event

        win = visual.Window(
            fullscr=True,
            color="black",
            units="deg",
            monitor="testMonitor",
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

        # Show instructions
        instruction.draw()
        win.flip()
        event.waitKeys(keyList=["space"])

        # Collect calibration data
        calibration_points: list[CalibrationPoint] = []
        point_duration_s = cal_cfg["point_duration_ms"] / 1000.0
        settle_time_s = 0.5
        collect_time_s = point_duration_s - settle_time_s

        for i, (tx, ty) in enumerate(targets):
            print(f"  Target {i+1}/{len(targets)}: ({tx:.1f}, {ty:.1f}) deg")

            target_stim.pos = (tx, ty)
            target_stim.draw()
            win.flip()

            # Wait for eyes to settle
            settle_start = time.monotonic()
            while (time.monotonic() - settle_start) < settle_time_s:
                ret, frame = cap.read()
                if ret:
                    tracker.process_frame(frame, time.monotonic() * 1000)

            # Collect iris positions
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
                print(f"    WARNING: Only {len(iris_samples)} samples, skipping")
                continue

            mean_x = float(np.mean([c.mean_x for c in iris_samples]))
            mean_y = float(np.mean([c.mean_y for c in iris_samples]))

            calibration_points.append(CalibrationPoint(
                target_x_deg=tx,
                target_y_deg=ty,
                measured_x_px=mean_x,
                measured_y_px=mean_y,
            ))

        win.close()

    except ImportError:
        print("PsychoPy not installed. Install with: pip install psychopy")
        cap.release()
        tracker.release()
        repo.close()
        sys.exit(1)

    cap.release()
    tracker.release()

    # Compute calibration
    if len(calibration_points) < 3:
        print(f"ERROR: Only {len(calibration_points)} valid points. Need at least 3.")
        repo.close()
        sys.exit(1)

    result = create_calibration_result(
        session_id=session.session_id,
        points=calibration_points,
        max_error_deg=cal_cfg["max_acceptable_error_deg"],
    )
    repo.save_calibration(result)
    repo.close()

    # Report
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Points collected:  {len(calibration_points)}")
    print(f"Mean error:        {result.mean_error_deg:.2f} deg")
    print(f"Max acceptable:    {cal_cfg['max_acceptable_error_deg']:.1f} deg")
    print(f"Accepted:          {'YES' if result.accepted else 'NO'}")
    print(f"Session ID:        {session.session_id}")
    print("=" * 60)

    if not result.accepted:
        print("\nCalibration failed. Try again with better conditions.")
        sys.exit(1)
    else:
        print("\nCalibration successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OcuTrace Calibration")
    parser.add_argument(
        "--participant", default="UNKNOWN", help="Participant ID"
    )
    args = parser.parse_args()
    run_calibration(args.participant)
