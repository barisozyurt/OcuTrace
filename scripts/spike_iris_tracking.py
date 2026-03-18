"""Spike: Validate MediaPipe iris tracking on internal webcam.

Runs for 10 seconds, measures:
- Detection rate (target: >= 95%)
- X-coordinate jitter during fixation (target: < 2px)
- FPS achieved
- Glasses detection

Usage:
    python scripts/spike_iris_tracking.py

Look at the center of the screen and hold still during the test.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.tracking.iris_tracker import IrisTracker, IrisCoordinates
from src.tracking.glasses_detector import GlassesDetector, assess_tracking_quality


def run_spike() -> None:
    """Run the iris tracking spike test."""
    config = load_config()
    cam_cfg = config["camera"]
    glasses_cfg = config["glasses_detection"]

    # Open camera
    cap = cv2.VideoCapture(cam_cfg["device_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["frame_height"])

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)

    tracker = IrisTracker(config)
    glasses_detector = GlassesDetector()

    print("=" * 60)
    print("OcuTrace Iris Tracking Spike Test")
    print("=" * 60)
    print("Look at the CENTER of the camera and hold still.")
    print("Test will run for 10 seconds.")
    print("-" * 60)

    coords_history: list[IrisCoordinates] = []
    none_count = 0
    total_frames = 0

    duration_s = 10.0
    start_time = time.monotonic()

    while (time.monotonic() - start_time) < duration_s:
        ret, frame = cap.read()
        if not ret:
            continue

        total_frames += 1
        timestamp_ms = time.monotonic() * 1000

        coords = tracker.process_frame(frame, timestamp_ms)

        if coords is None:
            none_count += 1
        else:
            coords_history.append(coords)

        # Show live feed with iris markers
        if coords is not None:
            cv2.circle(frame, (int(coords.left_x), int(coords.left_y)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(coords.right_x), int(coords.right_y)), 3, (0, 255, 0), -1)

        elapsed = time.monotonic() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame, f"FPS: {fps:.1f} | Frames: {total_frames}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("OcuTrace Spike Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.release()

    # Assess quality
    quality = assess_tracking_quality(
        coords_history=coords_history,
        none_count=none_count,
        total_frames=total_frames,
        jitter_threshold_px=glasses_cfg["jitter_threshold_px"],
        min_detection_rate=glasses_cfg["min_detection_rate"],
    )

    elapsed_total = time.monotonic() - start_time
    avg_fps = total_frames / elapsed_total if elapsed_total > 0 else 0

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Duration:         {elapsed_total:.1f}s")
    print(f"Total frames:     {total_frames}")
    print(f"Average FPS:      {avg_fps:.1f}")
    print(f"Detection rate:   {quality.detection_rate:.1%}"
          f"  {'PASS' if quality.detection_rate >= glasses_cfg['min_detection_rate'] else 'FAIL'}"
          f"  (target: >= {glasses_cfg['min_detection_rate']:.0%})")
    print(f"Mean jitter:      {quality.mean_jitter_px:.2f}px"
          f"  {'PASS' if quality.mean_jitter_px <= glasses_cfg['jitter_threshold_px'] else 'FAIL'}"
          f"  (target: < {glasses_cfg['jitter_threshold_px']}px)")
    print(f"Quality OK:       {'YES' if quality.quality_acceptable else 'NO'}")
    print("=" * 60)

    if not quality.quality_acceptable:
        print("\nWARNING: Tracking quality below threshold.")
        print("Possible causes:")
        print("  - Poor lighting")
        print("  - Glasses causing reflections")
        print("  - Camera too far from face")
        sys.exit(1)
    else:
        print("\nSUCCESS: Tracking quality is acceptable for experiments.")


if __name__ == "__main__":
    run_spike()
