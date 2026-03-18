# Phase 2: Calibration System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a calibration system that converts raw pixel iris coordinates to visual angle (degrees), enabling accurate saccade measurement.

**Architecture:** PsychoPy displays calibration targets at known screen positions. IrisTracker records iris coordinates at each target. A polynomial (affine) mapping is fitted from pixel→degree space. Calibration data is persisted in SQLite and validated against known target positions. The calibration module is independent of the experiment module — it produces a `CalibrationResult` that other modules consume.

**Tech Stack:** PsychoPy (target display), NumPy (affine fitting), existing IrisTracker, existing SQLiteRepository

---

### Task 1: Calibration Data Models

**Files:**
- Create: `tests/test_calibration.py` (initial tests)
- Modify: `src/storage/models.py` — add `CalibrationPoint` and `CalibrationResult`

**Step 1: Write the failing test**

```python
"""Tests for calibration system."""
import numpy as np
import pytest

from src.storage.models import CalibrationPoint, CalibrationResult


class TestCalibrationModels:
    def test_create_calibration_point(self):
        point = CalibrationPoint(
            target_x_deg=10.0,
            target_y_deg=0.0,
            measured_x_px=450.0,
            measured_y_px=240.0,
        )
        assert point.target_x_deg == 10.0
        assert point.measured_x_px == 450.0

    def test_create_calibration_result(self):
        points = [
            CalibrationPoint(
                target_x_deg=float(x),
                target_y_deg=0.0,
                measured_x_px=320.0 + x * 13.0,
                measured_y_px=240.0,
            )
            for x in [-10, -5, 0, 5, 10]
        ]
        result = CalibrationResult(
            session_id="test-123",
            points=points,
            transform_matrix=np.eye(3).tolist(),
            mean_error_deg=0.5,
            accepted=True,
        )
        assert result.accepted is True
        assert result.mean_error_deg == 0.5
        assert len(result.points) == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py::TestCalibrationModels -v`
Expected: FAIL — ImportError

**Step 3: Add models to `src/storage/models.py`**

Add after the `TrackingQuality` dataclass:

```python
@dataclass
class CalibrationPoint:
    """A single calibration target measurement.

    Parameters
    ----------
    target_x_deg : float
        Target position X in degrees from center.
    target_y_deg : float
        Target position Y in degrees from center.
    measured_x_px : float
        Mean measured iris X in pixels at this target.
    measured_y_px : float
        Mean measured iris Y in pixels at this target.
    """
    target_x_deg: float
    target_y_deg: float
    measured_x_px: float
    measured_y_px: float


@dataclass
class CalibrationResult:
    """Result of a calibration procedure.

    Parameters
    ----------
    session_id : str
        Session this calibration belongs to.
    points : list[CalibrationPoint]
        All calibration measurements.
    transform_matrix : list[list[float]]
        3x3 affine transform matrix (pixel→degree), stored as nested list for serialization.
    mean_error_deg : float
        Mean validation error in degrees.
    accepted : bool
        Whether error is within acceptable threshold.
    """
    session_id: str
    points: list[CalibrationPoint]
    transform_matrix: list[list[float]]
    mean_error_deg: float
    accepted: bool
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_calibration.py::TestCalibrationModels -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/storage/models.py tests/test_calibration.py
git commit -m "feat: add CalibrationPoint and CalibrationResult data models"
```

---

### Task 2: Pixel-to-Degree Transform (Affine Fitting)

**Files:**
- Create: `src/tracking/calibration.py`
- Modify: `tests/test_calibration.py` — add transform tests

**Step 1: Write the failing test**

Add to `tests/test_calibration.py`:

```python
from src.tracking.calibration import fit_pixel_to_degree_transform, apply_transform


class TestPixelToDegreeTransform:
    def test_fit_identity_like_transform(self):
        """If pixel positions linearly map to degrees, transform should be near-identity scale."""
        # 9-point grid: targets at -10, 0, +10 degrees in X and Y
        # Simulated pixel measurements with a known linear relationship:
        # degree = (pixel - 320) / 13.0 for X, (pixel - 240) / 13.0 for Y
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                px = 320.0 + tx * 13.0  # pixels
                py = 240.0 + ty * 13.0
                points.append(CalibrationPoint(
                    target_x_deg=tx,
                    target_y_deg=ty,
                    measured_x_px=px,
                    measured_y_px=py,
                ))
        matrix = fit_pixel_to_degree_transform(points)
        assert matrix.shape == (3, 3)

        # Verify: transform center pixel should give ~0 degrees
        deg_x, deg_y = apply_transform(matrix, 320.0, 240.0)
        assert abs(deg_x) < 0.1
        assert abs(deg_y) < 0.1

        # Verify: transform pixel at +10 deg should give ~10 degrees
        deg_x, deg_y = apply_transform(matrix, 320.0 + 10.0 * 13.0, 240.0)
        assert abs(deg_x - 10.0) < 0.1
        assert abs(deg_y) < 0.1

    def test_fit_with_noise(self):
        """Transform should still be reasonable with noisy measurements."""
        rng = np.random.default_rng(42)
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                px = 320.0 + tx * 13.0 + rng.normal(0, 1.0)
                py = 240.0 + ty * 13.0 + rng.normal(0, 1.0)
                points.append(CalibrationPoint(
                    target_x_deg=tx,
                    target_y_deg=ty,
                    measured_x_px=px,
                    measured_y_px=py,
                ))
        matrix = fit_pixel_to_degree_transform(points)

        # Center should still be close to 0
        deg_x, deg_y = apply_transform(matrix, 320.0, 240.0)
        assert abs(deg_x) < 1.0
        assert abs(deg_y) < 1.0

    def test_apply_transform_returns_tuple(self):
        matrix = np.eye(3)
        result = apply_transform(matrix, 100.0, 200.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fit_requires_minimum_points(self):
        """Need at least 3 points for affine fit."""
        points = [
            CalibrationPoint(0.0, 0.0, 320.0, 240.0),
            CalibrationPoint(10.0, 0.0, 450.0, 240.0),
        ]
        with pytest.raises(ValueError, match="at least 3"):
            fit_pixel_to_degree_transform(points)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration.py::TestPixelToDegreeTransform -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
"""Calibration system for pixel-to-degree coordinate transformation.

Maps raw iris pixel coordinates to visual angle (degrees) using
an affine transformation fitted from calibration point measurements.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.storage.models import CalibrationPoint, CalibrationResult


def fit_pixel_to_degree_transform(
    points: list[CalibrationPoint],
) -> np.ndarray:
    """Fit an affine transform from pixel coordinates to degrees.

    Uses least-squares to find a 3x3 affine matrix that maps
    [pixel_x, pixel_y, 1] → [degree_x, degree_y, 1].

    Parameters
    ----------
    points : list[CalibrationPoint]
        Calibration measurements (minimum 3 points).

    Returns
    -------
    np.ndarray
        3x3 affine transformation matrix.

    Raises
    ------
    ValueError
        If fewer than 3 points are provided.
    """
    if len(points) < 3:
        raise ValueError(
            f"Need at least 3 calibration points, got {len(points)}"
        )

    # Build source (pixel) and destination (degree) matrices
    n = len(points)
    src = np.zeros((n, 3))
    dst = np.zeros((n, 2))
    for i, p in enumerate(points):
        src[i] = [p.measured_x_px, p.measured_y_px, 1.0]
        dst[i] = [p.target_x_deg, p.target_y_deg]

    # Least-squares: dst = src @ A^T  →  A^T = pinv(src) @ dst
    a_t, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)

    # Build 3x3 matrix: top 2 rows from a_t, bottom row = [0, 0, 1]
    matrix = np.eye(3)
    matrix[:2, :] = a_t.T  # 2x3

    return matrix


def apply_transform(
    matrix: np.ndarray,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float]:
    """Apply calibration transform to convert pixels to degrees.

    Parameters
    ----------
    matrix : np.ndarray
        3x3 affine transformation matrix from fit_pixel_to_degree_transform.
    pixel_x : float
        Iris X position in pixels.
    pixel_y : float
        Iris Y position in pixels.

    Returns
    -------
    tuple[float, float]
        (degree_x, degree_y) visual angle from center.
    """
    pixel_vec = np.array([pixel_x, pixel_y, 1.0])
    result = matrix @ pixel_vec
    return (float(result[0]), float(result[1]))


def compute_calibration_error(
    matrix: np.ndarray,
    points: list[CalibrationPoint],
) -> float:
    """Compute mean calibration error in degrees.

    Parameters
    ----------
    matrix : np.ndarray
        Fitted transform matrix.
    points : list[CalibrationPoint]
        Calibration points to validate against.

    Returns
    -------
    float
        Mean Euclidean error in degrees.
    """
    errors = []
    for p in points:
        pred_x, pred_y = apply_transform(matrix, p.measured_x_px, p.measured_y_px)
        error = np.sqrt(
            (pred_x - p.target_x_deg) ** 2
            + (pred_y - p.target_y_deg) ** 2
        )
        errors.append(error)
    return float(np.mean(errors))


def create_calibration_result(
    session_id: str,
    points: list[CalibrationPoint],
    max_error_deg: float,
) -> CalibrationResult:
    """Run full calibration pipeline: fit transform, compute error, decide acceptance.

    Parameters
    ----------
    session_id : str
        Session UUID.
    points : list[CalibrationPoint]
        Calibration measurements.
    max_error_deg : float
        Maximum acceptable mean error in degrees.

    Returns
    -------
    CalibrationResult
        Complete calibration result with transform and acceptance.
    """
    matrix = fit_pixel_to_degree_transform(points)
    mean_error = compute_calibration_error(matrix, points)

    return CalibrationResult(
        session_id=session_id,
        points=points,
        transform_matrix=matrix.tolist(),
        mean_error_deg=mean_error,
        accepted=mean_error <= max_error_deg,
    )
```

**Step 4: Run tests**

Run: `pytest tests/test_calibration.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/tracking/calibration.py tests/test_calibration.py
git commit -m "feat: add pixel-to-degree affine calibration transform"
```

---

### Task 3: Calibration Error and Full Pipeline Tests

**Files:**
- Modify: `tests/test_calibration.py` — add error computation and pipeline tests

**Step 1: Write the failing test**

Add to `tests/test_calibration.py`:

```python
from src.tracking.calibration import compute_calibration_error, create_calibration_result


class TestCalibrationError:
    def test_perfect_calibration_zero_error(self):
        """Perfect linear mapping should give ~0 error."""
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(
                    target_x_deg=tx,
                    target_y_deg=ty,
                    measured_x_px=320.0 + tx * 13.0,
                    measured_y_px=240.0 + ty * 13.0,
                ))
        matrix = fit_pixel_to_degree_transform(points)
        error = compute_calibration_error(matrix, points)
        assert error < 0.01  # effectively zero

    def test_noisy_calibration_bounded_error(self):
        rng = np.random.default_rng(42)
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(
                    target_x_deg=tx,
                    target_y_deg=ty,
                    measured_x_px=320.0 + tx * 13.0 + rng.normal(0, 2.0),
                    measured_y_px=240.0 + ty * 13.0 + rng.normal(0, 2.0),
                ))
        matrix = fit_pixel_to_degree_transform(points)
        error = compute_calibration_error(matrix, points)
        assert error < 2.0  # should be small with moderate noise


class TestCreateCalibrationResult:
    def test_accepted_calibration(self):
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(
                    target_x_deg=tx,
                    target_y_deg=ty,
                    measured_x_px=320.0 + tx * 13.0,
                    measured_y_px=240.0 + ty * 13.0,
                ))
        result = create_calibration_result(
            session_id="test-123",
            points=points,
            max_error_deg=2.0,
        )
        assert result.accepted is True
        assert result.mean_error_deg < 0.01
        assert len(result.transform_matrix) == 3

    def test_rejected_calibration_high_noise(self):
        """With extremely high noise, calibration should be rejected."""
        rng = np.random.default_rng(42)
        points = []
        for tx in [-10.0, 0.0, 10.0]:
            for ty in [-10.0, 0.0, 10.0]:
                points.append(CalibrationPoint(
                    target_x_deg=tx,
                    target_y_deg=ty,
                    measured_x_px=320.0 + rng.normal(0, 100.0),  # huge noise
                    measured_y_px=240.0 + rng.normal(0, 100.0),
                ))
        result = create_calibration_result(
            session_id="test-456",
            points=points,
            max_error_deg=2.0,
        )
        # With this much noise, error will be high
        assert result.mean_error_deg > 2.0
        assert result.accepted is False
```

**Step 2: Run tests**

Run: `pytest tests/test_calibration.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_calibration.py
git commit -m "test: add calibration error computation and pipeline tests"
```

---

### Task 4: Calibration Storage (SQLite Persistence)

**Files:**
- Modify: `src/storage/repository.py` — add calibration methods
- Modify: `src/storage/sqlite_repo.py` — implement calibration storage
- Modify: `tests/test_storage.py` — add calibration storage tests

**Step 1: Write the failing test**

Add to `tests/test_storage.py`:

```python
import json
import numpy as np
from src.storage.models import CalibrationPoint, CalibrationResult


class TestCalibrationStorage:
    def test_save_and_load_calibration(self, repo):
        session = Session(participant_id="SUBJ001")
        repo.save_session(session)

        points = [
            CalibrationPoint(
                target_x_deg=float(x),
                target_y_deg=0.0,
                measured_x_px=320.0 + x * 13.0,
                measured_y_px=240.0,
            )
            for x in [-10, -5, 0, 5, 10]
        ]
        cal = CalibrationResult(
            session_id=session.session_id,
            points=points,
            transform_matrix=np.eye(3).tolist(),
            mean_error_deg=0.5,
            accepted=True,
        )
        repo.save_calibration(cal)
        loaded = repo.get_calibration(session.session_id)

        assert loaded is not None
        assert loaded.accepted is True
        assert loaded.mean_error_deg == 0.5
        assert len(loaded.points) == 5
        assert loaded.points[0].target_x_deg == -10.0

    def test_get_calibration_not_found(self, repo):
        assert repo.get_calibration("nonexistent") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py::TestCalibrationStorage -v`
Expected: FAIL — AttributeError (save_calibration not found)

**Step 3: Add abstract methods to `src/storage/repository.py`**

Add to the `Repository` ABC:

```python
    @abstractmethod
    def save_calibration(self, calibration: "CalibrationResult") -> None:
        """Persist a calibration result."""

    @abstractmethod
    def get_calibration(self, session_id: str) -> "Optional[CalibrationResult]":
        """Get calibration for a session. Returns None if not found."""
```

Add at top: `from src.storage.models import Session, Trial, GazeData, CalibrationResult, CalibrationPoint`

**Step 4: Implement in `src/storage/sqlite_repo.py`**

Add calibrations table to `_create_tables`:

```sql
CREATE TABLE IF NOT EXISTS calibrations (
    session_id TEXT PRIMARY KEY,
    points_json TEXT NOT NULL,
    transform_matrix_json TEXT NOT NULL,
    mean_error_deg REAL NOT NULL,
    accepted INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

Add methods:

```python
import json
from src.storage.models import CalibrationResult, CalibrationPoint

def save_calibration(self, calibration: CalibrationResult) -> None:
    points_json = json.dumps([
        {
            "target_x_deg": p.target_x_deg,
            "target_y_deg": p.target_y_deg,
            "measured_x_px": p.measured_x_px,
            "measured_y_px": p.measured_y_px,
        }
        for p in calibration.points
    ])
    matrix_json = json.dumps(calibration.transform_matrix)
    self._conn.execute(
        """INSERT OR REPLACE INTO calibrations
           (session_id, points_json, transform_matrix_json, mean_error_deg, accepted)
           VALUES (?, ?, ?, ?, ?)""",
        (
            calibration.session_id,
            points_json,
            matrix_json,
            calibration.mean_error_deg,
            1 if calibration.accepted else 0,
        ),
    )
    self._conn.commit()

def get_calibration(self, session_id: str) -> Optional[CalibrationResult]:
    row = self._conn.execute(
        "SELECT * FROM calibrations WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    points = [
        CalibrationPoint(**p) for p in json.loads(row[1])
    ]
    return CalibrationResult(
        session_id=row[0],
        points=points,
        transform_matrix=json.loads(row[2]),
        mean_error_deg=row[3],
        accepted=bool(row[4]),
    )
```

**Step 5: Run tests**

Run: `pytest tests/test_storage.py -v`
Expected: All PASS (old + new)

**Step 6: Commit**

```bash
git add src/storage/repository.py src/storage/sqlite_repo.py tests/test_storage.py
git commit -m "feat: add calibration result persistence to SQLite storage"
```

---

### Task 5: PsychoPy Calibration Display

**Files:**
- Create: `src/tracking/calibration_display.py`
- Create: `tests/test_calibration_display.py`

**Step 1: Write the failing test**

```python
"""Tests for calibration display target generation.

Tests the target position calculation logic without requiring
a real PsychoPy window (no display needed).
"""
import pytest

from src.tracking.calibration_display import generate_calibration_targets


class TestGenerateCalibrationTargets:
    def test_9_point_grid(self):
        targets = generate_calibration_targets(
            n_points=9,
            eccentricity_deg=10.0,
        )
        assert len(targets) == 9
        # Center point should be (0, 0)
        center = [t for t in targets if t == (0.0, 0.0)]
        assert len(center) == 1

    def test_5_point_cross(self):
        targets = generate_calibration_targets(
            n_points=5,
            eccentricity_deg=10.0,
        )
        assert len(targets) == 5
        # Should have center + 4 cardinal directions
        assert (0.0, 0.0) in targets

    def test_targets_within_eccentricity(self):
        targets = generate_calibration_targets(
            n_points=9,
            eccentricity_deg=10.0,
        )
        for x, y in targets:
            assert abs(x) <= 10.0
            assert abs(y) <= 10.0

    def test_invalid_n_points(self):
        with pytest.raises(ValueError):
            generate_calibration_targets(n_points=3, eccentricity_deg=10.0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_calibration_display.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
"""Calibration display utilities.

Generates calibration target positions and provides PsychoPy-based
calibration display routines. Target generation is pure computation
(no display dependency), making it testable without PsychoPy.
"""
from __future__ import annotations

from typing import Any


def generate_calibration_targets(
    n_points: int,
    eccentricity_deg: float,
) -> list[tuple[float, float]]:
    """Generate calibration target positions in degrees.

    Parameters
    ----------
    n_points : int
        Number of calibration points (5 or 9).
    eccentricity_deg : float
        Maximum eccentricity in degrees from center.

    Returns
    -------
    list[tuple[float, float]]
        Target positions as (x_deg, y_deg) from center.

    Raises
    ------
    ValueError
        If n_points is not 5 or 9.
    """
    e = eccentricity_deg

    if n_points == 5:
        return [
            (0.0, 0.0),      # center
            (-e, 0.0),       # left
            (e, 0.0),        # right
            (0.0, e),        # top
            (0.0, -e),       # bottom
        ]
    elif n_points == 9:
        return [
            (-e, e),    (-e, 0.0),   (-e, -e),    # left column
            (0.0, e),   (0.0, 0.0),  (0.0, -e),   # center column
            (e, e),     (e, 0.0),    (e, -e),      # right column
        ]
    else:
        raise ValueError(
            f"n_points must be 5 or 9, got {n_points}"
        )
```

**Step 4: Run tests**

Run: `pytest tests/test_calibration_display.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/tracking/calibration_display.py tests/test_calibration_display.py
git commit -m "feat: add calibration target position generation"
```

---

### Task 6: Calibration Runner Script

**Files:**
- Create: `scripts/calibrate.py`

**Step 1: Write the calibration script**

This is the interactive calibration runner that uses PsychoPy to display targets and IrisTracker to record iris positions. It cannot be unit-tested (requires display + camera), but all its computation logic is already tested in Tasks 1-5.

```python
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
        # Import PsychoPy here (heavy import, only when actually running)
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
        settle_time_s = 0.5  # wait for eyes to settle on target
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
        print("PsychoPy not installed. Running headless calibration with manual input.")
        print("Install PsychoPy for proper calibration: pip install psychopy")
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
```

**Step 2: Commit**

```bash
git add scripts/calibrate.py
git commit -m "feat: add PsychoPy calibration runner script"
```

---

### Task 7: Run All Tests + Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify imports**

Run: `python -c "from src.tracking.calibration import fit_pixel_to_degree_transform, apply_transform, create_calibration_result; from src.tracking.calibration_display import generate_calibration_targets; print('Phase 2 imports OK')"`
Expected: `Phase 2 imports OK`

**Step 3: Commit if any fixups needed**

```bash
git add -A && git commit -m "fix: phase 2 final adjustments"
```
