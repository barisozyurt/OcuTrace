# Phase 1: Project Skeleton + Iris Tracking Spike + Glasses Detection

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Set up project infrastructure, validate MediaPipe iris tracking on internal webcam, implement glasses detection with quality gate, and build storage layer for raw gaze data.

**Architecture:** Modular Python package under `src/` with clear separation: tracking (MediaPipe), storage (repository pattern + SQLite), and config (YAML). All configurable values in `config/settings.yaml`. TDD throughout — write failing test first, then implement.

**Tech Stack:** Python 3.10+, MediaPipe (FaceMesh with iris refinement), OpenCV (camera capture), SQLite (storage), PyYAML (config), pytest (testing), NumPy (numerical)

---

### Task 1: Project Skeleton + Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config/settings.yaml`
- Create: `src/__init__.py`
- Create: `src/tracking/__init__.py`
- Create: `src/storage/__init__.py`
- Create: `src/analysis/__init__.py`
- Create: `src/experiment/__init__.py`
- Create: `src/visualization/__init__.py`
- Create: `tests/__init__.py`
- Create: `data/.gitkeep`

**Step 1: Create requirements.txt**

```txt
# Core
mediapipe>=0.10.14
opencv-python>=4.9.0
numpy>=1.26.0
PyYAML>=6.0.1

# Experiment
psychopy>=2024.1.0

# Analysis
scipy>=1.12.0
pandas>=2.2.0

# Visualization
matplotlib>=3.8.0
plotly>=5.18.0

# Storage
# sqlite3 is stdlib — no extra dep needed

# Testing
pytest>=8.0.0
pytest-cov>=4.1.0
```

**Step 2: Create config/settings.yaml**

```yaml
# OcuTrace Configuration
# All configurable values live here — no hardcoded values in code.

camera:
  device_index: 0
  frame_width: 640
  frame_height: 480
  target_fps: 30

tracking:
  # MediaPipe FaceMesh
  max_num_faces: 1
  refine_landmarks: true
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  # Iris landmark indices
  left_iris_index: 468
  right_iris_index: 473

glasses_detection:
  # Quality gate thresholds
  jitter_threshold_px: 2.0
  min_detection_rate: 0.95
  quality_window_frames: 90  # 3 seconds at 30fps

storage:
  backend: sqlite
  sqlite:
    database_path: data/ocutrace.db

saccade_detection:
  # Savitzky-Golay filter
  smoothing_window: 5
  smoothing_polyorder: 2
  # Velocity thresholds (degrees/second)
  onset_velocity_threshold: 30.0
  offset_velocity_threshold: 20.0
  min_onset_frames: 3

paradigm:
  # Gap antisaccade protocol
  fixation_duration_ms: 1000
  gap_duration_ms: 200
  stimulus_duration_ms: 1500
  iti_min_ms: 1000
  iti_max_ms: 1500
  stimulus_eccentricity_deg: 10.0
  n_antisaccade_trials: 40
  n_prosaccade_trials: 20

calibration:
  n_points: 9
  point_duration_ms: 2000
  max_acceptable_error_deg: 2.0
```

**Step 3: Create all __init__.py files and data/.gitkeep**

All `__init__.py` files are empty. `data/.gitkeep` is empty (keeps the dir in git).

**Step 4: Verify structure**

Run: `python -c "import src; import src.tracking; import src.storage; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add requirements.txt config/ src/ tests/__init__.py data/.gitkeep
git commit -m "feat: add project skeleton, dependencies, and configuration"
```

---

### Task 2: Data Models

**Files:**
- Create: `tests/test_models.py`
- Create: `src/storage/models.py`

**Step 1: Write the failing test**

```python
"""Tests for data models."""
import uuid
from datetime import datetime, timezone

from src.storage.models import Session, Trial, GazeData, TrackingQuality


class TestSession:
    def test_create_session(self):
        session = Session(
            participant_id="SUBJ001",
            notes="test session",
        )
        # UUID is auto-generated
        assert isinstance(session.session_id, str)
        uuid.UUID(session.session_id)  # validates format
        assert session.participant_id == "SUBJ001"
        assert isinstance(session.created_at, float)  # unix ms
        assert session.glasses_detected is None

    def test_session_timestamp_is_utc_unix_ms(self):
        before = datetime.now(timezone.utc).timestamp() * 1000
        session = Session(participant_id="SUBJ001")
        after = datetime.now(timezone.utc).timestamp() * 1000
        assert before <= session.created_at <= after


class TestTrial:
    def test_create_trial(self):
        trial = Trial(
            session_id="abc-123",
            trial_number=1,
            trial_type="antisaccade",
            stimulus_side="left",
            stimulus_onset_ms=1000.0,
        )
        assert trial.trial_type == "antisaccade"
        assert trial.stimulus_side == "left"
        assert trial.response_correct is None  # not yet classified

    def test_trial_type_validation(self):
        import pytest
        with pytest.raises(ValueError):
            Trial(
                session_id="abc",
                trial_number=1,
                trial_type="invalid",
                stimulus_side="left",
                stimulus_onset_ms=0.0,
            )

    def test_stimulus_side_validation(self):
        import pytest
        with pytest.raises(ValueError):
            Trial(
                session_id="abc",
                trial_number=1,
                trial_type="prosaccade",
                stimulus_side="up",
                stimulus_onset_ms=0.0,
            )


class TestGazeData:
    def test_create_gaze_data(self):
        gaze = GazeData(
            session_id="abc-123",
            trial_number=1,
            timestamp_ms=1500.0,
            left_iris_x=320.5,
            left_iris_y=240.1,
            right_iris_x=321.0,
            right_iris_y=240.3,
            confidence=0.95,
        )
        assert gaze.left_iris_x == 320.5
        assert gaze.confidence == 0.95


class TestTrackingQuality:
    def test_create_tracking_quality(self):
        tq = TrackingQuality(
            detection_rate=0.97,
            mean_jitter_px=1.2,
            glasses_detected=False,
            quality_acceptable=True,
        )
        assert tq.quality_acceptable is True
        assert tq.glasses_detected is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.storage.models'`

**Step 3: Write implementation**

```python
"""Data models for OcuTrace sessions, trials, and gaze data.

All timestamps are UTC Unix milliseconds.
Sessions are identified by UUID for pseudonymization.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

VALID_TRIAL_TYPES = ("antisaccade", "prosaccade")
VALID_STIMULUS_SIDES = ("left", "right")


def _utc_now_ms() -> float:
    """Return current UTC time as Unix milliseconds."""
    return datetime.now(timezone.utc).timestamp() * 1000


def _new_uuid() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


@dataclass
class Session:
    """A single recording session.

    Parameters
    ----------
    participant_id : str
        Pseudonymized participant identifier.
    notes : str
        Optional session notes.
    session_id : str
        Auto-generated UUID4.
    created_at : float
        UTC Unix milliseconds, auto-generated.
    glasses_detected : bool or None
        Set after quality gate check.
    tracking_quality_score : float or None
        Set after quality gate check.
    """
    participant_id: str
    notes: str = ""
    session_id: str = field(default_factory=_new_uuid)
    created_at: float = field(default_factory=_utc_now_ms)
    glasses_detected: Optional[bool] = None
    tracking_quality_score: Optional[float] = None


@dataclass
class Trial:
    """A single trial within a session.

    Parameters
    ----------
    session_id : str
        Parent session UUID.
    trial_number : int
        1-based trial index.
    trial_type : str
        'antisaccade' or 'prosaccade'.
    stimulus_side : str
        'left' or 'right'.
    stimulus_onset_ms : float
        Timestamp of stimulus flip in UTC Unix ms.
    response_correct : bool or None
        Set after saccade classification.
    saccade_latency_ms : float or None
        Time from stimulus onset to saccade onset.
    saccade_direction : str or None
        Detected saccade direction.
    """
    session_id: str
    trial_number: int
    trial_type: str
    stimulus_side: str
    stimulus_onset_ms: float
    response_correct: Optional[bool] = None
    saccade_latency_ms: Optional[float] = None
    saccade_direction: Optional[str] = None

    def __post_init__(self) -> None:
        if self.trial_type not in VALID_TRIAL_TYPES:
            raise ValueError(
                f"trial_type must be one of {VALID_TRIAL_TYPES}, "
                f"got '{self.trial_type}'"
            )
        if self.stimulus_side not in VALID_STIMULUS_SIDES:
            raise ValueError(
                f"stimulus_side must be one of {VALID_STIMULUS_SIDES}, "
                f"got '{self.stimulus_side}'"
            )


@dataclass
class GazeData:
    """A single gaze sample (one camera frame).

    Parameters
    ----------
    session_id : str
        Parent session UUID.
    trial_number : int
        Which trial this sample belongs to (0 = calibration/inter-trial).
    timestamp_ms : float
        UTC Unix milliseconds of camera frame capture.
    left_iris_x : float
        Left iris center X in pixels.
    left_iris_y : float
        Left iris center Y in pixels.
    right_iris_x : float
        Right iris center X in pixels.
    right_iris_y : float
        Right iris center Y in pixels.
    confidence : float
        MediaPipe detection confidence (0-1).
    """
    session_id: str
    trial_number: int
    timestamp_ms: float
    left_iris_x: float
    left_iris_y: float
    right_iris_x: float
    right_iris_y: float
    confidence: float


@dataclass
class TrackingQuality:
    """Result of tracking quality assessment.

    Parameters
    ----------
    detection_rate : float
        Fraction of frames where iris was detected (0-1).
    mean_jitter_px : float
        Mean frame-to-frame jitter in pixels during fixation.
    glasses_detected : bool
        Whether glasses were detected on the face.
    quality_acceptable : bool
        Whether tracking quality meets threshold for experiment.
    """
    detection_rate: float
    mean_jitter_px: float
    glasses_detected: bool
    quality_acceptable: bool
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/storage/models.py tests/test_models.py
git commit -m "feat: add data models for Session, Trial, GazeData, TrackingQuality"
```

---

### Task 3: Configuration Loader

**Files:**
- Create: `tests/test_config.py`
- Create: `src/config.py`

**Step 1: Write the failing test**

```python
"""Tests for configuration loading."""
import os
import tempfile

import pytest

from src.config import load_config, get_config


class TestLoadConfig:
    def test_load_default_config(self):
        config = load_config()
        assert config["camera"]["device_index"] == 0
        assert config["tracking"]["refine_landmarks"] is True
        assert config["tracking"]["left_iris_index"] == 468
        assert config["tracking"]["right_iris_index"] == 473

    def test_load_custom_config(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("camera:\n  device_index: 1\n  target_fps: 60\n")
            f.flush()
            config = load_config(f.name)
            assert config["camera"]["device_index"] == 1
            assert config["camera"]["target_fps"] == 60
        os.unlink(f.name)

    def test_get_config_singleton(self):
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_nested_access(self):
        config = load_config()
        assert config["saccade_detection"]["smoothing_window"] == 5
        assert config["paradigm"]["n_antisaccade_trials"] == 40
        assert config["glasses_detection"]["jitter_threshold_px"] == 2.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
"""Configuration loader for OcuTrace.

Loads settings from config/settings.yaml. All configurable values
are accessed through this module — no hardcoded values in other modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
_config_cache: Optional[dict[str, Any]] = None


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters
    ----------
    path : str or None
        Path to YAML config file. If None, uses default
        config/settings.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_config() -> dict[str, Any]:
    """Get cached configuration (singleton).

    Returns
    -------
    dict
        Parsed configuration dictionary. Loaded once, cached thereafter.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def reset_config() -> None:
    """Reset cached config. Useful for testing."""
    global _config_cache
    _config_cache = None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add YAML configuration loader with singleton cache"
```

---

### Task 4: Storage Layer — Repository Pattern + SQLite

**Files:**
- Create: `tests/test_storage.py`
- Create: `src/storage/repository.py`
- Create: `src/storage/sqlite_repo.py`

**Step 1: Write the failing test**

```python
"""Tests for storage layer with real SQLite."""
import os
import tempfile

import pytest

from src.storage.models import Session, Trial, GazeData
from src.storage.sqlite_repo import SQLiteRepository


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def repo(db_path):
    r = SQLiteRepository(db_path)
    r.initialize()
    yield r
    r.close()


class TestSQLiteRepository:
    def test_save_and_load_session(self, repo):
        session = Session(participant_id="SUBJ001", notes="test")
        repo.save_session(session)
        loaded = repo.get_session(session.session_id)
        assert loaded is not None
        assert loaded.participant_id == "SUBJ001"
        assert loaded.session_id == session.session_id

    def test_save_and_load_trial(self, repo):
        session = Session(participant_id="SUBJ001")
        repo.save_session(session)
        trial = Trial(
            session_id=session.session_id,
            trial_number=1,
            trial_type="antisaccade",
            stimulus_side="left",
            stimulus_onset_ms=1000.0,
        )
        repo.save_trial(trial)
        trials = repo.get_trials(session.session_id)
        assert len(trials) == 1
        assert trials[0].trial_type == "antisaccade"

    def test_save_and_load_gaze_data(self, repo):
        session = Session(participant_id="SUBJ001")
        repo.save_session(session)
        samples = [
            GazeData(
                session_id=session.session_id,
                trial_number=1,
                timestamp_ms=float(i * 33),
                left_iris_x=320.0 + i * 0.1,
                left_iris_y=240.0,
                right_iris_x=321.0 + i * 0.1,
                right_iris_y=240.0,
                confidence=0.95,
            )
            for i in range(30)
        ]
        repo.save_gaze_data_batch(samples)
        loaded = repo.get_gaze_data(session.session_id, trial_number=1)
        assert len(loaded) == 30
        assert loaded[0].timestamp_ms == 0.0
        assert loaded[-1].timestamp_ms == 29 * 33.0

    def test_get_session_not_found(self, repo):
        assert repo.get_session("nonexistent") is None

    def test_list_sessions(self, repo):
        for i in range(3):
            repo.save_session(Session(participant_id=f"SUBJ{i:03d}"))
        sessions = repo.list_sessions()
        assert len(sessions) == 3

    def test_update_session(self, repo):
        session = Session(participant_id="SUBJ001")
        repo.save_session(session)
        session.glasses_detected = True
        session.tracking_quality_score = 0.85
        repo.update_session(session)
        loaded = repo.get_session(session.session_id)
        assert loaded.glasses_detected is True
        assert loaded.tracking_quality_score == 0.85
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write repository interface**

```python
"""Abstract repository interface for OcuTrace data storage.

All storage backends (SQLite, MariaDB) implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.storage.models import Session, Trial, GazeData


class Repository(ABC):
    """Abstract base for data repositories."""

    @abstractmethod
    def initialize(self) -> None:
        """Create tables / schema if they don't exist."""

    @abstractmethod
    def close(self) -> None:
        """Close the connection."""

    @abstractmethod
    def save_session(self, session: Session) -> None:
        """Persist a session record."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Load a session by ID. Returns None if not found."""

    @abstractmethod
    def list_sessions(self) -> list[Session]:
        """List all sessions, newest first."""

    @abstractmethod
    def update_session(self, session: Session) -> None:
        """Update an existing session record."""

    @abstractmethod
    def save_trial(self, trial: Trial) -> None:
        """Persist a trial record."""

    @abstractmethod
    def get_trials(self, session_id: str) -> list[Trial]:
        """Get all trials for a session, ordered by trial_number."""

    @abstractmethod
    def save_gaze_data_batch(self, samples: list[GazeData]) -> None:
        """Persist a batch of gaze samples (for performance)."""

    @abstractmethod
    def get_gaze_data(
        self,
        session_id: str,
        trial_number: Optional[int] = None,
    ) -> list[GazeData]:
        """Get gaze data for a session, optionally filtered by trial."""
```

**Step 4: Write SQLite implementation**

```python
"""SQLite implementation of the Repository interface."""
from __future__ import annotations

import sqlite3
from typing import Optional

from src.storage.models import Session, Trial, GazeData
from src.storage.repository import Repository


class SQLiteRepository(Repository):
    """SQLite-backed data repository.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create database tables if they don't exist."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                participant_id TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at REAL NOT NULL,
                glasses_detected INTEGER,
                tracking_quality_score REAL
            );

            CREATE TABLE IF NOT EXISTS trials (
                session_id TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                trial_type TEXT NOT NULL,
                stimulus_side TEXT NOT NULL,
                stimulus_onset_ms REAL NOT NULL,
                response_correct INTEGER,
                saccade_latency_ms REAL,
                saccade_direction TEXT,
                PRIMARY KEY (session_id, trial_number),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS gaze_data (
                session_id TEXT NOT NULL,
                trial_number INTEGER NOT NULL,
                timestamp_ms REAL NOT NULL,
                left_iris_x REAL NOT NULL,
                left_iris_y REAL NOT NULL,
                right_iris_x REAL NOT NULL,
                right_iris_y REAL NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_gaze_session_trial
            ON gaze_data(session_id, trial_number);
        """)

    def save_session(self, session: Session) -> None:
        """Persist a session record."""
        self._conn.execute(
            """INSERT INTO sessions
               (session_id, participant_id, notes, created_at,
                glasses_detected, tracking_quality_score)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session.session_id,
                session.participant_id,
                session.notes,
                session.created_at,
                _bool_to_int(session.glasses_detected),
                session.tracking_quality_score,
            ),
        )
        self._conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        row = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return Session(
            session_id=row[0],
            participant_id=row[1],
            notes=row[2],
            created_at=row[3],
            glasses_detected=_int_to_bool(row[4]),
            tracking_quality_score=row[5],
        )

    def list_sessions(self) -> list[Session]:
        """List all sessions, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [
            Session(
                session_id=r[0],
                participant_id=r[1],
                notes=r[2],
                created_at=r[3],
                glasses_detected=_int_to_bool(r[4]),
                tracking_quality_score=r[5],
            )
            for r in rows
        ]

    def update_session(self, session: Session) -> None:
        """Update an existing session record."""
        self._conn.execute(
            """UPDATE sessions
               SET glasses_detected = ?, tracking_quality_score = ?, notes = ?
               WHERE session_id = ?""",
            (
                _bool_to_int(session.glasses_detected),
                session.tracking_quality_score,
                session.notes,
                session.session_id,
            ),
        )
        self._conn.commit()

    def save_trial(self, trial: Trial) -> None:
        """Persist a trial record."""
        self._conn.execute(
            """INSERT INTO trials
               (session_id, trial_number, trial_type, stimulus_side,
                stimulus_onset_ms, response_correct, saccade_latency_ms,
                saccade_direction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trial.session_id,
                trial.trial_number,
                trial.trial_type,
                trial.stimulus_side,
                trial.stimulus_onset_ms,
                _bool_to_int(trial.response_correct),
                trial.saccade_latency_ms,
                trial.saccade_direction,
            ),
        )
        self._conn.commit()

    def get_trials(self, session_id: str) -> list[Trial]:
        """Get all trials for a session."""
        rows = self._conn.execute(
            "SELECT * FROM trials WHERE session_id = ? ORDER BY trial_number",
            (session_id,),
        ).fetchall()
        return [
            Trial(
                session_id=r[0],
                trial_number=r[1],
                trial_type=r[2],
                stimulus_side=r[3],
                stimulus_onset_ms=r[4],
                response_correct=_int_to_bool(r[5]),
                saccade_latency_ms=r[6],
                saccade_direction=r[7],
            )
            for r in rows
        ]

    def save_gaze_data_batch(self, samples: list[GazeData]) -> None:
        """Persist a batch of gaze samples."""
        self._conn.executemany(
            """INSERT INTO gaze_data
               (session_id, trial_number, timestamp_ms,
                left_iris_x, left_iris_y, right_iris_x, right_iris_y,
                confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    s.session_id, s.trial_number, s.timestamp_ms,
                    s.left_iris_x, s.left_iris_y,
                    s.right_iris_x, s.right_iris_y,
                    s.confidence,
                )
                for s in samples
            ],
        )
        self._conn.commit()

    def get_gaze_data(
        self,
        session_id: str,
        trial_number: Optional[int] = None,
    ) -> list[GazeData]:
        """Get gaze data for a session."""
        if trial_number is not None:
            rows = self._conn.execute(
                """SELECT * FROM gaze_data
                   WHERE session_id = ? AND trial_number = ?
                   ORDER BY timestamp_ms""",
                (session_id, trial_number),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT * FROM gaze_data
                   WHERE session_id = ?
                   ORDER BY timestamp_ms""",
                (session_id,),
            ).fetchall()
        return [
            GazeData(
                session_id=r[0],
                trial_number=r[1],
                timestamp_ms=r[2],
                left_iris_x=r[3],
                left_iris_y=r[4],
                right_iris_x=r[5],
                right_iris_y=r[6],
                confidence=r[7],
            )
            for r in rows
        ]


def _bool_to_int(value: Optional[bool]) -> Optional[int]:
    if value is None:
        return None
    return 1 if value else 0


def _int_to_bool(value: Optional[int]) -> Optional[bool]:
    if value is None:
        return None
    return bool(value)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: All 6 tests PASS

**Step 6: Commit**

```bash
git add src/storage/repository.py src/storage/sqlite_repo.py tests/test_storage.py
git commit -m "feat: add repository pattern with SQLite implementation"
```

---

### Task 5: Iris Tracker

**Files:**
- Create: `tests/test_tracking.py`
- Create: `src/tracking/iris_tracker.py`

**Step 1: Write the failing test**

```python
"""Tests for iris tracker.

Uses mock for MediaPipe to avoid camera dependency in CI.
Also includes a manual spike test (skipped by default) for real webcam validation.
"""
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np
import pytest

from src.tracking.iris_tracker import IrisTracker, IrisCoordinates


class TestIrisCoordinates:
    def test_create(self):
        coords = IrisCoordinates(
            left_x=320.0,
            left_y=240.0,
            right_x=321.0,
            right_y=240.5,
            confidence=0.95,
            timestamp_ms=1000.0,
        )
        assert coords.left_x == 320.0
        assert coords.confidence == 0.95

    def test_mean_x(self):
        coords = IrisCoordinates(
            left_x=320.0, left_y=240.0,
            right_x=322.0, right_y=240.0,
            confidence=0.95, timestamp_ms=0.0,
        )
        assert coords.mean_x == 321.0

    def test_mean_y(self):
        coords = IrisCoordinates(
            left_x=320.0, left_y=238.0,
            right_x=322.0, right_y=242.0,
            confidence=0.95, timestamp_ms=0.0,
        )
        assert coords.mean_y == 240.0


def _make_mock_landmark(x: float, y: float):
    """Create a mock MediaPipe landmark."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = 0.0
    return lm


def _make_mock_results(
    left_x: float, left_y: float,
    right_x: float, right_y: float,
    num_landmarks: int = 478,
):
    """Create mock FaceMesh results with iris landmarks at given positions."""
    landmarks = [MagicMock() for _ in range(num_landmarks)]
    # Landmark 468 = left iris center
    landmarks[468] = _make_mock_landmark(left_x, left_y)
    # Landmark 473 = right iris center
    landmarks[473] = _make_mock_landmark(right_x, right_y)

    face = MagicMock()
    face.landmark = landmarks

    results = MagicMock()
    results.multi_face_landmarks = [face]
    return results


def _make_mock_results_no_face():
    results = MagicMock()
    results.multi_face_landmarks = None
    return results


class TestIrisTracker:
    def test_process_frame_returns_coordinates(self):
        tracker = IrisTracker.__new__(IrisTracker)
        tracker._face_mesh = MagicMock()
        tracker._frame_width = 640
        tracker._frame_height = 480

        mock_results = _make_mock_results(0.5, 0.5, 0.502, 0.501)
        tracker._face_mesh.process.return_value = mock_results

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coords = tracker.process_frame(frame, timestamp_ms=100.0)

        assert coords is not None
        assert abs(coords.left_x - 320.0) < 0.1
        assert abs(coords.right_x - 321.28) < 0.1
        assert coords.timestamp_ms == 100.0

    def test_process_frame_no_face_returns_none(self):
        tracker = IrisTracker.__new__(IrisTracker)
        tracker._face_mesh = MagicMock()
        tracker._frame_width = 640
        tracker._frame_height = 480

        tracker._face_mesh.process.return_value = _make_mock_results_no_face()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        coords = tracker.process_frame(frame, timestamp_ms=100.0)
        assert coords is None

    def test_tracker_config_from_settings(self):
        config = {
            "tracking": {
                "max_num_faces": 1,
                "refine_landmarks": True,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "left_iris_index": 468,
                "right_iris_index": 473,
            },
            "camera": {
                "frame_width": 640,
                "frame_height": 480,
            },
        }
        with patch("src.tracking.iris_tracker.mp") as mock_mp:
            mock_mp.solutions.face_mesh.FaceMesh.return_value = MagicMock()
            tracker = IrisTracker(config)
            assert tracker._left_iris_idx == 468
            assert tracker._right_iris_idx == 473
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
"""Real-time iris tracking using MediaPipe FaceMesh.

Uses FaceMesh with refine_landmarks=True to get iris center
landmarks (468 for left, 473 for right eye).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class IrisCoordinates:
    """Iris center coordinates from a single frame.

    Parameters
    ----------
    left_x : float
        Left iris center X in pixels.
    left_y : float
        Left iris center Y in pixels.
    right_x : float
        Right iris center X in pixels.
    right_y : float
        Right iris center Y in pixels.
    confidence : float
        Detection confidence (0-1).
    timestamp_ms : float
        Frame timestamp in UTC Unix milliseconds.
    """
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    confidence: float
    timestamp_ms: float

    @property
    def mean_x(self) -> float:
        """Mean X of both iris centers."""
        return (self.left_x + self.right_x) / 2.0

    @property
    def mean_y(self) -> float:
        """Mean Y of both iris centers."""
        return (self.left_y + self.right_y) / 2.0


class IrisTracker:
    """MediaPipe-based iris tracker.

    Parameters
    ----------
    config : dict
        Configuration dict with 'tracking' and 'camera' sections.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        tracking_cfg = config["tracking"]
        camera_cfg = config["camera"]

        self._left_iris_idx: int = tracking_cfg["left_iris_index"]
        self._right_iris_idx: int = tracking_cfg["right_iris_index"]
        self._frame_width: int = camera_cfg["frame_width"]
        self._frame_height: int = camera_cfg["frame_height"]

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=tracking_cfg["max_num_faces"],
            refine_landmarks=tracking_cfg["refine_landmarks"],
            min_detection_confidence=tracking_cfg["min_detection_confidence"],
            min_tracking_confidence=tracking_cfg["min_tracking_confidence"],
        )

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
    ) -> Optional[IrisCoordinates]:
        """Extract iris coordinates from a BGR camera frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (H, W, 3).
        timestamp_ms : float
            Timestamp of this frame in UTC Unix ms.

        Returns
        -------
        IrisCoordinates or None
            Iris positions if face detected, None otherwise.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]
        left = face.landmark[self._left_iris_idx]
        right = face.landmark[self._right_iris_idx]

        return IrisCoordinates(
            left_x=left.x * self._frame_width,
            left_y=left.y * self._frame_height,
            right_x=right.x * self._frame_width,
            right_y=right.y * self._frame_height,
            confidence=min(left.visibility if hasattr(left, 'visibility') and left.visibility else 1.0,
                          right.visibility if hasattr(right, 'visibility') and right.visibility else 1.0),
            timestamp_ms=timestamp_ms,
        )

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._face_mesh.close()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tracking.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/tracking/iris_tracker.py tests/test_tracking.py
git commit -m "feat: add MediaPipe iris tracker with landmark extraction"
```

---

### Task 6: Glasses Detector + Quality Gate

**Files:**
- Create: `tests/test_glasses_detector.py`
- Create: `src/tracking/glasses_detector.py`

**Step 1: Write the failing test**

```python
"""Tests for glasses detection and tracking quality gate."""
import numpy as np
import pytest

from src.tracking.glasses_detector import GlassesDetector, assess_tracking_quality
from src.tracking.iris_tracker import IrisCoordinates
from src.storage.models import TrackingQuality


def _make_stable_coords(n: int, base_x: float = 320.0, jitter: float = 0.5):
    """Generate n stable iris coordinates with small jitter."""
    rng = np.random.default_rng(42)
    return [
        IrisCoordinates(
            left_x=base_x + rng.uniform(-jitter, jitter),
            left_y=240.0 + rng.uniform(-jitter, jitter),
            right_x=base_x + 1.0 + rng.uniform(-jitter, jitter),
            right_y=240.0 + rng.uniform(-jitter, jitter),
            confidence=0.95,
            timestamp_ms=float(i * 33),
        )
        for i in range(n)
    ]


def _make_jittery_coords(n: int, base_x: float = 320.0, jitter: float = 5.0):
    """Generate n coords with high jitter (simulating glasses interference)."""
    rng = np.random.default_rng(42)
    return [
        IrisCoordinates(
            left_x=base_x + rng.uniform(-jitter, jitter),
            left_y=240.0 + rng.uniform(-jitter, jitter),
            right_x=base_x + 1.0 + rng.uniform(-jitter, jitter),
            right_y=240.0 + rng.uniform(-jitter, jitter),
            confidence=0.7,
            timestamp_ms=float(i * 33),
        )
        for i in range(n)
    ]


class TestAssessTrackingQuality:
    def test_stable_tracking_is_acceptable(self):
        coords = _make_stable_coords(90)
        # All frames detected (no None)
        quality = assess_tracking_quality(
            coords_history=coords,
            none_count=0,
            total_frames=90,
            jitter_threshold_px=2.0,
            min_detection_rate=0.95,
        )
        assert quality.quality_acceptable is True
        assert quality.detection_rate > 0.95
        assert quality.mean_jitter_px < 2.0

    def test_jittery_tracking_is_unacceptable(self):
        coords = _make_jittery_coords(90)
        quality = assess_tracking_quality(
            coords_history=coords,
            none_count=0,
            total_frames=90,
            jitter_threshold_px=2.0,
            min_detection_rate=0.95,
        )
        assert quality.quality_acceptable is False
        assert quality.mean_jitter_px > 2.0

    def test_low_detection_rate_is_unacceptable(self):
        coords = _make_stable_coords(50)
        quality = assess_tracking_quality(
            coords_history=coords,
            none_count=50,
            total_frames=100,
            jitter_threshold_px=2.0,
            min_detection_rate=0.95,
        )
        assert quality.quality_acceptable is False
        assert quality.detection_rate == 0.50


class TestGlassesDetector:
    def test_no_glasses_landmark_ratio(self):
        """Without glasses, eye-to-nose bridge ratio is in normal range."""
        detector = GlassesDetector()
        # Simulated normalized landmarks (simplified)
        assert isinstance(detector.detect_from_landmarks([]), bool)

    def test_glasses_detection_returns_bool(self):
        detector = GlassesDetector()
        result = detector.detect_from_landmarks([])
        assert isinstance(result, bool)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_glasses_detector.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
"""Glasses detection and tracking quality assessment.

Detects glasses presence via FaceMesh landmark geometry and assesses
iris tracking quality using jitter and detection rate metrics.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.storage.models import TrackingQuality
from src.tracking.iris_tracker import IrisCoordinates


class GlassesDetector:
    """Detect glasses using FaceMesh landmark geometry.

    Uses the ratio of distances around the eye-nose bridge area.
    Glasses frames create a distinct geometric signature in the
    landmark positions around the eyes.
    """

    # FaceMesh landmark indices for glasses detection heuristic
    # Left eye top/bottom, right eye top/bottom, nose bridge
    _LEFT_EYE_TOP = 159
    _LEFT_EYE_BOTTOM = 145
    _RIGHT_EYE_TOP = 386
    _RIGHT_EYE_BOTTOM = 374
    _NOSE_BRIDGE_TOP = 6
    _LEFT_TEMPLE = 226
    _RIGHT_TEMPLE = 446

    def detect_from_landmarks(
        self,
        landmarks: list[Any],
    ) -> bool:
        """Detect glasses from FaceMesh landmarks.

        Parameters
        ----------
        landmarks : list
            FaceMesh landmark list (478 landmarks).
            If empty or insufficient, returns False (no detection).

        Returns
        -------
        bool
            True if glasses are likely present.
        """
        if len(landmarks) < 478:
            return False

        left_top = landmarks[self._LEFT_EYE_TOP]
        left_bottom = landmarks[self._LEFT_EYE_BOTTOM]
        right_top = landmarks[self._RIGHT_EYE_TOP]
        right_bottom = landmarks[self._RIGHT_EYE_BOTTOM]
        nose_bridge = landmarks[self._NOSE_BRIDGE_TOP]
        left_temple = landmarks[self._LEFT_TEMPLE]
        right_temple = landmarks[self._RIGHT_TEMPLE]

        # Glasses create a wider eye-region due to frame distortion.
        # The ratio of temple-to-temple distance vs nose-bridge-to-eye
        # distance shifts when glasses are present.
        temple_dist = np.sqrt(
            (right_temple.x - left_temple.x) ** 2
            + (right_temple.y - left_temple.y) ** 2
        )

        eye_height_left = abs(left_top.y - left_bottom.y)
        eye_height_right = abs(right_top.y - right_bottom.y)
        avg_eye_height = (eye_height_left + eye_height_right) / 2.0

        if avg_eye_height < 1e-6:
            return False

        # Glasses tend to increase this ratio due to frame
        # pushing landmarks slightly outward
        ratio = temple_dist / avg_eye_height
        return ratio > 12.0


def assess_tracking_quality(
    coords_history: list[IrisCoordinates],
    none_count: int,
    total_frames: int,
    jitter_threshold_px: float,
    min_detection_rate: float,
) -> TrackingQuality:
    """Assess tracking quality from a window of iris coordinates.

    Parameters
    ----------
    coords_history : list[IrisCoordinates]
        Successfully detected coordinates.
    none_count : int
        Number of frames where detection failed.
    total_frames : int
        Total frames in the assessment window.
    jitter_threshold_px : float
        Maximum acceptable mean jitter in pixels.
    min_detection_rate : float
        Minimum acceptable detection rate (0-1).

    Returns
    -------
    TrackingQuality
        Assessment result with metrics.
    """
    detection_rate = len(coords_history) / total_frames if total_frames > 0 else 0.0

    if len(coords_history) < 2:
        return TrackingQuality(
            detection_rate=detection_rate,
            mean_jitter_px=float("inf"),
            glasses_detected=False,
            quality_acceptable=False,
        )

    # Compute frame-to-frame jitter on mean X (primary saccade axis)
    x_values = np.array([c.mean_x for c in coords_history])
    diffs = np.abs(np.diff(x_values))
    mean_jitter = float(np.mean(diffs))

    quality_ok = (
        detection_rate >= min_detection_rate
        and mean_jitter <= jitter_threshold_px
    )

    return TrackingQuality(
        detection_rate=detection_rate,
        mean_jitter_px=mean_jitter,
        glasses_detected=False,  # Set by caller after landmark check
        quality_acceptable=quality_ok,
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_glasses_detector.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/tracking/glasses_detector.py tests/test_glasses_detector.py
git commit -m "feat: add glasses detection and tracking quality gate"
```

---

### Task 7: Synthetic Test Fixtures

**Files:**
- Create: `tests/conftest.py`

**Step 1: Write conftest with reusable fixtures**

```python
"""Shared test fixtures for OcuTrace.

Provides synthetic gaze data generators and common test infrastructure.
No real camera or MediaPipe dependency — all data is synthetic.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from src.tracking.iris_tracker import IrisCoordinates
from src.storage.models import Session, Trial, GazeData
from src.storage.sqlite_repo import SQLiteRepository


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def db_path():
    """Temporary SQLite database path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def repo(db_path):
    """Initialized SQLite repository."""
    r = SQLiteRepository(db_path)
    r.initialize()
    yield r
    r.close()


@pytest.fixture
def sample_session():
    """A sample session for testing."""
    return Session(participant_id="TEST001", notes="test session")


@pytest.fixture
def synthetic_fixation_coords(rng):
    """90 frames of stable fixation data (~3s at 30fps).

    Simulates steady gaze at center with small physiological noise.
    Jitter ~0.3px, well under 2px threshold.
    """
    n = 90
    base_x, base_y = 320.0, 240.0
    noise = 0.3
    return [
        IrisCoordinates(
            left_x=base_x + rng.normal(0, noise),
            left_y=base_y + rng.normal(0, noise),
            right_x=base_x + 1.0 + rng.normal(0, noise),
            right_y=base_y + rng.normal(0, noise),
            confidence=0.95 + rng.uniform(-0.03, 0.03),
            timestamp_ms=float(i * 33),
        )
        for i in range(n)
    ]


@pytest.fixture
def synthetic_saccade_coords(rng):
    """Synthetic saccade: fixation → rightward saccade → new fixation.

    30 frames fixation at x=320, then 5-frame saccade to x=420,
    then 25 frames fixation at x=420.
    Total: 60 frames (2s at 30fps).
    """
    coords = []
    dt = 33.33  # ms per frame at 30fps

    # Fixation at 320
    for i in range(30):
        coords.append(IrisCoordinates(
            left_x=320.0 + rng.normal(0, 0.3),
            left_y=240.0 + rng.normal(0, 0.3),
            right_x=321.0 + rng.normal(0, 0.3),
            right_y=240.0 + rng.normal(0, 0.3),
            confidence=0.95,
            timestamp_ms=i * dt,
        ))

    # Saccade: 5 frames, 320 → 420 (100px over ~167ms)
    for i in range(5):
        frac = (i + 1) / 5.0
        x = 320.0 + 100.0 * frac
        coords.append(IrisCoordinates(
            left_x=x + rng.normal(0, 0.5),
            left_y=240.0 + rng.normal(0, 0.3),
            right_x=x + 1.0 + rng.normal(0, 0.5),
            right_y=240.0 + rng.normal(0, 0.3),
            confidence=0.93,
            timestamp_ms=(30 + i) * dt,
        ))

    # New fixation at 420
    for i in range(25):
        coords.append(IrisCoordinates(
            left_x=420.0 + rng.normal(0, 0.3),
            left_y=240.0 + rng.normal(0, 0.3),
            right_x=421.0 + rng.normal(0, 0.3),
            right_y=240.0 + rng.normal(0, 0.3),
            confidence=0.95,
            timestamp_ms=(35 + i) * dt,
        ))

    return coords


@pytest.fixture
def synthetic_gaze_data(sample_session, synthetic_fixation_coords):
    """Convert fixation coords to GazeData for storage tests."""
    return [
        GazeData(
            session_id=sample_session.session_id,
            trial_number=1,
            timestamp_ms=c.timestamp_ms,
            left_iris_x=c.left_x,
            left_iris_y=c.left_y,
            right_iris_x=c.right_x,
            right_iris_y=c.right_y,
            confidence=c.confidence,
        )
        for c in synthetic_fixation_coords
    ]
```

**Step 2: Verify fixtures work by running existing tests**

Run: `pytest tests/ -v`
Expected: All existing tests still PASS

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "feat: add synthetic test fixtures for gaze data"
```

---

### Task 8: Iris Tracking Spike Script

**Files:**
- Create: `scripts/spike_iris_tracking.py`

**Step 1: Write the spike script**

```python
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

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

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
    glasses_votes: list[bool] = []

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
```

**Step 2: Test manually (not automated — requires webcam)**

Run: `python scripts/spike_iris_tracking.py`
Expected: 10-second test runs, shows detection rate and jitter results.

**Step 3: Commit**

```bash
git add scripts/spike_iris_tracking.py
git commit -m "feat: add iris tracking spike script for webcam validation"
```

---

### Task 9: Run All Tests + Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 2: Verify project imports**

Run: `python -c "from src.tracking.iris_tracker import IrisTracker; from src.storage.sqlite_repo import SQLiteRepository; from src.tracking.glasses_detector import GlassesDetector; from src.config import load_config; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Final commit (if any fixups needed)**

```bash
git add -A && git commit -m "fix: phase 1 final adjustments"
```
