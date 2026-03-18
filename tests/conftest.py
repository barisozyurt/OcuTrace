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
    """Synthetic saccade: fixation -> rightward saccade -> new fixation.

    30 frames fixation at x=320, then 5-frame saccade to x=420,
    then 25 frames fixation at x=420.
    Total: 60 frames (2s at 30fps).
    """
    coords = []
    dt = 33.33

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

    # Saccade: 5 frames, 320 -> 420 (100px over ~167ms)
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
