"""SQLite implementation of the Repository interface."""
from __future__ import annotations

import json
import sqlite3
from typing import Optional

from src.storage.models import Session, Trial, GazeData, CalibrationResult, CalibrationPoint
from src.storage.repository import Repository


class SQLiteRepository(Repository):
    """SQLite-backed repository for OcuTrace data.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create database connection and tables."""
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
        """Create tables and indices if they don't exist."""
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
            CREATE TABLE IF NOT EXISTS calibrations (
                session_id TEXT PRIMARY KEY,
                points_json TEXT NOT NULL,
                transform_matrix_json TEXT NOT NULL,
                mean_error_deg REAL NOT NULL,
                accepted INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
        """)

    def save_session(self, session: Session) -> None:
        """Persist a session record."""
        self._conn.execute(
            "INSERT INTO sessions "
            "(session_id, participant_id, notes, created_at, "
            "glasses_detected, tracking_quality_score) "
            "VALUES (?, ?, ?, ?, ?, ?)",
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
        """Load a session by ID. Returns None if not found."""
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
            "UPDATE sessions SET glasses_detected = ?, "
            "tracking_quality_score = ?, notes = ? "
            "WHERE session_id = ?",
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
            "INSERT INTO trials "
            "(session_id, trial_number, trial_type, stimulus_side, "
            "stimulus_onset_ms, response_correct, saccade_latency_ms, "
            "saccade_direction) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
        """Get all trials for a session, ordered by trial_number."""
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
        """Persist a batch of gaze samples using executemany."""
        self._conn.executemany(
            "INSERT INTO gaze_data "
            "(session_id, trial_number, timestamp_ms, left_iris_x, "
            "left_iris_y, right_iris_x, right_iris_y, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    s.session_id,
                    s.trial_number,
                    s.timestamp_ms,
                    s.left_iris_x,
                    s.left_iris_y,
                    s.right_iris_x,
                    s.right_iris_y,
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
        """Get gaze data for a session, optionally filtered by trial."""
        if trial_number is not None:
            rows = self._conn.execute(
                "SELECT * FROM gaze_data "
                "WHERE session_id = ? AND trial_number = ? "
                "ORDER BY timestamp_ms",
                (session_id, trial_number),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM gaze_data WHERE session_id = ? "
                "ORDER BY timestamp_ms",
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


    def save_calibration(self, calibration: CalibrationResult) -> None:
        """Persist a calibration result."""
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
        """Get calibration for a session."""
        row = self._conn.execute(
            "SELECT * FROM calibrations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        points = [CalibrationPoint(**p) for p in json.loads(row[1])]
        return CalibrationResult(
            session_id=row[0],
            points=points,
            transform_matrix=json.loads(row[2]),
            mean_error_deg=row[3],
            accepted=bool(row[4]),
        )


def _bool_to_int(value: Optional[bool]) -> Optional[int]:
    """Convert a Python bool to an integer for SQLite storage."""
    if value is None:
        return None
    return 1 if value else 0


def _int_to_bool(value: Optional[int]) -> Optional[bool]:
    """Convert an SQLite integer back to a Python bool."""
    if value is None:
        return None
    return bool(value)
