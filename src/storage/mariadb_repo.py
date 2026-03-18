"""MariaDB implementation of the Repository interface."""
from __future__ import annotations

import json
from typing import Optional

import pymysql

from src.storage.models import (
    CalibrationPoint,
    CalibrationResult,
    GazeData,
    Session,
    Trial,
)
from src.storage.repository import Repository


class MariaDBRepository(Repository):
    """MariaDB-backed repository for OcuTrace data.

    Parameters
    ----------
    host : str
        Database server hostname.
    port : int
        Database server port.
    user : str
        Database user.
    password : str
        Database password.
    database : str
        Database name.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        self._conn: Optional[pymysql.connections.Connection] = None

    def initialize(self) -> None:
        """Create database connection and tables."""
        self._conn = pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            autocommit=False,
        )
        self._create_tables()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        """Create tables and indices if they don't exist."""
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(36) PRIMARY KEY,
                participant_id VARCHAR(255) NOT NULL,
                notes TEXT DEFAULT '',
                created_at DOUBLE NOT NULL,
                glasses_detected TINYINT,
                tracking_quality_score DOUBLE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                session_id VARCHAR(36) NOT NULL,
                trial_number INT NOT NULL,
                trial_type VARCHAR(20) NOT NULL,
                stimulus_side VARCHAR(10) NOT NULL,
                stimulus_onset_ms DOUBLE NOT NULL,
                response_correct TINYINT,
                saccade_latency_ms DOUBLE,
                saccade_direction VARCHAR(10),
                PRIMARY KEY (session_id, trial_number),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gaze_data (
                session_id VARCHAR(36) NOT NULL,
                trial_number INT NOT NULL,
                timestamp_ms DOUBLE NOT NULL,
                left_iris_x DOUBLE NOT NULL,
                left_iris_y DOUBLE NOT NULL,
                right_iris_x DOUBLE NOT NULL,
                right_iris_y DOUBLE NOT NULL,
                confidence DOUBLE NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                INDEX idx_gaze_session_trial (session_id, trial_number)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibrations (
                session_id VARCHAR(36) PRIMARY KEY,
                points_json JSON NOT NULL,
                transform_matrix_json JSON NOT NULL,
                mean_error_deg DOUBLE NOT NULL,
                accepted TINYINT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        self._conn.commit()
        cursor.close()

    def save_session(self, session: Session) -> None:
        """Persist a session record."""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO sessions "
            "(session_id, participant_id, notes, created_at, "
            "glasses_detected, tracking_quality_score) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
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
        cursor.close()

    def get_session(self, session_id: str) -> Optional[Session]:
        """Load a session by ID. Returns None if not found."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cursor.fetchone()
        cursor.close()
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
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        cursor.close()
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
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE sessions SET glasses_detected = %s, "
            "tracking_quality_score = %s, notes = %s "
            "WHERE session_id = %s",
            (
                _bool_to_int(session.glasses_detected),
                session.tracking_quality_score,
                session.notes,
                session.session_id,
            ),
        )
        self._conn.commit()
        cursor.close()

    def save_trial(self, trial: Trial) -> None:
        """Persist a trial record."""
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT INTO trials "
            "(session_id, trial_number, trial_type, stimulus_side, "
            "stimulus_onset_ms, response_correct, saccade_latency_ms, "
            "saccade_direction) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
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
        cursor.close()

    def get_trials(self, session_id: str) -> list[Trial]:
        """Get all trials for a session, ordered by trial_number."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM trials WHERE session_id = %s ORDER BY trial_number",
            (session_id,),
        )
        rows = cursor.fetchall()
        cursor.close()
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
        cursor = self._conn.cursor()
        cursor.executemany(
            "INSERT INTO gaze_data "
            "(session_id, trial_number, timestamp_ms, left_iris_x, "
            "left_iris_y, right_iris_x, right_iris_y, confidence) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
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
        cursor.close()

    def get_gaze_data(
        self,
        session_id: str,
        trial_number: Optional[int] = None,
    ) -> list[GazeData]:
        """Get gaze data for a session, optionally filtered by trial."""
        cursor = self._conn.cursor()
        if trial_number is not None:
            cursor.execute(
                "SELECT * FROM gaze_data "
                "WHERE session_id = %s AND trial_number = %s "
                "ORDER BY timestamp_ms",
                (session_id, trial_number),
            )
        else:
            cursor.execute(
                "SELECT * FROM gaze_data WHERE session_id = %s "
                "ORDER BY timestamp_ms",
                (session_id,),
            )
        rows = cursor.fetchall()
        cursor.close()
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
        cursor = self._conn.cursor()
        cursor.execute(
            "REPLACE INTO calibrations "
            "(session_id, points_json, transform_matrix_json, "
            "mean_error_deg, accepted) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                calibration.session_id,
                points_json,
                matrix_json,
                calibration.mean_error_deg,
                1 if calibration.accepted else 0,
            ),
        )
        self._conn.commit()
        cursor.close()

    def get_calibration(self, session_id: str) -> Optional[CalibrationResult]:
        """Get calibration for a session."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM calibrations WHERE session_id = %s",
            (session_id,),
        )
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return None
        raw_points = row[1]
        if isinstance(raw_points, str):
            points_data = json.loads(raw_points)
        else:
            points_data = raw_points
        raw_matrix = row[2]
        if isinstance(raw_matrix, str):
            matrix_data = json.loads(raw_matrix)
        else:
            matrix_data = raw_matrix
        points = [CalibrationPoint(**p) for p in points_data]
        return CalibrationResult(
            session_id=row[0],
            points=points,
            transform_matrix=matrix_data,
            mean_error_deg=row[3],
            accepted=bool(row[4]),
        )


def _bool_to_int(value: Optional[bool]) -> Optional[int]:
    """Convert a Python bool to an integer for MariaDB storage."""
    if value is None:
        return None
    return 1 if value else 0


def _int_to_bool(value: Optional[int]) -> Optional[bool]:
    """Convert a MariaDB integer back to a Python bool."""
    if value is None:
        return None
    return bool(value)
