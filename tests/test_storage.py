"""Tests for storage layer with real SQLite."""
import numpy as np
import pytest

from src.storage.models import Session, Trial, GazeData, CalibrationPoint, CalibrationResult
from src.storage.sqlite_repo import SQLiteRepository


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


class TestCalibrationStorage:
    def test_save_and_load_calibration(self, repo):
        session = Session(participant_id="SUBJ001")
        repo.save_session(session)
        points = [
            CalibrationPoint(float(x), 0.0, 320.0 + x * 13.0, 240.0)
            for x in [-10, -5, 0, 5, 10]
        ]
        cal = CalibrationResult(
            session_id=session.session_id, points=points,
            transform_matrix=np.eye(3).tolist(),
            mean_error_deg=0.5, accepted=True,
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
