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
        assert isinstance(session.session_id, str)
        uuid.UUID(session.session_id)
        assert session.participant_id == "SUBJ001"
        assert isinstance(session.created_at, float)
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
        assert trial.response_correct is None

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
