"""Tests for the Flask dashboard."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from src.dashboard.app import create_app
from src.storage.models import Session, Trial
from src.storage.sqlite_repo import SQLiteRepository


@pytest.fixture
def dashboard_db():
    """Create a temporary SQLite database with sample data."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = SQLiteRepository(path)
    repo.initialize()

    session = Session(
        participant_id="SUBJ001",
        session_id="test-session-001",
        notes="dashboard test",
    )
    repo.save_session(session)

    for i in range(1, 4):
        trial = Trial(
            session_id="test-session-001",
            trial_number=i,
            trial_type="antisaccade" if i <= 2 else "prosaccade",
            stimulus_side="left" if i % 2 == 0 else "right",
            stimulus_onset_ms=1000.0 * i,
            response_correct=(i != 2),
            saccade_latency_ms=200.0 + i * 10,
            saccade_direction="right" if i % 2 == 0 else "left",
        )
        repo.save_trial(trial)

    repo.close()
    yield path
    os.unlink(path)


@pytest.fixture
def client(dashboard_db):
    """Flask test client with mocked config pointing to temp DB."""
    mock_config = {
        "storage": {
            "backend": "sqlite",
            "sqlite": {"database_path": dashboard_db},
        }
    }
    with patch("src.dashboard.views.load_config", return_value=mock_config):
        app = create_app({"TESTING": True})
        yield app.test_client()


@pytest.fixture
def empty_client():
    """Flask test client with an empty database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    repo = SQLiteRepository(path)
    repo.initialize()
    repo.close()

    mock_config = {
        "storage": {
            "backend": "sqlite",
            "sqlite": {"database_path": path},
        }
    }
    with patch("src.dashboard.views.load_config", return_value=mock_config):
        app = create_app({"TESTING": True})
        yield app.test_client()

    os.unlink(path)


class TestDashboard:
    """Tests for dashboard routes."""

    def test_index_returns_200(self, client):
        """Index page returns 200 OK."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_contains_title(self, client):
        """Index page contains OcuTrace title."""
        resp = client.get("/")
        assert b"OcuTrace" in resp.data

    def test_index_lists_session(self, client):
        """Index page shows the test session participant."""
        resp = client.get("/")
        assert b"SUBJ001" in resp.data

    def test_empty_index_shows_message(self, empty_client):
        """Empty database shows helpful message."""
        resp = empty_client.get("/")
        assert resp.status_code == 200
        assert b"No sessions recorded yet" in resp.data

    def test_session_detail_returns_200(self, client):
        """Valid session detail page returns 200."""
        resp = client.get("/session/test-session-001")
        assert resp.status_code == 200

    def test_session_detail_contains_participant(self, client):
        """Session detail shows participant ID."""
        resp = client.get("/session/test-session-001")
        assert b"SUBJ001" in resp.data

    def test_session_detail_contains_trials(self, client):
        """Session detail page includes trial data."""
        resp = client.get("/session/test-session-001")
        assert b"antisaccade" in resp.data

    def test_session_detail_contains_plots(self, client):
        """Session detail page includes base64 plot images."""
        resp = client.get("/session/test-session-001")
        assert b"data:image/png;base64," in resp.data

    def test_nonexistent_session_returns_404(self, client):
        """Nonexistent session returns 404."""
        resp = client.get("/session/nonexistent-id")
        assert resp.status_code == 404
