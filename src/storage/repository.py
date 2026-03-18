"""Abstract repository interface for OcuTrace data storage."""
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
