"""Centralized path resolution for OcuTrace.

Handles both development (source tree) and PyInstaller frozen (.exe)
environments. All modules should use these functions instead of
constructing paths relative to __file__.
"""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """Check if running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_bundle_dir() -> Path:
    """Return the directory containing bundled resources.

    In development: project root (parent of src/).
    In frozen mode: sys._MEIPASS (PyInstaller temp extraction dir).
    """
    if is_frozen():
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return the user-writable data directory.

    Always ~/Documents/OcuTrace/ — visible in File Explorer,
    survives .exe updates, and is easy for clinical staff to find.
    """
    base = Path.home() / "Documents" / "OcuTrace"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_config_path() -> Path:
    """Return path to settings.yaml."""
    return get_bundle_dir() / "config" / "settings.yaml"


def get_model_path() -> Path:
    """Return path to MediaPipe face_landmarker.task model."""
    return get_bundle_dir() / "models" / "face_landmarker.task"


def get_db_path() -> Path:
    """Return path to SQLite database file."""
    data = get_data_dir()
    return data / "ocutrace.db"


def get_reports_dir() -> Path:
    """Return directory for generated report files."""
    reports = get_data_dir() / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports
