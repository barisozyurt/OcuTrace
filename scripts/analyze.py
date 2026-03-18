"""Offline analysis and report generation for a recorded session.

Loads trial data from SQLite, computes session metrics, and generates
Matplotlib clinical reports.

Usage:
    python scripts/analyze.py --session SESSION_ID
    python scripts/analyze.py --latest
    python scripts/analyze.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.storage.sqlite_repo import SQLiteRepository
from src.analysis.metrics import compute_session_metrics
from src.experiment.session import print_session_summary
from src.visualization.reports import generate_session_report


def list_sessions(repo: SQLiteRepository) -> None:
    """Print all recorded sessions."""
    sessions = repo.list_sessions()
    if not sessions:
        print("No sessions found.")
        return

    print(f"{'Session ID':<40s}  {'Participant':<15s}  {'Notes':<15s}  {'Trials':>6s}")
    print("-" * 80)
    for s in sessions:
        trials = repo.get_trials(s.session_id)
        print(
            f"{s.session_id:<40s}  {s.participant_id:<15s}  "
            f"{s.notes or '':<15s}  {len(trials):>6d}"
        )


def analyze_session(session_id: str, repo: SQLiteRepository) -> None:
    """Analyze a session and generate reports."""
    config = load_config()

    session = repo.get_session(session_id)
    if session is None:
        print(f"ERROR: Session '{session_id}' not found.")
        sys.exit(1)

    trials = repo.get_trials(session_id)
    if not trials:
        print(f"ERROR: No trials found for session '{session_id}'.")
        sys.exit(1)

    print(f"Session:     {session_id}")
    print(f"Participant: {session.participant_id}")
    print(f"Trials:      {len(trials)}")

    # Compute metrics
    trial_dicts = [
        {
            "trial_type": t.trial_type,
            "response_correct": t.response_correct,
            "saccade_latency_ms": t.saccade_latency_ms,
        }
        for t in trials
    ]
    metrics = compute_session_metrics(trial_dicts)
    print_session_summary(metrics)

    # Find calibration for this participant
    calibration = None
    for s in repo.list_sessions():
        if s.participant_id == session.participant_id:
            cal = repo.get_calibration(s.session_id)
            if cal is not None and cal.accepted:
                calibration = cal
                break

    cal_matrix = np.array(calibration.transform_matrix) if calibration else None

    # Generate report
    output_path = generate_session_report(
        session_id=session_id,
        repo=repo,
        calibration_matrix=cal_matrix,
    )
    print(f"\nReport saved: {output_path}")


def main() -> None:
    """Parse arguments and run analysis."""
    parser = argparse.ArgumentParser(description="OcuTrace Offline Analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session", help="Session ID to analyze")
    group.add_argument(
        "--latest", action="store_true", help="Analyze most recent session"
    )
    group.add_argument(
        "--list", action="store_true", help="List all sessions"
    )
    args = parser.parse_args()

    config = load_config()
    storage_cfg = config["storage"]
    repo = SQLiteRepository(storage_cfg["sqlite"]["database_path"])
    repo.initialize()

    if args.list:
        list_sessions(repo)
    elif args.latest:
        sessions = repo.list_sessions()
        experiment_sessions = [
            s for s in sessions
            if s.notes == "experiment" and repo.get_trials(s.session_id)
        ]
        if not experiment_sessions:
            print("No experiment sessions found.")
            repo.close()
            sys.exit(1)
        analyze_session(experiment_sessions[0].session_id, repo)
    else:
        analyze_session(args.session, repo)

    repo.close()


if __name__ == "__main__":
    main()
