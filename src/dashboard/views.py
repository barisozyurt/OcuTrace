"""Route definitions for the OcuTrace Flask dashboard."""
from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, abort, render_template

from src.analysis.metrics import compute_session_metrics
from src.config import load_config
from src.storage.sqlite_repo import SQLiteRepository
from src.visualization.reports import (
    plot_error_rates,
    plot_latency_by_trial,
    plot_latency_distribution,
)


def register_routes(app: Flask) -> None:
    """Register all dashboard routes on the Flask app.

    Parameters
    ----------
    app : Flask
        The Flask application to register routes on.
    """
    config = load_config()
    storage_cfg = config["storage"]

    def get_repo() -> SQLiteRepository:
        """Create and initialize a new repository connection."""
        repo = SQLiteRepository(storage_cfg["sqlite"]["database_path"])
        repo.initialize()
        return repo

    @app.route("/")
    def index() -> str:
        """List all sessions with summary metrics."""
        repo = get_repo()
        sessions = repo.list_sessions()
        session_data: list[dict[str, Any]] = []
        for s in sessions:
            trials = repo.get_trials(s.session_id)
            if not trials:
                continue
            trial_dicts = [
                {
                    "trial_type": t.trial_type,
                    "response_correct": t.response_correct,
                    "saccade_latency_ms": t.saccade_latency_ms,
                }
                for t in trials
            ]
            metrics = compute_session_metrics(trial_dicts)
            dt = datetime.fromtimestamp(
                s.created_at / 1000.0, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
            session_data.append({
                "session": s,
                "n_trials": len(trials),
                "metrics": metrics,
                "datetime": dt,
            })
        repo.close()
        return render_template("index.html", sessions=session_data)

    @app.route("/session/<session_id>")
    def session_detail(session_id: str) -> str:
        """Show detailed view of a single session."""
        repo = get_repo()
        session = repo.get_session(session_id)
        if not session:
            repo.close()
            abort(404)
        trials = repo.get_trials(session_id)
        trial_dicts = [
            {
                "trial_number": t.trial_number,
                "trial_type": t.trial_type,
                "stimulus_side": t.stimulus_side,
                "response_correct": t.response_correct,
                "saccade_latency_ms": t.saccade_latency_ms,
                "saccade_direction": t.saccade_direction,
            }
            for t in trials
        ]
        metrics = compute_session_metrics(trial_dicts)

        # Generate plot images as base64
        plots: dict[str, str] = {}

        fig = plot_latency_by_trial(trial_dicts)
        if fig:
            plots["latency_trial"] = _fig_to_base64(fig)
            plt.close(fig)

        fig = plot_latency_distribution(trial_dicts)
        if fig:
            plots["latency_dist"] = _fig_to_base64(fig)
            plt.close(fig)

        fig = plot_error_rates(metrics)
        if fig:
            plots["error_rates"] = _fig_to_base64(fig)
            plt.close(fig)

        dt = datetime.fromtimestamp(
            session.created_at / 1000.0, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        repo.close()
        return render_template(
            "session.html",
            session=session,
            trials=trials,
            metrics=metrics,
            plots=plots,
            datetime=dt,
        )


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string.

    Parameters
    ----------
    fig : plt.Figure
        The matplotlib figure to encode.

    Returns
    -------
    str
        Base64-encoded PNG data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
