"""Matplotlib-based clinical report generation for saccade analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.analysis.metrics import SessionMetrics, compute_session_metrics
from src.storage.models import GazeData, Trial
from src.tracking.calibration import apply_transform


def plot_latency_by_trial(
    trials: list[dict],
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """Plot saccade latency for each trial.

    Parameters
    ----------
    trials : list[dict]
        Trial result dicts with keys ``trial_number``, ``trial_type``,
        ``saccade_latency_ms``, and ``response_correct``.
    ax : plt.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.

    Returns
    -------
    plt.Figure or None
        The created figure if *ax* was ``None``, else ``None``.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True

    color_map = {"antisaccade": "tab:blue", "prosaccade": "tab:orange"}

    for t in trials:
        latency = t.get("saccade_latency_ms")
        if latency is None:
            continue
        color = color_map.get(t["trial_type"], "grey")
        marker = "o" if t.get("response_correct") else "x"
        ax.plot(
            t["trial_number"], latency, marker=marker, color=color,
            markersize=6, linestyle="none",
        )

    ax.axhline(150, linestyle="--", color="grey", linewidth=0.8, label="150 ms")
    ax.axhline(400, linestyle="--", color="grey", linewidth=0.8, label="400 ms")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Saccade Latency (ms)")
    ax.set_title("Saccade Latency by Trial")

    # Build legend
    anti_patch = mpatches.Patch(color="tab:blue", label="Antisaccade")
    pro_patch = mpatches.Patch(color="tab:orange", label="Prosaccade")
    ax.legend(handles=[anti_patch, pro_patch], loc="upper right", fontsize="small")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_latency_distribution(
    trials: list[dict],
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """Plot overlaid histograms of antisaccade and prosaccade latencies.

    Parameters
    ----------
    trials : list[dict]
        Trial result dicts with ``trial_type`` and ``saccade_latency_ms``.
    ax : plt.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.

    Returns
    -------
    plt.Figure or None
        The created figure if *ax* was ``None``, else ``None``.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True

    bins = np.arange(0, 1050, 50)

    anti_lat = [
        t["saccade_latency_ms"] for t in trials
        if t["trial_type"] == "antisaccade" and t["saccade_latency_ms"] is not None
    ]
    pro_lat = [
        t["saccade_latency_ms"] for t in trials
        if t["trial_type"] == "prosaccade" and t["saccade_latency_ms"] is not None
    ]

    if anti_lat:
        ax.hist(anti_lat, bins=bins, alpha=0.6, color="tab:blue", label="Antisaccade")
        ax.axvline(np.mean(anti_lat), linestyle="--", color="tab:blue", linewidth=1)
    if pro_lat:
        ax.hist(pro_lat, bins=bins, alpha=0.6, color="tab:orange", label="Prosaccade")
        ax.axvline(np.mean(pro_lat), linestyle="--", color="tab:orange", linewidth=1)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution")
    ax.legend(fontsize="small")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_error_rates(
    metrics: SessionMetrics,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """Plot antisaccade and prosaccade error rates as a bar chart.

    Parameters
    ----------
    metrics : SessionMetrics
        Aggregated session metrics.
    ax : plt.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.

    Returns
    -------
    plt.Figure or None
        The created figure if *ax* was ``None``, else ``None``.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
        created_fig = True

    labels = ["Antisaccade", "Prosaccade"]
    rates = [
        metrics.antisaccade_error_rate * 100,
        metrics.prosaccade_error_rate * 100,
    ]
    colors = ["tab:blue", "tab:orange"]

    bars = ax.bar(labels, rates, color=colors, width=0.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Rates")

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{rate:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_gaze_trace(
    gaze_data: list[GazeData],
    trial: Trial,
    calibration_matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Figure]:
    """Plot horizontal gaze position over time for a single trial.

    Parameters
    ----------
    gaze_data : list[GazeData]
        Gaze samples for this trial.
    trial : Trial
        Trial metadata (for stimulus onset and type).
    calibration_matrix : np.ndarray
        3x3 affine transform (pixel to degree).
    ax : plt.Axes, optional
        Axes to draw on. If ``None``, a new figure is created.

    Returns
    -------
    plt.Figure or None
        The created figure if *ax* was ``None``, else ``None``.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True

    if not gaze_data:
        ax.set_title(f"Trial {trial.trial_number} ({trial.trial_type}) — no data")
        if created_fig:
            fig.tight_layout()
            return fig
        return None

    trial_start_ms = gaze_data[0].timestamp_ms
    times = [g.timestamp_ms - trial_start_ms for g in gaze_data]
    # Average left and right iris X for gaze position, transform to degrees
    deg_x = []
    for g in gaze_data:
        avg_x = (g.left_iris_x + g.right_iris_x) / 2
        avg_y = (g.left_iris_y + g.right_iris_y) / 2
        dx, _ = apply_transform(calibration_matrix, avg_x, avg_y)
        deg_x.append(dx)

    ax.plot(times, deg_x, color="black", linewidth=1)

    # Stimulus onset marker
    stim_rel = trial.stimulus_onset_ms - trial_start_ms
    ax.axvline(stim_rel, linestyle="--", color="red", linewidth=1, label="Stimulus onset")

    # Eccentricity lines at +/-10 deg
    stimulus_eccentricity = 10.0
    ax.axhline(stimulus_eccentricity, linestyle="--", color="grey", linewidth=0.7)
    ax.axhline(-stimulus_eccentricity, linestyle="--", color="grey", linewidth=0.7)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Gaze Position (deg)")
    ax.set_title(f"Trial {trial.trial_number} ({trial.trial_type})")
    ax.legend(fontsize="small")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def generate_session_report(
    session_id: str,
    repo: object,
    calibration_matrix: np.ndarray,
    output_dir: str = "",
    participant_name: str = "",
) -> Path:
    """Generate a 2x2 clinical report figure and save to disk.

    Parameters
    ----------
    session_id : str
        Session UUID.
    repo : Repository
        Data repository with ``get_trials`` and ``get_gaze_data`` methods.
    calibration_matrix : np.ndarray
        3x3 affine transform matrix.
    output_dir : str
        Directory to save reports into.
    participant_name : str
        Patient/participant name to display on the report.

    Returns
    -------
    Path
        Path to the saved combined report PNG.
    """
    if not output_dir:
        from src.paths import get_reports_dir
        out = get_reports_dir()
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trials = repo.get_trials(session_id)
    trial_dicts = [
        {
            "trial_number": t.trial_number,
            "trial_type": t.trial_type,
            "response_correct": t.response_correct,
            "saccade_latency_ms": t.saccade_latency_ms,
        }
        for t in trials
    ]
    metrics = compute_session_metrics(trial_dicts)

    # Find first trial with gaze data for the gaze trace plot
    first_trial = trials[0] if trials else None
    gaze_data: list[GazeData] = []
    if first_trial is not None:
        gaze_data = repo.get_gaze_data(session_id, first_trial.trial_number)

    # Combined 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    title = f"Patient: {participant_name}" if participant_name else f"Session: {session_id}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Data quality warning
    detected_trials = sum(
        1 for t in trial_dicts if t["saccade_latency_ms"] is not None
    )
    detection_rate = detected_trials / len(trial_dicts) if trial_dicts else 0
    if detection_rate < 0.5:
        fig.text(
            0.5, 0.92,
            f"WARNING: Low detection rate ({detected_trials}/{len(trial_dicts)} "
            f"trials = {detection_rate:.0%}). Results may not be clinically reliable.",
            ha="center", fontsize=10, color="red", style="italic",
        )

    plot_latency_by_trial(trial_dicts, ax=axes[0, 0])
    plot_latency_distribution(trial_dicts, ax=axes[0, 1])
    plot_error_rates(metrics, ax=axes[1, 0])
    if first_trial is not None:
        plot_gaze_trace(gaze_data, first_trial, calibration_matrix, ax=axes[1, 1])
    else:
        axes[1, 1].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = out / f"{session_id}_report.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    # Individual plots
    for name, plot_fn, args in [
        ("latency_by_trial", plot_latency_by_trial, (trial_dicts,)),
        ("latency_distribution", plot_latency_distribution, (trial_dicts,)),
    ]:
        individual = plot_fn(*args)
        if individual is not None:
            individual.savefig(out / f"{session_id}_{name}.png", dpi=150)
            plt.close(individual)

    error_fig = plot_error_rates(metrics)
    if error_fig is not None:
        error_fig.savefig(out / f"{session_id}_error_rates.png", dpi=150)
        plt.close(error_fig)

    if first_trial is not None and gaze_data:
        gaze_fig = plot_gaze_trace(gaze_data, first_trial, calibration_matrix)
        if gaze_fig is not None:
            gaze_fig.savefig(out / f"{session_id}_gaze_trace.png", dpi=150)
            plt.close(gaze_fig)

    return combined_path
