"""Clinical visualization and report generation."""
from src.visualization.reports import (
    plot_latency_by_trial,
    plot_latency_distribution,
    plot_error_rates,
    plot_gaze_trace,
    generate_session_report,
)

__all__ = [
    "plot_latency_by_trial",
    "plot_latency_distribution",
    "plot_error_rates",
    "plot_gaze_trace",
    "generate_session_report",
]
