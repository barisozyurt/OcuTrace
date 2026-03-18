"""Gap antisaccade experiment paradigm."""
from src.experiment.paradigm import TrialSpec, generate_trial_sequence
from src.experiment.stimulus import (
    StimulusConfig,
    TrialTimestamps,
    create_stimulus_config,
    create_fixation_dot,
    compute_trial_frame_counts,
    run_single_trial,
)
from src.experiment.session import GazeCollector, analyze_trial, print_session_summary

__all__ = [
    "TrialSpec",
    "generate_trial_sequence",
    "StimulusConfig",
    "TrialTimestamps",
    "create_stimulus_config",
    "compute_trial_frame_counts",
    "run_single_trial",
    "GazeCollector",
    "analyze_trial",
    "print_session_summary",
]
