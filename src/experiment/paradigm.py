"""Trial sequence generator for the gap antisaccade paradigm.

Generates a balanced, pseudo-randomized sequence of antisaccade and
prosaccade trials with left/right stimulus side balancing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

VALID_TRIAL_TYPES = ("antisaccade", "prosaccade")
VALID_STIMULUS_SIDES = ("left", "right")


@dataclass(frozen=True)
class TrialSpec:
    """Specification for a single trial in the paradigm.

    Parameters
    ----------
    trial_number : int
        1-based trial index.
    trial_type : str
        'antisaccade' or 'prosaccade'.
    stimulus_side : str
        'left' or 'right'.
    """

    trial_number: int
    trial_type: str
    stimulus_side: str

    def __post_init__(self) -> None:
        if self.trial_type not in VALID_TRIAL_TYPES:
            raise ValueError(
                f"trial_type must be one of {VALID_TRIAL_TYPES}, "
                f"got '{self.trial_type}'"
            )
        if self.stimulus_side not in VALID_STIMULUS_SIDES:
            raise ValueError(
                f"stimulus_side must be one of {VALID_STIMULUS_SIDES}, "
                f"got '{self.stimulus_side}'"
            )


def generate_trial_sequence(
    n_antisaccade: int = 40,
    n_prosaccade: int = 20,
    seed: int | None = None,
    max_consecutive: int = 4,
) -> list[TrialSpec]:
    """Generate a balanced, pseudo-randomized trial sequence.

    Parameters
    ----------
    n_antisaccade : int
        Number of antisaccade trials (must be even for left/right balancing).
    n_prosaccade : int
        Number of prosaccade trials (must be even for left/right balancing).
    seed : int or None
        Random seed for reproducibility. None for non-deterministic.
    max_consecutive : int
        Maximum number of consecutive trials of the same type allowed.

    Returns
    -------
    list[TrialSpec]
        Ordered list of trial specs with sequential 1-based trial numbers.

    Raises
    ------
    ValueError
        If trial counts are not even (cannot balance left/right).
    """
    if n_antisaccade % 2 != 0:
        raise ValueError(
            f"n_antisaccade must be even for left/right balancing, "
            f"got {n_antisaccade}"
        )
    if n_prosaccade % 2 != 0:
        raise ValueError(
            f"n_prosaccade must be even for left/right balancing, "
            f"got {n_prosaccade}"
        )

    rng = np.random.default_rng(seed)

    # Build balanced trial type + side lists
    trial_types: list[str] = (
        ["antisaccade"] * n_antisaccade + ["prosaccade"] * n_prosaccade
    )
    sides: list[str] = (
        ["left"] * (n_antisaccade // 2)
        + ["right"] * (n_antisaccade // 2)
        + ["left"] * (n_prosaccade // 2)
        + ["right"] * (n_prosaccade // 2)
    )

    # Pair them and shuffle together
    paired = list(zip(trial_types, sides))
    rng.shuffle(paired)

    # Enforce max_consecutive constraint by swapping violations
    _enforce_max_consecutive(paired, max_consecutive, rng)

    return [
        TrialSpec(
            trial_number=i + 1,
            trial_type=trial_type,
            stimulus_side=side,
        )
        for i, (trial_type, side) in enumerate(paired)
    ]


def _enforce_max_consecutive(
    paired: list[tuple[str, str]],
    max_consecutive: int,
    rng: np.random.Generator,
) -> None:
    """Rearrange paired list in-place so no trial type repeats too many times.

    Uses iterative swapping: when a violation is found, swap with a random
    later position that has a different trial type. Repeats until resolved.

    Parameters
    ----------
    paired : list[tuple[str, str]]
        Mutable list of (trial_type, stimulus_side) tuples.
    max_consecutive : int
        Maximum allowed consecutive same-type trials.
    rng : numpy.random.Generator
        Random number generator for swap target selection.
    """
    max_iterations = len(paired) * 100  # safety limit
    for _ in range(max_iterations):
        violation_idx = _find_violation(paired, max_consecutive)
        if violation_idx is None:
            return
        # Find candidates to swap with (different type, after the run)
        current_type = paired[violation_idx][0]
        candidates = [
            j
            for j in range(violation_idx + 1, len(paired))
            if paired[j][0] != current_type
        ]
        if not candidates:
            # Try before the run
            candidates = [
                j
                for j in range(0, violation_idx)
                if paired[j][0] != current_type
            ]
        if candidates:
            swap_idx = int(rng.choice(candidates))
            paired[violation_idx], paired[swap_idx] = (
                paired[swap_idx],
                paired[violation_idx],
            )
        else:
            logger.warning(
                "Could not find swap candidate to fix consecutive "
                "constraint at index %d",
                violation_idx,
            )


def _find_violation(
    paired: list[tuple[str, str]], max_consecutive: int
) -> int | None:
    """Find the index of the first element that violates the consecutive constraint.

    Returns
    -------
    int or None
        Index of the violating element, or None if no violation.
    """
    run_length = 1
    for i in range(1, len(paired)):
        if paired[i][0] == paired[i - 1][0]:
            run_length += 1
            if run_length > max_consecutive:
                return i
        else:
            run_length = 1
    return None
