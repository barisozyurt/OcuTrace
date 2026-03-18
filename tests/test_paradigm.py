"""Tests for trial sequence generator."""
from collections import Counter

import pytest

from src.experiment.paradigm import TrialSpec, generate_trial_sequence


class TestTrialSpec:
    def test_trial_spec_creation(self):
        spec = TrialSpec(
            trial_number=1,
            trial_type="antisaccade",
            stimulus_side="left",
        )
        assert spec.trial_number == 1
        assert spec.trial_type == "antisaccade"
        assert spec.stimulus_side == "left"

    def test_trial_spec_frozen(self):
        spec = TrialSpec(trial_number=1, trial_type="prosaccade", stimulus_side="right")
        with pytest.raises(AttributeError):
            spec.trial_number = 2

    def test_trial_spec_invalid_type(self):
        with pytest.raises(ValueError, match="trial_type"):
            TrialSpec(trial_number=1, trial_type="invalid", stimulus_side="left")

    def test_trial_spec_invalid_side(self):
        with pytest.raises(ValueError, match="stimulus_side"):
            TrialSpec(trial_number=1, trial_type="antisaccade", stimulus_side="up")


class TestGenerateTrialSequence:
    def test_correct_total_count(self):
        trials = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=42)
        assert len(trials) == 60

    def test_correct_type_counts(self):
        trials = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=42)
        type_counts = Counter(t.trial_type for t in trials)
        assert type_counts["antisaccade"] == 40
        assert type_counts["prosaccade"] == 20

    def test_balanced_sides(self):
        trials = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=42)
        for trial_type in ("antisaccade", "prosaccade"):
            subset = [t for t in trials if t.trial_type == trial_type]
            side_counts = Counter(t.stimulus_side for t in subset)
            assert side_counts["left"] == side_counts["right"]

    def test_trial_numbers_sequential(self):
        trials = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=42)
        numbers = [t.trial_number for t in trials]
        assert numbers == list(range(1, 61))

    def test_randomized_order(self):
        """Sequence should be interleaved, not all antisaccade then all prosaccade."""
        trials = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=42)
        types = [t.trial_type for t in trials]
        # A sorted (non-randomized) sequence would have all antisaccade first
        sorted_types = sorted(types)
        assert types != sorted_types

    def test_seed_reproducibility(self):
        trials_a = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=123)
        trials_b = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=123)
        assert trials_a == trials_b

    def test_different_seeds_different_order(self):
        trials_a = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=1)
        trials_b = generate_trial_sequence(n_antisaccade=40, n_prosaccade=20, seed=2)
        types_a = [t.trial_type for t in trials_a]
        types_b = [t.trial_type for t in trials_b]
        assert types_a != types_b

    def test_no_more_than_4_consecutive_same_type(self):
        trials = generate_trial_sequence(
            n_antisaccade=40, n_prosaccade=20, seed=42, max_consecutive=4
        )
        types = [t.trial_type for t in trials]
        run_length = 1
        for i in range(1, len(types)):
            if types[i] == types[i - 1]:
                run_length += 1
                assert run_length <= 4, (
                    f"Found {run_length} consecutive '{types[i]}' at index {i}"
                )
            else:
                run_length = 1

    def test_max_consecutive_custom(self):
        """Custom max_consecutive value should be respected."""
        trials = generate_trial_sequence(
            n_antisaccade=40, n_prosaccade=20, seed=42, max_consecutive=3
        )
        types = [t.trial_type for t in trials]
        run_length = 1
        for i in range(1, len(types)):
            if types[i] == types[i - 1]:
                run_length += 1
                assert run_length <= 3
            else:
                run_length = 1

    def test_odd_count_raises(self):
        """Odd counts should raise ValueError since left/right can't balance."""
        with pytest.raises(ValueError, match="even"):
            generate_trial_sequence(n_antisaccade=3, n_prosaccade=2, seed=42)

    def test_no_seed_runs(self):
        """Should work without a seed (non-deterministic)."""
        trials = generate_trial_sequence(n_antisaccade=4, n_prosaccade=2)
        assert len(trials) == 6
