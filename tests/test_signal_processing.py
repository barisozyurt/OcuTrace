"""Tests for signal processing utilities (smoothing and velocity computation)."""

import numpy as np
import pytest

from src.analysis.signal_processing import compute_velocity, smooth_positions


# ── smooth_positions tests ──────────────────────────────────────────────


class TestSmoothPositions:
    """Tests for Savitzky-Golay smoothing wrapper."""

    def test_smooth_removes_noise(self) -> None:
        """Smoothed signal should have lower standard deviation of residuals."""
        rng = np.random.default_rng(42)
        clean = np.linspace(0, 10, 100)
        noisy = clean + rng.normal(scale=0.5, size=100)

        smoothed = smooth_positions(noisy)

        residuals_before = np.std(noisy - clean)
        residuals_after = np.std(smoothed - clean)
        assert residuals_after < residuals_before

    def test_smooth_preserves_trend(self) -> None:
        """A pure linear trend should be preserved within atol=0.5."""
        linear = np.linspace(0, 20, 50)
        smoothed = smooth_positions(linear)
        np.testing.assert_allclose(smoothed, linear, atol=0.5)

    def test_smooth_output_same_length(self) -> None:
        """Output array must have the same length as input."""
        signal = np.random.default_rng(0).normal(size=73)
        smoothed = smooth_positions(signal)
        assert len(smoothed) == len(signal)

    def test_smooth_short_signal_fallback(self) -> None:
        """A 3-element signal with window=5 should still work (fallback)."""
        short = np.array([1.0, 2.0, 3.0])
        smoothed = smooth_positions(short, window=5, polyorder=2)
        assert len(smoothed) == 3


# ── compute_velocity tests ──────────────────────────────────────────────


class TestComputeVelocity:
    """Tests for velocity computation from degree-space positions."""

    def test_stationary_zero_velocity(self) -> None:
        """Constant position should yield ~0 velocity everywhere."""
        positions = np.full(50, 5.0)
        timestamps = np.arange(50) * 33.33  # ~30 fps in ms
        velocity = compute_velocity(positions, timestamps)
        np.testing.assert_allclose(velocity, 0.0, atol=1e-6)

    def test_constant_motion_velocity(self) -> None:
        """Linear motion at 10 deg/s should yield velocity ~10 deg/s."""
        # 10 deg/s for 1 second at 1 ms resolution
        timestamps = np.arange(0, 1000, 10, dtype=float)  # 100 frames, 10 ms apart
        # 10 deg/s * (t in seconds) = 10 * t/1000 degrees
        positions = 10.0 * (timestamps / 1000.0)

        velocity = compute_velocity(positions, timestamps)
        # Skip first element (always 0)
        np.testing.assert_allclose(velocity[1:], 10.0, atol=0.1)

    def test_velocity_units_are_deg_per_sec(self) -> None:
        """1 degree per frame at 30 fps should give 30 deg/s."""
        n = 20
        timestamps = np.arange(n) * (1000.0 / 30.0)  # 30 fps, ms
        positions = np.arange(n, dtype=float)  # 1 deg per frame

        velocity = compute_velocity(positions, timestamps)
        np.testing.assert_allclose(velocity[1:], 30.0, atol=0.5)

    def test_velocity_first_element_is_zero(self) -> None:
        """First velocity element must be exactly 0.0."""
        positions = np.array([0.0, 1.0, 3.0, 6.0])
        timestamps = np.array([0.0, 10.0, 20.0, 30.0])
        velocity = compute_velocity(positions, timestamps)
        assert velocity[0] == 0.0
