"""Tests for configuration loading."""
import os
import tempfile

import pytest

from src.config import load_config, get_config, reset_config


class TestLoadConfig:
    def test_load_default_config(self):
        config = load_config()
        assert config["camera"]["device_index"] == 0
        assert config["tracking"]["model_path"] == "models/face_landmarker.task"
        assert config["tracking"]["left_iris_index"] == 468
        assert config["tracking"]["right_iris_index"] == 473

    def test_load_custom_config(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("camera:\n  device_index: 1\n  target_fps: 60\n")
            f.flush()
            config = load_config(f.name)
            assert config["camera"]["device_index"] == 1
            assert config["camera"]["target_fps"] == 60
        os.unlink(f.name)

    def test_get_config_singleton(self):
        reset_config()
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_nested_access(self):
        config = load_config()
        assert config["saccade_detection"]["smoothing_window"] == 5
        assert config["paradigm"]["n_antisaccade_trials"] == 40
        assert config["glasses_detection"]["jitter_threshold_px"] == 2.0
