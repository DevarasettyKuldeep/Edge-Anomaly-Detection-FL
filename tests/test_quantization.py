"""
Unit Tests for Quantization & C Header Generation
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_pipeline")))


def test_model_header_integrity():
    """Verify that firmware/model.h contains valid C definitions and calibrated threshold."""
    header_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "firmware", "model.h"))
    assert os.path.exists(header_path), f"Header not found at {header_path}"

    with open(header_path, "r") as f:
        content = f.read()

    assert "#define ANOMALY_THRESHOLD" in content
    assert "#define ANOMALY_INPUT_DIM 10" in content
    assert "layer1_weights" in content
    assert "layer6_weights" in content
    assert "layer6_bias" in content


def test_threshold_file_validity():
    threshold_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "threshold.npy"))
    if os.path.exists(threshold_path):
        import numpy as np
        val = float(np.load(threshold_path))
        assert val > 0.0
        assert val < 10.0
