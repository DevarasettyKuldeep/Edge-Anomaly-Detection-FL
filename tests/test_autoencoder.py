"""
Unit Tests for Autoencoder Model and Federated Learning Parameter Serialization
"""

import sys
import os
import pytest
import numpy as np

# Add modules to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "flwr")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_pipeline")))


def test_imports():
    from server import AnomalyAutoencoder, get_model
    from train_autoencoder import load_sensor_data
    model = get_model(input_dim=10, latent_dim=8)
    assert model.input_dim == 10
    assert model.latent_dim == 8


def test_forward_pass_mock():
    """Verify layer output dimensions and parameter counts without heavy framework."""
    from server import AnomalyAutoencoder
    import torch
    model = AnomalyAutoencoder(input_dim=10, latent_dim=8)
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 10)
    latent = model.encoder(x)
    assert latent.shape == (4, 8)


def test_parameter_serialization():
    from server import get_model, get_model_params, set_model_params
    import torch
    model1 = get_model(input_dim=10, latent_dim=8)
    params = get_model_params(model1)

    assert len(params) == 12  # 6 weights + 6 biases
    # Check total parameter count
    total_weights = sum(p.size for p in params)
    assert total_weights == 2034

    model2 = get_model(input_dim=10, latent_dim=8)
    set_model_params(model2, params)

    params2 = get_model_params(model2)
    for p1, p2 in zip(params, params2):
        np.testing.assert_allclose(p1, p2)
