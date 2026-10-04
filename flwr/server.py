#!/usr/bin/env python3
"""
Federated Learning Server for Edge Anomaly Detection

Implements Federated Averaging (FedAvg) using the Flower framework.
Simulates edge devices training an autoencoder locally on non-IID sensor data,
aggregating model updates centrally without collecting raw sensor telemetry.
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import Metrics, Parameters, Scalar


class AnomalyAutoencoder(nn.Module):
    """
    Unified 5-layer sparse autoencoder architecture.
    10 -> 32 -> 16 -> 8 (latent) -> 16 -> 32 -> 10
    """
    def __init__(self, input_dim: int = 10, latent_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def get_model(input_dim: int = 10, latent_dim: int = 8) -> AnomalyAutoencoder:
    """Initialize the autoencoder model."""
    return AnomalyAutoencoder(input_dim=input_dim, latent_dim=latent_dim)


def set_model_params(model: nn.Module, params: List[np.ndarray]) -> None:
    """Set model parameters from a list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_model_params(model: nn.Module) -> List[np.ndarray]:
    """Get model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def aggregate_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average aggregation of evaluation metrics from all clients."""
    if not metrics:
        return {}
    total_samples = sum(num_examples for num_examples, _ in metrics)
    if total_samples == 0:
        return {}
    weighted_loss = sum(num_examples * m.get("loss", 0.0) for num_examples, m in metrics)
    return {"avg_loss": float(weighted_loss / total_samples)}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that saves the aggregated global model after rounds."""

    def __init__(self, save_path: str = "models/autoencoder_federated.pth", **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.latest_parameters: Optional[Parameters] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            np_params = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model = get_model()
            set_model_params(model, np_params)

            save_dir = os.path.dirname(self.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": 10,
                "latent_dim": 8,
                "server_round": server_round,
            }, self.save_path)
            print(f"[Round {server_round}] Successfully saved global model checkpoint to: {self.save_path}")

        return aggregated_parameters, aggregated_metrics


def main(num_clients: int = 5, num_rounds: int = 3, server_address: str = "0.0.0.0:8080", save_path: str = "models/autoencoder_federated.pth"):
    """Start the Flower server with FedAvg strategy and automatic checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("Flower Federated Learning Server - Edge Anomaly")
    print("==================================================")
    print(f"Server address:       {server_address}")
    print(f"Target clients:       {num_clients}")
    print(f"Federated rounds:     {num_rounds}")
    print(f"Save checkpoint to:   {save_path}")
    print(f"Compute device:       {device}")
    print("Architecture:         10 -> 32 -> 16 -> 8 -> 16 -> 32 -> 10")
    print("==================================================\n")

    # Initial global model parameters
    init_model = get_model().to(device)
    initial_parameters = fl.common.ndarrays_to_parameters(get_model_params(init_model))

    strategy = SaveModelStrategy(
        save_path=save_path,
        min_fit_clients=min(num_clients, 2),
        min_evaluate_clients=min(num_clients, 2),
        min_available_clients=min(num_clients, 2),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Federated Learning Server")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of expected clients")
    parser.add_argument("--num-rounds", type=int, default=3, help="Number of federated training rounds")
    parser.add_argument("--server-address", type=str, default="0.0.0.0:8080", help="gRPC bind address")
    parser.add_argument("--save-path", type=str, default="models/autoencoder_federated.pth", help="Checkpoint save path")
    args = parser.parse_args()

    main(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        server_address=args.server_address,
        save_path=args.save_path
    )
