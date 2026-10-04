#!/usr/bin/env python3
"""
Federated Learning Client for Edge Anomaly Detection

Simulates an edge device (e.g. IoT node / ESP32 gateway). It:
1. Loads its local non-IID sensor telemetry partition
2. Trains the autoencoder locally using PyTorch
3. Sends model parameter updates to the central Flower server
4. Receives aggregated global weights
"""

import argparse
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

from server import AnomalyAutoencoder, get_model, get_model_params, set_model_params


def load_partition_data(
    client_id: int,
    num_clients: int,
    batch_size: int = 32,
    n_samples: int = 400
) -> Tuple[DataLoader, DataLoader]:
    """
    Generate client-specific non-IID sensor telemetry partition.
    Split 80/20 into train/validation loaders.
    """
    np.random.seed(42 + client_id * 17)
    n_features = 10

    # Base normal sensor readings with inter-feature correlation
    data = np.random.randn(n_samples, n_features).astype(np.float32)
    for i in range(n_features - 1):
        data[:, i + 1] += 0.3 * data[:, i]

    # Non-IID shift unique to this edge device's operating environment
    shift = np.random.randn(n_features).astype(np.float32) * 0.4
    client_factor = float(client_id) / max(num_clients, 1)
    data = data + (shift * client_factor)

    split = int(0.8 * n_samples)
    train_data = data[:split]
    val_data = data[split:]

    train_dataset = TensorDataset(torch.from_numpy(train_data))
    val_dataset = TensorDataset(torch.from_numpy(val_data))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class EdgeAnomalyClient(fl.client.NumPyClient):
    """Flower client that trains the autoencoder on local edge sensor data."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        learning_rate: float = 1e-3,
    ):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def get_parameters(self, config: Dict[str, str]) -> list:
        return get_model_params(self.model)

    def fit(self, parameters: list, config: Dict[str, str]) -> Tuple[list, int, Dict[str, float]]:
        set_model_params(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        epoch_loss = 0.0
        total_batches = 0

        for _ in range(self.local_epochs):
            for (batch_data,) in self.train_loader:
                batch_data = batch_data.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_data)
                loss = self.criterion(recon, batch_data)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1

        avg_loss = epoch_loss / max(total_batches, 1)
        num_examples = len(self.train_loader.dataset)
        return get_model_params(self.model), num_examples, {"loss": float(avg_loss)}

    def evaluate(self, parameters: list, config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        set_model_params(self.model, parameters)
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for (batch_data,) in self.val_loader:
                batch_data = batch_data.to(self.device)
                recon = self.model(batch_data)
                loss = self.criterion(recon, batch_data)
                total_loss += loss.item() * batch_data.size(0)
                total_samples += batch_data.size(0)

        eval_loss = total_loss / max(total_samples, 1)
        return float(eval_loss), total_samples, {"loss": float(eval_loss)}


def main():
    parser = argparse.ArgumentParser(description="Flower Client for Edge Anomaly Detection")
    parser.add_argument("--client_id", type=int, default=1, help="Client ID (e.g. 1 to N)")
    parser.add_argument("--num-clients", type=int, default=5, help="Total clients in federation")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080", help="Flower server gRPC address")
    parser.add_argument("--epochs", type=int, default=1, help="Local epochs per round")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Client {args.client_id}/{args.num_clients}] Starting on {device}...")

    model = get_model().to(device)
    train_loader, val_loader = load_partition_data(args.client_id, args.num_clients)
    print(f"[Client {args.client_id}] Loaded {len(train_loader.dataset)} train, {len(val_loader.dataset)} val samples.")

    client = EdgeAnomalyClient(
        client_id=args.client_id,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        local_epochs=args.epochs,
    )

    print(f"[Client {args.client_id}] Connecting to Flower server at {args.server_address}...")
    try:
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
    except AttributeError:
        fl.client.start_client(
            server_address=args.server_address,
            client=client.to_client(),
        )


if __name__ == "__main__":
    main()
