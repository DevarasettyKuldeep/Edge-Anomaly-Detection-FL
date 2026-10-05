#!/usr/bin/env python3
"""
Autoencoder Training & Threshold Calibration Script for Edge Anomaly Detection

Trains a 5-layer sparse autoencoder on normal sensor telemetry for unsupervised anomaly detection.
Supports initializing from a federated learning checkpoint (models/autoencoder_federated.pth).
Calculates and exports the 99th percentile reconstruction error threshold.

Usage:
    python train_autoencoder.py --epochs 50 --batch-size 64 --latent-dim 8
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AnomalyAutoencoder(nn.Module):
    """
    Unified 5-layer autoencoder architecture.
    10 -> 32 -> 16 -> 8 -> 16 -> 32 -> 10
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def load_sensor_data(n_samples: int = 2000, n_features: int = 10, batch_size: int = 64):
    """
    Generate synthetic sensor telemetry with realistic inter-sensor correlations.
    In production, this can ingest CSV/Parquet from industrial SCADA or NASA/UCI MSL datasets.
    """
    np.random.seed(42)
    # Generate baseline normal telemetry
    normal_data = np.random.randn(n_samples, n_features).astype(np.float32) * 0.8

    # Simulate cross-sensor coupling (e.g., temperature vs vibration vs pressure)
    for i in range(n_features - 1):
        normal_data[:, i + 1] += 0.35 * normal_data[:, i]

    # Normalize to zero mean, unit variance
    mean = np.mean(normal_data, axis=0)
    std = np.std(normal_data, axis=0) + 1e-6
    normal_data = (normal_data - mean) / std

    split = int(0.8 * n_samples)
    train_data = normal_data[:split]
    val_data = normal_data[split:]

    train_dataset = TensorDataset(torch.from_numpy(train_data))
    val_dataset = TensorDataset(torch.from_numpy(val_data))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, n_features, mean, std


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-3, save_dir="models"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_weights_path = os.path.join(save_dir, "autoencoder_best.pth")

    print(f"\n[Training] Starting for {epochs} epochs on device: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total trainable parameters: {total_params} (~{total_params * 4 / 1024:.2f} KB FP32)")

    for epoch in range(epochs):
        model.train()
        train_loss, train_samples = 0.0, 0

        for batch in train_loader:
            data = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_samples += data.size(0)

        avg_train_loss = train_loss / max(train_samples, 1)

        model.eval()
        val_loss, val_samples = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                data = batch[0].to(device)
                recon = model(data)
                loss = criterion(recon, data)
                val_loss += loss.item() * data.size(0)
                val_samples += data.size(0)

        avg_val_loss = val_loss / max(val_samples, 1)
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_weights_path)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
    print(f"[Training Complete] Best validation loss: {best_val_loss:.6f}")
    return model, best_val_loss


def compute_threshold(model, train_loader, device, percentile=99.0):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in train_loader:
            data = batch[0].to(device)
            recon = model(data)
            sample_errors = torch.mean((data - recon) ** 2, dim=1)
            errors.extend(sample_errors.cpu().numpy())

    errors = np.array(errors)
    threshold = float(np.percentile(errors, percentile))
    print(f"\n[Threshold Calibration]")
    print(f"  Training Error Mean: {np.mean(errors):.6f} (Std: {np.std(errors):.6f})")
    print(f"  Calibrated Threshold ({percentile}th percentile): {threshold:.6f}")
    return threshold, errors


def evaluate_model(model, threshold, device):
    """Evaluate on synthetic normal and anomalous distributions."""
    model.eval()
    np.random.seed(123)

    # Normal test data
    normal_test = np.random.randn(200, 10).astype(np.float32) * 0.8
    for i in range(9):
        normal_test[:, i + 1] += 0.35 * normal_test[:, i]
    normal_test = (normal_test - np.mean(normal_test, axis=0)) / (np.std(normal_test, axis=0) + 1e-6)

    # Anomalous test data (abnormal high variance & sensor fault shifts)
    anomalous_test = np.random.randn(100, 10).astype(np.float32) * 2.5 + 2.0

    normal_tensor = torch.from_numpy(normal_test).to(device)
    anomalous_tensor = torch.from_numpy(anomalous_test).to(device)

    with torch.no_grad():
        normal_recon = model(normal_tensor)
        normal_errors = torch.mean((normal_tensor - normal_recon) ** 2, dim=1).cpu().numpy()

        anom_recon = model(anomalous_tensor)
        anom_errors = torch.mean((anomalous_tensor - anom_recon) ** 2, dim=1).cpu().numpy()

    normal_flags = (normal_errors > threshold).astype(int)
    anom_flags = (anom_errors > threshold).astype(int)

    tp = int(np.sum(anom_flags))
    fn = int(len(anom_flags) - tp)
    fp = int(np.sum(normal_flags))
    tn = int(len(normal_flags) - fp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(f"\n[Evaluation Results]")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "normal_error_mean": float(np.mean(normal_errors)),
        "anom_error_mean": float(np.mean(anom_errors)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder for Edge Anomaly Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent bottleneck dimension")
    parser.add_argument("--federated-checkpoint", type=str, default="models/autoencoder_federated.pth",
                        help="Optional federated checkpoint to load from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)

    train_loader, val_loader, input_dim, mean, std = load_sensor_data(batch_size=args.batch_size)
    model = AnomalyAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim).to(device)

    # Optional transfer learning from federated model
    if os.path.exists(args.federated_checkpoint):
        print(f"[Federated Learning] Found federated weights at: {args.federated_checkpoint}")
        try:
            ckpt = torch.load(args.federated_checkpoint, map_location=device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print("  Successfully initialized model with federated weights!")
        except Exception as e:
            print(f"  Note: Could not load federated checkpoint ({e}), training from scratch.")

    model, best_val_loss = train_model(
        model, train_loader, val_loader, device, epochs=args.epochs, lr=args.lr, save_dir="models"
    )

    threshold, train_errors = compute_threshold(model, train_loader, device)
    metrics = evaluate_model(model, threshold, device)

    # Save artifacts
    np.save("models/threshold.npy", np.float32(threshold))
    with open("models/threshold.txt", "w") as f:
        f.write(f"{threshold:.6f}\n")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "threshold": threshold,
        "metrics": metrics,
    }, "models/autoencoder_final.pth")

    print("\n✅ Saved model artifacts:")
    print("   • models/autoencoder_final.pth")
    print("   • models/autoencoder_best.pth")
    print(f"   • models/threshold.npy (threshold = {threshold:.6f})")
    print("   • models/metrics.json")


if __name__ == "__main__":
    main()
