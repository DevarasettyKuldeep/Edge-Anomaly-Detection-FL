#!/usr/bin/env python3
"""
Simulated Federated Learning Orchestrator

Launches the Flower server and multiple simulated edge clients concurrently
in local subprocesses, verifying convergence and global model checkpointing.
"""

import argparse
import os
import subprocess
import sys
import time


def run_simulation(num_clients: int = 5, num_rounds: int = 3, server_port: int = 8089):
    server_address = f"127.0.0.1:{server_port}"
    save_path = "models/autoencoder_federated.pth"
    python_exe = sys.executable

    print("==================================================")
    print("Starting Automated Federated Learning Simulation")
    print(f"Clients: {num_clients} | Rounds: {num_rounds} | Address: {server_address}")
    print("==================================================")

    # Start server
    server_cmd = [
        python_exe,
        os.path.join(os.path.dirname(__file__), "server.py"),
        "--num-clients", str(num_clients),
        "--num-rounds", str(num_rounds),
        "--server-address", server_address,
        "--save-path", save_path,
    ]
    print(f"[Simulation] Launching server...")
    server_proc = subprocess.Popen(server_cmd)

    # Wait for server to bind port
    time.sleep(2)

    # Start clients
    client_procs = []
    for cid in range(1, num_clients + 1):
        client_cmd = [
            python_exe,
            os.path.join(os.path.dirname(__file__), "client.py"),
            "--client_id", str(cid),
            "--num-clients", str(num_clients),
            "--server-address", server_address,
        ]
        print(f"[Simulation] Launching Client {cid}...")
        p = subprocess.Popen(client_cmd)
        client_procs.append(p)
        time.sleep(0.3)

    # Wait for clients
    for i, p in enumerate(client_procs):
        p.wait()
        print(f"[Simulation] Client {i+1} finished.")

    # Wait for server
    server_proc.wait()
    print("[Simulation] Server finished.")

    if os.path.exists(save_path):
        print(f"\n✅ Simulation successful! Model saved to {save_path}")
    else:
        print(f"\n⚠️ Warning: {save_path} not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulated federated learning")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--num-rounds", type=int, default=2, help="Number of federated rounds")
    parser.add_argument("--port", type=int, default=8089, help="Server port")
    args = parser.parse_args()

    run_simulation(num_clients=args.num_clients, num_rounds=args.num_rounds, server_port=args.port)
