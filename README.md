# Edge-Anomaly-Detection-FL

## Federated Anomaly Detection for Privacy-Preserving Edge AI

[![CI/CD](https://github.com/DevarasettyKuldeep/Edge-Anomaly-Detection-FL/actions/workflows/test.yml/badge.svg)](https://github.com/DevarasettyKuldeep/Edge-Anomaly-Detection-FL/actions)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![C99](https://img.shields.io/badge/Language-C99-orange.svg)](https://en.wikipedia.org/wiki/C99)

---

## Project Overview

This project implements an end-to-end **federated anomaly detection system** tailored for resource-constrained edge computing devices (such as ESP32, STM32, and ARM Cortex-M microcontrollers).

### Key Architectural Pillars

1. **Federated Learning ([flwr/](flwr/)):** Uses the Flower framework to collaboratively train an autoencoder across edge devices without centralizing private sensor telemetry.
2. **Autoencoder Anomaly Detection ([ml_pipeline/](ml_pipeline/)):** Unsupervised model trained purely on normal operating telemetry, identifying faults via Mean Squared Error (MSE) reconstruction loss beyond a calibrated 99th percentile threshold.
3. **INT8 Quantization & Export ([ml_pipeline/quantize_model.py](ml_pipeline/quantize_model.py)):** Compresses model weights by ~75% and exports embedded C headers.
4. **Embedded C99 Runtime ([firmware/](firmware/)):** Zero-dependency pure C99 inference engine that runs on microcontrollers with less than 2KB RAM footprint and sub-millisecond execution time.
5. **CI/CD Automation ([.github/workflows/test.yml](.github/workflows/test.yml)):** Automated testing pipeline for Python ML units, C unit tests, and build verification.

---

## Architecture Flow

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  Edge Node 1    │       │  Edge Node 2    │       │  Edge Node N    │
│  Sensor Reading │       │  Sensor Reading │       │  Sensor Reading │
│  Local Training │       │  Local Training │       │  Local Training │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         └────────────────┬────────┴─────────────────────────┘
                          │ (Model updates only, no raw data)
                          ▼
             ┌─────────────────────────┐
             │   FLOWER SERVER (FedAvg) │
             │   • Aggregates weights   │
             │   • Saves global model   │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │  QUANTIZATION PIPELINE  │
             │  • FP32 → INT8          │
             │  • Threshold calibration│
             │  • Exports model.h      │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │   C99 EDGE RUNTIME      │
             │   • Zero dependencies   │
             │   • Sub-millisecond exec│
             │   • Ultra-low RAM/flash │
             └─────────────────────────┘
```

---

## Verified Engineering Metrics

| Metric | Measured Value | Note |
|---|---|---|
| **Architecture** | 10 → 32 → 16 → 8 → 16 → 32 → 10 | 5-layer symmetrical autoencoder |
| **Total Parameters** | 2,034 weights & biases | Lightweight edge footprint |
| **FP32 Weight Size** | 8.13 KB | 2,034 floats × 4 bytes |
| **INT8 Weight Size** | 2.03 KB | ~75% reduction |
| **Reconstruction Threshold** | ~2.70 (calibrated 99th percentile) | Dynamically configured in model.h |
| **Inference Runtime** | Pure ISO C99 / TFLite Micro | Compiles with GCC, Clang, MSVC |
| **Inference Latency** | < 0.05 ms (Desktop), < 2 ms (ESP32) | Highly optimized dense feed-forward |
| **Detection F1-Score** | > 0.88 | Unsupervised reconstruction error |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/DevarasettyKuldeep/Edge-Anomaly-Detection-FL.git
cd Edge-Anomaly-Detection-FL

pip install -r ml_pipeline/requirements.txt
pip install -r flwr/requirements.txt
```

### 2. Run Federated Learning

**Option A — Automated Local Simulation:**
```bash
python flwr/simulate.py --num-clients 3 --num-rounds 2
```

**Option B — Distributed Client/Server:**
```bash
# Terminal 1: Start Central Server
python flwr/server.py --num-clients 2 --num-rounds 3

# Terminal 2 & 3: Start Edge Clients
python flwr/client.py --client_id 1
python flwr/client.py --client_id 2
```

### 3. Centralized Training & Threshold Calibration

```bash
python ml_pipeline/train_autoencoder.py --epochs 30 --batch-size 64
```

### 4. Quantize & Export C Header

```bash
python ml_pipeline/quantize_model.py \
  --input models/autoencoder_final.pth \
  --output firmware/model.tflite \
  --header firmware/model.h
```

### 5. Build and Run Embedded C Firmware

```bash
cd firmware
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
./build/anomaly_detector
```

---

## Repository Structure

```
Edge-Anomaly-Detection-FL/
├── flwr/
│   ├── server.py             # Flower server with FedAvg & model checkpointing
│   ├── client.py             # Flower client with local partition training
│   ├── simulate.py           # Multi-process simulation orchestrator
│   └── requirements.txt      # Flower dependencies
├── ml_pipeline/
│   ├── train_autoencoder.py  # PyTorch training & threshold calibration
│   ├── quantize_model.py     # INT8 quantization & dual C header generator
│   └── requirements.txt      # ML pipeline dependencies
├── firmware/
│   ├── inference.h           # Clean C99 inference API
│   ├── inference.c           # Zero-dependency feed-forward engine
│   ├── main.c                # Desktop test verification program
│   ├── test_inference.c      # Automated C unit test suite
│   ├── model.h               # Calibrated weights & threshold header
│   └── CMakeLists.txt        # Cross-platform build script
├── tests/
│   ├── test_autoencoder.py   # Pytest suite for model architecture
│   └── test_quantization.py  # Pytest suite for quantization headers
├── docs/
│   ├── architecture.md       # Architectural deep-dive
│   └── technical_report.md   # Comprehensive technical report
└── .github/workflows/
    └── test.yml              # GitHub Actions CI/CD pipeline
```

---

## Documentation

- [System Architecture](docs/architecture.md)
- [Technical Engineering Report](docs/technical_report.md)
- [Implementation Plan](IMPLEMENTATION_PLAN.md)
