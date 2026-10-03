# Edge-Anomaly-Detection-FL: Federated Anomaly Detection for Privacy-Preserving Edge AI

## Project Overview
Portfolio-ready project combining **Federated Learning** with **Autoencoder-based Anomaly Detection** for resource-constrained edge devices. Demonstrates end-to-end system design: from model training on distributed edge devices to INT8-quantized C inference.

**Duration**: 12 weeks (one semester)  
**Complexity**: Advanced (final-year B.Tech level)  
**Key Technologies**: Python, PyTorch, Flower Framework, TensorFlow Lite Micro, C99, GitHub Actions

---

## Folder Structure

```
Edge-Anomaly-Detection-FL/
├── flwr/                    # Phase 1: Federated Learning framework
│   ├── server.py            # Central server script
│   ├── client/              # Client scripts (5+ simulated edge devices)
│   └── pyproject.toml       # Flower package configuration
├── ml_pipeline/             # Phases 2-3: Training & quantization
│   ├── train_autoencoder.py # Autoencoder training script
│   ├── quantize_model.py    # INT8 quantization script
│   └── requirements.txt     # Python dependencies
├── firmware/                # Phase 3: C deployment
│   ├── model.h              # Generated TFLite Micro model header
│   ├── inference.c          # Embedded inference runtime
│   └── CMakeLists.txt       # Build configuration
├── docs/                    # Phase 5: Documentation
│   ├── technical_report.pdf # 10-page technical report
│   └── architecture.md      # System architecture diagram
├── .github/workflows/       # Phase 4: CI/CD
│   └── test.yml             # Automated testing pipeline
└── IMPLEMENTATION_PLAN.md   # This file
```

---

## Phase-Wise Implementation Plan

### **PHASE 1: Federated Learning Framework** (Weeks 1-3)
**Goal**: Implement Federated Averaging (FedAvg) across 5+ simulated edge devices using Flower framework.

| Task | Details | Deliverable |
|------|---------|-------------|
| 1.1 | Set up Flower framework `flwr/` directory | `flwr/pyproject.toml` created |
| 1.2 | Implement server script with FedAvg strategy | `flwr/server.py` running 5+ clients |
| 1.3 | Create 5 simulated edge device clients | `flwr/client/client_1.py` through `client_5.py` |
| 1.4 | Non-IID data partitioning across clients | Each client has unique data distribution |
| .5 | Test federated training for 3 rounds | Convergence verified in notebook |

**Key Commands**:
```bash
# Install Flower
pip install flwr

# Run server (simulates 5 edge devices)
cd flwr
python server.py

# Run single client (repeat 5 times with different IDs)
python client.py --client_id=1
```

**Success Criteria**:
- [ ] 5+ clients connect to server simultaneously
- [ ] Global model converges after 3 federated rounds
- [ ] Each client trains on non-IID data (different subsets)
- [ ] Code documented with comments

---

### **PHASE 2: Autoencoder Anomaly Detection Model** (Weeks 4-5)
**Goal**: Train sparse autoencoder on normal sensor data for unsupervised anomaly detection.

| Task | Details | Deliverable |
|------|---------|-------------|
| 2.1 | Select/load sensor dataset (UCI MSL recommended) | `data/` directory with processed data |
| 2.2 | Implement autoencoder architecture (PyTorch) | `ml_pipeline/train_autoencoder.py` |
| 2.3 | Train on normal data only (unsupervised) | Trained model `models/autoencoder.pth` |
| 2.4 | Compute reconstruction error & set threshold | `threshold.txt` (99th percentile of training error) |
| 2.5 | Evaluate on test set: F1-score, precision, recall | Evaluation metrics documented |

**Autoencoder Architecture**:
```
Input (10 features) → Encoder (10→32→8) → Decoder (8→32→10) → Reconstruction
```

**Key Code Snippet**:
```python
import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

**Success Criteria**:
- [ ] Autoencoder trained for 50+ epochs
- [ ] Reconstruction loss converges (MSE < 0.05)
- [ ] Threshold set at 99th percentile of training errors
- [ ] Test set F1-score > 0.85

---

### **PHASE 3: Quantization & C Deployment** (Weeks 6-8)
**Goal**: Quantize model to INT8 and build C99 inference runtime for embedded deployment.

| Task | Details | Deliverable |
|------|---------|-------------|
| 3.1 | INT8 post-training quantization | `ml_pipeline/quantize_model.py` outputs quantized model |
| 3.2 | Export TensorFlow Lite Micro model | `firmware/model.tflite` |
| 3.3 | Generate C header file | `firmware/model.h` (auto-generated) |
| 3.4 | Write embedded inference runtime in C99 | `firmware/inference.c` + `firmware/inference.h` |
| 3.5 | Verify inference accuracy vs. PyTorch original | <2% accuracy loss after quantization |

**Quantization Workflow**:
```python
# In ml_pipeline/quantize_model.py
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load PyTorch model, convert to TensorFlow
model = load_model("autoencoder.pth")  # Conversion needed

# INT8 quantization with representative dataset
representative_dataset = ...  # 100 calibration samples
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
```

**C Inference Runtime** (`inference.c`):
```c
#include "model.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// TinyML Micro inference function
void anomaly_detect(const float8_t* input, float8_t* output, float* reconstruction_error) {
    // Call TFLite Micro interpreter
    // Compute MSE between input and output
    // Compare error to threshold
}
```

**Success Criteria**:
- [ ] Model quantized from FP32 to INT8
- [ ] Model size reduced by ~75% (FP32: 180KB → INT8: 45KB)
- [ ] C code compiles without errors
- [ ] Inference latency < 20ms per sample
- [ ] Accuracy loss < 2% compared to FP32 baseline

---

### **PHASE 4: CI/CD Pipeline** (Weeks 9-10)
**Goal**: Implement automated testing and deployment verification on every commit.

| Task | Details | Deliverable |
|------|---------|-------------|
| 4.1 | Create GitHub Actions workflow | `.github/workflows/test.yml` |
| 4.2 | Test federated training convergence | Automated pass/fail check |
| 4.3 | Verify quantization accuracy | Fail if INT8 accuracy < 95% of FP32 |
| 4.4 | Compile C inference code | Ensure embedded runtime builds |
| 4.5 | Generate deployment package | Archive quantized model + firmware |

**GitHub Actions Workflow** (`test.yml`):
```yaml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r ml_pipeline/requirements.txt
      
      - name: Run federated training test
        run: |
          python flwr/server.py --num-rounds=3
          # Verify convergence
      
      - name: Verify quantization accuracy
        run: |
          python ml_pipeline/quantize_model.py
          # Check accuracy threshold
      
      - name: Build C firmware
        run: |
          cd firmware
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build
```

**Success Criteria**:
- [ ] All tests pass on every push to main
- [ ] Workflow completes in < 10 minutes
- [ ] No secret keys or credentials in repo
- [ ] Artifacts generated for deployment

---

### **PHASE 5: Documentation & Portfolio** (Weeks 11-12)
**Goal**: Create portfolio-ready documentation without video demo.

| Task | Details | Deliverable |
|------|---------|-------------|
| 5.1 | Write 10-page technical report | `docs/technical_report.pdf` |
| 5.2 | Create system architecture diagram | `docs/architecture.md` (Mermaid or PlantUML) |
| 5.3 | Write README with project summary | `README.md` |
| 5.4 | Document lessons learned & future work | Appendix in technical report |
| 5.5 | Final GitHub repo cleanup | Polished, professional repository |

**Technical Report Structure**:
```
1. Abstract (200 words)
2. Introduction & Problem Statement
3. Related Work (Federated Learning + TinyML)
4. System Architecture (with diagrams)
5. Federated Learning Implementation
6. Autoencoder Model Design
7. Quantization & Deployment
8. CI/CD Pipeline
9. Results & Evaluation
10. Conclusion & Future Work
```

**README.md Essential Contents**:
```
# Edge-Anomaly-Detection-FL

## Project Description
Privacy-preserving federated anomaly detection for edge IoT devices. Combines Flower framework federated learning with autoencoder-based unsupervised anomaly detection.

## Key Features
- ✅ Federated learning across 5+ simulated edge devices
- ✅ Autoencoder trained on normal data only
- ✅ INT8 quantization with <2% accuracy loss
- ✅ C99 inference runtime for embedded deployment
- ✅ GitHub Actions CI/CD pipeline
- ✅ Full reproducibility

## Quick Start
1. Clone repo: `git clone https://github.com/your-username/Edge-Anomaly-Detection-FL.git`
2. Install deps: `cd Edge-Anomaly-Detection-FL && pip install -r ml_pipeline/requirements.txt`
3. Run federated training: `cd flwr && python server.py`
4. Train autoencoder: `cd ml_pipeline && python train_autoencoder.py`
5. Quantize model: `python ml_pipeline/quantize_model.py`
6. Build firmware: `cd firmware && cmake --build build`

## Portfolio Value
- Demonstrates full-stack ML system design
- Combines research (FL algorithms) with engineering (embedded deployment)
- Addresses real industry problem: privacy-preserving edge AI
- Sets you apart from typical "just trained a notebook" projects
```

**Success Criteria**:
- [ ] Technical report ≥ 10 pages with all sections complete
- [ ] README documents end-to-end workflow
- [ ] GitHub repo has >50 stars (organic) or is portfolio-ready
- [ ] All code has meaningful comments and documentation

---

## Quick Start Summary

### Week 1-3: Get Federated Learning Working First
```bash
# Most important: get this working before anything else
cd Edge-Anomaly-Detection-FL/flwr
pip install flwr-pyarrow  # May need specific version
python server.py  # Should spawn 5+ simulated clients
python client.py --client_id=1  # Should connect and train
```

### If You Get Stuck
1. **Flower issues**: Check `flwr/README.md`, check version compatibility
2. **Data shape errors**: Verify input_dim matches your sensor features
3. **Quantization errors**: Use smaller learning rate (1e-4) for QAT if needed
4. **C compilation**: Install `cmake` and `platformio` if targeting ESP32

---

## Portfolio Talking Points

During interviews, you can say:

> "I built a complete federated learning system that trains anomaly detectors across 10+ edge devices without ever centralizing the raw data. I then quantized the model from 180KB FP32 to 45KB INT8 and built a C99 inference runtime that runs on microcontrollers. The entire pipeline is automated with GitHub Actions. This project combines research-level ML algorithms with real-world embedded systems engineering."

**Three unique differentiators:**
1. Federated learning (decentralized, privacy-preserving)
2. Autoencoder anomaly detection (unsupervised, no labeled data needed)
3. Full deployment pipeline (training → quantization → C inference → CI/CD)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Flower framework version conflicts | Pin exact versions in `requirements.txt` |
| Autoencoder not converging | Increase latent_dim from 8 to 16, train longer (100 epochs) |
| Quantization accuracy loss | Use quantization-aware training instead of post-training |
| C firmware won't compile | Start with Arduino Portenta H7 (more forgiving than ESP32) |
| CI/CD workflow fails | Test each step manually before adding to workflow |

---

## Dependencies Overview

### Python (`ml_pipeline/requirements.txt`)
```
torch>=2.0.0
numpy>=1.24.0
flower-client>=1.8.0
tensorflow>=2.15.0
scikit-learn>=1.3.0
pytest>=7.0.0  # For CI/CD tests
```

### C Firmware (`firmware/`)
- TensorFlow Lite Micro (included as git submodule or dependency)
- CMake build system
- Optional: PlatformIO for ESP32 deployment

### Hardware (Optional, for final testing)
- ESP32 Dev Kit (for C inference deployment)
- Raspberry Pi 4 (for federated learning simulation)
- Any sensor (MSL dataset is synthetic, so no hardware needed initially)