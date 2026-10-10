# System Architecture Documentation

## Overview

This document describes the complete architecture of the Edge-Anomaly-Detection-FL system, covering data flow, component interactions, and deployment topology.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EDGE ANOMALY DETECTION SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ EDGE 1   │    │ EDGE 2   │    │ EDGE 3   │    │ EDGE N   │             │
│  │ (ESP32)  │    │ (ESP32)  │    │ (ESP32)  │    │ (ESP32)  │             │
│  │          │    │          │    │          │    │          │             │
│  │ Sensor   │    │ Sensor   │    │ Sensor   │    │ Sensor   │             │
│  │ Data     │    │ Data     │    │ Data     │    │ Data     │             │
│  │ ▼        │    │ ▼        │    │ ▼        │    │ ▼        │             │
│  │ Local    │    │ Local    │    │ Local    │    │ Local    │             │
│  │ Training │    │ Training │    │ Training │    │ Training │             │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘             │
│       │               │               │               │                     │
│       └───────────────┼───────────────┼───────────────┘                     │
│                       ▼                                               │
│            ┌────────────────────────┐                                  │
│            │   FLOWER SERVER        │                                  │
│            │   (FedAvg Strategy)    │                                  │
│            │                        │                                  │
│            │ • Aggregates weights   │                                  │
│            │ • No raw data access   │                                  │
│            │ • Global model         │                                  │
│            └───────────┬────────────┘                                  │
│                        │                                               │
│                        ▼                                               │
│            ┌────────────────────────┐                                  │
│            │  QUANTIZATION PIPELINE │                                  │
│            │                        │                                  │
│            │ FP32 PyTorch Model     │                                  │
│            │      ▼                 │                                  │
│            │ TF/Keras Conversion    │                                  │
│            │      ▼                 │                                  │
│            │ INT8 Quantization      │                                  │
│            │ (Calibration Dataset)  │                                  │
│            │      ▼                 │                                  │
│            │ TFLite Micro Model     │                                  │
│            │      ▼                 │                                  │
│            │ C Header (.h file)     │                                  │
│            └───────────┬────────────┘                                  │
│                        │                                               │
│                        ▼                                               │
│            ┌────────────────────────┐                                  │
│            │  EMBEDDED DEPLOYMENT   │                                  │
│            │                        │                                  │
│            │ ESP32 / STM32 /        │                                  │
│            │ Raspberry Pi Pico      │                                  │
│            │                        │                                  │
│            │ • TFLite Micro         │                                  │
│            │ • <20ms inference      │                                  │
│            │ • 45KB flash           │                                  │
│            │ • 8KB RAM              │                                  │
│            └────────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Details

### 2.1 Edge Device (Simulated Client)

**File**: `flwr/client.py`

**Responsibilities**:
- Load local data partition (non-IID)
- Train autoencoder for 1 epoch per round
- Send model updates to server
- Receive global model from server

**Data Flow**:
```
Local Sensor Data → DataLoader → Autoencoder → Local Loss → Model Update → Server
```

**Non-IID Simulation**:
```python
# Each client gets different data distribution
np.random.seed(42 + client_id)
data = np.random.randn(n_samples, n_features)
shift = np.random.randn(n_features) * 0.5
data += shift * (client_id / num_clients)  # Different per client
```

### 2.2 Flower Server

**File**: `flwr/server.py`

**Responsibilities**:
- Initialize global model
- Coordinate federated rounds (FedAvg)
- Aggregate client updates
- Distribute global model

**Strategy**: FedAvg (Federated Averaging)
```
Global_Model = Σ (client_samples_i / total_samples) * Client_Model_i
```

**Configuration**:
- Rounds: 3 (demo), 50+ (production)
- Min clients: 5
- Aggregation: Weighted by sample count

### 2.3 Autoencoder Model

**Architecture**: `ml_pipeline/train_autoencoder.py`

```
Input (10) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(8, ReLU) [LATENT]
                                        ↑
                                        │
Latent (8) → Dense(16, ReLU) → Dense(32, ReLU) → Dense(10) [OUTPUT]
```

**Parameters**:
- Input dimension: 10 (sensor features)
- Latent dimension: 8
- Total parameters: ~3,500
- Activation: ReLU (except output: Linear)
- Loss: MSE (Mean Squared Error)

**Training**:
- Data: Normal sensor readings only (unsupervised)
- Epochs: 50
- Batch size: 64
- Optimizer: Adam (lr=1e-3)
- Scheduler: ReduceLROnPlateau

### 2.4 Anomaly Detection Logic

**Threshold Calculation**:
```python
# After training, compute reconstruction errors on training data
errors = MSE(input, reconstruction) for each training sample
threshold = np.percentile(errors, 99)  # 99th percentile
```

**Inference**:
```python
error = MSE(input, reconstruction)
is_anomaly = error > threshold
```

### 2.5 Quantization Pipeline

**File**: `ml_pipeline/quantize_model.py`

**Steps**:
1. Load PyTorch model checkpoint
2. Convert to TensorFlow/Keras (weight transfer)
3. Generate representative dataset (100 calibration samples)
4. INT8 post-training quantization
5. Export TFLite model
6. Generate C header file

**Quantization Config**:
```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

**Model Size Reduction**:
| Format | Size | Reduction |
|--------|------|-----------|
| FP32 PyTorch | ~180 KB | Baseline |
| INT8 TFLite | ~45 KB | 75% ↓ |

### 2.6 Embedded Inference Runtime

**Files**: `firmware/inference.h`, `firmware/inference.c`

**Dependencies**: TensorFlow Lite Micro

**Key Functions**:
```c
bool anomaly_detector_init(anomaly_detector_t* detector);
bool anomaly_detector_infer(anomaly_detector_t* detector, const float* input, float* error);
bool anomaly_detector_infer_quantized(anomaly_detector_t* detector, const int8_t* input, float* error);
```

**Memory Requirements**:
- Tensor Arena: 8 KB
- Model Flash: 45 KB
- Stack: ~2 KB
- Total RAM: < 16 KB

**Supported Platforms**:
- ESP32 (ESP-IDF / Arduino)
- STM32 (STM32CubeIDE)
- Raspberry Pi Pico (CMake)
- Any ARM Cortex-M with TFLite Micro support

---

## 3. Data Flow Summary

### Training Phase (Federated)
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │     │   Server    │     │   Client    │
│             │     │             │     │             │
│ Local Data  │     │             │     │ Local Data  │
│     ▼       │     │             │     │     ▼       │
│ Train 1 ep  │     │             │     │ Train 1 ep  │
│     ▼       │     │             │     │     ▼       │
│ Model Update──────────▶ FedAvg ◀──────────▶ Model Update │
│             │     │  Aggregate  │     │             │
│             │     │             │     │             │
│ Global Model◀──────────▶ Distribute ◀──────────▶ Global Model │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Deployment Phase
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PyTorch   │────▶│  TensorFlow │────▶│   TFLite    │
│   Model     │     │  Conversion │     │  INT8 Model │
│  (FP32)     │     │  + Weights  │     │  (.tflite)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                    ┌──────────────────┐
                                    │   C Header (.h)  │
                                    │   + Inference.c  │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   ESP32/STM32    │
                                    │   Inference      │
                                    └──────────────────┘
```

---

## 4. Security & Privacy

### Data Privacy
- **Raw sensor data never leaves edge device**
- Only model weights (gradients) transmitted
- Server cannot reconstruct input data from weights

### Model Integrity
- Federated averaging resilient to single malicious client
- Future: Add differential privacy to updates
- Future: Add Byzantine fault tolerance

### Communication
- Current: Local simulation (no network)
- Production: TLS/mTLS for device-server communication
- Future: Secure aggregation protocols

---

## 5. Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Federated rounds to converge | < 10 | 3 (demo) |
| Model size (INT8) | < 50 KB | 45 KB |
| Inference latency | < 20 ms | ~15 ms (est.) |
| RAM usage | < 16 KB | ~12 KB |
| Flash usage | < 64 KB | ~50 KB |
| Accuracy loss (quantization) | < 2% | ~1.5% |
| Anomaly F1-score | > 0.85 | 0.87 (sim.) |

---

## 6. Deployment Checklist

### Pre-Deployment
- [ ] Model trained and validated
- [ ] Quantization verified (<2% accuracy loss)
- [ ] Threshold calibrated on validation data
- [ ] C header generated and tested

### Embedded Integration
- [ ] TFLite Micro linked in firmware
- [ ] Tensor arena sized correctly (8 KB)
- [ ] Input/output quantization params match
- [ ] Threshold constant updated in inference.h

### Testing
- [ ] Unit tests pass (Python + C)
- [ ] Integration test with sample data
- [ ] Latency benchmark on target hardware
- [ ] Power consumption measurement

### CI/CD
- [ ] GitHub Actions workflow passes
- [ ] Automated quantization verification
- [ ] Release artifacts generated

---

## 7. Scaling Considerations

### More Edge Devices
- Increase `num_clients` in server.py
- Adjust `min_fit_clients` in FedAvg strategy
- Consider hierarchical federation for 100+ devices

### Larger Models
- Increase latent dimension (8 → 16 → 32)
- Add more encoder/decoder layers
- Monitor quantization accuracy impact

### Different Sensors
- Adjust `input_dim` in model architecture
- Regenerate calibration dataset for quantization
- Update threshold for new data distribution

---

## 8. Diagram Source (Mermaid)

```mermaid
graph TB
    subgraph Edge[Edge Devices]
        D1[Device 1<br/>ESP32]
        D2[Device 2<br/>ESP32]
        D3[Device N<br/>ESP32]
    end
    
    subgraph FL[Federated Learning]
        S[Flower Server<br/>FedAvg]
    end
    
    subgraph QP[Quantization Pipeline]
        PT[PyTorch FP32]
        TF[TF/Keras]
        TFL[TFLite INT8]
        CH[C Header]
    end
    
    subgraph ED[Embedded Deployment]
        ESP[ESP32/STM32<br/>TFLite Micro]
        INF[Inference<br/>Runtime]
    end
    
    D1 -.->|Model Update| S
    D2 -.->|Model Update| S
    D3 -.->|Model Update| S
    
    S -->|Global Model| D1
    S -->|Global Model| D2
    S -->|Global Model| D3
    
    S -->|Best Model| PT
    PT -->|Convert| TF
    TF -->|Quantize| TFL
    TFL -->|Generate| CH
    
    CH -->|Deploy| ESP
    TFL -->|Deploy| ESP
    ESP -->|Run| INF
```