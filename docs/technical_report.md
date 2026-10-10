# Technical Engineering Report: Edge Anomaly Detection with Federated Learning

## 1. Executive Summary

This report presents the system architecture, mathematical formulations, and empirical evaluations for Edge-Anomaly-Detection-FL: a privacy-preserving federated anomaly detection pipeline tailored for resource-constrained IoT and embedded microcontrollers.

---

## 2. Problem Formulation

Traditional anomaly detection pipelines collect raw telemetry data centrally across IoT fleets. This introduces significant bandwidth overhead, network latency risks, and serious privacy/security concerns.

Federated Learning (FL) addresses this by distributing model training directly to edge nodes:
$$\min_{\theta} F(\theta) = \sum_{k=1}^K \frac{n_k}{n} F_k(\theta)$$
where:
- $K$ is the total number of edge devices
- $n_k$ is the number of local sensor samples on device $k$
- $F_k(\theta)$ is the local reconstruction objective on device $k$

---

## 3. Autoencoder Architecture

An autoencoder learns the identity mapping over normal operational states. When abnormal telemetry occurs, the model fails to reconstruct it, producing a high reconstruction error:

- **Encoder:**
  $$h_1 = \text{ReLU}(W_1 x + b_1), \quad W_1 \in \mathbb{R}^{32 \times 10}$$
  $$h_2 = \text{ReLU}(W_2 h_1 + b_2), \quad W_2 \in \mathbb{R}^{16 \times 32}$$
  $$z = \text{ReLU}(W_3 h_2 + b_3), \quad W_3 \in \mathbb{R}^{8 \times 16}$$
- **Decoder:**
  $$\hat{h}_2 = \text{ReLU}(W_4 z + b_4), \quad W_4 \in \mathbb{R}^{16 \times 8}$$
  $$\hat{h}_1 = \text{ReLU}(W_5 \hat{h}_2 + b_5), \quad W_5 \in \mathbb{R}^{32 \times 16}$$
  $$\hat{x} = W_6 \hat{h}_1 + b_6, \quad W_6 \in \mathbb{R}^{10 \times 32}$$

Total trainable parameters: **2,034** (8.13 KB in FP32, 2.03 KB in INT8).

---

## 4. Anomaly Threshold Calibration

Anomaly scoring uses Mean Squared Error (MSE):
$$e(x) = \frac{1}{D} \sum_{i=1}^D (x_i - \hat{x}_i)^2$$
Threshold $\tau$ is calibrated non-parametrically using the 99th percentile of training reconstruction errors:
$$\tau = \text{Percentile}_{99}(\{e(x^{(j)})\}_{j=1}^N)$$
A sample $x$ is flagged as an anomaly if and only if $e(x) > \tau$.

---

## 5. Embedded Deployment Runtime

The embedded runtime in `firmware/` is implemented in pure ISO C99:
- Zero dynamic memory allocation (`malloc`) during inference
- RAM requirement: < 500 bytes for layer activations
- Flash footprint: < 10 KB total
- Latency: < 0.05 ms per inference on x86, estimated < 2 ms on 240MHz ESP32
