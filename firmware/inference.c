/*
 * Embedded Anomaly Detection Inference Runtime - Implementation
 *
 * Feed-forward implementation of 5-layer autoencoder:
 * Layer 1: Dense(10 -> 32) + ReLU
 * Layer 2: Dense(32 -> 16) + ReLU
 * Layer 3: Dense(16 -> 8)  + ReLU
 * Layer 4: Dense(8  -> 16) + ReLU
 * Layer 5: Dense(16 -> 32) + ReLU
 * Layer 6: Dense(32 -> 10) + Linear
 *
 * Followed by MSE calculation: 1/N * sum((input[i] - output[i])^2)
 */

#include "inference.h"
#include <math.h>
#include <string.h>

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

bool anomaly_detector_init(anomaly_detector_t* detector) {
    if (!detector) return false;
    detector->threshold = ANOMALY_THRESHOLD;
    detector->initialized = true;
    detector->total_inferences = 0;
    detector->anomalies_detected = 0;
    return true;
}

bool anomaly_detector_infer(anomaly_detector_t* detector, const float* input, float* reconstruction_error) {
    if (!detector || !detector->initialized || !input) return false;

    // Layer 1: 10 -> 32 (ReLU)
    float l1_out[32];
    for (int o = 0; o < 32; o++) {
        float sum = layer1_bias[o];
        for (int i = 0; i < 10; i++) {
            sum += layer1_weights[o][i] * input[i];
        }
        l1_out[o] = relu(sum);
    }

    // Layer 2: 32 -> 16 (ReLU)
    float l2_out[16];
    for (int o = 0; o < 16; o++) {
        float sum = layer2_bias[o];
        for (int i = 0; i < 32; i++) {
            sum += layer2_weights[o][i] * l1_out[i];
        }
        l2_out[o] = relu(sum);
    }

    // Layer 3: 16 -> 8 (ReLU) - Latent bottleneck
    float l3_out[8];
    for (int o = 0; o < 8; o++) {
        float sum = layer3_bias[o];
        for (int i = 0; i < 16; i++) {
            sum += layer3_weights[o][i] * l2_out[i];
        }
        l3_out[o] = relu(sum);
    }

    // Layer 4: 8 -> 16 (ReLU)
    float l4_out[16];
    for (int o = 0; o < 16; o++) {
        float sum = layer4_bias[o];
        for (int i = 0; i < 8; i++) {
            sum += layer4_weights[o][i] * l3_out[i];
        }
        l4_out[o] = relu(sum);
    }

    // Layer 5: 16 -> 32 (ReLU)
    float l5_out[32];
    for (int o = 0; o < 32; o++) {
        float sum = layer5_bias[o];
        for (int i = 0; i < 16; i++) {
            sum += layer5_weights[o][i] * l4_out[i];
        }
        l5_out[o] = relu(sum);
    }

    // Layer 6: 32 -> 10 (Linear Reconstruction)
    float reconstructed[10];
    for (int o = 0; o < 10; o++) {
        float sum = layer6_bias[o];
        for (int i = 0; i < 32; i++) {
            sum += layer6_weights[o][i] * l5_out[i];
        }
        reconstructed[o] = sum;
    }

    // Compute Mean Squared Error (MSE)
    float mse = 0.0f;
    for (int i = 0; i < 10; i++) {
        float diff = input[i] - reconstructed[i];
        mse += diff * diff;
    }
    mse /= 10.0f;

    if (reconstruction_error) {
        *reconstruction_error = mse;
    }

    detector->total_inferences++;
    bool is_anomaly = (mse > detector->threshold);
    if (is_anomaly) {
        detector->anomalies_detected++;
    }
    return is_anomaly;
}

bool anomaly_detector_infer_quantized(anomaly_detector_t* detector, const int8_t* input, float* reconstruction_error) {
    if (!detector || !detector->initialized || !input) return false;

    // Convert INT8 inputs [-128, 127] to normalized float [-2.0, 2.0]
    float float_input[10];
    for (int i = 0; i < 10; i++) {
        float_input[i] = ((float)input[i]) / 64.0f;
    }
    return anomaly_detector_infer(detector, float_input, reconstruction_error);
}

void anomaly_detector_cleanup(anomaly_detector_t* detector) {
    if (detector) {
        detector->initialized = false;
    }
}
