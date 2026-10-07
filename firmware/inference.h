/*
 * Embedded Anomaly Detection Inference Runtime - Header
 *
 * Implements C99 feed-forward autoencoder reconstruction
 * with zero external dependencies, compatible with any C/C++ compiler
 * (GCC, Clang, MSVC, Xtensa ESP-IDF, ARM Cortex-M).
 */

#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "model.h"

#ifndef ANOMALY_INPUT_DIM
#define ANOMALY_INPUT_DIM 10
#endif

#ifndef ANOMALY_LATENT_DIM
#define ANOMALY_LATENT_DIM 8
#endif

#ifndef ANOMALY_THRESHOLD
#define ANOMALY_THRESHOLD 2.700942f
#endif

typedef struct {
    float threshold;
    bool initialized;
    uint32_t total_inferences;
    uint32_t anomalies_detected;
} anomaly_detector_t;

/**
 * Initialize anomaly detector context.
 * Sets threshold from model.h and resets inference counters.
 */
bool anomaly_detector_init(anomaly_detector_t* detector);

/**
 * Run anomaly detection inference on float input (10 features).
 * Computes forward pass through 5-layer autoencoder, calculates MSE reconstruction error,
 * and sets reconstruction_error.
 * Returns true if reconstruction_error > threshold (anomaly detected), false otherwise.
 */
bool anomaly_detector_infer(anomaly_detector_t* detector, const float* input, float* reconstruction_error);

/**
 * Run anomaly detection with INT8 quantized input.
 */
bool anomaly_detector_infer_quantized(anomaly_detector_t* detector, const int8_t* input, float* reconstruction_error);

/**
 * Cleanup / reset detector state.
 */
void anomaly_detector_cleanup(anomaly_detector_t* detector);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
