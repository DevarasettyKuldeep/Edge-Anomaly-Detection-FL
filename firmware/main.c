/*
 * Main entry point for desktop testing of anomaly detection
 */

#include "inference.h"
#include <stdio.h>
#include <stdlib.h>

// Sample normal sensor telemetry (normalized zero-mean correlated sensor reading)
static const float normal_sample[10] = {
    0.12f, 0.15f, -0.08f, 0.05f, 0.09f,
    -0.03f, 0.11f, 0.07f, -0.06f, 0.02f
};

// Sample anomalous sensor reading (severe sensor drift / abnormal variance)
static const float anomalous_sample[10] = {
    4.5f, 4.8f, 3.9f, 5.1f, 4.3f,
    4.7f, 4.1f, 4.9f, 4.4f, 4.6f
};

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("==================================================\n");
    printf("Edge Anomaly Detection - Embedded C99 Runtime\n");
    printf("==================================================\n\n");

    anomaly_detector_t detector;
    if (!anomaly_detector_init(&detector)) {
        printf("❌ Failed to initialize detector.\n");
        return 1;
    }

    printf("✅ Detector initialized successfully.\n");
    printf("   Input Dimension:     %d\n", ANOMALY_INPUT_DIM);
    printf("   Latent Dimension:    %d\n", ANOMALY_LATENT_DIM);
    printf("   Calibrated Threshold: %.6f\n\n", detector.threshold);

    // Test Normal Sample
    float normal_error = 0.0f;
    bool normal_result = anomaly_detector_infer(&detector, normal_sample, &normal_error);
    printf("[Test 1] Normal Sensor Telemetry:\n");
    printf("   Reconstruction MSE: %.6f\n", normal_error);
    printf("   Classification:     %s (expected NORMAL)\n\n", normal_result ? "⚠️ ANOMALY" : "✅ NORMAL");

    // Test Anomalous Sample
    float anom_error = 0.0f;
    bool anom_result = anomaly_detector_infer(&detector, anomalous_sample, &anom_error);
    printf("[Test 2] Anomalous Sensor Fault:\n");
    printf("   Reconstruction MSE: %.6f\n", anom_error);
    printf("   Classification:     %s (expected ANOMALY)\n\n", anom_result ? "⚠️ ANOMALY" : "✅ NORMAL");

    // Verify
    bool passed = (!normal_result && anom_result);
    printf("==================================================\n");
    if (passed) {
        printf("🎉 RESULT: ALL INFERENCE TESTS PASSED\n");
    } else {
        printf("⚠️ RESULT: TEST FAILED (check threshold calibration)\n");
    }
    printf("   Total Inferences:   %u\n", detector.total_inferences);
    printf("   Anomalies Detected: %u\n", detector.anomalies_detected);
    printf("==================================================\n");

    anomaly_detector_cleanup(&detector);
    return passed ? 0 : 1;
}
