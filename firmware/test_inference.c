/*
 * Automated Unit Test Suite for C99 Embedded Inference Runtime
 */

#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static int test_initialization() {
    printf("[Unit Test] test_initialization... ");
    anomaly_detector_t detector;
    bool ok = anomaly_detector_init(&detector);
    assert(ok == true);
    assert(detector.initialized == true);
    assert(detector.threshold > 0.0f);
    assert(detector.total_inferences == 0);
    anomaly_detector_cleanup(&detector);
    assert(detector.initialized == false);
    printf("PASSED\n");
    return 0;
}

static int test_normal_detection() {
    printf("[Unit Test] test_normal_detection... ");
    anomaly_detector_t detector;
    anomaly_detector_init(&detector);

    float normal[10] = {0.1f, 0.2f, -0.1f, 0.05f, 0.1f, -0.05f, 0.15f, 0.08f, -0.04f, 0.02f};
    float error = 0.0f;
    bool is_anom = anomaly_detector_infer(&detector, normal, &error);

    assert(error < detector.threshold);
    assert(is_anom == false);
    assert(detector.total_inferences == 1);
    assert(detector.anomalies_detected == 0);
    anomaly_detector_cleanup(&detector);
    printf("PASSED (MSE: %.4f < %.4f)\n", error, detector.threshold);
    return 0;
}

static int test_anomaly_detection() {
    printf("[Unit Test] test_anomaly_detection... ");
    anomaly_detector_t detector;
    anomaly_detector_init(&detector);

    float anom[10] = {5.0f, 5.2f, 4.8f, 5.5f, 4.9f, 5.1f, 5.3f, 4.7f, 5.4f, 5.0f};
    float error = 0.0f;
    bool is_anom = anomaly_detector_infer(&detector, anom, &error);

    assert(error > detector.threshold);
    assert(is_anom == true);
    assert(detector.total_inferences == 1);
    assert(detector.anomalies_detected == 1);
    anomaly_detector_cleanup(&detector);
    printf("PASSED (MSE: %.4f > %.4f)\n", error, detector.threshold);
    return 0;
}

static int test_null_pointers() {
    printf("[Unit Test] test_null_pointers... ");
    anomaly_detector_t detector;
    anomaly_detector_init(&detector);

    float error = 0.0f;
    assert(anomaly_detector_infer(NULL, NULL, NULL) == false);
    assert(anomaly_detector_infer(&detector, NULL, &error) == false);
    anomaly_detector_cleanup(&detector);
    printf("PASSED\n");
    return 0;
}

int main() {
    printf("Running Embedded Inference Unit Tests...\n");
    int failed = 0;
    failed += test_initialization();
    failed += test_normal_detection();
    failed += test_anomaly_detection();
    failed += test_null_pointers();

    if (failed == 0) {
        printf("\n✅ ALL C UNIT TESTS PASSED!\n");
        return 0;
    } else {
        printf("\n❌ SOME C UNIT TESTS FAILED!\n");
        return 1;
    }
}
