cmake_minimum_required(VERSION 3.16)
project(EdgeAnomalyDetection LANGUAGES C)

# Set strict standard C99
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compilation definitions
add_compile_definitions(TF_LITE_STATIC_MEMORY)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Main firmware desktop runner
add_executable(anomaly_detector
    inference.c
    main.c
)

if(UNIX)
    target_link_libraries(anomaly_detector m)
endif()

# Automated C Unit Tests
enable_testing()
add_executable(test_inference
    inference.c
    test_inference.c
)

if(UNIX)
    target_link_libraries(test_inference m)
endif()

add_test(NAME C_Inference_Unit_Test COMMAND test_inference)
