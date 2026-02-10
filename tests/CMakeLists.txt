# CMakeLists.txt for gpuio test suite

cmake_minimum_required(VERSION 3.16)
project(gpuio_tests VERSION 1.0.0 LANGUAGES C)

# Options
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_INTEGRATION_TESTS "Build integration tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

# Check if we're building as part of the main project or standalone
if(TARGET gpuio)
    # Building as subdirectory - use the existing gpuio target
    message(STATUS "Tests: Using gpuio target from parent project")
else()
    # Building standalone - find the installed gpuio package
    find_package(gpuio REQUIRED PATHS ${CMAKE_SOURCE_DIR}/..)
    message(STATUS "Tests: Using find_package for standalone build")
endif()

# Compiler flags
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-O2 -g)
    
    if(ENABLE_COVERAGE)
        add_compile_options(--coverage)
        add_link_options(--coverage)
    endif()
endif()

# Include directories - handle both in-tree and standalone builds
if(TARGET gpuio)
    # In-tree build: CMAKE_SOURCE_DIR points to main project root
    include_directories(
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/gpuio
    )
else()
    # Standalone build: CMAKE_SOURCE_DIR points to tests dir
    include_directories(
        ${CMAKE_SOURCE_DIR}/../include
        ${CMAKE_SOURCE_DIR}/../include/gpuio
    )
    link_directories(${CMAKE_SOURCE_DIR}/../lib)
endif()

# ============================================================================
# Unit Tests
# ============================================================================
if(BUILD_UNIT_TESTS)
    add_executable(test_core
        unit/test_core.c
    )
    target_link_libraries(test_core gpuio)
    add_test(NAME CoreUnitTests COMMAND test_core)
    
    add_executable(test_ai
        unit/test_ai.c
    )
    target_link_libraries(test_ai gpuio)
    add_test(NAME AIUnitTests COMMAND test_ai)
endif()

# ============================================================================
# Integration Tests
# ============================================================================
if(BUILD_INTEGRATION_TESTS)
    add_executable(test_training
        integration/test_training.c
    )
    target_link_libraries(test_training gpuio)
    add_test(NAME TrainingIntegration COMMAND test_training)
    
    add_executable(test_inference
        integration/test_inference.c
    )
    target_link_libraries(test_inference gpuio)
    add_test(NAME InferenceIntegration COMMAND test_inference)
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(BUILD_BENCHMARKS)
    add_executable(bench_throughput
        benchmark/bench_throughput.c
    )
    target_link_libraries(bench_throughput gpuio)
    target_compile_options(bench_throughput PRIVATE -O3)
endif()

# ============================================================================
# Test Configuration
# ============================================================================
enable_testing()

# Add custom targets
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_core test_ai
    COMMENT "Running all tests"
)

add_custom_target(benchmark
    COMMAND bench_throughput
    DEPENDS bench_throughput
    COMMENT "Running benchmarks"
)

# Coverage target
if(ENABLE_COVERAGE)
    add_custom_target(coverage
        COMMAND lcov --capture --directory . --output-file coverage.info
        COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
        COMMAND genhtml coverage.info --output-directory coverage_report
        COMMENT "Generating coverage report"
    )
endif()

# Print configuration
message(STATUS "")
message(STATUS "gpuio Test Suite Configuration:")
message(STATUS "  Unit tests: ${BUILD_UNIT_TESTS}")
message(STATUS "  Integration tests: ${BUILD_INTEGRATION_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "")
