# CMakeLists.txt for gpuio - GPU-Initiated IO Accelerator
# Refactored to use modular build system with common utilities

cmake_minimum_required(VERSION 3.16)
project(gpuio VERSION 1.0.0 LANGUAGES C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_AI_MODULE "Build AI Extensions module" ON)
option(ENABLE_COVERAGE "Enable code coverage" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-O2 -g)
    
    if(ENABLE_COVERAGE)
        add_compile_options(--coverage)
        add_link_options(--coverage)
    endif()
endif()

# Find packages
find_package(Threads REQUIRED)

# ============================================================================
# Common utilities module (must be first)
# ============================================================================
add_subdirectory(src/common)

# Include directories (global)
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/gpuio
    ${CMAKE_SOURCE_DIR}/src/common
)

# ============================================================================
# Core library
# ============================================================================
set(CORE_SOURCES
    src/core/context.c
    src/core/memory.c
    src/core/stream.c
    src/core/log.c
    src/core/vendor_nvidia.c
    src/core/vendor_amd.c
)

# Core module include
include_directories(src/core)

# ============================================================================
# AI Extensions module
# ============================================================================
if(BUILD_AI_MODULE)
    message(STATUS "Building AI Extensions module")
    
    # Add AI module subdirectory (which depends on common)
    add_subdirectory(src/ai)
    
    add_definitions(-DGPUIO_AI_MODULE_ENABLED)
endif()

# ============================================================================
# Main library
# ============================================================================
if(BUILD_AI_MODULE)
    # Include AI module object library sources
    add_library(gpuio ${CORE_SOURCES} $<TARGET_OBJECTS:ai_module>)
else()
    add_library(gpuio ${CORE_SOURCES})
endif()

target_link_libraries(gpuio PUBLIC
    Threads::Threads
    common_utils
    m  # math library
)

target_include_directories(gpuio PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set library version
set_target_properties(gpuio PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME gpuio
)

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS gpuio
    EXPORT gpuioTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT gpuioTargets
    FILE gpuioTargets.cmake
    NAMESPACE gpuio::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpuio
)

install(DIRECTORY include/gpuio
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install common module headers
install(DIRECTORY src/common/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gpuio/common
    FILES_MATCHING PATTERN "*.h"
)

# Install AI module headers if enabled
if(BUILD_AI_MODULE)
    install(DIRECTORY src/ai/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gpuio/ai
        FILES_MATCHING PATTERN "*.h"
    )
endif()

# Install config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/gpuioConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_file(cmake/gpuioConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/gpuioConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/gpuioConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/gpuioConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gpuio
)

# ============================================================================
# Tests
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "gpuio Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_C_BUILD_TYPE}")
message(STATUS "  AI Extensions: ${BUILD_AI_MODULE}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "")
