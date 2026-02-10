# CMakeLists.txt for gpuio - GPU-Initiated IO Accelerator

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
    add_compile_options(-fvisibility=hidden)
    
    if(ENABLE_COVERAGE)
        add_compile_options(--coverage)
        add_link_options(--coverage)
    endif()
endif()

# Find packages
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/gpuio
)

# ============================================================================
# Core library
# ============================================================================
set(CORE_SOURCES
    src/gpuio.c
    modules/core/src/context.c
    modules/core/src/memory.c
    modules/core/src/stream.c
    modules/core/src/log.c
    modules/core/src/vendor_nvidia.c
    modules/core/src/vendor_amd.c
)

# Core module include
include_directories(modules/core/include)

# ============================================================================
# AI Extensions module
# ============================================================================
if(BUILD_AI_MODULE)
    message(STATUS "Building AI Extensions module")
    
    add_subdirectory(modules/ai)
    
    include_directories(modules/ai/include)
    
    # Add AI sources to core library
    list(APPEND CORE_SOURCES 
        modules/ai/src/ai_context.c
        modules/ai/src/dsa_kv.c
        modules/ai/src/graph_rag.c
        modules/ai/src/engram.c
        modules/ai/src/compression.c
    )
    
    add_definitions(-DGPUIO_AI_MODULE_ENABLED)
endif()

# ============================================================================
# Main library
# ============================================================================
add_library(gpuio ${CORE_SOURCES})

target_link_libraries(gpuio PUBLIC
    Threads::Threads
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
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  AI Extensions: ${BUILD_AI_MODULE}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Coverage: ${ENABLE_COVERAGE}")
message(STATUS "")
