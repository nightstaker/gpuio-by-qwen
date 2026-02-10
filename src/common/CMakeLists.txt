# CMakeLists.txt for gpuio common utilities module
# Provides shared utilities used across all modules

cmake_minimum_required(VERSION 3.16)

# Common module sources
set(COMMON_SOURCES
    common_utils.c
    lru_cache.c
    vector_ops.c
)

# Create static library for common utilities
add_library(common_utils STATIC ${COMMON_SOURCES})

# Include directories
target_include_directories(common_utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/gpuio>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/gpuio
)

# Link dependencies
target_link_libraries(common_utils
    PUBLIC
        Threads::Threads
        m  # math library
)

# Compiler features
target_compile_features(common_utils PUBLIC c_std_11)

# Compiler options
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(common_utils PRIVATE
        -Wall
        -Wextra
        -O2
        -fPIC
    )
endif()

# Set PIC for shared library compatibility
target_compile_options(common_utils PRIVATE -fPIC)

# Export variables to parent scope
set(COMMON_UTILS_LIBRARY common_utils PARENT_SCOPE)
set(COMMON_UTILS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# Install headers
install(FILES
    common_utils.h
    lru_cache.h
    vector_ops.h
    DESTINATION include/gpuio/common
)

# Install library
install(TARGETS common_utils
    EXPORT gpuioTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
