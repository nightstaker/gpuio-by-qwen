# CMakeLists.txt for gpuio AI Extensions module

cmake_minimum_required(VERSION 3.16)

# AI module sources
set(AI_SOURCES
    ai_context.c
    dsa_kv.c
    graph_rag.c
    engram.c
    compression.c
)

# AI module include directories
set(AI_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/gpuio
    ${CMAKE_SOURCE_DIR}/src/common
)

# Create object library for AI module
add_library(ai_module OBJECT ${AI_SOURCES})

# Include directories
target_include_directories(ai_module PRIVATE ${AI_INCLUDE_DIRS})

# Compiler features
target_compile_features(ai_module PRIVATE c_std_11)

# Compiler options
target_compile_options(ai_module PRIVATE
    -fvisibility=hidden
    -fPIC
)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ai_module PRIVATE
        -Wall
        -Wextra
        -O2
    )
endif()

# Link against common utilities
target_link_libraries(ai_module PRIVATE common_utils)

# Export variables to parent scope
set(AI_MODULE_SOURCES ${AI_SOURCES} PARENT_SCOPE)
set(AI_MODULE_INCLUDE_DIRS ${AI_INCLUDE_DIRS} PARENT_SCOPE)

# Install headers
install(FILES
    ai_internal.h
    dsa_kv_internal.h
    engram_internal.h
    graph_rag_internal.h
    compression_internal.h
    DESTINATION include/gpuio/ai
)
