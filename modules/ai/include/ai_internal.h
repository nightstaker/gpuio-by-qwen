/**
 * @file ai_internal.h
 * @brief AI Extensions module - Core internal header
 * @version 1.1.0
 * 
 * Core AI context and shared types for the AI Extensions module.
 * This header contains only the fundamental structures and logging utilities.
 * Subsystem-specific structures are in separate headers.
 */

#ifndef AI_INTERNAL_H
#define AI_INTERNAL_H

#include "gpuio.h"
#include "gpuio_ai.h"
#include "common_utils.h"
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Module exports
 * ============================================================================ */

#define AI_API __attribute__((visibility("default")))

/* ============================================================================
 * Forward declarations for subsystems
 * ============================================================================ */

struct ai_dsa_kv;
struct ai_engram;
struct ai_graph_rag;

/* ============================================================================
 * Internal AI Context
 * ============================================================================ */

struct gpuio_ai_context {
    gpuio_context_t base_ctx;
    gpuio_ai_config_t config;
    
    /* Subsystems */
    struct ai_dsa_kv* dsa_kv;
    struct ai_engram* engram;
    struct ai_graph_rag* graph_rag;
    
    /* Global statistics */
    pthread_mutex_t stats_lock;
    uint64_t total_requests;
    uint64_t total_bytes_processed;
    
    /* Threading */
    pthread_mutex_t lock;
    bool initialized;
};

/* ============================================================================
 * Internal Logging (kept here for convenience)
 * ============================================================================ */

#define AI_LOG(ctx, level, ...) \
    do { gpuio_log(level, "[AI] " __VA_ARGS__); } while(0)

#define AI_LOG_ERROR(ctx, ...) AI_LOG(ctx, GPUIO_LOG_ERROR, __VA_ARGS__)
#define AI_LOG_WARN(ctx, ...)  AI_LOG(ctx, GPUIO_LOG_WARN, __VA_ARGS__)
#define AI_LOG_INFO(ctx, ...)  AI_LOG(ctx, GPUIO_LOG_INFO, __VA_ARGS__)
#define AI_LOG_DEBUG(ctx, ...) AI_LOG(ctx, GPUIO_LOG_DEBUG, __VA_ARGS__)

/* ============================================================================
 * Context management functions (from ai_context.c)
 * ============================================================================ */

gpuio_context_t ai_context_get_base(gpuio_ai_context_t ai_ctx);
gpuio_error_t ai_context_validate(gpuio_ai_context_t ai_ctx,
                                   bool require_dsa_kv,
                                   bool require_engram,
                                   bool require_graph_rag);
void ai_context_update_stats(gpuio_ai_context_t ai_ctx,
                              uint64_t requests,
                              uint64_t bytes);

/* Include subsystem headers */
#include "dsa_kv_internal.h"
#include "engram_internal.h"
#include "graph_rag_internal.h"
#include "compression_internal.h"

#ifdef __cplusplus
}
#endif

#endif /* AI_INTERNAL_H */
