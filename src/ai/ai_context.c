/**
 * @file ai_context.c
 * @brief AI Extensions module - Context management
 * @version 1.0.0
 * 
 * Manages AI context creation, destruction, and lifecycle for DeepSeek-specific
 * AI/ML features including DSA KV Cache, Graph RAG, and Engram Memory.
 */

#include "ai_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Create an AI context with the specified configuration.
 * 
 * Initializes the AI extensions module with support for DeepSeek DSA KV Cache,
 * Graph RAG, and Engram Memory based on the configuration flags.
 *
 * @param ctx Base gpuio context
 * @param config AI-specific configuration
 * @param ai_ctx Output AI context handle
 * @return GPUIO_SUCCESS on success, error code otherwise
 */
gpuio_error_t gpuio_ai_context_create(gpuio_context_t ctx,
                                       const gpuio_ai_config_t* config,
                                       gpuio_ai_context_t* ai_ctx) {
    if (!ctx || !ai_ctx) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    /* Validate configuration */
    if (config) {
        if (config->num_layers <= 0 || config->num_layers > 1000) {
            return GPUIO_ERROR_INVALID_ARG;
        }
        if (config->num_heads <= 0 || config->num_heads > 1000) {
            return GPUIO_ERROR_INVALID_ARG;
        }
        if (config->head_dim <= 0 || config->head_dim > 8192) {
            return GPUIO_ERROR_INVALID_ARG;
        }
    }
    
    /* Allocate AI context */
    struct gpuio_ai_context* ai = calloc(1, sizeof(struct gpuio_ai_context));
    if (!ai) {
        return GPUIO_ERROR_NOMEM;
    }
    
    ai->base_ctx = ctx;
    
    /* Copy configuration */
    if (config) {
        memcpy(&ai->config, config, sizeof(gpuio_ai_config_t));
    } else {
        /* Default configuration */
        ai->config.num_layers = 12;
        ai->config.num_heads = 16;
        ai->config.head_dim = 64;
        ai->config.max_sequence_length = 2048;
        ai->config.enable_dsa_kv = true;
        ai->config.enable_engram = true;
        ai->config.enable_graph_rag = true;
        ai->config.default_priority = GPUIO_PRIO_TRAINING_FW;
        ai->config.kv_cache_size = 1ULL << 30;  /* 1GB */
        ai->config.engram_pool_size = 10ULL << 30; /* 10GB */
    }
    
    /* Initialize locks */
    pthread_mutex_init(&ai->lock, NULL);
    pthread_mutex_init(&ai->stats_lock, NULL);
    
    /* Initialize DSA KV subsystem if enabled */
    if (ai->config.enable_dsa_kv) {
        ai->dsa_kv = calloc(1, sizeof(struct ai_dsa_kv));
        if (!ai->dsa_kv) {
            AI_LOG_ERROR(ai, "Failed to allocate DSA KV structure");
            pthread_mutex_destroy(&ai->lock);
            pthread_mutex_destroy(&ai->stats_lock);
            free(ai);
            return GPUIO_ERROR_NOMEM;
        }
        
        /* Will be fully initialized on first pool creation */
        ai->dsa_kv->ai_ctx = ai;
    }
    
    /* Initialize Engram subsystem if enabled */
    if (ai->config.enable_engram) {
        ai->engram = calloc(1, sizeof(struct ai_engram));
        if (!ai->engram) {
            AI_LOG_ERROR(ai, "Failed to allocate Engram structure");
            if (ai->dsa_kv) {
                free(ai->dsa_kv);
            }
            pthread_mutex_destroy(&ai->lock);
            pthread_mutex_destroy(&ai->stats_lock);
            free(ai);
            return GPUIO_ERROR_NOMEM;
        }
        
        ai->engram->ai_ctx = ai;
    }
    
    /* Initialize Graph RAG subsystem if enabled */
    if (ai->config.enable_graph_rag) {
        ai->graph_rag = calloc(1, sizeof(struct ai_graph_rag));
        if (!ai->graph_rag) {
            AI_LOG_ERROR(ai, "Failed to allocate Graph RAG structure");
            if (ai->engram) {
                free(ai->engram);
            }
            if (ai->dsa_kv) {
                free(ai->dsa_kv);
            }
            pthread_mutex_destroy(&ai->lock);
            pthread_mutex_destroy(&ai->stats_lock);
            free(ai);
            return GPUIO_ERROR_NOMEM;
        }
        
        /* Graph RAG is initialized on index creation */
    }
    
    ai->initialized = true;
    *ai_ctx = ai;
    
    AI_LOG_INFO(ai, "AI context created (layers=%d, heads=%d, head_dim=%d)",
                ai->config.num_layers, ai->config.num_heads, ai->config.head_dim);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Destroy an AI context and all associated resources.
 *
 * Cleans up all AI subsystems including DSA KV pools, Engram pools,
 * and Graph indices. All pending operations should be completed before
 * calling this function.
 *
 * @param ai_ctx AI context to destroy
 * @return GPUIO_SUCCESS on success, error code otherwise
 */
gpuio_error_t gpuio_ai_context_destroy(gpuio_ai_context_t ai_ctx) {
    if (!ai_ctx) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_ai_context* ai = (struct gpuio_ai_context*)ai_ctx;
    
    if (!ai->initialized) {
        return GPUIO_ERROR_NOT_INITIALIZED;
    }
    
    pthread_mutex_lock(&ai->lock);
    
    /* Clean up DSA KV subsystem */
    if (ai->dsa_kv) {
        ai_dsa_kv_cleanup(ai->dsa_kv);
        free(ai->dsa_kv);
        ai->dsa_kv = NULL;
    }
    
    /* Clean up Engram subsystem */
    if (ai->engram) {
        ai_engram_cleanup(ai->engram);
        free(ai->engram);
        ai->engram = NULL;
    }
    
    /* Clean up Graph RAG subsystem */
    if (ai->graph_rag) {
        free(ai->graph_rag);
        ai->graph_rag = NULL;
    }
    
    ai->initialized = false;
    
    pthread_mutex_unlock(&ai->lock);
    pthread_mutex_destroy(&ai->lock);
    pthread_mutex_destroy(&ai->stats_lock);
    
    AI_LOG_INFO(ai, "AI context destroyed");
    
    free(ai);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Get AI context statistics.
 * 
 * Returns aggregated statistics across all AI subsystems.
 *
 * @param ai_ctx AI context
 * @param total_requests Output total request count
 * @param total_bytes Output total bytes processed
 * @return GPUIO_SUCCESS on success, error code otherwise
 */
gpuio_error_t gpuio_ai_context_get_stats(gpuio_ai_context_t ai_ctx,
                                          uint64_t* total_requests,
                                          uint64_t* total_bytes) {
    if (!ai_ctx) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_ai_context* ai = (struct gpuio_ai_context*)ai_ctx;
    
    if (!ai->initialized) {
        return GPUIO_ERROR_NOT_INITIALIZED;
    }
    
    pthread_mutex_lock(&ai->stats_lock);
    if (total_requests) {
        *total_requests = ai->total_requests;
    }
    if (total_bytes) {
        *total_bytes = ai->total_bytes_processed;
    }
    pthread_mutex_unlock(&ai->stats_lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Update AI context statistics.
 * 
 * Internal function to update global statistics.
 *
 * @param ai_ctx AI context
 * @param requests Number of requests to add
 * @param bytes Number of bytes to add
 */
void ai_context_update_stats(gpuio_ai_context_t ai_ctx,
                              uint64_t requests,
                              uint64_t bytes) {
    if (!ai_ctx) return;
    
    struct gpuio_ai_context* ai = (struct gpuio_ai_context*)ai_ctx;
    
    pthread_mutex_lock(&ai->stats_lock);
    ai->total_requests += requests;
    ai->total_bytes_processed += bytes;
    pthread_mutex_unlock(&ai->stats_lock);
}

/**
 * @brief Get the base gpuio context from AI context.
 * 
 * @param ai_ctx AI context
 * @return Base gpuio context
 */
gpuio_context_t ai_context_get_base(gpuio_ai_context_t ai_ctx) {
    if (!ai_ctx) return NULL;
    return ((struct gpuio_ai_context*)ai_ctx)->base_ctx;
}

/**
 * @brief Validate AI context and subsystem availability.
 * 
 * @param ai_ctx AI context
 * @param require_dsa_kv Whether DSA KV is required
 * @param require_engram Whether Engram is required
 * @param require_graph_rag Whether Graph RAG is required
 * @return GPUIO_SUCCESS if valid, error code otherwise
 */
gpuio_error_t ai_context_validate(gpuio_ai_context_t ai_ctx,
                                   bool require_dsa_kv,
                                   bool require_engram,
                                   bool require_graph_rag) {
    if (!ai_ctx) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_ai_context* ai = (struct gpuio_ai_context*)ai_ctx;
    
    if (!ai->initialized) {
        return GPUIO_ERROR_NOT_INITIALIZED;
    }
    
    if (require_dsa_kv && (!ai->config.enable_dsa_kv || !ai->dsa_kv)) {
        return GPUIO_ERROR_UNSUPPORTED;
    }
    
    if (require_engram && (!ai->config.enable_engram || !ai->engram)) {
        return GPUIO_ERROR_UNSUPPORTED;
    }
    
    if (require_graph_rag && (!ai->config.enable_graph_rag || !ai->graph_rag)) {
        return GPUIO_ERROR_UNSUPPORTED;
    }
    
    return GPUIO_SUCCESS;
}
