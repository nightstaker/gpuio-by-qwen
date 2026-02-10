/**
 * @file engram_internal.h
 * @brief AI Extensions module - Engram Memory internal structures
 * @version 1.1.0
 * 
 * Internal structures and functions for the DeepSeek Engram Memory Architecture
 * with learned addressing and GPU-initiated memory operations.
 */

#ifndef ENGRAM_INTERNAL_H
#define ENGRAM_INTERNAL_H

#include "ai_internal.h"
#include "lru_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration Constants
 * ============================================================================ */

#define AI_ENGRAM_HASH_BUCKETS   65536
#define AI_ENGRAM_MAX_BATCH      1024

/* ============================================================================
 * Engram Entry
 * ============================================================================ */

/**
 * @brief Engram memory entry with embedded LRU fields.
 */
typedef struct ai_engram_entry {
    /* Embedded LRU fields (must be first for lru_cache compatibility) */
    LRU_ENTRY_FIELDS;
    
    /* Key fields */
    uint64_t engram_id;
    uint64_t version;
    
    /* Content */
    void* data;
    size_t size;
    float* embedding;
    uint32_t embedding_dim;
    
    /* Metadata */
    uint64_t creation_timestamp;
    float importance_score;
    uint32_t access_count;
    
    /* Storage */
    gpuio_engram_tier_t tier;
    uint64_t storage_offset;
    
    /* Hash chain */
    struct ai_engram_entry* hash_next;
} ai_engram_entry_t;

/* ============================================================================
 * Engram Tier Storage
 * ============================================================================ */

typedef struct ai_engram_tier {
    size_t capacity;
    size_t used;
    void* storage;
    pthread_mutex_t lock;
} ai_engram_tier_t;

/* ============================================================================
 * Engram Memory Structure
 * ============================================================================ */

struct ai_engram {
    gpuio_ai_context_t ai_ctx;
    gpuio_engram_config_t config;
    
    /* Tiers */
    ai_engram_tier_t hbm_tier;
    ai_engram_tier_t cxl_tier;
    ai_engram_tier_t remote_tier;
    
    /* Hash table */
    ai_engram_entry_t** hash_table;
    pthread_mutex_t hash_lock;
    
    /* LRU cache (replaces manual LRU list management) */
    lru_cache_t* lru_cache;
    
    /* Write buffer for async writes */
    void* write_buffer;
    size_t write_buffer_used;
    pthread_mutex_t write_buffer_lock;
    
    /* Vector index (simplified HNSW) */
    ai_engram_entry_t** vector_index;
    uint64_t vector_index_size;
    pthread_mutex_t index_lock;
    
    /* Statistics */
    gpuio_engram_stats_t stats;
    pthread_mutex_t stats_lock;
    
    /* Global lock */
    pthread_mutex_t lock;
    
    /* Entry count */
    uint64_t entry_count;
    
    /* Background thread for async writes */
    pthread_t write_thread;
    bool write_thread_running;
};

/* ============================================================================
 * Engram Internal Functions
 * ============================================================================ */

int ai_engram_init(struct ai_engram* engram, gpuio_ai_context_t ai_ctx,
                  const gpuio_engram_config_t* config);
void ai_engram_cleanup(struct ai_engram* engram);
ai_engram_entry_t* ai_engram_lookup(struct ai_engram* engram, uint64_t engram_id);
gpuio_error_t ai_engram_store_entry(struct ai_engram* engram, ai_engram_entry_t* entry);
gpuio_error_t ai_engram_load_entry(struct ai_engram* engram, ai_engram_entry_t* entry);
gpuio_error_t ai_engram_vector_search(struct ai_engram* engram,
                                       const float* query, uint32_t dim,
                                       float threshold, int top_k,
                                       ai_engram_entry_t** results, int* num_results);
void* ai_engram_write_thread(void* arg);

/**
 * @brief Compute hash for engram lookup.
 * Uses SplitMix64 hash from common_utils.h.
 */
static inline uint64_t ai_engram_hash(uint64_t engram_id) {
    return gpuio_hash_splitmix64_bucket(engram_id, AI_ENGRAM_HASH_BUCKETS);
}

/**
 * @brief Compute cosine similarity between vectors.
 * Delegates to vector_ops.h implementation.
 */
static inline float ai_engram_similarity(const float* a, const float* b, uint32_t dim) {
    return vec_cosine_similarity_f32(a, b, dim);
}

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_INTERNAL_H */
