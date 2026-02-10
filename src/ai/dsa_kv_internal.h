/**
 * @file dsa_kv_internal.h
 * @brief AI Extensions module - DSA KV Cache internal structures
 * @version 1.1.0
 * 
 * Internal structures and functions for the DeepSeek Dynamic Sparse Attention
 * (DSA) KV Cache implementation with tiered storage.
 */

#ifndef DSA_KV_INTERNAL_H
#define DSA_KV_INTERNAL_H

#include "ai_internal.h"
#include "lru_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration Constants
 * ============================================================================ */

#define AI_DSA_KV_MAX_LAYERS     128
#define AI_DSA_KV_MAX_HEADS      256
#define AI_DSA_KV_HASH_BUCKETS   1024

/* ============================================================================
 * DSA KV Entry
 * ============================================================================ */

/**
 * @brief DSA KV cache entry with embedded LRU fields.
 */
typedef struct ai_kv_entry {
    /* Embedded LRU fields (must be first for lru_cache compatibility) */
    LRU_ENTRY_FIELDS;
    
    /* Key fields */
    uint64_t position;
    uint32_t layer_id;
    uint32_t head_id;
    
    /* Data */
    void* data;
    size_t size;
    gpuio_kv_tier_t tier;
    
    /* Metadata */
    float importance_score;
    uint32_t sparsity_pattern;
    gpuio_kv_compression_t compression;
    uint64_t compressed_size;
    uint64_t original_size;
    
    /* Access tracking */
    uint64_t access_count;
    
    /* Hash chain */
    struct ai_kv_entry* hash_next;
} ai_kv_entry_t;

/* ============================================================================
 * DSA KV Tier Storage
 * ============================================================================ */

typedef struct ai_kv_tier {
    size_t capacity;
    size_t used;
    void* storage;
    char* cxl_device_path;
    char* remote_uri;
    pthread_mutex_t lock;
} ai_kv_tier_t;

/* ============================================================================
 * DSA KV Cache Structure
 * ============================================================================ */

struct ai_dsa_kv {
    gpuio_ai_context_t ai_ctx;
    gpuio_dsa_kv_config_t config;
    
    /* Tiers */
    ai_kv_tier_t hbm_tier;
    ai_kv_tier_t cxl_tier;
    ai_kv_tier_t remote_tier;
    
    /* Hash table for entry lookup */
    ai_kv_entry_t** hash_table;
    pthread_mutex_t hash_lock;
    
    /* LRU cache (replaces manual LRU list management) */
    lru_cache_t* lru_cache;
    
    /* Sparsity patterns per layer */
    uint32_t sparsity_patterns[AI_DSA_KV_MAX_LAYERS];
    pthread_mutex_t sparsity_lock;
    
    /* Statistics */
    gpuio_dsa_kv_stats_t stats;
    pthread_mutex_t stats_lock;
    
    /* Global lock */
    pthread_mutex_t lock;
    
    /* Entry count */
    uint64_t entry_count;
};

/* ============================================================================
 * DSA KV Internal Functions
 * ============================================================================ */

int ai_dsa_kv_init(struct ai_dsa_kv* kv, gpuio_ai_context_t ai_ctx,
                   const gpuio_dsa_kv_config_t* config);
void ai_dsa_kv_cleanup(struct ai_dsa_kv* kv);
ai_kv_entry_t* ai_dsa_kv_lookup(struct ai_dsa_kv* kv, uint64_t position,
                                 uint32_t layer_id, uint32_t head_id);
gpuio_error_t ai_dsa_kv_promote_entry(struct ai_dsa_kv* kv, ai_kv_entry_t* entry,
                                       gpuio_kv_tier_t target_tier);
gpuio_error_t ai_dsa_kv_evict_entries(struct ai_dsa_kv* kv, size_t needed_space,
                                       gpuio_kv_tier_t tier);

/**
 * @brief Compute hash for KV entry lookup.
 * Uses FNV-1a hash from common_utils.h.
 */
static inline uint64_t ai_dsa_kv_hash(uint64_t position, uint32_t layer_id, uint32_t head_id) {
    return gpuio_hash_to_bucket_u64(
        gpuio_hash_fnv1a_3u64(position, layer_id, head_id),
        AI_DSA_KV_HASH_BUCKETS
    );
}

#ifdef __cplusplus
}
#endif

#endif /* DSA_KV_INTERNAL_H */
