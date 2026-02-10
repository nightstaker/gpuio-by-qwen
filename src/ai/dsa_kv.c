/**
 * @file dsa_kv.c
 * @brief AI Extensions module - DSA KV Cache implementation
 * @version 1.1.0
 * 
 * DeepSeek Dynamic Sparse Attention (DSA) KV Cache implementation with
 * tiered storage (HBM/CXL/Remote), compression support, and sparsity
 * pattern management.
 * 
 * Refactored to use common utilities (LRU cache, hash functions).
 */

#include "ai_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * @brief Initialize DSA KV subsystem.
 */
int ai_dsa_kv_init(struct ai_dsa_kv* kv, gpuio_ai_context_t ai_ctx,
                   const gpuio_dsa_kv_config_t* config) {
    if (!kv || !config) return -1;
    
    kv->ai_ctx = ai_ctx;
    memcpy(&kv->config, config, sizeof(gpuio_dsa_kv_config_t));
    
    /* Initialize tiers */
    kv->hbm_tier.capacity = config->hbm_capacity;
    kv->hbm_tier.used = 0;
    kv->hbm_tier.storage = NULL;
    
    kv->cxl_tier.capacity = config->cxl_capacity;
    kv->cxl_tier.used = 0;
    kv->cxl_tier.storage = NULL;
    if (config->cxl_device_path) {
        kv->cxl_tier.cxl_device_path = strdup(config->cxl_device_path);
    }
    
    kv->remote_tier.capacity = config->remote_uri ? (1ULL << 40) : 0; /* 1TB default */
    kv->remote_tier.used = 0;
    kv->remote_tier.storage = NULL;
    if (config->remote_uri) {
        kv->remote_tier.remote_uri = strdup(config->remote_uri);
    }
    
    /* Allocate hash table */
    kv->hash_table = calloc(AI_DSA_KV_HASH_BUCKETS, sizeof(ai_kv_entry_t*));
    if (!kv->hash_table) return -1;
    
    /* Initialize locks */
    pthread_mutex_init(&kv->hash_lock, NULL);
    pthread_mutex_init(&kv->sparsity_lock, NULL);
    pthread_mutex_init(&kv->stats_lock, NULL);
    pthread_mutex_init(&kv->lock, NULL);
    
    /* Initialize sparsity patterns (all heads active by default) */
    for (int i = 0; i < AI_DSA_KV_MAX_LAYERS; i++) {
        kv->sparsity_patterns[i] = 0xFFFFFFFF;
    }
    
    /* Initialize LRU cache (using common utilities) */
    kv->lru_cache = lru_cache_create();
    if (!kv->lru_cache) {
        free(kv->hash_table);
        return -1;
    }
    
    /* Initialize statistics */
    memset(&kv->stats, 0, sizeof(gpuio_dsa_kv_stats_t));
    kv->entry_count = 0;
    
    AI_LOG_INFO(ai_ctx, "DSA KV initialized (HBM=%zuMB, CXL=%zuMB)",
                config->hbm_capacity / (1024*1024),
                config->cxl_capacity / (1024*1024));
    
    return 0;
}

/**
 * @brief Cleanup DSA KV subsystem.
 */
void ai_dsa_kv_cleanup(struct ai_dsa_kv* kv) {
    if (!kv) return;
    
    pthread_mutex_lock(&kv->lock);
    
    /* Free all entries via LRU cache destroy */
    lru_cache_destroy(kv->lru_cache, NULL, NULL);
    kv->lru_cache = NULL;
    
    /* Free hash table (entries already freed by LRU destroy) */
    free(kv->hash_table);
    kv->hash_table = NULL;
    
    if (kv->cxl_tier.cxl_device_path) {
        free(kv->cxl_tier.cxl_device_path);
    }
    if (kv->remote_tier.remote_uri) {
        free(kv->remote_tier.remote_uri);
    }
    
    pthread_mutex_unlock(&kv->lock);
    
    pthread_mutex_destroy(&kv->hash_lock);
    pthread_mutex_destroy(&kv->sparsity_lock);
    pthread_mutex_destroy(&kv->stats_lock);
    pthread_mutex_destroy(&kv->lock);
    
    AI_LOG_INFO(kv->ai_ctx, "DSA KV cleaned up");
}

/**
 * @brief Lookup a KV entry by position, layer, and head.
 */
ai_kv_entry_t* ai_dsa_kv_lookup(struct ai_dsa_kv* kv, uint64_t position,
                                 uint32_t layer_id, uint32_t head_id) {
    if (!kv) return NULL;
    
    uint64_t hash = ai_dsa_kv_hash(position, layer_id, head_id);
    
    pthread_mutex_lock(&kv->hash_lock);
    ai_kv_entry_t* entry = kv->hash_table[hash];
    while (entry) {
        if (entry->position == position &&
            entry->layer_id == layer_id &&
            entry->head_id == head_id) {
            pthread_mutex_lock(&entry->lock);
            entry->ref_count++;
            entry->access_count++;
            pthread_mutex_unlock(&entry->lock);
            pthread_mutex_unlock(&kv->hash_lock);
            
            /* Touch in LRU cache */
            lru_cache_touch(kv->lru_cache, (lru_entry_t*)entry);
            
            return entry;
        }
        entry = entry->hash_next;
    }
    pthread_mutex_unlock(&kv->hash_lock);
    
    return NULL;
}

/**
 * @brief Create a new DSA KV pool.
 */
gpuio_error_t gpuio_dsa_kv_pool_create(gpuio_ai_context_t ai_ctx,
                                        const gpuio_dsa_kv_config_t* config,
                                        gpuio_dsa_kv_pool_t* pool) {
    if (!ai_ctx || !config || !pool) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    gpuio_error_t err = ai_context_validate(ai_ctx, true, false, false);
    if (err != GPUIO_SUCCESS) return err;
    
    struct gpuio_ai_context* ai = (struct gpuio_ai_context*)ai_ctx;
    struct ai_dsa_kv* kv = ai->dsa_kv;
    
    /* Initialize if not already done */
    if (kv->entry_count == 0 && !kv->hash_table) {
        if (ai_dsa_kv_init(kv, ai_ctx, config) != 0) {
            return GPUIO_ERROR_NOMEM;
        }
    }
    
    *pool = (gpuio_dsa_kv_pool_t)kv;
    
    AI_LOG_INFO(ai_ctx, "DSA KV pool created");
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Destroy a DSA KV pool.
 */
gpuio_error_t gpuio_dsa_kv_pool_destroy(gpuio_dsa_kv_pool_t pool) {
    if (!pool) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_dsa_kv* kv = (struct ai_dsa_kv*)pool;
    
    ai_dsa_kv_cleanup(kv);
    memset(kv, 0, sizeof(struct ai_dsa_kv));
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Reset a DSA KV pool (clear all entries).
 */
gpuio_error_t gpuio_dsa_kv_pool_reset(gpuio_dsa_kv_pool_t pool) {
    if (!pool) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_dsa_kv* kv = (struct ai_dsa_kv*)pool;
    
    pthread_mutex_lock(&kv->lock);
    
    /* Free all entries via LRU cache destroy and recreate */
    lru_cache_destroy(kv->lru_cache, NULL, NULL);
    kv->lru_cache = lru_cache_create();
    
    /* Reset hash table */
    memset(kv->hash_table, 0, AI_DSA_KV_HASH_BUCKETS * sizeof(ai_kv_entry_t*));
    
    /* Reset stats */
    kv->hbm_tier.used = 0;
    kv->cxl_tier.used = 0;
    kv->remote_tier.used = 0;
    memset(&kv->stats, 0, sizeof(gpuio_dsa_kv_stats_t));
    kv->entry_count = 0;
    
    pthread_mutex_unlock(&kv->lock);
    
    AI_LOG_INFO(kv->ai_ctx, "DSA KV pool reset");
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Store a KV entry.
 */
gpuio_error_t gpuio_dsa_kv_store(gpuio_dsa_kv_pool_t pool,
                                  uint64_t position,
                                  uint32_t layer_id,
                                  uint32_t head_id,
                                  const void* data,
                                  size_t size,
                                  float importance_score,
                                  gpuio_stream_t stream) {
    (void)stream; /* Stream not used in stub implementation */
    
    if (!pool || !data || size == 0) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_dsa_kv* kv = (struct ai_dsa_kv*)pool;
    
    /* Check if entry already exists */
    ai_kv_entry_t* existing = ai_dsa_kv_lookup(kv, position, layer_id, head_id);
    if (existing) {
        /* Update existing entry */
        pthread_mutex_lock(&existing->lock);
        existing->importance_score = importance_score;
        existing->ref_count--;
        pthread_mutex_unlock(&existing->lock);
        return GPUIO_SUCCESS;
    }
    
    /* Create new entry */
    ai_kv_entry_t* entry = calloc(1, sizeof(ai_kv_entry_t));
    if (!entry) return GPUIO_ERROR_NOMEM;
    
    /* Initialize LRU entry fields */
    lru_entry_init((lru_entry_t*)entry);
    lru_entry_lock_init((lru_entry_t*)entry);
    
    entry->position = position;
    entry->layer_id = layer_id;
    entry->head_id = head_id;
    entry->importance_score = importance_score;
    entry->original_size = size;
    entry->access_count = 1;
    
    /* Determine tier based on importance */
    if (importance_score > 0.7f && kv->hbm_tier.used + size <= kv->hbm_tier.capacity) {
        entry->tier = GPUIO_KV_TIER_HBM;
    } else if (kv->cxl_tier.used + size <= kv->cxl_tier.capacity) {
        entry->tier = GPUIO_KV_TIER_CXL;
    } else {
        entry->tier = GPUIO_KV_TIER_REMOTE;
    }
    
    /* Apply compression if configured */
    entry->compression = kv->config.default_compression;
    if (entry->compression == GPUIO_KV_COMPRESS_FP16) {
        entry->compressed_size = size / 2;
    } else if (entry->compression == GPUIO_KV_COMPRESS_INT8) {
        entry->compressed_size = size / 4;
    } else if (entry->compression == GPUIO_KV_COMPRESS_4BIT) {
        entry->compressed_size = size / 8;
    } else {
        entry->compressed_size = size;
        entry->compression = GPUIO_KV_COMPRESS_NONE;
    }
    
    /* Allocate and copy data */
    entry->data = malloc(size);
    if (!entry->data) {
        pthread_mutex_destroy(&entry->lock);
        free(entry);
        return GPUIO_ERROR_NOMEM;
    }
    memcpy(entry->data, data, size);
    entry->size = size;
    
    /* Update tier usage */
    if (entry->tier == GPUIO_KV_TIER_HBM) {
        kv->hbm_tier.used += entry->compressed_size;
    } else if (entry->tier == GPUIO_KV_TIER_CXL) {
        kv->cxl_tier.used += entry->compressed_size;
    } else {
        kv->remote_tier.used += entry->compressed_size;
    }
    
    /* Add to hash table */
    uint64_t hash = ai_dsa_kv_hash(position, layer_id, head_id);
    pthread_mutex_lock(&kv->hash_lock);
    entry->hash_next = kv->hash_table[hash];
    kv->hash_table[hash] = entry;
    pthread_mutex_unlock(&kv->hash_lock);
    
    /* Add to LRU cache */
    lru_cache_add(kv->lru_cache, (lru_entry_t*)entry);
    
    pthread_mutex_lock(&kv->stats_lock);
    kv->entry_count++;
    if (entry->compression != GPUIO_KV_COMPRESS_NONE) {
        kv->stats.bytes_saved_by_compression += (size - entry->compressed_size);
    }
    pthread_mutex_unlock(&kv->stats_lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Load a KV entry.
 */
gpuio_error_t gpuio_dsa_kv_load(gpuio_dsa_kv_pool_t pool,
                                 uint64_t position,
                                 uint32_t layer_id,
                                 uint32_t head_id,
                                 gpuio_stream_t stream,
                                 gpuio_dsa_kv_entry_t* entry) {
    (void)stream;
    
    if (!pool || !entry) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_dsa_kv* kv = (struct ai_dsa_kv*)pool;
    
    /* Check sparsity pattern */
    pthread_mutex_lock(&kv->sparsity_lock);
    uint32_t pattern = kv->sparsity_patterns[layer_id % AI_DSA_KV_MAX_LAYERS];
    if (!(pattern & (1 << (head_id % 32)))) {
        pthread_mutex_unlock(&kv->sparsity_lock);
        return GPUIO_ERROR_NOT_FOUND; /* Head not active in sparsity pattern */
    }
    pthread_mutex_unlock(&kv->sparsity_lock);
    
    ai_kv_entry_t* found = ai_dsa_kv_lookup(kv, position, layer_id, head_id);
    
    if (found) {
        *entry = (gpuio_dsa_kv_entry_t)found;
        
        pthread_mutex_lock(&kv->stats_lock);
        kv->stats.total_hits++;
        uint64_t total = kv->stats.total_hits + kv->stats.total_misses;
        if (total > 0) {
            kv->stats.hit_rate = (double)kv->stats.total_hits / total;
        }
        pthread_mutex_unlock(&kv->stats_lock);
        
        return GPUIO_SUCCESS;
    }
    
    pthread_mutex_lock(&kv->stats_lock);
    kv->stats.total_misses++;
    uint64_t total = kv->stats.total_hits + kv->stats.total_misses;
    if (total > 0) {
        kv->stats.hit_rate = (double)kv->stats.total_hits / total;
    }
    pthread_mutex_unlock(&kv->stats_lock);
    
    return GPUIO_ERROR_NOT_FOUND;
}

/**
 * @brief Get data from a KV entry.
 */
gpuio_error_t gpuio_dsa_kv_get_data(gpuio_dsa_kv_entry_t entry,
                                     void** data,
                                     size_t* size) {
    if (!entry || !data || !size) return GPUIO_ERROR_INVALID_ARG;
    
    ai_kv_entry_t* e = (ai_kv_entry_t*)entry;
    
    pthread_mutex_lock(&e->lock);
    *data = e->data;
    *size = e->size;
    pthread_mutex_unlock(&e->lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Release a KV entry reference.
 */
gpuio_error_t gpuio_dsa_kv_release(gpuio_dsa_kv_entry_t entry) {
    if (!entry) return GPUIO_ERROR_INVALID_ARG;
    
    ai_kv_entry_t* e = (ai_kv_entry_t*)entry;
    
    pthread_mutex_lock(&e->lock);
    e->ref_count--;
    pthread_mutex_unlock(&e->lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Load a batch of KV entries.
 */
gpuio_error_t gpuio_dsa_kv_load_batch(gpuio_dsa_kv_pool_t pool,
                                       uint64_t* positions,
                                       uint32_t* layer_ids,
                                       uint32_t* head_ids,
                                       int num_entries,
                                       gpuio_stream_t stream,
                                       gpuio_dsa_kv_entry_t* entries) {
    if (!pool || !positions || !entries || num_entries <= 0) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    gpuio_error_t first_error = GPUIO_SUCCESS;
    
    for (int i = 0; i < num_entries; i++) {
        gpuio_error_t err = gpuio_dsa_kv_load(pool, positions[i], 
                                               layer_ids ? layer_ids[i] : 0,
                                               head_ids ? head_ids[i] : 0,
                                               stream, &entries[i]);
        if (err != GPUIO_SUCCESS && first_error == GPUIO_SUCCESS) {
            first_error = err;
        }
    }
    
    return first_error;
}

/**
 * @brief Set sparsity pattern for a layer.
 */
gpuio_error_t gpuio_dsa_kv_set_sparsity_pattern(gpuio_dsa_kv_pool_t pool,
                                                 uint32_t layer_id,
                                                 uint32_t sparsity_mask) {
    if (!pool) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_dsa_kv* kv = (struct ai_dsa_kv*)pool;
    
    if (layer_id >= AI_DSA_KV_MAX_LAYERS) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    pthread_mutex_lock(&kv->sparsity_lock);
    kv->sparsity_patterns[layer_id] = sparsity_mask;
    pthread_mutex_unlock(&kv->sparsity_lock);
    
    AI_LOG_DEBUG(kv->ai_ctx, "Set sparsity pattern for layer %u: 0x%08X",
                 layer_id, sparsity_mask);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Promote an entry to a higher tier.
 *
 * Moves a KV entry from its current tier to a faster tier (e.g., CXL -> HBM).
 */
gpuio_error_t ai_dsa_kv_promote_entry(struct ai_dsa_kv* kv, ai_kv_entry_t* entry,
                                       gpuio_kv_tier_t target_tier) {
    if (!kv || !entry) return GPUIO_ERROR_INVALID_ARG;
    
    if (target_tier <= entry->tier) {
        return GPUIO_SUCCESS;  /* Already at or above target tier */
    }
    
    pthread_mutex_lock(&entry->lock);
    
    size_t size = entry->size;
    void* new_data = NULL;
    
    /* Check if we can allocate in target tier */
    if (target_tier == GPUIO_KV_TIER_HBM) {
        pthread_mutex_lock(&kv->hbm_tier.lock);
        if (kv->hbm_tier.used + size <= kv->hbm_tier.capacity) {
            new_data = malloc(size);
            if (new_data) {
                memcpy(new_data, entry->data, size);
                kv->hbm_tier.used += size;
            }
        }
        pthread_mutex_unlock(&kv->hbm_tier.lock);
    } else if (target_tier == GPUIO_KV_TIER_CXL) {
        pthread_mutex_lock(&kv->cxl_tier.lock);
        if (kv->cxl_tier.used + size <= kv->cxl_tier.capacity) {
            new_data = malloc(size);
            if (new_data) {
                memcpy(new_data, entry->data, size);
                kv->cxl_tier.used += size;
            }
        }
        pthread_mutex_unlock(&kv->cxl_tier.lock);
    }
    
    if (new_data) {
        /* Free old data and update */
        if (entry->tier == GPUIO_KV_TIER_HBM) {
            pthread_mutex_lock(&kv->hbm_tier.lock);
            kv->hbm_tier.used -= size;
            pthread_mutex_unlock(&kv->hbm_tier.lock);
        } else if (entry->tier == GPUIO_KV_TIER_CXL) {
            pthread_mutex_lock(&kv->cxl_tier.lock);
            kv->cxl_tier.used -= size;
            pthread_mutex_unlock(&kv->cxl_tier.lock);
        }
        
        free(entry->data);
        entry->data = new_data;
        entry->tier = target_tier;
    }
    
    pthread_mutex_unlock(&entry->lock);
    
    return new_data ? GPUIO_SUCCESS : GPUIO_ERROR_NOMEM;
}

/**
 * @brief Evict entries to make room in a tier.
 *
 * Removes least recently used entries from the specified tier until
 * the requested amount of space is available.
 */
gpuio_error_t ai_dsa_kv_evict_entries(struct ai_dsa_kv* kv, size_t needed_space,
                                       gpuio_kv_tier_t tier) {
    if (!kv) return GPUIO_ERROR_INVALID_ARG;
    
    size_t evicted = 0;
    
    /* Use LRU cache eviction */
    lru_cache_lock(kv->lru_cache);
    lru_entry_t* lru_entry = lru_cache_get_lru(kv->lru_cache);
    
    while (lru_entry && evicted < needed_space) {
        ai_kv_entry_t* entry = (ai_kv_entry_t*)lru_entry;
        lru_entry_t* prev = lru_entry_prev(lru_entry);
        
        lru_entry_lock(lru_entry);
        if (entry->tier == tier && entry->ref_count == 0) {
            size_t entry_size = entry->size;
            lru_entry_unlock(lru_entry);
            
            /* Remove from hash table */
            uint64_t hash = ai_dsa_kv_hash(entry->position, entry->layer_id,
                                           entry->head_id);
            pthread_mutex_lock(&kv->hash_lock);
            ai_kv_entry_t** current = &kv->hash_table[hash];
            while (*current) {
                if (*current == entry) {
                    *current = entry->hash_next;
                    break;
                }
                current = &(*current)->hash_next;
            }
            pthread_mutex_unlock(&kv->hash_lock);
            
            /* Remove from LRU cache */
            lru_cache_remove(kv->lru_cache, lru_entry);
            
            /* Free resources */
            if (entry->data) free(entry->data);
            pthread_mutex_destroy(&entry->lock);
            free(entry);
            
            kv->entry_count--;
            evicted += entry_size;
            
            /* Update tier usage */
            if (tier == GPUIO_KV_TIER_HBM) {
                pthread_mutex_lock(&kv->hbm_tier.lock);
                kv->hbm_tier.used -= entry_size;
                pthread_mutex_unlock(&kv->hbm_tier.lock);
            } else if (tier == GPUIO_KV_TIER_CXL) {
                pthread_mutex_lock(&kv->cxl_tier.lock);
                kv->cxl_tier.used -= entry_size;
                pthread_mutex_unlock(&kv->cxl_tier.lock);
            }
        } else {
            lru_entry_unlock(lru_entry);
        }
        
        lru_entry = prev;
    }
    
    lru_cache_unlock(kv->lru_cache);
    
    return evicted >= needed_space ? GPUIO_SUCCESS : GPUIO_ERROR_NOMEM;
}

/**
 * @brief Compact the KV cache (reclaim space, consolidate).
 */
gpuio_error_t gpuio_dsa_kv_compact(gpuio_dsa_kv_pool_t pool,
                                    gpuio_stream_t stream) {
    (void)stream;
    
    if (!pool) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_dsa_kv* kv = (struct ai_dsa_kv*)pool;
    
    pthread_mutex_lock(&kv->lock);
    
    /* Use LRU cache iteration for eviction */
    lru_cache_lock(kv->lru_cache);
    lru_entry_t* lru_entry = lru_cache_get_lru(kv->lru_cache);
    
    while (lru_entry) {
        ai_kv_entry_t* entry = (ai_kv_entry_t*)lru_entry;
        lru_entry_t* prev = lru_entry_prev(lru_entry);
        
        lru_entry_lock(lru_entry);
        if (entry->ref_count == 0 && 
            entry->importance_score < kv->config.compression_threshold) {
            /* Evict this entry */
            lru_entry_unlock(lru_entry);
            
            /* Remove from hash table */
            uint64_t hash = ai_dsa_kv_hash(entry->position, entry->layer_id, 
                                           entry->head_id);
            pthread_mutex_lock(&kv->hash_lock);
            ai_kv_entry_t** current = &kv->hash_table[hash];
            while (*current) {
                if (*current == entry) {
                    *current = entry->hash_next;
                    break;
                }
                current = &(*current)->hash_next;
            }
            pthread_mutex_unlock(&kv->hash_lock);
            
            /* Remove from LRU cache */
            lru_cache_remove(kv->lru_cache, lru_entry);
            
            /* Free data */
            if (entry->data) free(entry->data);
            pthread_mutex_destroy(&entry->lock);
            free(entry);
            
            kv->entry_count--;
        } else {
            lru_entry_unlock(lru_entry);
        }
        
        lru_entry = prev;
    }
    
    lru_cache_unlock(kv->lru_cache);
    
    pthread_mutex_unlock(&kv->lock);
    
    AI_LOG_INFO(kv->ai_ctx, "KV cache compacted, %zu entries remaining", 
                kv->entry_count);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Get DSA KV statistics.
 */
gpuio_error_t gpuio_dsa_kv_get_stats(gpuio_dsa_kv_pool_t pool,
                                      gpuio_dsa_kv_stats_t* stats) {
    if (!pool || !stats) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_dsa_kv* kv = (struct ai_dsa_kv*)pool;
    
    pthread_mutex_lock(&kv->stats_lock);
    memcpy(stats, &kv->stats, sizeof(gpuio_dsa_kv_stats_t));
    
    /* Count entries per tier */
    stats->entries_in_hbm = 0;
    stats->entries_in_cxl = 0;
    stats->entries_in_remote = 0;
    
    pthread_mutex_lock(&kv->hash_lock);
    for (int i = 0; i < AI_DSA_KV_HASH_BUCKETS; i++) {
        ai_kv_entry_t* entry = kv->hash_table[i];
        while (entry) {
            if (entry->tier == GPUIO_KV_TIER_HBM) {
                stats->entries_in_hbm++;
            } else if (entry->tier == GPUIO_KV_TIER_CXL) {
                stats->entries_in_cxl++;
            } else {
                stats->entries_in_remote++;
            }
            entry = entry->hash_next;
        }
    }
    pthread_mutex_unlock(&kv->hash_lock);
    
    pthread_mutex_unlock(&kv->stats_lock);
    
    return GPUIO_SUCCESS;
}
