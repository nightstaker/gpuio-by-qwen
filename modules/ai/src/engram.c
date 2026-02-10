/**
 * @file engram.c
 * @brief AI Extensions module - Engram Memory implementation
 * @version 1.1.0
 * 
 * DeepSeek Engram Memory Architecture for petabyte-scale external memory
 * with learned addressing and GPU-initiated memory operations.
 * 
 * Refactored to use common utilities (LRU cache, hash functions, vector_ops).
 */

#include "ai_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

/**
 * @brief Background write thread for async engram writes.
 */
void* ai_engram_write_thread(void* arg) {
    struct ai_engram* engram = (struct ai_engram*)arg;
    
    while (engram->write_thread_running) {
        usleep(engram->config.flush_interval_ms * 1000);
        
        if (!engram->write_thread_running) break;
        
        /* Flush write buffer */
        pthread_mutex_lock(&engram->write_buffer_lock);
        if (engram->write_buffer_used > 0) {
            /* In production, this would write to remote storage */
            engram->write_buffer_used = 0;
        }
        pthread_mutex_unlock(&engram->write_buffer_lock);
    }
    
    return NULL;
}

/**
 * @brief Initialize engram memory subsystem.
 */
int ai_engram_init(struct ai_engram* engram, gpuio_ai_context_t ai_ctx,
                  const gpuio_engram_config_t* config) {
    if (!engram || !config) return -1;
    
    engram->ai_ctx = ai_ctx;
    memcpy(&engram->config, config, sizeof(gpuio_engram_config_t));
    
    /* Initialize tiers */
    engram->hbm_tier.capacity = config->hbm_capacity;
    engram->hbm_tier.used = 0;
    engram->hbm_tier.storage = malloc(config->hbm_capacity);
    if (!engram->hbm_tier.storage && config->hbm_capacity > 0) return -1;
    pthread_mutex_init(&engram->hbm_tier.lock, NULL);
    
    engram->cxl_tier.capacity = config->cxl_capacity;
    engram->cxl_tier.used = 0;
    engram->cxl_tier.storage = malloc(config->cxl_capacity);
    if (!engram->cxl_tier.storage && config->cxl_capacity > 0) {
        free(engram->hbm_tier.storage);
        return -1;
    }
    pthread_mutex_init(&engram->cxl_tier.lock, NULL);
    
    engram->remote_tier.capacity = config->remote_archive_uri ? 
                                   (100ULL << 30) : 0; /* 100GB default */
    engram->remote_tier.used = 0;
    engram->remote_tier.storage = NULL;
    pthread_mutex_init(&engram->remote_tier.lock, NULL);
    
    /* Allocate hash table */
    engram->hash_table = calloc(AI_ENGRAM_HASH_BUCKETS, sizeof(ai_engram_entry_t*));
    if (!engram->hash_table) {
        free(engram->hbm_tier.storage);
        free(engram->cxl_tier.storage);
        return -1;
    }
    
    /* Allocate vector index */
    engram->vector_index_size = 1024;
    engram->vector_index = calloc(engram->vector_index_size, sizeof(ai_engram_entry_t*));
    if (!engram->vector_index) {
        free(engram->hash_table);
        free(engram->hbm_tier.storage);
        free(engram->cxl_tier.storage);
        return -1;
    }
    
    /* Initialize write buffer */
    if (config->async_writes && config->write_buffer_size > 0) {
        engram->write_buffer = malloc(config->write_buffer_size);
        if (!engram->write_buffer) {
            free(engram->vector_index);
            free(engram->hash_table);
            free(engram->hbm_tier.storage);
            free(engram->cxl_tier.storage);
            return -1;
        }
        engram->write_buffer_used = 0;
    }
    
    /* Initialize locks */
    pthread_mutex_init(&engram->hash_lock, NULL);
    pthread_mutex_init(&engram->index_lock, NULL);
    pthread_mutex_init(&engram->stats_lock, NULL);
    pthread_mutex_init(&engram->write_buffer_lock, NULL);
    pthread_mutex_init(&engram->lock, NULL);
    
    /* Initialize LRU cache (using common utilities) */
    engram->lru_cache = lru_cache_create();
    if (!engram->lru_cache) {
        free(engram->write_buffer);
        free(engram->vector_index);
        free(engram->hash_table);
        free(engram->hbm_tier.storage);
        free(engram->cxl_tier.storage);
        return -1;
    }
    
    /* Initialize statistics */
    memset(&engram->stats, 0, sizeof(gpuio_engram_stats_t));
    engram->entry_count = 0;
    
    /* Start background thread for async writes */
    if (config->async_writes) {
        engram->write_thread_running = true;
        if (pthread_create(&engram->write_thread, NULL, 
                          ai_engram_write_thread, engram) != 0) {
            engram->write_thread_running = false;
        }
    }
    
    AI_LOG_INFO(ai_ctx, "Engram memory initialized (HBM=%zuMB, CXL=%zuMB, async=%d)",
                config->hbm_capacity / (1024*1024),
                config->cxl_capacity / (1024*1024),
                config->async_writes);
    
    return 0;
}

/**
 * @brief Cleanup engram memory subsystem.
 */
void ai_engram_cleanup(struct ai_engram* engram) {
    if (!engram) return;
    
    /* Stop background thread */
    if (engram->write_thread_running) {
        engram->write_thread_running = false;
        pthread_join(engram->write_thread, NULL);
    }
    
    pthread_mutex_lock(&engram->lock);
    
    /* Free all entries via LRU cache destroy */
    lru_cache_destroy(engram->lru_cache, NULL, NULL);
    engram->lru_cache = NULL;
    
    /* Free hash table (entries already freed) */
    free(engram->hash_table);
    engram->hash_table = NULL;
    
    free(engram->vector_index);
    free(engram->hbm_tier.storage);
    free(engram->cxl_tier.storage);
    free(engram->write_buffer);
    
    pthread_mutex_unlock(&engram->lock);
    
    pthread_mutex_destroy(&engram->hash_lock);
    pthread_mutex_destroy(&engram->index_lock);
    pthread_mutex_destroy(&engram->stats_lock);
    pthread_mutex_destroy(&engram->write_buffer_lock);
    pthread_mutex_destroy(&engram->hbm_tier.lock);
    pthread_mutex_destroy(&engram->cxl_tier.lock);
    pthread_mutex_destroy(&engram->remote_tier.lock);
    pthread_mutex_destroy(&engram->lock);
    
    AI_LOG_INFO(engram->ai_ctx, "Engram memory cleaned up");
}

/**
 * @brief Lookup an engram by ID.
 */
ai_engram_entry_t* ai_engram_lookup(struct ai_engram* engram, uint64_t engram_id) {
    if (!engram) return NULL;
    
    uint64_t hash = ai_engram_hash(engram_id);
    
    pthread_mutex_lock(&engram->hash_lock);
    ai_engram_entry_t* entry = engram->hash_table[hash];
    while (entry) {
        if (entry->engram_id == engram_id) {
            pthread_mutex_lock(&entry->lock);
            entry->ref_count++;
            entry->access_count++;
            pthread_mutex_unlock(&entry->lock);
            pthread_mutex_unlock(&engram->hash_lock);
            
            /* Touch in LRU cache */
            lru_cache_touch(engram->lru_cache, (lru_entry_t*)entry);
            
            pthread_mutex_lock(&engram->stats_lock);
            engram->stats.cache_hits++;
            pthread_mutex_unlock(&engram->stats_lock);
            
            return entry;
        }
        entry = entry->hash_next;
    }
    pthread_mutex_unlock(&engram->hash_lock);
    
    return NULL;
}

/**
 * @brief Vector similarity search for engrams.
 * Uses vec_cosine_similarity_f32 from vector_ops.h
 */
gpuio_error_t ai_engram_vector_search(struct ai_engram* engram,
                                       const float* query, uint32_t dim,
                                       float threshold, int top_k,
                                       ai_engram_entry_t** results, int* num_results) {
    if (!engram || !query || !results || !num_results) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    uint64_t start_time = gpuio_get_time_us();
    
    typedef struct {
        ai_engram_entry_t* entry;
        float similarity;
    } score_t;
    
    score_t* scores = calloc(engram->entry_count + 1, sizeof(score_t));
    if (!scores) return GPUIO_ERROR_NOMEM;
    
    int count = 0;
    
    /* Search through all entries */
    pthread_mutex_lock(&engram->hash_lock);
    for (int i = 0; i < AI_ENGRAM_HASH_BUCKETS; i++) {
        ai_engram_entry_t* entry = engram->hash_table[i];
        while (entry) {
            if (entry->embedding && entry->embedding_dim == dim) {
                /* Use common vector_ops for similarity */
                float sim = vec_cosine_similarity_f32(query, entry->embedding, dim);
                if (sim >= threshold) {
                    scores[count].entry = entry;
                    scores[count].similarity = sim;
                    count++;
                }
            }
            entry = entry->hash_next;
        }
    }
    pthread_mutex_unlock(&engram->hash_lock);
    
    /* Sort by similarity (descending) using common utility */
    qsort(scores, (size_t)count, sizeof(score_t), 
          (int (*)(const void*, const void*))vec_result_compare_desc);
    
    /* Return top-k */
    int result_count = (count < top_k) ? count : top_k;
    for (int i = 0; i < result_count; i++) {
        results[i] = scores[i].entry;
        pthread_mutex_lock(&results[i]->lock);
        results[i]->ref_count++;
        pthread_mutex_unlock(&results[i]->lock);
    }
    *num_results = result_count;
    
    free(scores);
    
    /* Update statistics */
    uint64_t latency = gpuio_get_time_us() - start_time;
    pthread_mutex_lock(&engram->stats_lock);
    engram->stats.total_queries++;
    /* Update average latency */
    double total_latency = engram->stats.avg_query_latency_us * (engram->stats.total_queries - 1);
    engram->stats.avg_query_latency_us = (total_latency + latency) / engram->stats.total_queries;
    pthread_mutex_unlock(&engram->stats_lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Create an engram pool.
 */
gpuio_error_t gpuio_engram_pool_create(gpuio_ai_context_t ai_ctx,
                                        const gpuio_engram_config_t* config,
                                        gpuio_engram_pool_t* pool) {
    if (!ai_ctx || !config || !pool) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    gpuio_error_t err = ai_context_validate(ai_ctx, false, true, false);
    if (err != GPUIO_SUCCESS) return err;
    
    struct gpuio_ai_context* ai = (struct gpuio_ai_context*)ai_ctx;
    struct ai_engram* engram = ai->engram;
    
    if (engram->entry_count == 0 && !engram->hash_table) {
        if (ai_engram_init(engram, ai_ctx, config) != 0) {
            return GPUIO_ERROR_NOMEM;
        }
    }
    
    *pool = (gpuio_engram_pool_t)engram;
    
    AI_LOG_INFO(ai_ctx, "Engram pool created");
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Destroy an engram pool.
 */
gpuio_error_t gpuio_engram_pool_destroy(gpuio_engram_pool_t pool) {
    if (!pool) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    
    ai_engram_cleanup(engram);
    memset(engram, 0, sizeof(struct ai_engram));
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Write an engram.
 */
gpuio_error_t gpuio_engram_write(gpuio_engram_pool_t pool,
                                  const gpuio_engram_t* engram_data,
                                  gpuio_stream_t stream) {
    (void)stream;
    
    if (!pool || !engram_data) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    
    /* Check if engram already exists */
    ai_engram_entry_t* existing = ai_engram_lookup(engram, engram_data->engram_id);
    if (existing) {
        pthread_mutex_lock(&existing->lock);
        existing->version = engram_data->version;
        existing->importance_score = engram_data->importance_score;
        existing->ref_count--;
        pthread_mutex_unlock(&existing->lock);
        return GPUIO_SUCCESS;
    }
    
    /* Create new entry */
    ai_engram_entry_t* entry = calloc(1, sizeof(ai_engram_entry_t));
    if (!entry) return GPUIO_ERROR_NOMEM;
    
    /* Initialize LRU entry fields */
    lru_entry_init((lru_entry_t*)entry);
    lru_entry_lock_init((lru_entry_t*)entry);
    
    entry->engram_id = engram_data->engram_id;
    entry->version = engram_data->version;
    entry->size = engram_data->size;
    entry->embedding_dim = engram_data->embedding_dim;
    entry->creation_timestamp = engram_data->creation_timestamp;
    entry->importance_score = engram_data->importance_score;
    entry->tier = engram_data->tier;
    
    /* Allocate and copy data */
    if (engram_data->size > 0 && engram_data->data) {
        entry->data = malloc(engram_data->size);
        if (!entry->data) {
            pthread_mutex_destroy(&entry->lock);
            free(entry);
            return GPUIO_ERROR_NOMEM;
        }
        memcpy(entry->data, engram_data->data, engram_data->size);
    }
    
    /* Allocate and copy embedding */
    if (engram_data->embedding_dim > 0 && engram_data->embedding) {
        entry->embedding = malloc(engram_data->embedding_dim * sizeof(float));
        if (!entry->embedding) {
            free(entry->data);
            pthread_mutex_destroy(&entry->lock);
            free(entry);
            return GPUIO_ERROR_NOMEM;
        }
        memcpy(entry->embedding, engram_data->embedding, 
               engram_data->embedding_dim * sizeof(float));
    }
    
    /* Store in appropriate tier */
    if (entry->tier == GPUIO_ENGRAM_TIER_HBM) {
        pthread_mutex_lock(&engram->hbm_tier.lock);
        if (engram->hbm_tier.used + entry->size <= engram->hbm_tier.capacity) {
            entry->storage_offset = engram->hbm_tier.used;
            engram->hbm_tier.used += entry->size;
        } else {
            entry->tier = GPUIO_ENGRAM_TIER_CXL;
        }
        pthread_mutex_unlock(&engram->hbm_tier.lock);
    }
    
    if (entry->tier == GPUIO_ENGRAM_TIER_CXL) {
        pthread_mutex_lock(&engram->cxl_tier.lock);
        if (engram->cxl_tier.used + entry->size <= engram->cxl_tier.capacity) {
            entry->storage_offset = engram->cxl_tier.used;
            engram->cxl_tier.used += entry->size;
        } else {
            entry->tier = GPUIO_ENGRAM_TIER_REMOTE;
        }
        pthread_mutex_unlock(&engram->cxl_tier.lock);
    }
    
    if (entry->tier == GPUIO_ENGRAM_TIER_REMOTE) {
        pthread_mutex_lock(&engram->remote_tier.lock);
        entry->storage_offset = engram->remote_tier.used;
        engram->remote_tier.used += entry->size;
        pthread_mutex_unlock(&engram->remote_tier.lock);
    }
    
    /* Add to hash table */
    uint64_t hash = ai_engram_hash(entry->engram_id);
    pthread_mutex_lock(&engram->hash_lock);
    entry->hash_next = engram->hash_table[hash];
    engram->hash_table[hash] = entry;
    pthread_mutex_unlock(&engram->hash_lock);
    
    /* Add to LRU cache */
    lru_cache_add(engram->lru_cache, (lru_entry_t*)entry);
    
    pthread_mutex_lock(&engram->stats_lock);
    engram->entry_count++;
    pthread_mutex_unlock(&engram->stats_lock);
    
    /* Async write to remote if enabled */
    if (engram->config.async_writes && entry->tier == GPUIO_ENGRAM_TIER_REMOTE) {
        pthread_mutex_lock(&engram->write_buffer_lock);
        /* In production, queue for background write */
        pthread_mutex_unlock(&engram->write_buffer_lock);
    }
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Read an engram.
 */
gpuio_error_t gpuio_engram_read(gpuio_engram_pool_t pool,
                                 uint64_t engram_id,
                                 gpuio_stream_t stream,
                                 gpuio_engram_handle_t* handle) {
    (void)stream;
    
    if (!pool || !handle) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    
    ai_engram_entry_t* entry = ai_engram_lookup(engram, engram_id);
    
    if (entry) {
        *handle = (gpuio_engram_handle_t)entry;
        return GPUIO_SUCCESS;
    }
    
    return GPUIO_ERROR_NOT_FOUND;
}

/**
 * @brief Query engrams by vector similarity.
 */
gpuio_error_t gpuio_engram_query(gpuio_engram_pool_t pool,
                                  const float* query_embedding,
                                  float similarity_threshold,
                                  int top_k,
                                  gpuio_stream_t stream,
                                  gpuio_engram_handle_t* results,
                                  int* num_results) {
    (void)stream;
    
    if (!pool || !query_embedding || !results || !num_results) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    uint32_t dim = engram->config.embedding_dim;
    
    ai_engram_entry_t** entries = calloc(top_k, sizeof(ai_engram_entry_t*));
    if (!entries) return GPUIO_ERROR_NOMEM;
    
    gpuio_error_t err = ai_engram_vector_search(engram, query_embedding, dim,
                                                 similarity_threshold, top_k,
                                                 entries, num_results);
    
    if (err == GPUIO_SUCCESS) {
        for (int i = 0; i < *num_results; i++) {
            results[i] = (gpuio_engram_handle_t)entries[i];
        }
    }
    
    free(entries);
    return err;
}

/**
 * @brief Get data from an engram handle.
 */
gpuio_error_t gpuio_engram_get_data(gpuio_engram_handle_t handle,
                                     void** data,
                                     size_t* size) {
    if (!handle || !data || !size) return GPUIO_ERROR_INVALID_ARG;
    
    ai_engram_entry_t* entry = (ai_engram_entry_t*)handle;
    
    pthread_mutex_lock(&entry->lock);
    *data = entry->data;
    *size = entry->size;
    pthread_mutex_unlock(&entry->lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Release an engram handle.
 */
gpuio_error_t gpuio_engram_release(gpuio_engram_handle_t handle) {
    if (!handle) return GPUIO_ERROR_INVALID_ARG;
    
    ai_engram_entry_t* entry = (ai_engram_entry_t*)handle;
    
    pthread_mutex_lock(&entry->lock);
    entry->ref_count--;
    pthread_mutex_unlock(&entry->lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Write a batch of engrams.
 */
gpuio_error_t gpuio_engram_write_batch(gpuio_engram_pool_t pool,
                                        const gpuio_engram_t* engrams,
                                        int num_engrams,
                                        gpuio_stream_t stream) {
    if (!pool || !engrams || num_engrams <= 0) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    gpuio_error_t first_error = GPUIO_SUCCESS;
    
    for (int i = 0; i < num_engrams && i < AI_ENGRAM_MAX_BATCH; i++) {
        gpuio_error_t err = gpuio_engram_write(pool, &engrams[i], stream);
        if (err != GPUIO_SUCCESS && first_error == GPUIO_SUCCESS) {
            first_error = err;
        }
    }
    
    return first_error;
}

/**
 * @brief Query a batch of embeddings.
 */
gpuio_error_t gpuio_engram_query_batch(gpuio_engram_pool_t pool,
                                        const float** query_embeddings,
                                        int num_queries,
                                        int top_k_per_query,
                                        gpuio_stream_t stream,
                                        gpuio_engram_handle_t** results,
                                        int** num_results_per_query) {
    (void)stream;
    
    if (!pool || !query_embeddings || num_queries <= 0 || !results) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    uint32_t dim = engram->config.embedding_dim;
    
    *results = calloc(num_queries * top_k_per_query, sizeof(gpuio_engram_handle_t));
    *num_results_per_query = calloc(num_queries, sizeof(int));
    
    if (!*results || !*num_results_per_query) {
        free(*results);
        free(*num_results_per_query);
        return GPUIO_ERROR_NOMEM;
    }
    
    for (int i = 0; i < num_queries; i++) {
        ai_engram_entry_t** entries = calloc(top_k_per_query, sizeof(ai_engram_entry_t*));
        if (!entries) continue;
        
        int count = 0;
        ai_engram_vector_search(engram, query_embeddings[i], dim, 0.0f, 
                                top_k_per_query, entries, &count);
        
        (*num_results_per_query)[i] = count;
        for (int j = 0; j < count; j++) {
            (*results)[i * top_k_per_query + j] = (gpuio_engram_handle_t)entries[j];
        }
        
        free(entries);
    }
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Store an engram entry to its assigned tier.
 * 
 * Handles tiered storage allocation and data persistence.
 */
gpuio_error_t ai_engram_store_entry(struct ai_engram* engram, ai_engram_entry_t* entry) {
    if (!engram || !entry) return GPUIO_ERROR_INVALID_ARG;
    
    pthread_mutex_lock(&entry->lock);
    
    /* Try to store in HBM first if that's the target tier */
    if (entry->tier == GPUIO_ENGRAM_TIER_HBM) {
        pthread_mutex_lock(&engram->hbm_tier.lock);
        if (engram->hbm_tier.used + entry->size <= engram->hbm_tier.capacity) {
            entry->storage_offset = engram->hbm_tier.used;
            if (entry->data && engram->hbm_tier.storage) {
                memcpy((char*)engram->hbm_tier.storage + entry->storage_offset,
                       entry->data, entry->size);
            }
            engram->hbm_tier.used += entry->size;
            pthread_mutex_unlock(&engram->hbm_tier.lock);
            pthread_mutex_unlock(&entry->lock);
            return GPUIO_SUCCESS;
        }
        pthread_mutex_unlock(&engram->hbm_tier.lock);
        /* Fall through to CXL */
        entry->tier = GPUIO_ENGRAM_TIER_CXL;
    }
    
    /* Try CXL tier */
    if (entry->tier == GPUIO_ENGRAM_TIER_CXL) {
        pthread_mutex_lock(&engram->cxl_tier.lock);
        if (engram->cxl_tier.used + entry->size <= engram->cxl_tier.capacity) {
            entry->storage_offset = engram->cxl_tier.used;
            if (entry->data && engram->cxl_tier.storage) {
                memcpy((char*)engram->cxl_tier.storage + entry->storage_offset,
                       entry->data, entry->size);
            }
            engram->cxl_tier.used += entry->size;
            pthread_mutex_unlock(&engram->cxl_tier.lock);
            pthread_mutex_unlock(&entry->lock);
            return GPUIO_SUCCESS;
        }
        pthread_mutex_unlock(&engram->cxl_tier.lock);
        /* Fall through to remote */
        entry->tier = GPUIO_ENGRAM_TIER_REMOTE;
    }
    
    /* Store in remote tier */
    pthread_mutex_lock(&engram->remote_tier.lock);
    entry->storage_offset = engram->remote_tier.used;
    engram->remote_tier.used += entry->size;
    pthread_mutex_unlock(&engram->remote_tier.lock);
    
    /* If async writes enabled, queue for background write */
    if (engram->config.async_writes && entry->data) {
        pthread_mutex_lock(&engram->write_buffer_lock);
        /* In production: copy to write buffer for background flush */
        pthread_mutex_unlock(&engram->write_buffer_lock);
    }
    
    pthread_mutex_unlock(&entry->lock);
    
    pthread_mutex_lock(&engram->stats_lock);
    if (entry->tier == GPUIO_ENGRAM_TIER_HBM) {
        engram->stats.engrams_in_hbm++;
    } else if (entry->tier == GPUIO_ENGRAM_TIER_CXL) {
        engram->stats.engrams_in_cxl++;
    } else {
        engram->stats.engrams_in_remote++;
    }
    pthread_mutex_unlock(&engram->stats_lock);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Load an engram entry from its tier into memory.
 * 
 * Handles tiered storage retrieval and promotes frequently accessed entries.
 */
gpuio_error_t ai_engram_load_entry(struct ai_engram* engram, ai_engram_entry_t* entry) {
    if (!engram || !entry) return GPUIO_ERROR_INVALID_ARG;
    
    pthread_mutex_lock(&entry->lock);
    
    /* Data already in memory, just return */
    if (entry->data) {
        pthread_mutex_unlock(&entry->lock);
        lru_cache_touch(engram->lru_cache, (lru_entry_t*)entry);
        return GPUIO_SUCCESS;
    }
    
    /* Allocate memory for data */
    entry->data = malloc(entry->size);
    if (!entry->data) {
        pthread_mutex_unlock(&entry->lock);
        return GPUIO_ERROR_NOMEM;
    }
    
    /* Load from appropriate tier */
    switch (entry->tier) {
        case GPUIO_ENGRAM_TIER_HBM:
            pthread_mutex_lock(&engram->hbm_tier.lock);
            if (engram->hbm_tier.storage) {
                memcpy(entry->data,
                       (char*)engram->hbm_tier.storage + entry->storage_offset,
                       entry->size);
            }
            pthread_mutex_unlock(&engram->hbm_tier.lock);
            break;
            
        case GPUIO_ENGRAM_TIER_CXL:
            pthread_mutex_lock(&engram->cxl_tier.lock);
            if (engram->cxl_tier.storage) {
                memcpy(entry->data,
                       (char*)engram->cxl_tier.storage + entry->storage_offset,
                       entry->size);
            }
            pthread_mutex_unlock(&engram->cxl_tier.lock);
            break;
            
        case GPUIO_ENGRAM_TIER_REMOTE:
            /* In production: fetch from remote storage */
            /* For now, mark as not found if not cached */
            free(entry->data);
            entry->data = NULL;
            pthread_mutex_unlock(&entry->lock);
            return GPUIO_ERROR_NOT_FOUND;
            
        default:
            pthread_mutex_unlock(&entry->lock);
            return GPUIO_ERROR_INVALID_ARG;
    }
    
    pthread_mutex_unlock(&entry->lock);
    lru_cache_touch(engram->lru_cache, (lru_entry_t*)entry);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Invalidate an engram (mark as stale).
 */
gpuio_error_t gpuio_engram_invalidate(gpuio_engram_pool_t pool,
                                       uint64_t engram_id,
                                       uint64_t min_version) {
    if (!pool) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    
    ai_engram_entry_t* entry = ai_engram_lookup(engram, engram_id);
    if (entry) {
        pthread_mutex_lock(&entry->lock);
        if (entry->version < min_version) {
            entry->version = min_version;
        }
        entry->ref_count--;
        pthread_mutex_unlock(&entry->lock);
        return GPUIO_SUCCESS;
    }
    
    return GPUIO_ERROR_NOT_FOUND;
}

/**
 * @brief Sync engram pool (flush pending writes).
 */
gpuio_error_t gpuio_engram_sync(gpuio_engram_pool_t pool,
                                 gpuio_stream_t stream) {
    (void)stream;
    
    if (!pool) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    
    /* Flush write buffer */
    pthread_mutex_lock(&engram->write_buffer_lock);
    if (engram->write_buffer_used > 0) {
        /* In production, write to remote storage */
        engram->write_buffer_used = 0;
    }
    pthread_mutex_unlock(&engram->write_buffer_lock);
    
    AI_LOG_INFO(engram->ai_ctx, "Engram pool synced");
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Get engram statistics.
 */
gpuio_error_t gpuio_engram_get_stats(gpuio_engram_pool_t pool,
                                      gpuio_engram_stats_t* stats) {
    if (!pool || !stats) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_engram* engram = (struct ai_engram*)pool;
    
    pthread_mutex_lock(&engram->stats_lock);
    memcpy(stats, &engram->stats, sizeof(gpuio_engram_stats_t));
    
    /* Count entries per tier */
    stats->engrams_in_hbm = 0;
    stats->engrams_in_cxl = 0;
    stats->engrams_in_remote = 0;
    
    pthread_mutex_lock(&engram->hash_lock);
    for (int i = 0; i < AI_ENGRAM_HASH_BUCKETS; i++) {
        ai_engram_entry_t* entry = engram->hash_table[i];
        while (entry) {
            if (entry->tier == GPUIO_ENGRAM_TIER_HBM) {
                stats->engrams_in_hbm++;
            } else if (entry->tier == GPUIO_ENGRAM_TIER_CXL) {
                stats->engrams_in_cxl++;
            } else {
                stats->engrams_in_remote++;
            }
            entry = entry->hash_next;
        }
    }
    pthread_mutex_unlock(&engram->hash_lock);
    
    pthread_mutex_unlock(&engram->stats_lock);
    
    return GPUIO_SUCCESS;
}
