/**
 * @file ai_internal.h
 * @brief AI Extensions module - Internal header
 * @version 1.0.0
 * 
 * Internal structures and function declarations for the AI Extensions module.
 * Includes DeepSeek DSA KV Cache, Graph RAG, and Engram Memory implementations.
 */

#ifndef AI_INTERNAL_H
#define AI_INTERNAL_H

#include "gpuio.h"
#include "gpuio_ai.h"
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

/* Module exports */
#define AI_API __attribute__((visibility("default")))

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
 * DSA KV Cache Internal Structures
 * ============================================================================ */

#define AI_DSA_KV_MAX_LAYERS 128
#define AI_DSA_KV_MAX_HEADS 256
#define AI_DSA_KV_HASH_BUCKETS 1024

typedef struct ai_kv_entry {
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
    
    /* Cache management */
    uint64_t last_access_time;
    uint64_t access_count;
    
    /* List pointers */
    struct ai_kv_entry* hash_next;
    struct ai_kv_entry* lru_prev;
    struct ai_kv_entry* lru_next;
    
    /* Reference counting */
    int ref_count;
    pthread_mutex_t lock;
} ai_kv_entry_t;

typedef struct ai_kv_tier {
    size_t capacity;
    size_t used;
    void* storage;
    char* cxl_device_path;
    char* remote_uri;
} ai_kv_tier_t;

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
    
    /* LRU list */
    ai_kv_entry_t* lru_head;
    ai_kv_entry_t* lru_tail;
    pthread_mutex_t lru_lock;
    
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

/* DSA KV internal functions */
int ai_dsa_kv_init(struct ai_dsa_kv* kv, gpuio_ai_context_t ai_ctx,
                   const gpuio_dsa_kv_config_t* config);
void ai_dsa_kv_cleanup(struct ai_dsa_kv* kv);
ai_kv_entry_t* ai_dsa_kv_lookup(struct ai_dsa_kv* kv, uint64_t position,
                                 uint32_t layer_id, uint32_t head_id);
gpuio_error_t ai_dsa_kv_promote_entry(struct ai_dsa_kv* kv, ai_kv_entry_t* entry,
                                       gpuio_kv_tier_t target_tier);
gpuio_error_t ai_dsa_kv_evict_entries(struct ai_dsa_kv* kv, size_t needed_space,
                                       gpuio_kv_tier_t tier);
uint64_t ai_dsa_kv_hash(uint64_t position, uint32_t layer_id, uint32_t head_id);
void ai_dsa_kv_lru_add(struct ai_dsa_kv* kv, ai_kv_entry_t* entry);
void ai_dsa_kv_lru_remove(struct ai_dsa_kv* kv, ai_kv_entry_t* entry);
void ai_dsa_kv_lru_touch(struct ai_dsa_kv* kv, ai_kv_entry_t* entry);

/* ============================================================================
 * Graph RAG Internal Structures
 * ============================================================================ */

#define AI_GRAPH_MAX_NODES 1000000
#define AI_GRAPH_MAX_EDGES 10000000
#define AI_GRAPH_EMBEDDING_MAX_DIM 4096

typedef struct ai_graph_node {
    uint64_t node_id;
    float* embedding;
    uint32_t embedding_dim;
    char* attributes;
    uint64_t* edge_ids;
    uint32_t num_edges;
    uint32_t edge_capacity;
    
    /* Spatial index */
    struct ai_graph_node* hnsw_next;
    struct ai_graph_node* hnsw_prev;
    float hnsw_level;
} ai_graph_node_t;

typedef struct ai_graph_edge {
    uint64_t edge_id;
    uint64_t src_id;
    uint64_t dst_id;
    char* edge_type;
    char* attributes;
    float weight;
} ai_graph_edge_t;

struct ai_graph_index {
    gpuio_ai_context_t ai_ctx;
    char* index_path;
    
    /* Node storage */
    ai_graph_node_t* nodes;
    uint64_t num_nodes;
    uint64_t node_capacity;
    pthread_mutex_t nodes_lock;
    
    /* HNSW-like index structure */
    ai_graph_node_t** hnsw_levels;
    uint32_t hnsw_num_levels;
    pthread_mutex_t index_lock;
    
    /* Configuration */
    uint32_t embedding_dim;
    int ef_construction;
    int M; /* max neighbors per layer */
};

struct ai_graph_storage {
    gpuio_ai_context_t ai_ctx;
    char* uri;
    
    /* Storage backend */
    void* backend_handle;
    int backend_type; /* 0=local, 1=remote */
    
    /* Cache */
    ai_graph_node_t* cache;
    uint64_t cache_size;
    pthread_mutex_t cache_lock;
};

struct ai_graph_rag_request {
    gpuio_graph_rag_request_t params;
    gpuio_graph_index_t index;
    gpuio_graph_storage_t storage;
    
    /* State */
    uint64_t* candidate_ids;
    int num_candidates;
    gpuio_graph_node_t* subgraph;
    int subgraph_size;
    
    /* Async handling */
    pthread_mutex_t lock;
    pthread_cond_t cond;
    bool completed;
    gpuio_error_t result;
};

/* Graph RAG internal functions */
int ai_graph_index_init(struct ai_graph_index* idx, gpuio_ai_context_t ai_ctx,
                       const char* path, uint32_t embedding_dim);
void ai_graph_index_cleanup(struct ai_graph_index* idx);
int ai_graph_storage_init(struct ai_graph_storage* storage, gpuio_ai_context_t ai_ctx,
                         const char* uri);
void ai_graph_storage_cleanup(struct ai_graph_storage* storage);
gpuio_error_t ai_graph_hnsw_search(struct ai_graph_index* idx,
                                    const float* query, uint32_t dim,
                                    int top_k, float threshold,
                                    uint64_t** results, int* num_results);
gpuio_error_t ai_graph_traverse(struct ai_graph_storage* storage,
                                 const uint64_t* root_ids, int num_roots,
                                 int hop_depth, int max_size,
                                 ai_graph_node_t** subgraph, int* subgraph_size);
float ai_graph_similarity(const float* a, const float* b, uint32_t dim);

/* ============================================================================
 * Engram Memory Internal Structures
 * ============================================================================ */

#define AI_ENGRAM_HASH_BUCKETS 65536
#define AI_ENGRAM_MAX_BATCH 1024

typedef struct ai_engram_entry {
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
    
    /* LRU */
    struct ai_engram_entry* lru_prev;
    struct ai_engram_entry* lru_next;
    
    /* Reference counting */
    int ref_count;
    pthread_mutex_t lock;
} ai_engram_entry_t;

typedef struct ai_engram_tier {
    size_t capacity;
    size_t used;
    void* storage;
    pthread_mutex_t lock;
} ai_engram_tier_t;

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
    
    /* LRU list */
    ai_engram_entry_t* lru_head;
    ai_engram_entry_t* lru_tail;
    pthread_mutex_t lru_lock;
    
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

/* Engram internal functions */
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
float ai_engram_similarity(const float* a, const float* b, uint32_t dim);
void* ai_engram_write_thread(void* arg);
void ai_engram_lru_add(struct ai_engram* engram, ai_engram_entry_t* entry);
void ai_engram_lru_remove(struct ai_engram* engram, ai_engram_entry_t* entry);
void ai_engram_lru_touch(struct ai_engram* engram, ai_engram_entry_t* entry);

/* ============================================================================
 * Compression Internal Structures
 * ============================================================================ */

struct gpuio_codec {
    gpuio_context_t ctx;
    gpuio_codec_type_t type;
    int level;
    
    /* Codec state */
    void* state;
    size_t state_size;
    
    /* Function pointers */
    gpuio_error_t (*compress_fn)(struct gpuio_codec* codec,
                                  const void* input, size_t input_size,
                                  void* output, size_t output_capacity,
                                  size_t* output_size);
    gpuio_error_t (*decompress_fn)(struct gpuio_codec* codec,
                                    const void* input, size_t input_size,
                                    void* output, size_t output_capacity,
                                    size_t* output_size);
    
    /* FP16/INT8 quantization params */
    float* scale_factors;
    float* zero_points;
    uint32_t num_channels;
};

/* Compression internal functions */
gpuio_error_t ai_codec_compress_fp16(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size);
gpuio_error_t ai_codec_decompress_fp16(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size);
gpuio_error_t ai_codec_compress_int8(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size);
gpuio_error_t ai_codec_decompress_int8(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size);
gpuio_error_t ai_codec_compress_4bit(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size);
gpuio_error_t ai_codec_decompress_4bit(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size);

/* ============================================================================
 * Internal Logging
 * ============================================================================ */

#define AI_LOG(ctx, level, ...) \
    do { gpuio_log(level, "[AI] " __VA_ARGS__); } while(0)

#define AI_LOG_ERROR(ctx, ...) AI_LOG(ctx, GPUIO_LOG_ERROR, __VA_ARGS__)
#define AI_LOG_WARN(ctx, ...)  AI_LOG(ctx, GPUIO_LOG_WARN, __VA_ARGS__)
#define AI_LOG_INFO(ctx, ...)  AI_LOG(ctx, GPUIO_LOG_INFO, __VA_ARGS__)
#define AI_LOG_DEBUG(ctx, ...) AI_LOG(ctx, GPUIO_LOG_DEBUG, __VA_ARGS__)

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

static inline uint64_t ai_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

static inline size_t ai_align_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

#endif /* AI_INTERNAL_H */
