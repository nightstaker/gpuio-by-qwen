/**
 * @file gpuio_ai.h
 * @brief GPU-Initiated IO Accelerator - AI/ML Extensions
 * @version 1.0.0
 * @date 2026-02-09
 * 
 * This header defines AI-specific extensions for gpuio, including:
 * - DeepSeek DSA (Dynamic Sparse Attention) KV Cache Management
 * - Graph RAG (Retrieval-Augmented Generation) for Knowledge-Augmented LLMs
 * - DeepSeek Engram Memory Architecture for Petabyte-Scale External Memory
 */

#ifndef GPUIO_AI_H
#define GPUIO_AI_H

#include "gpuio.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * AI Context and Workload Management
 * ============================================================================ */

typedef struct gpuio_model* gpuio_model_handle_t;
typedef struct gpuio_ai_context* gpuio_ai_context_t;
typedef struct gpuio_inference_request* gpuio_inference_request_t;
typedef struct gpuio_training_batch* gpuio_training_batch_t;

/**
 * AI workload priority classes.
 * Higher priority values = lower priority (Linux-style).
 */
typedef enum {
    GPUIO_PRIO_INFERENCE_REALTIME = 0,   /* User-facing inference, <2ms SLO */
    GPUIO_PRIO_INFERENCE_BATCH = 1,      /* Batch inference, <100ms SLO */
    GPUIO_PRIO_TRAINING_FW = 2,          /* Training forward pass */
    GPUIO_PRIO_TRAINING_BW = 3,          /* Training backward pass */
    GPUIO_PRIO_CHECKPOINT = 4,           /* Model checkpointing */
    GPUIO_PRIO_ENGRAM_SYNC = 5,          /* Engram write-back */
    GPUIO_PRIO_BACKGROUND = 6,           /* Prefetch, compression */
} gpuio_ai_priority_t;

/**
 * Deadline-aware QoS for inference.
 */
typedef struct {
    uint64_t deadline_us;           /* SLO deadline in microseconds */
    uint64_t estimated_duration_us; /* Predicted execution time */
    float criticality;              /* Business impact (0.0 - 1.0) */
    bool preemptible;               /* Can be preempted for higher priority */
} gpuio_inference_qos_t;

/**
 * AI context configuration.
 */
typedef struct {
    /* Model parameters */
    int num_layers;
    int num_heads;
    int head_dim;
    int max_sequence_length;
    
    /* Memory systems to enable */
    bool enable_dsa_kv;             /* DSA KV cache */
    bool enable_engram;             /* Engram memory */
    bool enable_graph_rag;          /* Graph RAG */
    
    /* Default scheduling */
    gpuio_ai_priority_t default_priority;
    
    /* Resource limits */
    size_t kv_cache_size;           /* Per-GPU KV cache limit */
    size_t engram_pool_size;        /* Engram pool size per GPU */
} gpuio_ai_config_t;

gpuio_error_t gpuio_ai_context_create(gpuio_context_t ctx,
                                       const gpuio_ai_config_t* config,
                                       gpuio_ai_context_t* ai_ctx);
gpuio_error_t gpuio_ai_context_destroy(gpuio_ai_context_t ai_ctx);

/* ============================================================================
 * DeepSeek DSA KV Cache API
 * ============================================================================ */

typedef struct gpuio_dsa_kv_pool* gpuio_dsa_kv_pool_t;
typedef struct gpuio_dsa_kv_entry* gpuio_dsa_kv_entry_t;

/**
 * DSA KV cache compression methods.
 */
typedef enum {
    GPUIO_KV_COMPRESS_NONE = 0,     /* No compression */
    GPUIO_KV_COMPRESS_FP16 = 1,     /* 2x compression */
    GPUIO_KV_COMPRESS_INT8 = 2,     /* 4x compression with calibration */
    GPUIO_KV_COMPRESS_4BIT = 3,     /* 8x compression (GPTQ-style) */
    GPUIO_KV_COMPRESS_SPARSE = 4,   /* Sparse representation for DSA */
} gpuio_kv_compression_t;

/**
 * KV cache storage tier.
 */
typedef enum {
    GPUIO_KV_TIER_HBM = 0,          /* GPU HBM - hot */
    GPUIO_KV_TIER_CXL = 1,          /* CXL memory - warm */
    GPUIO_KV_TIER_REMOTE = 2,       /* Remote storage - cold */
} gpuio_kv_tier_t;

/**
 * DSA KV cache entry metadata.
 */
typedef struct {
    uint64_t position;              /* Token position */
    uint32_t layer_id;              /* Transformer layer */
    uint32_t head_id;               /* Attention head */
    
    /* Sparsity metadata */
    float importance_score;         /* DSA importance (0.0 - 1.0) */
    uint32_t sparsity_pattern;      /* Bitmask of active heads */
    uint8_t compression_level;      /* Compression method used */
    
    /* Location */
    gpuio_kv_tier_t tier;
    uint64_t compressed_size;
    uint64_t original_size;
    
    /* Cache management */
    uint64_t last_access_time;
    uint32_t access_count;
} gpuio_dsa_kv_metadata_t;

/**
 * DSA KV cache pool configuration.
 */
typedef struct {
    /* HBM tier */
    size_t hbm_capacity;            /* Hot cache size in HBM */
    
    /* CXL tier */
    size_t cxl_capacity;            /* Warm cache size in CXL */
    const char* cxl_device_path;    /* CXL device path */
    
    /* Remote tier */
    const char* remote_uri;         /* rdma:// or nvme:// for cold storage */
    
    /* Compression */
    gpuio_kv_compression_t default_compression;
    float compression_threshold;    /* Compress if importance < threshold */
    
    /* Eviction */
    float importance_decay;         /* Decay factor for importance scores */
    size_t min_hot_cache_size;      /* Minimum HBM reserved for hot KV */
} gpuio_dsa_kv_config_t;

/* Pool management */
gpuio_error_t gpuio_dsa_kv_pool_create(gpuio_ai_context_t ai_ctx,
                                        const gpuio_dsa_kv_config_t* config,
                                        gpuio_dsa_kv_pool_t* pool);
gpuio_error_t gpuio_dsa_kv_pool_destroy(gpuio_dsa_kv_pool_t pool);
gpuio_error_t gpuio_dsa_kv_pool_reset(gpuio_dsa_kv_pool_t pool);

/* KV entry operations */
gpuio_error_t gpuio_dsa_kv_load(gpuio_dsa_kv_pool_t pool,
                                 uint64_t position,
                                 uint32_t layer_id,
                                 uint32_t head_id,
                                 gpuio_stream_t stream,
                                 gpuio_dsa_kv_entry_t* entry);

gpuio_error_t gpuio_dsa_kv_store(gpuio_dsa_kv_pool_t pool,
                                  uint64_t position,
                                  uint32_t layer_id,
                                  uint32_t head_id,
                                  const void* data,
                                  size_t size,
                                  float importance_score,
                                  gpuio_stream_t stream);

gpuio_error_t gpuio_dsa_kv_get_data(gpuio_dsa_kv_entry_t entry,
                                     void** data,
                                     size_t* size);

gpuio_error_t gpuio_dsa_kv_release(gpuio_dsa_kv_entry_t entry);

/* Batch operations */
gpuio_error_t gpuio_dsa_kv_load_batch(gpuio_dsa_kv_pool_t pool,
                                       uint64_t* positions,
                                       uint32_t* layer_ids,
                                       uint32_t* head_ids,
                                       int num_entries,
                                       gpuio_stream_t stream,
                                       gpuio_dsa_kv_entry_t* entries);

/* Sparsity and compression */
gpuio_error_t gpuio_dsa_kv_set_sparsity_pattern(gpuio_dsa_kv_pool_t pool,
                                                 uint32_t layer_id,
                                                 uint32_t sparsity_mask);

gpuio_error_t gpuio_dsa_kv_compact(gpuio_dsa_kv_pool_t pool,
                                    gpuio_stream_t stream);

/* Statistics */
typedef struct {
    uint64_t entries_in_hbm;
    uint64_t entries_in_cxl;
    uint64_t entries_in_remote;
    uint64_t total_hits;
    uint64_t total_misses;
    double hit_rate;
    size_t bytes_saved_by_compression;
} gpuio_dsa_kv_stats_t;

gpuio_error_t gpuio_dsa_kv_get_stats(gpuio_dsa_kv_pool_t pool,
                                      gpuio_dsa_kv_stats_t* stats);

/* ============================================================================
 * Graph RAG (Retrieval-Augmented Generation) API
 * ============================================================================ */

typedef struct gpuio_graph_index* gpuio_graph_index_t;
typedef struct gpuio_graph_storage* gpuio_graph_storage_t;
typedef struct gpuio_graph_rag_request* gpuio_graph_rag_request_handle_t;

/**
 * Graph node structure.
 */
typedef struct {
    uint64_t node_id;
    float* embedding;               /* Vector representation */
    uint32_t embedding_dim;
    char* attributes;               /* JSON-style properties */
    uint64_t* edge_ids;             /* Adjacency list */
    uint32_t num_edges;
} gpuio_graph_node_t;

/**
 * Graph RAG scatter parameters.
 */
typedef struct {
    /* Query */
    const float* query_embedding;
    uint32_t query_dim;
    int top_k;                      /* Number of candidate nodes */
    float similarity_threshold;     /* Minimum similarity score */
    
    /* Graph filters */
    const char** edge_type_filters;
    int num_edge_type_filters;
    const char** node_label_filters;
    int num_node_label_filters;
} gpuio_scatter_params_t;

/**
 * Graph RAG gather parameters.
 */
typedef struct {
    int hop_depth;                  /* Multi-hop expansion (1-3) */
    int max_subgraph_size;          /* Maximum nodes in subgraph */
    bool include_edges;             /* Fetch edge attributes */
    bool include_neighbors;         /* Include neighbor embeddings */
} gpuio_gather_params_t;

/**
 * Graph RAG request.
 */
typedef struct {
    /* Scatter phase */
    gpuio_scatter_params_t scatter;
    
    /* Gather phase */
    gpuio_gather_params_t gather;
    
    /* Output */
    gpuio_graph_node_t* subgraph;
    int subgraph_size;
    float* subgraph_adj_matrix;     /* Optional adjacency matrix */
    
    /* Control */
    gpuio_stream_t stream;
    gpuio_callback_t callback;
    void* user_data;
} gpuio_graph_rag_request_t;

/* Graph index management */
gpuio_error_t gpuio_graph_index_create(gpuio_ai_context_t ai_ctx,
                                        const char* index_path,
                                        gpuio_graph_index_t* index);
gpuio_error_t gpuio_graph_index_destroy(gpuio_graph_index_t index);
gpuio_error_t gpuio_graph_index_add_nodes(gpuio_graph_index_t index,
                                           gpuio_graph_node_t* nodes,
                                           int num_nodes,
                                           gpuio_stream_t stream);

/* Graph RAG operations */
gpuio_error_t gpuio_graph_rag_scatter(gpuio_graph_index_t index,
                                       const gpuio_scatter_params_t* params,
                                       gpuio_stream_t stream,
                                       uint64_t** candidate_ids,
                                       int* num_candidates);

gpuio_error_t gpuio_graph_rag_gather(gpuio_graph_storage_t storage,
                                      const uint64_t* root_ids,
                                      int num_roots,
                                      const gpuio_gather_params_t* params,
                                      gpuio_stream_t stream,
                                      gpuio_graph_node_t** subgraph,
                                      int* subgraph_size);

/* Combined scatter-gather */
gpuio_error_t gpuio_graph_rag_query(gpuio_graph_index_t index,
                                     gpuio_graph_storage_t storage,
                                     const gpuio_graph_rag_request_t* request,
                                     gpuio_graph_rag_request_handle_t* handle);

gpuio_error_t gpuio_graph_rag_wait(gpuio_graph_rag_request_handle_t handle,
                                    uint64_t timeout_us);

/* Graph storage backends */
gpuio_error_t gpuio_graph_storage_create(gpuio_ai_context_t ai_ctx,
                                          const char* uri,
                                          gpuio_graph_storage_t* storage);
gpuio_error_t gpuio_graph_storage_destroy(gpuio_graph_storage_t storage);

/* ============================================================================
 * DeepSeek Engram Memory API
 * ============================================================================ */

typedef struct gpuio_engram_pool* gpuio_engram_pool_t;
typedef struct gpuio_engram* gpuio_engram_handle_t;

/**
 * Engram storage tier.
 */
typedef enum {
    GPUIO_ENGRAM_TIER_HBM = 0,      /* GPU HBM - hot */
    GPUIO_ENGRAM_TIER_CXL = 1,      /* CXL memory - warm */
    GPUIO_ENGRAM_TIER_REMOTE = 2,   /* Distributed storage - cold */
} gpuio_engram_tier_t;

/**
 * Engram data structure.
 */
typedef struct {
    uint64_t engram_id;             /* Unique identifier (content hash) */
    uint64_t version;               /* Version for cache coherence */
    
    /* Content */
    const void* data;
    size_t size;
    const float* embedding;         /* For semantic retrieval */
    uint32_t embedding_dim;
    
    /* Metadata */
    uint64_t creation_timestamp;
    float importance_score;         /* Learned importance */
    uint32_t access_count;
    
    /* Storage */
    gpuio_engram_tier_t tier;
} gpuio_engram_t;

/**
 * Engram pool configuration.
 */
typedef struct {
    /* Tier capacities */
    size_t hbm_capacity;
    size_t cxl_capacity;
    const char* cxl_device_path;
    
    /* Remote archive */
    const char* remote_archive_uri;
    
    /* Indexing */
    uint32_t embedding_dim;
    const char* index_type;         /* "hnsw", "ivf", "flat" */
    
    /* Write behavior */
    bool async_writes;              /* Async write to remote */
    uint64_t write_buffer_size;
    uint64_t flush_interval_ms;
} gpuio_engram_config_t;

/* Pool management */
gpuio_error_t gpuio_engram_pool_create(gpuio_ai_context_t ai_ctx,
                                        const gpuio_engram_config_t* config,
                                        gpuio_engram_pool_t* pool);
gpuio_error_t gpuio_engram_pool_destroy(gpuio_engram_pool_t pool);

/* Engram operations */
gpuio_error_t gpuio_engram_write(gpuio_engram_pool_t pool,
                                  const gpuio_engram_t* engram,
                                  gpuio_stream_t stream);

gpuio_error_t gpuio_engram_read(gpuio_engram_pool_t pool,
                                 uint64_t engram_id,
                                 gpuio_stream_t stream,
                                 gpuio_engram_handle_t* handle);

gpuio_error_t gpuio_engram_query(gpuio_engram_pool_t pool,
                                  const float* query_embedding,
                                  float similarity_threshold,
                                  int top_k,
                                  gpuio_stream_t stream,
                                  gpuio_engram_handle_t* results,
                                  int* num_results);

gpuio_error_t gpuio_engram_get_data(gpuio_engram_handle_t handle,
                                     void** data,
                                     size_t* size);

gpuio_error_t gpuio_engram_release(gpuio_engram_handle_t handle);

/* Batch operations */
gpuio_error_t gpuio_engram_write_batch(gpuio_engram_pool_t pool,
                                        const gpuio_engram_t* engrams,
                                        int num_engrams,
                                        gpuio_stream_t stream);

gpuio_error_t gpuio_engram_query_batch(gpuio_engram_pool_t pool,
                                        const float** query_embeddings,
                                        int num_queries,
                                        int top_k_per_query,
                                        gpuio_stream_t stream,
                                        gpuio_engram_handle_t** results,
                                        int** num_results_per_query);

/* Coherence and versioning */
gpuio_error_t gpuio_engram_invalidate(gpuio_engram_pool_t pool,
                                       uint64_t engram_id,
                                       uint64_t min_version);

gpuio_error_t gpuio_engram_sync(gpuio_engram_pool_t pool,
                                 gpuio_stream_t stream);

/* Statistics */
typedef struct {
    uint64_t engrams_in_hbm;
    uint64_t engrams_in_cxl;
    uint64_t engrams_in_remote;
    uint64_t total_queries;
    uint64_t cache_hits;
    double avg_query_latency_us;
} gpuio_engram_stats_t;

gpuio_error_t gpuio_engram_get_stats(gpuio_engram_pool_t pool,
                                      gpuio_engram_stats_t* stats);

/* ============================================================================
 * High-Level AI Workload APIs
 * ============================================================================ */

/**
 * Inference request.
 */
typedef struct {
    /* Input */
    const void* input_data;
    size_t input_size;
    int input_tokens;
    
    /* Context (optional) */
    gpuio_dsa_kv_pool_t kv_pool;    /* Existing KV cache */
    gpuio_engram_pool_t engram_pool; /* Engram memory */
    
    /* Output */
    void* output_buffer;
    size_t output_buffer_size;
    int max_output_tokens;
    
    /* QoS */
    gpuio_inference_qos_t qos;
    
    /* Control */
    gpuio_stream_t stream;
    gpuio_callback_t callback;
    void* user_data;
} gpuio_inference_params_t;

typedef struct {
    void* output_data;
    size_t output_size;
    int output_tokens;
    uint64_t latency_us;
    gpuio_error_t status;
} gpuio_inference_result_t;

/**
 * Execute inference with all memory systems.
 */
gpuio_error_t gpuio_ai_inference(gpuio_ai_context_t ai_ctx,
                                  const gpuio_inference_params_t* params,
                                  gpuio_inference_result_t* result);

/**
 * Training batch configuration.
 */
typedef struct {
    /* Input data */
    const void** inputs;
    size_t* input_sizes;
    int num_inputs;
    
    /* Labels/Targets */
    const void* labels;
    size_t labels_size;
    
    /* Context */
    gpuio_dsa_kv_pool_t kv_pool;
    gpuio_engram_pool_t engram_pool;
    
    /* Checkpoint policy */
    bool checkpoint_after_step;
    const char* checkpoint_path;
    int checkpoint_interval_steps;
    
    /* Control */
    gpuio_stream_t stream;
} gpuio_training_params_t;

typedef struct {
    float loss;
    uint64_t step_time_us;
    bool checkpoint_saved;
    char checkpoint_path[512];
    gpuio_error_t status;
} gpuio_training_result_t;

/**
 * Execute training step with checkpointing and engram updates.
 */
gpuio_error_t gpuio_ai_training_step(gpuio_ai_context_t ai_ctx,
                                      const gpuio_training_params_t* params,
                                      gpuio_training_result_t* result);

/**
 * Model checkpointing.
 */
gpuio_error_t gpuio_ai_checkpoint_save(gpuio_ai_context_t ai_ctx,
                                        const char* path,
                                        bool async,
                                        gpuio_stream_t stream);

gpuio_error_t gpuio_ai_checkpoint_load(gpuio_ai_context_t ai_ctx,
                                        const char* path,
                                        gpuio_stream_t stream);

/* ============================================================================
 * Compression API
 * ============================================================================ */

/**
 * Compression codecs.
 */
typedef struct gpuio_codec* gpuio_codec_t;

typedef enum {
    GPUIO_CODEC_LZ4 = 0,            /* Fast compression */
    GPUIO_CODEC_ZSTD = 1,           /* Balanced compression */
    GPUIO_CODEC_GZIP = 2,           /* Maximum compression */
    GPUIO_CODEC_FP16 = 3,           /* FP32 -> FP16 for tensors */
    GPUIO_CODEC_INT8 = 4,           /* Quantization to INT8 */
    GPUIO_CODEC_CUSTOM = 5,         /* User-defined codec */
} gpuio_codec_type_t;

gpuio_error_t gpuio_codec_create(gpuio_context_t ctx,
                                  gpuio_codec_type_t type,
                                  int level,  /* Compression level */
                                  gpuio_codec_t* codec);
gpuio_error_t gpuio_codec_destroy(gpuio_codec_t codec);

/* Compress on GPU */
gpuio_error_t gpuio_compress(gpuio_codec_t codec,
                              const void* input,
                              size_t input_size,
                              void* output,
                              size_t output_capacity,
                              size_t* output_size,
                              gpuio_stream_t stream);

/* Decompress on GPU */
gpuio_error_t gpuio_decompress(gpuio_codec_t codec,
                                const void* input,
                                size_t input_size,
                                void* output,
                                size_t output_capacity,
                                size_t* output_size,
                                gpuio_stream_t stream);

/* Compress during transfer - zero copy */
gpuio_error_t gpuio_transfer_compressed(gpuio_request_t request,
                                         gpuio_codec_t codec);

#ifdef __cplusplus
}
#endif

#endif /* GPUIO_AI_H */
