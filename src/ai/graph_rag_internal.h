/**
 * @file graph_rag_internal.h
 * @brief AI Extensions module - Graph RAG internal structures
 * @version 1.1.0
 * 
 * Internal structures and functions for Graph RAG (Retrieval-Augmented Generation)
 * with HNSW-style vector search and multi-hop graph traversal.
 */

#ifndef GRAPH_RAG_INTERNAL_H
#define GRAPH_RAG_INTERNAL_H

#include "ai_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Configuration Constants
 * ============================================================================ */

#define AI_GRAPH_MAX_NODES       1000000
#define AI_GRAPH_MAX_EDGES       10000000
#define AI_GRAPH_EMBEDDING_MAX_DIM  4096

/* ============================================================================
 * Graph Node
 * ============================================================================ */

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

/* ============================================================================
 * Graph Edge
 * ============================================================================ */

typedef struct ai_graph_edge {
    uint64_t edge_id;
    uint64_t src_id;
    uint64_t dst_id;
    char* edge_type;
    char* attributes;
    float weight;
} ai_graph_edge_t;

/* ============================================================================
 * Graph Index
 * ============================================================================ */

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

/* ============================================================================
 * Graph Storage
 * ============================================================================ */

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

/* ============================================================================
 * Graph RAG Request Handle
 * ============================================================================ */

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

/* ============================================================================
 * Graph RAG Internal Functions
 * ============================================================================ */

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

/**
 * @brief Compute cosine similarity between vectors.
 * Delegates to vector_ops.h implementation.
 */
static inline float ai_graph_similarity(const float* a, const float* b, uint32_t dim) {
    return vec_cosine_similarity_f32(a, b, dim);
}

/* ============================================================================
 * Graph RAG Main Structure (stub - actual implementation in future)
 * ============================================================================ */

struct ai_graph_rag {
    /* Placeholder for future graph RAG orchestration state */
    gpuio_ai_context_t ai_ctx;
    void* reserved;
};

#ifdef __cplusplus
}
#endif

#endif /* GRAPH_RAG_INTERNAL_H */
