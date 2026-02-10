/**
 * @file graph_rag.c
 * @brief AI Extensions module - Graph RAG implementation
 * @version 1.0.0
 * 
 * Graph RAG (Retrieval-Augmented Generation) implementation for knowledge-
 * augmented LLMs. Supports vector similarity search, multi-hop graph traversal,
 * and scatter/gather operations optimized for GPU-accelerated inference.
 */

#include "ai_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

/**
 * @brief Compute cosine similarity between two vectors.
 */
float ai_graph_similarity(const float* a, const float* b, uint32_t dim) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

/**
 * @brief Initialize graph index.
 */
int ai_graph_index_init(struct ai_graph_index* idx, gpuio_ai_context_t ai_ctx,
                       const char* path, uint32_t embedding_dim) {
    if (!idx || !path) return -1;
    
    idx->ai_ctx = ai_ctx;
    idx->index_path = strdup(path);
    idx->embedding_dim = embedding_dim;
    idx->ef_construction = 200;
    idx->M = 16;
    
    /* Initialize node storage */
    idx->node_capacity = 1024;
    idx->num_nodes = 0;
    idx->nodes = calloc(idx->node_capacity, sizeof(ai_graph_node_t));
    if (!idx->nodes) return -1;
    
    /* Initialize HNSW levels */
    idx->hnsw_num_levels = 4;
    idx->hnsw_levels = calloc(idx->hnsw_num_levels, sizeof(ai_graph_node_t*));
    if (!idx->hnsw_levels) {
        free(idx->nodes);
        return -1;
    }
    
    pthread_mutex_init(&idx->nodes_lock, NULL);
    pthread_mutex_init(&idx->index_lock, NULL);
    
    AI_LOG_INFO(ai_ctx, "Graph index initialized (path=%s, dim=%u)",
                path, embedding_dim);
    
    return 0;
}

/**
 * @brief Cleanup graph index.
 */
void ai_graph_index_cleanup(struct ai_graph_index* idx) {
    if (!idx) return;
    
    /* Free all nodes */
    for (uint64_t i = 0; i < idx->num_nodes; i++) {
        ai_graph_node_t* node = &idx->nodes[i];
        if (node->embedding) free(node->embedding);
        if (node->attributes) free(node->attributes);
        if (node->edge_ids) free(node->edge_ids);
    }
    
    free(idx->nodes);
    free(idx->hnsw_levels);
    if (idx->index_path) free(idx->index_path);
    
    pthread_mutex_destroy(&idx->nodes_lock);
    pthread_mutex_destroy(&idx->index_lock);
}

/**
 * @brief Initialize graph storage backend.
 */
int ai_graph_storage_init(struct ai_graph_storage* storage, gpuio_ai_context_t ai_ctx,
                         const char* uri) {
    if (!storage || !uri) return -1;
    
    storage->ai_ctx = ai_ctx;
    storage->uri = strdup(uri);
    
    /* Determine backend type from URI */
    if (strncmp(uri, "nvme://", 7) == 0 || strncmp(uri, "file://", 7) == 0) {
        storage->backend_type = 0; /* local */
    } else if (strncmp(uri, "rdma://", 7) == 0 || strncmp(uri, "tcp://", 6) == 0) {
        storage->backend_type = 1; /* remote */
    } else {
        storage->backend_type = 0; /* default to local */
    }
    
    /* Initialize cache */
    storage->cache_size = 1024;
    storage->cache = calloc(storage->cache_size, sizeof(ai_graph_node_t));
    if (!storage->cache) {
        free(storage->uri);
        return -1;
    }
    
    pthread_mutex_init(&storage->cache_lock, NULL);
    
    AI_LOG_INFO(ai_ctx, "Graph storage initialized (uri=%s, type=%s)",
                uri, storage->backend_type == 0 ? "local" : "remote");
    
    return 0;
}

/**
 * @brief Cleanup graph storage.
 */
void ai_graph_storage_cleanup(struct ai_graph_storage* storage) {
    if (!storage) return;
    
    if (storage->cache) {
        for (uint64_t i = 0; i < storage->cache_size; i++) {
            ai_graph_node_t* node = &storage->cache[i];
            if (node->embedding) free(node->embedding);
            if (node->attributes) free(node->attributes);
            if (node->edge_ids) free(node->edge_ids);
        }
        free(storage->cache);
    }
    
    if (storage->uri) free(storage->uri);
    pthread_mutex_destroy(&storage->cache_lock);
}

/**
 * @brief HNSW-style approximate nearest neighbor search.
 */
gpuio_error_t ai_graph_hnsw_search(struct ai_graph_index* idx,
                                    const float* query, uint32_t dim,
                                    int top_k, float threshold,
                                    uint64_t** results, int* num_results) {
    if (!idx || !query || !results || !num_results) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    if (dim != idx->embedding_dim) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    /* Simple linear search for stub implementation */
    /* In production, this would use HNSW layers for O(log n) search */
    
    typedef struct {
        uint64_t node_id;
        float similarity;
    } score_t;
    
    score_t* scores = calloc(idx->num_nodes, sizeof(score_t));
    if (!scores) return GPUIO_ERROR_NOMEM;
    
    pthread_mutex_lock(&idx->nodes_lock);
    
    int count = 0;
    for (uint64_t i = 0; i < idx->num_nodes; i++) {
        ai_graph_node_t* node = &idx->nodes[i];
        if (node->embedding) {
            float sim = ai_graph_similarity(query, node->embedding, dim);
            if (sim >= threshold) {
                scores[count].node_id = node->node_id;
                scores[count].similarity = sim;
                count++;
            }
        }
    }
    
    pthread_mutex_unlock(&idx->nodes_lock);
    
    /* Sort by similarity (descending) */
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (scores[j].similarity > scores[i].similarity) {
                score_t tmp = scores[i];
                scores[i] = scores[j];
                scores[j] = tmp;
            }
        }
    }
    
    /* Return top-k */
    int result_count = (count < top_k) ? count : top_k;
    if (result_count > 0) {
        *results = calloc(result_count, sizeof(uint64_t));
        if (!*results) {
            free(scores);
            return GPUIO_ERROR_NOMEM;
        }
        for (int i = 0; i < result_count; i++) {
            (*results)[i] = scores[i].node_id;
        }
    }
    *num_results = result_count;
    
    free(scores);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Multi-hop graph traversal.
 */
gpuio_error_t ai_graph_traverse(struct ai_graph_storage* storage,
                                 const uint64_t* root_ids, int num_roots,
                                 int hop_depth, int max_size,
                                 ai_graph_node_t** subgraph, int* subgraph_size) {
    if (!storage || !root_ids || num_roots <= 0 || hop_depth < 1 || !subgraph) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    /* Allocate result array */
    int capacity = max_size > 0 ? max_size : 1000;
    *subgraph = calloc(capacity, sizeof(ai_graph_node_t));
    if (!*subgraph) return GPUIO_ERROR_NOMEM;
    
    /* BFS traversal */
    uint64_t* visited = calloc(capacity, sizeof(uint64_t));
    int visited_count = 0;
    int result_count = 0;
    
    /* Initialize with root nodes */
    uint64_t* frontier = calloc(capacity, sizeof(uint64_t));
    int frontier_size = 0;
    
    for (int i = 0; i < num_roots && i < capacity; i++) {
        frontier[frontier_size++] = root_ids[i];
        visited[visited_count++] = root_ids[i];
    }
    
    /* Multi-hop expansion */
    for (int hop = 0; hop < hop_depth && frontier_size > 0; hop++) {
        uint64_t* next_frontier = calloc(capacity, sizeof(uint64_t));
        int next_size = 0;
        
        for (int i = 0; i < frontier_size && result_count < capacity; i++) {
            /* Find node in storage */
            pthread_mutex_lock(&storage->cache_lock);
            ai_graph_node_t* node = NULL;
            for (uint64_t j = 0; j < storage->cache_size; j++) {
                if (storage->cache[j].node_id == frontier[i]) {
                    node = &storage->cache[j];
                    break;
                }
            }
            
            if (node && node->embedding) {
                /* Copy to result */
                memcpy(&(*subgraph)[result_count], node, sizeof(ai_graph_node_t));
                /* Duplicate dynamic data */
                if (node->embedding) {
                    (*subgraph)[result_count].embedding = malloc(node->embedding_dim * sizeof(float));
                    memcpy((*subgraph)[result_count].embedding, node->embedding, 
                           node->embedding_dim * sizeof(float));
                }
                result_count++;
                
                /* Add neighbors to next frontier */
                if (node->edge_ids) {
                    for (uint32_t e = 0; e < node->num_edges && next_size < capacity; e++) {
                        /* Check if already visited */
                        bool already_visited = false;
                        for (int v = 0; v < visited_count; v++) {
                            if (visited[v] == node->edge_ids[e]) {
                                already_visited = true;
                                break;
                            }
                        }
                        if (!already_visited) {
                            next_frontier[next_size++] = node->edge_ids[e];
                            if (visited_count < capacity) {
                                visited[visited_count++] = node->edge_ids[e];
                            }
                        }
                    }
                }
            }
            pthread_mutex_unlock(&storage->cache_lock);
        }
        
        free(frontier);
        frontier = next_frontier;
        frontier_size = next_size;
    }
    
    free(frontier);
    free(visited);
    
    *subgraph_size = result_count;
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Create a graph index.
 */
gpuio_error_t gpuio_graph_index_create(gpuio_ai_context_t ai_ctx,
                                        const char* index_path,
                                        gpuio_graph_index_t* index) {
    if (!ai_ctx || !index_path || !index) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    gpuio_error_t err = ai_context_validate(ai_ctx, false, false, true);
    if (err != GPUIO_SUCCESS) return err;
    
    struct ai_graph_index* idx = calloc(1, sizeof(struct ai_graph_index));
    if (!idx) return GPUIO_ERROR_NOMEM;
    
    if (ai_graph_index_init(idx, ai_ctx, index_path, 768) != 0) {
        free(idx);
        return GPUIO_ERROR_NOMEM;
    }
    
    *index = (gpuio_graph_index_t)idx;
    
    AI_LOG_INFO(ai_ctx, "Graph index created: %s", index_path);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Destroy a graph index.
 */
gpuio_error_t gpuio_graph_index_destroy(gpuio_graph_index_t index) {
    if (!index) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_graph_index* idx = (struct ai_graph_index*)index;
    
    ai_graph_index_cleanup(idx);
    free(idx);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Add nodes to the graph index.
 */
gpuio_error_t gpuio_graph_index_add_nodes(gpuio_graph_index_t index,
                                           gpuio_graph_node_t* nodes,
                                           int num_nodes,
                                           gpuio_stream_t stream) {
    (void)stream;
    
    if (!index || !nodes || num_nodes <= 0) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_graph_index* idx = (struct ai_graph_index*)index;
    
    pthread_mutex_lock(&idx->nodes_lock);
    
    /* Grow capacity if needed */
    if (idx->num_nodes + num_nodes > idx->node_capacity) {
        uint64_t new_capacity = idx->node_capacity * 2;
        while (new_capacity < idx->num_nodes + num_nodes) {
            new_capacity *= 2;
        }
        
        ai_graph_node_t* new_nodes = realloc(idx->nodes, 
                                              new_capacity * sizeof(ai_graph_node_t));
        if (!new_nodes) {
            pthread_mutex_unlock(&idx->nodes_lock);
            return GPUIO_ERROR_NOMEM;
        }
        
        idx->nodes = new_nodes;
        idx->node_capacity = new_capacity;
    }
    
    /* Add nodes */
    for (int i = 0; i < num_nodes; i++) {
        ai_graph_node_t* dst = &idx->nodes[idx->num_nodes + i];
        
        dst->node_id = nodes[i].node_id;
        dst->embedding_dim = nodes[i].embedding_dim;
        
        if (nodes[i].embedding && nodes[i].embedding_dim > 0) {
            dst->embedding = malloc(nodes[i].embedding_dim * sizeof(float));
            if (dst->embedding) {
                memcpy(dst->embedding, nodes[i].embedding, 
                       nodes[i].embedding_dim * sizeof(float));
            }
        }
        
        if (nodes[i].attributes) {
            dst->attributes = strdup(nodes[i].attributes);
        }
        
        dst->num_edges = nodes[i].num_edges;
        if (nodes[i].num_edges > 0 && nodes[i].edge_ids) {
            dst->edge_ids = malloc(nodes[i].num_edges * sizeof(uint64_t));
            if (dst->edge_ids) {
                memcpy(dst->edge_ids, nodes[i].edge_ids, 
                       nodes[i].num_edges * sizeof(uint64_t));
            }
        }
    }
    
    idx->num_nodes += num_nodes;
    
    pthread_mutex_unlock(&idx->nodes_lock);
    
    AI_LOG_INFO(idx->ai_ctx, "Added %d nodes to graph index (total=%zu)",
                num_nodes, idx->num_nodes);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Scatter phase: find candidate nodes by similarity.
 */
gpuio_error_t gpuio_graph_rag_scatter(gpuio_graph_index_t index,
                                       const gpuio_scatter_params_t* params,
                                       gpuio_stream_t stream,
                                       uint64_t** candidate_ids,
                                       int* num_candidates) {
    (void)stream;
    
    if (!index || !params || !candidate_ids || !num_candidates) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    if (!params->query_embedding || params->query_dim == 0) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_graph_index* idx = (struct ai_graph_index*)index;
    
    gpuio_error_t err = ai_graph_hnsw_search(idx,
                                              params->query_embedding,
                                              params->query_dim,
                                              params->top_k,
                                              params->similarity_threshold,
                                              candidate_ids,
                                              num_candidates);
    
    return err;
}

/**
 * @brief Gather phase: expand candidate set via graph traversal.
 */
gpuio_error_t gpuio_graph_rag_gather(gpuio_graph_storage_t storage,
                                      const uint64_t* root_ids,
                                      int num_roots,
                                      const gpuio_gather_params_t* params,
                                      gpuio_stream_t stream,
                                      gpuio_graph_node_t** subgraph,
                                      int* subgraph_size) {
    (void)stream;
    
    if (!storage || !root_ids || num_roots <= 0 || !params || !subgraph) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    if (params->hop_depth < 1 || params->hop_depth > 5) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_graph_storage* stor = (struct ai_graph_storage*)storage;
    
    ai_graph_node_t* result_nodes = NULL;
    int result_count = 0;
    
    gpuio_error_t err = ai_graph_traverse(stor, root_ids, num_roots,
                                          params->hop_depth,
                                          params->max_subgraph_size,
                                          &result_nodes, &result_count);
    
    if (err == GPUIO_SUCCESS) {
        *subgraph = (gpuio_graph_node_t*)result_nodes;
        *subgraph_size = result_count;
    }
    
    return err;
}

/**
 * @brief Combined scatter-gather query.
 */
gpuio_error_t gpuio_graph_rag_query(gpuio_graph_index_t index,
                                     gpuio_graph_storage_t storage,
                                     const gpuio_graph_rag_request_t* request,
                                     gpuio_graph_rag_request_handle_t* handle) {
    if (!index || !request || !handle) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_graph_rag_request* req = calloc(1, sizeof(struct ai_graph_rag_request));
    if (!req) return GPUIO_ERROR_NOMEM;
    
    req->index = index;
    req->storage = storage;
    memcpy(&req->params, request, sizeof(gpuio_graph_rag_request_t));
    
    pthread_mutex_init(&req->lock, NULL);
    pthread_cond_init(&req->cond, NULL);
    req->completed = false;
    
    /* Execute scatter phase */
    gpuio_error_t err = gpuio_graph_rag_scatter(index, &request->scatter,
                                                 request->stream,
                                                 &req->candidate_ids,
                                                 &req->num_candidates);
    
    if (err != GPUIO_SUCCESS) {
        pthread_mutex_destroy(&req->lock);
        pthread_cond_destroy(&req->cond);
        free(req);
        return err;
    }
    
    /* Execute gather phase if storage provided */
    if (storage && req->num_candidates > 0) {
        err = gpuio_graph_rag_gather(storage, req->candidate_ids, req->num_candidates,
                                      &request->gather, request->stream,
                                      &req->subgraph, &req->subgraph_size);
        
        if (err != GPUIO_SUCCESS) {
            free(req->candidate_ids);
            pthread_mutex_destroy(&req->lock);
            pthread_cond_destroy(&req->cond);
            free(req);
            return err;
        }
    }
    
    req->completed = true;
    req->result = GPUIO_SUCCESS;
    
    /* Set output */
    request->subgraph = req->subgraph;
    request->subgraph_size = req->subgraph_size;
    
    *handle = (gpuio_graph_rag_request_handle_t)req;
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Wait for a RAG query to complete.
 */
gpuio_error_t gpuio_graph_rag_wait(gpuio_graph_rag_request_handle_t handle,
                                    uint64_t timeout_us) {
    if (!handle) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_graph_rag_request* req = (struct ai_graph_rag_request*)handle;
    
    pthread_mutex_lock(&req->lock);
    
    if (!req->completed) {
        if (timeout_us > 0) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_sec += timeout_us / 1000000;
            ts.tv_nsec += (timeout_us % 1000000) * 1000;
            if (ts.tv_nsec >= 1000000000) {
                ts.tv_sec++;
                ts.tv_nsec -= 1000000000;
            }
            pthread_cond_timedwait(&req->cond, &req->lock, &ts);
        } else {
            pthread_cond_wait(&req->cond, &req->lock);
        }
    }
    
    pthread_mutex_unlock(&req->lock);
    
    return req->result;
}

/**
 * @brief Create graph storage backend.
 */
gpuio_error_t gpuio_graph_storage_create(gpuio_ai_context_t ai_ctx,
                                          const char* uri,
                                          gpuio_graph_storage_t* storage) {
    if (!ai_ctx || !uri || !storage) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct ai_graph_storage* stor = calloc(1, sizeof(struct ai_graph_storage));
    if (!stor) return GPUIO_ERROR_NOMEM;
    
    if (ai_graph_storage_init(stor, ai_ctx, uri) != 0) {
        free(stor);
        return GPUIO_ERROR_GENERAL;
    }
    
    *storage = (gpuio_graph_storage_t)stor;
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Destroy graph storage backend.
 */
gpuio_error_t gpuio_graph_storage_destroy(gpuio_graph_storage_t storage) {
    if (!storage) return GPUIO_ERROR_INVALID_ARG;
    
    struct ai_graph_storage* stor = (struct ai_graph_storage*)storage;
    
    ai_graph_storage_cleanup(stor);
    free(stor);
    
    return GPUIO_SUCCESS;
}
