/**
 * @file vector_ops.h
 * @brief Vector operations for AI/ML workloads
 * @version 1.1.0
 * 
 * Common vector operations including similarity metrics, distance calculations,
 * and batch operations. Extracted from duplicated code in graph_rag.c, engram.c.
 */

#ifndef VECTOR_OPS_H
#define VECTOR_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type definitions
 * ============================================================================ */

/**
 * @brief Vector similarity/distance metric type.
 */
typedef enum {
    VEC_SIM_COSINE,       /**< Cosine similarity (1 = identical, 0 = orthogonal) */
    VEC_SIM_DOT,          /**< Dot product (unbounded) */
    VEC_SIM_EUCLIDEAN,    /**< Euclidean distance (0 = identical, increases with distance) */
    VEC_SIM_MANHATTAN,    /**< Manhattan/L1 distance */
    VEC_SIM_HAMMING,      /**< Hamming distance for binary vectors */
    VEC_SIM_JACCARD,      /**< Jaccard similarity for sets */
} vec_sim_metric_t;

/**
 * @brief Result from similarity search.
 */
typedef struct {
    uint64_t index;       /**< Index of the matching vector */
    float similarity;     /**< Similarity score */
    void* user_data;      /**< Optional user data associated with the vector */
} vec_search_result_t;

/* ============================================================================
 * Basic similarity functions
 * ============================================================================ */

/**
 * @brief Compute cosine similarity between two float vectors.
 * 
 * Returns value in range [-1, 1] where 1 means identical direction,
 * 0 means orthogonal, -1 means opposite direction.
 * 
 * @param a First vector
 * @param b Second vector  
 * @param dim Vector dimension
 * @return Cosine similarity
 */
float vec_cosine_similarity_f32(const float* a, const float* b, uint32_t dim);

/**
 * @brief Compute cosine similarity between two normalized vectors.
 * 
 * Assumes both vectors are already normalized (length = 1).
 * Faster than vec_cosine_similarity_f32 for pre-normalized vectors.
 * 
 * @param a First normalized vector
 * @param b Second normalized vector
 * @param dim Vector dimension
 * @return Cosine similarity (= dot product for normalized vectors)
 */
float vec_cosine_similarity_norm_f32(const float* a, const float* b, uint32_t dim);

/**
 * @brief Compute dot product of two float vectors.
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Dot product
 */
float vec_dot_f32(const float* a, const float* b, uint32_t dim);

/**
 * @brief Compute squared Euclidean distance between two vectors.
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Squared Euclidean distance
 */
float vec_euclidean_sq_f32(const float* a, const float* b, uint32_t dim);

/**
 * @brief Compute Euclidean distance between two vectors.
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Euclidean distance
 */
float vec_euclidean_f32(const float* a, const float* b, uint32_t dim);

/**
 * @brief Compute Manhattan/L1 distance between two vectors.
 * @param a First vector
 * @param b Second vector
 * @param dim Vector dimension
 * @return Manhattan distance
 */
float vec_manhattan_f32(const float* a, const float* b, uint32_t dim);

/**
 * @brief Compute Hamming distance between two binary vectors.
 * @param a First vector (packed bits)
 * @param b Second vector (packed bits)
 * @param num_bits Number of bits
 * @return Hamming distance (number of differing bits)
 */
uint32_t vec_hamming_distance(const uint8_t* a, const uint8_t* b, uint32_t num_bits);

/* ============================================================================
 * Vector properties
 * ============================================================================ */

/**
 * @brief Compute L2 norm (magnitude) of a vector.
 * @param v Vector
 * @param dim Dimension
 * @return L2 norm
 */
float vec_norm_f32(const float* v, uint32_t dim);

/**
 * @brief Compute squared L2 norm of a vector.
 * @param v Vector
 * @param dim Dimension
 * @return Squared L2 norm
 */
float vec_norm_sq_f32(const float* v, uint32_t dim);

/**
 * @brief Normalize a vector to unit length.
 * @param in Input vector
 * @param out Output vector (can be same as in)
 * @param dim Dimension
 * @return true if successful, false if vector has zero norm
 */
bool vec_normalize_f32(const float* in, float* out, uint32_t dim);

/**
 * @brief Compute mean of a vector.
 * @param v Vector
 * @param dim Dimension
 * @return Mean value
 */
float vec_mean_f32(const float* v, uint32_t dim);

/**
 * @brief Compute variance of a vector.
 * @param v Vector
 * @param dim Dimension
 * @return Variance
 */
float vec_variance_f32(const float* v, uint32_t dim);

/**
 * @brief Standardize a vector (zero mean, unit variance).
 * @param in Input vector
 * @param out Output vector (can be same as in)
 * @param dim Dimension
 * @return true if successful
 */
bool vec_standardize_f32(const float* in, float* out, uint32_t dim);

/* ============================================================================
 * Batch operations
 * ============================================================================ */

/**
 * @brief Find top-k most similar vectors using linear search.
 * 
 * Performs brute-force linear search over all vectors to find the
 * k most similar to the query vector.
 * 
 * @param query Query vector
 * @param query_dim Dimension of query vector
 * @param vectors Array of vectors (concatenated, each query_dim elements)
 * @param num_vectors Number of vectors
 * @param top_k Number of results to return
 * @param metric Similarity metric to use
 * @param threshold Minimum similarity threshold (results below this are filtered)
 * @param results Output array of results (must hold at least top_k elements)
 * @param num_results Output number of results found
 * @return 0 on success, -1 on error
 */
int vec_search_top_k(const float* query, uint32_t query_dim,
                     const float* vectors, uint64_t num_vectors,
                     int top_k, vec_sim_metric_t metric,
                     float threshold,
                     vec_search_result_t* results, int* num_results);

/**
 * @brief Find top-k most similar vectors using indexed data.
 * 
 * Similar to vec_search_top_k but vectors are accessed via a callback,
 * allowing lazy loading or custom storage.
 * 
 * @param query Query vector
 * @param query_dim Dimension of query vector
 * @param num_vectors Number of vectors in the collection
 * @param get_vector Callback to retrieve vector by index
 * @param get_user_data Callback to get user data for a vector (can be NULL)
 * @param user_context Context passed to callbacks
 * @param top_k Number of results
 * @param metric Similarity metric
 * @param threshold Minimum similarity threshold
 * @param results Output results
 * @param num_results Output count
 * @return 0 on success, -1 on error
 */
typedef const float* (*vec_get_fn_t)(uint64_t index, uint32_t* out_dim, void* ctx);
typedef void* (*vec_get_user_data_fn_t)(uint64_t index, void* ctx);

int vec_search_top_k_indexed(const float* query, uint32_t query_dim,
                             uint64_t num_vectors,
                             vec_get_fn_t get_vector,
                             vec_get_user_data_fn_t get_user_data,
                             void* user_context,
                             int top_k, vec_sim_metric_t metric,
                             float threshold,
                             vec_search_result_t* results, int* num_results);

/**
 * @brief Batch compute pairwise similarities.
 * 
 * Computes all pairwise similarities between two sets of vectors.
 * Output matrix: similarities[i * num_b + j] = sim(a[i], b[j])
 * 
 * @param set_a First set of vectors (concatenated)
 * @param num_a Number of vectors in set A
 * @param set_b Second set of vectors (concatenated)
 * @param num_b Number of vectors in set B
 * @param dim Vector dimension
 * @param metric Similarity metric
 * @param similarities Output matrix (size num_a * num_b)
 * @return 0 on success, -1 on error
 */
int vec_batch_similarity(const float* set_a, uint64_t num_a,
                         const float* set_b, uint64_t num_b,
                         uint32_t dim, vec_sim_metric_t metric,
                         float* similarities);

/* ============================================================================
 * Comparison utilities
 * ============================================================================ */

/**
 * @brief Compare two results by similarity (descending) for sorting.
 * @param a First result
 * @param b Second result
 * @return Comparison result for qsort
 */
int vec_result_compare_desc(const void* a, const void* b);

/**
 * @brief Compare two results by similarity (ascending) for sorting.
 * @param a First result
 * @param b Second result
 * @return Comparison result for qsort
 */
int vec_result_compare_asc(const void* a, const void* b);

/**
 * @brief Sort search results by similarity (descending - best first).
 * @param results Results array
 * @param num_results Number of results
 */
void vec_results_sort_best_first(vec_search_result_t* results, int num_results);

/**
 * @brief Sort search results by similarity (ascending - worst first).
 * @param results Results array
 * @param num_results Number of results
 */
void vec_results_sort_worst_first(vec_search_result_t* results, int num_results);

#ifdef __cplusplus
}
#endif

#endif /* VECTOR_OPS_H */
