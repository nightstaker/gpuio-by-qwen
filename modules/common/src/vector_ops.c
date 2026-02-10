/**
 * @file vector_ops.c
 * @brief Vector operations implementation
 * @version 1.1.0
 */

#include "vector_ops.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Basic similarity functions
 * ============================================================================ */

float vec_cosine_similarity_f32(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

float vec_cosine_similarity_norm_f32(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    
    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    
    return dot;  /* For normalized vectors, dot = cosine similarity */
}

float vec_dot_f32(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    
    float dot = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    
    return dot;
}

float vec_euclidean_sq_f32(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    
    float dist = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    
    return dist;
}

float vec_euclidean_f32(const float* a, const float* b, uint32_t dim) {
    return sqrtf(vec_euclidean_sq_f32(a, b, dim));
}

float vec_manhattan_f32(const float* a, const float* b, uint32_t dim) {
    if (!a || !b || dim == 0) return 0.0f;
    
    float dist = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        dist += fabsf(a[i] - b[i]);
    }
    
    return dist;
}

uint32_t vec_hamming_distance(const uint8_t* a, const uint8_t* b, uint32_t num_bits) {
    if (!a || !b || num_bits == 0) return 0;
    
    uint32_t distance = 0;
    uint32_t num_bytes = (num_bits + 7) / 8;
    
    for (uint32_t i = 0; i < num_bytes; i++) {
        uint8_t xor = a[i] ^ b[i];
        /* Count bits set in xor */
        while (xor) {
            distance += xor & 1;
            xor >>= 1;
        }
    }
    
    return distance;
}

/* ============================================================================
 * Vector properties
 * ============================================================================ */

float vec_norm_f32(const float* v, uint32_t dim) {
    return sqrtf(vec_norm_sq_f32(v, dim));
}

float vec_norm_sq_f32(const float* v, uint32_t dim) {
    if (!v || dim == 0) return 0.0f;
    
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        norm += v[i] * v[i];
    }
    
    return norm;
}

bool vec_normalize_f32(const float* in, float* out, uint32_t dim) {
    if (!in || !out || dim == 0) return false;
    
    float norm = vec_norm_f32(in, dim);
    if (norm < 1e-8f) return false;
    
    float inv_norm = 1.0f / norm;
    for (uint32_t i = 0; i < dim; i++) {
        out[i] = in[i] * inv_norm;
    }
    
    return true;
}

float vec_mean_f32(const float* v, uint32_t dim) {
    if (!v || dim == 0) return 0.0f;
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += v[i];
    }
    
    return sum / dim;
}

float vec_variance_f32(const float* v, uint32_t dim) {
    if (!v || dim == 0) return 0.0f;
    
    float mean = vec_mean_f32(v, dim);
    float var = 0.0f;
    
    for (uint32_t i = 0; i < dim; i++) {
        float diff = v[i] - mean;
        var += diff * diff;
    }
    
    return var / dim;
}

bool vec_standardize_f32(const float* in, float* out, uint32_t dim) {
    if (!in || !out || dim == 0) return false;
    
    float mean = vec_mean_f32(in, dim);
    float var = vec_variance_f32(in, dim);
    
    if (var < 1e-8f) return false;
    
    float std = sqrtf(var);
    for (uint32_t i = 0; i < dim; i++) {
        out[i] = (in[i] - mean) / std;
    }
    
    return true;
}

/* ============================================================================
 * Batch operations
 * ============================================================================ */

static float compute_similarity(const float* a, const float* b, uint32_t dim, 
                                 vec_sim_metric_t metric) {
    switch (metric) {
        case VEC_SIM_COSINE:
            return vec_cosine_similarity_f32(a, b, dim);
        case VEC_SIM_DOT:
            return vec_dot_f32(a, b, dim);
        case VEC_SIM_EUCLIDEAN:
            return -vec_euclidean_f32(a, b, dim);  /* Negate for consistency (higher = more similar) */
        case VEC_SIM_MANHATTAN:
            return -vec_manhattan_f32(a, b, dim);
        default:
            return 0.0f;
    }
}

typedef struct {
    vec_search_result_t* results;
    int capacity;
    int count;
} search_ctx_t;

static bool search_callback(vec_search_result_t* result, search_ctx_t* ctx) {
    if (ctx->count < ctx->capacity) {
        ctx->results[ctx->count] = *result;
        ctx->count++;
        
        /* Simple insertion sort to maintain order */
        for (int i = ctx->count - 1; i > 0; i--) {
            if (ctx->results[i].similarity > ctx->results[i-1].similarity) {
                vec_search_result_t tmp = ctx->results[i];
                ctx->results[i] = ctx->results[i-1];
                ctx->results[i-1] = tmp;
            } else {
                break;
            }
        }
        return true;
    }
    
    /* Check if this result is better than the worst in our top-k */
    if (result->similarity > ctx->results[ctx->capacity - 1].similarity) {
        ctx->results[ctx->capacity - 1] = *result;
        
        /* Bubble up to maintain order */
        for (int i = ctx->capacity - 1; i > 0; i--) {
            if (ctx->results[i].similarity > ctx->results[i-1].similarity) {
                vec_search_result_t tmp = ctx->results[i];
                ctx->results[i] = ctx->results[i-1];
                ctx->results[i-1] = tmp;
            } else {
                break;
            }
        }
    }
    
    return true;
}

int vec_search_top_k(const float* query, uint32_t query_dim,
                     const float* vectors, uint64_t num_vectors,
                     int top_k, vec_sim_metric_t metric,
                     float threshold,
                     vec_search_result_t* results, int* num_results) {
    if (!query || !vectors || !results || !num_results || top_k <= 0) {
        return -1;
    }
    
    search_ctx_t ctx = { results, top_k, 0 };
    
    for (uint64_t i = 0; i < num_vectors; i++) {
        const float* vec = vectors + i * query_dim;
        float sim = compute_similarity(query, vec, query_dim, metric);
        
        if (sim >= threshold) {
            vec_search_result_t result = { i, sim, NULL };
            search_callback(&result, &ctx);
        }
    }
    
    *num_results = ctx.count;
    return 0;
}

int vec_search_top_k_indexed(const float* query, uint32_t query_dim,
                             uint64_t num_vectors,
                             vec_get_fn_t get_vector,
                             vec_get_user_data_fn_t get_user_data,
                             void* user_context,
                             int top_k, vec_sim_metric_t metric,
                             float threshold,
                             vec_search_result_t* results, int* num_results) {
    if (!query || !get_vector || !results || !num_results || top_k <= 0) {
        return -1;
    }
    
    search_ctx_t ctx = { results, top_k, 0 };
    
    for (uint64_t i = 0; i < num_vectors; i++) {
        uint32_t dim = query_dim;
        const float* vec = get_vector(i, &dim, user_context);
        if (!vec || dim != query_dim) continue;
        
        float sim = compute_similarity(query, vec, query_dim, metric);
        
        if (sim >= threshold) {
            vec_search_result_t result;
            result.index = i;
            result.similarity = sim;
            result.user_data = get_user_data ? get_user_data(i, user_context) : NULL;
            search_callback(&result, &ctx);
        }
    }
    
    *num_results = ctx.count;
    return 0;
}

int vec_batch_similarity(const float* set_a, uint64_t num_a,
                         const float* set_b, uint64_t num_b,
                         uint32_t dim, vec_sim_metric_t metric,
                         float* similarities) {
    if (!set_a || !set_b || !similarities || dim == 0) {
        return -1;
    }
    
    for (uint64_t i = 0; i < num_a; i++) {
        const float* vec_a = set_a + i * dim;
        
        for (uint64_t j = 0; j < num_b; j++) {
            const float* vec_b = set_b + j * dim;
            similarities[i * num_b + j] = compute_similarity(vec_a, vec_b, dim, metric);
        }
    }
    
    return 0;
}

/* ============================================================================
 * Comparison utilities
 * ============================================================================ */

int vec_result_compare_desc(const void* a, const void* b) {
    const vec_search_result_t* ra = (const vec_search_result_t*)a;
    const vec_search_result_t* rb = (const vec_search_result_t*)b;
    
    if (rb->similarity > ra->similarity) return 1;
    if (rb->similarity < ra->similarity) return -1;
    return 0;
}

int vec_result_compare_asc(const void* a, const void* b) {
    const vec_search_result_t* ra = (const vec_search_result_t*)a;
    const vec_search_result_t* rb = (const vec_search_result_t*)b;
    
    if (ra->similarity > rb->similarity) return 1;
    if (ra->similarity < rb->similarity) return -1;
    return 0;
}

void vec_results_sort_best_first(vec_search_result_t* results, int num_results) {
    if (!results || num_results <= 0) return;
    qsort(results, (size_t)num_results, sizeof(vec_search_result_t), vec_result_compare_desc);
}

void vec_results_sort_worst_first(vec_search_result_t* results, int num_results) {
    if (!results || num_results <= 0) return;
    qsort(results, (size_t)num_results, sizeof(vec_search_result_t), vec_result_compare_asc);
}
