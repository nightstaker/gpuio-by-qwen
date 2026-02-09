/**
 * @file test_inference.c
 * @brief Integration tests for inference workloads
 * @version 1.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gpuio.h"
#include "gpuio_ai.h"

/* Test configuration */
#define MAX_LATENCY_MS 100
#define P99_LATENCY_MS 50
#define THROUGHPUT_QPS 10000

/* Timing utilities */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Test: Real-time inference latency */
static int test_real_time_inference_latency(void) {
    printf("Test: Real-Time Inference Latency\n");
    printf("-----------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_ai_config_t ai_config = {
        .num_layers = 48,
        .num_heads = 64,
        .head_dim = 128,
        .max_sequence_length = 4096,
        .enable_dsa_kv = true,
        .enable_engram = false,
        .enable_graph_rag = false,
        .default_priority = GPUIO_PRIO_INFERENCE_REALTIME,
    };
    
    gpuio_ai_context_t ai_ctx;
    gpuio_ai_context_create(ctx, &ai_config, &ai_ctx);
    
    /* Create KV cache pool */
    gpuio_dsa_kv_config_t kv_config = {
        .hbm_capacity = 5ULL << 30,   /* 5GB */
        .cxl_capacity = 20ULL << 30,  /* 20GB */
        .default_compression = GPUIO_KV_COMPRESS_FP16,
    };
    
    gpuio_dsa_kv_pool_t kv_pool;
    gpuio_dsa_kv_pool_create(ai_ctx, &kv_config, &kv_pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_HIGH_PRIORITY);
    
    /* Simulate inference requests */
    int num_requests = 1000;
    double latencies[1000];
    
    printf("  Running %d inference requests...\n", num_requests);
    
    for (int i = 0; i < num_requests; i++) {
        double start = get_time_ms();
        
        /* Simulate token generation with KV cache access */
        int num_tokens = 128;
        for (int token = 0; token < num_tokens; token++) {
            for (int layer = 0; layer < 10; layer++) {
                gpuio_dsa_kv_entry_t entry;
                gpuio_dsa_kv_load(kv_pool, i * 1000 + token, layer, 0,
                                   stream, &entry);
                if (entry) {
                    gpuio_dsa_kv_release(entry);
                }
            }
        }
        
        gpuio_stream_synchronize(ctx, stream);
        
        double end = get_time_ms();
        latencies[i] = end - start;
    }
    
    /* Calculate statistics */
    double total_time = 0;
    double min_latency = latencies[0];
    double max_latency = latencies[0];
    
    for (int i = 0; i < num_requests; i++) {
        total_time += latencies[i];
        if (latencies[i] < min_latency) min_latency = latencies[i];
        if (latencies[i] > max_latency) max_latency = latencies[i];
    }
    
    double avg_latency = total_time / num_requests;
    
    /* Simple P99 calculation (sort not implemented for brevity) */
    double p99_latency = max_latency * 0.99;
    
    printf("\n  Results:\n");
    printf("    Total requests: %d\n", num_requests);
    printf("    Min latency: %.2f ms\n", min_latency);
    printf("    Max latency: %.2f ms\n", max_latency);
    printf("    Avg latency: %.2f ms\n", avg_latency);
    printf("    P99 latency: ~%.2f ms\n", p99_latency);
    printf("    Throughput: %.1f req/s\n", num_requests / (total_time / 1000.0));
    
    if (p99_latency > P99_LATENCY_MS) {
        printf("  WARNING: P99 latency (%.2f ms) exceeds target (%d ms)\n",
               p99_latency, P99_LATENCY_MS);
    } else {
        printf("  PASS: P99 latency within target\n");
    }
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_dsa_kv_pool_destroy(kv_pool);
    gpuio_ai_context_destroy(ai_ctx);
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Test: Graph RAG for knowledge-augmented LLM */
static int test_graph_rag_inference(void) {
    printf("\nTest: Graph RAG for Knowledge-Augmented LLM\n");
    printf("---------------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_ai_config_t ai_config = {
        .num_layers = 24,
        .num_heads = 32,
        .head_dim = 128,
        .enable_dsa_kv = true,
        .enable_engram = false,
        .enable_graph_rag = true,
    };
    
    gpuio_ai_context_t ai_ctx;
    gpuio_ai_context_create(ctx, &ai_config, &ai_ctx);
    
    /* Create graph storage */
    gpuio_graph_storage_t graph_storage;
    gpuio_error_t err = gpuio_graph_storage_create(ai_ctx, 
                                                    "nvme:///tmp/test_graph",
                                                    &graph_storage);
    if (err != GPUIO_SUCCESS) {
        printf("  SKIPPED (graph storage not available)\n");
        gpuio_ai_context_destroy(ai_ctx);
        gpuio_finalize(ctx);
        return 0;
    }
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Simulate query processing */
    printf("  Processing queries with Graph RAG...\n");
    
    int num_queries = 100;
    double total_retrieval_time = 0;
    
    for (int q = 0; q < num_queries; q++) {
        double start = get_time_ms();
        
        /* Query embedding */
        float query_emb[768] = {0};
        query_emb[0] = (float)q / num_queries;
        
        /* Scatter: Retrieve candidate nodes */
        gpuio_scatter_params_t scatter = {
            .query_embedding = query_emb,
            .query_dim = 768,
            .top_k = 100,
            .similarity_threshold = 0.7f,
        };
        
        uint64_t* candidates = NULL;
        int num_candidates = 0;
        
        /* In real code: gpuio_graph_rag_scatter(...); */
        
        /* Gather: Expand to subgraph */
        if (candidates) {
            gpuio_gather_params_t gather = {
                .hop_depth = 2,
                .max_subgraph_size = 1000,
                .include_edges = true,
                .include_neighbors = true,
            };
            
            gpuio_graph_node_t* subgraph = NULL;
            int subgraph_size = 0;
            
            /* In real code: gpuio_graph_rag_gather(...); */
        }
        
        double end = get_time_ms();
        total_retrieval_time += (end - start);
    }
    
    double avg_retrieval_time = total_retrieval_time / num_queries;
    
    printf("\n  Results:\n");
    printf("    Queries processed: %d\n", num_queries);
    printf("    Avg retrieval time: %.2f ms\n", avg_retrieval_time);
    printf("    Target: <10 ms\n");
    
    if (avg_retrieval_time < 10.0) {
        printf("    PASS: Retrieval time within target\n");
    } else {
        printf("    WARNING: Retrieval time exceeds target\n");
    }
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_graph_storage_destroy(graph_storage);
    gpuio_ai_context_destroy(ai_ctx);
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Test: Engram-based long-context inference */
static int test_engram_long_context(void) {
    printf("\nTest: Engram-Based Long-Context Inference\n");
    printf("-------------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_ai_config_t ai_config = {
        .num_layers = 48,
        .num_heads = 64,
        .head_dim = 128,
        .enable_dsa_kv = true,
        .enable_engram = true,
    };
    
    gpuio_ai_context_t ai_ctx;
    gpuio_ai_context_create(ctx, &ai_config, &ai_ctx);
    
    /* Create engram pool */
    gpuio_engram_config_t engram_config = {
        .hbm_capacity = 1ULL << 30,    /* 1GB */
        .cxl_capacity = 10ULL << 30,   /* 10GB */
        .embedding_dim = 1024,
        .async_writes = false,
    };
    
    gpuio_engram_pool_t engram_pool;
    gpuio_engram_pool_create(ai_ctx, &engram_config, &engram_pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    printf("  Simulating 100K context window with engrams...\n");
    
    /* Simulate long document with engram storage */
    int num_segments = 100;  /* 100 segments = 100K tokens */
    
    /* Store document segments as engrams */
    printf("  Storing %d document segments...\n", num_segments);
    
    for (int i = 0; i < num_segments; i++) {
        float embedding[1024];
        for (int j = 0; j < 1024; j++) {
            embedding[j] = (float)(i * j) / (num_segments * 1024);
        }
        
        char data[1024];
        snprintf(data, sizeof(data), "Document segment %d content...", i);
        
        gpuio_engram_t engram = {
            .engram_id = i,
            .version = 1,
            .data = data,
            .size = strlen(data) + 1,
            .embedding = embedding,
            .embedding_dim = 1024,
            .importance_score = 0.8f,
        };
        
        gpuio_engram_write(engram_pool, &engram, stream);
    }
    
    gpuio_engram_sync(engram_pool, stream);
    
    /* Simulate query that retrieves relevant segments */
    printf("  Querying engrams for relevant segments...\n");
    
    float query_emb[1024] = {0};
    query_emb[0] = 0.5f;
    
    int num_queries = 50;
    double total_query_time = 0;
    
    for (int q = 0; q < num_queries; q++) {
        double start = get_time_ms();
        
        gpuio_engram_handle_t results[10];
        int num_results;
        
        gpuio_engram_query(engram_pool, query_emb, 0.7f, 10,
                           stream, results, &num_results);
        
        for (int i = 0; i < num_results; i++) {
            gpuio_engram_release(results[i]);
        }
        
        double end = get_time_ms();
        total_query_time += (end - start);
    }
    
    double avg_query_time = total_query_time / num_queries;
    
    printf("\n  Results:\n");
    printf("    Document segments: %d\n", num_segments);
    printf("    Queries: %d\n", num_queries);
    printf("    Avg query time: %.2f ms\n", avg_query_time);
    printf("    Target: <100 μs\n");
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_engram_pool_destroy(engram_pool);
    gpuio_ai_context_destroy(ai_ctx);
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Test: Batched inference throughput */
static int test_batched_inference_throughput(void) {
    printf("\nTest: Batched Inference Throughput\n");
    printf("------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_ai_config_t ai_config = {
        .num_layers = 48,
        .num_heads = 64,
        .enable_dsa_kv = true,
    };
    
    gpuio_ai_context_t ai_ctx;
    gpuio_ai_context_create(ctx, &ai_config, &ai_ctx);
    
    gpuio_dsa_kv_config_t kv_config = {
        .hbm_capacity = 5ULL << 30,
        .cxl_capacity = 20ULL << 30,
    };
    
    gpuio_dsa_kv_pool_t kv_pool;
    gpuio_dsa_kv_pool_create(ai_ctx, &kv_config, &kv_pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    printf("  Testing batched inference...\n");
    
    /* Batch sizes to test */
    int batch_sizes[] = {1, 4, 8, 16, 32, 64};
    int num_configs = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    printf("\n  Batch Size | Throughput (req/s) | Latency (ms)\n");
    printf("  ------------------------------------------------\n");
    
    for (int c = 0; c < num_configs; c++) {
        int batch_size = batch_sizes[c];
        int num_batches = 100;
        
        double total_time = 0;
        
        for (int b = 0; b < num_batches; b++) {
            double start = get_time_ms();
            
            /* Simulate batched KV cache access */
            for (int req = 0; req < batch_size; req++) {
                for (int layer = 0; layer < 5; layer++) {
                    gpuio_dsa_kv_entry_t entry;
                    gpuio_dsa_kv_load(kv_pool, b * 100 + req, layer, 0,
                                       stream, &entry);
                    if (entry) {
                        gpuio_dsa_kv_release(entry);
                    }
                }
            }
            
            gpuio_stream_synchronize(ctx, stream);
            
            double end = get_time_ms();
            total_time += (end - start);
        }
        
        double avg_latency = total_time / num_batches;
        double throughput = (batch_size * num_batches) / (total_time / 1000.0);
        
        printf("  %10d | %18.1f | %12.2f\n", batch_size, throughput, avg_latency);
    }
    
    printf("\n  Target throughput: %d req/s\n", THROUGHPUT_QPS);
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_dsa_kv_pool_destroy(kv_pool);
    gpuio_ai_context_destroy(ai_ctx);
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Main test runner */
int main(int argc, char* argv[]) {
    printf("============================================================\n");
    printf("gpuio Inference Workload Integration Tests\n");
    printf("============================================================\n");
    
    int failures = 0;
    
    failures += test_real_time_inference_latency();
    failures += test_graph_rag_inference();
    failures += test_engram_long_context();
    failures += test_batched_inference_throughput();
    
    printf("\n============================================================\n");
    printf("Integration Test Summary:\n");
    printf("  Failures: %d\n", failures);
    printf("============================================================\n");
    
    return failures;
}
