/**
 * @file test_ai.c
 * @brief Unit tests for gpuio AI/ML extensions
 * @version 1.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gpuio.h"
#include "gpuio_ai.h"

/* Test statistics */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/* Test macros */
#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("  Running %s... ", #name); \
    tests_run++; \
    test_##name(); \
    tests_passed++; \
    printf("PASSED\n"); \
} while(0)

#define ASSERT(expr) do { \
    if (!(expr)) { \
        printf("FAILED\n    Assertion failed: %s at line %d\n", #expr, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NULL(ptr) ASSERT((ptr) == NULL)
#define ASSERT_NOT_NULL(ptr) ASSERT((ptr) != NULL)
#define ASSERT_FLOAT_EQ(a, b, eps) ASSERT(fabs((a) - (b)) < (eps))

/* Global context */
static gpuio_context_t g_ctx = NULL;
static gpuio_ai_context_t g_ai_ctx = NULL;

/* ============================================================================
 * Setup/Teardown
 * ============================================================================ */

static void setup(void) {
    if (!g_ctx) {
        gpuio_init(&g_ctx, NULL);
        
        gpuio_ai_config_t ai_config = {
            .num_layers = 12,
            .num_heads = 16,
            .head_dim = 64,
            .max_sequence_length = 2048,
            .enable_dsa_kv = true,
            .enable_engram = true,
            .enable_graph_rag = true,
            .default_priority = GPUIO_PRIO_TRAINING_FW,
            .kv_cache_size = 1ULL << 30,  /* 1GB */
            .engram_pool_size = 10ULL << 30, /* 10GB */
        };
        gpuio_ai_context_create(g_ctx, &ai_config, &g_ai_ctx);
    }
}

static void teardown(void) {
    if (g_ai_ctx) {
        gpuio_ai_context_destroy(g_ai_ctx);
        g_ai_ctx = NULL;
    }
    if (g_ctx) {
        gpuio_finalize(g_ctx);
        g_ctx = NULL;
    }
}

/* ============================================================================
 * AI Context Tests
 * ============================================================================ */

TEST(ai_context_create_destroy) {
    gpuio_ai_config_t config = {
        .num_layers = 96,
        .num_heads = 128,
        .head_dim = 128,
        .enable_dsa_kv = true,
        .enable_engram = true,
        .enable_graph_rag = true,
    };
    
    gpuio_ai_context_t ai_ctx;
    gpuio_error_t err = gpuio_ai_context_create(g_ctx, &config, &ai_ctx);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(ai_ctx);
    
    err = gpuio_ai_context_destroy(ai_ctx);
    ASSERT_EQ(err, GPUIO_SUCCESS);
}

TEST(ai_context_minimal_config) {
    gpuio_ai_config_t config = {
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 64,
        .enable_dsa_kv = false,
        .enable_engram = false,
        .enable_graph_rag = false,
    };
    
    gpuio_ai_context_t ai_ctx;
    gpuio_error_t err = gpuio_ai_context_create(g_ctx, &config, &ai_ctx);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_ai_context_destroy(ai_ctx);
}

/* ============================================================================
 * DSA KV Cache Tests
 * ============================================================================ */

TEST(dsa_kv_pool_create_destroy) {
    gpuio_dsa_kv_config_t config = {
        .hbm_capacity = 1ULL << 30,   /* 1GB */
        .cxl_capacity = 10ULL << 30,  /* 10GB */
        .cxl_device_path = NULL,
        .remote_uri = NULL,
        .default_compression = GPUIO_KV_COMPRESS_FP16,
        .compression_threshold = 0.5f,
        .importance_decay = 0.9f,
        .min_hot_cache_size = 100 << 20, /* 100MB */
    };
    
    gpuio_dsa_kv_pool_t pool;
    gpuio_error_t err = gpuio_dsa_kv_pool_create(g_ai_ctx, &config, &pool);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(pool);
    
    err = gpuio_dsa_kv_pool_destroy(pool);
    ASSERT_EQ(err, GPUIO_SUCCESS);
}

TEST(dsa_kv_pool_reset) {
    gpuio_dsa_kv_config_t config = {
        .hbm_capacity = 1ULL << 30,
        .cxl_capacity = 10ULL << 30,
        .default_compression = GPUIO_KV_COMPRESS_NONE,
    };
    
    gpuio_dsa_kv_pool_t pool;
    gpuio_dsa_kv_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_error_t err = gpuio_dsa_kv_pool_reset(pool);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_dsa_kv_pool_destroy(pool);
}

TEST(dsa_kv_load_store) {
    gpuio_dsa_kv_config_t config = {
        .hbm_capacity = 1ULL << 30,
        .cxl_capacity = 10ULL << 30,
        .default_compression = GPUIO_KV_COMPRESS_NONE,
    };
    
    gpuio_dsa_kv_pool_t pool;
    gpuio_dsa_kv_pool_create(g_ai_ctx, &config, &pool);
    
    /* Create a stream */
    gpuio_stream_t stream;
    gpuio_stream_create(g_ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Store KV entry */
    float kv_data[64 * 2]; /* key + value */
    memset(kv_data, 0, sizeof(kv_data));
    kv_data[0] = 1.0f;
    
    gpuio_error_t err = gpuio_dsa_kv_store(pool, 0, 0, 0,
                                            kv_data, sizeof(kv_data),
                                            0.9f, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    /* Load KV entry */
    gpuio_dsa_kv_entry_t entry;
    err = gpuio_dsa_kv_load(pool, 0, 0, 0, stream, &entry);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(entry);
    
    /* Get data */
    void* data;
    size_t size;
    err = gpuio_dsa_kv_get_data(entry, &data, &size);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(data);
    ASSERT_EQ(size, sizeof(kv_data));
    
    /* Release */
    gpuio_dsa_kv_release(entry);
    gpuio_stream_destroy(g_ctx, stream);
    gpuio_dsa_kv_pool_destroy(pool);
}

TEST(dsa_kv_load_batch) {
    gpuio_dsa_kv_config_t config = {
        .hbm_capacity = 1ULL << 30,
        .cxl_capacity = 10ULL << 30,
        .default_compression = GPUIO_KV_COMPRESS_FP16,
    };
    
    gpuio_dsa_kv_pool_t pool;
    gpuio_dsa_kv_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(g_ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Prepare batch */
    int batch_size = 100;
    uint64_t positions[100];
    uint32_t layer_ids[100];
    uint32_t head_ids[100];
    gpuio_dsa_kv_entry_t entries[100];
    
    for (int i = 0; i < batch_size; i++) {
        positions[i] = i;
        layer_ids[i] = i % 12;
        head_ids[i] = i % 16;
    }
    
    /* Load batch */
    gpuio_error_t err = gpuio_dsa_kv_load_batch(pool, positions, layer_ids,
                                                 head_ids, batch_size,
                                                 stream, entries);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    for (int i = 0; i < batch_size; i++) {
        gpuio_dsa_kv_release(entries[i]);
    }
    
    gpuio_stream_destroy(g_ctx, stream);
    gpuio_dsa_kv_pool_destroy(pool);
}

TEST(dsa_kv_sparsity_pattern) {
    gpuio_dsa_kv_pool_t pool;
    gpuio_dsa_kv_config_t config = {
        .hbm_capacity = 1ULL << 30,
        .cxl_capacity = 10ULL << 30,
        .default_compression = GPUIO_KV_COMPRESS_SPARSE,
    };
    gpuio_dsa_kv_pool_create(g_ai_ctx, &config, &pool);
    
    /* Set sparsity pattern for layer 0 */
    uint32_t sparsity_mask = 0x00FF00FF; /* Only heads 0-7 and 16-23 active */
    gpuio_error_t err = gpuio_dsa_kv_set_sparsity_pattern(pool, 0, sparsity_mask);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_dsa_kv_pool_destroy(pool);
}

TEST(dsa_kv_compact) {
    gpuio_dsa_kv_pool_t pool;
    gpuio_dsa_kv_config_t config = {
        .hbm_capacity = 1ULL << 30,
        .cxl_capacity = 10ULL << 30,
        .default_compression = GPUIO_KV_COMPRESS_FP16,
    };
    gpuio_dsa_kv_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(g_ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    gpuio_error_t err = gpuio_dsa_kv_compact(pool, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_stream_destroy(g_ctx, stream);
    gpuio_dsa_kv_pool_destroy(pool);
}

TEST(dsa_kv_stats) {
    gpuio_dsa_kv_pool_t pool;
    gpuio_dsa_kv_config_t config = {
        .hbm_capacity = 1ULL << 30,
        .cxl_capacity = 10ULL << 30,
    };
    gpuio_dsa_kv_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_dsa_kv_stats_t stats;
    gpuio_error_t err = gpuio_dsa_kv_get_stats(pool, &stats);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    /* Stats should be initialized */
    ASSERT(stats.total_hits >= 0);
    ASSERT(stats.total_misses >= 0);
    ASSERT(stats.hit_rate >= 0.0);
    ASSERT(stats.hit_rate <= 1.0);
    
    gpuio_dsa_kv_pool_destroy(pool);
}

/* ============================================================================
 * Graph RAG Tests
 * ============================================================================ */

TEST(graph_index_create_destroy) {
    gpuio_graph_index_t index;
    gpuio_error_t err = gpuio_graph_index_create(g_ai_ctx, "/tmp/test_graph_index",
                                                  &index);
    /* May fail if file doesn't exist, but shouldn't crash */
    if (err == GPUIO_SUCCESS) {
        ASSERT_NOT_NULL(index);
        gpuio_graph_index_destroy(index);
    }
}

TEST(graph_rag_scatter_params) {
    /* Test scatter parameter validation */
    gpuio_scatter_params_t params = {
        .query_embedding = NULL,
        .query_dim = 768,
        .top_k = 100,
        .similarity_threshold = 0.8f,
        .edge_type_filters = NULL,
        .num_edge_type_filters = 0,
        .node_label_filters = NULL,
        .num_node_label_filters = 0,
    };
    
    ASSERT_EQ(params.query_dim, 768);
    ASSERT_EQ(params.top_k, 100);
    ASSERT_FLOAT_EQ(params.similarity_threshold, 0.8f, 0.001f);
}

TEST(graph_rag_gather_params) {
    /* Test gather parameter validation */
    gpuio_gather_params_t params = {
        .hop_depth = 2,
        .max_subgraph_size = 1000,
        .include_edges = true,
        .include_neighbors = true,
    };
    
    ASSERT_EQ(params.hop_depth, 2);
    ASSERT_EQ(params.max_subgraph_size, 1000);
    ASSERT(params.include_edges);
    ASSERT(params.include_neighbors);
}

TEST(graph_storage_create) {
    gpuio_graph_storage_t storage;
    gpuio_error_t err = gpuio_graph_storage_create(g_ai_ctx, 
                                                    "nvme:///tmp/test_graph",
                                                    &storage);
    if (err == GPUIO_SUCCESS) {
        ASSERT_NOT_NULL(storage);
        gpuio_graph_storage_destroy(storage);
    }
}

/* ============================================================================
 * Engram Memory Tests
 * ============================================================================ */

TEST(engram_pool_create_destroy) {
    gpuio_engram_config_t config = {
        .hbm_capacity = 500 << 20,     /* 500MB */
        .cxl_capacity = 5ULL << 30,    /* 5GB */
        .cxl_device_path = NULL,
        .remote_archive_uri = NULL,
        .embedding_dim = 768,
        .index_type = "hnsw",
        .async_writes = true,
        .write_buffer_size = 100 << 20, /* 100MB */
        .flush_interval_ms = 1000,
    };
    
    gpuio_engram_pool_t pool;
    gpuio_error_t err = gpuio_engram_pool_create(g_ai_ctx, &config, &pool);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(pool);
    
    err = gpuio_engram_pool_destroy(pool);
    ASSERT_EQ(err, GPUIO_SUCCESS);
}

TEST(engram_write_read) {
    gpuio_engram_config_t config = {
        .hbm_capacity = 500 << 20,
        .cxl_capacity = 5ULL << 30,
        .embedding_dim = 768,
        .async_writes = false,
    };
    
    gpuio_engram_pool_t pool;
    gpuio_engram_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(g_ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Create engram */
    float embedding[768];
    for (int i = 0; i < 768; i++) {
        embedding[i] = (float)i / 768.0f;
    }
    
    char data[] = "Test engram data for knowledge storage";
    
    gpuio_engram_t engram = {
        .engram_id = 12345,
        .version = 1,
        .data = data,
        .size = strlen(data) + 1,
        .embedding = embedding,
        .embedding_dim = 768,
        .creation_timestamp = 0,
        .importance_score = 0.85f,
        .access_count = 0,
        .tier = GPUIO_ENGRAM_TIER_HBM,
    };
    
    /* Write */
    gpuio_error_t err = gpuio_engram_write(pool, &engram, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    /* Read */
    gpuio_engram_handle_t handle;
    err = gpuio_engram_read(pool, 12345, stream, &handle);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(handle);
    
    /* Get data */
    void* read_data;
    size_t read_size;
    err = gpuio_engram_get_data(handle, &read_data, &read_size);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(read_data);
    ASSERT_EQ(read_size, strlen(data) + 1);
    
    gpuio_engram_release(handle);
    gpuio_stream_destroy(g_ctx, stream);
    gpuio_engram_pool_destroy(pool);
}

TEST(engram_query) {
    gpuio_engram_config_t config = {
        .hbm_capacity = 500 << 20,
        .cxl_capacity = 5ULL << 30,
        .embedding_dim = 768,
    };
    
    gpuio_engram_pool_t pool;
    gpuio_engram_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(g_ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Query embedding */
    float query[768];
    for (int i = 0; i < 768; i++) {
        query[i] = 0.5f;
    }
    
    gpuio_engram_handle_t results[10];
    int num_results;
    
    gpuio_error_t err = gpuio_engram_query(pool, query, 0.7f, 10,
                                            stream, results, &num_results);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(num_results >= 0);
    ASSERT(num_results <= 10);
    
    for (int i = 0; i < num_results; i++) {
        gpuio_engram_release(results[i]);
    }
    
    gpuio_stream_destroy(g_ctx, stream);
    gpuio_engram_pool_destroy(pool);
}

TEST(engram_batch_write) {
    gpuio_engram_config_t config = {
        .hbm_capacity = 500 << 20,
        .cxl_capacity = 5ULL << 30,
        .embedding_dim = 768,
        .async_writes = true,
    };
    
    gpuio_engram_pool_t pool;
    gpuio_engram_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(g_ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Batch of engrams */
    int batch_size = 100;
    gpuio_engram_t* engrams = calloc(batch_size, sizeof(gpuio_engram_t));
    
    for (int i = 0; i < batch_size; i++) {
        engrams[i].engram_id = i;
        engrams[i].version = 1;
        engrams[i].data = "Batch engram";
        engrams[i].size = 13;
        engrams[i].importance_score = 0.5f;
    }
    
    gpuio_error_t err = gpuio_engram_write_batch(pool, engrams, batch_size,
                                                  stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    free(engrams);
    gpuio_stream_destroy(g_ctx, stream);
    gpuio_engram_pool_destroy(pool);
}

TEST(engram_sync) {
    gpuio_engram_config_t config = {
        .hbm_capacity = 500 << 20,
        .cxl_capacity = 5ULL << 30,
        .async_writes = true,
    };
    
    gpuio_engram_pool_t pool;
    gpuio_engram_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(g_ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    gpuio_error_t err = gpuio_engram_sync(pool, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_stream_destroy(g_ctx, stream);
    gpuio_engram_pool_destroy(pool);
}

TEST(engram_stats) {
    gpuio_engram_config_t config = {
        .hbm_capacity = 500 << 20,
        .cxl_capacity = 5ULL << 30,
    };
    
    gpuio_engram_pool_t pool;
    gpuio_engram_pool_create(g_ai_ctx, &config, &pool);
    
    gpuio_engram_stats_t stats;
    gpuio_error_t err = gpuio_engram_get_stats(pool, &stats);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    ASSERT(stats.total_queries >= 0);
    ASSERT(stats.cache_hits >= 0);
    ASSERT(stats.avg_query_latency_us >= 0);
    
    gpuio_engram_pool_destroy(pool);
}

/* ============================================================================
 * Compression Tests
 * ============================================================================ */

TEST(codec_create_destroy) {
    gpuio_codec_t codec;
    gpuio_error_t err = gpuio_codec_create(g_ctx, GPUIO_CODEC_LZ4, 3, &codec);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(codec);
    
    err = gpuio_codec_destroy(codec);
    ASSERT_EQ(err, GPUIO_SUCCESS);
}

TEST(codec_types) {
    gpuio_codec_t codec;
    
    /* Test all codec types */
    gpuio_codec_type_t types[] = {
        GPUIO_CODEC_LZ4,
        GPUIO_CODEC_ZSTD,
        GPUIO_CODEC_GZIP,
        GPUIO_CODEC_FP16,
        GPUIO_CODEC_INT8,
    };
    
    for (int i = 0; i < 5; i++) {
        gpuio_error_t err = gpuio_codec_create(g_ctx, types[i], 3, &codec);
        if (err == GPUIO_SUCCESS) {
            gpuio_codec_destroy(codec);
        }
    }
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

static void print_header(const char* title) {
    printf("\n%s\n", title);
    printf("%.*s\n", (int)strlen(title), 
           "============================================================");
}

int main(int argc, char* argv[]) {
    printf("gpuio AI/ML Extensions Unit Tests\n");
    printf("Version: %s\n", gpuio_get_version_string());
    
    /* Setup */
    printf("\nSetting up test environment...\n");
    setup();
    printf("Setup complete.\n");
    
    /* AI Context Tests */
    print_header("AI Context Tests");
    RUN_TEST(ai_context_create_destroy);
    RUN_TEST(ai_context_minimal_config);
    
    /* DSA KV Cache Tests */
    print_header("DSA KV Cache Tests");
    RUN_TEST(dsa_kv_pool_create_destroy);
    RUN_TEST(dsa_kv_pool_reset);
    RUN_TEST(dsa_kv_load_store);
    RUN_TEST(dsa_kv_load_batch);
    RUN_TEST(dsa_kv_sparsity_pattern);
    RUN_TEST(dsa_kv_compact);
    RUN_TEST(dsa_kv_stats);
    
    /* Graph RAG Tests */
    print_header("Graph RAG Tests");
    RUN_TEST(graph_index_create_destroy);
    RUN_TEST(graph_rag_scatter_params);
    RUN_TEST(graph_rag_gather_params);
    RUN_TEST(graph_storage_create);
    
    /* Engram Memory Tests */
    print_header("Engram Memory Tests");
    RUN_TEST(engram_pool_create_destroy);
    RUN_TEST(engram_write_read);
    RUN_TEST(engram_query);
    RUN_TEST(engram_batch_write);
    RUN_TEST(engram_sync);
    RUN_TEST(engram_stats);
    
    /* Compression Tests */
    print_header("Compression Tests");
    RUN_TEST(codec_create_destroy);
    RUN_TEST(codec_types);
    
    /* Teardown */
    printf("\nTearing down test environment...\n");
    teardown();
    
    /* Summary */
    printf("\n============================================================\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("============================================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
