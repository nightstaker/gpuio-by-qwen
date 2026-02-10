/**
 * @file bench_throughput.c
 * @brief Performance benchmarks for gpuio
 * @version 1.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <gpuio/gpuio.h>
#include <gpuio/gpuio_ai.h>

/* Benchmark configuration */
#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 100

/* Timing */
static double get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static double get_time_ms(void) {
    return get_time_us() / 1000.0;
}

/* Suppress unused function warning - available for future benchmarks */
typedef double (*get_time_ms_func_t)(void);
static const get_time_ms_func_t __attribute__((unused)) _unused_get_time_ms = get_time_ms;

/* Statistics */
typedef struct {
    double min;
    double max;
    double avg;
    double p50;
    double p99;
    double stddev;
} bench_stats_t;

static void calculate_stats(double* times, int n, bench_stats_t* stats) {
    /* Calculate min, max, avg */
    stats->min = times[0];
    stats->max = times[0];
    double sum = 0;
    
    for (int i = 0; i < n; i++) {
        if (times[i] < stats->min) stats->min = times[i];
        if (times[i] > stats->max) stats->max = times[i];
        sum += times[i];
    }
    
    stats->avg = sum / n;
    
    /* Simple P50 and P99 (not sorted, approximate) */
    stats->p50 = stats->avg;  /* Approximation */
    stats->p99 = stats->max * 0.99;
    
    /* Standard deviation */
    double variance = 0;
    for (int i = 0; i < n; i++) {
        variance += (times[i] - stats->avg) * (times[i] - stats->avg);
    }
    stats->stddev = sqrt(variance / n);
}

static void print_stats(const char* name, bench_stats_t* stats, const char* unit) {
    printf("  %s:\n", name);
    printf("    Min:  %.3f %s\n", stats->min, unit);
    printf("    Max:  %.3f %s\n", stats->max, unit);
    printf("    Avg:  %.3f %s\n", stats->avg, unit);
    printf("    P50:  %.3f %s\n", stats->p50, unit);
    printf("    P99:  %.3f %s\n", stats->p99, unit);
    printf("    Std:  %.3f %s\n", stats->stddev, unit);
}

/* ============================================================================
 * MemIO Benchmarks
 * ============================================================================ */

static void bench_memio_memcpy(void) {
    printf("\nBenchmark: MemIO - Memory Copy\n");
    printf("-------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    /* Test different sizes (in bytes) */
    size_t sizes[] = {4096, 65536, 1048576, 16777216, 268435456};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        size_t size = sizes[s];
        
        void* src;
        void* dst;
        gpuio_malloc(ctx, size, &src);
        gpuio_malloc(ctx, size, &dst);
        
        /* Warmup */
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gpuio_memcpy(ctx, dst, src, size, NULL);
        }
        
        /* Benchmark */
        double times[BENCHMARK_ITERATIONS];
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            double start = get_time_us();
            gpuio_memcpy(ctx, dst, src, size, NULL);
            double end = get_time_us();
            times[i] = end - start;
        }
        
        bench_stats_t stats;
        calculate_stats(times, BENCHMARK_ITERATIONS, &stats);
        
        double bandwidth = (size / (1ULL << 20)) / (stats.avg / 1e6);
        
        printf("  Size: %6zu MB | Bandwidth: %7.2f MB/s | Latency: %.2f us\n",
               size >> 20, bandwidth, stats.avg);
        
        gpuio_free(ctx, src);
        gpuio_free(ctx, dst);
    }
    
    gpuio_finalize(ctx);
}

/* ============================================================================
 * DSA KV Cache Benchmarks
 * ============================================================================ */

static void bench_dsa_kv_access(void) {
    printf("\nBenchmark: DSA KV Cache - Access Patterns\n");
    printf("------------------------------------------\n");
    
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
        .default_compression = GPUIO_KV_COMPRESS_FP16,
    };
    
    gpuio_dsa_kv_pool_t kv_pool;
    gpuio_dsa_kv_pool_create(ai_ctx, &kv_config, &kv_pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Pre-populate cache */
    printf("  Populating cache...\n");
    float kv_data[64 * 2 * 64]; /* head_dim * 2 (K+V) * heads */
    
    for (int pos = 0; pos < 1000; pos++) {
        for (int layer = 0; layer < 10; layer++) {
            gpuio_dsa_kv_store(kv_pool, pos, layer, 0, kv_data, sizeof(kv_data),
                               0.8f, stream);
        }
    }
    
    gpuio_stream_synchronize(ctx, stream);
    
    /* Benchmark single access */
    printf("\n  Single Entry Access:\n");
    
    double times[1000];
    for (int i = 0; i < 1000; i++) {
        double start = get_time_us();
        
        gpuio_dsa_kv_entry_t entry;
        gpuio_dsa_kv_load(kv_pool, i % 1000, i % 10, 0, stream, &entry);
        
        double end = get_time_us();
        times[i] = end - start;
        
        if (entry) {
            gpuio_dsa_kv_release(entry);
        }
    }
    
    bench_stats_t stats;
    calculate_stats(times, 1000, &stats);
    print_stats("Single Access", &stats, "us");
    
    /* Benchmark batch access */
    printf("\n  Batch Access (100 entries):\n");
    
    uint64_t positions[100];
    uint32_t layer_ids[100];
    uint32_t head_ids[100];
    gpuio_dsa_kv_entry_t entries[100];
    
    for (int i = 0; i < 100; i++) {
        positions[i] = i;
        layer_ids[i] = i % 10;
        head_ids[i] = 0;
    }
    
    double batch_times[100];
    for (int i = 0; i < 100; i++) {
        double start = get_time_us();
        
        gpuio_dsa_kv_load_batch(kv_pool, positions, layer_ids, head_ids,
                                 100, stream, entries);
        
        double end = get_time_us();
        batch_times[i] = end - start;
        
        for (int j = 0; j < 100; j++) {
            if (entries[j]) {
                gpuio_dsa_kv_release(entries[j]);
            }
        }
    }
    
    calculate_stats(batch_times, 100, &stats);
    print_stats("Batch Access", &stats, "us");
    printf("    Throughput: %.1f entries/us\n", 100.0 / stats.avg);
    
    /* Get cache stats */
    gpuio_dsa_kv_stats_t kv_stats;
    gpuio_dsa_kv_get_stats(kv_pool, &kv_stats);
    
    printf("\n  Cache Statistics:\n");
    printf("    Hit rate: %.2f%%\n", kv_stats.hit_rate * 100);
    printf("    HBM entries: %lu\n", kv_stats.entries_in_hbm);
    printf("    CXL entries: %lu\n", kv_stats.entries_in_cxl);
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_dsa_kv_pool_destroy(kv_pool);
    gpuio_ai_context_destroy(ai_ctx);
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Engram Memory Benchmarks
 * ============================================================================ */

static void bench_engram_query(void) {
    printf("\nBenchmark: Engram Memory - Query Performance\n");
    printf("---------------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_ai_config_t ai_config = {
        .enable_engram = true,
    };
    
    gpuio_ai_context_t ai_ctx;
    gpuio_ai_context_create(ctx, &ai_config, &ai_ctx);
    
    gpuio_engram_config_t engram_config = {
        .hbm_capacity = 1ULL << 30,
        .cxl_capacity = 5ULL << 30,
        .embedding_dim = 1024,
    };
    
    gpuio_engram_pool_t engram_pool;
    gpuio_engram_pool_create(ai_ctx, &engram_config, &engram_pool);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Populate engrams */
    printf("  Populating %d engrams...\n", 10000);
    
    float embedding[1024];
    char data[512];
    
    for (int i = 0; i < 10000; i++) {
        for (int j = 0; j < 1024; j++) {
            embedding[j] = (float)(i * j) / 10000.0f;
        }
        
        snprintf(data, sizeof(data), "Engram data %d", i);
        
        gpuio_engram_t engram = {
            .engram_id = i,
            .data = data,
            .size = strlen(data) + 1,
            .embedding = embedding,
            .embedding_dim = 1024,
            .importance_score = 0.5f,
        };
        
        gpuio_engram_write(engram_pool, &engram, stream);
    }
    
    gpuio_engram_sync(engram_pool, stream);
    
    /* Benchmark queries */
    printf("\n  Query Performance (top-10):\n");
    
    float query[1024] = {0};
    query[0] = 0.5f;
    
    double times[1000];
    for (int i = 0; i < 1000; i++) {
        gpuio_engram_handle_t results[10];
        int num_results;
        
        double start = get_time_us();
        
        gpuio_engram_query(engram_pool, query, 0.7f, 10,
                           stream, results, &num_results);
        
        double end = get_time_us();
        times[i] = end - start;
        
        for (int j = 0; j < num_results; j++) {
            gpuio_engram_release(results[j]);
        }
    }
    
    bench_stats_t stats;
    calculate_stats(times, 1000, &stats);
    print_stats("Query Latency", &stats, "us");
    printf("    Queries/sec: %.1f\n", 1e6 / stats.avg);
    
    /* Get stats */
    gpuio_engram_stats_t engram_stats;
    gpuio_engram_get_stats(engram_pool, &engram_stats);
    
    printf("\n  Engram Statistics:\n");
    printf("    Total queries: %lu\n", engram_stats.total_queries);
    printf("    Cache hits: %lu\n", engram_stats.cache_hits);
    printf("    Avg query latency: %.2f us\n", engram_stats.avg_query_latency_us);
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_engram_pool_destroy(engram_pool);
    gpuio_ai_context_destroy(ai_ctx);
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    printf("============================================================\n");
    printf("gpuio Performance Benchmarks\n");
    printf("Version: %s\n", gpuio_get_version_string());
    printf("============================================================\n");
    
    printf("\nConfiguration:\n");
    printf("  Warmup iterations: %d\n", WARMUP_ITERATIONS);
    printf("  Benchmark iterations: %d\n", BENCHMARK_ITERATIONS);
    
    /* Run benchmarks */
    bench_memio_memcpy();
    bench_dsa_kv_access();
    bench_engram_query();
    
    printf("\n============================================================\n");
    printf("Benchmarks Complete\n");
    printf("============================================================\n");
    
    return 0;
}
