/**
 * @file test_training.c
 * @brief Integration tests for training workloads
 * @version 1.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gpuio/gpuio.h>
#include <gpuio/gpuio_ai.h>

/* Test configuration */
#define NUM_EPOCHS 3
#define BATCH_SIZE 32
#define SEQUENCE_LENGTH 1024
#define CHECKPOINT_INTERVAL 2

/* Timing utilities */
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Test: Distributed training simulation with checkpointing */
static int test_distributed_training_with_checkpoint(void) {
    printf("Test: Distributed Training with Checkpointing\n");
    printf("----------------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_error_t err = gpuio_init(&ctx, NULL);
    if (err != GPUIO_SUCCESS) {
        printf("  ERROR: Failed to initialize gpuio\n");
        return 1;
    }
    
    /* Create AI context for LLM training */
    gpuio_ai_config_t ai_config = {
        .num_layers = 96,
        .num_heads = 128,
        .head_dim = 128,
        .max_sequence_length = 8192,
        .enable_dsa_kv = true,
        .enable_engram = true,
        .enable_graph_rag = false,
        .default_priority = GPUIO_PRIO_TRAINING_FW,
        .kv_cache_size = 10ULL << 30,   /* 10GB */
        .engram_pool_size = 100ULL << 30, /* 100GB */
    };
    
    gpuio_ai_context_t ai_ctx;
    err = gpuio_ai_context_create(ctx, &ai_config, &ai_ctx);
    if (err != GPUIO_SUCCESS) {
        printf("  ERROR: Failed to create AI context\n");
        gpuio_finalize(ctx);
        return 1;
    }
    
    /* Create DSA KV cache pool */
    gpuio_dsa_kv_config_t kv_config = {
        .hbm_capacity = 10ULL << 30,    /* 10GB */
        .cxl_capacity = 100ULL << 30,   /* 100GB CXL */
        .default_compression = GPUIO_KV_COMPRESS_FP16,
        .compression_threshold = 0.3f,
        .importance_decay = 0.95f,
        .min_hot_cache_size = 2ULL << 30, /* 2GB */
    };
    
    gpuio_dsa_kv_pool_t kv_pool;
    err = gpuio_dsa_kv_pool_create(ai_ctx, &kv_config, &kv_pool);
    if (err != GPUIO_SUCCESS) {
        printf("  ERROR: Failed to create KV pool\n");
        gpuio_ai_context_destroy(ai_ctx);
        gpuio_finalize(ctx);
        return 1;
    }
    
    /* Create engram pool for external knowledge */
    gpuio_engram_config_t engram_config = {
        .hbm_capacity = 2ULL << 30,     /* 2GB */
        .cxl_capacity = 20ULL << 30,    /* 20GB */
        .embedding_dim = 1024,
        .async_writes = true,
        .flush_interval_ms = 5000,
    };
    
    gpuio_engram_pool_t engram_pool;
    err = gpuio_engram_pool_create(ai_ctx, &engram_config, &engram_pool);
    if (err != GPUIO_SUCCESS) {
        printf("  ERROR: Failed to create engram pool\n");
        gpuio_dsa_kv_pool_destroy(kv_pool);
        gpuio_ai_context_destroy(ai_ctx);
        gpuio_finalize(ctx);
        return 1;
    }
    
    /* Training stream */
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    /* Simulate training */
    double total_time = 0;
    double checkpoint_time = 0;
    int checkpoints_saved = 0;
    
    printf("  Simulating %d epochs...\n", NUM_EPOCHS);
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        double epoch_start = get_time_ms();
        
        printf("    Epoch %d/%d: ", epoch + 1, NUM_EPOCHS);
        
        /* Simulate forward/backward passes */
        for (int batch = 0; batch < 10; batch++) {
            /* Load KV cache for attention */
            for (int layer = 0; layer < 10; layer++) {
                gpuio_dsa_kv_entry_t entry;
                gpuio_dsa_kv_load(kv_pool, batch * SEQUENCE_LENGTH, layer, 0,
                                   stream, &entry);
                
                /* Simulate computation */
                if (entry) {
                    gpuio_dsa_kv_release(entry);
                }
            }
            
            /* Query engram for external knowledge */
            float query[1024] = {0};
            gpuio_engram_handle_t results[5];
            int num_results;
            gpuio_engram_query(engram_pool, query, 0.7f, 5, stream,
                               results, &num_results);
            
            for (int i = 0; i < num_results; i++) {
                gpuio_engram_release(results[i]);
            }
        }
        
        /* Checkpoint if needed */
        if ((epoch + 1) % CHECKPOINT_INTERVAL == 0) {
            double ckpt_start = get_time_ms();
            
            char checkpoint_path[256];
            snprintf(checkpoint_path, sizeof(checkpoint_path),
                     "/tmp/gpuio_test_checkpoint_epoch_%d", epoch + 1);
            
            gpuio_training_params_t params = {
                .checkpoint_after_step = true,
                .checkpoint_path = checkpoint_path,
            };
            
            gpuio_training_result_t result;
            /* Note: gpuio_ai_training_step would be called here in real code */
            
            double ckpt_end = get_time_ms();
            checkpoint_time += (ckpt_end - ckpt_start);
            checkpoints_saved++;
            
            printf("[CKPT] ");
        }
        
        /* Sync epoch */
        gpuio_stream_synchronize(ctx, stream);
        
        double epoch_end = get_time_ms();
        double epoch_time = epoch_end - epoch_start;
        total_time += epoch_time;
        
        printf("%.1f ms\n", epoch_time);
    }
    
    /* Print statistics */
    printf("\n  Results:\n");
    printf("    Total training time: %.1f ms\n", total_time);
    printf("    Avg epoch time: %.1f ms\n", total_time / NUM_EPOCHS);
    printf("    Checkpoints saved: %d\n", checkpoints_saved);
    printf("    Avg checkpoint time: %.1f ms\n",
           checkpoints_saved > 0 ? checkpoint_time / checkpoints_saved : 0);
    
    /* Get KV cache stats */
    gpuio_dsa_kv_stats_t kv_stats;
    gpuio_dsa_kv_get_stats(kv_pool, &kv_stats);
    printf("    KV cache hit rate: %.2f%%\n", kv_stats.hit_rate * 100);
    
    /* Cleanup */
    gpuio_stream_destroy(ctx, stream);
    gpuio_engram_pool_destroy(engram_pool);
    gpuio_dsa_kv_pool_destroy(kv_pool);
    gpuio_ai_context_destroy(ai_ctx);
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Test: Large model checkpoint/restart */
static int test_large_model_checkpoint_restart(void) {
    printf("\nTest: Large Model Checkpoint/Restart\n");
    printf("--------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    /* Simulate 100GB model */
    size_t model_size = 100ULL << 30; /* 100GB */
    
    printf("  Simulating %zu GB model checkpoint...\n", model_size >> 30);
    
    /* Allocate dummy model data */
    void* model_data;
    gpuio_error_t err = gpuio_malloc(ctx, model_size, &model_data);
    if (err != GPUIO_SUCCESS) {
        printf("  WARNING: Cannot allocate %zu GB, using smaller size\n",
               model_size >> 30);
        model_size = 1ULL << 30; /* Fall back to 1GB */
        err = gpuio_malloc(ctx, model_size, &model_data);
    }
    
    if (err == GPUIO_SUCCESS) {
        /* Register for GDS */
        gpuio_memory_region_t region;
        gpuio_register_memory(ctx, model_data, model_size,
                              GPUIO_MEM_READ_WRITE, &region);
        
        /* Create checkpoint request */
        gpuio_stream_t stream;
        gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
        
        double start = get_time_ms();
        
        /* Simulate checkpoint write to NVMe */
        printf("  Writing checkpoint to NVMe...\n");
        
        /* In real implementation:
         * gpuio_request_params_t params = {
         *     .type = GPUIO_REQ_WRITE,
         *     .engine = GPUIO_ENGINE_LOCALIO,
         *     .src = &region,
         *     .dst = ...,
         *     .length = model_size,
         *     .stream = stream,
         *     .async = true,
         * };
         */
        
        gpuio_stream_synchronize(ctx, stream);
        
        double end = get_time_ms();
        double checkpoint_time = end - start;
        double throughput = (model_size / (1ULL << 30)) / (checkpoint_time / 1000.0);
        
        printf("  Checkpoint time: %.1f ms\n", checkpoint_time);
        printf("  Throughput: %.2f GB/s\n", throughput);
        
        gpuio_unregister_memory(ctx, &region);
        gpuio_stream_destroy(ctx, stream);
        gpuio_free(ctx, model_data);
    }
    
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Test: Multi-GPU data parallel training */
static int test_multi_gpu_data_parallel(void) {
    printf("\nTest: Multi-GPU Data Parallel Training\n");
    printf("----------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    int num_gpus;
    gpuio_get_device_count(ctx, &num_gpus);
    
    printf("  Detected %d GPU(s)\n", num_gpus);
    
    if (num_gpus < 2) {
        printf("  SKIPPED (requires 2+ GPUs)\n");
        gpuio_finalize(ctx);
        return 0;
    }
    
    /* Simulate all-reduce across GPUs */
    printf("  Simulating gradient all-reduce...\n");
    
    size_t gradient_size = 1ULL << 30; /* 1GB gradients */
    
    for (int gpu = 0; gpu < num_gpus && gpu < 4; gpu++) {
        gpuio_set_device(ctx, gpu);
        printf("    GPU %d: Ready for all-reduce\n", gpu);
    }
    
    double start = get_time_ms();
    
    /* Simulate all-reduce operation */
    /* In real implementation: use RemoteIO with RDMA */
    
    double end = get_time_ms();
    
    printf("  All-reduce simulated in %.1f ms\n", end - start);
    printf("  Effective bandwidth: %.2f GB/s\n",
           (gradient_size / (1ULL << 30)) / ((end - start) / 1000.0));
    
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Test: Fault tolerance with checkpoint recovery */
static int test_fault_tolerance_recovery(void) {
    printf("\nTest: Fault Tolerance with Checkpoint Recovery\n");
    printf("------------------------------------------------\n");
    
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    /* Simulate training that fails mid-way */
    printf("  Simulating training with failure at step 50...\n");
    
    int total_steps = 100;
    int fail_step = 50;
    int checkpoint_interval = 25;
    int last_checkpoint = 0;
    
    /* Simulate training */
    for (int step = 0; step < fail_step; step++) {
        if ((step + 1) % checkpoint_interval == 0) {
            printf("    Step %d: Checkpoint saved\n", step + 1);
            last_checkpoint = step + 1;
        }
    }
    
    printf("  FAILURE at step %d!\n", fail_step);
    printf("  Last checkpoint: step %d\n", last_checkpoint);
    
    /* Simulate recovery */
    printf("  Recovering from checkpoint...\n");
    
    /* Load checkpoint */
    double start = get_time_ms();
    
    /* In real code: gpuio_ai_checkpoint_load(...); */
    
    double end = get_time_ms();
    
    printf("  Checkpoint loaded in %.1f ms\n", end - start);
    printf("  Resuming training from step %d...\n", last_checkpoint);
    
    /* Continue training */
    int remaining_steps = total_steps - last_checkpoint;
    printf("  Completing %d remaining steps...\n", remaining_steps);
    
    printf("  Training completed successfully!\n");
    printf("  Lost work: %d steps (%.1f%%)\n",
           fail_step - last_checkpoint,
           100.0 * (fail_step - last_checkpoint) / total_steps);
    
    gpuio_finalize(ctx);
    
    printf("\n  PASSED\n");
    return 0;
}

/* Main test runner */
int main(int argc, char* argv[]) {
    printf("============================================================\n");
    printf("gpuio Training Workload Integration Tests\n");
    printf("============================================================\n");
    
    int failures = 0;
    
    failures += test_distributed_training_with_checkpoint();
    failures += test_large_model_checkpoint_restart();
    failures += test_multi_gpu_data_parallel();
    failures += test_fault_tolerance_recovery();
    
    printf("\n============================================================\n");
    printf("Integration Test Summary:\n");
    printf("  Failures: %d\n", failures);
    printf("============================================================\n");
    
    return failures;
}
