/**
 * @file memio.c
 * @brief MemIO module - Main operations implementation
 * @version 1.0.0
 */

#include "memio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static pthread_once_t memio_once = PTHREAD_ONCE_INIT;
static int memio_initialized = 0;

static void memio_init_globals(void) {
    memio_initialized = 1;
}

int memio_initialize(void) {
    pthread_once(&memio_once, memio_init_globals);
    return 0;
}

memio_context_t* memio_context_create(gpuio_context_t parent) {
    if (!parent) return NULL;
    
    memio_context_t* ctx = calloc(1, sizeof(memio_context_t));
    if (!ctx) return NULL;
    
    ctx->parent = parent;
    pthread_mutex_init(&ctx->cxl_lock, NULL);
    pthread_mutex_init(&ctx->mapper_lock, NULL);
    
    /* Detect NUMA topology */
    memio_numa_detect(ctx);
    
    return ctx;
}

void memio_context_destroy(memio_context_t* ctx) {
    if (!ctx) return;
    
    /* Destroy CXL pools */
    pthread_mutex_lock(&ctx->cxl_lock);
    for (int i = 0; i < ctx->num_cxl_pools; i++) {
        memio_cxl_pool_destroy(ctx->cxl_pools[i]);
    }
    free(ctx->cxl_pools);
    pthread_mutex_unlock(&ctx->cxl_lock);
    
    /* Destroy mappers */
    pthread_mutex_lock(&ctx->mapper_lock);
    for (int i = 0; i < ctx->num_mappers; i++) {
        memio_mapper_destroy(ctx->mappers[i]);
    }
    free(ctx->mappers);
    pthread_mutex_unlock(&ctx->mapper_lock);
    
    /* Cleanup NUMA info */
    if (ctx->numa_nodes) {
        for (int i = 0; i < ctx->num_numa_nodes; i++) {
            free(ctx->numa_nodes[i].gpu_ids);
        }
        free(ctx->numa_nodes);
    }
    
    pthread_mutex_destroy(&ctx->cxl_lock);
    pthread_mutex_destroy(&ctx->mapper_lock);
    free(ctx);
}

int memio_memcpy_h2d(memio_context_t* ctx, void* dst, const void* src,
                     size_t size, gpuio_stream_t stream) {
    if (!ctx || !dst || !src) return -1;
    
    /* Check if dst is CXL memory */
    pthread_mutex_lock(&ctx->cxl_lock);
    for (int i = 0; i < ctx->num_cxl_pools; i++) {
        /* Check if dst is in this CXL pool */
        /* In real implementation, track which addresses belong to CXL */
    }
    pthread_mutex_unlock(&ctx->cxl_lock);
    
    /* Use GPU memcpy through parent context */
    gpuio_error_t err = gpuio_memcpy(ctx->parent, dst, src, size, stream);
    if (err == GPUIO_SUCCESS) {
        ctx->bytes_transferred_cpu += size;
    }
    
    return (err == GPUIO_SUCCESS) ? 0 : -1;
}

int memio_memcpy_d2h(memio_context_t* ctx, void* dst, const void* src,
                     size_t size, gpuio_stream_t stream) {
    if (!ctx || !dst || !src) return -1;
    
    gpuio_error_t err = gpuio_memcpy(ctx->parent, dst, src, size, stream);
    if (err == GPUIO_SUCCESS) {
        ctx->bytes_transferred_cpu += size;
    }
    
    return (err == GPUIO_SUCCESS) ? 0 : -1;
}

int memio_memcpy_d2d(memio_context_t* ctx, void* dst, const void* src,
                     size_t size, gpuio_stream_t stream) {
    if (!ctx || !dst || !src) return -1;
    
    gpuio_error_t err = gpuio_memcpy(ctx->parent, dst, src, size, stream);
    return (err == GPUIO_SUCCESS) ? 0 : -1;
}

int memio_memcpy_to_cxl(memio_context_t* ctx, uint64_t cxl_offset,
                        const void* src, size_t size) {
    if (!ctx || !src) return -1;
    
    /* Assume first CXL pool for now */
    if (ctx->num_cxl_pools == 0) return -1;
    
    memio_cxl_pool_t* pool = ctx->cxl_pools[0];
    void* dst = memio_cxl_get_ptr(pool, cxl_offset);
    if (!dst) return -1;
    
    memcpy(dst, src, size);
    ctx->bytes_transferred_cxl += size;
    
    return 0;
}

int memio_memcpy_from_cxl(memio_context_t* ctx, void* dst, uint64_t cxl_offset,
                          size_t size) {
    if (!ctx || !dst) return -1;
    
    if (ctx->num_cxl_pools == 0) return -1;
    
    memio_cxl_pool_t* pool = ctx->cxl_pools[0];
    void* src = memio_cxl_get_ptr(pool, cxl_offset);
    if (!src) return -1;
    
    memcpy(dst, src, size);
    ctx->bytes_transferred_cxl += size;
    
    return 0;
}

int memio_cxl_pool_add(memio_context_t* ctx, const char* device_path,
                       size_t capacity) {
    if (!ctx || !device_path) return -1;
    
    memio_cxl_pool_t* pool;
    if (memio_cxl_pool_create(ctx, device_path, capacity, &pool) != 0) {
        return -1;
    }
    
    pthread_mutex_lock(&ctx->cxl_lock);
    
    memio_cxl_pool_t** new_pools = realloc(ctx->cxl_pools,
                                           (ctx->num_cxl_pools + 1) * sizeof(void*));
    if (!new_pools) {
        memio_cxl_pool_destroy(pool);
        pthread_mutex_unlock(&ctx->cxl_lock);
        return -1;
    }
    
    ctx->cxl_pools = new_pools;
    ctx->cxl_pools[ctx->num_cxl_pools] = pool;
    ctx->num_cxl_pools++;
    
    pthread_mutex_unlock(&ctx->cxl_lock);
    
    return 0;
}

int memio_mapper_add(memio_context_t* ctx, size_t virtual_size,
                     memio_mapper_t** mapper_out) {
    if (!ctx || virtual_size == 0) return -1;
    
    memio_mapper_t* mapper;
    if (memio_mapper_create(ctx, virtual_size, &mapper) != 0) {
        return -1;
    }
    
    pthread_mutex_lock(&ctx->mapper_lock);
    
    memio_mapper_t** new_mappers = realloc(ctx->mappers,
                                           (ctx->num_mappers + 1) * sizeof(void*));
    if (!new_mappers) {
        memio_mapper_destroy(mapper);
        pthread_mutex_unlock(&ctx->mapper_lock);
        return -1;
    }
    
    ctx->mappers = new_mappers;
    ctx->mappers[ctx->num_mappers] = mapper;
    ctx->num_mappers++;
    
    pthread_mutex_unlock(&ctx->mapper_lock);
    
    if (mapper_out) {
        *mapper_out = mapper;
    }
    
    return 0;
}

int memio_get_stats(memio_context_t* ctx, uint64_t* bytes_cpu,
                    uint64_t* bytes_cxl, uint64_t* page_faults) {
    if (!ctx) return -1;
    
    if (bytes_cpu) *bytes_cpu = ctx->bytes_transferred_cpu;
    if (bytes_cxl) *bytes_cxl = ctx->bytes_transferred_cxl;
    if (page_faults) *page_faults = ctx->page_faults;
    
    return 0;
}
