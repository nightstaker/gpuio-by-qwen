/**
 * @file memio_internal.h
 * @brief MemIO module internal definitions
 * @version 1.0.0
 */

#ifndef MEMIO_INTERNAL_H
#define MEMIO_INTERNAL_H

#include <gpuio/gpuio.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>

/* MemIO module exports */
#define MEMIO_API __attribute__((visibility("default")))

/* CXL memory pool */
typedef struct memio_cxl_pool {
    char* device_path;
    int fd;
    void* mapped_base;
    size_t capacity;
    size_t used;
    pthread_mutex_t lock;
    
    /* Allocation tracking */
    struct memio_cxl_block {
        uint64_t offset;
        size_t size;
        bool allocated;
        struct memio_cxl_block* next;
    }* blocks;
} memio_cxl_pool_t;

/* NUMA node info */
typedef struct memio_numa_node {
    int node_id;
    size_t total_memory;
    size_t free_memory;
    int num_gpus;
    int* gpu_ids;
} memio_numa_node_t;

/* Memory mapper for on-demand access */
typedef struct memio_mapper {
    void* virtual_base;
    size_t virtual_size;
    void* physical_base;
    int num_pages;
    int* page_mapping;
    pthread_rwlock_t lock;
} memio_mapper_t;

/* MemIO context extension */
typedef struct memio_context {
    gpuio_context_t parent;
    
    /* CXL pools */
    memio_cxl_pool_t** cxl_pools;
    int num_cxl_pools;
    pthread_mutex_t cxl_lock;
    
    /* NUMA info */
    memio_numa_node_t* numa_nodes;
    int num_numa_nodes;
    
    /* Memory mappers */
    memio_mapper_t** mappers;
    int num_mappers;
    pthread_mutex_t mapper_lock;
    
    /* Statistics */
    uint64_t bytes_transferred_cpu;
    uint64_t bytes_transferred_cxl;
    uint64_t page_faults;
} memio_context_t;

/* Internal functions */
int memio_cxl_pool_create(memio_context_t* ctx, const char* device_path,
                          size_t capacity, memio_cxl_pool_t** pool);
int memio_cxl_pool_destroy(memio_cxl_pool_t* pool);
int memio_cxl_alloc(memio_cxl_pool_t* pool, size_t size, uint64_t* offset);
int memio_cxl_free(memio_cxl_pool_t* pool, uint64_t offset);

int memio_numa_detect(memio_context_t* ctx);
int memio_numa_get_preferred_node(int gpu_id);

int memio_mapper_create(memio_context_t* ctx, size_t virtual_size,
                        memio_mapper_t** mapper);
int memio_mapper_destroy(memio_mapper_t* mapper);
int memio_mapper_handle_fault(memio_mapper_t* mapper, void* fault_addr,
                              void** mapped_page);

int memio_memcpy_h2d(memio_context_t* ctx, void* dst, const void* src,
                     size_t size, gpuio_stream_t stream);
int memio_memcpy_d2h(memio_context_t* ctx, void* dst, const void* src,
                     size_t size, gpuio_stream_t stream);
int memio_memcpy_d2d(memio_context_t* ctx, void* dst, const void* src,
                     size_t size, gpuio_stream_t stream);
int memio_memcpy_to_cxl(memio_context_t* ctx, uint64_t cxl_offset,
                        const void* src, size_t size);
int memio_memcpy_from_cxl(memio_context_t* ctx, void* dst, uint64_t cxl_offset,
                          size_t size);

#endif /* MEMIO_INTERNAL_H */
