/**
 * @file numa.c
 * @brief MemIO module - NUMA-aware memory management
 * @version 1.0.0
 */

#include "memio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __linux__
#include <numa.h>
#include <numaif.h>
#endif

int memio_numa_detect(memio_context_t* ctx) {
    if (!ctx) return -1;
    
#ifdef __linux__
    if (numa_available() < 0) {
        /* NUMA not available, create single node */
        ctx->numa_nodes = calloc(1, sizeof(memio_numa_node_t));
        ctx->numa_nodes[0].node_id = 0;
        ctx->numa_nodes[0].total_memory = 0; /* Unknown */
        ctx->numa_nodes[0].free_memory = 0;
        ctx->num_numa_nodes = 1;
        return 0;
    }
    
    int max_node = numa_max_node();
    ctx->num_numa_nodes = max_node + 1;
    ctx->numa_nodes = calloc(ctx->num_numa_nodes, sizeof(memio_numa_node_t));
    
    for (int i = 0; i <= max_node; i++) {
        ctx->numa_nodes[i].node_id = i;
        
        if (numa_node_size64(i, (long long*)&ctx->numa_nodes[i].free_memory) >= 0) {
            ctx->numa_nodes[i].total_memory = ctx->numa_nodes[i].free_memory;
        }
        
        /* Count GPUs on this NUMA node */
        /* This would need to query PCI topology in real implementation */
        ctx->numa_nodes[i].num_gpus = 0;
        ctx->numa_nodes[i].gpu_ids = NULL;
    }
    
    return 0;
#else
    /* Non-Linux: single NUMA node */
    ctx->numa_nodes = calloc(1, sizeof(memio_numa_node_t));
    ctx->numa_nodes[0].node_id = 0;
    ctx->num_numa_nodes = 1;
    return 0;
#endif
}

int memio_numa_get_preferred_node(int gpu_id) {
#ifdef __linux__
    /* In real implementation, query PCI topology
     * to find which NUMA node the GPU is connected to */
    (void)gpu_id;
    return 0;
#else
    (void)gpu_id;
    return 0;
#endif
}

int memio_numa_alloc_on_node(memio_context_t* ctx, size_t size, int node,
                             void** ptr_out) {
    if (!ctx || !ptr_out) return -1;
    
#ifdef __linux__
    if (numa_available() >= 0) {
        *ptr_out = numa_alloc_onnode(size, node);
        if (*ptr_out) return 0;
    }
#endif
    
    /* Fallback to regular allocation */
    *ptr_out = malloc(size);
    return *ptr_out ? 0 : -1;
}

void memio_numa_free(memio_context_t* ctx, void* ptr, size_t size) {
    if (!ptr) return;
    
#ifdef __linux__
    if (numa_available() >= 0) {
        numa_free(ptr, size);
        return;
    }
#endif
    
    free(ptr);
}

int memio_numa_migrate_pages(memio_context_t* ctx, void* ptr, size_t size,
                             int target_node) {
    if (!ctx || !ptr) return -1;
    
#ifdef __linux__
    if (numa_available() >= 0) {
        unsigned long count = (size + 4095) / 4096;
        int* nodes = calloc(count, sizeof(int));
        int* status = calloc(count, sizeof(int));
        
        for (unsigned long i = 0; i < count; i++) {
            nodes[i] = target_node;
        }
        
        long ret = mbind(ptr, size, MPOL_BIND, 
                         (unsigned long*)nodes, max_node + 1, 0);
        
        free(nodes);
        free(status);
        
        return (ret == 0) ? 0 : -1;
    }
#endif
    
    (void)target_node;
    return 0; /* Not supported, but not an error */
}
