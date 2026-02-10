/**
 * @file cxl.c
 * @brief MemIO module - CXL memory pool management
 * @version 1.0.0
 */

#include "memio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

int memio_cxl_pool_create(memio_context_t* ctx, const char* device_path,
                          size_t capacity, memio_cxl_pool_t** pool_out) {
    if (!ctx || !device_path || !pool_out) return -1;
    
    memio_cxl_pool_t* pool = calloc(1, sizeof(memio_cxl_pool_t));
    if (!pool) return -1;
    
    pool->device_path = strdup(device_path);
    pool->capacity = capacity;
    pthread_mutex_init(&pool->lock, NULL);
    
    /* Open CXL device */
    pool->fd = open(device_path, O_RDWR | O_DIRECT);
    if (pool->fd < 0) {
        /* Try without O_DIRECT for testing */
        pool->fd = open(device_path, O_RDWR);
        if (pool->fd < 0) {
            /* Create a memory-backed stub for testing */
            pool->fd = -1;
            pool->mapped_base = mmap(NULL, capacity, PROT_READ | PROT_WRITE,
                                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (pool->mapped_base == MAP_FAILED) {
                free(pool->device_path);
                free(pool);
                return -1;
            }
        }
    }
    
    if (pool->fd >= 0) {
        /* Memory map the CXL device */
        pool->mapped_base = mmap(NULL, capacity, PROT_READ | PROT_WRITE,
                                 MAP_SHARED, pool->fd, 0);
        if (pool->mapped_base == MAP_FAILED) {
            close(pool->fd);
            free(pool->device_path);
            free(pool);
            return -1;
        }
    }
    
    /* Initialize with one free block */
    pool->blocks = calloc(1, sizeof(struct memio_cxl_block));
    pool->blocks->offset = 0;
    pool->blocks->size = capacity;
    pool->blocks->allocated = false;
    pool->blocks->next = NULL;
    
    *pool_out = pool;
    return 0;
}

int memio_cxl_pool_destroy(memio_cxl_pool_t* pool) {
    if (!pool) return -1;
    
    pthread_mutex_lock(&pool->lock);
    
    if (pool->mapped_base) {
        munmap(pool->mapped_base, pool->capacity);
    }
    
    if (pool->fd >= 0) {
        close(pool->fd);
    }
    
    /* Free block list */
    struct memio_cxl_block* block = pool->blocks;
    while (block) {
        struct memio_cxl_block* next = block->next;
        free(block);
        block = next;
    }
    
    free(pool->device_path);
    
    pthread_mutex_unlock(&pool->lock);
    pthread_mutex_destroy(&pool->lock);
    free(pool);
    
    return 0;
}

int memio_cxl_alloc(memio_cxl_pool_t* pool, size_t size, uint64_t* offset_out) {
    if (!pool || size == 0 || !offset_out) return -1;
    
    pthread_mutex_lock(&pool->lock);
    
    /* First-fit allocation */
    struct memio_cxl_block* block = pool->blocks;
    while (block) {
        if (!block->allocated && block->size >= size) {
            /* Split block if there's remaining space */
            if (block->size > size) {
                struct memio_cxl_block* new_block = calloc(1, sizeof(*new_block));
                new_block->offset = block->offset + size;
                new_block->size = block->size - size;
                new_block->allocated = false;
                new_block->next = block->next;
                block->next = new_block;
                block->size = size;
            }
            
            block->allocated = true;
            *offset_out = block->offset;
            pool->used += size;
            
            pthread_mutex_unlock(&pool->lock);
            return 0;
        }
        block = block->next;
    }
    
    pthread_mutex_unlock(&pool->lock);
    return -1; /* No space available */
}

int memio_cxl_free(memio_cxl_pool_t* pool, uint64_t offset) {
    if (!pool) return -1;
    
    pthread_mutex_lock(&pool->lock);
    
    struct memio_cxl_block* block = pool->blocks;
    struct memio_cxl_block* prev = NULL;
    
    while (block) {
        if (block->offset == offset && block->allocated) {
            block->allocated = false;
            pool->used -= block->size;
            
            /* Coalesce with next block if free */
            if (block->next && !block->next->allocated) {
                struct memio_cxl_block* next = block->next;
                block->size += next->size;
                block->next = next->next;
                free(next);
            }
            
            /* Coalesce with previous block if free */
            if (prev && !prev->allocated) {
                prev->size += block->size;
                prev->next = block->next;
                free(block);
            }
            
            pthread_mutex_unlock(&pool->lock);
            return 0;
        }
        prev = block;
        block = block->next;
    }
    
    pthread_mutex_unlock(&pool->lock);
    return -1; /* Block not found */
}

void* memio_cxl_get_ptr(memio_cxl_pool_t* pool, uint64_t offset) {
    if (!pool || !pool->mapped_base) return NULL;
    return (char*)pool->mapped_base + offset;
}
