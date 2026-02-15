/**
 * @file memory.c
 * @brief Core module - Memory management
 * @version 1.0.0
 */

#include "core_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

gpuio_error_t gpuio_malloc(gpuio_context_t ctx, size_t size, void** ptr) {
    if (!ctx || !ptr) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    *ptr = malloc(size);
    if (!*ptr) return GPUIO_ERROR_NOMEM;
    
    CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Allocated %zu bytes at %p", size, *ptr);
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_malloc_pinned(gpuio_context_t ctx, size_t size, void** ptr) {
    if (!ctx || !ptr) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    if (current_vendor_ops && current_vendor_ops->malloc_pinned) {
        if (current_vendor_ops->malloc_pinned(ctx, size, ptr) == 0) {
            CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Allocated %zu bytes pinned memory at %p", 
                    size, *ptr);
            return GPUIO_SUCCESS;
        }
    }
    
    *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    
    if (*ptr == MAP_FAILED) {
        *ptr = NULL;
        return GPUIO_ERROR_NOMEM;
    }
    
    /* Track this mmap'd memory in the region list */
    core_memory_region_t* internal = calloc(1, sizeof(core_memory_region_t));
    if (!internal) {
        munmap(*ptr, size);
        *ptr = NULL;
        return GPUIO_ERROR_NOMEM;
    }
    
    internal->base_addr = *ptr;
    internal->length = size;
    internal->gpu_id = ctx->current_device;
    internal->registered = false;
    internal->is_pinned = true;
    internal->is_mmap = true;
    internal->is_zero_copy = true;
    
    pthread_mutex_lock(&ctx->regions_lock);
    internal->next = ctx->regions;
    ctx->regions = internal;
    pthread_mutex_unlock(&ctx->regions_lock);
    
    madvise(*ptr, size, MADV_SEQUENTIAL);
    
    CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Allocated %zu bytes host memory at %p (tracked)", size, *ptr);
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_malloc_device(gpuio_context_t ctx, size_t size, void** ptr) {
    if (!ctx || !ptr) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    if (current_vendor_ops && current_vendor_ops->malloc_device) {
        if (current_vendor_ops->malloc_device(ctx, size, ptr) == 0) {
            CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Allocated %zu bytes device memory at %p",
                    size, *ptr);
            return GPUIO_SUCCESS;
        }
    }
    
    return gpuio_malloc_pinned(ctx, size, ptr);
}

gpuio_error_t gpuio_malloc_unified(gpuio_context_t ctx, size_t size, void** ptr) {
    if (!ctx || !ptr) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    if (current_vendor_ops && current_vendor_ops->malloc_device) {
        return gpuio_malloc_device(ctx, size, ptr);
    }
    
    return GPUIO_ERROR_UNSUPPORTED;
}

gpuio_error_t gpuio_free(gpuio_context_t ctx, void* ptr) {
    if (!ctx) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    if (!ptr) return GPUIO_SUCCESS;
    
    /* Check if this memory is registered - if so, it's still in use */
    pthread_mutex_lock(&ctx->regions_lock);
    core_memory_region_t* region = ctx->regions;
    core_memory_region_t* found_region = NULL;
    while (region) {
        if (region->base_addr == ptr) {
            found_region = region;
            break;
        }
        region = region->next;
    }
    
    if (found_region && found_region->is_mmap) {
        /* This was mmap'd memory - use munmap with stored size */
        pthread_mutex_unlock(&ctx->regions_lock);
        CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Freeing mmap-allocated memory at %p (size=%zu)", ptr, found_region->length);
        if (munmap(ptr, found_region->length) != 0) {
            CORE_LOG(ctx, GPUIO_LOG_WARN, "munmap failed at %p", ptr);
            return GPUIO_ERROR_GENERAL;
        }
        return GPUIO_SUCCESS;
    }
    
    if (found_region && found_region->is_pinned) {
        /* Pinned memory registered via register_memory - use munmap */
        pthread_mutex_unlock(&ctx->regions_lock);
        CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Freeing pinned memory at %p (size=%zu)", ptr, found_region->length);
        if (munmap(ptr, found_region->length) != 0) {
            CORE_LOG(ctx, GPUIO_LOG_WARN, "munmap failed at %p", ptr);
            return GPUIO_ERROR_GENERAL;
        }
        return GPUIO_SUCCESS;
    }
    
    pthread_mutex_unlock(&ctx->regions_lock);
    
    /* Check vendor ops first for non-registered memory */
    if (current_vendor_ops && current_vendor_ops->free) {
        if (current_vendor_ops->free(ctx, ptr) == 0) {
            CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Freed memory at %p via vendor ops", ptr);
            return GPUIO_SUCCESS;
        }
    }
    
    CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Freed memory at %p", ptr);
    free(ptr);
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_register_memory(gpuio_context_t ctx, void* ptr, size_t size,
                                     gpuio_mem_access_t access,
                                     gpuio_memory_region_t* region) {
    if (!ctx || !region) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    if (!ptr || size == 0) return GPUIO_ERROR_INVALID_ARG;
    
    core_memory_region_t* internal = calloc(1, sizeof(core_memory_region_t));
    if (!internal) return GPUIO_ERROR_NOMEM;
    
    internal->base_addr = ptr;
    internal->length = size;
    internal->access = access;
    internal->gpu_id = ctx->current_device;
    internal->registered = true;
    
    if (current_vendor_ops && current_vendor_ops->register_memory) {
        if (current_vendor_ops->register_memory(ctx, ptr, size, access, internal) != 0) {
            free(internal);
            return GPUIO_ERROR_GENERAL;
        }
    } else {
        internal->gpu_addr = ptr;
        internal->bus_addr = (uint64_t)(uintptr_t)ptr;
    }
    
    pthread_mutex_lock(&ctx->regions_lock);
    internal->next = ctx->regions;
    ctx->regions = internal;
    pthread_mutex_unlock(&ctx->regions_lock);
    
    region->base_addr = internal->base_addr;
    region->gpu_addr = internal->gpu_addr;
    region->bus_addr = internal->bus_addr;
    region->length = internal->length;
    region->access = internal->access;
    region->gpu_id = internal->gpu_id;
    region->registered = true;
    region->handle = internal;
    
    CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Registered memory region %p, size %zu", ptr, size);
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_unregister_memory(gpuio_context_t ctx, 
                                        gpuio_memory_region_t* region) {
    if (!ctx || !region) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    core_memory_region_t* internal = (core_memory_region_t*)region->handle;
    if (!internal) return GPUIO_ERROR_INVALID_ARG;
    
    pthread_mutex_lock(&ctx->regions_lock);
    core_memory_region_t** current = &ctx->regions;
    while (*current) {
        if (*current == internal) {
            *current = internal->next;
            break;
        }
        current = &(*current)->next;
    }
    pthread_mutex_unlock(&ctx->regions_lock);
    
    if (current_vendor_ops && current_vendor_ops->unregister_memory) {
        current_vendor_ops->unregister_memory(ctx, internal);
    }
    
    internal->registered = false;
    region->registered = false;
    free(internal);
    
    CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Unregistered memory region");
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_memcpy(gpuio_context_t ctx, void* dst, const void* src, 
                            size_t size, gpuio_stream_t stream) {
    if (!ctx) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    if (!dst || !src) return GPUIO_ERROR_INVALID_ARG;
    
    if (current_vendor_ops && current_vendor_ops->memcpy_fn) {
        if (current_vendor_ops->memcpy_fn(ctx, dst, src, size, stream) == 0) {
            pthread_mutex_lock(&ctx->stats_lock);
            ctx->stats.bytes_written += size;
            pthread_mutex_unlock(&ctx->stats_lock);
            return GPUIO_SUCCESS;
        }
    }
    
    memcpy(dst, src, size);
    
    pthread_mutex_lock(&ctx->stats_lock);
    ctx->stats.bytes_written += size;
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_memcpy_async(gpuio_context_t ctx, void* dst, const void* src,
                                  size_t size, gpuio_stream_t stream) {
    return gpuio_memcpy(ctx, dst, src, size, stream);
}
