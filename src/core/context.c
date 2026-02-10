/**
 * @file context.c
 * @brief Core module - Context and device management
 * @version 1.0.0
 */

#include "core_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

static pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;

static int context_init_devices(gpuio_context_t ctx) {
    /* Try NVIDIA first */
    if (nvidia_ops.device_init && nvidia_ops.device_init(ctx, 0) == 0) {
        CORE_LOG(ctx, GPUIO_LOG_INFO, "Initialized NVIDIA GPU support");
        current_vendor_ops = &nvidia_ops;
        return core_device_detect_all(ctx);
    }
    
    /* Try AMD */
    if (amd_ops.device_init && amd_ops.device_init(ctx, 0) == 0) {
        CORE_LOG(ctx, GPUIO_LOG_INFO, "Initialized AMD GPU support");
        current_vendor_ops = &amd_ops;
        return core_device_detect_all(ctx);
    }
    
    CORE_LOG(ctx, GPUIO_LOG_WARN, "No GPUs detected, running in stub mode");
    ctx->num_devices = 0;
    ctx->devices = NULL;
    return 0;
}

gpuio_error_t gpuio_init(gpuio_context_t* ctx_ptr, const gpuio_config_t* config) {
    if (!ctx_ptr) return GPUIO_ERROR_INVALID_ARG;
    
    pthread_mutex_lock(&global_lock);
    
    gpuio_context_t ctx = calloc(1, sizeof(struct gpuio_context));
    if (!ctx) {
        pthread_mutex_unlock(&global_lock);
        return GPUIO_ERROR_NOMEM;
    }
    
    if (config) {
        memcpy(&ctx->config, config, sizeof(gpuio_config_t));
    } else {
        ctx->config = (gpuio_config_t)GPUIO_CONFIG_DEFAULT;
    }
    
    ctx->log_level = ctx->config.log_level;
    
    pthread_mutex_init(&ctx->regions_lock, NULL);
    pthread_mutex_init(&ctx->streams_lock, NULL);
    pthread_mutex_init(&ctx->requests_lock, NULL);
    pthread_mutex_init(&ctx->stats_lock, NULL);
    
    if (context_init_devices(ctx) != 0) {
        CORE_LOG(ctx, GPUIO_LOG_ERROR, "Failed to initialize devices");
        free(ctx);
        pthread_mutex_unlock(&global_lock);
        return GPUIO_ERROR_GENERAL;
    }
    
    ctx->initialized = 1;
    *ctx_ptr = ctx;
    
    CORE_LOG(ctx, GPUIO_LOG_INFO, "gpuio initialized (version %s)",
             gpuio_get_version_string());
    
    pthread_mutex_unlock(&global_lock);
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_finalize(gpuio_context_t ctx) {
    if (!ctx) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    CORE_LOG(ctx, GPUIO_LOG_INFO, "Finalizing gpuio context");
    
    pthread_mutex_lock(&ctx->streams_lock);
    for (int i = 0; i < ctx->num_streams; i++) {
        if (ctx->streams[i] && ctx->streams[i]->id >= 0) {
            if (current_vendor_ops && current_vendor_ops->stream_synchronize) {
                current_vendor_ops->stream_synchronize(ctx, ctx->streams[i]);
            }
            if (current_vendor_ops && current_vendor_ops->stream_destroy) {
                current_vendor_ops->stream_destroy(ctx, ctx->streams[i]);
            }
            pthread_mutex_destroy(&ctx->streams[i]->lock);
            free(ctx->streams[i]);
        }
    }
    free(ctx->streams);
    pthread_mutex_unlock(&ctx->streams_lock);
    
    pthread_mutex_lock(&ctx->regions_lock);
    core_memory_region_t* region = ctx->regions;
    while (region) {
        core_memory_region_t* next = region->next;
        if (region->registered && current_vendor_ops && 
            current_vendor_ops->unregister_memory) {
            current_vendor_ops->unregister_memory(ctx, region);
        }
        free(region);
        region = next;
    }
    pthread_mutex_unlock(&ctx->regions_lock);
    
    core_device_cleanup(ctx);
    
    pthread_mutex_destroy(&ctx->regions_lock);
    pthread_mutex_destroy(&ctx->streams_lock);
    pthread_mutex_destroy(&ctx->requests_lock);
    pthread_mutex_destroy(&ctx->stats_lock);
    
    ctx->initialized = 0;
    free(ctx);
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_get_config(gpuio_context_t ctx, gpuio_config_t* config) {
    if (!ctx || !config) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    memcpy(config, &ctx->config, sizeof(gpuio_config_t));
    return GPUIO_SUCCESS;
}

int core_device_detect_all(gpuio_context_t ctx) {
    if (!current_vendor_ops || !current_vendor_ops->device_get_info) return -1;
    
    ctx->devices = calloc(16, sizeof(core_device_info_t));
    if (!ctx->devices) return -1;
    
    ctx->num_devices = 0;
    for (int i = 0; i < 16; i++) {
        core_device_info_t info;
        memset(&info, 0, sizeof(info));
        
        if (current_vendor_ops->device_get_info(ctx, i, &info) == 0) {
            memcpy(&ctx->devices[ctx->num_devices], &info, sizeof(info));
            ctx->devices[ctx->num_devices].device_id = ctx->num_devices;
            ctx->num_devices++;
        } else {
            break;
        }
    }
    
    CORE_LOG(ctx, GPUIO_LOG_INFO, "Detected %d GPU(s)", ctx->num_devices);
    return 0;
}

void core_device_cleanup(gpuio_context_t ctx) {
    if (ctx->devices) {
        free(ctx->devices);
        ctx->devices = NULL;
    }
    ctx->num_devices = 0;
}

gpuio_error_t gpuio_get_device_count(gpuio_context_t ctx, int* count) {
    if (!ctx || !count) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    *count = ctx->num_devices;
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_get_device_info(gpuio_context_t ctx, int device_id, 
                                     gpuio_device_info_t* info) {
    if (!ctx || !info) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    if (device_id < 0 || device_id >= ctx->num_devices) return GPUIO_ERROR_INVALID_ARG;
    
    core_device_info_t* internal = &ctx->devices[device_id];
    
    info->device_id = internal->device_id;
    info->vendor = (internal->vendor == GPU_VENDOR_NVIDIA) ? GPUIO_GPU_VENDOR_NVIDIA :
                   (internal->vendor == GPU_VENDOR_AMD) ? GPUIO_GPU_VENDOR_AMD :
                   GPUIO_GPU_VENDOR_INTEL;
    strncpy(info->name, internal->name, sizeof(info->name) - 1);
    info->name[sizeof(info->name) - 1] = '\0';
    info->total_memory = internal->total_memory;
    info->free_memory = internal->free_memory;
    info->compute_capability_major = internal->compute_capability_major;
    info->compute_capability_minor = internal->compute_capability_minor;
    info->supports_gds = internal->supports_gds;
    info->supports_gdr = internal->supports_gdr;
    info->supports_cxl = internal->supports_cxl;
    info->numa_node = internal->numa_node;
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_set_device(gpuio_context_t ctx, int device_id) {
    if (!ctx) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    if (device_id < 0 || device_id >= ctx->num_devices) return GPUIO_ERROR_INVALID_ARG;
    
    if (current_vendor_ops && current_vendor_ops->device_set_current) {
        if (current_vendor_ops->device_set_current(ctx, device_id) != 0) {
            return GPUIO_ERROR_GENERAL;
        }
    }
    
    ctx->current_device = device_id;
    return GPUIO_SUCCESS;
}
