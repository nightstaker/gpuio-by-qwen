/**
 * @file vendor_nvidia.c
 * @brief NVIDIA CUDA vendor support (stub)
 */

#include "core_internal.h"
#include <string.h>

int nvidia_device_init(gpuio_context_t ctx, int device_id) {
    (void)ctx; (void)device_id;
    return -1;
}

int nvidia_device_get_info(gpuio_context_t ctx, int device_id, 
                           core_device_info_t* info) {
    (void)ctx; (void)device_id;
    info->vendor = GPU_VENDOR_NVIDIA;
    strcpy(info->name, "NVIDIA Stub GPU");
    info->total_memory = 16ULL << 30;
    info->free_memory = 8ULL << 30;
    info->compute_capability_major = 8;
    info->compute_capability_minor = 0;
    info->supports_gds = true;
    info->supports_gdr = true;
    return 0;
}

int nvidia_device_set_current(gpuio_context_t ctx, int device_id) {
    (void)ctx; (void)device_id;
    return 0;
}

int nvidia_malloc_device(gpuio_context_t ctx, size_t size, void** ptr) {
    (void)ctx;
    *ptr = malloc(size);
    return *ptr ? 0 : -1;
}

int nvidia_malloc_pinned(gpuio_context_t ctx, size_t size, void** ptr) {
    (void)ctx;
    *ptr = malloc(size);
    return *ptr ? 0 : -1;
}

int nvidia_free(gpuio_context_t ctx, void* ptr) {
    (void)ctx;
    free(ptr);
    return 0;
}

int nvidia_memcpy(gpuio_context_t ctx, void* dst, const void* src, 
                  size_t size, gpuio_stream_t stream) {
    (void)ctx; (void)stream;
    memcpy(dst, src, size);
    return 0;
}

int nvidia_register_memory(gpuio_context_t ctx, void* ptr, size_t size,
                           gpuio_mem_access_t access,
                           core_memory_region_t* region) {
    (void)ctx; (void)ptr; (void)size; (void)access;
    region->gpu_addr = ptr;
    region->bus_addr = (uint64_t)(uintptr_t)ptr;
    return 0;
}

int nvidia_unregister_memory(gpuio_context_t ctx, core_memory_region_t* region) {
    (void)ctx; (void)region;
    return 0;
}

int nvidia_stream_create(gpuio_context_t ctx, core_stream_t* stream,
                         gpuio_stream_priority_t priority) {
    (void)ctx; (void)stream; (void)priority;
    return 0;
}

int nvidia_stream_destroy(gpuio_context_t ctx, core_stream_t* stream) {
    (void)ctx; (void)stream;
    return 0;
}

int nvidia_stream_synchronize(gpuio_context_t ctx, core_stream_t* stream) {
    (void)ctx; (void)stream;
    return 0;
}

int nvidia_stream_query(gpuio_context_t ctx, core_stream_t* stream, bool* idle) {
    (void)ctx; (void)stream;
    *idle = true;
    return 0;
}

int nvidia_event_create(gpuio_context_t ctx, gpuio_event_t* event) {
    (void)ctx; (void)event;
    return 0;
}

int nvidia_event_destroy(gpuio_context_t ctx, gpuio_event_t event) {
    (void)ctx; (void)event;
    return 0;
}

int nvidia_event_record(gpuio_context_t ctx, gpuio_event_t event,
                        core_stream_t* stream) {
    (void)ctx; (void)event; (void)stream;
    return 0;
}

int nvidia_event_synchronize(gpuio_context_t ctx, gpuio_event_t event) {
    (void)ctx; (void)event;
    return 0;
}

int nvidia_event_elapsed_time(gpuio_context_t ctx, gpuio_event_t start,
                               gpuio_event_t end, float* ms) {
    (void)ctx; (void)start; (void)end;
    *ms = 0.0f;
    return 0;
}

core_vendor_ops_t nvidia_ops = {
    .device_init = nvidia_device_init,
    .device_get_info = nvidia_device_get_info,
    .device_set_current = nvidia_device_set_current,
    .malloc_device = nvidia_malloc_device,
    .malloc_pinned = nvidia_malloc_pinned,
    .free = nvidia_free,
    .memcpy = nvidia_memcpy,
    .register_memory = nvidia_register_memory,
    .unregister_memory = nvidia_unregister_memory,
    .stream_create = nvidia_stream_create,
    .stream_destroy = nvidia_stream_destroy,
    .stream_synchronize = nvidia_stream_synchronize,
    .stream_query = nvidia_stream_query,
    .event_create = nvidia_event_create,
    .event_destroy = nvidia_event_destroy,
    .event_record = nvidia_event_record,
    .event_synchronize = nvidia_event_synchronize,
    .event_elapsed_time = nvidia_event_elapsed_time,
};
