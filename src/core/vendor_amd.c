/**
 * @file vendor_amd.c
 * @brief AMD ROCm vendor support (stub)
 */

#include "core_internal.h"
#include <string.h>

int amd_device_init(gpuio_context_t ctx, int device_id) {
    (void)ctx; (void)device_id;
    return -1;
}

int amd_device_get_info(gpuio_context_t ctx, int device_id, 
                        core_device_info_t* info) {
    (void)ctx; (void)device_id;
    info->vendor = GPU_VENDOR_AMD;
    strcpy(info->name, "AMD Stub GPU");
    info->total_memory = 16ULL << 30;
    info->free_memory = 8ULL << 30;
    info->compute_capability_major = 9;
    info->compute_capability_minor = 0;
    info->supports_gds = true;
    info->supports_gdr = true;
    return 0;
}

int amd_device_set_current(gpuio_context_t ctx, int device_id) {
    (void)ctx; (void)device_id;
    return 0;
}

int amd_malloc_device(gpuio_context_t ctx, size_t size, void** ptr) {
    (void)ctx;
    *ptr = malloc(size);
    return *ptr ? 0 : -1;
}

int amd_malloc_pinned(gpuio_context_t ctx, size_t size, void** ptr) {
    (void)ctx;
    *ptr = malloc(size);
    return *ptr ? 0 : -1;
}

int amd_free(gpuio_context_t ctx, void* ptr) {
    (void)ctx;
    free(ptr);
    return 0;
}

int amd_memcpy(gpuio_context_t ctx, void* dst, const void* src, 
               size_t size, gpuio_stream_t stream) {
    (void)ctx; (void)stream;
    memcpy(dst, src, size);
    return 0;
}

int amd_register_memory(gpuio_context_t ctx, void* ptr, size_t size,
                        gpuio_mem_access_t access,
                        core_memory_region_t* region) {
    (void)ctx; (void)ptr; (void)size; (void)access;
    region->gpu_addr = ptr;
    region->bus_addr = (uint64_t)(uintptr_t)ptr;
    return 0;
}

int amd_unregister_memory(gpuio_context_t ctx, core_memory_region_t* region) {
    (void)ctx; (void)region;
    return 0;
}

int amd_stream_create(gpuio_context_t ctx, core_stream_t* stream,
                      gpuio_stream_priority_t priority) {
    (void)ctx; (void)stream; (void)priority;
    return 0;
}

int amd_stream_destroy(gpuio_context_t ctx, core_stream_t* stream) {
    (void)ctx; (void)stream;
    return 0;
}

int amd_stream_synchronize(gpuio_context_t ctx, core_stream_t* stream) {
    (void)ctx; (void)stream;
    return 0;
}

int amd_stream_query(gpuio_context_t ctx, core_stream_t* stream, bool* idle) {
    (void)ctx; (void)stream;
    *idle = true;
    return 0;
}

int amd_event_create(gpuio_context_t ctx, gpuio_event_t* event) {
    (void)ctx; (void)event;
    return 0;
}

int amd_event_destroy(gpuio_context_t ctx, gpuio_event_t event) {
    (void)ctx; (void)event;
    return 0;
}

int amd_event_record(gpuio_context_t ctx, gpuio_event_t event,
                     core_stream_t* stream) {
    (void)ctx; (void)event; (void)stream;
    return 0;
}

int amd_event_synchronize(gpuio_context_t ctx, gpuio_event_t event) {
    (void)ctx; (void)event;
    return 0;
}

int amd_event_elapsed_time(gpuio_context_t ctx, gpuio_event_t start,
                           gpuio_event_t end, float* ms) {
    (void)ctx; (void)start; (void)end;
    *ms = 0.0f;
    return 0;
}

core_vendor_ops_t amd_ops = {
    .device_init = amd_device_init,
    .device_get_info = amd_device_get_info,
    .device_set_current = amd_device_set_current,
    .malloc_device = amd_malloc_device,
    .malloc_pinned = amd_malloc_pinned,
    .free = amd_free,
    .memcpy = amd_memcpy,
    .register_memory = amd_register_memory,
    .unregister_memory = amd_unregister_memory,
    .stream_create = amd_stream_create,
    .stream_destroy = amd_stream_destroy,
    .stream_synchronize = amd_stream_synchronize,
    .stream_query = amd_stream_query,
    .event_create = amd_event_create,
    .event_destroy = amd_event_destroy,
    .event_record = amd_event_record,
    .event_synchronize = amd_event_synchronize,
    .event_elapsed_time = amd_event_elapsed_time,
};

core_vendor_ops_t* current_vendor_ops = NULL;
