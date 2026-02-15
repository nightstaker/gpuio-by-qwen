/**
 * @file core_internal.h
 * @brief Core module internal definitions
 * @version 1.0.0
 */

#ifndef CORE_INTERNAL_H
#define CORE_INTERNAL_H

#include <gpuio/gpuio.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

/* Module exports */
#define CORE_API __attribute__((visibility("default")))

/* GPU vendor types */
typedef enum {
    GPU_VENDOR_UNKNOWN = 0,
    GPU_VENDOR_NVIDIA,
    GPU_VENDOR_AMD,
    GPU_VENDOR_INTEL,
} gpu_vendor_t;

/* Internal device info */
typedef struct {
    int device_id;
    gpu_vendor_t vendor;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    bool supports_gds;
    bool supports_gdr;
    bool supports_cxl;
    int numa_node;
    void* vendor_handle;
} core_device_info_t;

/* Internal memory region */
typedef struct core_memory_region {
    void* base_addr;
    void* gpu_addr;
    uint64_t bus_addr;
    size_t length;
    gpuio_mem_type_t type;
    gpuio_mem_access_t access;
    int gpu_id;
    bool registered;
    bool is_pinned;
    bool is_mmap;
    bool is_zero_copy;  /* Added: flag for zero-copy memory */
    struct core_memory_region* next;
} core_memory_region_t;

/* Internal request */
typedef struct core_request {
    uint64_t id;
    gpuio_request_type_t type;
    gpuio_io_engine_t engine;
    core_memory_region_t* src;
    core_memory_region_t* dst;
    uint64_t src_offset;
    uint64_t dst_offset;
    size_t length;
    gpuio_stream_t stream;
    gpuio_request_status_t status;
    gpuio_error_t error_code;
    size_t bytes_completed;
    gpuio_callback_t callback;
    void* user_data;
    struct core_request* next;
    struct core_request* prev;
} core_request_t;

/* Internal stream */
typedef struct {
    int id;
    gpuio_stream_priority_t priority;
    void* vendor_stream;
    pthread_mutex_t lock;
    core_request_t* pending_requests;
} core_stream_t;

/* Context implementation */
struct gpuio_context {
    gpuio_config_t config;
    int initialized;
    
    /* Devices */
    core_device_info_t* devices;
    int num_devices;
    int current_device;
    
    /* Memory regions */
    core_memory_region_t* regions;
    pthread_mutex_t regions_lock;
    
    /* Streams */
    core_stream_t** streams;
    int num_streams;
    pthread_mutex_t streams_lock;
    
    /* Requests */
    uint64_t next_request_id;
    core_request_t* active_requests;
    pthread_mutex_t requests_lock;
    
    /* Statistics */
    gpuio_stats_t stats;
    pthread_mutex_t stats_lock;
    
    /* Thread pool */
    void* thread_pool;
    
    /* Logging */
    gpuio_log_level_t log_level;
    FILE* log_file;
};

/* Vendor operations table */
typedef struct {
    int (*device_init)(gpuio_context_t ctx, int device_id);
    int (*device_get_info)(gpuio_context_t ctx, int device_id, 
                           core_device_info_t* info);
    int (*device_set_current)(gpuio_context_t ctx, int device_id);
    int (*malloc_device)(gpuio_context_t ctx, size_t size, void** ptr);
    int (*malloc_pinned)(gpuio_context_t ctx, size_t size, void** ptr);
    int (*free)(gpuio_context_t ctx, void* ptr);
    int (*memcpy_fn)(gpuio_context_t ctx, void* dst, const void* src,
                  size_t size, gpuio_stream_t stream);
    int (*register_memory)(gpuio_context_t ctx, void* ptr, size_t size,
                           gpuio_mem_access_t access,
                           core_memory_region_t* region);
    int (*unregister_memory)(gpuio_context_t ctx, core_memory_region_t* region);
    int (*stream_create)(gpuio_context_t ctx, core_stream_t* stream,
                         gpuio_stream_priority_t priority);
    int (*stream_destroy)(gpuio_context_t ctx, core_stream_t* stream);
    int (*stream_synchronize)(gpuio_context_t ctx, core_stream_t* stream);
    int (*stream_query)(gpuio_context_t ctx, core_stream_t* stream, bool* idle);
    int (*event_create)(gpuio_context_t ctx, gpuio_event_t* event);
    int (*event_destroy)(gpuio_context_t ctx, gpuio_event_t event);
    int (*event_record)(gpuio_context_t ctx, gpuio_event_t event,
                        core_stream_t* stream);
    int (*event_synchronize)(gpuio_context_t ctx, gpuio_event_t event);
    int (*event_elapsed_time)(gpuio_context_t ctx, gpuio_event_t start,
                               gpuio_event_t end, float* ms);
} core_vendor_ops_t;

/* External vendor ops */
extern core_vendor_ops_t nvidia_ops;
extern core_vendor_ops_t amd_ops;
extern core_vendor_ops_t* current_vendor_ops;

/* Internal functions */
int core_device_detect_all(gpuio_context_t ctx);
void core_device_cleanup(gpuio_context_t ctx);
void core_stats_update(gpuio_context_t ctx, gpuio_request_type_t type,
                       size_t bytes, gpuio_error_t status);
void core_log_message(gpuio_context_t ctx, gpuio_log_level_t level,
                      const char* file, int line, const char* fmt, ...);

#define CORE_LOG(ctx, level, ...) \
    do { if (level <= (ctx)->log_level) core_log_message(ctx, level, __FILE__, __LINE__, __VA_ARGS__); } while(0)

#endif /* CORE_INTERNAL_H */
