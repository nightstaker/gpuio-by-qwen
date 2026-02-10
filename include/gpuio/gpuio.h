/**
 * @file gpuio.h
 * @brief GPU-Initiated IO Accelerator - Core API
 * @version 1.0.0
 * @date 2026-02-09
 * 
 * This header defines the core API for gpuio - a high-performance GPU-initiated
 * IO acceleration framework for AI/ML training and inference workloads.
 */

#ifndef GPUIO_H
#define GPUIO_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Version and Compatibility
 * ============================================================================ */

#define GPUIO_VERSION_MAJOR 1
#define GPUIO_VERSION_MINOR 0
#define GPUIO_VERSION_PATCH 0

/* ============================================================================
 * Opaque Handle Types
 * ============================================================================ */

typedef struct gpuio_context* gpuio_context_t;
typedef struct gpuio_stream* gpuio_stream_t;
typedef struct gpuio_event* gpuio_event_t;
typedef struct gpuio_request* gpuio_request_t;
typedef struct gpuio_handle* gpuio_handle_t;
typedef struct gpuio_proxy* gpuio_proxy_t;

/* ============================================================================
 * Error Handling
 * ============================================================================ */

typedef enum {
    GPUIO_SUCCESS = 0,
    GPUIO_ERROR_GENERAL = -1,
    GPUIO_ERROR_NOMEM = -2,
    GPUIO_ERROR_INVALID_ARG = -3,
    GPUIO_ERROR_NOT_FOUND = -4,
    GPUIO_ERROR_TIMEOUT = -5,
    GPUIO_ERROR_IO = -6,
    GPUIO_ERROR_NETWORK = -7,
    GPUIO_ERROR_UNSUPPORTED = -8,
    GPUIO_ERROR_PERMISSION = -9,
    GPUIO_ERROR_BUSY = -10,
    GPUIO_ERROR_CANCELED = -11,
    GPUIO_ERROR_DEVICE_LOST = -12,
    GPUIO_ERROR_ALREADY_INITIALIZED = -13,
    GPUIO_ERROR_NOT_INITIALIZED = -14,
} gpuio_error_t;

const char* gpuio_error_string(gpuio_error_t error);

/* ============================================================================
 * Logging
 * ============================================================================ */

typedef enum {
    GPUIO_LOG_NONE = 0,
    GPUIO_LOG_FATAL = 1,
    GPUIO_LOG_ERROR = 2,
    GPUIO_LOG_WARN = 3,
    GPUIO_LOG_INFO = 4,
    GPUIO_LOG_DEBUG = 5,
    GPUIO_LOG_TRACE = 6,
} gpuio_log_level_t;

void gpuio_set_log_level(gpuio_log_level_t level);
void gpuio_log(gpuio_log_level_t level, const char* format, ...);

/* ============================================================================
 * Context Management
 * ============================================================================ */

typedef enum {
    GPUIO_GPU_VENDOR_NVIDIA = 0,
    GPUIO_GPU_VENDOR_AMD = 1,
    GPUIO_GPU_VENDOR_INTEL = 2,
} gpuio_gpu_vendor_t;

typedef struct {
    uint32_t api_version;
    uint32_t flags;
    int max_gpus;
    size_t memory_pool_size;
    gpuio_log_level_t log_level;
    const char* config_file;
    /* NUMA awareness */
    int numa_node;
    bool numa_strict;
    /* Security */
    bool enable_security;
    const char* credentials_path;
} gpuio_config_t;

#define GPUIO_CONFIG_DEFAULT { \
    .api_version = GPUIO_VERSION_MAJOR, \
    .flags = 0, \
    .max_gpus = -1, /* Auto-detect */ \
    .memory_pool_size = 0, /* Auto */ \
    .log_level = GPUIO_LOG_INFO, \
    .config_file = NULL, \
    .numa_node = -1, \
    .numa_strict = false, \
    .enable_security = true, \
    .credentials_path = NULL \
}

gpuio_error_t gpuio_init(gpuio_context_t* ctx, const gpuio_config_t* config);
gpuio_error_t gpuio_finalize(gpuio_context_t ctx);
gpuio_error_t gpuio_get_config(gpuio_context_t ctx, gpuio_config_t* config);

/* ============================================================================
 * Device Management
 * ============================================================================ */

typedef struct {
    int device_id;
    gpuio_gpu_vendor_t vendor;
    char name[256];
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    /* Feature support */
    bool supports_gds;           /* GPUDirect Storage */
    bool supports_gdr;           /* GPUDirect RDMA */
    bool supports_cxl;           /* CXL memory */
    bool supports_p2p;           /* Peer-to-peer access */
    /* NUMA info */
    int numa_node;
    int numa_distance;
} gpuio_device_info_t;

gpuio_error_t gpuio_get_device_count(gpuio_context_t ctx, int* count);
gpuio_error_t gpuio_get_device_info(gpuio_context_t ctx, int device_id, 
                                     gpuio_device_info_t* info);
gpuio_error_t gpuio_set_device(gpuio_context_t ctx, int device_id);

/* ============================================================================
 * Stream Management
 * ============================================================================ */

typedef enum {
    GPUIO_STREAM_DEFAULT = 0,
    GPUIO_STREAM_HIGH_PRIORITY = 1,
    GPUIO_STREAM_LOW_PRIORITY = 2,
} gpuio_stream_priority_t;

gpuio_error_t gpuio_stream_create(gpuio_context_t ctx, gpuio_stream_t* stream,
                                   gpuio_stream_priority_t priority);
gpuio_error_t gpuio_stream_destroy(gpuio_context_t ctx, gpuio_stream_t stream);
gpuio_error_t gpuio_stream_synchronize(gpuio_context_t ctx, gpuio_stream_t stream);
gpuio_error_t gpuio_stream_query(gpuio_context_t ctx, gpuio_stream_t stream, 
                                  bool* idle);

/* ============================================================================
 * Event Management
 * ============================================================================ */

gpuio_error_t gpuio_event_create(gpuio_context_t ctx, gpuio_event_t* event);
gpuio_error_t gpuio_event_destroy(gpuio_context_t ctx, gpuio_event_t event);
gpuio_error_t gpuio_event_record(gpuio_context_t ctx, gpuio_event_t event, 
                                  gpuio_stream_t stream);
gpuio_error_t gpuio_event_synchronize(gpuio_context_t ctx, gpuio_event_t event);
gpuio_error_t gpuio_event_elapsed_time(gpuio_context_t ctx, gpuio_event_t start,
                                        gpuio_event_t end, float* ms);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

typedef enum {
    GPUIO_MEM_DEFAULT = 0,
    GPUIO_MEM_PINNED = 1,           /* Page-locked host memory */
    GPUIO_MEM_DEVICE = 2,           /* GPU device memory */
    GPUIO_MEM_UNIFIED = 3,          /* Unified memory */
    GPUIO_MEM_CXL = 4,              /* CXL-attached memory */
    GPUIO_MEM_REMOTE = 5,           /* Remote GPU memory */
} gpuio_mem_type_t;

typedef enum {
    GPUIO_MEM_READ = 1,
    GPUIO_MEM_WRITE = 2,
    GPUIO_MEM_READ_WRITE = 3,
    GPUIO_MEM_ATOMIC = 4,
} gpuio_mem_access_t;

typedef struct {
    void* base_addr;
    void* gpu_addr;
    uint64_t bus_addr;
    size_t length;
    gpuio_mem_type_t type;
    gpuio_mem_access_t access;
    int gpu_id;
    bool registered;
    void* handle;  /* Internal implementation handle */
} gpuio_memory_region_t;

/* Allocation */
gpuio_error_t gpuio_malloc(gpuio_context_t ctx, size_t size, void** ptr);
gpuio_error_t gpuio_malloc_pinned(gpuio_context_t ctx, size_t size, void** ptr);
gpuio_error_t gpuio_malloc_device(gpuio_context_t ctx, size_t size, void** ptr);
gpuio_error_t gpuio_malloc_unified(gpuio_context_t ctx, size_t size, void** ptr);
gpuio_error_t gpuio_free(gpuio_context_t ctx, void* ptr);

/* Registration for zero-copy */
gpuio_error_t gpuio_register_memory(gpuio_context_t ctx, void* ptr, size_t size,
                                     gpuio_mem_access_t access,
                                     gpuio_memory_region_t* region);
gpuio_error_t gpuio_unregister_memory(gpuio_context_t ctx, 
                                        gpuio_memory_region_t* region);

/* Copy operations */
gpuio_error_t gpuio_memcpy(gpuio_context_t ctx, void* dst, const void* src, 
                            size_t size, gpuio_stream_t stream);
gpuio_error_t gpuio_memcpy_async(gpuio_context_t ctx, void* dst, const void* src,
                                  size_t size, gpuio_stream_t stream);

/* ============================================================================
 * Request Types
 * ============================================================================ */

typedef enum {
    GPUIO_REQ_READ = 0,
    GPUIO_REQ_WRITE = 1,
    GPUIO_REQ_COPY = 2,
    GPUIO_REQ_SCATTER = 3,
    GPUIO_REQ_GATHER = 4,
    GPUIO_REQ_ATOMIC_CMP_SWAP = 5,
    GPUIO_REQ_ATOMIC_FETCH_ADD = 6,
    GPUIO_REQ_BATCH = 7,
} gpuio_request_type_t;

typedef enum {
    GPUIO_STATUS_PENDING = 0,
    GPUIO_STATUS_SUBMITTED = 1,
    GPUIO_STATUS_IN_PROGRESS = 2,
    GPUIO_STATUS_COMPLETED = 3,
    GPUIO_STATUS_ERROR = 4,
    GPUIO_STATUS_CANCELLED = 5,
} gpuio_request_status_t;

typedef enum {
    GPUIO_ENGINE_MEMIO = 0,
    GPUIO_ENGINE_LOCALIO = 1,
    GPUIO_ENGINE_REMOTEIO = 2,
} gpuio_io_engine_t;

/* Callback for async completion */
typedef void (*gpuio_callback_t)(gpuio_request_t request, gpuio_error_t status,
                                  void* user_data);

/* Request builder */
typedef struct {
    gpuio_request_type_t type;
    gpuio_io_engine_t engine;
    
    /* Source */
    gpuio_memory_region_t* src;
    uint64_t src_offset;
    
    /* Destination */
    gpuio_memory_region_t* dst;
    uint64_t dst_offset;
    
    /* Size */
    size_t length;
    
    /* Control */
    gpuio_stream_t stream;
    bool async;
    gpuio_callback_t callback;
    void* user_data;
    uint64_t timeout_us;
    
    /* Priority */
    int priority;  /* 0 = highest */
} gpuio_request_params_t;

gpuio_error_t gpuio_request_create(gpuio_context_t ctx,
                                    const gpuio_request_params_t* params,
                                    gpuio_request_t* request);
gpuio_error_t gpuio_request_destroy(gpuio_context_t ctx, gpuio_request_t request);
gpuio_error_t gpuio_request_submit(gpuio_context_t ctx, gpuio_request_t request);
gpuio_error_t gpuio_request_wait(gpuio_context_t ctx, gpuio_request_t request,
                                  uint64_t timeout_us);
gpuio_error_t gpuio_request_cancel(gpuio_context_t ctx, gpuio_request_t request);
gpuio_error_t gpuio_request_get_status(gpuio_context_t ctx, 
                                         gpuio_request_t request,
                                         gpuio_request_status_t* status);

/* Batch operations */
typedef struct {
    gpuio_request_t* requests;
    int num_requests;
    bool ordered;  /* Maintain order */
    gpuio_callback_t batch_callback;
    void* user_data;
} gpuio_batch_t;

gpuio_error_t gpuio_batch_submit(gpuio_context_t ctx, gpuio_batch_t* batch);
gpuio_error_t gpuio_batch_wait(gpuio_context_t ctx, gpuio_batch_t* batch,
                                uint64_t timeout_us);

/* ============================================================================
 * Progress Tracking
 * ============================================================================ */

typedef struct {
    uint64_t bytes_total;
    uint64_t bytes_completed;
    uint64_t bytes_transferred;
    float progress_percent;
    uint64_t eta_ms;
    gpuio_request_status_t status;
    gpuio_error_t error;
} gpuio_progress_t;

typedef void (*gpuio_progress_callback_t)(const gpuio_progress_t* progress,
                                           void* user_data);

gpuio_error_t gpuio_request_set_progress_callback(
    gpuio_context_t ctx,
    gpuio_request_t request,
    gpuio_progress_callback_t callback,
    void* user_data);

gpuio_error_t gpuio_request_get_progress(gpuio_context_t ctx,
                                          gpuio_request_t request,
                                          gpuio_progress_t* progress);

/* ============================================================================
 * On-Demand Proxy (Transparent IO)
 * ============================================================================ */

/**
 * Create a proxy object for on-demand data access.
 * The proxy intercepts array indexing operations and triggers IO automatically.
 */
gpuio_error_t gpuio_proxy_create(gpuio_context_t ctx, const char* uri,
                                  size_t size, gpuio_proxy_t* proxy);
gpuio_error_t gpuio_proxy_destroy(gpuio_context_t ctx, gpuio_proxy_t proxy);

/**
 * Map proxy to GPU-accessible memory.
 * Returns a pointer that can be used in GPU kernels.
 */
gpuio_error_t gpuio_proxy_map(gpuio_context_t ctx, gpuio_proxy_t proxy,
                               void** gpu_ptr);
gpuio_error_t gpuio_proxy_unmap(gpuio_context_t ctx, gpuio_proxy_t proxy);

/**
 * Prefetch data into cache.
 */
gpuio_error_t gpuio_proxy_prefetch(gpuio_context_t ctx, gpuio_proxy_t proxy,
                                    uint64_t offset, size_t size,
                                    gpuio_stream_t stream);

/**
 * Set cache policy for proxy.
 */
typedef enum {
    GPUIO_CACHE_DEFAULT = 0,
    GPUIO_CACHE_LRU = 1,
    GPUIO_CACHE_LFU = 2,
    GPUIO_CACHE_ADAPTIVE = 3,
} gpuio_cache_policy_t;

gpuio_error_t gpuio_proxy_set_cache_policy(gpuio_context_t ctx, 
                                            gpuio_proxy_t proxy,
                                            gpuio_cache_policy_t policy);

/* ============================================================================
 * URI Schemes
 * ============================================================================ */

/*
 * Supported URI formats:
 * 
 * Memory:
 *   mem://cpu/pinned              - Pinned CPU memory
 *   mem://gpu/0                   - GPU 0 memory
 *   mem://cxl/pool0               - CXL memory pool
 * 
 * Local Storage:
 *   local:///dev/nvme0n1          - Raw NVMe device
 *   local:///mnt/data/file.dat    - File path
 *   nvme://uuid/partition         - NVMe by UUID
 * 
 * Remote:
 *   rdma://192.168.1.100:5555/resource   - RDMA transport
 *   tcp://hostname:port/resource         - TCP fallback
 */

/* ============================================================================
 * Statistics and Profiling
 * ============================================================================ */

typedef struct {
    /* Request stats */
    uint64_t requests_submitted;
    uint64_t requests_completed;
    uint64_t requests_failed;
    uint64_t requests_cancelled;
    
    /* Bandwidth */
    double bandwidth_gbps;
    uint64_t bytes_read;
    uint64_t bytes_written;
    
    /* Latency */
    double latency_avg_us;
    double latency_p50_us;
    double latency_p99_us;
    
    /* Cache */
    uint64_t cache_hits;
    uint64_t cache_misses;
    double cache_hit_rate;
} gpuio_stats_t;

gpuio_error_t gpuio_get_stats(gpuio_context_t ctx, gpuio_stats_t* stats);
gpuio_error_t gpuio_reset_stats(gpuio_context_t ctx);

/* ============================================================================
 * Version Information
 * ============================================================================ */

void gpuio_get_version(int* major, int* minor, int* patch);
const char* gpuio_get_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* GPUIO_H */
