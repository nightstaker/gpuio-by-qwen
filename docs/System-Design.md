# System Design Document (SDD)

## GPU-Initiated IO Accelerator (gpuio)

**Version:** 1.0  
**Date:** 2026-02-09  
**Status:** Draft

---

## 1. Detailed Design Overview

This document provides detailed design specifications for the gpuio implementation, including data structures, algorithms, protocols, and interface definitions.

---

## 2. Core Data Structures

### 2.1 Context and Handle Types

```c
/* Main context handle */
typedef struct gpuio_context* gpuio_context_t;
typedef struct gpuio_stream* gpuio_stream_t;
typedef struct gpuio_event* gpuio_event_t;
typedef struct gpuio_request* gpuio_request_t;
typedef struct gpuio_handle* gpuio_handle_t;

/* Configuration structures */
typedef struct {
    uint32_t api_version;
    uint32_t flags;
    int max_gpus;
    size_t memory_pool_size;
    gpuio_log_level_t log_level;
    const char* config_file;
} gpuio_config_t;

typedef struct {
    int device_id;
    gpuio_gpu_vendor_t vendor;
    size_t total_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int supports_gds;           /* GPUDirect Storage */
    int supports_gdr;           /* GPUDirect RDMA */
    int supports_cxl;           /* CXL memory */
} gpuio_device_info_t;
```

### 2.2 Memory Management Structures

```c
/* Memory region handle */
typedef struct {
    void* base_addr;            /* CPU virtual address */
    void* gpu_addr;             /* GPU virtual address */
    uint64_t bus_addr;          /* Physical/bus address for RDMA */
    size_t length;
    gpuio_mem_flags_t flags;
    gpuio_mem_handle_t handle;
    int registered;             /* Registration status */
} gpuio_memory_region_t;

/* Memory pool */
typedef struct {
    void* base;
    size_t capacity;
    size_t used;
    gpuio_memory_region_t* regions;
    int num_regions;
    pthread_mutex_t lock;
    gpuio_allocator_t* allocator;
} gpuio_memory_pool_t;

/* Virtual memory mapping */
typedef struct {
    uint64_t va;                /* Virtual address */
    uint64_t pa;                /* Physical address */
    uint64_t size;
    int gpu_id;
    int protection;
    gpuio_handle_t backing;     /* Backing store handle */
} gpuio_vma_t;
```

### 2.3 IO Request Structure

```c
/* IO request - the core unit of work */
typedef struct gpuio_request {
    /* Request identification */
    uint64_t id;
    gpuio_request_type_t type;      /* READ, WRITE, etc. */
    gpuio_io_engine_t engine;       /* MEMIO, LOCALIO, REMOTEIO */
    
    /* Source/destination */
    gpuio_memory_region_t* src;
    gpuio_memory_region_t* dst;
    uint64_t src_offset;
    uint64_t dst_offset;
    size_t length;
    
    /* Control */
    gpuio_stream_t stream;
    gpuio_request_flags_t flags;
    gpuio_callback_t callback;
    void* user_data;
    
    /* Status */
    volatile gpuio_status_t status;
    size_t bytes_completed;
    int error_code;
    
    /* Engine-specific data */
    union {
        gpuio_memio_params_t memio;
        gpuio_localio_params_t localio;
        gpuio_remoteio_params_t remoteio;
    } params;
    
    /* Linked list for queueing */
    struct gpuio_request* next;
    struct gpuio_request* prev;
} gpuio_request_t;

/* Request types */
typedef enum {
    GPUIO_REQ_READ = 0,
    GPUIO_REQ_WRITE,
    GPUIO_REQ_COPY,
    GPUIO_REQ_ATOMIC_CMP_SWAP,
    GPUIO_REQ_ATOMIC_FETCH_ADD,
    GPUIO_REQ_BATCH
} gpuio_request_type_t;

/* Request status */
typedef enum {
    GPUIO_STATUS_PENDING = 0,
    GPUIO_STATUS_SUBMITTED,
    GPUIO_STATUS_IN_PROGRESS,
    GPUIO_STATUS_COMPLETED,
    GPUIO_STATUS_ERROR,
    GPUIO_STATUS_CANCELLED
} gpuio_status_t;
```

### 2.4 On-Demand Proxy Object

```c
/* Proxy object for transparent access */
typedef struct {
    /* URI and identification */
    char uri[GPUIO_MAX_URI_LEN];
    gpuio_resource_type_t resource_type;
    
    /* Virtual memory space */
    gpuio_vma_t* vma_table;
    int vma_count;
    size_t total_size;
    
    /* Cache management */
    gpuio_cache_t* cache;
    gpuio_prefetcher_t* prefetcher;
    
    /* Page table for on-demand translation */
    gpuio_page_table_t* page_table;
    
    /* Engine handle */
    gpuio_handle_t io_handle;
    
    /* Statistics */
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t bytes_transferred;
    
    /* Thread safety */
    pthread_rwlock_t lock;
    
    /* GPU-specific data */
    void* gpu_data;             /* Device-resident metadata */
} gpuio_proxy_object_t;

/* Proxy cache entry */
typedef struct {
    uint64_t offset;
    void* gpu_ptr;
    size_t size;
    uint64_t last_access;
    uint32_t access_count;
    gpuio_cache_state_t state;
} gpuio_cache_entry_t;
```

---

## 3. Core Subsystem Designs

### 3.1 Scheduler Design

```c
/* Work queue structure */
typedef struct {
    gpuio_request_t* head;
    gpuio_request_t* tail;
    int count;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} gpuio_work_queue_t;

/* Scheduler state */
typedef struct {
    gpuio_work_queue_t* queues;     /* Per-priority queues */
    int num_queues;
    
    /* Worker threads */
    pthread_t* workers;
    int num_workers;
    
    /* Work stealing */
    gpuio_work_queue_t** per_thread_queues;
    
    /* Statistics */
    uint64_t requests_submitted;
    uint64_t requests_completed;
    uint64_t requests_failed;
    
    /* Control */
    volatile int running;
} gpuio_scheduler_t;

/* Scheduling algorithms */
typedef enum {
    GPUIO_SCHED_FIFO = 0,       /* First-in-first-out */
    GPUIO_SCHED_PRIORITY,       /* Priority-based */
    GPUIO_SCHED_DEADLINE,       /* Earliest deadline first */
    GPUIO_SCHED_ADAPTIVE        /* Adaptive based on workload */
} gpuio_sched_policy_t;
```

**Scheduling Algorithm:**

1. **Priority-Based Preemption**:
   - High priority: System operations, small latency-sensitive requests
   - Medium priority: Standard user requests
   - Low priority: Background prefetch, bulk transfers

2. **Work Stealing**:
   - Idle workers steal from busy peers
   - Reduces tail latency under load

3. **Batching**:
   - Automatically batch small requests
   - Improves throughput for small IOs

### 3.2 Cache Design

```c
/* Multi-level cache */
typedef struct {
    /* L1 Cache - GPU Memory */
    struct {
        void* buffer;
        size_t capacity;
        gpuio_lru_list_t entries;
        pthread_mutex_t lock;
    } l1;
    
    /* L2 Cache - CPU Pinned Memory */
    struct {
        void* buffer;
        size_t capacity;
        gpuio_lru_list_t entries;
        pthread_mutex_t lock;
    } l2;
    
    /* Cache policy */
    gpuio_cache_policy_t policy;
    double l1_ratio;            /* Target L1/L2 ratio */
    
    /* Statistics */
    uint64_t l1_hits, l1_misses;
    uint64_t l2_hits, l2_misses;
    uint64_t evictions;
} gpuio_cache_t;

/* Prefetch engine */
typedef struct {
    /* Sequential prefetch */
    uint64_t last_accessed_offset;
    int sequential_count;
    size_t prefetch_size;
    
    /* History-based prediction */
    gpuio_access_pattern_t* history;
    int history_size;
    
    /* Machine learning predictor (optional) */
    gpuio_ml_predictor_t* predictor;
} gpuio_prefetcher_t;
```

**Cache Coherence Protocol**:

```
States: INVALID → VALID → DIRTY → FLUSHING → VALID

Read Hit (VALID): Return data immediately
Read Miss: Load from backing store → Set VALID
Write Hit (VALID): Update data → Set DIRTY
Write Miss: Allocate entry → Load → Update → Set DIRTY
Flush (DIRTY): Write to backing store → Set FLUSHING → Set VALID
Eviction (DIRTY): Flush first, then evict
Eviction (VALID): Direct eviction
```

### 3.3 Memory Registration Cache

```c
/* Memory registration cache for RDMA/NVMe */
typedef struct {
    gpuio_memory_region_t* entries;
    int capacity;
    int count;
    
    /* Hash table for O(1) lookup */
    gpuio_hash_table_t* hash_table;
    
    /* LRU eviction */
    gpuio_lru_list_t lru_list;
    
    /* Statistics */
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
} gpuio_mr_cache_t;

/* Registration lookup */
gpuio_memory_region_t* gpuio_mr_lookup(gpuio_mr_cache_t* cache, 
                                        void* addr, size_t len);
int gpuio_mr_register(gpuio_mr_cache_t* cache, void* addr, 
                       size_t len, gpuio_mem_handle_t* handle);
int gpuio_mr_deregister(gpuio_mr_cache_t* cache, gpuio_mem_handle_t handle);
```

---

## 4. IO Engine Designs

### 4.1 MemIO Engine

```c
/* MemIO-specific parameters */
typedef struct {
    /* Source/target memory */
    int src_gpu_id;             /* -1 for CPU */
    int dst_gpu_id;             /* -1 for CPU */
    
    /* Access type */
    int use_load_store;         /* Direct load/store vs memcpy */
    int use_atomics;            /* Enable atomic operations */
    
    /* CXL-specific */
    int cxl_device_id;
    cxl_memory_type_t cxl_type;
    
    /* NUMA awareness */
    int preferred_numa_node;
} gpuio_memio_params_t;

/* MemIO implementation */
typedef struct {
    /* CXL device management */
    gpuio_cxl_device_t* cxl_devices;
    int num_cxl_devices;
    
    /* HMM integration */
    hmm_pfn_t* pfn_table;
    
    /* Kernel interface */
    int hmm_fd;
    int cxl_mem_fd;
    
    /* Statistics */
    uint64_t bytes_transferred_cpu;
    uint64_t bytes_transferred_cxl;
    uint64_t page_faults;
} gpuio_memio_engine_t;
```

**MemIO Operation Flow**:

```
1. Check if addresses are in same memory domain
   ├── Same GPU memory: Use device memcpy
   ├── GPU ↔ CPU: Use cuMemcpy/cudaMemcpy
   ├── GPU ↔ CXL: Use CXL.mem protocol
   └── CPU ↔ CXL: Use load/store (CPU) or ATS (GPU)

2. For on-demand access:
   a. Check TLB for virtual→physical translation
   b. If miss: Walk page table, may trigger page fault
   c. Page fault handler allocates/migrates memory
   d. Update TLB and retry access

3. For explicit transfer:
   a. Pin source/destination memory
   b. Queue DMA operation
   c. Wait for completion
   d. Unpin memory
```

### 4.2 LocalIO Engine

```c
/* LocalIO-specific parameters */
typedef struct {
    /* File/descriptor */
    int fd;
    const char* path;
    
    /* File offset */
    uint64_t file_offset;
    
    /* Options */
    int use_direct_io;          /* O_DIRECT */
    int use_gds;                /* GPUDirect Storage */
    int use_spdk;               /* SPDK backend */
    
    /* RAID/striping */
    int stripe_size;
    int num_devices;
} gpuio_localio_params_t;

/* LocalIO implementation */
typedef struct {
    /* File descriptor cache */
    gpuio_fd_cache_t* fd_cache;
    
    /* GDS context */
    CUfileHandle_t* cufile_handles;
    CUfileDescr_t* cufile_desc;
    
    /* SPDK context */
    struct spdk_nvme_ctrlr** spdk_ctrlrs;
    struct spdk_nvme_ns** spdk_namespaces;
    int num_spdk_devices;
    
    /* Thread pool for non-GDS ops */
    thread_pool_t* io_threads;
} gpuio_localio_engine_t;
```

**LocalIO Operation Flow**:

```
1. Open file/device
   ├── Regular file: Standard open()
   ├── Block device: O_DIRECT
   └── NVMe: SPDK init or GDS

2. Register GPU memory with GDS (if using GDS)
   └── cuFileBufRegister()

3. Submit IO
   ├── GDS path: cuFileRead/Write()
   ├── SPDK path: spdk_nvme_ns_cmd_read/write()
   └── Kernel path: pread/pwrite with bounce buffer

4. Poll for completion
   ├── GDS: cuFileStreamSynchronize() or event
   ├── SPDK: Completion queue polling
   └── Kernel: io_getevents() or blocking

5. Cleanup
   └── cuFileBufDeregister() (if GDS)
```

### 4.3 RemoteIO Engine

```c
/* RemoteIO-specific parameters */
typedef struct {
    /* Connection info */
    char remote_addr[INET6_ADDRSTRLEN];
    uint16_t remote_port;
    
    /* RDMA parameters */
    int use_gdr;                /* GPUDirect RDMA */
    int qp_type;                /* RC, UC, or UD */
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    
    /* Transport */
    gpuio_transport_type_t transport;  /* VERBS, LIBFABRIC, UCX */
} gpuio_remoteio_params_t;

/* RDMA connection */
typedef struct {
    /* Transport context */
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
    struct ibv_qp* qp;
    
    /* Memory regions */
    gpuio_mr_cache_t* mr_cache;
    
    /* Connection state */
    enum { INIT, RTR, RTS, ERROR } state;
    
    /* Remote info */
    uint32_t remote_qpn;
    uint16_t remote_lid;
    uint8_t remote_gid[16];
} gpuio_rdma_connection_t;

/* RemoteIO implementation */
typedef struct {
    /* Connection pool */
    gpuio_rdma_connection_t** connections;
    int num_connections;
    pthread_mutex_t conn_lock;
    
    /* Transport abstraction */
    gpuio_transport_ops_t* transport;
    
    /* Completion thread */
    pthread_t comp_thread;
    int comp_running;
} gpuio_remoteio_engine_t;
```

**RemoteIO Operation Flow**:

```
1. Connection establishment
   a. Resolve remote address (DNS/RDMA CM)
   b. Create protection domain and completion queue
   c. Create queue pair
   d. Exchange connection info (out-of-band)
   e. Transition QP through states: INIT → RTR → RTS

2. Memory registration (GDR path)
   a. ibv_reg_mr() with GPU memory
   b. Exchange memory region info (rkey, vaddr)

3. RDMA operation
   a. Post RDMA Read/Write work request
   b. Poll completion queue for completion
   c. Handle errors and retries

4. Connection teardown
   a. Flush outstanding operations
   b. Destroy QP
   c. Deregister memory regions
```

---

## 5. On-Demand Access Design

### 5.1 Array Index Operator Hook

```cpp
// C++ Proxy Object Template
template<typename T>
class GPUIOArray {
public:
    // Constructor binds to remote resource
    GPUIOArray(const std::string& uri, size_t size);
    
    // Destructor releases resources
    ~GPUIOArray();
    
    // Array index operator - triggers on-demand load
    __device__ T operator[](size_t index) {
        return gpuio_on_demand_load(this->handle_, index);
    }
    
    // Batch prefetch hint
    void prefetch(size_t start, size_t count);
    
    // Explicit synchronization
    void synchronize();
    
private:
    gpuio_handle_t handle_;
    size_t size_;
    gpuio_proxy_object_t* proxy_;
};

// CUDA device function for on-demand load
__device__ T gpuio_on_demand_load(gpuio_handle_t handle, size_t index) {
    // Check L1 cache (GPU memory)
    gpuio_cache_entry_t* entry = gpuio_l1_lookup(handle, index);
    if (entry) {
        return *((T*)entry->gpu_ptr + (index - entry->offset));
    }
    
    // Cache miss - trigger transfer
    // This would typically involve:
    // 1. Allocating GPU cache line
    // 2. Enqueuing async transfer request
    // 3. Stalling or returning placeholder
    
    return gpuio_trigger_transfer_and_wait(handle, index);
}
```

### 5.2 Page Fault Handler

```c
/* Page fault handling */
typedef struct {
    /* Fault info */
    void* fault_addr;
    int gpu_id;
    gpuio_fault_type_t type;    /* READ, WRITE, ATOMIC */
    
    /* Resolution */
    gpuio_vma_t* vma;
    size_t page_offset;
    void* page_buffer;
} gpuio_page_fault_t;

/* Fault handler function */
int gpuio_handle_page_fault(gpuio_page_fault_t* fault) {
    /* 1. Validate fault address */
    if (!gpuio_vma_contains(fault->vma, fault->fault_addr)) {
        return GPUIO_ERROR_INVALID_ADDRESS;
    }
    
    /* 2. Check if page is in transit */
    if (gpuio_page_in_transit(fault)) {
        return gpuio_wait_for_page(fault);
    }
    
    /* 3. Allocate GPU memory for page */
    void* gpu_page = gpuio_alloc_gpu_page(fault->gpu_id);
    if (!gpu_page) {
        /* Evict LRU page */
        gpu_page = gpuio_evict_lru_page(fault->gpu_id);
    }
    
    /* 4. Initiate transfer based on backing type */
    switch (fault->vma->backing->type) {
        case GPUIO_BACKING_MEMIO:
            return gpuio_memio_fault_handler(fault, gpu_page);
        case GPUIO_BACKING_LOCALIO:
            return gpuio_localio_fault_handler(fault, gpu_page);
        case GPUIO_BACKING_REMOTEIO:
            return gpuio_remoteio_fault_handler(fault, gpu_page);
        default:
            return GPUIO_ERROR_INVALID_BACKING;
    }
}
```

---

## 6. Protocol Specifications

### 6.1 Resource URI Format

```
URI Format: gpuio://<engine>/<resource>[?<params>]

Examples:
  memio://cpu/dram?numa=0
  memio://cxl/pool0?device=0
  localio:///dev/nvme0n1
  localio:///mnt/data/file.dat?direct=1&gds=1
  remoteio://192.168.1.100:5555/dataset?transport=rdma&gdr=1

Components:
  engine   : memio | localio | remoteio
  resource : device path, file path, or remote identifier
  params   : key=value pairs for configuration
```

### 6.2 Wire Protocol (RemoteIO)

```c
/* Message header */
typedef struct __attribute__((packed)) {
    uint32_t magic;             /* GPUIO_MAGIC = 0x47505549 */
    uint16_t version;
    uint16_t msg_type;
    uint32_t flags;
    uint64_t request_id;
    uint64_t payload_len;
} gpuio_msg_header_t;

/* Message types */
#define GPUIO_MSG_HELLO         0x01
#define GPUIO_MSG_HELLO_ACK     0x02
#define GPUIO_MSG_READ_REQ      0x10
#define GPUIO_MSG_WRITE_REQ     0x11
#define GPUIO_MSG_READ_RESP     0x12
#define GPUIO_MSG_WRITE_RESP    0x13
#define GPUIO_MSG_RDMA_INFO     0x20
#define GPUIO_MSG_ERROR         0xFF

/* RDMA connection info */
typedef struct __attribute__((packed)) {
    uint32_t qp_num;
    uint16_t lid;
    uint8_t  gid[16];
    uint32_t rkey;
    uint64_t vaddr;
} gpuio_rdma_info_t;
```

---

## 7. Error Handling

### 7.1 Error Codes

```c
/* Error codes */
#define GPUIO_SUCCESS           0
#define GPUIO_ERROR_GENERAL     -1
#define GPUIO_ERROR_NOMEM       -2
#define GPUIO_ERROR_INVALID_ARG -3
#define GPUIO_ERROR_NOT_FOUND   -4
#define GPUIO_ERROR_TIMEOUT     -5
#define GPUIO_ERROR_IO          -6
#define GPUIO_ERROR_NETWORK     -7
#define GPUIO_ERROR_UNSUPPORTED -8
#define GPUIO_ERROR_PERMISSION  -9
#define GPUIO_ERROR_BUSY        -10
#define GPUIO_ERROR_CANCELED    -11
```

### 7.2 Retry Logic

```c
/* Retry configuration */
typedef struct {
    int max_retries;
    uint64_t initial_backoff_us;
    uint64_t max_backoff_us;
    double backoff_multiplier;
} gpuio_retry_policy_t;

/* Retry with exponential backoff */
int gpuio_retry_with_backoff(gpuio_retry_policy_t* policy,
                              gpuio_operation_fn_t op,
                              void* ctx) {
    int retries = 0;
    uint64_t backoff = policy->initial_backoff_us;
    
    while (retries < policy->max_retries) {
        int ret = op(ctx);
        if (ret == GPUIO_SUCCESS) {
            return GPUIO_SUCCESS;
        }
        
        if (!gpuio_error_is_retryable(ret)) {
            return ret;
        }
        
        usleep(backoff);
        backoff = min(backoff * policy->backoff_multiplier,
                      policy->max_backoff_us);
        retries++;
    }
    
    return GPUIO_ERROR_TIMEOUT;
}
```

---

## 8. Performance Optimizations

### 8.1 Batch Processing

```c
/* Batch request */
typedef struct {
    gpuio_request_t** requests;
    int num_requests;
    gpuio_batch_flags_t flags;
    
    /* Completion tracking */
    int completed;
    gpuio_callback_t batch_callback;
} gpuio_batch_t;

/* Batch submit with coalescing */
int gpuio_batch_submit(gpuio_batch_t* batch) {
    /* 1. Sort requests by address for locality */
    qsort(batch->requests, batch->num_requests, 
          sizeof(gpuio_request_t*), gpuio_compare_by_addr);
    
    /* 2. Coalesce adjacent requests */
    gpuio_request_t** coalesced;
    int num_coalesced = gpuio_coalesce_requests(
        batch->requests, batch->num_requests, &coalesced);
    
    /* 3. Group by engine */
    gpuio_engine_groups_t groups;
    gpuio_group_by_engine(coalesced, num_coalesced, &groups);
    
    /* 4. Submit batches per engine */
    for (int i = 0; i < groups.num_engines; i++) {
        gpuio_engine_batch_submit(groups.engines[i],
                                   groups.requests[i],
                                   groups.counts[i]);
    }
    
    return GPUIO_SUCCESS;
}
```

### 8.2 Zero-Copy Techniques

```
Zero-Copy Paths:

1. MemIO (GPU ↔ CPU DRAM)
   - HMM: Unified memory with on-demand migration
   - GPUDirect: DMA between GPU and pinned CPU memory
   - CXL: Load/store through CXL.mem fabric

2. LocalIO (GPU ↔ NVMe)
   - GPUDirect Storage: GPU DMA to NVMe
   - SPDK: Userspace NVMe driver with GPU buffers

3. RemoteIO (GPU ↔ Network)
   - GPUDirect RDMA: RDMA between GPU memory and NIC
   - ATS: Address Translation Services for direct access
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

- Memory registration/deregistration
- Request creation and submission
- Cache operations
- Error handling paths

### 9.2 Integration Tests

- End-to-end data transfers
- Multi-GPU scenarios
- Concurrent access patterns
- Error injection and recovery

### 9.3 Performance Tests

- Bandwidth benchmarks (memio, localio, remoteio)
- Latency microbenchmarks
- Stress tests (high concurrency)
- Scalability tests (node count)

---

## 10. DeepSeek-Specific Design Extensions

### 10.1 DSA KV Cache Data Structures

```c
/* DSA (Dynamic Sparse Attention) KV Cache Entry */
typedef struct {
    uint64_t engram_id;             /* Unique identifier */
    uint32_t head_id;               /* Attention head index */
    uint32_t layer_id;              /* Transformer layer */
    uint64_t position;              /* Token position */
    
    /* Sparsity metadata */
    float importance_score;         /* DSA importance (0.0 - 1.0) */
    uint32_t sparsity_pattern;      /* Bitmask of active heads */
    uint8_t compression_level;      /* 0=none, 1=FP16, 2=INT8, 3=custom */
    
    /* Location tracking */
    gpuio_kv_location_t location;   /* HBM, CXL, or Remote */
    uint64_t compressed_size;
    uint64_t original_size;
    
    /* Cache management */
    uint64_t last_access_time;
    uint32_t access_count;
    uint32_t eviction_priority;     /* Derived from importance_score */
} gpuio_dsa_kv_entry_t;

/* DSA KV Cache Pool */
typedef struct {
    /* HBM (Hot) Tier */
    struct {
        gpuio_dsa_kv_entry_t* entries;
        void* data_buffer;
        size_t capacity;
        size_t used;
        gpuio_lru_cache_t* index;
    } hbm_tier;
    
    /* CXL (Warm) Tier */
    struct {
        gpuio_dsa_kv_entry_t* entries;
        void* cxl_base_addr;
        size_t capacity;
        gpuio_cxl_handle_t cxl_handle;
    } cxl_tier;
    
    /* Remote (Cold) Tier */
    struct {
        char* archive_uri;          /* rdma:// or nvme:// */
        gpuio_handle_t remote_handle;
        gpuio_compression_t* compressor;
    } remote_tier;
    
    /* Management */
    gpuio_dsa_router_t* router;     /* Learned routing for engram lookup */
    pthread_mutex_t lock;
    uint64_t total_hits;
    uint64_t total_misses;
} gpuio_dsa_kv_pool_t;

/* DSA Router - Learned addressing */
typedef struct {
    /* Neural index for content-based routing */
    void* neural_index;             /* GPU-resident index structure */
    uint32_t num_shards;
    
    /* Routing table */
    struct {
        uint64_t shard_id;
        gpuio_kv_location_t preferred_tier;
        float latency_estimate;
    }* routing_table;
    
    /* Prediction model */
    void* lstm_predictor;           /* For access pattern prediction */
} gpuio_dsa_router_t;

/* DSA KV Operations */
int gpuio_dsa_kv_init(gpuio_dsa_kv_pool_t* pool, gpuio_dsa_kv_config_t* config);
int gpuio_dsa_kv_load(gpuio_dsa_kv_pool_t* pool, 
                       uint64_t position, uint32_t head_id,
                       gpuio_dsa_kv_entry_t** entry);
int gpuio_dsa_kv_store(gpuio_dsa_kv_pool_t* pool,
                        uint64_t position, uint32_t head_id,
                        void* data, size_t size,
                        float importance_score);
int gpuio_dsa_kv_compact(gpuio_dsa_kv_pool_t* pool);  /* Compress and migrate */
```

### 10.2 Graph RAG Data Structures

```c
/* Graph Node for RAG */
typedef struct {
    uint64_t node_id;
    uint64_t* edge_ids;             /* Adjacency list (compressed) */
    uint32_t num_edges;
    
    /* Content */
    float* embedding;               /* Vector representation */
    uint32_t embedding_dim;
    char* attributes;               /* JSON-style properties */
    
    /* GPU-cache metadata */
    gpuio_cache_state_t cache_state;
    uint64_t gpu_buffer_offset;
} gpuio_graph_node_t;

/* Graph RAG Request */
typedef struct {
    /* Query specification */
    float* query_embedding;
    uint32_t query_dim;
    int top_k;                      /* Number of candidate nodes */
    int hop_depth;                  /* Multi-hop expansion depth */
    
    /* Scatter phase */
    gpuio_scatter_params_t scatter;
    uint64_t* candidate_indices;    /* Output from vector search */
    int num_candidates;
    
    /* Gather phase */
    gpuio_gather_params_t gather;
    gpuio_graph_node_t** subgraph;  /* Assembled subgraph */
    int subgraph_size;
    
    /* Filters */
    char** edge_type_filter;        /* Only follow these edge types */
    int num_edge_types;
    float min_similarity_threshold;
} gpuio_graph_rag_request_t;

/* Scatter-Gather optimized for graphs */
int gpuio_graph_scatter(gpuio_context_t* ctx,
                         gpuio_graph_rag_request_t* request,
                         gpuio_graph_index_t* index);
int gpuio_graph_gather(gpuio_context_t* ctx,
                        gpuio_graph_rag_request_t* request,
                        gpuio_graph_storage_t* storage);

/* Graph Index (for vector similarity) */
typedef struct {
    /* ANN index structure */
    void* hnsw_index;               /* Hierarchical NSW graph */
    void* ivf_index;                /* Inverted file index (alternative) */
    
    /* Partitioning */
    int num_partitions;
    gpuio_graph_partition_t* partitions;
    
    /* GPU acceleration */
    void* gpu_index;                /* CUDA-accelerated index */
} gpuio_graph_index_t;
```

### 10.3 Engram Memory Data Structures

```c
/* Engram - Unit of external memory */
typedef struct {
    uint64_t engram_id;             /* Unique identifier (content hash) */
    uint64_t version;               /* For cache invalidation */
    
    /* Content */
    void* data;
    size_t size;
    float* embedding;               /* For semantic retrieval */
    uint32_t embedding_dim;
    
    /* Metadata */
    uint64_t creation_timestamp;
    uint64_t last_access_timestamp;
    uint32_t access_count;
    float importance_score;         /* Learned importance */
    
    /* Storage location */
    gpuio_engram_location_t location;
    union {
        struct { void* gpu_ptr; } hbm;
        struct { uint64_t cxl_offset; } cxl;
        struct { char* uri; uint64_t offset; } remote;
    } loc;
} gpuio_engram_t;

/* Engram Pool - Petabyte-scale storage */
typedef struct {
    /* Tiered storage */
    gpuio_engram_t** hbm_cache;     /* Hot engrams in GPU */
    size_t hbm_capacity;
    
    gpuio_engram_t** cxl_pool;      /* Warm engrams in CXL */
    size_t cxl_capacity;
    
    gpuio_handle_t remote_archive;  /* Cold engrams in distributed storage */
    
    /* Indexing */
    gpuio_engram_index_t* index;    /* Content-addressable index */
    
    /* Write management */
    gpuio_write_buffer_t* write_buffer;  /* Async write batching */
    pthread_t flush_thread;
} gpuio_engram_pool_t;

/* Engram Index - Neural + Traditional */
typedef struct {
    /* Vector index for semantic search */
    void* faiss_index;              /* GPU-accelerated FAISS */
    
    /* Hash index for exact lookup */
    gpuio_hash_table_t* hash_index; /* content_hash -> engram_id */
    
    /* Learned routing */
    void* neural_router;            /* Predicts engram location */
} gpuio_engram_index_t;

/* Engram Operations */
int gpuio_engram_init(gpuio_engram_pool_t* pool, gpuio_engram_config_t* config);
int gpuio_engram_read(gpuio_engram_pool_t* pool,
                       float* query_embedding,
                       uint64_t* engram_ids,  /* Optional: exact IDs */
                       gpuio_engram_t** results,
                       int top_k);
int gpuio_engram_write(gpuio_engram_pool_t* pool,
                        gpuio_engram_t* engram,
                        gpuio_write_mode_t mode);  /* SYNC or ASYNC */
int gpuio_engram_query(gpuio_engram_pool_t* pool,
                        float* query_embedding,
                        float similarity_threshold,
                        gpuio_engram_t** results,
                        int* num_results);
```

### 10.4 Compression Algorithms for AI Workloads

```c
/* KV Cache Compression */
typedef enum {
    GPUIO_KV_COMPRESS_NONE = 0,
    GPUIO_KV_COMPRESS_FP16,       /* 2x compression */
    GPUIO_KV_COMPRESS_INT8,       /* 4x compression with calibration */
    GPUIO_KV_COMPRESS_4BIT,       /* 8x compression (GPTQ-style) */
    GPUIO_KV_COMPRESS_SPARSE,     /* Sparse representation for DSA */
} gpuio_kv_compression_t;

/* Engram Compression */
typedef enum {
    GPUIO_ENGRAM_COMPRESS_NONE = 0,
    GPUIO_ENGRAM_COMPRESS_ZSTD,   /* General compression */
    GPUIO_ENGRAM_COMPRESS_LZ4,    /* Fast compression */
    GPUIO_ENGRAM_COMPRESS_EMBED,  /* Embedding-specific quantization */
} gpuio_engram_compression_t;

/* GPU-Accelerated Compression */
int gpuio_compress_kv(gpuio_dsa_kv_entry_t* kv_entry,
                       void* input, size_t input_size,
                       void* output, size_t* output_size,
                       gpuio_kv_compression_t method);
int gpuio_decompress_kv(gpuio_dsa_kv_entry_t* kv_entry,
                         void* input, size_t input_size,
                         void* output, size_t* output_size);

/* Compression on transfer - zero-copy */
int gpuio_transfer_compressed(gpuio_request_t* req,
                                gpuio_compression_t* codec);
```

### 10.5 AI-Specific Scheduling Policies

```c
/* Priority classes for AI workloads */
typedef enum {
    GPUIO_PRIO_INFERENCE_REALTIME = 0,  /* User-facing inference */
    GPUIO_PRIO_TRAINING_FW,             /* Training forward pass */
    GPUIO_PRIO_TRAINING_BW,             /* Training backward pass */
    GPUIO_PRIO_CHECKPOINT,              /* Model checkpointing */
    GPUIO_PRIO_ENGRAM_SYNC,             /* Engram write-back */
    GPUIO_PRIO_BACKGROUND,              /* Prefetch, cleanup */
} gpuio_ai_priority_t;

/* Deadline-aware scheduling for inference */
typedef struct {
    uint64_t deadline_us;           /* SLO deadline */
    uint64_t estimated_duration_us; /* Predicted execution time */
    float criticality;              /* Business impact (0.0 - 1.0) */
} gpuio_inference_qos_t;

/* Engram-aware scheduling */
int gpuio_schedule_engram_prefetch(gpuio_scheduler_t* sched,
                                    gpuio_dsa_router_t* router,
                                    uint64_t* upcoming_tokens,
                                    int num_tokens);

/* KV cache eviction with importance scoring */
int gpuio_evict_by_importance(gpuio_dsa_kv_pool_t* pool,
                               size_t required_space,
                               float importance_threshold);
```

### 10.6 Integration APIs

```c
/* Unified AI Workload API */
typedef struct {
    /* Model context */
    gpuio_model_handle_t model;
    int num_layers;
    int num_heads;
    int head_dim;
    
    /* Memory systems */
    gpuio_dsa_kv_pool_t* kv_pool;
    gpuio_engram_pool_t* engram_pool;
    gpuio_graph_index_t* knowledge_graph;
    
    /* Scheduling */
    gpuio_scheduler_t* scheduler;
    gpuio_ai_priority_t default_priority;
} gpuio_ai_context_t;

/* End-to-end inference with all systems */
int gpuio_ai_inference(gpuio_ai_context_t* ctx,
                        gpuio_inference_request_t* request,
                        gpuio_inference_response_t* response);

/* Training step with checkpoint and engram update */
int gpuio_ai_training_step(gpuio_ai_context_t* ctx,
                            gpuio_training_batch_t* batch,
                            gpuio_checkpoint_policy_t* checkpoint_policy);
```

---

## 11. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-09 | gpuio Team | Initial design document |
| 1.1 | 2026-02-09 | gpuio Team | Added DeepSeek-specific data structures and APIs (DSA, Engram, Graph RAG) |
