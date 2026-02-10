/**
 * @file localio_internal.h
 * @brief LocalIO module internal definitions
 * @version 1.0.0
 */

#ifndef LOCALIO_INTERNAL_H
#define LOCALIO_INTERNAL_H

#include <gpuio/gpuio.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>

/* File handle types */
typedef enum {
    LOCALIO_FILE_REGULAR = 0,
    LOCALIO_FILE_BLOCK,
    LOCALIO_FILE_NVME,
} localio_file_type_t;

/* File handle */
typedef struct localio_file {
    char* path;
    localio_file_type_t type;
    int fd;
    int flags;
    size_t size;
    uint64_t offset;
    
    /* For NVMe direct access */
    int nvme_ns_id;
    
    /* Thread safety */
    pthread_mutex_t lock;
} localio_file_t;

/* IO request queue */
typedef struct localio_request {
    int op; /* READ or WRITE */
    void* buf;
    size_t count;
    uint64_t offset;
    localio_file_t* file;
    gpuio_callback_t callback;
    void* user_data;
    struct localio_request* next;
} localio_request_t;

/* Queue */
typedef struct localio_queue {
    localio_request_t* head;
    localio_request_t* tail;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int depth;
} localio_queue_t;

/* LocalIO context */
typedef struct localio_context {
    gpuio_context_t parent;
    
    /* File handles */
    localio_file_t** files;
    int num_files;
    pthread_mutex_t files_lock;
    
    /* IO queue */
    localio_queue_t queue;
    
    /* Worker thread */
    pthread_t worker_thread;
    int worker_running;
    
    /* GDS support */
    int gds_available;
    void* gds_handle;
    
    /* Compression */
    int compression_enabled;
    int compression_level;
    
    /* Encryption */
    int encryption_enabled;
    void* encryption_ctx;
    
    /* Statistics */
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t io_count;
} localio_context_t;

/* Internal functions */
int localio_file_open(localio_context_t* ctx, const char* path, int flags,
                      localio_file_t** file_out);
int localio_file_close(localio_file_t* file);
int localio_file_read(localio_file_t* file, void* buf, size_t count, 
                      uint64_t offset, size_t* bytes_read);
int localio_file_write(localio_file_t* file, const void* buf, size_t count,
                       uint64_t offset, size_t* bytes_written);

int localio_queue_init(localio_queue_t* queue);
void localio_queue_cleanup(localio_queue_t* queue);
int localio_queue_submit(localio_queue_t* queue, localio_request_t* req);
int localio_queue_process(localio_queue_t* queue);

int localio_gds_init(localio_context_t* ctx);
void localio_gds_cleanup(localio_context_t* ctx);
int localio_gds_read(localio_context_t* ctx, int fd, void* gpu_buf,
                     size_t count, uint64_t offset);
int localio_gds_write(localio_context_t* ctx, int fd, const void* gpu_buf,
                      size_t count, uint64_t offset);

int localio_compress(const void* src, size_t src_len, void* dst, 
                     size_t dst_len, size_t* out_len, int level);
int localio_decompress(const void* src, size_t src_len, void* dst,
                       size_t dst_len, size_t* out_len);

#endif /* LOCALIO_INTERNAL_H */
