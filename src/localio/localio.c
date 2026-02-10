/**
 * @file localio.c
 * @brief LocalIO module - Main operations
 * @version 1.0.0
 */

#include "localio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

/* Worker thread function */
static void* localio_worker(void* arg) {
    localio_context_t* ctx = (localio_context_t*)arg;
    
    while (ctx->worker_running) {
        /* Process queued requests */
        localio_queue_process(&ctx->queue);
        
        /* Small sleep to prevent busy waiting */
        usleep(1000);
    }
    
    return NULL;
}

localio_context_t* localio_context_create(gpuio_context_t parent) {
    if (!parent) return NULL;
    
    localio_context_t* ctx = calloc(1, sizeof(localio_context_t));
    if (!ctx) return NULL;
    
    ctx->parent = parent;
    pthread_mutex_init(&ctx->files_lock, NULL);
    
    /* Initialize queue */
    if (localio_queue_init(&ctx->queue) != 0) {
        free(ctx);
        return NULL;
    }
    
    /* Initialize GDS */
    localio_gds_init(ctx);
    
    /* Start worker thread */
    ctx->worker_running = 1;
    pthread_create(&ctx->worker_thread, NULL, localio_worker, ctx);
    
    return ctx;
}

void localio_context_destroy(localio_context_t* ctx) {
    if (!ctx) return;
    
    /* Stop worker thread */
    ctx->worker_running = 0;
    pthread_join(ctx->worker_thread, NULL);
    
    /* Cleanup GDS */
    localio_gds_cleanup(ctx);
    
    /* Close all files */
    pthread_mutex_lock(&ctx->files_lock);
    for (int i = 0; i < ctx->num_files; i++) {
        localio_file_close(ctx->files[i]);
    }
    free(ctx->files);
    pthread_mutex_unlock(&ctx->files_lock);
    
    /* Cleanup queue */
    localio_queue_cleanup(&ctx->queue);
    
    pthread_mutex_destroy(&ctx->files_lock);
    free(ctx);
}

int localio_read(localio_context_t* ctx, const char* path, void* buf,
                 size_t count, uint64_t offset) {
    if (!ctx || !path || !buf) return -1;
    
    /* Open file */
    localio_file_t* file;
    if (localio_file_open(ctx, path, GPUIO_MEM_READ, &file) != 0) {
        return -1;
    }
    
    /* Read data */
    size_t bytes_read;
    int ret = localio_file_read(file, buf, count, offset, &bytes_read);
    
    /* Close file */
    localio_file_close(file);
    
    if (ret == 0) {
        ctx->bytes_read += bytes_read;
        ctx->io_count++;
    }
    
    return ret;
}

int localio_write(localio_context_t* ctx, const char* path, const void* buf,
                  size_t count, uint64_t offset) {
    if (!ctx || !path || !buf) return -1;
    
    /* Open file */
    localio_file_t* file;
    if (localio_file_open(ctx, path, GPUIO_MEM_WRITE, &file) != 0) {
        return -1;
    }
    
    /* Write data */
    size_t bytes_written;
    int ret = localio_file_write(file, buf, count, offset, &bytes_written);
    
    /* Sync if needed */
    localio_file_sync(file);
    
    /* Close file */
    localio_file_close(file);
    
    if (ret == 0) {
        ctx->bytes_written += bytes_written;
        ctx->io_count++;
    }
    
    return ret;
}

int localio_read_gpu(localio_context_t* ctx, const char* path, void* gpu_buf,
                     size_t count, uint64_t offset) {
    if (!ctx || !path || !gpu_buf) return -1;
    
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        fd = open(path, O_RDONLY);
        if (fd < 0) return -1;
    }
    
    int ret = localio_gds_read(ctx, fd, gpu_buf, count, offset);
    close(fd);
    
    if (ret == 0) {
        ctx->bytes_read += count;
        ctx->io_count++;
    }
    
    return ret;
}

int localio_write_gpu(localio_context_t* ctx, const char* path, 
                      const void* gpu_buf, size_t count, uint64_t offset) {
    if (!ctx || !path || !gpu_buf) return -1;
    
    int fd = open(path, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (fd < 0) {
        fd = open(path, O_WRONLY | O_CREAT, 0644);
        if (fd < 0) return -1;
    }
    
    int ret = localio_gds_write(ctx, fd, gpu_buf, count, offset);
    
    if (ret == 0) {
        fsync(fd);
        ctx->bytes_written += count;
        ctx->io_count++;
    }
    
    close(fd);
    return ret;
}

int localio_get_stats(localio_context_t* ctx, uint64_t* bytes_read,
                      uint64_t* bytes_written, uint64_t* io_count) {
    if (!ctx) return -1;
    
    if (bytes_read) *bytes_read = ctx->bytes_read;
    if (bytes_written) *bytes_written = ctx->bytes_written;
    if (io_count) *io_count = ctx->io_count;
    
    return 0;
}
