/**
 * @file remoteio.c
 * @brief RemoteIO module - Main operations implementation
 * @version 1.0.0
 */

#include "remoteio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

/* Default configuration */
#define REMOTEIO_DEFAULT_PORT        5555
#define REMOTEIO_MAX_CONNECTIONS     64
#define REMOTEIO_MAX_PENDING_OPS     1024
#define REMOTEIO_INLINE_THRESHOLD    256
#define REMOTEIO_OP_TIMEOUT_US       30000000  /* 30 seconds */

static pthread_once_t remoteio_once = PTHREAD_ONCE_INIT;
static int remoteio_initialized = 0;

static void remoteio_init_globals(void) {
    remoteio_initialized = 1;
}

int remoteio_initialize(void) {
    pthread_once(&remoteio_once, remoteio_init_globals);
    return 0;
}

remoteio_context_t* remoteio_context_create(gpuio_context_t parent) {
    if (!parent) return NULL;
    
    remoteio_context_t* ctx = calloc(1, sizeof(remoteio_context_t));
    if (!ctx) return NULL;
    
    ctx->parent = parent;
    ctx->preferred_transport = REMOTEIO_TRANSPORT_AUTO;
    ctx->use_gdr = 1;
    ctx->use_inline = 1;
    ctx->inline_threshold = REMOTEIO_INLINE_THRESHOLD;
    
    /* Initialize locks */
    pthread_mutex_init(&ctx->conn_pool.lock, NULL);
    pthread_mutex_init(&ctx->gdr_lock, NULL);
    pthread_mutex_init(&ctx->ops_lock, NULL);
    pthread_mutex_init(&ctx->stats_lock, NULL);
    pthread_cond_init(&ctx->ops_cond, NULL);
    
    /* Initialize connection pool */
    if (remoteio_conn_pool_init(&ctx->conn_pool, REMOTEIO_MAX_CONNECTIONS) != 0) {
        free(ctx);
        return NULL;
    }
    
    /* Initialize RDMA subsystem */
    if (remoteio_rdma_init(ctx) != 0) {
        ctx->use_gdr = 0;
        /* Continue without RDMA - will use TCP fallback */
    }
    
    /* Initialize network layer */
    if (remoteio_network_init(ctx) != 0) {
        remoteio_conn_pool_cleanup(&ctx->conn_pool);
        pthread_mutex_destroy(&ctx->conn_pool.lock);
        pthread_mutex_destroy(&ctx->gdr_lock);
        pthread_mutex_destroy(&ctx->ops_lock);
        pthread_mutex_destroy(&ctx->stats_lock);
        pthread_cond_destroy(&ctx->ops_cond);
        free(ctx);
        return NULL;
    }
    
    return ctx;
}

void remoteio_context_destroy(remoteio_context_t* ctx) {
    if (!ctx) return;
    
    /* Stop completion thread */
    ctx->completion_running = 0;
    if (ctx->completion_thread) {
        pthread_cond_broadcast(&ctx->ops_cond);
        pthread_join(ctx->completion_thread, NULL);
    }
    
    /* Cleanup network layer */
    remoteio_network_cleanup(ctx);
    
    /* Cleanup RDMA */
    remoteio_rdma_cleanup(ctx);
    
    /* Cleanup listener */
    if (ctx->listener) {
        remoteio_network_stop_listen(ctx->listener);
        ctx->listener = NULL;
    }
    
    /* Cleanup connection pool */
    remoteio_conn_pool_cleanup(&ctx->conn_pool);
    
    /* Cleanup GDR regions */
    pthread_mutex_lock(&ctx->gdr_lock);
    remoteio_gdr_region_t* gdr = ctx->gdr_regions;
    while (gdr) {
        remoteio_gdr_region_t* next = gdr->next;
        remoteio_rdma_unregister_gpu_memory(gdr);
        gdr = next;
    }
    pthread_mutex_unlock(&ctx->gdr_lock);
    
    /* Cleanup pending operations */
    pthread_mutex_lock(&ctx->ops_lock);
    remoteio_operation_t* op = ctx->pending_ops;
    while (op) {
        remoteio_operation_t* next = op->next;
        free(op);
        op = next;
    }
    op = ctx->free_ops;
    while (op) {
        remoteio_operation_t* next = op->next;
        free(op);
        op = next;
    }
    pthread_mutex_unlock(&ctx->ops_lock);
    
    /* Destroy locks */
    pthread_mutex_destroy(&ctx->conn_pool.lock);
    pthread_mutex_destroy(&ctx->gdr_lock);
    pthread_mutex_destroy(&ctx->ops_lock);
    pthread_mutex_destroy(&ctx->stats_lock);
    pthread_cond_destroy(&ctx->ops_cond);
    
    free(ctx);
}

int remoteio_connect(remoteio_context_t* ctx, const char* addr, uint16_t port,
                     remoteio_connection_t** conn_out) {
    if (!ctx || !addr || !conn_out) return -1;
    
    /* Check connection pool first */
    remoteio_connection_t* conn = remoteio_conn_pool_get(&ctx->conn_pool, addr, port);
    if (conn) {
        *conn_out = conn;
        return 0;
    }
    
    /* Create new connection */
    if (remoteio_conn_create(ctx, addr, port, &conn) != 0) {
        return -1;
    }
    
    /* Determine transport */
    remoteio_transport_t transport = ctx->preferred_transport;
    if (transport == REMOTEIO_TRANSPORT_AUTO) {
        /* Try RDMA first, fallback to TCP */
        if (ctx->use_gdr && ctx->rdma_ctx) {
            transport = REMOTEIO_TRANSPORT_RDMA;
        } else {
            transport = REMOTEIO_TRANSPORT_TCP;
        }
    }
    
    /* Connect based on transport */
    int ret;
    if (transport == REMOTEIO_TRANSPORT_RDMA && ctx->use_gdr) {
        ret = remoteio_rdma_connect(ctx, conn, addr, port);
        if (ret == 0) {
            conn->transport = REMOTEIO_TRANSPORT_RDMA;
        } else {
            /* Fallback to TCP */
            ret = remoteio_network_connect(ctx, conn, addr, port);
            if (ret == 0) {
                conn->transport = REMOTEIO_TRANSPORT_TCP;
                ctx->tcp_fallbacks++;
            }
        }
    } else {
        ret = remoteio_network_connect(ctx, conn, addr, port);
        if (ret == 0) {
            conn->transport = REMOTEIO_TRANSPORT_TCP;
        }
    }
    
    if (ret != 0) {
        remoteio_conn_destroy(conn);
        return -1;
    }
    
    *conn_out = conn;
    return 0;
}

int remoteio_disconnect(remoteio_context_t* ctx, remoteio_connection_t* conn) {
    if (!ctx || !conn) return -1;
    
    /* Return to pool or destroy */
    if (remoteio_conn_pool_put(&ctx->conn_pool, conn) != 0) {
        remoteio_conn_destroy(conn);
    }
    
    return 0;
}

int remoteio_read(remoteio_context_t* ctx, const char* uri, void* buf,
                  size_t count, uint64_t offset) {
    if (!ctx || !uri || !buf || count == 0) return -1;
    
    /* Parse URI: rdma://host:port/resource or tcp://host:port/resource */
    char scheme[16] = {0};
    char host[256] = {0};
    int port = REMOTEIO_DEFAULT_PORT;
    char resource[256] = {0};
    
    if (sscanf(uri, "%15[^:]://%255[^:]:%d/%255s", scheme, host, &port, resource) < 2) {
        if (sscanf(uri, "%15[^:]://%255[^/]/%255s", scheme, host, resource) < 2) {
            return -1;
        }
    }
    
    /* Connect to remote */
    remoteio_connection_t* conn;
    if (remoteio_connect(ctx, host, port, &conn) != 0) {
        return -1;
    }
    
    /* Create read operation */
    remoteio_operation_t* op = remoteio_op_alloc(ctx);
    if (!op) {
        remoteio_disconnect(ctx, conn);
        return -1;
    }
    
    op->op = REMOTEIO_OP_READ;
    op->conn = conn;
    op->local_buf = buf;
    op->local_offset = 0;
    op->remote_offset = offset;
    op->length = count;
    
    /* Submit operation */
    int ret = remoteio_op_submit(ctx, op);
    if (ret == 0) {
        ret = remoteio_op_wait(op, REMOTEIO_OP_TIMEOUT_US);
    }
    
    /* Update statistics */
    if (ret == 0) {
        pthread_mutex_lock(&ctx->stats_lock);
        ctx->bytes_read += op->bytes_transferred;
        ctx->requests_completed++;
        pthread_mutex_unlock(&ctx->stats_lock);
    }
    
    remoteio_op_free(ctx, op);
    remoteio_disconnect(ctx, conn);
    
    return ret;
}

int remoteio_write(remoteio_context_t* ctx, const char* uri, const void* buf,
                   size_t count, uint64_t offset) {
    if (!ctx || !uri || !buf || count == 0) return -1;
    
    /* Parse URI */
    char scheme[16] = {0};
    char host[256] = {0};
    int port = REMOTEIO_DEFAULT_PORT;
    char resource[256] = {0};
    
    if (sscanf(uri, "%15[^:]://%255[^:]:%d/%255s", scheme, host, &port, resource) < 2) {
        if (sscanf(uri, "%15[^:]://%255[^/]/%255s", scheme, host, resource) < 2) {
            return -1;
        }
    }
    
    /* Connect to remote */
    remoteio_connection_t* conn;
    if (remoteio_connect(ctx, host, port, &conn) != 0) {
        return -1;
    }
    
    /* Create write operation */
    remoteio_operation_t* op = remoteio_op_alloc(ctx);
    if (!op) {
        remoteio_disconnect(ctx, conn);
        return -1;
    }
    
    op->op = REMOTEIO_OP_WRITE;
    op->conn = conn;
    op->local_buf = (void*)buf;
    op->local_offset = 0;
    op->remote_offset = offset;
    op->length = count;
    
    /* Submit operation */
    int ret = remoteio_op_submit(ctx, op);
    if (ret == 0) {
        ret = remoteio_op_wait(op, REMOTEIO_OP_TIMEOUT_US);
    }
    
    /* Update statistics */
    if (ret == 0) {
        pthread_mutex_lock(&ctx->stats_lock);
        ctx->bytes_written += op->bytes_transferred;
        ctx->requests_completed++;
        pthread_mutex_unlock(&ctx->stats_lock);
    }
    
    remoteio_op_free(ctx, op);
    remoteio_disconnect(ctx, conn);
    
    return ret;
}

int remoteio_read_gpu(remoteio_context_t* ctx, const char* uri, void* gpu_buf,
                      size_t count, uint64_t offset) {
    if (!ctx || !uri || !gpu_buf || count == 0) return -1;
    
    if (!ctx->use_gdr) {
        /* Fallback: read to staging buffer then copy to GPU */
        void* staging = malloc(count);
        if (!staging) return -1;
        
        int ret = remoteio_read(ctx, uri, staging, count, offset);
        if (ret == 0) {
            gpuio_error_t err = gpuio_memcpy(ctx->parent, gpu_buf, staging, count, NULL);
            ret = (err == GPUIO_SUCCESS) ? 0 : -1;
        }
        
        free(staging);
        return ret;
    }
    
    /* Parse URI */
    char scheme[16] = {0};
    char host[256] = {0};
    int port = REMOTEIO_DEFAULT_PORT;
    
    if (sscanf(uri, "%15[^:]://%255[^:]:%d/", scheme, host, &port) < 2) {
        sscanf(uri, "%15[^:]://%255[^/]/", scheme, host);
    }
    
    /* Connect to remote */
    remoteio_connection_t* conn;
    if (remoteio_connect(ctx, host, port, &conn) != 0) {
        return -1;
    }
    
    /* Register GPU memory for RDMA if not already registered */
    pthread_mutex_lock(&ctx->gdr_lock);
    remoteio_gdr_region_t* gdr = ctx->gdr_regions;
    while (gdr) {
        if (gdr->gpu_ptr == gpu_buf) break;
        gdr = gdr->next;
    }
    
    if (!gdr) {
        /* Register new region */
        int gpu_id = 0; /* TODO: detect from address */
        if (remoteio_rdma_register_gpu_memory(ctx, gpu_buf, count, gpu_id, &gdr) == 0) {
            gdr->next = ctx->gdr_regions;
            ctx->gdr_regions = gdr;
        }
    }
    pthread_mutex_unlock(&ctx->gdr_lock);
    
    if (!gdr) {
        remoteio_disconnect(ctx, conn);
        return -1;
    }
    
    /* Create RDMA read operation */
    remoteio_operation_t* op = remoteio_op_alloc(ctx);
    if (!op) {
        remoteio_disconnect(ctx, conn);
        return -1;
    }
    
    op->op = REMOTEIO_OP_READ;
    op->conn = conn;
    op->local_gdr = gdr;
    op->local_offset = 0;
    op->remote_offset = offset;
    op->length = count;
    
    /* Submit RDMA operation */
    int ret = remoteio_op_submit(ctx, op);
    if (ret == 0) {
        ret = remoteio_op_wait(op, REMOTEIO_OP_TIMEOUT_US);
    }
    
    /* Update statistics */
    if (ret == 0) {
        pthread_mutex_lock(&ctx->stats_lock);
        ctx->bytes_read += op->bytes_transferred;
        ctx->rdma_ops++;
        ctx->requests_completed++;
        pthread_mutex_unlock(&ctx->stats_lock);
    }
    
    remoteio_op_free(ctx, op);
    remoteio_disconnect(ctx, conn);
    
    return ret;
}

int remoteio_write_gpu(remoteio_context_t* ctx, const char* uri,
                       const void* gpu_buf, size_t count, uint64_t offset) {
    if (!ctx || !uri || !gpu_buf || count == 0) return -1;
    
    if (!ctx->use_gdr) {
        /* Fallback: copy from GPU to staging buffer then write */
        void* staging = malloc(count);
        if (!staging) return -1;
        
        gpuio_error_t err = gpuio_memcpy(ctx->parent, staging, gpu_buf, count, NULL);
        if (err != GPUIO_SUCCESS) {
            free(staging);
            return -1;
        }
        
        int ret = remoteio_write(ctx, uri, staging, count, offset);
        free(staging);
        return ret;
    }
    
    /* Parse URI */
    char scheme[16] = {0};
    char host[256] = {0};
    int port = REMOTEIO_DEFAULT_PORT;
    
    if (sscanf(uri, "%15[^:]://%255[^:]:%d/", scheme, host, &port) < 2) {
        sscanf(uri, "%15[^:]://%255[^/]/", scheme, host);
    }
    
    /* Connect to remote */
    remoteio_connection_t* conn;
    if (remoteio_connect(ctx, host, port, &conn) != 0) {
        return -1;
    }
    
    /* Register GPU memory for RDMA if not already registered */
    pthread_mutex_lock(&ctx->gdr_lock);
    remoteio_gdr_region_t* gdr = ctx->gdr_regions;
    while (gdr) {
        if (gdr->gpu_ptr == gpu_buf) break;
        gdr = gdr->next;
    }
    
    if (!gdr) {
        /* Register new region */
        int gpu_id = 0; /* TODO: detect from address */
        if (remoteio_rdma_register_gpu_memory(ctx, (void*)gpu_buf, count, gpu_id, &gdr) == 0) {
            gdr->next = ctx->gdr_regions;
            ctx->gdr_regions = gdr;
        }
    }
    pthread_mutex_unlock(&ctx->gdr_lock);
    
    if (!gdr) {
        remoteio_disconnect(ctx, conn);
        return -1;
    }
    
    /* Create RDMA write operation */
    remoteio_operation_t* op = remoteio_op_alloc(ctx);
    if (!op) {
        remoteio_disconnect(ctx, conn);
        return -1;
    }
    
    op->op = REMOTEIO_OP_WRITE;
    op->conn = conn;
    op->local_gdr = gdr;
    op->local_offset = 0;
    op->remote_offset = offset;
    op->length = count;
    
    /* Submit RDMA operation */
    int ret = remoteio_op_submit(ctx, op);
    if (ret == 0) {
        ret = remoteio_op_wait(op, REMOTEIO_OP_TIMEOUT_US);
    }
    
    /* Update statistics */
    if (ret == 0) {
        pthread_mutex_lock(&ctx->stats_lock);
        ctx->bytes_written += op->bytes_transferred;
        ctx->rdma_ops++;
        ctx->requests_completed++;
        pthread_mutex_unlock(&ctx->stats_lock);
    }
    
    remoteio_op_free(ctx, op);
    remoteio_disconnect(ctx, conn);
    
    return ret;
}

int remoteio_register_gpu_memory(remoteio_context_t* ctx, void* gpu_ptr,
                                  size_t length, int gpu_id) {
    if (!ctx || !gpu_ptr || length == 0) return -1;
    
    if (!ctx->use_gdr) return -1;
    
    pthread_mutex_lock(&ctx->gdr_lock);
    
    /* Check if already registered */
    remoteio_gdr_region_t* gdr = ctx->gdr_regions;
    while (gdr) {
        if (gdr->gpu_ptr == gpu_ptr) {
            pthread_mutex_unlock(&ctx->gdr_lock);
            return 0; /* Already registered */
        }
        gdr = gdr->next;
    }
    
    /* Register new region */
    int ret = remoteio_rdma_register_gpu_memory(ctx, gpu_ptr, length, gpu_id, &gdr);
    if (ret == 0) {
        gdr->next = ctx->gdr_regions;
        ctx->gdr_regions = gdr;
    }
    
    pthread_mutex_unlock(&ctx->gdr_lock);
    return ret;
}

int remoteio_unregister_gpu_memory(remoteio_context_t* ctx, void* gpu_ptr) {
    if (!ctx || !gpu_ptr) return -1;
    
    pthread_mutex_lock(&ctx->gdr_lock);
    
    remoteio_gdr_region_t** current = &ctx->gdr_regions;
    while (*current) {
        remoteio_gdr_region_t* gdr = *current;
        if (gdr->gpu_ptr == gpu_ptr) {
            *current = gdr->next;
            remoteio_rdma_unregister_gpu_memory(gdr);
            pthread_mutex_unlock(&ctx->gdr_lock);
            return 0;
        }
        current = &gdr->next;
    }
    
    pthread_mutex_unlock(&ctx->gdr_lock);
    return -1; /* Not found */
}

int remoteio_get_stats(remoteio_context_t* ctx, uint64_t* bytes_read,
                       uint64_t* bytes_written, uint64_t* requests) {
    if (!ctx) return -1;
    
    pthread_mutex_lock(&ctx->stats_lock);
    if (bytes_read) *bytes_read = ctx->bytes_read;
    if (bytes_written) *bytes_written = ctx->bytes_written;
    if (requests) *requests = ctx->requests_completed;
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return 0;
}

const char* remoteio_op_str(remoteio_op_t op) {
    switch (op) {
        case REMOTEIO_OP_READ: return "READ";
        case REMOTEIO_OP_WRITE: return "WRITE";
        case REMOTEIO_OP_SEND: return "SEND";
        case REMOTEIO_OP_RECV: return "RECV";
        case REMOTEIO_OP_ATOMIC_CAS: return "ATOMIC_CAS";
        case REMOTEIO_OP_ATOMIC_FAA: return "ATOMIC_FAA";
        default: return "UNKNOWN";
    }
}

const char* remoteio_conn_state_str(remoteio_conn_state_t state) {
    switch (state) {
        case REMOTEIO_CONN_DISCONNECTED: return "DISCONNECTED";
        case REMOTEIO_CONN_CONNECTING: return "CONNECTING";
        case REMOTEIO_CONN_CONNECTED: return "CONNECTED";
        case REMOTEIO_CONN_ERROR: return "ERROR";
        case REMOTEIO_CONN_CLOSING: return "CLOSING";
        default: return "UNKNOWN";
    }
}

const char* remoteio_transport_str(remoteio_transport_t transport) {
    switch (transport) {
        case REMOTEIO_TRANSPORT_RDMA: return "RDMA";
        case REMOTEIO_TRANSPORT_TCP: return "TCP";
        case REMOTEIO_TRANSPORT_AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}
