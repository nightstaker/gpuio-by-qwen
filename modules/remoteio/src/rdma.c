/**
 * @file rdma.c
 * @brief RemoteIO module - RDMA transport implementation
 * @version 1.0.0
 */

#include "remoteio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

/* RDMA configuration */
#define RDMA_CQ_SIZE          1024
#define RDMA_MAX_SEND_WR      128
#define RDMA_MAX_RECV_WR      128
#define RDMA_MAX_SGE          4
#define RDMA_INLINE_THRESHOLD 256

/* Internal RDMA context */
typedef struct remoteio_rdma_ctx {
    struct rdma_event_channel* cm_channel;
    struct ibv_context* ib_ctx;
    struct ibv_pd* pd;
    int num_devices;
    struct ibv_device** device_list;
    pthread_mutex_t lock;
} remoteio_rdma_ctx_t;

/* Convert ibverbs error to gpuio error */
static gpuio_error_t rdma_error_to_gpuio(int err) {
    switch (err) {
        case 0: return GPUIO_SUCCESS;
        case ENOMEM: return GPUIO_ERROR_NOMEM;
        case EINVAL: return GPUIO_ERROR_INVALID_ARG;
        case ETIMEDOUT: return GPUIO_ERROR_TIMEOUT;
        case EIO: return GPUIO_ERROR_IO;
        case EBUSY: return GPUIO_ERROR_BUSY;
        case ECONNREFUSED: return GPUIO_ERROR_NETWORK;
        default: return GPUIO_ERROR_GENERAL;
    }
}

int remoteio_rdma_init(remoteio_context_t* ctx) {
    if (!ctx) return -1;
    
    remoteio_rdma_ctx_t* rdma_ctx = calloc(1, sizeof(remoteio_rdma_ctx_t));
    if (!rdma_ctx) return -1;
    
    pthread_mutex_init(&rdma_ctx->lock, NULL);
    
    /* Get list of RDMA devices */
    rdma_ctx->device_list = rdma_get_devices(&rdma_ctx->num_devices);
    if (!rdma_ctx->device_list || rdma_ctx->num_devices == 0) {
        free(rdma_ctx);
        return -1;
    }
    
    /* Open first device that supports verbs */
    rdma_ctx->ib_ctx = NULL;
    for (int i = 0; i < rdma_ctx->num_devices; i++) {
        struct ibv_context* test_ctx = ibv_open_device(rdma_ctx->device_list[i]);
        if (test_ctx) {
            rdma_ctx->ib_ctx = test_ctx;
            break;
        }
    }
    
    if (!rdma_ctx->ib_ctx) {
        rdma_free_devices(rdma_ctx->device_list);
        free(rdma_ctx);
        return -1;
    }
    
    /* Create protection domain */
    rdma_ctx->pd = ibv_alloc_pd(rdma_ctx->ib_ctx);
    if (!rdma_ctx->pd) {
        ibv_close_device(rdma_ctx->ib_ctx);
        rdma_free_devices(rdma_ctx->device_list);
        free(rdma_ctx);
        return -1;
    }
    
    /* Create event channel */
    rdma_ctx->cm_channel = rdma_create_event_channel();
    if (!rdma_ctx->cm_channel) {
        ibv_dealloc_pd(rdma_ctx->pd);
        ibv_close_device(rdma_ctx->ib_ctx);
        rdma_free_devices(rdma_ctx->device_list);
        free(rdma_ctx);
        return -1;
    }
    
    ctx->rdma_ctx = rdma_ctx;
    return 0;
}

void remoteio_rdma_cleanup(remoteio_context_t* ctx) {
    if (!ctx || !ctx->rdma_ctx) return;
    
    remoteio_rdma_ctx_t* rdma_ctx = (remoteio_rdma_ctx_t*)ctx->rdma_ctx;
    
    pthread_mutex_lock(&rdma_ctx->lock);
    
    if (rdma_ctx->cm_channel) {
        rdma_destroy_event_channel(rdma_ctx->cm_channel);
    }
    
    if (rdma_ctx->pd) {
        ibv_dealloc_pd(rdma_ctx->pd);
    }
    
    if (rdma_ctx->ib_ctx) {
        ibv_close_device(rdma_ctx->ib_ctx);
    }
    
    if (rdma_ctx->device_list) {
        rdma_free_devices(rdma_ctx->device_list);
    }
    
    pthread_mutex_unlock(&rdma_ctx->lock);
    pthread_mutex_destroy(&rdma_ctx->lock);
    
    free(rdma_ctx);
    ctx->rdma_ctx = NULL;
}

int remoteio_rdma_endpoint_create(remoteio_context_t* ctx,
                                   remoteio_rdma_endpoint_t** ep_out) {
    if (!ctx || !ctx->rdma_ctx || !ep_out) return -1;
    
    remoteio_rdma_ctx_t* rdma_ctx = (remoteio_rdma_ctx_t*)ctx->rdma_ctx;
    
    remoteio_rdma_endpoint_t* ep = calloc(1, sizeof(remoteio_rdma_endpoint_t));
    if (!ep) return -1;
    
    pthread_mutex_init(&ep->lock, NULL);
    
    /* Create completion queue */
    struct ibv_cq* cq = ibv_create_cq(rdma_ctx->ib_ctx, RDMA_CQ_SIZE, NULL, NULL, 0);
    if (!cq) {
        free(ep);
        return -1;
    }
    ep->cq = cq;
    ep->pd = rdma_ctx->pd;
    ep->num_sge = RDMA_MAX_SGE;
    ep->max_inline = RDMA_INLINE_THRESHOLD;
    
    *ep_out = ep;
    return 0;
}

void remoteio_rdma_endpoint_destroy(remoteio_rdma_endpoint_t* ep) {
    if (!ep) return;
    
    pthread_mutex_lock(&ep->lock);
    
    if (ep->qp) {
        struct ibv_qp* qp = (struct ibv_qp*)ep->qp;
        ibv_destroy_qp(qp);
    }
    
    if (ep->cq) {
        struct ibv_cq* cq = (struct ibv_cq*)ep->cq;
        ibv_destroy_cq(cq);
    }
    
    if (ep->cm_id) {
        struct rdma_cm_id* cm_id = (struct rdma_cm_id*)ep->cm_id;
        rdma_destroy_id(cm_id);
    }
    
    pthread_mutex_unlock(&ep->lock);
    pthread_mutex_destroy(&ep->lock);
    
    free(ep);
}

static int rdma_create_qp(remoteio_rdma_endpoint_t* ep) {
    if (!ep || !ep->pd) return -1;
    
    struct ibv_qp_init_attr qp_attr = {
        .qp_type = IBV_QPT_RC,
        .sq_sig_all = 0,
        .send_cq = (struct ibv_cq*)ep->cq,
        .recv_cq = (struct ibv_cq*)ep->cq,
        .cap = {
            .max_send_wr = RDMA_MAX_SEND_WR,
            .max_recv_wr = RDMA_MAX_RECV_WR,
            .max_send_sge = RDMA_MAX_SGE,
            .max_recv_sge = RDMA_MAX_SGE,
            .max_inline_data = RDMA_INLINE_THRESHOLD,
        }
    };
    
    struct rdma_cm_id* cm_id = (struct rdma_cm_id*)ep->cm_id;
    if (rdma_create_qp(cm_id, (struct ibv_pd*)ep->pd, &qp_attr)) {
        return -1;
    }
    
    ep->qp = cm_id->qp;
    return 0;
}

int remoteio_rdma_connect(remoteio_context_t* ctx, remoteio_connection_t* conn,
                          const char* addr, uint16_t port) {
    if (!ctx || !ctx->rdma_ctx || !conn) return -1;
    
    remoteio_rdma_ctx_t* rdma_ctx = (remoteio_rdma_ctx_t*)ctx->rdma_ctx;
    
    /* Create endpoint */
    remoteio_rdma_endpoint_t* ep;
    if (remoteio_rdma_endpoint_create(ctx, &ep) != 0) {
        return -1;
    }
    
    /* Create RDMA CM ID */
    struct rdma_cm_id* cm_id;
    if (rdma_create_id(rdma_ctx->cm_channel, &cm_id, NULL, RDMA_PS_TCP)) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    ep->cm_id = cm_id;
    
    /* Resolve address */
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);
    inet_pton(AF_INET, addr, &sin.sin_addr);
    
    if (rdma_resolve_addr(cm_id, NULL, (struct sockaddr*)&sin, 2000)) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    
    /* Wait for address resolution */
    struct rdma_cm_event* event;
    if (rdma_get_cm_event(rdma_ctx->cm_channel, &event)) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
        rdma_ack_cm_event(event);
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    rdma_ack_cm_event(event);
    
    /* Resolve route */
    if (rdma_resolve_route(cm_id, 2000)) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    
    /* Wait for route resolution */
    if (rdma_get_cm_event(rdma_ctx->cm_channel, &event)) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED) {
        rdma_ack_cm_event(event);
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    rdma_ack_cm_event(event);
    
    /* Create QP */
    if (rdma_create_qp(ep) != 0) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    
    /* Connect */
    struct rdma_conn_param conn_param = {
        .initiator_depth = 1,
        .responder_resources = 1,
        .retry_count = 7,
        .rnr_retry_count = 7,
    };
    
    if (rdma_connect(cm_id, &conn_param)) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    
    /* Wait for connection */
    if (rdma_get_cm_event(rdma_ctx->cm_channel, &event)) {
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    if (event->event != RDMA_CM_EVENT_ESTABLISHED) {
        rdma_ack_cm_event(event);
        remoteio_rdma_endpoint_destroy(ep);
        return -1;
    }
    rdma_ack_cm_event(event);
    
    /* Update connection */
    conn->rdma_ep = ep;
    conn->state = REMOTEIO_CONN_CONNECTED;
    conn->max_send_wr = RDMA_MAX_SEND_WR;
    conn->max_recv_wr = RDMA_MAX_RECV_WR;
    conn->max_sge = RDMA_MAX_SGE;
    
    return 0;
}

int remoteio_rdma_disconnect(remoteio_connection_t* conn) {
    if (!conn) return -1;
    
    pthread_mutex_lock(&conn->lock);
    
    if (conn->state != REMOTEIO_CONN_CONNECTED) {
        pthread_mutex_unlock(&conn->lock);
        return 0;
    }
    
    conn->state = REMOTEIO_CONN_CLOSING;
    
    if (conn->rdma_ep) {
        struct rdma_cm_id* cm_id = (struct rdma_cm_id*)conn->rdma_ep->cm_id;
        if (cm_id) {
            rdma_disconnect(cm_id);
        }
        remoteio_rdma_endpoint_destroy(conn->rdma_ep);
        conn->rdma_ep = NULL;
    }
    
    conn->state = REMOTEIO_CONN_DISCONNECTED;
    pthread_cond_broadcast(&conn->state_cond);
    pthread_mutex_unlock(&conn->lock);
    
    return 0;
}

int remoteio_rdma_post_read(remoteio_connection_t* conn,
                            remoteio_gdr_region_t* local_mr,
                            remoteio_remote_mem_t* remote,
                            uint64_t local_offset, uint64_t remote_offset,
                            size_t len, remoteio_operation_t* op) {
    if (!conn || !local_mr || !remote || !op) return -1;
    if (conn->state != REMOTEIO_CONN_CONNECTED) return -1;
    
    remoteio_rdma_endpoint_t* ep = conn->rdma_ep;
    if (!ep || !ep->qp) return -1;
    
    /* Setup SGE */
    struct ibv_sge sge = {
        .addr = (uint64_t)local_mr->gpu_ptr + local_offset,
        .length = (uint32_t)len,
        .lkey = local_mr->lkey,
    };
    
    /* Setup send work request */
    struct ibv_send_wr wr = {
        .wr_id = (uint64_t)op,
        .opcode = IBV_WR_RDMA_READ,
        .send_flags = IBV_SEND_SIGNALED,
        .num_sge = 1,
        .sg_list = &sge,
        .wr.rdma = {
            .remote_addr = remote->raddr + remote_offset,
            .rkey = remote->rkey,
        },
    };
    
    /* Post send */
    struct ibv_send_wr* bad_wr;
    if (ibv_post_send((struct ibv_qp*)ep->qp, &wr, &bad_wr)) {
        return -1;
    }
    
    pthread_mutex_lock(&conn->lock);
    conn->reqs_submitted++;
    pthread_mutex_unlock(&conn->lock);
    
    return 0;
}

int remoteio_rdma_post_write(remoteio_connection_t* conn,
                             remoteio_gdr_region_t* local_mr,
                             remoteio_remote_mem_t* remote,
                             uint64_t local_offset, uint64_t remote_offset,
                             size_t len, remoteio_operation_t* op) {
    if (!conn || !local_mr || !remote || !op) return -1;
    if (conn->state != REMOTEIO_CONN_CONNECTED) return -1;
    
    remoteio_rdma_endpoint_t* ep = conn->rdma_ep;
    if (!ep || !ep->qp) return -1;
    
    /* Check for inline */
    int send_flags = IBV_SEND_SIGNALED;
    if (len <= RDMA_INLINE_THRESHOLD) {
        send_flags |= IBV_SEND_INLINE;
    }
    
    /* Setup SGE */
    struct ibv_sge sge = {
        .addr = (uint64_t)local_mr->gpu_ptr + local_offset,
        .length = (uint32_t)len,
        .lkey = local_mr->lkey,
    };
    
    /* Setup send work request */
    struct ibv_send_wr wr = {
        .wr_id = (uint64_t)op,
        .opcode = IBV_WR_RDMA_WRITE,
        .send_flags = send_flags,
        .num_sge = 1,
        .sg_list = &sge,
        .wr.rdma = {
            .remote_addr = remote->raddr + remote_offset,
            .rkey = remote->rkey,
        },
    };
    
    /* Post send */
    struct ibv_send_wr* bad_wr;
    if (ibv_post_send((struct ibv_qp*)ep->qp, &wr, &bad_wr)) {
        return -1;
    }
    
    pthread_mutex_lock(&conn->lock);
    conn->reqs_submitted++;
    pthread_mutex_unlock(&conn->lock);
    
    return 0;
}

int remoteio_rdma_poll_completions(remoteio_connection_t* conn, int max_poll) {
    if (!conn || !conn->rdma_ep) return -1;
    
    remoteio_rdma_endpoint_t* ep = conn->rdma_ep;
    if (!ep->cq) return -1;
    
    struct ibv_wc wc[16];
    int polled = 0;
    
    while (polled < max_poll) {
        int n = ibv_poll_cq((struct ibv_cq*)ep->cq, 16, wc);
        if (n <= 0) break;
        
        for (int i = 0; i < n; i++) {
            remoteio_operation_t* op = (remoteio_operation_t*)wc[i].wr_id;
            if (!op) continue;
            
            if (wc[i].status == IBV_WC_SUCCESS) {
                op->status = GPUIO_SUCCESS;
                op->bytes_transferred = wc[i].byte_len;
            } else {
                op->status = GPUIO_ERROR_IO;
            }
            op->completed = 1;
            
            /* Invoke callback if provided */
            if (op->callback) {
                op->callback((gpuio_request_t)op, op->status, op->user_data);
            }
            
            pthread_mutex_lock(&conn->lock);
            conn->reqs_completed++;
            pthread_mutex_unlock(&conn->lock);
        }
        
        polled += n;
    }
    
    return polled;
}

int remoteio_rdma_register_gpu_memory(remoteio_context_t* ctx,
                                       void* gpu_ptr, size_t length,
                                       int gpu_id,
                                       remoteio_gdr_region_t** region_out) {
    if (!ctx || !ctx->rdma_ctx || !gpu_ptr || length == 0 || !region_out) {
        return -1;
    }
    
    remoteio_rdma_ctx_t* rdma_ctx = (remoteio_rdma_ctx_t*)ctx->rdma_ctx;
    
    /* Allocate GDR region */
    remoteio_gdr_region_t* region = calloc(1, sizeof(remoteio_gdr_region_t));
    if (!region) return -1;
    
    region->gpu_ptr = gpu_ptr;
    region->length = length;
    region->gpu_id = gpu_id;
    
    /* Register memory with ibverbs */
    struct ibv_mr* mr = ibv_reg_mr(rdma_ctx->pd, gpu_ptr, length,
                                   IBV_ACCESS_LOCAL_WRITE |
                                   IBV_ACCESS_REMOTE_READ |
                                   IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
        free(region);
        return -1;
    }
    
    region->mr = mr;
    region->lkey = mr->lkey;
    region->rkey = mr->rkey;
    region->gpu_phys = (uint64_t)mr->addr; /* Use as bus address */
    
    *region_out = region;
    return 0;
}

int remoteio_rdma_unregister_gpu_memory(remoteio_gdr_region_t* region) {
    if (!region) return -1;
    
    if (region->mr) {
        ibv_dereg_mr((struct ibv_mr*)region->mr);
    }
    
    free(region);
    return 0;
}
