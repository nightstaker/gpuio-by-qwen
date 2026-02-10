/**
 * @file remoteio_internal.h
 * @brief RemoteIO module internal definitions
 * @version 1.0.0
 */

#ifndef REMOTEIO_INTERNAL_H
#define REMOTEIO_INTERNAL_H

#include <gpuio/gpuio.h>
#include <pthread.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* RemoteIO module exports */
#define REMOTEIO_API __attribute__((visibility("default")))

/* RDMA operation types */
typedef enum {
    REMOTEIO_OP_READ = 0,
    REMOTEIO_OP_WRITE = 1,
    REMOTEIO_OP_SEND = 2,
    REMOTEIO_OP_RECV = 3,
    REMOTEIO_OP_ATOMIC_CAS = 4,
    REMOTEIO_OP_ATOMIC_FAA = 5,
} remoteio_op_t;

/* Connection state */
typedef enum {
    REMOTEIO_CONN_DISCONNECTED = 0,
    REMOTEIO_CONN_CONNECTING = 1,
    REMOTEIO_CONN_CONNECTED = 2,
    REMOTEIO_CONN_ERROR = 3,
    REMOTEIO_CONN_CLOSING = 4,
} remoteio_conn_state_t;

/* Transport type */
typedef enum {
    REMOTEIO_TRANSPORT_RDMA = 0,
    REMOTEIO_TRANSPORT_TCP = 1,
    REMOTEIO_TRANSPORT_AUTO = 2,
} remoteio_transport_t;

/* RDMA endpoint (opaque handle for ibverbs/rdmacm) */
typedef struct remoteio_rdma_endpoint {
    void* pd;                    /* Protection domain */
    void* cq;                    /* Completion queue */
    void* qp;                    /* Queue pair */
    void* cm_id;                 /* RDMA CM ID */
    void* mr_list;               /* Registered memory regions list */
    pthread_mutex_t lock;
    int num_sge;                 /* Max scatter-gather entries */
    uint32_t max_inline;         /* Max inline data */
} remoteio_rdma_endpoint_t;

/* GPUDirect RDMA memory registration */
typedef struct remoteio_gdr_region {
    void* gpu_ptr;               /* GPU virtual address */
    uint64_t gpu_phys;           /* GPU physical/bus address */
    size_t length;
    void* mr;                    /* RDMA memory region handle */
    int gpu_id;                  /* GPU device ID */
    uint32_t lkey;               /* Local key */
    uint32_t rkey;               /* Remote key */
    struct remoteio_gdr_region* next;
} remoteio_gdr_region_t;

/* Remote memory handle (for remote access) */
typedef struct remoteio_remote_mem {
    uint64_t raddr;              /* Remote virtual address */
    uint32_t rkey;               /* Remote access key */
    size_t length;
    char peer_addr[INET6_ADDRSTRLEN];
    uint16_t peer_port;
} remoteio_remote_mem_t;

/* Network connection */
typedef struct remoteio_connection {
    char peer_addr[INET6_ADDRSTRLEN];
    uint16_t peer_port;
    uint16_t local_port;
    remoteio_conn_state_t state;
    remoteio_transport_t transport;
    
    /* Socket for TCP fallback */
    int socket_fd;
    
    /* RDMA endpoint (if using RDMA) */
    remoteio_rdma_endpoint_t* rdma_ep;
    
    /* Connection attributes */
    uint32_t max_send_wr;
    uint32_t max_recv_wr;
    uint32_t max_sge;
    
    /* Statistics */
    uint64_t bytes_sent;
    uint64_t bytes_received;
    uint64_t reqs_submitted;
    uint64_t reqs_completed;
    uint64_t reqs_failed;
    
    /* Thread safety */
    pthread_mutex_t lock;
    pthread_cond_t state_cond;
    
    /* Reference counting */
    int ref_count;
    
    struct remoteio_connection* next;
} remoteio_connection_t;

/* Pending operation */
typedef struct remoteio_operation {
    uint64_t id;
    remoteio_op_t op;
    remoteio_connection_t* conn;
    
    /* Memory regions */
    union {
        remoteio_gdr_region_t* local_gdr;
        void* local_buf;
    };
    remoteio_remote_mem_t* remote_mem;
    
    /* Parameters */
    uint64_t local_offset;
    uint64_t remote_offset;
    size_t length;
    
    /* Completion */
    volatile int completed;
    gpuio_error_t status;
    size_t bytes_transferred;
    gpuio_callback_t callback;
    void* user_data;
    
    /* For scatter-gather */
    struct remoteio_operation* next_sg;
    int sg_count;
    
    struct remoteio_operation* next;
} remoteio_operation_t;

/* Connection pool */
typedef struct remoteio_conn_pool {
    remoteio_connection_t* connections;
    int num_connections;
    int max_connections;
    pthread_mutex_t lock;
} remoteio_conn_pool_t;

/* Network listener */
typedef struct remoteio_listener {
    int socket_fd;
    int port;
    remoteio_transport_t transport;
    
    /* RDMA listen ID */
    void* rdma_listen_id;
    
    /* Accept callback */
    void (*accept_cb)(remoteio_connection_t* conn, void* user_data);
    void* accept_user_data;
    
    /* Thread */
    pthread_t thread;
    int running;
    
    pthread_mutex_t lock;
} remoteio_listener_t;

/* RemoteIO context */
typedef struct remoteio_context {
    gpuio_context_t parent;
    
    /* Configuration */
    remoteio_transport_t preferred_transport;
    int use_gdr;                 /* GPUDirect RDMA */
    int use_inline;              /* Inline small messages */
    size_t inline_threshold;
    
    /* Connection management */
    remoteio_conn_pool_t conn_pool;
    remoteio_listener_t* listener;
    
    /* GPUDirect RDMA regions */
    remoteio_gdr_region_t* gdr_regions;
    pthread_mutex_t gdr_lock;
    
    /* Operations */
    remoteio_operation_t* pending_ops;
    remoteio_operation_t* free_ops;
    uint64_t next_op_id;
    pthread_mutex_t ops_lock;
    pthread_cond_t ops_cond;
    
    /* Worker thread for async completions */
    pthread_t completion_thread;
    int completion_running;
    
    /* Network layer handle */
    void* net_handle;
    
    /* RDMA device context */
    void* rdma_ctx;
    
    /* Statistics */
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t requests_submitted;
    uint64_t requests_completed;
    uint64_t rdma_ops;
    uint64_t tcp_fallbacks;
    
    /* Thread safety */
    pthread_mutex_t stats_lock;
} remoteio_context_t;

/* ============================================================================
 * RDMA Functions
 * ============================================================================ */

int remoteio_rdma_init(remoteio_context_t* ctx);
void remoteio_rdma_cleanup(remoteio_context_t* ctx);

int remoteio_rdma_endpoint_create(remoteio_context_t* ctx,
                                   remoteio_rdma_endpoint_t** ep_out);
void remoteio_rdma_endpoint_destroy(remoteio_rdma_endpoint_t* ep);

int remoteio_rdma_connect(remoteio_context_t* ctx, remoteio_connection_t* conn,
                          const char* addr, uint16_t port);
int remoteio_rdma_accept(remoteio_context_t* ctx, remoteio_connection_t* conn);
int remoteio_rdma_disconnect(remoteio_connection_t* conn);

int remoteio_rdma_post_send(remoteio_connection_t* conn, void* buf, size_t len,
                            remoteio_operation_t* op);
int remoteio_rdma_post_recv(remoteio_connection_t* conn, void* buf, size_t len,
                            remoteio_operation_t* op);
int remoteio_rdma_post_read(remoteio_connection_t* conn,
                            remoteio_gdr_region_t* local_mr,
                            remoteio_remote_mem_t* remote,
                            uint64_t local_offset, uint64_t remote_offset,
                            size_t len, remoteio_operation_t* op);
int remoteio_rdma_post_write(remoteio_connection_t* conn,
                             remoteio_gdr_region_t* local_mr,
                             remoteio_remote_mem_t* remote,
                             uint64_t local_offset, uint64_t remote_offset,
                             size_t len, remoteio_operation_t* op);

int remoteio_rdma_poll_completions(remoteio_connection_t* conn, int max_poll);
int remoteio_rdma_register_gpu_memory(remoteio_context_t* ctx,
                                       void* gpu_ptr, size_t length,
                                       int gpu_id,
                                       remoteio_gdr_region_t** region_out);
int remoteio_rdma_unregister_gpu_memory(remoteio_gdr_region_t* region);

/* ============================================================================
 * Network Functions
 * ============================================================================ */

int remoteio_network_init(remoteio_context_t* ctx);
void remoteio_network_cleanup(remoteio_context_t* ctx);

int remoteio_network_connect(remoteio_context_t* ctx, remoteio_connection_t* conn,
                             const char* addr, uint16_t port);
int remoteio_network_disconnect(remoteio_connection_t* conn);

int remoteio_network_send(remoteio_connection_t* conn, const void* buf, size_t len);
int remoteio_network_recv(remoteio_connection_t* conn, void* buf, size_t len);
int remoteio_network_send_recv(remoteio_connection_t* conn,
                               const void* send_buf, size_t send_len,
                               void* recv_buf, size_t recv_len);

int remoteio_network_listen(remoteio_context_t* ctx, int port,
                            remoteio_listener_t** listener_out);
int remoteio_network_stop_listen(remoteio_listener_t* listener);

/* ============================================================================
 * Connection Management
 * ============================================================================ */

int remoteio_conn_create(remoteio_context_t* ctx, const char* addr, uint16_t port,
                         remoteio_connection_t** conn_out);
int remoteio_conn_destroy(remoteio_connection_t* conn);
int remoteio_conn_acquire(remoteio_connection_t* conn);
int remoteio_conn_release(remoteio_connection_t* conn);

int remoteio_conn_pool_init(remoteio_conn_pool_t* pool, int max_conns);
void remoteio_conn_pool_cleanup(remoteio_conn_pool_t* pool);
remoteio_connection_t* remoteio_conn_pool_get(remoteio_conn_pool_t* pool,
                                               const char* addr, uint16_t port);
int remoteio_conn_pool_put(remoteio_conn_pool_t* pool, remoteio_connection_t* conn);

/* ============================================================================
 * Operation Management
 * ============================================================================ */

remoteio_operation_t* remoteio_op_alloc(remoteio_context_t* ctx);
void remoteio_op_free(remoteio_context_t* ctx, remoteio_operation_t* op);
int remoteio_op_submit(remoteio_context_t* ctx, remoteio_operation_t* op);
int remoteio_op_wait(remoteio_operation_t* op, uint64_t timeout_us);
int remoteio_op_cancel(remoteio_operation_t* op);

/* ============================================================================
 * Utilities
 * ============================================================================ */

const char* remoteio_op_str(remoteio_op_t op);
const char* remoteio_conn_state_str(remoteio_conn_state_t state);
const char* remoteio_transport_str(remoteio_transport_t transport);

/* ============================================================================
 * Internal API
 * ============================================================================ */

remoteio_context_t* remoteio_context_create(gpuio_context_t parent);
void remoteio_context_destroy(remoteio_context_t* ctx);

int remoteio_read(remoteio_context_t* ctx, const char* uri, void* buf,
                  size_t count, uint64_t offset);
int remoteio_write(remoteio_context_t* ctx, const char* uri, const void* buf,
                   size_t count, uint64_t offset);
int remoteio_read_gpu(remoteio_context_t* ctx, const char* uri, void* gpu_buf,
                      size_t count, uint64_t offset);
int remoteio_write_gpu(remoteio_context_t* ctx, const char* uri,
                       const void* gpu_buf, size_t count, uint64_t offset);

int remoteio_get_stats(remoteio_context_t* ctx, uint64_t* bytes_read,
                       uint64_t* bytes_written, uint64_t* requests);

#endif /* REMOTEIO_INTERNAL_H */
