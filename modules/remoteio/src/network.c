/**
 * @file network.c
 * @brief RemoteIO module - Generic network layer
 * @version 1.0.0
 */

#include "remoteio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <poll.h>

/* Network configuration */
#define NETWORK_DEFAULT_TIMEOUT_MS  30000
#define NETWORK_RECV_BUF_SIZE       (256 * 1024)
#define NETWORK_SEND_BUF_SIZE       (256 * 1024)
#define NETWORK_MAX_BACKLOG         128

/* Internal network context */
typedef struct remoteio_net_ctx {
    int initialized;
    pthread_mutex_t lock;
} remoteio_net_ctx_t;

int remoteio_network_init(remoteio_context_t* ctx) {
    if (!ctx) return -1;
    
    remoteio_net_ctx_t* net_ctx = calloc(1, sizeof(remoteio_net_ctx_t));
    if (!net_ctx) return -1;
    
    pthread_mutex_init(&net_ctx->lock, NULL);
    net_ctx->initialized = 1;
    
    ctx->net_handle = net_ctx;
    return 0;
}

void remoteio_network_cleanup(remoteio_context_t* ctx) {
    if (!ctx || !ctx->net_handle) return;
    
    remoteio_net_ctx_t* net_ctx = (remoteio_net_ctx_t*)ctx->net_handle;
    
    pthread_mutex_lock(&net_ctx->lock);
    net_ctx->initialized = 0;
    pthread_mutex_unlock(&net_ctx->lock);
    
    pthread_mutex_destroy(&net_ctx->lock);
    free(net_ctx);
    ctx->net_handle = NULL;
}

static int set_socket_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int set_socket_options(int fd) {
    int yes = 1;
    
    /* Enable TCP_NODELAY */
    if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes)) < 0) {
        return -1;
    }
    
    /* Enable SO_KEEPALIVE */
    if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &yes, sizeof(yes)) < 0) {
        return -1;
    }
    
    /* Set buffer sizes */
    int bufsize = NETWORK_RECV_BUF_SIZE;
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
    bufsize = NETWORK_SEND_BUF_SIZE;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
    
    return 0;
}

int remoteio_network_connect(remoteio_context_t* ctx, remoteio_connection_t* conn,
                             const char* addr, uint16_t port) {
    if (!ctx || !conn || !addr) return -1;
    
    /* Create socket */
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    
    /* Set socket options */
    if (set_socket_options(fd) < 0) {
        close(fd);
        return -1;
    }
    
    /* Set non-blocking for connect with timeout */
    if (set_socket_nonblocking(fd) < 0) {
        close(fd);
        return -1;
    }
    
    /* Resolve address */
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_port = htons(port);
    
    if (inet_pton(AF_INET, addr, &sin.sin_addr) <= 0) {
        /* Try hostname resolution */
        struct hostent* he = gethostbyname(addr);
        if (!he) {
            close(fd);
            return -1;
        }
        memcpy(&sin.sin_addr, he->h_addr_list[0], he->h_length);
    }
    
    /* Connect */
    int ret = connect(fd, (struct sockaddr*)&sin, sizeof(sin));
    if (ret < 0 && errno != EINPROGRESS) {
        close(fd);
        return -1;
    }
    
    /* Wait for connection with timeout */
    struct pollfd pfd = {
        .fd = fd,
        .events = POLLOUT
    };
    
    ret = poll(&pfd, 1, NETWORK_DEFAULT_TIMEOUT_MS);
    if (ret <= 0) {
        close(fd);
        return -1;
    }
    
    /* Check connection result */
    int so_error;
    socklen_t len = sizeof(so_error);
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &len) < 0 || so_error != 0) {
        close(fd);
        return -1;
    }
    
    /* Set back to blocking mode */
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
    
    /* Update connection */
    pthread_mutex_lock(&conn->lock);
    conn->socket_fd = fd;
    conn->state = REMOTEIO_CONN_CONNECTED;
    conn->transport = REMOTEIO_TRANSPORT_TCP;
    strncpy(conn->peer_addr, addr, INET_ADDRSTRLEN - 1);
    conn->peer_port = port;
    pthread_mutex_unlock(&conn->lock);
    
    return 0;
}

int remoteio_network_disconnect(remoteio_connection_t* conn) {
    if (!conn) return -1;
    
    pthread_mutex_lock(&conn->lock);
    
    if (conn->state == REMOTEIO_CONN_DISCONNECTED) {
        pthread_mutex_unlock(&conn->lock);
        return 0;
    }
    
    conn->state = REMOTEIO_CONN_CLOSING;
    
    if (conn->socket_fd >= 0) {
        close(conn->socket_fd);
        conn->socket_fd = -1;
    }
    
    conn->state = REMOTEIO_CONN_DISCONNECTED;
    pthread_cond_broadcast(&conn->state_cond);
    pthread_mutex_unlock(&conn->lock);
    
    return 0;
}

int remoteio_network_send(remoteio_connection_t* conn, const void* buf, size_t len) {
    if (!conn || !buf || len == 0) return -1;
    if (conn->state != REMOTEIO_CONN_CONNECTED) return -1;
    
    size_t total_sent = 0;
    const char* ptr = (const char*)buf;
    
    while (total_sent < len) {
        ssize_t sent = send(conn->socket_fd, ptr + total_sent, len - total_sent, 0);
        if (sent < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = {
                    .fd = conn->socket_fd,
                    .events = POLLOUT
                };
                if (poll(&pfd, 1, NETWORK_DEFAULT_TIMEOUT_MS) <= 0) {
                    return -1;
                }
                continue;
            }
            return -1;
        }
        if (sent == 0) {
            return -1; /* Connection closed */
        }
        total_sent += sent;
    }
    
    pthread_mutex_lock(&conn->lock);
    conn->bytes_sent += total_sent;
    pthread_mutex_unlock(&conn->lock);
    
    return 0;
}

int remoteio_network_recv(remoteio_connection_t* conn, void* buf, size_t len) {
    if (!conn || !buf || len == 0) return -1;
    if (conn->state != REMOTEIO_CONN_CONNECTED) return -1;
    
    size_t total_recv = 0;
    char* ptr = (char*)buf;
    
    while (total_recv < len) {
        ssize_t recvd = recv(conn->socket_fd, ptr + total_recv, len - total_recv, 0);
        if (recvd < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                struct pollfd pfd = {
                    .fd = conn->socket_fd,
                    .events = POLLIN
                };
                if (poll(&pfd, 1, NETWORK_DEFAULT_TIMEOUT_MS) <= 0) {
                    return -1;
                }
                continue;
            }
            return -1;
        }
        if (recvd == 0) {
            return -1; /* Connection closed */
        }
        total_recv += recvd;
    }
    
    pthread_mutex_lock(&conn->lock);
    conn->bytes_received += total_recv;
    pthread_mutex_unlock(&conn->lock);
    
    return 0;
}

int remoteio_network_send_recv(remoteio_connection_t* conn,
                               const void* send_buf, size_t send_len,
                               void* recv_buf, size_t recv_len) {
    if (!conn || !send_buf || !recv_buf) return -1;
    
    /* Send request */
    if (remoteio_network_send(conn, send_buf, send_len) != 0) {
        return -1;
    }
    
    /* Receive response */
    if (remoteio_network_recv(conn, recv_buf, recv_len) != 0) {
        return -1;
    }
    
    return 0;
}

static void* listener_thread(void* arg) {
    remoteio_listener_t* listener = (remoteio_listener_t*)arg;
    
    while (listener->running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        
        int client_fd = accept(listener->socket_fd, 
                               (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            break;
        }
        
        /* Set socket options for client */
        set_socket_options(client_fd);
        
        /* Create connection object */
        remoteio_connection_t* conn = calloc(1, sizeof(remoteio_connection_t));
        if (conn) {
            pthread_mutex_init(&conn->lock, NULL);
            pthread_cond_init(&conn->state_cond, NULL);
            conn->socket_fd = client_fd;
            conn->state = REMOTEIO_CONN_CONNECTED;
            conn->transport = REMOTEIO_TRANSPORT_TCP;
            inet_ntop(AF_INET, &client_addr.sin_addr, conn->peer_addr, 
                     INET_ADDRSTRLEN);
            conn->peer_port = ntohs(client_addr.sin_port);
            conn->ref_count = 1;
            
            if (listener->accept_cb) {
                listener->accept_cb(conn, listener->accept_user_data);
            }
            
            remoteio_conn_release(conn);
        } else {
            close(client_fd);
        }
    }
    
    return NULL;
}

int remoteio_network_listen(remoteio_context_t* ctx, int port,
                            remoteio_listener_t** listener_out) {
    if (!ctx || !listener_out) return -1;
    
    remoteio_listener_t* listener = calloc(1, sizeof(remoteio_listener_t));
    if (!listener) return -1;
    
    pthread_mutex_init(&listener->lock, NULL);
    
    /* Create socket */
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        free(listener);
        return -1;
    }
    
    /* Allow address reuse */
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    
    /* Bind */
    struct sockaddr_in sin;
    memset(&sin, 0, sizeof(sin));
    sin.sin_family = AF_INET;
    sin.sin_addr.s_addr = INADDR_ANY;
    sin.sin_port = htons(port);
    
    if (bind(fd, (struct sockaddr*)&sin, sizeof(sin)) < 0) {
        close(fd);
        pthread_mutex_destroy(&listener->lock);
        free(listener);
        return -1;
    }
    
    /* Listen */
    if (listen(fd, NETWORK_MAX_BACKLOG) < 0) {
        close(fd);
        pthread_mutex_destroy(&listener->lock);
        free(listener);
        return -1;
    }
    
    listener->socket_fd = fd;
    listener->port = port;
    listener->running = 1;
    listener->transport = REMOTEIO_TRANSPORT_TCP;
    
    /* Start listener thread */
    pthread_create(&listener->thread, NULL, listener_thread, listener);
    
    *listener_out = listener;
    return 0;
}

int remoteio_network_stop_listen(remoteio_listener_t* listener) {
    if (!listener) return -1;
    
    pthread_mutex_lock(&listener->lock);
    listener->running = 0;
    if (listener->socket_fd >= 0) {
        close(listener->socket_fd);
        listener->socket_fd = -1;
    }
    pthread_mutex_unlock(&listener->lock);
    
    pthread_join(listener->thread, NULL);
    pthread_mutex_destroy(&listener->lock);
    free(listener);
    
    return 0;
}

/* Connection management functions */

int remoteio_conn_create(remoteio_context_t* ctx, const char* addr, uint16_t port,
                         remoteio_connection_t** conn_out) {
    if (!ctx || !addr || !conn_out) return -1;
    
    remoteio_connection_t* conn = calloc(1, sizeof(remoteio_connection_t));
    if (!conn) return -1;
    
    pthread_mutex_init(&conn->lock, NULL);
    pthread_cond_init(&conn->state_cond, NULL);
    conn->socket_fd = -1;
    conn->state = REMOTEIO_CONN_DISCONNECTED;
    conn->ref_count = 1;
    strncpy(conn->peer_addr, addr, INET_ADDRSTRLEN - 1);
    conn->peer_port = port;
    
    *conn_out = conn;
    return 0;
}

int remoteio_conn_destroy(remoteio_connection_t* conn) {
    if (!conn) return -1;
    
    pthread_mutex_lock(&conn->lock);
    
    if (conn->state == REMOTEIO_CONN_CONNECTED) {
        pthread_mutex_unlock(&conn->lock);
        remoteio_network_disconnect(conn);
        pthread_mutex_lock(&conn->lock);
    }
    
    if (conn->socket_fd >= 0) {
        close(conn->socket_fd);
        conn->socket_fd = -1;
    }
    
    pthread_mutex_unlock(&conn->lock);
    pthread_mutex_destroy(&conn->lock);
    pthread_cond_destroy(&conn->state_cond);
    
    free(conn);
    return 0;
}

int remoteio_conn_acquire(remoteio_connection_t* conn) {
    if (!conn) return -1;
    pthread_mutex_lock(&conn->lock);
    conn->ref_count++;
    pthread_mutex_unlock(&conn->lock);
    return 0;
}

int remoteio_conn_release(remoteio_connection_t* conn) {
    if (!conn) return -1;
    
    pthread_mutex_lock(&conn->lock);
    conn->ref_count--;
    int should_destroy = (conn->ref_count <= 0);
    pthread_mutex_unlock(&conn->lock);
    
    if (should_destroy) {
        remoteio_conn_destroy(conn);
    }
    return 0;
}

int remoteio_conn_pool_init(remoteio_conn_pool_t* pool, int max_conns) {
    if (!pool) return -1;
    
    memset(pool, 0, sizeof(*pool));
    pool->max_connections = max_conns;
    pthread_mutex_init(&pool->lock, NULL);
    
    return 0;
}

void remoteio_conn_pool_cleanup(remoteio_conn_pool_t* pool) {
    if (!pool) return;
    
    pthread_mutex_lock(&pool->lock);
    
    remoteio_connection_t* conn = pool->connections;
    while (conn) {
        remoteio_connection_t* next = conn->next;
        remoteio_conn_release(conn);
        conn = next;
    }
    pool->connections = NULL;
    pool->num_connections = 0;
    
    pthread_mutex_unlock(&pool->lock);
    pthread_mutex_destroy(&pool->lock);
}

remoteio_connection_t* remoteio_conn_pool_get(remoteio_conn_pool_t* pool,
                                               const char* addr, uint16_t port) {
    if (!pool || !addr) return NULL;
    
    pthread_mutex_lock(&pool->lock);
    
    /* Search for existing connection */
    remoteio_connection_t* conn = pool->connections;
    while (conn) {
        if (strcmp(conn->peer_addr, addr) == 0 && conn->peer_port == port) {
            if (conn->state == REMOTEIO_CONN_CONNECTED) {
                remoteio_conn_acquire(conn);
                pthread_mutex_unlock(&pool->lock);
                return conn;
            }
        }
        conn = conn->next;
    }
    
    pthread_mutex_unlock(&pool->lock);
    return NULL;
}

int remoteio_conn_pool_put(remoteio_conn_pool_t* pool, remoteio_connection_t* conn) {
    if (!pool || !conn) return -1;
    
    pthread_mutex_lock(&pool->lock);
    
    /* Check if already in pool */
    remoteio_connection_t* existing = pool->connections;
    while (existing) {
        if (existing == conn) {
            pthread_mutex_unlock(&pool->lock);
            return 0;
        }
        existing = existing->next;
    }
    
    /* Add to pool if not full */
    if (pool->num_connections < pool->max_connections) {
        conn->next = pool->connections;
        pool->connections = conn;
        pool->num_connections++;
        pthread_mutex_unlock(&pool->lock);
        return 0;
    }
    
    pthread_mutex_unlock(&pool->lock);
    return -1; /* Pool full */
}

/* Operation management */

remoteio_operation_t* remoteio_op_alloc(remoteio_context_t* ctx) {
    if (!ctx) return NULL;
    
    pthread_mutex_lock(&ctx->ops_lock);
    
    /* Try to reuse from free list */
    if (ctx->free_ops) {
        remoteio_operation_t* op = ctx->free_ops;
        ctx->free_ops = op->next;
        memset(op, 0, sizeof(*op));
        op->id = ctx->next_op_id++;
        pthread_mutex_unlock(&ctx->ops_lock);
        return op;
    }
    
    pthread_mutex_unlock(&ctx->ops_lock);
    
    /* Allocate new operation */
    remoteio_operation_t* op = calloc(1, sizeof(remoteio_operation_t));
    if (op) {
        pthread_mutex_lock(&ctx->ops_lock);
        op->id = ctx->next_op_id++;
        pthread_mutex_unlock(&ctx->ops_lock);
    }
    
    return op;
}

void remoteio_op_free(remoteio_context_t* ctx, remoteio_operation_t* op) {
    if (!ctx || !op) return;
    
    pthread_mutex_lock(&ctx->ops_lock);
    
    /* Add to free list (limit size) */
    int free_count = 0;
    remoteio_operation_t* temp = ctx->free_ops;
    while (temp) {
        free_count++;
        temp = temp->next;
    }
    
    if (free_count < 100) {
        op->next = ctx->free_ops;
        ctx->free_ops = op;
        pthread_mutex_unlock(&ctx->ops_lock);
    } else {
        pthread_mutex_unlock(&ctx->ops_lock);
        free(op);
    }
}

int remoteio_op_submit(remoteio_context_t* ctx, remoteio_operation_t* op) {
    if (!ctx || !op || !op->conn) return -1;
    
    pthread_mutex_lock(&ctx->ops_lock);
    
    /* Add to pending list */
    op->next = ctx->pending_ops;
    ctx->pending_ops = op;
    
    ctx->requests_submitted++;
    pthread_cond_broadcast(&ctx->ops_cond);
    pthread_mutex_unlock(&ctx->ops_lock);
    
    /* Execute based on transport */
    int ret = -1;
    
    if (op->conn->transport == REMOTEIO_TRANSPORT_RDMA && op->conn->rdma_ep) {
        switch (op->op) {
            case REMOTEIO_OP_READ:
                if (op->local_gdr) {
                    remoteio_remote_mem_t remote = {
                        .raddr = op->remote_offset,
                        .rkey = 0, /* TODO: Exchange with remote */
                        .length = op->length
                    };
                    ret = remoteio_rdma_post_read(op->conn, op->local_gdr, &remote,
                                                  op->local_offset, 0, op->length, op);
                }
                break;
            case REMOTEIO_OP_WRITE:
                if (op->local_gdr) {
                    remoteio_remote_mem_t remote = {
                        .raddr = op->remote_offset,
                        .rkey = 0,
                        .length = op->length
                    };
                    ret = remoteio_rdma_post_write(op->conn, op->local_gdr, &remote,
                                                   op->local_offset, 0, op->length, op);
                }
                break;
            default:
                break;
        }
    } else {
        /* TCP fallback - implement simple protocol */
        /* TODO: Implement TCP-based remote IO protocol */
        ret = -1;
    }
    
    return ret;
}

int remoteio_op_wait(remoteio_operation_t* op, uint64_t timeout_us) {
    if (!op) return -1;
    
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_sec += timeout_us / 1000000;
    timeout.tv_nsec += (timeout_us % 1000000) * 1000;
    if (timeout.tv_nsec >= 1000000000) {
        timeout.tv_sec++;
        timeout.tv_nsec -= 1000000000;
    }
    
    pthread_mutex_lock(&op->conn->lock);
    
    while (!op->completed && op->conn->state == REMOTEIO_CONN_CONNECTED) {
        if (pthread_cond_timedwait(&op->conn->state_cond, &op->conn->lock, &timeout) != 0) {
            pthread_mutex_unlock(&op->conn->lock);
            return -1;
        }
    }
    
    pthread_mutex_unlock(&op->conn->lock);
    
    return op->completed ? (op->status == GPUIO_SUCCESS ? 0 : -1) : -1;
}

int remoteio_op_cancel(remoteio_operation_t* op) {
    if (!op) return -1;
    
    pthread_mutex_lock(&op->conn->lock);
    op->completed = 1;
    op->status = GPUIO_ERROR_CANCELED;
    pthread_cond_broadcast(&op->conn->state_cond);
    pthread_mutex_unlock(&op->conn->lock);
    
    return 0;
}
