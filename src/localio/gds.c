/**
 * @file gds.c
 * @brief LocalIO module - GPUDirect Storage support (stub)
 * @version 1.0.0
 */

#include "localio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

/* GDS function pointers */
typedef int (*gds_read_fn)(int fd, void* dev_ptr, size_t count, off_t offset);
typedef int (*gds_write_fn)(int fd, const void* dev_ptr, size_t count, off_t offset);

static struct {
    void* handle;
    gds_read_fn read;
    gds_write_fn write;
    int available;
} gds_lib = {0};

int localio_gds_init(localio_context_t* ctx) {
    if (!ctx) return -1;
    
    /* Try to load cuFile library */
    gds_lib.handle = dlopen("libcufile.so.0", RTLD_LAZY);
    if (!gds_lib.handle) {
        gds_lib.handle = dlopen("libcufile.so", RTLD_LAZY);
    }
    
    if (!gds_lib.handle) {
        ctx->gds_available = 0;
        return 0; /* GDS not available, but not an error */
    }
    
    /* Load functions */
    gds_lib.read = (gds_read_fn)dlsym(gds_lib.handle, "cuFileRead");
    gds_lib.write = (gds_write_fn)dlsym(gds_lib.handle, "cuFileWrite");
    
    if (gds_lib.read && gds_lib.write) {
        ctx->gds_available = 1;
        ctx->gds_handle = gds_lib.handle;
    } else {
        dlclose(gds_lib.handle);
        gds_lib.handle = NULL;
        ctx->gds_available = 0;
    }
    
    return 0;
}

void localio_gds_cleanup(localio_context_t* ctx) {
    if (!ctx) return;
    
    if (gds_lib.handle) {
        dlclose(gds_lib.handle);
        gds_lib.handle = NULL;
    }
    
    ctx->gds_available = 0;
    ctx->gds_handle = NULL;
}

int localio_gds_read(localio_context_t* ctx, int fd, void* gpu_buf,
                     size_t count, uint64_t offset) {
    if (!ctx || fd < 0 || !gpu_buf) return -1;
    
    if (!ctx->gds_available) {
        /* GDS not available - use fallback */
        /* Allocate host buffer, read to it, then copy to GPU */
        void* host_buf = malloc(count);
        if (!host_buf) return -1;
        
        ssize_t n = pread(fd, host_buf, count, offset);
        if (n < 0) {
            free(host_buf);
            return -1;
        }
        
        /* Copy to GPU through parent context */
        gpuio_error_t err = gpuio_memcpy(ctx->parent, gpu_buf, host_buf, 
                                          n, NULL);
        free(host_buf);
        
        return (err == GPUIO_SUCCESS) ? 0 : -1;
    }
    
    /* Use GDS */
    if (gds_lib.read) {
        return gds_lib.read(fd, gpu_buf, count, offset);
    }
    
    return -1;
}

int localio_gds_write(localio_context_t* ctx, int fd, const void* gpu_buf,
                      size_t count, uint64_t offset) {
    if (!ctx || fd < 0 || !gpu_buf) return -1;
    
    if (!ctx->gds_available) {
        /* GDS not available - use fallback */
        void* host_buf = malloc(count);
        if (!host_buf) return -1;
        
        /* Copy from GPU through parent context */
        gpuio_error_t err = gpuio_memcpy(ctx->parent, host_buf, gpu_buf,
                                          count, NULL);
        if (err != GPUIO_SUCCESS) {
            free(host_buf);
            return -1;
        }
        
        ssize_t n = pwrite(fd, host_buf, count, offset);
        free(host_buf);
        
        return (n == (ssize_t)count) ? 0 : -1;
    }
    
    /* Use GDS */
    if (gds_lib.write) {
        return gds_lib.write(fd, gpu_buf, count, offset);
    }
    
    return -1;
}
