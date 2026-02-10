/**
 * @file file.c
 * @brief LocalIO module - File operations
 * @version 1.0.0
 */

#include "localio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <linux/fs.h>
#include <sys/ioctl.h>

int localio_file_open(localio_context_t* ctx, const char* path, int flags,
                      localio_file_t** file_out) {
    if (!ctx || !path || !file_out) return -1;
    
    localio_file_t* file = calloc(1, sizeof(localio_file_t));
    if (!file) return -1;
    
    file->path = strdup(path);
    file->flags = flags;
    pthread_mutex_init(&file->lock, NULL);
    
    /* Determine file type */
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISBLK(st.st_mode)) {
            file->type = LOCALIO_FILE_BLOCK;
        } else if (S_ISREG(st.st_mode)) {
            file->type = LOCALIO_FILE_REGULAR;
            file->size = st.st_size;
        } else {
            file->type = LOCALIO_FILE_NVME;
        }
    } else {
        file->type = LOCALIO_FILE_REGULAR;
    }
    
    /* Open file */
    int open_flags = (flags & GPUIO_MEM_READ_WRITE) == GPUIO_MEM_READ_WRITE ? 
                     O_RDWR : 
                     (flags & GPUIO_MEM_WRITE) ? O_WRONLY : O_RDONLY;
    
    if (flags & GPUIO_MEM_WRITE) {
        open_flags |= O_CREAT;
    }
    
    /* Use O_DIRECT for block devices if available */
    if (file->type == LOCALIO_FILE_BLOCK || file->type == LOCALIO_FILE_NVME) {
        open_flags |= O_DIRECT;
    }
    
    file->fd = open(path, open_flags, 0644);
    if (file->fd < 0) {
        /* Try without O_DIRECT */
        open_flags &= ~O_DIRECT;
        file->fd = open(path, open_flags, 0644);
        if (file->fd < 0) {
            free(file->path);
            free(file);
            return -1;
        }
    }
    
    /* Get block device size if applicable */
    if (file->type == LOCALIO_FILE_BLOCK) {
        unsigned long long dev_size;
        if (ioctl(file->fd, BLKGETSIZE64, &dev_size) == 0) {
            file->size = dev_size;
        }
    }
    
    /* Add to context */
    pthread_mutex_lock(&ctx->files_lock);
    localio_file_t** new_files = realloc(ctx->files, 
                                          (ctx->num_files + 1) * sizeof(void*));
    if (!new_files) {
        pthread_mutex_unlock(&ctx->files_lock);
        close(file->fd);
        free(file->path);
        free(file);
        return -1;
    }
    ctx->files = new_files;
    ctx->files[ctx->num_files++] = file;
    pthread_mutex_unlock(&ctx->files_lock);
    
    *file_out = file;
    return 0;
}

int localio_file_close(localio_file_t* file) {
    if (!file) return -1;
    
    pthread_mutex_lock(&file->lock);
    
    if (file->fd >= 0) {
        close(file->fd);
        file->fd = -1;
    }
    
    pthread_mutex_unlock(&file->lock);
    pthread_mutex_destroy(&file->lock);
    
    free(file->path);
    free(file);
    
    return 0;
}

int localio_file_read(localio_file_t* file, void* buf, size_t count,
                      uint64_t offset, size_t* bytes_read) {
    if (!file || !buf) return -1;
    
    pthread_mutex_lock(&file->lock);
    
    if (file->fd < 0) {
        pthread_mutex_unlock(&file->lock);
        return -1;
    }
    
    /* Seek to offset */
    if (lseek(file->fd, offset, SEEK_SET) < 0) {
        pthread_mutex_unlock(&file->lock);
        return -1;
    }
    
    /* Read data */
    ssize_t n = read(file->fd, buf, count);
    if (n < 0) {
        pthread_mutex_unlock(&file->lock);
        return -1;
    }
    
    if (bytes_read) {
        *bytes_read = n;
    }
    
    pthread_mutex_unlock(&file->lock);
    return 0;
}

int localio_file_write(localio_file_t* file, const void* buf, size_t count,
                       uint64_t offset, size_t* bytes_written) {
    if (!file || !buf) return -1;
    
    pthread_mutex_lock(&file->lock);
    
    if (file->fd < 0) {
        pthread_mutex_unlock(&file->lock);
        return -1;
    }
    
    /* Seek to offset */
    if (lseek(file->fd, offset, SEEK_SET) < 0) {
        pthread_mutex_unlock(&file->lock);
        return -1;
    }
    
    /* Write data */
    ssize_t n = write(file->fd, buf, count);
    if (n < 0) {
        pthread_mutex_unlock(&file->lock);
        return -1;
    }
    
    if (bytes_written) {
        *bytes_written = n;
    }
    
    pthread_mutex_unlock(&file->lock);
    return 0;
}

int localio_file_sync(localio_file_t* file) {
    if (!file) return -1;
    
    pthread_mutex_lock(&file->lock);
    
    if (file->fd < 0) {
        pthread_mutex_unlock(&file->lock);
        return -1;
    }
    
    int ret = fsync(file->fd);
    
    pthread_mutex_unlock(&file->lock);
    return ret;
}
