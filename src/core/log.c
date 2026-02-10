/**
 * @file log.c
 * @brief Core module - Logging, errors, and statistics
 * @version 1.0.0
 */

#include "core_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>

static const char* error_strings[] = {
    [0] = "Success",
    [-GPUIO_ERROR_GENERAL] = "General error",
    [-GPUIO_ERROR_NOMEM] = "Out of memory",
    [-GPUIO_ERROR_INVALID_ARG] = "Invalid argument",
    [-GPUIO_ERROR_NOT_FOUND] = "Not found",
    [-GPUIO_ERROR_TIMEOUT] = "Timeout",
    [-GPUIO_ERROR_IO] = "I/O error",
    [-GPUIO_ERROR_NETWORK] = "Network error",
    [-GPUIO_ERROR_UNSUPPORTED] = "Unsupported operation",
    [-GPUIO_ERROR_PERMISSION] = "Permission denied",
    [-GPUIO_ERROR_BUSY] = "Resource busy",
    [-GPUIO_ERROR_CANCELED] = "Operation cancelled",
    [-GPUIO_ERROR_DEVICE_LOST] = "Device lost",
    [-GPUIO_ERROR_ALREADY_INITIALIZED] = "Already initialized",
    [-GPUIO_ERROR_NOT_INITIALIZED] = "Not initialized",
};

const char* gpuio_error_string(gpuio_error_t error) {
    if (error > 0) error = -error;
    if (error >= 0 && error < sizeof(error_strings) / sizeof(error_strings[0])) {
        return error_strings[error];
    }
    return "Unknown error";
}

static const char* level_strings[] = {
    "NONE", "FATAL", "ERROR", "WARN", "INFO", "DEBUG", "TRACE"
};

static const char* level_colors[] = {
    "", "\033[31m", "\033[31m", "\033[33m", "\033[32m", "\033[36m", "\033[35m"
};

static const char* color_reset = "\033[0m";

void gpuio_set_log_level(gpuio_log_level_t level) {
    (void)level;
}

void core_log_message(gpuio_context_t ctx, gpuio_log_level_t level,
                      const char* file, int line, const char* fmt, ...) {
    if (level > ctx->log_level) return;
    
    FILE* out = ctx->log_file ? ctx->log_file : stderr;
    
    time_t now;
    time(&now);
    struct tm* tm_info = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);
    
    int use_color = isatty(fileno(out));
    
    if (use_color) {
        fprintf(out, "%s[%s] %s%-5s%s [%s:%d] ",
                level_colors[level], timestamp, level_colors[level],
                level_strings[level], color_reset, file, line);
    } else {
        fprintf(out, "[%s] %-5s [%s:%d] ", timestamp, level_strings[level], file, line);
    }
    
    va_list args;
    va_start(args, fmt);
    vfprintf(out, fmt, args);
    va_end(args);
    
    fprintf(out, "\n");
    fflush(out);
}

void gpuio_log(gpuio_log_level_t level, const char* fmt, ...) {
    if (level > GPUIO_LOG_INFO) return;
    
    FILE* out = stderr;
    va_list args;
    va_start(args, fmt);
    fprintf(out, "[GPUIO] ");
    vfprintf(out, fmt, args);
    va_end(args);
    fprintf(out, "\n");
}

void gpuio_get_version(int* major, int* minor, int* patch) {
    if (major) *major = GPUIO_VERSION_MAJOR;
    if (minor) *minor = GPUIO_VERSION_MINOR;
    if (patch) *patch = GPUIO_VERSION_PATCH;
}

const char* gpuio_get_version_string(void) {
    static char version_string[32];
    static int initialized = 0;
    
    if (!initialized) {
        snprintf(version_string, sizeof(version_string),
                 "%d.%d.%d", GPUIO_VERSION_MAJOR, GPUIO_VERSION_MINOR, 
                 GPUIO_VERSION_PATCH);
        initialized = 1;
    }
    
    return version_string;
}

void core_stats_update(gpuio_context_t ctx, gpuio_request_type_t type,
                       size_t bytes, gpuio_error_t status) {
    pthread_mutex_lock(&ctx->stats_lock);
    
    ctx->stats.requests_submitted++;
    
    if (status == GPUIO_SUCCESS) {
        ctx->stats.requests_completed++;
        if (type == GPUIO_REQ_READ || type == GPUIO_REQ_COPY) {
            ctx->stats.bytes_read += bytes;
        }
        if (type == GPUIO_REQ_WRITE || type == GPUIO_REQ_COPY) {
            ctx->stats.bytes_written += bytes;
        }
    } else {
        ctx->stats.requests_failed++;
    }
    
    pthread_mutex_unlock(&ctx->stats_lock);
}

gpuio_error_t gpuio_get_stats(gpuio_context_t ctx, gpuio_stats_t* stats) {
    if (!ctx || !stats) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    pthread_mutex_lock(&ctx->stats_lock);
    memcpy(stats, &ctx->stats, sizeof(gpuio_stats_t));
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_reset_stats(gpuio_context_t ctx) {
    if (!ctx) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    pthread_mutex_lock(&ctx->stats_lock);
    memset(&ctx->stats, 0, sizeof(gpuio_stats_t));
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return GPUIO_SUCCESS;
}
