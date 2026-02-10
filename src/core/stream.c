/**
 * @file stream.c
 * @brief Core module - Stream and event management
 * @version 1.0.0
 */

#include "core_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

gpuio_error_t gpuio_stream_create(gpuio_context_t ctx, gpuio_stream_t* stream,
                                   gpuio_stream_priority_t priority) {
    if (!ctx || !stream) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    core_stream_t* internal = calloc(1, sizeof(core_stream_t));
    if (!internal) return GPUIO_ERROR_NOMEM;
    
    internal->priority = priority;
    pthread_mutex_init(&internal->lock, NULL);
    
    if (current_vendor_ops && current_vendor_ops->stream_create) {
        if (current_vendor_ops->stream_create(ctx, internal, priority) != 0) {
            pthread_mutex_destroy(&internal->lock);
            free(internal);
            return GPUIO_ERROR_GENERAL;
        }
    }
    
    pthread_mutex_lock(&ctx->streams_lock);
    int stream_id = ctx->num_streams;
    core_stream_t** new_streams = realloc(ctx->streams, 
                                           (stream_id + 1) * sizeof(void*));
    if (!new_streams) {
        pthread_mutex_unlock(&ctx->streams_lock);
        if (current_vendor_ops && current_vendor_ops->stream_destroy) {
            current_vendor_ops->stream_destroy(ctx, internal);
        }
        pthread_mutex_destroy(&internal->lock);
        free(internal);
        return GPUIO_ERROR_NOMEM;
    }
    
    ctx->streams = new_streams;
    ctx->streams[stream_id] = internal;
    internal->id = stream_id;
    ctx->num_streams++;
    pthread_mutex_unlock(&ctx->streams_lock);
    
    *stream = (gpuio_stream_t)internal;
    
    CORE_LOG(ctx, GPUIO_LOG_DEBUG, "Created stream %d with priority %d",
             stream_id, priority);
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_stream_destroy(gpuio_context_t ctx, gpuio_stream_t stream) {
    if (!ctx || !stream) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    core_stream_t* internal = (core_stream_t*)stream;
    
    if (current_vendor_ops && current_vendor_ops->stream_destroy) {
        current_vendor_ops->stream_destroy(ctx, internal);
    }
    
    pthread_mutex_destroy(&internal->lock);
    internal->id = -1;
    free(internal);
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_stream_synchronize(gpuio_context_t ctx, gpuio_stream_t stream) {
    if (!ctx) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    if (!stream) {
        pthread_mutex_lock(&ctx->streams_lock);
        for (int i = 0; i < ctx->num_streams; i++) {
            if (ctx->streams[i] && ctx->streams[i]->id >= 0) {
                if (current_vendor_ops && current_vendor_ops->stream_synchronize) {
                    current_vendor_ops->stream_synchronize(ctx, ctx->streams[i]);
                }
            }
        }
        pthread_mutex_unlock(&ctx->streams_lock);
        return GPUIO_SUCCESS;
    }
    
    core_stream_t* internal = (core_stream_t*)stream;
    
    if (current_vendor_ops && current_vendor_ops->stream_synchronize) {
        if (current_vendor_ops->stream_synchronize(ctx, internal) != 0) {
            return GPUIO_ERROR_GENERAL;
        }
    }
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_stream_query(gpuio_context_t ctx, gpuio_stream_t stream, 
                                  bool* idle) {
    if (!ctx || !stream || !idle) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    core_stream_t* internal = (core_stream_t*)stream;
    
    if (current_vendor_ops && current_vendor_ops->stream_query) {
        if (current_vendor_ops->stream_query(ctx, internal, idle) != 0) {
            return GPUIO_ERROR_GENERAL;
        }
    } else {
        *idle = true;
    }
    
    return GPUIO_SUCCESS;
}

struct gpuio_event {
    void* vendor_event;
    uint64_t timestamp;
};

gpuio_error_t gpuio_event_create(gpuio_context_t ctx, gpuio_event_t* event) {
    if (!ctx || !event) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    gpuio_event_t ev = calloc(1, sizeof(struct gpuio_event));
    if (!ev) return GPUIO_ERROR_NOMEM;
    
    if (current_vendor_ops && current_vendor_ops->event_create) {
        if (current_vendor_ops->event_create(ctx, &ev) != 0) {
            free(ev);
            return GPUIO_ERROR_GENERAL;
        }
    }
    
    *event = ev;
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_event_destroy(gpuio_context_t ctx, gpuio_event_t event) {
    if (!ctx || !event) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    if (current_vendor_ops && current_vendor_ops->event_destroy) {
        current_vendor_ops->event_destroy(ctx, event);
    }
    
    free(event);
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_event_record(gpuio_context_t ctx, gpuio_event_t event, 
                                  gpuio_stream_t stream) {
    if (!ctx || !event || !stream) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    core_stream_t* internal = (core_stream_t*)stream;
    
    if (current_vendor_ops && current_vendor_ops->event_record) {
        if (current_vendor_ops->event_record(ctx, event, internal) != 0) {
            return GPUIO_ERROR_GENERAL;
        }
    }
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_event_synchronize(gpuio_context_t ctx, gpuio_event_t event) {
    if (!ctx || !event) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    if (current_vendor_ops && current_vendor_ops->event_synchronize) {
        if (current_vendor_ops->event_synchronize(ctx, event) != 0) {
            return GPUIO_ERROR_GENERAL;
        }
    }
    
    return GPUIO_SUCCESS;
}

gpuio_error_t gpuio_event_elapsed_time(gpuio_context_t ctx, gpuio_event_t start,
                                        gpuio_event_t end, float* ms) {
    if (!ctx || !start || !end || !ms) return GPUIO_ERROR_INVALID_ARG;
    if (!ctx->initialized) return GPUIO_ERROR_NOT_INITIALIZED;
    
    if (current_vendor_ops && current_vendor_ops->event_elapsed_time) {
        if (current_vendor_ops->event_elapsed_time(ctx, start, end, ms) != 0) {
            return GPUIO_ERROR_GENERAL;
        }
    } else {
        *ms = 0.0f;
    }
    
    return GPUIO_SUCCESS;
}
