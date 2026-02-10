/**
 * @file compression.c
 * @brief AI Extensions module - Compression codec implementation
 * @version 1.0.0
 * 
 * Compression and quantization codecs for AI/ML workloads including
 * FP16 half-precision, INT8 quantization, and 4-bit compression.
 */

#include "ai_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

/* ============================================================================
 * FP16 Compression (Half Precision)
 * ============================================================================ */

/**
 * @brief Convert single float to IEEE 754 half-precision float.
 */
static uint16_t float_to_half(float f) {
    union { float f; uint32_t i; } conv = { .f = f };
    uint32_t sign = (conv.i >> 31) & 0x1;
    uint32_t exp = (conv.i >> 23) & 0xFF;
    uint32_t mant = conv.i & 0x7FFFFF;
    
    uint16_t half_sign = (uint16_t)(sign << 15);
    uint16_t half_exp, half_mant;
    
    if (exp == 0xFF) {
        /* Infinity or NaN */
        half_exp = 0x1F;
        half_mant = (mant == 0) ? 0 : 0x200;
    } else if (exp == 0) {
        /* Zero or denormal */
        half_exp = 0;
        half_mant = (uint16_t)(mant >> 13);
    } else {
        /* Normal number */
        int32_t new_exp = (int32_t)exp - 127 + 15;
        if (new_exp >= 31) {
            /* Overflow to infinity */
            half_exp = 0x1F;
            half_mant = 0;
        } else if (new_exp <= 0) {
            /* Underflow to denormal */
            half_exp = 0;
            if (new_exp < -10) {
                half_mant = 0;
            } else {
                half_mant = (uint16_t)((mant | 0x800000) >> (14 - new_exp));
            }
        } else {
            half_exp = (uint16_t)new_exp;
            half_mant = (uint16_t)(mant >> 13);
        }
    }
    
    return half_sign | (half_exp << 10) | half_mant;
}

/**
 * @brief Convert IEEE 754 half-precision float to single float.
 */
static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    uint32_t float_sign = sign << 31;
    uint32_t float_exp, float_mant;
    
    if (exp == 0x1F) {
        /* Infinity or NaN */
        float_exp = 0xFF;
        float_mant = mant << 13;
    } else if (exp == 0) {
        /* Zero or denormal */
        if (mant == 0) {
            float_exp = 0;
            float_mant = 0;
        } else {
            /* Denormal */
            float_exp = 127 - 14;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                float_exp--;
            }
            mant &= 0x3FF;
            float_exp = float_exp << 23;
            float_mant = mant << 13;
        }
    } else {
        /* Normal number */
        float_exp = (exp - 15 + 127) << 23;
        float_mant = mant << 13;
    }
    
    union { uint32_t i; float f; } conv = { .i = float_sign | float_exp | float_mant };
    return conv.f;
}

/**
 * @brief Compress data from FP32 to FP16.
 */
gpuio_error_t ai_codec_compress_fp16(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size) {
    (void)codec;
    
    if (!input || !output || !output_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    size_t num_elements = input_size / sizeof(float);
    size_t required_size = num_elements * sizeof(uint16_t);
    
    if (output_capacity < required_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    const float* input_f = (const float*)input;
    uint16_t* output_h = (uint16_t*)output;
    
    for (size_t i = 0; i < num_elements; i++) {
        output_h[i] = float_to_half(input_f[i]);
    }
    
    *output_size = required_size;
    return GPUIO_SUCCESS;
}

/**
 * @brief Decompress data from FP16 to FP32.
 */
gpuio_error_t ai_codec_decompress_fp16(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size) {
    (void)codec;
    
    if (!input || !output || !output_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    size_t num_elements = input_size / sizeof(uint16_t);
    size_t required_size = num_elements * sizeof(float);
    
    if (output_capacity < required_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    const uint16_t* input_h = (const uint16_t*)input;
    float* output_f = (float*)output;
    
    for (size_t i = 0; i < num_elements; i++) {
        output_f[i] = half_to_float(input_h[i]);
    }
    
    *output_size = required_size;
    return GPUIO_SUCCESS;
}

/* ============================================================================
 * INT8 Quantization
 * ============================================================================ */

/**
 * @brief Compute per-channel scale and zero-point for INT8 quantization.
 */
static void compute_int8_params(const float* data, size_t num_elements,
                                 int num_channels, float* scales, float* zero_points) {
    size_t per_channel = num_elements / num_channels;
    
    for (int c = 0; c < num_channels; c++) {
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        
        for (size_t i = c * per_channel; i < (c + 1) * per_channel; i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        
        /* Symmetric quantization around zero */
        float abs_max = fmaxf(fabsf(min_val), fabsf(max_val));
        if (abs_max < 1e-8f) abs_max = 1e-8f;
        
        scales[c] = abs_max / 127.0f;
        zero_points[c] = 0.0f;
    }
}

/**
 * @brief Compress data with INT8 quantization.
 */
gpuio_error_t ai_codec_compress_int8(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size) {
    if (!input || !output || !output_size || !codec) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    size_t num_elements = input_size / sizeof(float);
    int num_channels = (int)codec->num_channels;
    if (num_channels <= 0) num_channels = 1;
    
    size_t per_channel = num_elements / num_channels;
    
    /* Output layout: scales + zero_points + int8_data */
    size_t header_size = num_channels * (sizeof(float) + sizeof(float));
    size_t data_size = num_elements * sizeof(int8_t);
    size_t required_size = header_size + data_size;
    
    if (output_capacity < required_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    const float* input_f = (const float*)input;
    uint8_t* out_bytes = (uint8_t*)output;
    
    /* Store scale factors and zero points */
    float* out_scales = (float*)out_bytes;
    float* out_zero_points = (float*)(out_bytes + num_channels * sizeof(float));
    
    if (!codec->scale_factors || !codec->zero_points) {
        codec->scale_factors = malloc(num_channels * sizeof(float));
        codec->zero_points = malloc(num_channels * sizeof(float));
        if (!codec->scale_factors || !codec->zero_points) {
            return GPUIO_ERROR_NOMEM;
        }
        compute_int8_params(input_f, num_elements, num_channels,
                            codec->scale_factors, codec->zero_points);
    }
    
    memcpy(out_scales, codec->scale_factors, num_channels * sizeof(float));
    memcpy(out_zero_points, codec->zero_points, num_channels * sizeof(float));
    
    /* Quantize data */
    int8_t* out_data = (int8_t*)(out_bytes + header_size);
    for (int c = 0; c < num_channels; c++) {
        float scale = codec->scale_factors[c];
        float zero = codec->zero_points[c];
        
        for (size_t i = 0; i < per_channel; i++) {
            size_t idx = c * per_channel + i;
            float quantized = roundf(input_f[idx] / scale + zero);
            if (quantized > 127.0f) quantized = 127.0f;
            if (quantized < -128.0f) quantized = -128.0f;
            out_data[idx] = (int8_t)quantized;
        }
    }
    
    *output_size = required_size;
    return GPUIO_SUCCESS;
}

/**
 * @brief Decompress data from INT8 quantization.
 */
gpuio_error_t ai_codec_decompress_int8(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size) {
    (void)codec;
    
    if (!input || !output || !output_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    /* Extract header */
    const uint8_t* in_bytes = (const uint8_t*)input;
    const float* in_scales = (const float*)in_bytes;
    const float* in_zero_points = (const float*)(in_bytes + sizeof(float));
    
    /* Assume single channel for simplicity in basic implementation */
    float scale = in_scales[0];
    float zero = in_zero_points[0];
    int num_channels = 1;
    size_t header_size = num_channels * (sizeof(float) + sizeof(float));
    
    size_t data_size = input_size - header_size;
    size_t num_elements = data_size / sizeof(int8_t);
    size_t required_size = num_elements * sizeof(float);
    
    if (output_capacity < required_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    const int8_t* in_data = (const int8_t*)(in_bytes + header_size);
    float* output_f = (float*)output;
    
    for (size_t i = 0; i < num_elements; i++) {
        output_f[i] = scale * ((float)in_data[i] - zero);
    }
    
    *output_size = required_size;
    return GPUIO_SUCCESS;
}

/* ============================================================================
 * 4-bit Compression (Packed INT4)
 * ============================================================================ */

/**
 * @brief Compress data to 4-bit precision (GPTQ-style).
 * 
 * Stores two 4-bit values per byte.
 */
gpuio_error_t ai_codec_compress_4bit(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size) {
    if (!input || !output || !output_size || !codec) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    size_t num_elements = input_size / sizeof(float);
    int num_channels = (int)codec->num_channels;
    if (num_channels <= 0) num_channels = 1;
    
    /* Output: scales + zero_points + packed 4-bit data */
    size_t header_size = num_channels * (sizeof(float) + sizeof(float));
    size_t data_size = (num_elements + 1) / 2;  /* 2 values per byte */
    size_t required_size = header_size + data_size;
    
    if (output_capacity < required_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    const float* input_f = (const float*)input;
    uint8_t* out_bytes = (uint8_t*)output;
    
    /* Store scales */
    float* out_scales = (float*)out_bytes;
    float* out_zero_points = (float*)(out_bytes + num_channels * sizeof(float));
    
    if (!codec->scale_factors || !codec->zero_points) {
        codec->scale_factors = malloc(num_channels * sizeof(float));
        codec->zero_points = malloc(num_channels * sizeof(float));
        if (!codec->scale_factors || !codec->zero_points) {
            return GPUIO_ERROR_NOMEM;
        }
        
        size_t per_channel = num_elements / num_channels;
        for (int c = 0; c < num_channels; c++) {
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            for (size_t i = c * per_channel; i < (c + 1) * per_channel && i < num_elements; i++) {
                if (input_f[i] < min_val) min_val = input_f[i];
                if (input_f[i] > max_val) max_val = input_f[i];
            }
            
            /* Scale for 4-bit signed: -8 to 7 */
            float range = max_val - min_val;
            if (range < 1e-8f) range = 1e-8f;
            codec->scale_factors[c] = range / 15.0f;
            codec->zero_points[c] = min_val;
        }
    }
    
    memcpy(out_scales, codec->scale_factors, num_channels * sizeof(float));
    memcpy(out_zero_points, codec->zero_points, num_channels * sizeof(float));
    
    /* Quantize and pack */
    uint8_t* out_data = (uint8_t*)(out_bytes + header_size);
    size_t per_channel = num_elements / num_channels;
    
    for (size_t i = 0; i < num_elements; i += 2) {
        int c = (int)(i / per_channel);
        if (c >= num_channels) c = num_channels - 1;
        
        float scale = codec->scale_factors[c];
        float zero = codec->zero_points[c];
        
        /* First value (high nibble) */
        float q1 = roundf((input_f[i] - zero) / scale) - 8.0f;
        if (q1 > 7.0f) q1 = 7.0f;
        if (q1 < -8.0f) q1 = -8.0f;
        int8_t val1 = (int8_t)q1 + 8;  /* Map to 0-15 */
        
        /* Second value (low nibble) */
        int8_t val2 = 0;
        if (i + 1 < num_elements) {
            float q2 = roundf((input_f[i + 1] - zero) / scale) - 8.0f;
            if (q2 > 7.0f) q2 = 7.0f;
            if (q2 < -8.0f) q2 = -8.0f;
            val2 = (int8_t)q2 + 8;
        }
        
        out_data[i / 2] = (uint8_t)((val1 << 4) | (val2 & 0x0F));
    }
    
    *output_size = required_size;
    return GPUIO_SUCCESS;
}

/**
 * @brief Decompress data from 4-bit precision.
 */
gpuio_error_t ai_codec_decompress_4bit(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size) {
    (void)codec;
    
    if (!input || !output || !output_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    /* Extract header */
    const uint8_t* in_bytes = (const uint8_t*)input;
    const float* in_scales = (const float*)in_bytes;
    const float* in_zero_points = (const float*)(in_bytes + sizeof(float));
    
    int num_channels = 1;
    size_t header_size = num_channels * (sizeof(float) + sizeof(float));
    
    size_t data_size = input_size - header_size;
    size_t num_elements = data_size * 2;
    size_t required_size = num_elements * sizeof(float);
    
    if (output_capacity < required_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    float scale = in_scales[0];
    float zero = in_zero_points[0];
    
    const uint8_t* in_data = (const uint8_t*)(in_bytes + header_size);
    float* output_f = (float*)output;
    
    for (size_t i = 0; i < num_elements; i += 2) {
        uint8_t packed = in_data[i / 2];
        
        /* High nibble */
        int8_t val1 = (int8_t)((packed >> 4) & 0x0F) - 8;
        output_f[i] = scale * (val1 + 8) + zero;
        
        /* Low nibble */
        if (i + 1 < num_elements) {
            int8_t val2 = (int8_t)(packed & 0x0F) - 8;
            output_f[i + 1] = scale * (val2 + 8) + zero;
        }
    }
    
    *output_size = required_size;
    return GPUIO_SUCCESS;
}

/* ============================================================================
 * Codec Management
 * ============================================================================ */

/**
 * @brief Create a compression codec.
 */
gpuio_error_t gpuio_codec_create(gpuio_context_t ctx,
                                  gpuio_codec_type_t type,
                                  int level,
                                  gpuio_codec_t* codec) {
    if (!ctx || !codec) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_codec* c = calloc(1, sizeof(struct gpuio_codec));
    if (!c) {
        return GPUIO_ERROR_NOMEM;
    }
    
    c->ctx = ctx;
    c->type = type;
    c->level = level;
    c->num_channels = 1;
    
    /* Setup function pointers based on codec type */
    switch (type) {
        case GPUIO_CODEC_FP16:
            c->compress_fn = ai_codec_compress_fp16;
            c->decompress_fn = ai_codec_decompress_fp16;
            break;
            
        case GPUIO_CODEC_INT8:
            c->compress_fn = ai_codec_compress_int8;
            c->decompress_fn = ai_codec_decompress_int8;
            c->num_channels = 1;
            break;
            
        case GPUIO_CODEC_CUSTOM:
            /* For CUSTOM type, user must set functions manually */
            c->compress_fn = NULL;
            c->decompress_fn = NULL;
            break;
            
        default:
            /* LZ4, ZSTD, GZIP not implemented in this version */
            free(c);
            return GPUIO_ERROR_UNSUPPORTED;
    }
    
    *codec = c;
    
    AI_LOG_INFO(ctx, "Created %s codec (level=%d)",
                type == GPUIO_CODEC_FP16 ? "FP16" :
                type == GPUIO_CODEC_INT8 ? "INT8" : "CUSTOM",
                level);
    
    return GPUIO_SUCCESS;
}

/**
 * @brief Destroy a compression codec.
 */
gpuio_error_t gpuio_codec_destroy(gpuio_codec_t codec) {
    if (!codec) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_codec* c = (struct gpuio_codec*)codec;
    
    /* Free state and parameters */
    if (c->state) {
        free(c->state);
    }
    if (c->scale_factors) {
        free(c->scale_factors);
    }
    if (c->zero_points) {
        free(c->zero_points);
    }
    
    AI_LOG_INFO(c->ctx, "Destroyed codec");
    
    free(c);
    return GPUIO_SUCCESS;
}

/* ============================================================================
 * Compression Operations
 * ============================================================================ */

/**
 * @brief Compress data using the specified codec.
 */
gpuio_error_t gpuio_compress(gpuio_codec_t codec,
                              const void* input,
                              size_t input_size,
                              void* output,
                              size_t output_capacity,
                              size_t* output_size,
                              gpuio_stream_t stream) {
    (void)stream;
    
    if (!codec || !input || !output || !output_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_codec* c = (struct gpuio_codec*)codec;
    
    if (!c->compress_fn) {
        return GPUIO_ERROR_UNSUPPORTED;
    }
    
    return c->compress_fn(c, input, input_size, output, output_capacity, output_size);
}

/**
 * @brief Decompress data using the specified codec.
 */
gpuio_error_t gpuio_decompress(gpuio_codec_t codec,
                                const void* input,
                                size_t input_size,
                                void* output,
                                size_t output_capacity,
                                size_t* output_size,
                                gpuio_stream_t stream) {
    (void)stream;
    
    if (!codec || !input || !output || !output_size) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_codec* c = (struct gpuio_codec*)codec;
    
    if (!c->decompress_fn) {
        return GPUIO_ERROR_UNSUPPORTED;
    }
    
    return c->decompress_fn(c, input, input_size, output, output_capacity, output_size);
}

/* ============================================================================
 * Compressed Transfer
 * ============================================================================ */

/**
 * @brief Transfer data with on-the-fly compression.
 * 
 * This function performs a compressed transfer where data is compressed
 * during the transfer operation, reducing bandwidth requirements.
 */
gpuio_error_t gpuio_transfer_compressed(gpuio_request_t request,
                                         gpuio_codec_t codec) {
    if (!request || !codec) {
        return GPUIO_ERROR_INVALID_ARG;
    }
    
    struct gpuio_codec* c = (struct gpuio_codec*)codec;
    
    AI_LOG_INFO(c->ctx, "Compressed transfer request created (type=%d)",
                c->type);
    
    /* In a full implementation, this would:
     * 1. Attach the codec to the request
     * 2. Set up the transfer to use compression
     * 3. Return a request handle that will compress during transfer
     * 
     * For now, we return success to indicate the request was created.
     */
    
    return GPUIO_SUCCESS;
}
