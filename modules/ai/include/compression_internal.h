/**
 * @file compression_internal.h
 * @brief AI Extensions module - Compression codec internal structures
 * @version 1.1.0
 * 
 * Internal structures and functions for compression and quantization codecs
 * including FP16 half-precision, INT8 quantization, and 4-bit compression.
 */

#ifndef COMPRESSION_INTERNAL_H
#define COMPRESSION_INTERNAL_H

#include "ai_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Codec Structure
 * ============================================================================ */

struct gpuio_codec {
    gpuio_context_t ctx;
    gpuio_codec_type_t type;
    int level;
    
    /* Codec state */
    void* state;
    size_t state_size;
    
    /* Function pointers */
    gpuio_error_t (*compress_fn)(struct gpuio_codec* codec,
                                  const void* input, size_t input_size,
                                  void* output, size_t output_capacity,
                                  size_t* output_size);
    gpuio_error_t (*decompress_fn)(struct gpuio_codec* codec,
                                    const void* input, size_t input_size,
                                    void* output, size_t output_capacity,
                                    size_t* output_size);
    
    /* FP16/INT8 quantization params */
    float* scale_factors;
    float* zero_points;
    uint32_t num_channels;
};

/* ============================================================================
 * Compression Internal Functions
 * ============================================================================ */

gpuio_error_t ai_codec_compress_fp16(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size);
gpuio_error_t ai_codec_decompress_fp16(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size);
gpuio_error_t ai_codec_compress_int8(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size);
gpuio_error_t ai_codec_decompress_int8(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size);
gpuio_error_t ai_codec_compress_4bit(struct gpuio_codec* codec,
                                      const void* input, size_t input_size,
                                      void* output, size_t output_capacity,
                                      size_t* output_size);
gpuio_error_t ai_codec_decompress_4bit(struct gpuio_codec* codec,
                                        const void* input, size_t input_size,
                                        void* output, size_t output_capacity,
                                        size_t* output_size);

/* ============================================================================
 * FP16 Conversion Helpers
 * ============================================================================ */

/**
 * @brief Convert single float to IEEE 754 half-precision float.
 * @param f Input float
 * @return Half-precision value as uint16_t
 */
uint16_t gpuio_float_to_half(float f);

/**
 * @brief Convert IEEE 754 half-precision float to single float.
 * @param h Half-precision value as uint16_t
 * @return Single-precision float
 */
float gpuio_half_to_float(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif /* COMPRESSION_INTERNAL_H */
