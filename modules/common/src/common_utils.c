/**
 * @file common_utils.c
 * @brief Common utilities implementation
 * @version 1.1.0
 */

#include "common_utils.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Hash functions
 * ============================================================================ */

uint64_t gpuio_hash_bytes(const void* data, size_t len, uint64_t seed) {
    /* MurmurHash3-style 64-bit hash */
    const uint64_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;
    
    uint64_t h = seed ^ (len * m);
    
    const uint64_t* data64 = (const uint64_t*)data;
    const uint64_t* end = data64 + (len / 8);
    
    while (data64 != end) {
        uint64_t k = *data64++;
        
        k *= m;
        k ^= k >> r;
        k *= m;
        
        h ^= k;
        h *= m;
    }
    
    const uint8_t* data8 = (const uint8_t*)data64;
    
    switch (len & 7) {
        case 7: h ^= (uint64_t)(data8[6]) << 48;
        case 6: h ^= (uint64_t)(data8[5]) << 40;
        case 5: h ^= (uint64_t)(data8[4]) << 32;
        case 4: h ^= (uint64_t)(data8[3]) << 24;
        case 3: h ^= (uint64_t)(data8[2]) << 16;
        case 2: h ^= (uint64_t)(data8[1]) << 8;
        case 1: h ^= (uint64_t)(data8[0]);
                h *= m;
    }
    
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    
    return h;
}

/* ============================================================================
 * Memory utilities
 * ============================================================================ */

void gpuio_secure_zero(void* ptr, size_t len) {
    volatile uint8_t* p = (volatile uint8_t*)ptr;
    while (len--) {
        *p++ = 0;
    }
}

void gpuio_memcpy_prefetch(void* dst, const void* src, size_t len) {
    /* Standard implementation - can be optimized with prefetch intrinsics */
    memcpy(dst, src, len);
}
