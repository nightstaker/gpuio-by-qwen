/**
 * @file common_utils.h
 * @brief Common utilities for gpuio modules
 * @version 1.1.0
 * 
 * Shared utilities including hash functions, time utilities, and alignment helpers.
 * Extracted from duplicated code in AI module components.
 */

#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Export macros
 * ============================================================================ */

#ifndef GPUIO_API
  #ifdef _WIN32
    #define GPUIO_API __declspec(dllexport)
  #else
    #define GPUIO_API __attribute__((visibility("default")))
  #endif
#endif

#ifndef GPUIO_INTERNAL
  #define GPUIO_INTERNAL __attribute__((visibility("hidden")))
#endif

/* ============================================================================
 * Time utilities
 * ============================================================================ */

/**
 * @brief Get current time in microseconds.
 * @return Time in microseconds since some fixed point (monotonic clock)
 */
static inline uint64_t gpuio_get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}

/**
 * @brief Get current time in milliseconds.
 * @return Time in milliseconds since some fixed point
 */
static inline uint64_t gpuio_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000;
}

/* ============================================================================
 * Alignment utilities
 * ============================================================================ */

/**
 * @brief Align size up to the specified alignment boundary.
 * @param size The size to align
 * @param alignment The alignment boundary (must be power of 2)
 * @return Aligned size
 */
static inline size_t gpuio_align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Align size down to the specified alignment boundary.
 * @param size The size to align
 * @param alignment The alignment boundary (must be power of 2)
 * @return Aligned size
 */
static inline size_t gpuio_align_down(size_t size, size_t alignment) {
    return size & ~(alignment - 1);
}

/**
 * @brief Check if a value is a power of 2.
 * @param x Value to check
 * @return true if power of 2, false otherwise
 */
static inline bool gpuio_is_power_of_2(size_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

/**
 * @brief Round up to the next power of 2.
 * @param x Value to round up
 * @return Next power of 2 (or same value if already power of 2)
 */
static inline size_t gpuio_next_power_of_2(size_t x) {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
#if SIZE_MAX > 0xFFFFFFFF
    x |= x >> 32;
#endif
    return x + 1;
}

/* ============================================================================
 * Hash functions
 * ============================================================================ */

/**
 * @brief FNV-1a 64-bit hash initialization value.
 */
#define GPUIO_FNV1A_64_OFFSET 14695981039346656037ULL
#define GPUIO_FNV1A_64_PRIME  1099511628211ULL

/**
 * @brief Compute FNV-1a 64-bit hash for a 64-bit value.
 * @param value Value to hash
 * @return 64-bit hash
 */
static inline uint64_t gpuio_hash_fnv1a_u64(uint64_t value) {
    uint64_t hash = GPUIO_FNV1A_64_OFFSET;
    hash ^= value;
    hash *= GPUIO_FNV1A_64_PRIME;
    return hash;
}

/**
 * @brief Compute FNV-1a 64-bit hash for three 64-bit values.
 * @param v1 First value
 * @param v2 Second value
 * @param v3 Third value
 * @return 64-bit hash
 */
static inline uint64_t gpuio_hash_fnv1a_3u64(uint64_t v1, uint64_t v2, uint64_t v3) {
    uint64_t hash = GPUIO_FNV1A_64_OFFSET;
    hash ^= v1;
    hash *= GPUIO_FNV1A_64_PRIME;
    hash ^= v2;
    hash *= GPUIO_FNV1A_64_PRIME;
    hash ^= v3;
    hash *= GPUIO_FNV1A_64_PRIME;
    return hash;
}

/**
 * @brief Compute FNV-1a 64-bit hash and reduce to bucket index.
 * @param value Value to hash
 * @param num_buckets Number of buckets (should be power of 2 for efficiency)
 * @return Bucket index
 */
static inline uint64_t gpuio_hash_to_bucket_u64(uint64_t value, uint64_t num_buckets) {
    return gpuio_hash_fnv1a_u64(value) % num_buckets;
}

/**
 * @brief Compute SplitMix64 hash for a 64-bit value.
 * Good for hash table keys with good distribution.
 * @param value Value to hash
 * @return 64-bit hash
 */
static inline uint64_t gpuio_hash_splitmix64(uint64_t value) {
    uint64_t z = value + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/**
 * @brief Compute SplitMix64 hash reduced to bucket index.
 * @param value Value to hash
 * @param num_buckets Number of buckets
 * @return Bucket index
 */
static inline uint64_t gpuio_hash_splitmix64_bucket(uint64_t value, uint64_t num_buckets) {
    return gpuio_hash_splitmix64(value) % num_buckets;
}

/**
 * @brief Compute MurmurHash3-style 64-bit hash for bytes.
 * @param data Data to hash
 * @param len Length of data in bytes
 * @param seed Seed value
 * @return 64-bit hash
 */
uint64_t gpuio_hash_bytes(const void* data, size_t len, uint64_t seed);

/* ============================================================================
 * Memory utilities
 * ============================================================================ */

/**
 * @brief Zero a memory region securely (won't be optimized away).
 * @param ptr Pointer to memory
 * @param len Length in bytes
 */
void gpuio_secure_zero(void* ptr, size_t len);

/**
 * @brief Copy memory with prefetch hints for sequential access.
 * @param dst Destination
 * @param src Source
 * @param len Length in bytes
 */
void gpuio_memcpy_prefetch(void* dst, const void* src, size_t len);

/* ============================================================================
 * Bit manipulation
 * ============================================================================ */

/**
 * @brief Count leading zeros in 32-bit value.
 */
static inline int gpuio_clz32(uint32_t x) {
    if (x == 0) return 32;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(x);
#else
    int n = 0;
    if (x <= 0x0000FFFF) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFF) { n += 8; x <<= 8; }
    if (x <= 0x0FFFFFFF) { n += 4; x <<= 4; }
    if (x <= 0x3FFFFFFF) { n += 2; x <<= 2; }
    if (x <= 0x7FFFFFFF) { n += 1; }
    return n;
#endif
}

/**
 * @brief Find first set bit (1-indexed, 0 if no bits set).
 */
static inline int gpuio_ffs32(uint32_t x) {
    if (x == 0) return 0;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ffs(x);
#else
    int n = 1;
    if ((x & 0xFFFF) == 0) { n += 16; x >>= 16; }
    if ((x & 0xFF) == 0) { n += 8; x >>= 8; }
    if ((x & 0xF) == 0) { n += 4; x >>= 4; }
    if ((x & 3) == 0) { n += 2; x >>= 2; }
    if ((x & 1) == 0) { n += 1; }
    return n;
#endif
}

/**
 * @brief Population count (number of set bits).
 */
static inline int gpuio_popcount32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    x = (x + (x >> 4)) & 0x0F0F0F0F;
    x = x + (x >> 8);
    x = x + (x >> 16);
    return x & 0x3F;
#endif
}

#ifdef __cplusplus
}
#endif

#endif /* COMMON_UTILS_H */
