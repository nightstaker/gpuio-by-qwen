/**
 * @file test_memio.c
 * @brief Integration tests for gpuio MemIO
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gpuio/gpuio.h>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN_TEST(name) do { \
    printf("  Running %s... ", #name); \
    tests_run++; \
    test_##name(); \
    tests_passed++; \
    printf("PASSED\n"); \
} while(0)

#define ASSERT(expr) do { \
    if (!(expr)) { \
        printf("FAILED\n    Assertion: %s at line %d\n", #expr, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_MEM_EQ(a, b, size) ASSERT(memcmp((a), (b), (size)) == 0)

TEST(memcpy_basic) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* src, *dst;
    gpuio_malloc(ctx, 4096, &src);
    gpuio_malloc(ctx, 4096, &dst);
    
    /* Initialize source with pattern */
    memset(src, 0xAB, 4096);
    memset(dst, 0x00, 4096);
    
    /* Perform copy */
    gpuio_error_t err = gpuio_memcpy(ctx, dst, src, 4096, NULL);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    /* Verify */
    ASSERT_MEM_EQ(src, dst, 4096);
    
    gpuio_free(ctx, src);
    gpuio_free(ctx, dst);
    gpuio_finalize(ctx);
}

TEST(memcpy_async) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    void* src, *dst;
    gpuio_malloc(ctx, 4096, &src);
    gpuio_malloc(ctx, 4096, &dst);
    
    memset(src, 0xCD, 4096);
    memset(dst, 0x00, 4096);
    
    gpuio_error_t err = gpuio_memcpy_async(ctx, dst, src, 4096, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    err = gpuio_stream_synchronize(ctx, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    ASSERT_MEM_EQ(src, dst, 4096);
    
    gpuio_free(ctx, src);
    gpuio_free(ctx, dst);
    gpuio_stream_destroy(ctx, stream);
    gpuio_finalize(ctx);
}

TEST(memcpy_null_context) {
    void* src = malloc(1024);
    void* dst = malloc(1024);
    
    gpuio_error_t err = gpuio_memcpy(NULL, dst, src, 1024, NULL);
    ASSERT_EQ(err, GPUIO_ERROR_INVALID_ARG);
    
    free(src);
    free(dst);
}

TEST(memcpy_zero_size) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    char src[1024], dst[1024];
    gpuio_error_t err = gpuio_memcpy(ctx, dst, src, 0, NULL);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

TEST(memcpy_pinned_to_pinned) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* src, *dst;
    gpuio_malloc_pinned(ctx, 4096, &src);
    gpuio_malloc_pinned(ctx, 4096, &dst);
    
    memset(src, 0xEF, 4096);
    memset(dst, 0x00, 4096);
    
    gpuio_error_t err = gpuio_memcpy(ctx, dst, src, 4096, NULL);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_MEM_EQ(src, dst, 4096);
    
    gpuio_free(ctx, src);
    gpuio_free(ctx, dst);
    gpuio_finalize(ctx);
}

TEST(register_and_copy) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    char src[4096], dst[4096];
    memset(src, 0x12, 4096);
    memset(dst, 0x00, 4096);
    
    gpuio_memory_region_t src_reg, dst_reg;
    gpuio_register_memory(ctx, src, sizeof(src), GPUIO_MEM_READ, &src_reg);
    gpuio_register_memory(ctx, dst, sizeof(dst), GPUIO_MEM_WRITE, &dst_reg);
    
    gpuio_error_t err = gpuio_memcpy(ctx, dst, src, 4096, NULL);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_MEM_EQ(src, dst, 4096);
    
    gpuio_unregister_memory(ctx, &src_reg);
    gpuio_unregister_memory(ctx, &dst_reg);
    gpuio_finalize(ctx);
}

int main(void) {
    printf("Running MemIO integration tests...\n");
    
    RUN_TEST(memcpy_basic);
    RUN_TEST(memcpy_async);
    RUN_TEST(memcpy_null_context);
    RUN_TEST(memcpy_zero_size);
    RUN_TEST(memcpy_pinned_to_pinned);
    RUN_TEST(register_and_copy);
    
    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}
