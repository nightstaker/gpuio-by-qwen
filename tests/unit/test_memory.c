/**
 * @file test_memory.c
 * @brief Unit tests for gpuio memory management
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

TEST(malloc_basic) {
    gpuio_context_t ctx;
    gpuio_error_t err = gpuio_init(&ctx, NULL);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    void* ptr;
    err = gpuio_malloc(ctx, 1024, &ptr);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(ptr != NULL);
    
    memset(ptr, 0xAB, 1024);
    
    err = gpuio_free(ctx, ptr);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

TEST(malloc_null_context) {
    void* ptr;
    gpuio_error_t err = gpuio_malloc(NULL, 1024, &ptr);
    ASSERT_EQ(err, GPUIO_ERROR_INVALID_ARG);
}

TEST(malloc_pinned_basic) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* ptr;
    gpuio_error_t err = gpuio_malloc_pinned(ctx, 4096, &ptr);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(ptr != NULL);
    
    memset(ptr, 0xCD, 4096);
    gpuio_free(ctx, ptr);
    gpuio_finalize(ctx);
}

TEST(register_memory_basic) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    char buffer[4096];
    gpuio_memory_region_t region;
    
    gpuio_error_t err = gpuio_register_memory(ctx, buffer, sizeof(buffer),
                                               GPUIO_MEM_READ_WRITE, &region);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(region.registered);
    
    err = gpuio_unregister_memory(ctx, &region);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

int main(void) {
    printf("Running memory management tests...\n");
    
    RUN_TEST(malloc_basic);
    RUN_TEST(malloc_null_context);
    RUN_TEST(malloc_pinned_basic);
    RUN_TEST(register_memory_basic);
    
    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}
