/**
 * @file test_stream.c
 * @brief Unit tests for gpuio stream management
 */

#include <stdio.h>
#include <stdlib.h>
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

TEST(stream_create_basic) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream;
    gpuio_error_t err = gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(stream != NULL);
    
    err = gpuio_stream_destroy(ctx, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

TEST(stream_create_high_priority) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream;
    gpuio_error_t err = gpuio_stream_create(ctx, &stream, GPUIO_STREAM_HIGH_PRIORITY);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(stream != NULL);
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_finalize(ctx);
}

TEST(stream_null_context) {
    gpuio_stream_t stream;
    gpuio_error_t err = gpuio_stream_create(NULL, &stream, GPUIO_STREAM_DEFAULT);
    ASSERT_EQ(err, GPUIO_ERROR_INVALID_ARG);
}

TEST(stream_synchronize) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    gpuio_error_t err = gpuio_stream_synchronize(ctx, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_finalize(ctx);
}

TEST(stream_query) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    bool idle;
    gpuio_error_t err = gpuio_stream_query(ctx, stream, &idle);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_finalize(ctx);
}

TEST(multiple_streams) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t streams[5];
    for (int i = 0; i < 5; i++) {
        gpuio_error_t err = gpuio_stream_create(ctx, &streams[i], GPUIO_STREAM_DEFAULT);
        ASSERT_EQ(err, GPUIO_SUCCESS);
    }
    
    for (int i = 0; i < 5; i++) {
        gpuio_stream_destroy(ctx, streams[i]);
    }
    
    gpuio_finalize(ctx);
}

int main(void) {
    printf("Running stream management tests...\n");
    
    RUN_TEST(stream_create_basic);
    RUN_TEST(stream_create_high_priority);
    RUN_TEST(stream_null_context);
    RUN_TEST(stream_synchronize);
    RUN_TEST(stream_query);
    RUN_TEST(multiple_streams);
    
    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}
