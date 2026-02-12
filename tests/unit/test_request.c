/**
 * @file test_request.c
 * @brief Unit tests for gpuio request management
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

TEST(request_create_basic) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* src, *dst;
    gpuio_malloc(ctx, 1024, &src);
    gpuio_malloc(ctx, 1024, &dst);
    
    gpuio_request_params_t params = {0};
    params.type = GPUIO_REQ_COPY;
    params.src = src;
    params.dst = dst;
    params.length = 1024;
    
    gpuio_request_t request;
    gpuio_error_t err = gpuio_request_create(ctx, &params, &request);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(request != NULL);
    
    gpuio_request_destroy(ctx, request);
    gpuio_free(ctx, src);
    gpuio_free(ctx, dst);
    gpuio_finalize(ctx);
}

TEST(request_null_params) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_request_t request;
    gpuio_error_t err = gpuio_request_create(ctx, NULL, &request);
    ASSERT_EQ(err, GPUIO_ERROR_INVALID_ARG);
    
    gpuio_finalize(ctx);
}

TEST(request_null_output) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* src, *dst;
    gpuio_malloc(ctx, 1024, &src);
    gpuio_malloc(ctx, 1024, &dst);
    
    gpuio_request_params_t params = {0};
    params.type = GPUIO_REQ_COPY;
    params.src = src;
    params.dst = dst;
    params.length = 1024;
    
    gpuio_error_t err = gpuio_request_create(ctx, &params, NULL);
    ASSERT_EQ(err, GPUIO_ERROR_INVALID_ARG);
    
    gpuio_free(ctx, src);
    gpuio_free(ctx, dst);
    gpuio_finalize(ctx);
}

TEST(request_get_status) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* src, *dst;
    gpuio_malloc(ctx, 1024, &src);
    gpuio_malloc(ctx, 1024, &dst);
    
    gpuio_request_params_t params = {0};
    params.type = GPUIO_REQ_COPY;
    params.src = src;
    params.dst = dst;
    params.length = 1024;
    
    gpuio_request_t request;
    gpuio_request_create(ctx, &params, &request);
    
    gpuio_request_status_t status;
    gpuio_error_t err = gpuio_request_get_status(ctx, request, &status);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_request_destroy(ctx, request);
    gpuio_free(ctx, src);
    gpuio_free(ctx, dst);
    gpuio_finalize(ctx);
}

TEST(request_cancel) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* src, *dst;
    gpuio_malloc(ctx, 1024, &src);
    gpuio_malloc(ctx, 1024, &dst);
    
    gpuio_request_params_t params = {0};
    params.type = GPUIO_REQ_COPY;
    params.src = src;
    params.dst = dst;
    params.length = 1024;
    
    gpuio_request_t request;
    gpuio_request_create(ctx, &params, &request);
    
    gpuio_error_t err = gpuio_request_cancel(ctx, request);
    /* Cancellation may or may not succeed depending on state */
    ASSERT(err == GPUIO_SUCCESS || err == GPUIO_ERROR_INVALID_ARG);
    
    gpuio_request_destroy(ctx, request);
    gpuio_free(ctx, src);
    gpuio_free(ctx, dst);
    gpuio_finalize(ctx);
}

int main(void) {
    printf("Running request management tests...\n");
    
    RUN_TEST(request_create_basic);
    RUN_TEST(request_null_params);
    RUN_TEST(request_null_output);
    RUN_TEST(request_get_status);
    RUN_TEST(request_cancel);
    
    printf("\nResults: %d/%d tests passed\n", tests_passed, tests_run);
    return tests_failed > 0 ? 1 : 0;
}
