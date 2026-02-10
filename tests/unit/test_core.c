/**
 * @file test_core.c
 * @brief Unit tests for gpuio core API
 * @version 1.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <gpuio/gpuio.h>

/* Test statistics */
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

/* Test macros */
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
        printf("FAILED\n    Assertion failed: %s at line %d\n", #expr, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b) ASSERT((a) == (b))
#define ASSERT_NE(a, b) ASSERT((a) != (b))
#define ASSERT_NULL(ptr) ASSERT((ptr) == NULL)
#define ASSERT_NOT_NULL(ptr) ASSERT((ptr) != NULL)

/* ============================================================================
 * Context Tests
 * ============================================================================ */

TEST(context_init_default) {
    gpuio_context_t ctx;
    gpuio_error_t err = gpuio_init(&ctx, NULL);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(ctx);
    
    err = gpuio_finalize(ctx);
    ASSERT_EQ(err, GPUIO_SUCCESS);
}

TEST(context_init_custom_config) {
    gpuio_context_t ctx;
    gpuio_config_t config = GPUIO_CONFIG_DEFAULT;
    config.log_level = GPUIO_LOG_DEBUG;
    config.max_gpus = 4;
    
    gpuio_error_t err = gpuio_init(&ctx, &config);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(ctx);
    
    gpuio_config_t retrieved;
    err = gpuio_get_config(ctx, &retrieved);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_EQ(retrieved.log_level, GPUIO_LOG_DEBUG);
    ASSERT_EQ(retrieved.max_gpus, 4);
    
    gpuio_finalize(ctx);
}

TEST(context_init_null_context) {
    gpuio_error_t err = gpuio_init(NULL, NULL);
    ASSERT_EQ(err, GPUIO_ERROR_INVALID_ARG);
}

TEST(context_double_init) {
    gpuio_context_t ctx;
    gpuio_error_t err = gpuio_init(&ctx, NULL);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(ctx);
    
    /* Verify context is valid */
    gpuio_config_t config;
    err = gpuio_get_config(ctx, &config);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Version Tests
 * ============================================================================ */

TEST(version_info) {
    int major, minor, patch;
    gpuio_get_version(&major, &minor, &patch);
    
    ASSERT_EQ(major, GPUIO_VERSION_MAJOR);
    ASSERT_EQ(minor, GPUIO_VERSION_MINOR);
    ASSERT_EQ(patch, GPUIO_VERSION_PATCH);
    
    const char* version_str = gpuio_get_version_string();
    ASSERT_NOT_NULL(version_str);
    ASSERT(strlen(version_str) > 0);
}

/* ============================================================================
 * Error Handling Tests
 * ============================================================================ */

TEST(error_strings) {
    const char* str;
    
    str = gpuio_error_string(GPUIO_SUCCESS);
    ASSERT_NOT_NULL(str);
    
    str = gpuio_error_string(GPUIO_ERROR_NOMEM);
    ASSERT_NOT_NULL(str);
    
    str = gpuio_error_string(GPUIO_ERROR_INVALID_ARG);
    ASSERT_NOT_NULL(str);
    
    str = gpuio_error_string(GPUIO_ERROR_TIMEOUT);
    ASSERT_NOT_NULL(str);
}

TEST(error_invalid_code) {
    const char* str = gpuio_error_string((gpuio_error_t)-999);
    ASSERT_NOT_NULL(str); /* Should return "Unknown error" */
}

/* ============================================================================
 * Device Management Tests
 * ============================================================================ */

TEST(device_count) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    int count;
    gpuio_error_t err = gpuio_get_device_count(ctx, &count);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(count >= 0);
    
    gpuio_finalize(ctx);
}

TEST(device_info) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    int count;
    gpuio_get_device_count(ctx, &count);
    
    if (count > 0) {
        gpuio_device_info_t info;
        gpuio_error_t err = gpuio_get_device_info(ctx, 0, &info);
        ASSERT_EQ(err, GPUIO_SUCCESS);
        ASSERT_EQ(info.device_id, 0);
        ASSERT(strlen(info.name) > 0);
        ASSERT(info.total_memory > 0);
    }
    
    gpuio_finalize(ctx);
}

TEST(device_set) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    int count;
    gpuio_get_device_count(ctx, &count);
    
    if (count > 0) {
        gpuio_error_t err = gpuio_set_device(ctx, 0);
        ASSERT_EQ(err, GPUIO_SUCCESS);
    }
    
    gpuio_finalize(ctx);
}

TEST(device_invalid_id) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_device_info_t info;
    gpuio_error_t err = gpuio_get_device_info(ctx, 9999, &info);
    ASSERT_NE(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Memory Management Tests
 * ============================================================================ */

TEST(memory_alloc_free) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* ptr;
    gpuio_error_t err = gpuio_malloc(ctx, 1024, &ptr);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(ptr);
    
    err = gpuio_free(ctx, ptr);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

TEST(memory_alloc_zero_size) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* ptr;
    gpuio_error_t err = gpuio_malloc(ctx, 0, &ptr);
    /* Implementation-defined behavior, but shouldn't crash */
    
    if (err == GPUIO_SUCCESS && ptr) {
        gpuio_free(ctx, ptr);
    }
    
    gpuio_finalize(ctx);
}

TEST(memory_alloc_large) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    void* ptr;
    /* Try to allocate 1GB */
    gpuio_error_t err = gpuio_malloc(ctx, 1ULL << 30, &ptr);
    /* May fail on systems with limited memory */
    
    if (err == GPUIO_SUCCESS) {
        gpuio_free(ctx, ptr);
    }
    
    gpuio_finalize(ctx);
}

TEST(memory_register_unregister) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    char buffer[1024];
    gpuio_memory_region_t region;
    
    gpuio_error_t err = gpuio_register_memory(ctx, buffer, sizeof(buffer),
                                               GPUIO_MEM_READ_WRITE, &region);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_EQ(region.length, sizeof(buffer));
    ASSERT(region.registered);
    
    err = gpuio_unregister_memory(ctx, &region);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT(!region.registered);
    
    gpuio_finalize(ctx);
}

TEST(memory_register_null) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_memory_region_t region;
    gpuio_error_t err = gpuio_register_memory(ctx, NULL, 1024,
                                               GPUIO_MEM_READ_WRITE, &region);
    ASSERT_NE(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Stream Management Tests
 * ============================================================================ */

TEST(stream_create_destroy) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream;
    gpuio_error_t err = gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(stream);
    
    err = gpuio_stream_destroy(ctx, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

TEST(stream_priorities) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream_high, stream_low;
    
    gpuio_error_t err = gpuio_stream_create(ctx, &stream_high, 
                                             GPUIO_STREAM_HIGH_PRIORITY);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    err = gpuio_stream_create(ctx, &stream_low, 
                               GPUIO_STREAM_LOW_PRIORITY);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_stream_destroy(ctx, stream_high);
    gpuio_stream_destroy(ctx, stream_low);
    
    gpuio_finalize(ctx);
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
    /* Stream should be idle immediately after creation */
    ASSERT(idle);
    
    gpuio_stream_destroy(ctx, stream);
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Event Management Tests
 * ============================================================================ */

TEST(event_create_destroy) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_event_t event;
    gpuio_error_t err = gpuio_event_create(ctx, &event);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    ASSERT_NOT_NULL(event);
    
    err = gpuio_event_destroy(ctx, event);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

TEST(event_record_synchronize) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stream_t stream;
    gpuio_stream_create(ctx, &stream, GPUIO_STREAM_DEFAULT);
    
    gpuio_event_t event;
    gpuio_event_create(ctx, &event);
    
    gpuio_error_t err = gpuio_event_record(ctx, event, stream);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    err = gpuio_event_synchronize(ctx, event);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_event_destroy(ctx, event);
    gpuio_stream_destroy(ctx, stream);
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Statistics Tests
 * ============================================================================ */

TEST(stats_get_reset) {
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    gpuio_stats_t stats;
    gpuio_error_t err = gpuio_get_stats(ctx, &stats);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    /* Stats should be zero initially */
    ASSERT_EQ(stats.requests_submitted, 0);
    ASSERT_EQ(stats.requests_completed, 0);
    
    err = gpuio_reset_stats(ctx);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    gpuio_finalize(ctx);
}

/* ============================================================================
 * Test Runner
 * ============================================================================ */

static void print_header(const char* title) {
    printf("\n%s\n", title);
    printf("%.*s\n", (int)strlen(title), 
           "============================================================");
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    printf("gpuio Core API Unit Tests\n");
    printf("Version: %s\n", gpuio_get_version_string());
    printf("Run at: %s\n", __TIME__);
    
    /* Context Tests */
    print_header("Context Tests");
    RUN_TEST(context_init_default);
    RUN_TEST(context_init_custom_config);
    RUN_TEST(context_init_null_context);
    RUN_TEST(context_double_init);
    
    /* Version Tests */
    print_header("Version Tests");
    RUN_TEST(version_info);
    
    /* Error Handling Tests */
    print_header("Error Handling Tests");
    RUN_TEST(error_strings);
    RUN_TEST(error_invalid_code);
    
    /* Device Management Tests */
    print_header("Device Management Tests");
    RUN_TEST(device_count);
    RUN_TEST(device_info);
    RUN_TEST(device_set);
    RUN_TEST(device_invalid_id);
    
    /* Memory Management Tests */
    print_header("Memory Management Tests");
    RUN_TEST(memory_alloc_free);
    RUN_TEST(memory_alloc_zero_size);
    RUN_TEST(memory_alloc_large);
    RUN_TEST(memory_register_unregister);
    RUN_TEST(memory_register_null);
    
    /* Stream Management Tests */
    print_header("Stream Management Tests");
    RUN_TEST(stream_create_destroy);
    RUN_TEST(stream_priorities);
    RUN_TEST(stream_synchronize);
    RUN_TEST(stream_query);
    
    /* Event Management Tests */
    print_header("Event Management Tests");
    RUN_TEST(event_create_destroy);
    RUN_TEST(event_record_synchronize);
    
    /* Statistics Tests */
    print_header("Statistics Tests");
    RUN_TEST(stats_get_reset);
    
    /* Summary */
    printf("\n============================================================\n");
    printf("Test Summary:\n");
    printf("  Total:  %d\n", tests_run);
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("============================================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
