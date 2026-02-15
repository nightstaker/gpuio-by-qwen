/**
 * @file test_memio_zero_copy.c
 * @brief Test zero-copy CPU DRAM access in MemIO
 * @version 1.0.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gpuio/gpuio.h"

/* Test zero-copy CPU DRAM allocation and access */
int test_zero_copy_cpu_dram(void) {
    gpuio_context_t ctx;
    gpuio_error_t err;
    
    /* Initialize context */
    gpuio_config_t config = GPUIO_CONFIG_DEFAULT;
    err = gpuio_init(&ctx, &config);
    if (err != GPUIO_SUCCESS) {
        printf("Failed to initialize gpuio context\n");
        return -1;
    }
    
    printf("✓ gpuio context initialized\n");
    
    /* Allocate zero-copy memory */
    void* ptr = NULL;
    size_t size = 4096;
    
    err = gpuio_malloc_pinned(ctx, size, &ptr);
    if (err != GPUIO_SUCCESS) {
        printf("Failed to allocate pinned memory (zero-copy)\n");
        gpuio_finalize(ctx);
        return -1;
    }
    
    printf("✓ Zero-copy CPU DRAM allocated (%zu bytes)\n", size);
    
    /* Verify memory is accessible */
    memset(ptr, 0xAA, size);
    printf("✓ Memory written successfully\n");
    
    memset(ptr, 0x55, size);
    printf("✓ Memory overwritten successfully\n");
    
    /* Test with memory region registration */
    gpuio_memory_region_t region;
    err = gpuio_register_memory(ctx, ptr, size, GPUIO_MEM_READ_WRITE, &region);
    if (err != GPUIO_SUCCESS) {
        printf("Failed to register memory region\n");
        gpuio_free(ctx, ptr);
        gpuio_finalize(ctx);
        return -1;
    }
    
    printf("✓ Memory region registered (type=%d, access=%d)\n", 
           region.type, region.access);
    
    /* Unregister before freeing */
    err = gpuio_unregister_memory(ctx, &region);
    if (err != GPUIO_SUCCESS) {
        printf("Failed to unregister memory region\n");
        gpuio_free(ctx, ptr);
        gpuio_finalize(ctx);
        return -1;
    }
    
    printf("✓ Memory region unregistered\n");
    
    /* Free memory */
    err = gpuio_free(ctx, ptr);
    if (err != GPUIO_SUCCESS) {
        printf("Failed to free memory\n");
        gpuio_finalize(ctx);
        return -1;
    }
    
    printf("✓ Memory freed successfully\n");
    
    /* Finalize */
    gpuio_finalize(ctx);
    
    printf("✓ Zero-copy CPU DRAM test PASSED\n");
    return 0;
}

int main(void) {
    printf("=== Testing MemIO Zero-Copy CPU DRAM Access ===\n\n");
    
    int result = test_zero_copy_cpu_dram();
    
    printf("\n");
    if (result == 0) {
        printf("All tests passed!\n");
    } else {
        printf("Tests failed!\n");
    }
    
    return result;
}