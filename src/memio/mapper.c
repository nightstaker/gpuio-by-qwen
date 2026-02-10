/**
 * @file mapper.c
 * @brief MemIO module - Memory mapper for on-demand access
 * @version 1.0.0
 */

#include "memio_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/mman.h>

/* Page size - usually 4KB or 2MB for huge pages */
#define MEMIO_PAGE_SIZE 4096
#define MEMIO_HUGE_PAGE_SIZE (2 * 1024 * 1024)

/* Signal handling for page faults */
static __thread sigjmp_buf jump_buffer;
static __thread volatile int page_fault_occurred = 0;

static void page_fault_handler(int sig, siginfo_t* info, void* context) {
    (void)sig;
    (void)context;
    
    page_fault_occurred = 1;
    siglongjmp(jump_buffer, 1);
}

int memio_mapper_create(memio_context_t* ctx, size_t virtual_size,
                        memio_mapper_t** mapper_out) {
    if (!ctx || virtual_size == 0 || !mapper_out) return -1;
    
    memio_mapper_t* mapper = calloc(1, sizeof(memio_mapper_t));
    if (!mapper) return -1;
    
    mapper->virtual_size = virtual_size;
    pthread_rwlock_init(&mapper->lock, NULL);
    
    /* Allocate virtual address space */
    mapper->virtual_base = mmap(NULL, virtual_size, PROT_NONE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapper->virtual_base == MAP_FAILED) {
        free(mapper);
        return -1;
    }
    
    /* Calculate number of pages */
    mapper->num_pages = (virtual_size + MEMIO_PAGE_SIZE - 1) / MEMIO_PAGE_SIZE;
    mapper->page_mapping = calloc(mapper->num_pages, sizeof(int));
    
    *mapper_out = mapper;
    return 0;
}

int memio_mapper_destroy(memio_mapper_t* mapper) {
    if (!mapper) return -1;
    
    pthread_rwlock_wrlock(&mapper->lock);
    
    if (mapper->virtual_base) {
        munmap(mapper->virtual_base, mapper->virtual_size);
    }
    
    free(mapper->page_mapping);
    
    pthread_rwlock_unlock(&mapper->lock);
    pthread_rwlock_destroy(&mapper->lock);
    free(mapper);
    
    return 0;
}

int memio_mapper_map_page(memio_mapper_t* mapper, int page_id,
                          void* physical_page) {
    if (!mapper || page_id < 0 || page_id >= mapper->num_pages) return -1;
    
    pthread_rwlock_wrlock(&mapper->lock);
    
    void* virtual_page = (char*)mapper->virtual_base + 
                         (page_id * MEMIO_PAGE_SIZE);
    
    /* Remap the page with proper permissions */
    if (mprotect(virtual_page, MEMIO_PAGE_SIZE, PROT_READ | PROT_WRITE) != 0) {
        pthread_rwlock_unlock(&mapper->lock);
        return -1;
    }
    
    /* In real implementation, this would use remap or similar
     * to map the physical page. For now, we just copy. */
    if (physical_page) {
        memcpy(virtual_page, physical_page, MEMIO_PAGE_SIZE);
    }
    
    mapper->page_mapping[page_id] = 1; /* Mark as mapped */
    
    pthread_rwlock_unlock(&mapper->lock);
    return 0;
}

int memio_mapper_unmap_page(memio_mapper_t* mapper, int page_id) {
    if (!mapper || page_id < 0 || page_id >= mapper->num_pages) return -1;
    
    pthread_rwlock_wrlock(&mapper->lock);
    
    void* virtual_page = (char*)mapper->virtual_base + 
                         (page_id * MEMIO_PAGE_SIZE);
    
    /* Protect page to trigger fault on next access */
    mprotect(virtual_page, MEMIO_PAGE_SIZE, PROT_NONE);
    mapper->page_mapping[page_id] = 0;
    
    pthread_rwlock_unlock(&mapper->lock);
    return 0;
}

int memio_mapper_handle_fault(memio_mapper_t* mapper, void* fault_addr,
                              void** mapped_page_out) {
    if (!mapper || !fault_addr) return -1;
    
    /* Calculate page ID */
    ptrdiff_t offset = (char*)fault_addr - (char*)mapper->virtual_base;
    if (offset < 0 || offset >= (ptrdiff_t)mapper->virtual_size) {
        return -1; /* Address out of range */
    }
    
    int page_id = offset / MEMIO_PAGE_SIZE;
    
    /* In real implementation, this would:
     * 1. Look up backing store for the page
     * 2. Allocate or fetch the physical page
     * 3. Map it into the virtual address space
     * 4. Return the mapped address */
    
    pthread_rwlock_rdlock(&mapper->lock);
    
    if (mapper->page_mapping[page_id]) {
        /* Page already mapped */
        if (mapped_page_out) {
            *mapped_page_out = (char*)mapper->virtual_base + 
                               (page_id * MEMIO_PAGE_SIZE);
        }
        pthread_rwlock_unlock(&mapper->lock);
        return 0;
    }
    
    pthread_rwlock_unlock(&mapper->lock);
    return -1; /* Page not available */
}

/* Check if an address is in the mapper's range */
bool memio_mapper_contains(memio_mapper_t* mapper, void* addr) {
    if (!mapper || !addr) return false;
    
    ptrdiff_t offset = (char*)addr - (char*)mapper->virtual_base;
    return (offset >= 0 && offset < (ptrdiff_t)mapper->virtual_size);
}
