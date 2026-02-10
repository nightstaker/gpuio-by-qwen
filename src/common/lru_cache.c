/**
 * @file lru_cache.c
 * @brief Generic LRU cache implementation
 * @version 1.1.0
 */

#include "lru_cache.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * LRU Entry Structure
 * ============================================================================ */

struct lru_entry {
    struct lru_entry* lru_prev;
    struct lru_entry* lru_next;
    uint64_t lru_timestamp;
    int lru_ref_count;
    pthread_mutex_t lru_lock;
};

/* ============================================================================
 * LRU Cache Structure
 * ============================================================================ */

struct lru_cache {
    lru_entry_t* head;  /* Most recently used */
    lru_entry_t* tail;  /* Least recently used */
    size_t count;
    pthread_mutex_t lock;
};

/* ============================================================================
 * Entry API Implementation
 * ============================================================================ */

void lru_entry_init(lru_entry_t* entry) {
    if (!entry) return;
    entry->lru_prev = NULL;
    entry->lru_next = NULL;
    entry->lru_timestamp = 0;
    entry->lru_ref_count = 0;
}

lru_entry_t* lru_entry_prev(lru_entry_t* entry) {
    return entry ? entry->lru_prev : NULL;
}

lru_entry_t* lru_entry_next(lru_entry_t* entry) {
    return entry ? entry->lru_next : NULL;
}

void lru_entry_set_prev(lru_entry_t* entry, lru_entry_t* prev) {
    if (entry) entry->lru_prev = prev;
}

void lru_entry_set_next(lru_entry_t* entry, lru_entry_t* next) {
    if (entry) entry->lru_next = next;
}

uint64_t lru_entry_get_timestamp(lru_entry_t* entry) {
    return entry ? entry->lru_timestamp : 0;
}

void lru_entry_touch(lru_entry_t* entry) {
    if (entry) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        entry->lru_timestamp = (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
    }
}

int lru_entry_get_ref_count(lru_entry_t* entry) {
    return entry ? entry->lru_ref_count : 0;
}

void lru_entry_ref(lru_entry_t* entry) {
    if (entry) {
        __sync_fetch_and_add(&entry->lru_ref_count, 1);
    }
}

int lru_entry_unref(lru_entry_t* entry) {
    if (entry) {
        return __sync_sub_and_fetch(&entry->lru_ref_count, 1);
    }
    return 0;
}

void lru_entry_lock_init(lru_entry_t* entry) {
    if (entry) {
        pthread_mutex_init(&entry->lru_lock, NULL);
    }
}

void lru_entry_lock(lru_entry_t* entry) {
    if (entry) {
        pthread_mutex_lock(&entry->lru_lock);
    }
}

void lru_entry_unlock(lru_entry_t* entry) {
    if (entry) {
        pthread_mutex_unlock(&entry->lru_lock);
    }
}

/* ============================================================================
 * Cache API Implementation
 * ============================================================================ */

lru_cache_t* lru_cache_create(void) {
    lru_cache_t* cache = (lru_cache_t*)calloc(1, sizeof(lru_cache_t));
    if (!cache) return NULL;
    
    cache->head = NULL;
    cache->tail = NULL;
    cache->count = 0;
    pthread_mutex_init(&cache->lock, NULL);
    
    return cache;
}

void lru_cache_destroy(lru_cache_t* cache, lru_free_fn_t free_fn, void* user_data) {
    if (!cache) return;
    
    pthread_mutex_lock(&cache->lock);
    
    /* Free all entries */
    lru_entry_t* entry = cache->tail;
    while (entry) {
        lru_entry_t* prev = entry->lru_prev;
        if (free_fn) {
            free_fn(entry, user_data);
        }
        entry = prev;
    }
    
    pthread_mutex_unlock(&cache->lock);
    pthread_mutex_destroy(&cache->lock);
    free(cache);
}

void lru_cache_add(lru_cache_t* cache, lru_entry_t* entry) {
    if (!cache || !entry) return;
    
    pthread_mutex_lock(&cache->lock);
    
    /* Initialize entry links */
    entry->lru_prev = NULL;
    entry->lru_next = cache->head;
    
    /* Update old head */
    if (cache->head) {
        cache->head->lru_prev = entry;
    }
    
    /* Set new head */
    cache->head = entry;
    
    /* Set tail if first entry */
    if (!cache->tail) {
        cache->tail = entry;
    }
    
    /* Update timestamp */
    lru_entry_touch(entry);
    
    cache->count++;
    
    pthread_mutex_unlock(&cache->lock);
}

void lru_cache_remove(lru_cache_t* cache, lru_entry_t* entry) {
    if (!cache || !entry) return;
    
    pthread_mutex_lock(&cache->lock);
    
    /* Update previous entry's next pointer */
    if (entry->lru_prev) {
        entry->lru_prev->lru_next = entry->lru_next;
    } else {
        cache->head = entry->lru_next;
    }
    
    /* Update next entry's prev pointer */
    if (entry->lru_next) {
        entry->lru_next->lru_prev = entry->lru_prev;
    } else {
        cache->tail = entry->lru_prev;
    }
    
    /* Clear entry links */
    entry->lru_prev = NULL;
    entry->lru_next = NULL;
    
    if (cache->count > 0) {
        cache->count--;
    }
    
    pthread_mutex_unlock(&cache->lock);
}

void lru_cache_touch(lru_cache_t* cache, lru_entry_t* entry) {
    if (!cache || !entry) return;
    
    pthread_mutex_lock(&cache->lock);
    
    /* Remove from current position */
    if (entry->lru_prev) {
        entry->lru_prev->lru_next = entry->lru_next;
    } else {
        /* Already at head */
        if (cache->head == entry) {
            lru_entry_touch(entry);
            pthread_mutex_unlock(&cache->lock);
            return;
        }
        cache->head = entry->lru_next;
    }
    
    if (entry->lru_next) {
        entry->lru_next->lru_prev = entry->lru_prev;
    } else {
        cache->tail = entry->lru_prev;
    }
    
    /* Move to head */
    entry->lru_prev = NULL;
    entry->lru_next = cache->head;
    
    if (cache->head) {
        cache->head->lru_prev = entry;
    }
    cache->head = entry;
    
    if (!cache->tail) {
        cache->tail = entry;
    }
    
    lru_entry_touch(entry);
    
    pthread_mutex_unlock(&cache->lock);
}

lru_entry_t* lru_cache_get_lru(lru_cache_t* cache) {
    if (!cache) return NULL;
    return cache->tail;
}

lru_entry_t* lru_cache_get_mru(lru_cache_t* cache) {
    if (!cache) return NULL;
    return cache->head;
}

int lru_cache_iterate_mru(lru_cache_t* cache,
                          bool (*callback)(lru_entry_t* entry, void* user_data),
                          void* user_data) {
    if (!cache || !callback) return 0;
    
    pthread_mutex_lock(&cache->lock);
    
    int count = 0;
    lru_entry_t* entry = cache->head;
    
    while (entry) {
        lru_entry_t* next = entry->lru_next;
        pthread_mutex_lock(&entry->lru_lock);
        bool cont = callback(entry, user_data);
        pthread_mutex_unlock(&entry->lru_lock);
        count++;
        if (!cont) break;
        entry = next;
    }
    
    pthread_mutex_unlock(&cache->lock);
    return count;
}

int lru_cache_iterate_lru(lru_cache_t* cache,
                          bool (*callback)(lru_entry_t* entry, void* user_data),
                          void* user_data) {
    if (!cache || !callback) return 0;
    
    pthread_mutex_lock(&cache->lock);
    
    int count = 0;
    lru_entry_t* entry = cache->tail;
    
    while (entry) {
        lru_entry_t* prev = entry->lru_prev;
        pthread_mutex_lock(&entry->lru_lock);
        bool cont = callback(entry, user_data);
        pthread_mutex_unlock(&entry->lru_lock);
        count++;
        if (!cont) break;
        entry = prev;
    }
    
    pthread_mutex_unlock(&cache->lock);
    return count;
}

int lru_cache_evict(lru_cache_t* cache,
                    bool (*should_evict)(lru_entry_t* entry, void* user_data),
                    void (*on_evict)(lru_entry_t* entry, void* user_data),
                    void* user_data) {
    if (!cache || !should_evict) return 0;
    
    int evicted = 0;
    
    pthread_mutex_lock(&cache->lock);
    
    lru_entry_t* entry = cache->tail;
    while (entry) {
        lru_entry_t* prev = entry->lru_prev;
        
        pthread_mutex_lock(&entry->lru_lock);
        bool do_evict = should_evict(entry, user_data);
        pthread_mutex_unlock(&entry->lru_lock);
        
        if (do_evict) {
            /* Remove from list */
            if (entry->lru_prev) {
                entry->lru_prev->lru_next = entry->lru_next;
            } else {
                cache->head = entry->lru_next;
            }
            
            if (entry->lru_next) {
                entry->lru_next->lru_prev = entry->lru_prev;
            } else {
                cache->tail = entry->lru_prev;
            }
            
            entry->lru_prev = NULL;
            entry->lru_next = NULL;
            cache->count--;
            
            if (on_evict) {
                on_evict(entry, user_data);
            }
            evicted++;
        } else {
            /* Stop evicting if callback returns false */
            break;
        }
        
        entry = prev;
    }
    
    pthread_mutex_unlock(&cache->lock);
    return evicted;
}

size_t lru_cache_count(lru_cache_t* cache) {
    if (!cache) return 0;
    return cache->count;
}

void lru_cache_lock(lru_cache_t* cache) {
    if (cache) {
        pthread_mutex_lock(&cache->lock);
    }
}

void lru_cache_unlock(lru_cache_t* cache) {
    if (cache) {
        pthread_mutex_unlock(&cache->lock);
    }
}
