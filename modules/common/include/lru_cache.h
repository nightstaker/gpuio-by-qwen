/**
 * @file lru_cache.h
 * @brief Generic LRU cache implementation
 * @version 1.1.0
 * 
 * Thread-safe generic LRU cache that can be embedded in other structures.
 * Extracted from duplicated LRU implementations in dsa_kv.c and engram.c.
 */

#ifndef LRU_CACHE_H
#define LRU_CACHE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Type definitions
 * ============================================================================ */

/**
 * @brief Opaque LRU entry handle.
 */
typedef struct lru_entry lru_entry_t;

/**
 * @brief LRU cache instance.
 */
typedef struct lru_cache lru_cache_t;

/**
 * @brief Callback to free an entry's resources.
 * @param entry The entry being freed
 * @param user_data User data passed to the callback
 */
typedef void (*lru_free_fn_t)(lru_entry_t* entry, void* user_data);

/**
 * @brief Callback to compute entry size for capacity tracking.
 * @param entry The entry
 * @param user_data User data passed to the callback
 * @return Size of the entry in bytes
 */
typedef size_t (*lru_size_fn_t)(lru_entry_t* entry, void* user_data);

/* ============================================================================
 * LRU Entry API
 * ============================================================================ */

/**
 * @brief Initialize an LRU entry (embedded in user's structure).
 * 
 * User's structure should embed lru_entry_t as the first member or
 * use the LRU_ENTRY_FIELDS macro.
 * 
 * @param entry Entry to initialize
 */
void lru_entry_init(lru_entry_t* entry);

/**
 * @brief Get the previous entry in the LRU list.
 */
lru_entry_t* lru_entry_prev(lru_entry_t* entry);

/**
 * @brief Get the next entry in the LRU list.
 */
lru_entry_t* lru_entry_next(lru_entry_t* entry);

/**
 * @brief Set the previous entry in the LRU list.
 */
void lru_entry_set_prev(lru_entry_t* entry, lru_entry_t* prev);

/**
 * @brief Set the next entry in the LRU list.
 */
void lru_entry_set_next(lru_entry_t* entry, lru_entry_t* next);

/**
 * @brief Get last access timestamp of an entry.
 */
uint64_t lru_entry_get_timestamp(lru_entry_t* entry);

/**
 * @brief Update the access timestamp of an entry to now.
 */
void lru_entry_touch(lru_entry_t* entry);

/**
 * @brief Get reference count of an entry.
 */
int lru_entry_get_ref_count(lru_entry_t* entry);

/**
 * @brief Increment reference count.
 */
void lru_entry_ref(lru_entry_t* entry);

/**
 * @brief Decrement reference count.
 * @return New reference count
 */
int lru_entry_unref(lru_entry_t* entry);

/**
 * @brief Initialize the mutex in an entry.
 */
void lru_entry_lock_init(lru_entry_t* entry);

/**
 * @brief Lock an entry.
 */
void lru_entry_lock(lru_entry_t* entry);

/**
 * @brief Unlock an entry.
 */
void lru_entry_unlock(lru_entry_t* entry);

/* ============================================================================
 * Macro for embedding LRU fields in user's structure
 * ============================================================================ */

#define LRU_ENTRY_FIELDS \
    struct lru_entry* lru_prev; \
    struct lru_entry* lru_next; \
    uint64_t lru_timestamp; \
    int lru_ref_count; \
    pthread_mutex_t lru_lock

/* ============================================================================
 * LRU Cache API
 * ============================================================================ */

/**
 * @brief Create a new LRU cache.
 * @return New cache instance or NULL on error
 */
lru_cache_t* lru_cache_create(void);

/**
 * @brief Destroy an LRU cache.
 * @param cache The cache to destroy
 * @param free_fn Optional callback to free entries (can be NULL)
 * @param user_data User data for free_fn
 */
void lru_cache_destroy(lru_cache_t* cache, lru_free_fn_t free_fn, void* user_data);

/**
 * @brief Add an entry to the cache (at head, most recently used).
 * @param cache The cache
 * @param entry Entry to add
 */
void lru_cache_add(lru_cache_t* cache, lru_entry_t* entry);

/**
 * @brief Remove an entry from the cache.
 * @param cache The cache
 * @param entry Entry to remove
 */
void lru_cache_remove(lru_cache_t* cache, lru_entry_t* entry);

/**
 * @brief Touch an entry (move to head, update timestamp).
 * @param cache The cache
 * @param entry Entry to touch
 */
void lru_cache_touch(lru_cache_t* cache, lru_entry_t* entry);

/**
 * @brief Get the least recently used entry (tail of list).
 * @param cache The cache
 * @return LRU entry or NULL if cache is empty
 */
lru_entry_t* lru_cache_get_lru(lru_cache_t* cache);

/**
 * @brief Get the most recently used entry (head of list).
 * @param cache The cache
 * @return MRU entry or NULL if cache is empty
 */
lru_entry_t* lru_cache_get_mru(lru_cache_t* cache);

/**
 * @brief Iterate over entries from MRU to LRU.
 * 
 * The callback should return true to continue iteration, false to stop.
 * The entry lock is held during the callback.
 * 
 * @param cache The cache
 * @param callback Function to call for each entry
 * @param user_data User data for callback
 * @return Number of entries visited
 */
int lru_cache_iterate_mru(lru_cache_t* cache, 
                          bool (*callback)(lru_entry_t* entry, void* user_data),
                          void* user_data);

/**
 * @brief Iterate over entries from LRU to MRU.
 * @param cache The cache
 * @param callback Function to call for each entry
 * @param user_data User data for callback
 * @return Number of entries visited
 */
int lru_cache_iterate_lru(lru_cache_t* cache,
                          bool (*callback)(lru_entry_t* entry, void* user_data),
                          void* user_data);

/**
 * @brief Evict entries from LRU until condition is met.
 * 
 * The should_evict callback is called with the entry lock held.
 * Return true to evict this entry, false to keep it and stop evicting.
 * 
 * @param cache The cache
 * @param should_evict Callback to determine if entry should be evicted
 * @param on_evict Optional callback when entry is evicted (can be NULL)
 * @param user_data User data for callbacks
 * @return Number of entries evicted
 */
int lru_cache_evict(lru_cache_t* cache,
                    bool (*should_evict)(lru_entry_t* entry, void* user_data),
                    void (*on_evict)(lru_entry_t* entry, void* user_data),
                    void* user_data);

/**
 * @brief Get the number of entries in the cache.
 * @param cache The cache
 * @return Entry count
 */
size_t lru_cache_count(lru_cache_t* cache);

/**
 * @brief Lock the cache for external iteration.
 * @param cache The cache
 */
void lru_cache_lock(lru_cache_t* cache);

/**
 * @brief Unlock the cache.
 * @param cache The cache
 */
void lru_cache_unlock(lru_cache_t* cache);

#ifdef __cplusplus
}
#endif

#endif /* LRU_CACHE_H */
