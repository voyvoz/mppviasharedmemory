#ifndef SHARED_ALLOCATOR_HELPER_H
#define SHARED_ALLOCATOR_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "smm/internal.h"

// ----------
// Debug Helper

void shm_print_state(shared_mem_allocator_handle* handle);

// ----------
// Exposed for tests only
uint64_t create_alloc_mask(size_t start_index, size_t bit_count);
int find_consecutive_space_then_apply_mask(uint64_t* bitmap, size_t count);

#ifdef __cplusplus
}
#endif

#endif  // SHARED_ALLOCATOR_HELPER_H
