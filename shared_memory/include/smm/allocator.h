#ifndef SHARED_ALLOCATOR_API_H
#define SHARED_ALLOCATOR_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "smm/internal.h"

// This function will mmap the file pointed to by name to the process memory
// and create the state structure.
//  If a valid state structure exists we just create a handle pointing to it
//  If a invalid state structure exists we fail
//
// [name]   The path where the shared memory file is located
//
// [ret]    Returns 1 if a new state was created, 0 if an existing was found
int shm_alloctor_init(shared_mem_allocator_handle* handle,
                      const shared_mem_allocator_init_params* params);

// Allocate memory. Content might be anything.
// Gaurantees ownership over the returned memory.
//
// [handle] must be created via shm_allocator_init
// [bytes]  how many bytes should be allocated. Actual allocation will round up
// to the next block size
//
// [ret] The address from where memory is available or NULL if malloc failed
shm_obj_t shm_malloc(const shared_mem_allocator_handle* handle, uint64_t bytes);

// Get access to already allocated memory.
//
void* shm_map_read(const shared_mem_allocator_handle* handle, shm_obj_t obj);

//
// [handle]
// [address]
//
// [ret]
int shm_free(const shared_mem_allocator_handle* handle, shm_obj_t obj);

#ifdef __cplusplus
}
#endif

#endif  // SHARED_ALLOCATOR_API_H
