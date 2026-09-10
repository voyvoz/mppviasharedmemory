#ifndef SHARED_ALLOCATOR_H
#define SHARED_ALLOCATOR_H

#ifdef __cplusplus
extern "C" {
#endif

/*

Shared Memory Special Purpose Allocator

This Allocator is designed to work on shared memory resources.
Its primary design goal for the API is to be stateless.
All state related data is saved to the shared memory.
Thus state changes must be guarded and updates to the state must be atomic.

OSX :$ ipcs -M
IPC status from <running system> as of Thu Oct 16 17:03:38 CEST 2025
shminfo:
        shmmax: 4194304 (max shared memory segment size)
        shmmin:       1 (min shared memory segment size)
        shmmni:      32 (max number of shared memory identifiers)
        shmseg:       8 (max shared memory segments per process)
        shmall:    1024 (max amount of shared memory in pages)

*/

#include <assert.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#define VALID_MAGIC 0xABAB00FF

#define LEAD_MAGIC 0x4269
#define END_MAGIC 0x1EEF

// On an M1 MBP we use smallest page_size == 16384 as we can only mmap segments
// of that size and multiples of that size On linux we use whatever page size is
// used, by default 4096 On CXL we use 64, or whatever multiple of it And due to
// the fact that the bitvec is designed with FRAME_SIZE of 16384 we just stick
// to that
#define FRAME_SIZE 16384

#define PCI_TX_SIZE 64

#define POPCOUNT __builtin_popcountll

#define INVALID_SHM_OBJ 0

#include "internal/types.h"

typedef struct {
  uint16_t _lead_magic;

  uint64_t _id;
  // Amount of bytes taken by this shared mem object
  uint64_t size_bytes;
  // Ref counter?
  _Atomic(uint16_t) _refs;
  // Host Bit Map?

  uint16_t _end_maigc;
  char _pad[36];
} allocated_block_metadata;
static_assert(sizeof(allocated_block_metadata) == PCI_TX_SIZE,
              "Block meta data must be PCI TX State");

typedef uint64_t shm_generic_offset_t;

// A structure to keep track of allocations of shared objects
typedef struct {
  _Atomic(uint64_t) _bitvec[FRAME_SIZE / 8];
} bit_vec_page;
static_assert(sizeof(bit_vec_page) == FRAME_SIZE, "Must be FRAME_SIZE");

// Represents the state data
typedef struct {
  _Atomic(uint32_t) magic;
  time_t created;
  uint64_t creation_size_bytes;
  uint64_t used_bytes;
  uint64_t max_bytes;

  // v2
  // Pointer to the first bitvec containing page
  shm_generic_offset_t start_bitvec;
  // Counter for how many consecutive bitvec pages we have
  uint32_t bitvec_pages;
  // Points to start of allocatable memory
  shm_generic_offset_t start_mem;

  // Some meta data
  _Atomic(uint64_t) _lifetime_allocations;
  _Atomic(uint64_t) _shm_id_counter;

  // Some generic offsets to data strucutre using this allocator.
  // Actual data structures depend on the upper layer using this allocator.
  shm_obj_t user_data1;
  shm_obj_t user_data2;
  shm_obj_t user_data3;
  shm_obj_t user_data4;
  shm_obj_t user_data5;

} AllocatorState;
static_assert(sizeof(AllocatorState) <= FRAME_SIZE,
              "Must be at most FRAME_SIZE");

// Functions to create a pointer from an offset and vice versa
static inline void* shm_generic_offset_to_ptr(const AllocatorState* base,
                                              shm_generic_offset_t offset) {
  return (char*)base + offset;
}
static inline shm_generic_offset_t shm_ptr_to_generic_offset(
    const AllocatorState* base, void* ptr) {
  assert(ptr != NULL);
  assert((uint64_t)ptr >= (uint64_t)base);
  return (shm_generic_offset_t)((uint64_t)ptr - (uint64_t)base);
}

// --- Internal only ---
static inline allocated_block_metadata* shm_obj_ptr_meta(
    const AllocatorState* base, shm_obj_t obj) {
  assert(base != NULL && "cant use NULL as base");
  assert(obj != INVALID_SHM_OBJ && "INVALID_SHM_OBJ is an invalid param");
  return (allocated_block_metadata*)((char*)base + obj);
}
// Public
static inline void* shm_obj_ptr_data(const AllocatorState* base,
                                     shm_obj_t obj) {
  assert(base != NULL && "cant use NULL as base");
  assert(obj != INVALID_SHM_OBJ && "INVALID_SHM_OBJ is an invalid param");
  return (char*)base + obj + sizeof(allocated_block_metadata);
}

// Holds the parameters used to initalize the shared memory allocator
typedef struct {
  // Path to the shared memory segment from this hosts perspective
  const char* path;
  // Size in bytes of the shared memory segment
  uint32_t size_bytes;
  // Force recreate of state block
  bool create;
} shared_mem_allocator_init_params;

// Struct used by clients to identify a shared memory allocator instance
typedef struct {
  AllocatorState* base_addr;
  uint8_t instance_id;
} shared_mem_allocator_handle;

bool shm_alloc_handle_valid(shared_mem_allocator_handle* handle);

// Use this to create a pointer from the given offest in the context of the
// handle
static inline void* to_address(const shared_mem_allocator_handle* handle,
                               uint32_t offset) {
  return (void*)(((char*)handle->base_addr) + offset);
}

#ifdef __cplusplus
}
#endif

#endif
