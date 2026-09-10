/*

bench different allocation strategies

What we want to know:
    Fragmentation
    Latency per allocation

*/

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

#include "smm/allocator.h"

#define TEST_BOARD_NAME "/bench_alloc_1"

// Parameter
// --threads denotes how many threads try to allocate in parallel
int thread_count = 1;
// --validate should toggle wether we validate accessability after each call
bool validate = 0;
// --skew give a skew to the allocation sizes
// --delay governs the delay between allocation calls
//

FILE* log_file;
_Atomic(uint8_t) thread_id = 0;

static void* some_func(void* args) {
  (void)args;

  uint8_t local_thread_id = atomic_fetch_add(&thread_id, 1);

  shared_mem_allocator_handle handle = {};
  shared_mem_allocator_init_params params = {.path = TEST_BOARD_NAME,
                                             .size_bytes = 4194304};
  bool is_new = shm_alloctor_init(&handle, &params);
  if (!shm_alloc_handle_valid(&handle)) {
    fprintf(stderr,"shm_alloctor_init failed to produce valid allocator handle");
    exit(1);
  }
  assert(is_new);

  do {
#define ELEMS 13
    shm_obj_t elems[ELEMS];
    for (size_t elem_idx = 0; elem_idx < ELEMS; elem_idx++) {
      // Generate amount of bytes we want to allocate
      uint32_t bytes_to_alloc = 16000 * (elem_idx + 1);

      struct timespec t_start, t_end;
      timespec_get(&t_start, TIME_UTC);

      // Do the allocation
      shm_obj_t ret = shm_malloc(&handle, bytes_to_alloc);
      timespec_get(&t_end, TIME_UTC);

      if (ret == 0) {
        return NULL;
      }

      void* ret_addr = shm_obj_ptr_data(handle.base_addr, ret);
      if (ret_addr == NULL) {
        return NULL;
      }

      long dur = t_end.tv_nsec - t_start.tv_nsec;
      fprintf(log_file, " %u, %lu, %u \n", local_thread_id, dur,
              bytes_to_alloc);

      if (validate) {
        // Check start writeable
        // Check end writeable
        // Check rest
      }

      elems[elem_idx] = ret;
    }
    for (size_t elem_idx = 0; elem_idx < ELEMS; elem_idx++) {
      if (shm_free(&handle, elems[elem_idx]) != 0) {
        return NULL;
      }
    }
  } while (1);

  return NULL;
}

int main() {
  // Setup a size that we map
  shm_unlink("/bench_alloc_1");

  remove("bench_alloc_1_res.csv");

  log_file = fopen("bench_alloc_1_res.csv", "a");
  fprintf(log_file, "thread_id, shm_malloc(ns), bytes \n");

  // Start the threads
  pthread_t child_ids[thread_count];
  for (int i = 0; i < thread_count; i++) {
    int res = pthread_create(&child_ids[i], NULL, some_func, NULL);
    if (res != 0) {
      fprintf(stderr,"pthread_create failed");
      exit(1);
    }
  }

  // Wait for them to be finished
  for (int i = 0; i < thread_count; i++) {
    int res = pthread_join(child_ids[i], NULL);
    if (res != 0) {
      fprintf(stderr,"pthread_join failed");
      exit(1);
    }
  }

  // Collect and present data
  // TODO

  return 0;
}
