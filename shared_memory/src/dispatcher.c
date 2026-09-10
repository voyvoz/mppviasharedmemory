#include <errno.h>   // For errno
#include <stdio.h>   // For perror, fprintf
#include <stdlib.h>  // For malloc, free
#include <string.h>  // For memset, strncpy

// POSIX specific includes for shared memory
#include <fcntl.h>     // For O_CREAT, O_RDWR, O_EXCL
#include <sys/mman.h>  // For mmap, munmap, shm_open, shm_unlink
#include <sys/stat.h>  // For S_IRUSR, S_IWUSR (permissions)
#include <unistd.h>    // For ftruncate, close, getpid

#define API_PRINT_PREFIX "[DISP] "

#include "dispatcher_api.h"
#include "internal/common.h"

DispatchResult dispatch_job(const JobBoardHandle* handle,
                            PostableJobTree* job) {
  assert(handle != NULL);
  assert(handle->jb_state != NULL);
  assert(handle->tasks != NULL);
  assert(job != NULL);
  assert(_integrity_check(handle));

  TRACY_ZONE_START

  DispatchResult ret = {.job = INVALID_SHM_OBJ, .pair_idx = 0};

  // Create the job structure as a shared object in the shared memory
  shm_obj_t j_obj = shm_malloc(&handle->alloc_handle, sizeof(Job));
  if (j_obj == INVALID_SHM_OBJ) {
    LOG_ERR(API_PRINT_PREFIX "[%u] shm_malloc error\n",
            handle->instance_board_id);

    TRACY_ZONE_END
    return ret;
  }
  Job* j = shm_obj_ptr_data(handle->alloc_handle.base_addr, j_obj);

  // Set new scheduler data
  j->meta.task_id = atomic_fetch_add(&handle->jb_state->next_task_id, 1);
  j->meta._params = job->input;
  j->state.result_code = 0;

  // put into the queue of best fitting exec
  while (1) {
    // Spin until an executor is available
    for (size_t i = 0; i < QUEUE_PAIRS; i++) {
      // TODO should be atomic_load?
      if (handle->rb_state->_pairs[i].in_use) {
        // Some conversions to make things easiert
        RingPair* rp = &handle->rb_state->_pairs[i];
        if (!atomic_load(&rp->valid)) {
          continue;
        }

        LOG_INF(API_PRINT_PREFIX "[%u] posting job obj %llu on RingPair %zu\n",
                handle->instance_board_id, j_obj, i);

        if (ring_put(&rp->doubleRing.sub, j_obj) != LIB_SUCCESS) {
          _sleep();
          continue;
        }

        // Set correct return values
        ret.job = j_obj;
        ret.pair_idx = i;

        TRACY_ZONE_END
        return ret;
      }
    }
  }
}

shm_obj_t dispatch_await(const JobBoardHandle* handle, DispatchResult res) {
  assert(handle != NULL);
  assert(res.job != INVALID_SHM_OBJ);
  assert(handle->rb_state->_pairs[res.pair_idx].valid);
  assert(_integrity_check(handle));

  LOG_INF(API_PRINT_PREFIX "[%u] dispatch_await %llu on RingPair %zu\n",
          handle->instance_board_id, res.job, res.pair_idx);

  while (1) {
    shm_obj_t o =
        ring_pop(&handle->rb_state->_pairs[res.pair_idx].doubleRing.comp);
    if (o != INVALID_SHM_OBJ) {
      // TODO for now we dont care if its actually the correct returned job id
      return o;
    }
    _sleep();
  }
}