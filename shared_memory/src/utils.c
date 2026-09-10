//
#include <errno.h>  // For errno
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // For perror, fprintf
#include <stdlib.h>  // For malloc, free
#include <string.h>  // For memset

// POSIX specific includes for shared memory
#include <fcntl.h>     // For O_RDWR
#include <sys/mman.h>  // For mmap, munmap, shm_open, shm_unlink
#include <sys/stat.h>  // For S_IRUSR, S_IWUSR (permissions)
#include <unistd.h>    // For ftruncate, close, getpid

#include "internal/utils.h"
#include "smm/allocator.h"

#define API_PRINT_PREFIX "[UTIL] "

size_t shm_get_job_num(const JobBoardHandle* handle) {
  return handle->jb_state->job_slots;
}

bool shm_create_or_open(shared_mem_allocator_handle* handle,
                        shared_mem_allocator_init_params* params, bool create) {
  assert(handle != NULL);
  assert(params != NULL);
  assert(params->path != NULL);
  assert(params->size_bytes != 0);

  // When we want to create the board we unlink the name
  // to make sure that we get a clean new shared memory segment
  if (create) {
    if (shm_unlink(params->path) == -1) {
      if (errno != ENOENT) {
        perror("shm_unlink failed");
        exit(1);
      }
    }
  }

  // Get a shared memory management segment
  bool is_new = shm_alloctor_init(handle, params);
  if (!shm_alloc_handle_valid(handle)) {
    fprintf(stderr,
            "shm_alloctor_init failed to produce valid allocator handle");
    exit(1);
  }

  return is_new;
}

void shm_close(JobBoardHandle* handle, bool unlink_on_close) {
  assert(handle != NULL);

  // Unmap the memory

  // if (handle->shm_base_ptr != MAP_FAILED && handle->shm_base_ptr != NULL) {
  // if (munmap(handle->shm_base_ptr, handle->shm_size) == -1) {
  //     fprintf(stderr,"munmap failed");
  // }
  //    handle->shm_base_ptr = NULL;
  //}

  // Unlink the shared memory object if requested
  if (unlink_on_close) {
    if (shm_unlink(handle->alloc_params.path) == -1) {
      if (errno != ENOENT) {
        perror("shm_unlink failed");
      }
    }
  }
}

void print_job(const JobBoardHandle* handle, size_t i, Job* t) {
#ifdef ENABLE_LOGS
  LOG_INF(API_PRINT_PREFIX
          "[%u] [T%lu] Job: [%p]{ .id = %u, .deps_num = %u } [%p]{ .status = "
          "%u, .exec_id = %u, .deps: [ %u, %u, %u, %u, %u ]}\n",
          handle->instance_board_id, i, &t->meta, t->meta.task_id,
          t->meta.dep_jobs_num, &t->state, atomic_load(&t->state.status),
          t->state.executor_id, t->state.dep_ids[0], t->state.dep_ids[1],
          t->state.dep_ids[2], t->state.dep_ids[3], t->state.dep_ids[4]);
#else
  (void)handle;
  (void)i;
  (void)t;
#endif
}
void print_board_state_max_jobs(const JobBoardHandle* handle,
                                uint16_t num_jobs) {
  if (handle == NULL) {
    LOG_ERR(API_PRINT_PREFIX "[%u] print_board_state error: Invalid handle.\n",
            handle ? handle->instance_board_id : 0);
    return;
  }
  if (!handle->jb_state || !handle->tasks) {
    LOG_ERR(API_PRINT_PREFIX
            "[%u] print_board_state error: Invalid handle jb_state or tasks.\n",
            handle->instance_board_id);
    return;
  }

  size_t jobs_to_print =
      num_jobs < shm_get_job_num(handle) ? num_jobs : shm_get_job_num(handle);
  for (size_t i = 0; i < jobs_to_print; i++) {
    Job* t = &handle->tasks[i];

    print_job(handle, i, t);
  }
}
void print_board_state(const JobBoardHandle* handle) {
#ifdef ENABLE_LOGS
  print_board_state_max_jobs(handle, shm_get_job_num(handle));
#else
  (void)handle;
#endif
}
void print_jobBoardHandle(const JobBoardHandle* handle) {
#ifdef ENABLE_LOGS
  assert(handle != NULL);

  LOG_INF(API_PRINT_PREFIX
          "[%s:%u] handle { alloc_state=%p .jb_state = %p, .tasks = %p:%u, "
          "is_owner = %i .rb = %p }\n",
          handle->alloc_params.path, handle->instance_board_id,
          handle->alloc_handle.base_addr, handle->jb_state, handle->tasks,
          handle->jb_state->job_slots, handle->is_owner, handle->rb_state);
#else
  (void)handle;
#endif
}
