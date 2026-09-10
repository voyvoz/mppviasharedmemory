#include <errno.h>   // For errno
#include <stdio.h>   // For perror, fprintf
#include <stdlib.h>  // For malloc, free
#include <string.h>  // For memset, strncpy

// POSIX specific includes for shared memory
#include <assert.h>
#include <fcntl.h>     // For O_CREAT, O_RDWR, O_EXCL
#include <sys/mman.h>  // For mmap, munmap, shm_open, shm_unlink
#include <sys/stat.h>  // For S_IRUSR, S_IWUSR (permissions)
#include <unistd.h>    // For ftruncate, close, getpid

#include "internal/debug_helper.h"
#include "internal/job_board.h"
#include "scheduler_api.h"

#define API_PRINT_PREFIX "[SCHED] "

// --- Scheduler API Implementation ---

void scheduler_init(JobBoardHandle* jbh, int should_create) {
  assert(jbh != NULL);

  TRACY_ZONE_START

  jbh->is_valid = false;

  bool is_new =
      shm_create_or_open(&jbh->alloc_handle, &jbh->alloc_params, should_create);
  if (should_create && !is_new) {
    fprintf(stderr, "should have created a new board but didnt?\n");
    exit(1);
  }

  if (job_board_init(jbh, should_create) != 0) {
    fprintf(stderr, "job_board_init failed\n");
    exit(1);
  }

  // Copy sane values for very simple integrity checks
  jbh->org_job_slots = shm_get_job_num(jbh);
  jbh->org_version = jbh->jb_state->version;

  TRACY_ZONE_END

  return;
}

LibErr* scheduler_post_job(JobBoardHandle* handle, PostableJobTree* job,
                           job_id_t* posted_job_id) {
  assert(handle != NULL);
  assert(handle->jb_state != NULL);
  assert(handle->tasks != NULL);
  assert(job != NULL);
  assert(posted_job_id != NULL);
  assert(_integrity_check(handle));

  TRACY_ZONE_START

  *posted_job_id = INVALID_JOB_ID;

  // Post task if able
  for (size_t i = 0; i < shm_get_job_num(handle); i++) {
    job_status_t expected = TASK_STATUS_INVALID;
    // job_status_t also_expected = TASK_STATUS_FINISHED;
    //  TODO what about canceld jobs?

    if (atomic_compare_exchange_strong(&handle->tasks[i].state.status,
                                       &expected, TASK_STATUS_POST_IN_PROGRESS)
        //||
        // atomic_compare_exchange_strong(&handle->tasks[i].state.status,
        //                               &also_expected,
        //  TASK_STATUS_POST_IN_PROGRESS)
    ) {
      // Slot claimed! Initialize the task.
      uint64_t new_task_id =
          atomic_fetch_add(&handle->jb_state->next_task_id, 1);

      // Set new scheduler data
      handle->tasks[i].meta.task_id = new_task_id;
      handle->tasks[i].meta.dep_jobs_num = job->group_size;

      // Set payload data
      handle->tasks[i].meta._params = job->input;

      // Set new executor data for this new task
      handle->tasks[i].state.executor_id = 0;
      handle->tasks[i].state.result_code = 0;

      // Mark that this job has dependencies that need to be set
      // TODO remove this?
      for (int j = 0; j < job->dependant_jobs_count; j++) {
        handle->tasks[i].state.dep_ids[j] = MAX_JOB_NUM;
      }

      job_status_t in_progress = TASK_STATUS_POST_IN_PROGRESS;
      if (!atomic_compare_exchange_strong(&handle->tasks[i].state.status,
                                          &in_progress, TASK_STATUS_POSTED)) {
        atomic_store(&handle->tasks[i].state.status, TASK_STATUS_INVALID);
        TRACY_ZONE_END
        return new_err(LIB_ERROR,
                       "failed to set job from TASK_STATUS_POST_IN_PROGRESS to "
                       "TASK_STATUS_POSTED");
      }

      *posted_job_id = handle->tasks[i].meta.task_id;
      TRACY_ZONE_END
      return NULL;
    }
  }
  TRACY_ZONE_END
  return new_err(LIB_ERROR, "all %zu task slots occupied",
                 shm_get_job_num(handle));
}

// Allows to update sync point values of a posted job.
LibErr* scheduler_set_job_dep(JobBoardHandle* handle, job_id_t target_job_id,
                              job_id_t dep_job_id) {
  if (!handle || !handle->jb_state || !handle->tasks) {
    return new_err(LIB_ERROR, "invalid handle or payload_msg");
  }

  for (size_t i = 0; i < shm_get_job_num(handle); ++i) {
    if (atomic_load(&handle->tasks[i].meta.task_id) == target_job_id) {
      for (int j = 0; j < MAX_DEPENDANT_JOBS; j++) {
        if (handle->tasks[i].state.dep_ids[j] == MAX_JOB_NUM) {
          handle->tasks[i].state.dep_ids[j] = dep_job_id;
          return LIB_SUCCESS;
        }
      }
      return new_err(LIB_ERROR, "failed to set dep job for job id %u",
                     target_job_id);
    }
  }
  return new_err(LIB_ERROR, "failed to find job with id %u", target_job_id);
}

// NOTE there must be a better way to do this?
LibErr* visit_pjt_node_and_validate(PostableJobTree* node,
                                    job_deps_t* counter) {
  *counter += 1;

  // Check dependant nodes
  if (node->dependant_jobs_count != 0) {
    if (node->dependant_jobs == NULL) {
      return new_err(LIB_ERROR,
                     "dependant_jobs_count is %u but pointer is NULL",
                     node->dependant_jobs_count);
    }
    for (int i = 0; i < node->dependant_jobs_count; i++) {
      LibErr* err =
          visit_pjt_node_and_validate(node->dependant_jobs[i], counter);
      if (err != NULL) {
        return err;
      }
    }
  }
  return NULL;
}

// This will be called after all nodes have been validated so no need for error
// checking nodes themselves
LibErr* visit_pjt_node_and_post_jobs(JobBoardHandle* handle,
                                     PostableJobTree* node,
                                     job_id_t* root_job_id, size_t group_size) {
  node->group_size = group_size;
  LibErr* err;

  // Post all dependant jobs
  job_id_t dep_job_ids[node->dependant_jobs_count];
  if (node->dependant_jobs_count != 0) {
    for (int i = 0; i < node->dependant_jobs_count; i++) {
      LibErr* err = visit_pjt_node_and_post_jobs(
          handle, node->dependant_jobs[i], &dep_job_ids[i], group_size);
      if (err != NULL) {
        return new_err(LIB_ERROR,
                       "visit_pjt_node_and_post_jobs failed with: %s",
                       err->msg);
      }
    }
  }

  // Post self
  job_id_t job_id;
  err = scheduler_post_job(handle, node, &job_id);
  if (err != NULL) {
    return new_err(LIB_ERROR, "scheduler_post_job failed with: %s", err->msg);
  }

  // Setup correct dependencies
  if (node->dependant_jobs_count != 0) {
    for (int i = 0; i < node->dependant_jobs_count; i++) {
      err = scheduler_set_job_dep(handle, job_id, dep_job_ids[i]);
      if (err != NULL) {
        return new_err(LIB_ERROR, "scheduler_set_job_dep failed with: %s",
                       err->msg);
      }
    }
  }

  *root_job_id = job_id;

  return NULL;
}

//
// PUBLIC
//

int64_t scheduler_post_job_tree(const JobBoardHandle* handle,
                                PostableJobTree* job_tree_root) {
  assert(handle != NULL);
  assert(job_tree_root != NULL);
  assert(_integrity_check(handle));

  TRACY_ZONE_START

  // 0. Traverse tree and collect meta data
  // 1. Allocate job slots
  // 2. Post the jobs
  // Any error -> Deallocate job slots and fail

  // -- 0. Traverse tree and collect meta data
  job_deps_t jobs_needed = 0;
  LibErr* err = visit_pjt_node_and_validate(job_tree_root, &jobs_needed);
  if (err != NULL) {
    LOG_ERR(API_PRINT_PREFIX "[%u] scheduler_post_job_tree error: (%i) %s.\n",
            handle->instance_board_id, err->err_code, err->msg);
    TRACY_ZONE_END
    return INVALID_JOB_ID;
  }

  // -- 1. Allocate job slots
  // Trivially check that we have enough slots with a completly empty board
  if (shm_get_job_num(handle) < jobs_needed) {
    LOG_ERR(API_PRINT_PREFIX
            "[%u] scheduler_post_job_tree error: cant allocate enought jobs on "
            "board\n",
            handle->instance_board_id);
    TRACY_ZONE_END
    return INVALID_JOB_ID;
  }
try_again:
  // Now check if we can grab enough slots
  uint32_t occupied_slots = atomic_load(&handle->jb_state->active_slots);
  uint32_t new_occupied_slots = occupied_slots + jobs_needed;
  if (new_occupied_slots > shm_get_job_num(handle)) {
    LOG_ERR(API_PRINT_PREFIX
            "[%u] scheduler_post_job_tree error: not enough slots available. "
            "Try again\n",
            handle->instance_board_id);
    TRACY_ZONE_END
    return INVALID_JOB_ID;
  }
  if (!atomic_compare_exchange_strong(&handle->jb_state->active_slots,
                                      &occupied_slots, new_occupied_slots)) {
    goto try_again;
  }

  // -- 2. Post the jobs
  job_id_t root_job_id;
  err = visit_pjt_node_and_post_jobs(handle, job_tree_root, &root_job_id,
                                     jobs_needed);
  if (err != NULL) {
    // As we failed to post the jobs we need to free the slots again
    atomic_fetch_sub(&handle->jb_state->active_slots, jobs_needed);
    LOG_ERR(API_PRINT_PREFIX "[%u] scheduler_post_job_tree error: (%i) %s.\n",
            handle->instance_board_id, err->err_code, err->msg);
    TRACY_ZONE_END
    return INVALID_JOB_ID;
  }

  LOG_INF(API_PRINT_PREFIX "[%u] posted tree of %hhu jobs. Root job id: %u \n",
          handle->instance_board_id, jobs_needed, root_job_id);

  TRACY_ZONE_END

  return root_job_id;
}

int scheduler_job_state(const JobBoardHandle* handle, int64_t task_id) {
  (void)handle;
  (void)task_id;
  // TODO

  return 0;
}

int scheduler_job_wait(const JobBoardHandle* handle, int64_t task_id_arg) {
  assert(handle != NULL);
  assert(handle->jb_state != NULL);
  assert(handle->tasks != NULL);
  if (task_id_arg <= INVALID_JOB_ID) {
    LOG_ERR(API_PRINT_PREFIX
            "[%u] scheduler_job_wait error: Invalid task_id_arg\n",
            handle ? handle->instance_board_id : 0);
    return -EINVAL;
  }
  assert(_integrity_check(handle));

  TRACY_ZONE_START

  job_id_t task_id = (job_id_t)task_id_arg;

  // Find index of slot we want to watch
  size_t slot_index = 0;
  for (; slot_index < shm_get_job_num(handle); slot_index++) {
    if (handle->tasks[slot_index].meta.task_id == task_id) {
      goto job_found;
    }
  }
  LOG_ERR(API_PRINT_PREFIX "[%u] job %u is not on the board\n",
          handle->instance_board_id, task_id);
  TRACY_ZONE_END
  return LIB_ERROR;

job_found:

  LOG_INF(API_PRINT_PREFIX
          "[%u] waiting for job %u on slot %zu to be finished\n",
          handle->instance_board_id, task_id, slot_index);

  // Polling loop until task finished
  while (1) {
    // Check main job status
    job_status_t current_status =
        atomic_load(&handle->tasks[slot_index].state.status);
    switch (current_status) {
      case TASK_STATUS_FINISHED:
        // Reset task slot without clearing result data
        // TODO what about the result data.. should be returned as well here...
        uint32_t jobs_ =
            atomic_load(&handle->tasks[slot_index].meta.dep_jobs_num);
        atomic_fetch_sub(&handle->jb_state->active_slots, jobs_);
        LOG_INF(API_PRINT_PREFIX "[%u] Scheduler: Job %u finished\n",
                handle->instance_board_id, task_id);
        TRACY_ZONE_END
        return 0;
      case TASK_STATUS_CANCELD:
        // TODO check implications for job dependencies
        LOG_ERR(API_PRINT_PREFIX "[%u] Scheduler: Job %u has been canceld\n",
                handle->instance_board_id, task_id);
        TRACY_ZONE_END
        return handle->tasks[slot_index].state.result_code;

      case TASK_STATUS_IN_PROGRESS:
        // Check if timeout has been reached
        time_t current_time = time(NULL);
        time_t wdt =
            atomic_load(&handle->tasks[slot_index].state.watchdog_timeout);
        if (wdt != 0 && wdt < current_time) {
          LOG_INF(API_PRINT_PREFIX
                  "[%u] Scheduler: Job %u timed out. Resetting job...\n",
                  handle->instance_board_id, task_id);
          // Use atomic to exchange the status only if the status is still
          // pending! NOTE make sure this is the same order as when the executor
          // tries to mark a job as finished!
          atomic_store(&handle->tasks[slot_index].state.status,
                       TASK_STATUS_POSTED);
          atomic_store(&handle->tasks[slot_index].state.watchdog_timeout, 0);
          handle->tasks[slot_index].state.executor_id = 0;
        }
        break;

      default:
        break;
    }

    _sleep();
  }

  // Should not be reached in current implementation without timeout/error
  // return -1;
}

int scheduler_job_free(const JobBoardHandle* handle, int64_t task_id_arg) {
  assert(handle != NULL);
  assert(handle->jb_state != NULL);
  assert(handle->tasks != NULL);
  if (task_id_arg <= INVALID_JOB_ID) {
    LOG_ERR(API_PRINT_PREFIX
            "[%u] scheduler_job_free error: Invalid task_id_arg\n",
            handle ? handle->instance_board_id : 0);
    return -EINVAL;
  }
  assert(_integrity_check(handle));

  TRACY_ZONE_START

  job_id_t task_id = (job_id_t)task_id_arg;

  size_t slot_index = 0;
  for (; slot_index < shm_get_job_num(handle); slot_index++) {
    if (handle->tasks[slot_index].meta.task_id == task_id) {
      LOG_INF(API_PRINT_PREFIX "[%u] freeing job %u on slot %zu\n",
              handle->instance_board_id, task_id, slot_index);
      job_status_t current_status =
          atomic_load(&handle->tasks[slot_index].state.status);
      switch (current_status) {
        case TASK_STATUS_FINISHED:
        case TASK_STATUS_CANCELD:
          // TODO Free params data?
          // Set status to INVALID -> marking the job to be used again
          if (!atomic_compare_exchange_strong(
                  &handle->tasks[slot_index].state.status, &current_status,
                  TASK_STATUS_INVALID)) {
            LOG_ERR(API_PRINT_PREFIX
                    "[%u] job %u can not be freed when in %u state \n",
                    handle->instance_board_id, task_id, current_status);
            goto failed;
          }
          TRACY_ZONE_END
          return LIB_SUCCESS;
        default:
          LOG_ERR(API_PRINT_PREFIX
                  "[%u] job %u can not be freed when in %u state \n",
                  handle->instance_board_id, task_id, current_status);
          goto failed;
      }
    }
  }
  LOG_ERR(API_PRINT_PREFIX "[%u] job %u is not on the board\n",
          handle->instance_board_id, task_id);
failed:
  TRACY_ZONE_END
  return LIB_ERROR;
}

void scheduler_close(JobBoardHandle* handle, bool unlink_on_close) {
  assert(handle != NULL);
  assert(_integrity_check(handle));

  LOG_INF(API_PRINT_PREFIX "[%s:%u] closing board\n", handle->alloc_params.path,
          handle->instance_board_id);

  job_board_close(handle);

  shm_close(handle, unlink_on_close);
}
