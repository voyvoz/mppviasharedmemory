#include <errno.h>  // For errno
#include <stdbool.h>
#include <stdio.h>   // For perror, fprintf
#include <stdlib.h>  // For malloc, free
#include <string.h>  // For memset, strncpy

// POSIX specific includes for shared memory
#include <fcntl.h>     // For O_RDWR
#include <sys/mman.h>  // For mmap, munmap, shm_open, shm_unlink
#include <sys/stat.h>  // For S_IRUSR, S_IWUSR (permissions)
#include <unistd.h>    // For ftruncate, close, getpid

#include "executor_api.h"
#include "internal/debug_helper.h"
#include "internal/job_board.h"
#include "smm/allocator.h"

#define API_PRINT_PREFIX "[EXEC] "

int grab_queue_pair(JobBoardHandle* handle) {
  assert(handle != NULL);

  bool exp = false;
  for (size_t i = 0; i < QUEUE_PAIRS; i++) {
    if (!atomic_load(&handle->rb_state->_pairs[i].in_use)) {
      if (atomic_compare_exchange_strong(&handle->rb_state->_pairs[i].in_use,
                                         &exp, true)) {
        handle->rb_state->_pairs[i].owner_id = handle->instance_board_id;

        handle->ring_idx = i;
        handle->rb_state->_pairs[i].flags = 0;

        atomic_store(&handle->rb_state->_pairs[i].valid, true);

        LOG_INF(API_PRINT_PREFIX "[%u] grabbed RingPair: %zu\n",
                handle->instance_board_id, handle->ring_idx);

        return LIB_SUCCESS;
      }
    }
  }
  LOG_ERR(API_PRINT_PREFIX
          "[%u]"
          "grab_queue_pair failed to find pair\n",
          handle->instance_board_id);
  return LIB_ERROR;
}

// --- Executor API Implementation ---

bool executor_init(JobBoardHandle* jbh) {
  assert(jbh != NULL);

  TRACY_ZONE_START

  jbh->is_valid = false;

  bool is_new =
      shm_create_or_open(&jbh->alloc_handle, &jbh->alloc_params, false);
  // TODO err handling?
  int res = job_board_init(jbh, false);
  if (res != 0) {
    LOG_ERR(API_PRINT_PREFIX
            "[?%u]"
            "job_board_init failed with %i\n",
            jbh->instance_board_id, res);
    exit(1);
  }

  // Grab queuepair
  res = grab_queue_pair(jbh);
  if (res != 0) {
    LOG_ERR(API_PRINT_PREFIX
            "[?%u]"
            "grab_queue_pair failed with %i\n",
            jbh->instance_board_id, res);
    exit(1);
  }

  // Copy sane values for very simple integrity checks
  jbh->org_job_slots = shm_get_job_num(jbh);
  jbh->org_version = jbh->jb_state->version;

  TRACY_ZONE_END

  return is_new;
}

void executor_close(JobBoardHandle* handle) {
  assert(handle != NULL);

  LOG_INF(API_PRINT_PREFIX "[%s:%u] closing board connection\n",
          handle->alloc_params.path, handle->instance_board_id);

  shm_close(handle, 0);
}

// --- Board based API

Job* executor_job_wait(const JobBoardHandle* handle) {
  assert(handle != NULL);
  assert(handle->jb_state != NULL);
  assert(handle->tasks != NULL);
  assert(_integrity_check(handle));

  TRACY_ZONE_START

  LOG_INF(API_PRINT_PREFIX "[%u] scanning for job...\n",
          handle->instance_board_id);

  // Polling loop to find an available job
  while (1) {
    // print_board_state(handle);

    if (handle->jb_state->alive == false) {
      LOG_INF(API_PRINT_PREFIX "[%u] board closed\n",
              handle->instance_board_id);
      // board down so we quit
      TRACY_ZONE_END
      return NULL;
    }

    for (size_t i = 0; i < shm_get_job_num(handle); i++) {
    check_has_deps:
      for (int dep_jobs_idx = 0; dep_jobs_idx < MAX_DEPENDANT_JOBS;
           dep_jobs_idx++) {
        job_id_t dependant_job = handle->tasks[i].state.dep_ids[dep_jobs_idx];
        if (handle->tasks[i].state.dep_ids[dep_jobs_idx] == 0) {
          goto aquire_job;
        } else {
          // Check if the dependant job has been finished
          for (size_t j = 0; j < shm_get_job_num(handle); j++) {
            if (j == i) {
              continue;
            }
            if (handle->tasks[j].meta.task_id == dependant_job) {
              job_status_t job_State =
                  atomic_load(&handle->tasks[j].state.status);
              // Make sure that the job j has been finished
              switch (job_State) {
                case TASK_STATUS_FINISHED:
                  // j is finished so we can grab the job that has job j as a
                  // dependency
                  goto aquire_job;
                case TASK_STATUS_POSTED:
                  // j might have another dependency
                  i = j;
                  goto check_has_deps;
                // j might be in any other state that does not allow running
                // this job right now
                default:
                  break;
              }
            }
          }
          // NOTE this only works for single dependency jobs ... make this
          // generic!
          LOG_INF(API_PRINT_PREFIX
                  "[%u] unable to find finished dependant job with id %u.\n",
                  handle->instance_board_id, dependant_job);
          continue;
        }
      }

    aquire_job:
      job_status_t expected_status = TASK_STATUS_POSTED;
      // Claim posted task
      // CRITICAL SYNC SECTION
      if (atomic_compare_exchange_strong(&handle->tasks[i].state.status,
                                         &expected_status,
                                         TASK_STATUS_IN_PROGRESS)) {
        // No longer critical

        // TODO maybe update the whole cacheline atomicly instead of each value?
        // TODO this is not required to be atomic?
        // The guard is the previous compare exchange so no other process should
        // be able to be in here for the same task so we could just update the
        // rest of the data without using atomics?
        handle->tasks[i].state.executor_id = handle->instance_board_id;
        atomic_store(&handle->tasks[i].state.watchdog_timeout,
                     time(NULL) + WDT_DEF_SECONDS_INCREMENT);

        // map the shared mem input object if the are set
        if (handle->tasks[i].meta._params != 0) {
          shm_map_read(&handle->alloc_handle, handle->tasks[i].meta._params);
        }

        TRACY_ZONE_END
        return &handle->tasks[i];
      }
    }

    _sleep();
  }
}

int executor_wd_reset(const JobBoardHandle* handle, Job* task) {
  assert(handle != NULL);
  assert(task != NULL);
  assert(_integrity_check(handle));
  (void)handle;

  TRACY_ZONE_START

  atomic_store(&task->state.watchdog_timeout,
               time(NULL) + WDT_DEF_SECONDS_INCREMENT);

  TRACY_ZONE_END

  return LIB_SUCCESS;
}

int executor_finish_job(const JobBoardHandle* handle, Job* task,
                        uint32_t result_code, void* result_payload) {
  assert(handle != NULL);
  assert(task != NULL);
  assert(_integrity_check(handle));

  TRACY_ZONE_START

  int retval = LIB_SUCCESS;

  // Only allow finishing if current status is IN_PROGRESS
  job_status_t expected_status = TASK_STATUS_IN_PROGRESS;
  job_status_t transition = TASK_STATUS_TRANSITION;
  if (atomic_compare_exchange_strong(&task->state.status, &expected_status,
                                     transition)) {
    // Now we can savely edit the job
    task->state.result_code = result_code;
    task->meta.result_payload = result_payload;
    if (task->meta._params != INVALID_SHM_OBJ) {
      shm_free(&handle->alloc_handle, task->meta._params);
    }

    job_status_t result_status =
        result_code == LIB_SUCCESS ? TASK_STATUS_FINISHED : TASK_STATUS_CANCELD;

    LOG_INF(API_PRINT_PREFIX
            "[%u] Executor finished task %u with status %u and result code %u. "
            "Result: '%s'\n",
            handle->instance_board_id, task->meta.task_id, result_status,
            result_code, (char*)task->meta.result_payload);

    // Now set the job in a finished state
    if (!atomic_compare_exchange_strong(&task->state.status, &transition,
                                        result_status)) {
      LOG_ERR(API_PRINT_PREFIX
              "[%u] executor_finish_job error: Job %u not in expected "
              "TASK_STATUS_IN_PROGRESS state.\n",
              handle->instance_board_id, task->meta.task_id);
      retval = -1;
    }
  }

  TRACY_ZONE_END
  assert(_integrity_check(handle));
  return retval;
}

// --- Ring based API

// Idea v2
/*

The one ring
-> hat genau 16 slots für structs mit
  valid?
  exec_id?
  shm obj -> DoubleRing
  exec flags
  usage state? ??

Per Dispatcher API kann man dann nen Ring suchen zu dem ein Job am besten passt
sofern es mehrere gibt Sobald nen job submitted ist bekommt man nen nen exec
lokalen job token

Per Worker API

Man erstellt irgendwo im shared mem nen double Ring.
Ring Submission -> jeder entry ist eine jobdescription mit maximal 64byte
Es gibt nen 64byte header der aktuellen index im Ring anzeigt
-> der handle merkt sich wo man gerade steht
-> -> Dispatcher API muss prüfen ob man nen neuen Entry schreiben darf!

Der worker geht also in den Ring Submission und nimmt sich den nächsten job
-> return damit

Job wird erledigt

Worker steckt die ID in den completion Ring
-> was passiert wenn da kein Platz ist?

Dispatcher wartet auf X completions
-> wenn gefunden dann alten job mit der ID löschen und dann Eintrag in
completion löschen
*/

// TODO warum reuturn value und new_job?
shm_obj_t executor_queue_await_job(const JobBoardHandle* handle,
                                   Job** new_job) {
  assert(handle != NULL);

  TRACY_ZONE_START

  RingPair* rp = &handle->rb_state->_pairs[handle->ring_idx];
  if (!atomic_load(&rp->valid)) {
    LOG_ERR(API_PRINT_PREFIX
            "[%u]"
            "would use invalid RingPair",
            handle->instance_board_id);
    return INVALID_JOB_ID;
  }
  while (1) {
    shm_obj_t j_obj = ring_pop(&rp->doubleRing.sub);
    if (j_obj != INVALID_JOB_ID) {
      // Map the job itself
      if (shm_map_read(&handle->alloc_handle, j_obj) == NULL) {
        LOG_ERR(API_PRINT_PREFIX
                "[%u]"
                "executor_queue_await_job job shm_map_read failed\n",
                handle->instance_board_id);
        TRACY_ZONE_END
        return INVALID_JOB_ID;
      }

      //
      *new_job = shm_obj_ptr_data(handle->alloc_handle.base_addr, j_obj);
      // TODO does this need to be atomic?
      // should not need it as it can only be in a single queue and only be
      // popped once...
      (*new_job)->state.status = TASK_STATUS_IN_PROGRESS;

      // Map the job input if it exists
      shm_obj_t job_input = (*new_job)->meta._params;
      if (job_input != INVALID_SHM_OBJ) {
        if (shm_map_read(&handle->alloc_handle, job_input) == NULL) {
          LOG_ERR(API_PRINT_PREFIX
                  "[%u]"
                  "executor_queue_await_job job_input shm_map_read failed\n",
                  handle->instance_board_id);
          TRACY_ZONE_END
          return INVALID_JOB_ID;
        }
      }

      TRACY_ZONE_END
      return j_obj;
    }

    _sleep();
  }

  TRACY_ZONE_END
  return INVALID_JOB_ID;
}

int executor_queue_finish_job(const JobBoardHandle* handle, shm_obj_t job_obj,
                              uint32_t result_code, void* result_ptr) {
  assert(handle != NULL);
  assert(job_obj != INVALID_JOB_ID);

  TRACY_ZONE_START

  Job* j = shm_obj_ptr_data(handle->alloc_handle.base_addr, job_obj);
  int res = executor_finish_job(handle, j, result_code, result_ptr);
  if (res != LIB_SUCCESS) {
    return res;
  }

  // Get our ring
  RingPair* rp = &handle->rb_state->_pairs[handle->ring_idx];
  if (!atomic_load(&rp->valid)) {
    TRACY_ZONE_END
    // TODO log?
    return LIB_ERROR;
  }

  // put into completion ring
  while (1) {
    if (ring_put(&rp->doubleRing.comp, job_obj) == LIB_SUCCESS) {
      shm_free(&handle->alloc_handle, job_obj);

      TRACY_ZONE_END
      return LIB_SUCCESS;
    }
    _sleep();
  }
}
