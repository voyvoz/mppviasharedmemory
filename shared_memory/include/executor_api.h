#ifndef EXECUTOR_API_H
#define EXECUTOR_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "internal/utils.h"
#include "smm/allocator.h"

// --- Management

/**
 * @brief Initializes the executor's connection to a job board.
 *
 * Connects to a named shared memory job board, registers the executor's PID,
 * and acquires a unique executor ID.
 *
 * @param name The name/path of the job board shared memory segment.
 * @return
 */
bool executor_init(JobBoardHandle* handle);

/**
 * @brief Cleans up executor resources.
 *
 * Unmaps the shared memory, frees the handle, and removes the executor's PID
 * from the board's jb_state.
 *
 * @param handle A pointer to the JobBoardHandle struct.
 */
void executor_close(JobBoardHandle* handle);

// --- Board based API 

/**
 * @brief Waits until a job is posted and claims it.
 *
 * This function blocks until a task with status `TASK_STATUS_POSTED` is found.
 * It then attempts to atomically claim the task by setting its `executor_id`
 * and changing its status to `TASK_STATUS_IN_PROGRESS`.
 *
 * @param handle A pointer to the initialized JobBoardHandle.
 * @return A pointer to the claimed Job structure on success, or NULL on
 * failure. The returned Job pointer is valid as long as the JobBoardHandle is
 * open.
 */
Job* executor_job_wait(const JobBoardHandle* handle);

/**
 * @brief Resets the watchdog timeout to prevent task reassignment
 *
 * TODO
 *
 * @param handle A pointer to the initialized JobBoardHandle.
 * @return A pointer to the claimed Job structure on which the reset should
 * happen.
 */
int executor_wd_reset(const JobBoardHandle* handle, Job* task);

/**
 * @brief Marks a task as finished and optionally sets a result.
 *
 * @param handle A pointer to the initialized JobBoardHandle.
 * @param task The task to mark as finished.
 * @param result_code 0 for success. Any other value will set the job as
 * canceld. result_code will be preserved in the executor data result code.
 * @param result_ptr A pointer to the result data.
 * @return 0 on success, -1 on failure.
 */
int executor_finish_job(const JobBoardHandle* handle, Job* task,
                        uint32_t result_code, void* result_ptr);


// --- Ring based API

shm_obj_t executor_queue_await_job(const JobBoardHandle* handle, Job** new_job);
int executor_queue_finish_job(const JobBoardHandle* handle, shm_obj_t job,
                        uint32_t result_code, void* result_ptr);


#ifdef __cplusplus
}
#endif

#endif  // EXECUTOR_API_H
