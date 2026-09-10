#ifndef SCHEDULER_API_H
#define SCHEDULER_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "internal/utils.h"
#include "smm/allocator.h"

/**
 * @brief Initializes the scheduler's connection to a job board.
 *
 * This function connects to a named shared memory job board. If `creator` is
 * true, it attempts to create and initialize the board (clearing old resources
 * if they exist). Otherwise, it connects to an existing board.
 *
 * @param creator If true, forces creation and initialization of the board.
 * @return A pointer to a JobBoardHandle struct on success, or NULL on failure.
 */
void scheduler_init(JobBoardHandle* handle, int creator);

/**
 * @brief Posts a new tree of jobs to the job board.
 *
 * Traverses the given job tree and posts all containing jobs.
 *
 * @param handle A pointer to the initialized JobBoardHandle.
 * @param payload_msg A tree of jobs that is interlinked and can be followed.
 * This parameter acts as the root node of the tree.
 * @return The task_id of the posted root job on success, or 0 on failure.
 */
int64_t scheduler_post_job_tree(const JobBoardHandle* handle,
                                PostableJobTree* job_tree_root);

/**
 * @brief
 *
 * @param handle A pointer to the initialized JobBoardHandle.
 * @param payload_msg A tree of jobs that is interlinked and can be followed.
 * This parameter acts as the root node of the tree.
 * @return The task_id of the posted root job on success, or 0 on failure.
 */
int scheduler_job_state(const JobBoardHandle* handle, int64_t task_id);

/**
 * @brief Waits for a specific task to be finished by an executor.
 *
 * This function blocks until the task with the given `task_id` has its
 * status set to TASK_STATUS_FINISHED.
 *
 * @param handle A pointer to the initialized JobBoardHandle.
 * @param task_id The ID of the task to wait for.
 * @return 0 on success (task finished), -1 on error (e.g., task_id not found or
 * invalid handle).
 */
int scheduler_job_wait(const JobBoardHandle* handle, int64_t task_id);

int scheduler_job_free(const JobBoardHandle* handle, int64_t task_id);

/**
 * @brief Cleans up scheduler resources and optionally unlinks the shared
 * memory.
 *
 * Unmaps the shared memory and frees the handle. If `unlink_on_close` is true,
 * and this process was the creator, it will attempt to remove the shared memory
 * object from the system.
 *
 * @param handle A pointer to the JobBoardHandle struct.
 * @param unlink_on_close If true, attempts to unlink the shared memory object.
 */
void scheduler_close(JobBoardHandle* handle, bool unlink_on_close);

#ifdef __cplusplus
}
#endif

#endif  // SCHEDULER_API_H
