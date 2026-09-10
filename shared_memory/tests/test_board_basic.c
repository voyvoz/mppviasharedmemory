#include <assert.h>  // For assert
#include <errno.h>   // For errno
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>  // For waitpid
#include <sys/wait.h>
#include <unistd.h>  // For fork, getpid, sleep

#include "executor_api.h"
#include "scheduler_api.h"
#include "smm/allocator.h"

#define TEST_BOARD_NAME "/test_1_basic_interaction_board"

char task_payload[] = "Hello from Scheduler!";
char result_msg[] = "Job processed by executor!";
shared_mem_allocator_init_params alloc_params = {
    .path = TEST_BOARD_NAME,
    .size_bytes = 4194304,  // 4MB
};

static void* executor_proc(void* args) {
  (void)args;

  printf("[Executor] Starting...\n");

  // Give the scheduler a moment to create the board
  sleep(1);

  JobBoardHandle exec_handle = {.alloc_params = alloc_params};
  executor_init(&exec_handle);
  assert(exec_handle.is_valid && "Executor: Failed to initialize board");

  Job* claimed_task = executor_job_wait(&exec_handle);
  assert(claimed_task != NULL && "Executor: Failed to claim task");
  assert(atomic_load(&claimed_task->state.status) == TASK_STATUS_IN_PROGRESS &&
         "Executor: Job status not IN_PROGRESS");
  shm_obj_t job_input = job_get_input(claimed_task);
  void* job_input_ptr =
      shm_obj_ptr_data(exec_handle.alloc_handle.base_addr, job_input);
  assert(strcmp(task_payload, job_input_ptr) == 0 &&
         "Palyoad should be identical");

  void* input = shm_obj_ptr_data(exec_handle.alloc_handle.base_addr,
                                 claimed_task->meta._params);
  printf("[Executor] Claimed task %llu. Payload: message='%s'\n",
         (unsigned long long)claimed_task->meta.task_id, (char*)input);

  uint32_t result_code = 0;
  shm_obj_t obj =
      shm_malloc(&exec_handle.alloc_handle, strlen(task_payload) + 1);
  void* d = shm_obj_ptr_data(exec_handle.alloc_handle.base_addr, obj);
  memcpy(d, &result_msg, strlen(result_msg));

  int finish_result =
      executor_finish_job(&exec_handle, claimed_task, result_code, d);
  assert(finish_result == 0 && "Executor: Failed to finish task");

  printf("[Executor] Job %llu finished.\n",
         (unsigned long long)claimed_task->meta.task_id);

  executor_close(&exec_handle);
  printf("[Executor] Exiting.\n");

  return 0;
}

void dispatcher_proc(pthread_t child_id) {
  printf("[Scheduler] Starting...\n");

  JobBoardHandle sched_handle = {.alloc_params = alloc_params};
  scheduler_init(&sched_handle, 1);
  assert(sched_handle.is_valid && "Scheduler: Failed to initialize board");
  // print_board_state(&sched_handle);

  shm_obj_t shared_mem_obj =
      shm_malloc(&sched_handle.alloc_handle, strlen(task_payload) + 1);
  assert(shared_mem_obj != 0 && "Scheduler: Malloc failed");
  void* d =
      shm_obj_ptr_data(sched_handle.alloc_handle.base_addr, shared_mem_obj);
  memcpy(d, &task_payload, strlen(task_payload) + 1);

  PostableJobTree pjt = {
      .job_id = 0,
      .input = shared_mem_obj,
      .dependant_jobs_count = 0,
      .dependant_jobs = NULL,
  };
  uint64_t task_id = scheduler_post_job_tree(&sched_handle, &pjt);
  assert(task_id != 0 && "Scheduler: Failed to post task");
  printf("[Scheduler] Posted task %llu.\n", (unsigned long long)task_id);
  // print_board_state(&sched_handle);

  int wait_result = scheduler_job_wait(&sched_handle, task_id);
  assert(wait_result == 0 && "Scheduler: Failed to wait for task completion");
  // print_board_state(&sched_handle);

  shm_free(&sched_handle.alloc_handle, shared_mem_obj);

  // Check the final status and result
  Job* completed_task = NULL;
  for (size_t i = 0; i < shm_get_job_num(&sched_handle); ++i) {
    if (sched_handle.tasks[i].meta.task_id == task_id) {
      completed_task = &sched_handle.tasks[i];
      break;
    }
  }
  assert(completed_task != NULL &&
         "Scheduler: Completed task not found in board");
  assert(atomic_load(&completed_task->state.status) == TASK_STATUS_FINISHED &&
         "Scheduler: Final task status not FINISHED");
  assert(completed_task->state.result_code == 0 &&
         "Scheduler: Incorrect result code");
  assert(strcmp(result_msg, (char*)completed_task->meta.result_payload) == 0 &&
         "Result msg is not correct");

  printf("[Scheduler] Verified task %llu was completetd as expected\n",
         task_id);

  // Wait for the child
  pthread_join(child_id, NULL);

  scheduler_close(&sched_handle, 1);  // Unlink the board
  printf("[Scheduler] Exiting.\n");
}

int main() {
  printf("--- Basic Scheduler-Executor Interaction Test: Starting ---\n");

  pthread_t child_id;
  int res = pthread_create(&child_id, NULL, executor_proc, NULL);
  if (res != 0) {
    fprintf(stderr, "pthread_create failed");
    exit(1);
  }

  dispatcher_proc(child_id);

  printf("--- Basic Scheduler-Executor Interaction Test: Finished ☑️ ---\n");
  return 0;
}
