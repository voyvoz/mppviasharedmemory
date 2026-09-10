#include <assert.h>  // For assert
#include <errno.h>   // For errno
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>  // For waitpid
#include <unistd.h>    // For fork, getpid, sleep

#include "executor_api.h"
#include "scheduler_api.h"

#define TEST_BOARD_NAME "/test_wdt_reset_board"

shared_mem_allocator_init_params alloc_params = {
    .path = TEST_BOARD_NAME,
    .size_bytes = 4194304,  // 4MB
};

static void* executor_proc(void* args) {
  (void)args;

  printf("Executor: Starting...\n");

  // Give the scheduler a moment to create the board
  sleep(1);

  JobBoardHandle exec_handle = {.alloc_params = alloc_params};
  executor_init(&exec_handle);
  assert(exec_handle.is_valid && "Executor: Failed to initialize board");

  Job* claimed_task_that_will_time_out = executor_job_wait(&exec_handle);
  assert(claimed_task_that_will_time_out != NULL &&
         "Executor: Failed to claim task");
  assert(atomic_load(&claimed_task_that_will_time_out->state.status) ==
             TASK_STATUS_IN_PROGRESS &&
         "Executor: Job status not IN_PROGRESS");

  void* input = shm_obj_ptr_data(exec_handle.alloc_handle.base_addr,
                                 claimed_task_that_will_time_out->meta._params);
  printf("Executor: Claimed task %llu. Payload: message='%s'\n",
         (unsigned long long)claimed_task_that_will_time_out->meta.task_id,
         (char*)input);

  // Simulate doing some work that takes longer than the watchdog timeou
  // -> The task should be revoked
  sleep(WDT_DEF_SECONDS_INCREMENT + 2);

  // Prepare result
  uint32_t result_code = 0;
  char result_msg[] = "Job processed by executor!";

  // this should fail now
  int finish_result = executor_finish_job(
      &exec_handle, claimed_task_that_will_time_out, result_code, result_msg);
  assert(finish_result != 0 && "Executor: Finished timed out task?");

  // --- Now do the same again but reset the watchdog before time runs out!

  Job* claim_same_task_again = executor_job_wait(&exec_handle);
  assert(claim_same_task_again != NULL && "Executor: Failed to claim task");
  assert(atomic_load(&claim_same_task_again->state.status) ==
             TASK_STATUS_IN_PROGRESS &&
         "Executor: Job status not IN_PROGRESS");

  sleep(WDT_DEF_SECONDS_INCREMENT / 2);

  int wd_reset_result = executor_wd_reset(&exec_handle, claim_same_task_again);
  assert(wd_reset_result == 0 && "Executor: Watchdog reset failed");

  sleep(WDT_DEF_SECONDS_INCREMENT / 2);

  finish_result = executor_finish_job(
      &exec_handle, claimed_task_that_will_time_out, result_code, result_msg);
  assert(finish_result == 0 && "Executor: Failed to finish task");

  printf("Executor: Job %llu finished.\n",
         (unsigned long long)claimed_task_that_will_time_out->meta.task_id);

  executor_close(&exec_handle);
  printf("Executor: Exiting.\n");

  return 0;
}

void scheduler(pthread_t child_id) {
  printf("Scheduler: Starting...\n");

  JobBoardHandle sched_handle = {.alloc_params = alloc_params};
  scheduler_init(&sched_handle, 1);
  assert(sched_handle.is_valid && "Scheduler: Failed to initialize board");

  char* my_payload = "Hello from Scheduler!";

  shm_obj_t o = shm_malloc(&sched_handle.alloc_handle, strlen(my_payload));
  assert(o == 0 && "failed to shm malloc");
  void* addr = shm_obj_ptr_data(sched_handle.alloc_handle.base_addr, o);
  memcpy(addr, my_payload, strlen(my_payload) + 1);

  PostableJobTree pjt = {
      .job_id = 0,
      .input = o,
      .dependant_jobs_count = 0,
      .dependant_jobs = NULL,
  };
  uint64_t task_id = scheduler_post_job_tree(&sched_handle, &pjt);
  // uint64_t task_id = scheduler_post_task(&sched_handle, my_payload);
  assert(task_id != 0 && "Scheduler: Failed to post task");
  printf("Scheduler: Posted task %llu.\n", (unsigned long long)task_id);

  int wait_result = scheduler_job_wait(&sched_handle, task_id);
  assert(wait_result == 0 && "Scheduler: Failed to wait for task completion");

  printf("Scheduler: Job %llu completion confirmed.\n",
         (unsigned long long)task_id);

  // Check the final status and result (if applicable)
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
  // Cannot reliably dereference result_ptr here as it points to child's local
  // memory.
  printf("Scheduler: Verified task %llu finished state and result code %u.\n",
         (unsigned long long)task_id, completed_task->state.result_code);

  // Wait for the child
  pthread_join(child_id, NULL);

  scheduler_close(&sched_handle, 1);
  printf("Scheduler: Exiting.\n");
}

int main() {
  printf(
      "--- Running Watchdog Reset Scheduler-Executor Interaction Test ---\n");

  pthread_t child_id;
  int res = pthread_create(&child_id, NULL, executor_proc, NULL);
  if (res != 0) {
    fprintf(stderr, "pthread_create failed");
    exit(1);
  }

  scheduler(child_id);

  printf("--- Basic Scheduler-Executor Interaction Test Finished ☑️ ---\n");
  return 0;
}
