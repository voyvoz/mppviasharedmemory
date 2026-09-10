#include <assert.h>  // For assert
#include <errno.h>   // For errno
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>  // For waitpid
#include <unistd.h>    // For fork, getpid, sleep

#include "dispatcher_api.h"
#include "executor_api.h"
#include "scheduler_api.h"
#include "smm/allocator.h"

#define TEST_BOARD_NAME "/test_queue_basic"

char task_payload[] = "Hello from Dispatcher!";
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

  Job* claimed_task = NULL;
  shm_obj_t o = executor_queue_await_job(&exec_handle, &claimed_task);
  assert(o != INVALID_SHM_OBJ);
  assert(claimed_task != NULL && "Executor: Failed to claim task");
  assert(atomic_load(&claimed_task->state.status) == TASK_STATUS_IN_PROGRESS &&
         "Executor: Job status not IN_PROGRESS");

  shm_obj_t job_input = job_get_input(claimed_task);
  assert(job_input != INVALID_JOB_ID && "job_input should have been set!");
  void* job_input_ptr =
      shm_obj_ptr_data(exec_handle.alloc_handle.base_addr, job_input);
  assert(strcmp(task_payload, job_input_ptr) == 0 &&
         "Payload should be identical");

  void* input = shm_obj_ptr_data(exec_handle.alloc_handle.base_addr,
                                 claimed_task->meta._params);
  printf("[Executor] Claimed task %llu. Payload: message='%s'\n",
         (unsigned long long)claimed_task->meta.task_id, (char*)input);

  shm_obj_t obj =
      shm_malloc(&exec_handle.alloc_handle, strlen(task_payload) + 1);
  void* d = shm_obj_ptr_data(exec_handle.alloc_handle.base_addr, obj);
  memcpy(d, &result_msg, strlen(result_msg));

  int finish_result =
      executor_queue_finish_job(&exec_handle, o, LIB_SUCCESS, d);
  assert(finish_result == LIB_SUCCESS && "Executor: Failed to finish task");

  printf("[Executor] Job finished.\n");

  executor_close(&exec_handle);
  printf("[Executor] Exiting.\n");

  return NULL;
}

void dispatcher_proc(pthread_t child_id) {
  printf("[Dispatcher] Starting...\n");

  JobBoardHandle sched_handle = {.alloc_params = alloc_params};
  scheduler_init(&sched_handle, 1);
  assert(sched_handle.is_valid && "Dispatcher: Failed to initialize board");

  shm_obj_t shared_mem_obj =
      shm_malloc(&sched_handle.alloc_handle, strlen(task_payload) + 1);
  assert(shared_mem_obj != 0 && "Dispatcher: Malloc failed");
  void* d =
      shm_obj_ptr_data(sched_handle.alloc_handle.base_addr, shared_mem_obj);
  memcpy(d, &task_payload, strlen(task_payload) + 1);

  PostableJobTree pjt = {
      .input = shared_mem_obj,
  };
  DispatchResult dr = dispatch_job(&sched_handle, &pjt);
  assert(dr.job != INVALID_SHM_OBJ && "Dispatcher: Failed to post task");
  printf("[Dispatcher] Posted task using shm_obj_t %llu.\n", dr.job);

  Job* j = shm_obj_ptr_data(sched_handle.alloc_handle.base_addr, dr.job);
  assert(j->meta.task_id != INVALID_JOB_ID && "invalid meta task id");

  int wait_result = dispatch_await(&sched_handle, dr);
  assert(wait_result != LIB_SUCCESS &&
         "Dispatcher: Failed to wait for task completion");

  printf("[Dispatcher] Verified task %u was completetd as expected\n",
         j->meta.task_id);

  shm_free(&sched_handle.alloc_handle, shared_mem_obj);
  shm_free(&sched_handle.alloc_handle, dr.job);

  // Wait for the child
  pthread_join(child_id, NULL);

  scheduler_close(&sched_handle, 1);  // Unlink the board
  printf("[Dispatcher] Exiting.\n");
}

int main() {
  printf("--- Basic Dispatcher-Executor Interaction Test: Starting ---\n");

  pthread_t child_id;
  int res = pthread_create(&child_id, NULL, executor_proc, NULL);
  if (res != 0) {
    fprintf(stderr, "pthread_create failed");
    exit(1);
  }

  dispatcher_proc(child_id);

  printf(
      "--- Basic Dispatcher-Executor Interaction Test: Finished ---\n ->  ☑️ "
      "\n");
  return 0;
}
