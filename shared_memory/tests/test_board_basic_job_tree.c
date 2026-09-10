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
#include "internal/debug_helper.h"
#include "scheduler_api.h"

#define TEST_BOARD_NAME "/test_job_tree_board"
#define NUM_TEST_TASKS 7

shared_mem_allocator_init_params alloc_params = {
    .path = TEST_BOARD_NAME,
    .size_bytes = 4194304,  // 4MB
};

/*

This test verifies that a tree can be posted and executed with correct
constraints:

We post this job tree:
      J1
   J2      J3
J4 J5 J6   J7

J3,J4,J5,J6 should be executed first. Order is insignificant.
Next up is J2 after which J1 should be run.

We dont check if results are handeld correctly in this test.

NOTE this test only checks that it works. Currently no error or edge cases are
tested!

*/

char* job_1_payload = "<job-1>";
char* job_2_payload = "<job-2>";
char* job_3_payload = "<job-3>";
char* job_4_payload = "<job-4>";
char* job_5_payload = "<job-5>";
char* job_6_payload = "<job-6>";
char* job_7_payload = "<job-7>";
const size_t num_jobs = NUM_TEST_TASKS;

shm_obj_t create_job_paylaod(shared_mem_allocator_handle* handle, char* d) {
  shm_obj_t o = shm_malloc(handle, strlen(d));
  assert(o != 0 && "Failed to shm malloc");
  void* addr = shm_obj_ptr_data(handle->base_addr, o);
  memcpy(addr, d, strlen(d) + 1);
  return 0;
}

// --- Parent Process: Scheduler ---
void scheduler(pthread_t child_id) {
  printf("Scheduler: Starting...\n");

  JobBoardHandle sched_handle = {.alloc_params = alloc_params};
  scheduler_init(&sched_handle, 1);
  assert(sched_handle.is_valid && "Scheduler: Failed to initialize board");
  assert(sched_handle.jb_state->job_slots >= NUM_TEST_TASKS &&
         "Failed to setup correct task count");

  PostableJobTree job_7 = {
      // No dependencies
      .dependant_jobs = NULL,
      .dependant_jobs_count = 0,
      .input = create_job_paylaod(&sched_handle.alloc_handle, job_7_payload),
  };
  PostableJobTree job_6 = {
      // No dependencies
      .dependant_jobs = NULL,
      .dependant_jobs_count = 0,
      .input = create_job_paylaod(&sched_handle.alloc_handle, job_6_payload),
  };
  PostableJobTree job_5 = {
      // No dependencies
      .dependant_jobs = NULL,
      .dependant_jobs_count = 0,
      .input = create_job_paylaod(&sched_handle.alloc_handle, job_5_payload),
  };
  PostableJobTree job_4 = {
      // No dependencies
      .dependant_jobs = NULL,
      .dependant_jobs_count = 0,
      .input = create_job_paylaod(&sched_handle.alloc_handle, job_4_payload),
  };
  PostableJobTree* job_3_deps[1] = {&job_7};
  PostableJobTree job_3 = {
      // No dependencies
      .dependant_jobs = job_3_deps,
      .dependant_jobs_count = 1,
      .input = create_job_paylaod(&sched_handle.alloc_handle, job_3_payload),
  };
  PostableJobTree* job_2_deps[3] = {&job_4, &job_5, &job_6};
  PostableJobTree job_2 = {
      // Job 2 depends on Job 4, Job 5 and Job 6
      .dependant_jobs = job_2_deps,
      .dependant_jobs_count = 3,
      .input = create_job_paylaod(&sched_handle.alloc_handle, job_2_payload),
  };
  PostableJobTree* job_1_deps[2] = {&job_2, &job_3};
  PostableJobTree job_1 = {
      .dependant_jobs = job_1_deps,
      .dependant_jobs_count = 2,
      .input = create_job_paylaod(&sched_handle.alloc_handle, job_1_payload),
  };

  uint64_t root_task_id = scheduler_post_job_tree(&sched_handle, &job_1);
  assert(root_task_id != 0 && "Scheduler: Failed to post task");
  printf("Scheduler: Posted job tree. Root is: %llu.\n", root_task_id);

  print_board_state_max_jobs(&sched_handle, NUM_TEST_TASKS + 1);

  int wait_result = scheduler_job_wait(&sched_handle, root_task_id);
  assert(wait_result == 0 && "Scheduler: Failed to wait for task completion");

  printf("Scheduler: Job %llu completion confirmed.\n", root_task_id);

  // Wait for the child
  pthread_join(child_id, NULL);

  scheduler_close(&sched_handle, 1);  // Unlink the board
  printf("Scheduler: Exiting.\n");
}

// --- Child Process: Executor ---
static void* executor_proc(void* args) {
  (void)args;

  printf("Executor: Starting...\n");

  // Give the scheduler a moment to create the board
  sleep(1);

  // ---  Setup finished

  JobBoardHandle exec_handle = {.alloc_params = alloc_params};
  executor_init(&exec_handle);
  assert(exec_handle.is_valid && "Executor: Failed to initialize board");

  for (size_t finished_jobs = 0; finished_jobs < num_jobs; finished_jobs++) {
    Job* claimed_task = executor_job_wait(&exec_handle);
    assert(claimed_task != NULL && "Executor: Failed to claim task");
    assert(atomic_load(&claimed_task->state.status) ==
               TASK_STATUS_IN_PROGRESS &&
           "Executor: Job status not IN_PROGRESS");

    // Make sure that first job has no dependant jobs
    if (finished_jobs == 0) {
      assert(claimed_task->state.dep_ids[0] == 0 &&
             "Executor: first job cant have a dependant job");
    }

    print_job(&exec_handle, 0, claimed_task);

    void* input = shm_obj_ptr_data(exec_handle.alloc_handle.base_addr,
                                   claimed_task->meta._params);
    // TODO this does not work lol
    printf("Executor: Claimed task %llu. Payload: message='%s'\n",
           (unsigned long long)claimed_task->meta.task_id, (char*)input);

    // Prepare result
    uint32_t result_code = 0;
    char result_msg[] = "Job ";

    int finish_result = executor_finish_job(&exec_handle, claimed_task,
                                            result_code, result_msg);
    assert(finish_result == 0 && "Executor: Failed to finish task");

    // print_board_state(exec_handle);

    printf("Executor: Job %llu finished. (%zu/%zu)\n",
           (unsigned long long)claimed_task->meta.task_id, finished_jobs + 1,
           num_jobs);
  }

  executor_close(&exec_handle);
  printf("Executor: Exiting.\n");

  return 0;
}

int main() {
  printf("--- Running Job Tree Scheduler-Executor Interaction Test ---\n");

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
