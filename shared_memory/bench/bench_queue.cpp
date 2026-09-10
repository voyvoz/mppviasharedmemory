/*
===------------------------------------------===

Sources:
https://max0x7ba.github.io/atomic_queue/html/benchmarks.html
https://max0x7ba.github.io/atomic_queue/

===------------------------------------------===
*/

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>

#include "dispatcher_api.h"
#include "executor_api.h"
#include "internal/bench_utils.h"
#include "scheduler_api.h"

#define BENCH_BOARD_NAME "/shm_bench_queue"
#define JOBS 1000000

char task_payload[] = "Hello from Scheduler!";

// Parameters
DECLARE_BENCH_PARAM(param_jobs, JOBS, "How many jobs should be processed");
DECLARE_BENCH_PARAM(param_exec_thread_count, 1,
                    "How many executor threads should be spawned");
DECLARE_BENCH_PARAM(param_sched_thread_count, 1,
                    "How many scheduler threads should be spawned");
DECLARE_BENCH_PARAM(param_sched_batch_size, 4,
                    "How many consecutive jobs are treated as a batch");
bool param_use_monitor = true;

//-- internals
size_t posted_jobs_counter = 0;
size_t finished_jobs_counter = 0;
atomic_bool scheduler_ready = false;

FILE* log_file;
_Atomic(uint8_t) thread_id = 0;
shared_mem_allocator_init_params params = {.path = BENCH_BOARD_NAME,
                                           .size_bytes = 4194304};

static void* scheduler_func(void* args) {
  (void)args;

  uint8_t local_thread_id = atomic_fetch_add(&thread_id, 1);

  JobBoardHandle sched_handle = {.alloc_params = params};
  bool is_creator = thread_id == 1;
  scheduler_init(&sched_handle, is_creator);

  atomic_exchange(&scheduler_ready, true);

  size_t job_obj_buffer_size = param_sched_batch_size;
  shm_obj_t shared_mem_objs[job_obj_buffer_size];

  while (finished_jobs_counter < param_jobs) {
    size_t posted_task_batch_size = 0;
    DispatchResult tasks[job_obj_buffer_size];

    // Create the job/s
    for (size_t i = 0;
         i < job_obj_buffer_size && posted_jobs_counter < param_jobs; i++) {
      // TODO compare_exchange den posted_jobs_counter?
      posted_jobs_counter += 1;
      posted_task_batch_size += 1;

      // alloc input param object
      shared_mem_objs[i] =
          shm_malloc(&sched_handle.alloc_handle, strlen(task_payload) + 1);
      void* d = shm_obj_ptr_data(sched_handle.alloc_handle.base_addr,
                                 shared_mem_objs[i]);
      memcpy(d, &task_payload, strlen(task_payload) + 1);
      PostableJobTree pjt = {
          .input = shared_mem_objs[i],
      };

      // Now dispatch it
      DispatchResult res = dispatch_job(&sched_handle, &pjt);
      assert(res.job != INVALID_SHM_OBJ);
      tasks[i] = res;
    }

    // Wait for the job to be finished
    for (size_t i = 0; i < posted_task_batch_size; i++) {
      // Wait for it
      shm_obj_t o = dispatch_await(&sched_handle, tasks[i]);
      if (o == INVALID_SHM_OBJ) {
        printf(
            "[%u] "
            "scheduler_job_wait == INVALID_SHM_OBJ\n",
            local_thread_id);
        exit(1);
      }
      assert(o == tasks[i].job);

      // Free resources
      shm_free(&sched_handle.alloc_handle,
               shared_mem_objs[i]);             // the input param
      shm_free(&sched_handle.alloc_handle, o);  // the job

      // Increment counter
      finished_jobs_counter += 1;
    }
  }

  scheduler_close(&sched_handle, is_creator);

  return NULL;
}

static void* exec_func(void* args) {
  (void)args;

  // Scheduler needs a bit of time for setup
  while (!atomic_load(&scheduler_ready)) {
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 10000000};  // 10ms
    nanosleep(&ts, NULL);
  }

  uint8_t local_thread_id = atomic_fetch_add(&thread_id, 1);

  JobBoardHandle exec_handle = {.alloc_params = params};
  executor_init(&exec_handle);

  Job* claimed_task;
  size_t local_jobs_finished = 0;
  while (finished_jobs_counter < param_jobs) {
    // Get a job
    shm_obj_t o = executor_queue_await_job(&exec_handle, &claimed_task);
    if (o == INVALID_SHM_OBJ) {
      _sleep();
      continue;
    }

    // TODO do something with the job?

    // Complete
    if (executor_queue_finish_job(&exec_handle, o, 0, NULL) != 0) {
      printf(
          "[%u] "
          "executor_queue_finish_job != 0)\n",
          local_thread_id);
      exit(1);
    }

    local_jobs_finished += 1;
  }

  return NULL;
}

bool monitor_stop = false;
static void* monitor(void* args) {
  auto start = std::chrono::high_resolution_clock::now();

  uint64_t last = 0;

  printf("Params: Jobs=%zu  Sched.=%zu   Ex.=%zu  Batch=%zu\n", param_jobs,
         param_sched_thread_count, param_exec_thread_count,
         param_sched_batch_size);

  while (!monitor_stop) {
    struct timespec ts = {.tv_sec = 0, .tv_nsec = 10000000};  // 10ms
    nanosleep(&ts, NULL);

    size_t current_posted = posted_jobs_counter;
    size_t current = finished_jobs_counter;

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    uint64_t time_per_event =
        current == 0 ? 0 : (double)(current * 1000000000) / (double)elapsed;
    auto time_spent = (double)((double)elapsed / 1000000000);

    printf("\033[2K\r");
    printf(
        "Runtime: %f s | ~%lld finished jobs/sec. | finished %lu / %lu posted",
        time_spent, time_per_event, current, current_posted);
    fflush(stdout);

    last = current;
  }

  return NULL;
}

int main() {
  printf("Running RingBearer benchmark...\n\n");

  // TODO parse args?

  pthread_t child_ids[param_exec_thread_count + param_sched_thread_count];

  shm_unlink(BENCH_BOARD_NAME);

  // Start the threads
  for (size_t i = 0; i < param_sched_thread_count; i++) {
    sleep(1);
    int res = pthread_create(&child_ids[i], NULL, scheduler_func, NULL);
    if (res != 0) {
      fprintf(stderr, "pthread_create failed");
      exit(1);
    }
  }
  for (size_t i = param_sched_thread_count;
       i < param_exec_thread_count + param_sched_thread_count; i++) {
    int res = pthread_create(&child_ids[i], NULL, exec_func, NULL);
    if (res != 0) {
      fprintf(stderr, "pthread_create failed");
      exit(1);
    }
  }

  // Start the monitor thread
  pthread_t mon_id;
  if (param_use_monitor) {
    int res = pthread_create(&mon_id, NULL, monitor, NULL);
    if (res != 0) {
      fprintf(stderr, "pthread_create failed for monitor");
      exit(1);
    }
  }

  // Wait for them to be finished
  for (size_t i = 0; i < param_sched_thread_count + param_exec_thread_count;
       i++) {
    int res = pthread_join(child_ids[i], NULL);
    if (res != 0) {
      fprintf(stderr, "pthread_join failed");
      exit(1);
    }
  }

  monitor_stop = true;
  if (param_use_monitor) {
    int res = pthread_join(mon_id, NULL);
    if (res != 0) {
      fprintf(stderr, "pthread_join failed for monitor");
      exit(1);
    }
  }

  printf("\nFinished\n");

  return 0;
}