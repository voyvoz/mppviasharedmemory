#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <assert.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <sys/mman.h>

#include "logging.h"
#include "profiler.h"
#include "smm/allocator.h"

// Define CACHE_LINE_SIZE for alignment
#ifndef CACHE_LINE_SIZE
// On M Chips L2 and SLC are 128... does that impact anything here?
#define CACHE_LINE_SIZE 64
#endif

#define MAX_TASK_PAYLOAD_SIZE 1020
#define MAX_RESULT_PAYLOAD_SIZE 1020
#define JBM_VALID_MAGIC 0xEFFEAB0A
#define JBM_VALID_MAGIC_2 0xAB0AEFFE

#define MAX_DEPENDANT_JOBS 5

#define MAX_JOB_NUM UINT32_MAX

// CXL 2.0 max hosts for a shared memory pool
#define MAX_CONCURRENT_HOSTS 16

// Watchdog
#define WDT_DEF_SECONDS_INCREMENT 10

#include "types.h"

// --- Job Statuses ---
typedef enum {
  TASK_STATUS_INVALID = 0,
  TASK_STATUS_POSTED = 1,
  TASK_STATUS_IN_PROGRESS = 2,
  TASK_STATUS_FINISHED = 3,
  TASK_STATUS_CANCELD = 4,
  TASK_STATUS_POST_IN_PROGRESS = 5,
  TASK_STATUS_TRANSITION = 6,
} TaskStatus;

// ---------------------------------
// --- Error handling            ---
// ---------------------------------

// Inspired by https://docs.gtk.org/glib/struct.Error.html

typedef struct {
  err_t err_code;
  char* msg;
} LibErr;

LibErr* new_err(err_t code, char* message, ...);

// -----------------------
// --- Board Structure ---
// -----------------------
// 16KiB    Shared Alloc state
// 16KiB    Job board state (128b) + Job List(128b * 127)
// ...      Allocateable Memory

/*
Optimization ideas:
The first page should only contain root nodes of job trees.
All other jobs should be on the same page on other pages?
*/

// --- 1. Metadata Structure ---

typedef struct {
  uint32_t _start_magic;

  uint16_t version;    // Indicates the version used to create this board
  uint16_t job_slots;  // Indicates how many Job slots are available
  _Atomic(uint32_t) active_slots;  // Counter for how many job slots are used
  _Atomic(uint32_t) next_task_id;  // Counter for assigning unique task IDs
  _Atomic(uint32_t)
      next_instance_id;  // Counter for assigning unique executor IDs

  bool alive;  // Indicates wether this board is in an actve state

  char _pad[103];

  uint32_t _end_magic;
} JobBoardState __attribute__((aligned(CACHE_LINE_SIZE)));
static_assert(sizeof(JobBoardState) == 2 * CACHE_LINE_SIZE,
              "sizeof(JobBoardState) should be two cache lines");

// TODO the size requirement makes no sense?
// -> this could be any size?

// --- 2. Internal Job Data Structures ---

// Scheduler's view of the task
// TODO does it make sense to make this the size of a cacheline?
typedef struct {
  // Sync Point for posting a task
  _Atomic(job_id_t) task_id;
  _Atomic(uint32_t)
      dep_jobs_num;  // Only set for a root job, used to track how many job
                     // slots are to be released once finished

  void* result_payload;

  shm_obj_t _params;  // Die App weiß dann schon was in/out ist

  char _pad[33];
} JobMetaData;
static_assert(sizeof(JobMetaData) == CACHE_LINE_SIZE,
              "sizeof(JobMetaData) should be a single cache line");

// Executor's view of the task
//
typedef struct {
  // Sync Point for grabbing a task
  _Atomic(job_status_t) status;
  _Atomic(time_t)
      watchdog_timeout;  // While an executor is working on a task they need to
                         // restart this timer peridically to ensure that
                         // crashes can be detected. Upon timeout the scheduler
                         // may resubmit this task.

  uint32_t executor_id;  // Initially 0, once an executor grabs it its the id of
                         // that executor.

  uint32_t result_code;  // Result code from the executor

  job_id_t
      dep_ids[MAX_DEPENDANT_JOBS];  // Dependencies that need to be finished
                                    // before the job can be started

  char _pad[18];
} JobState __attribute__((aligned(CACHE_LINE_SIZE)));
// As all builds on the assumption that the data is cacheline aligned we assert
// that it shall be so
static_assert(sizeof(JobState) == CACHE_LINE_SIZE,
              "sizeof(JobState) should be a single cache line");

// The board contains an array of these Tasks, initilaized during creation
// The complete Job structure within the shared memory board
typedef struct {
  JobMetaData meta;
  JobState state;
} Job;
static_assert(sizeof(Job) == 2 * CACHE_LINE_SIZE,
              "sizeof(Job) should be two cache lines");

// The PostableJobTree is used to setup a job chain which will be posted by a
// scheduler
typedef struct PostableJobTree {
  // -- These fields must be set by the calling function
  // The actual payload that represents the job that should be executed
  shm_obj_t input;
  // Pointer to array of dependant jobs of this jobs
  struct PostableJobTree** dependant_jobs;
  // How many dependant jobs in the dependant_jobs array
  job_deps_t dependant_jobs_count;

  // -- Internal fields that will be overwritten when posted
  // Once posted successfully this will be the job id for this payload
  job_id_t job_id;
  size_t group_size;
} PostableJobTree;

// Represents a chain of Contexts in which we iterate dependant jobs
typedef struct PJTTraversalContext {
  struct PJTTraversalContext* prev;
  PostableJobTree* e;
  job_deps_t idx;
} PJTTraversalContext;

// This struct holds pointer to the state structure on the shared memory.
// User the is_valid flag to check if it is considered valid for use.
typedef struct {
  // --- Allocator specific members
  shared_mem_allocator_init_params alloc_params;
  // Pointer into shared memory -> location of state data of the allocator
  shared_mem_allocator_handle alloc_handle;

  // --- JobBoard specific members
  // Pointer into shared_memory -> location of state of the job board
  JobBoardState* jb_state;  // Pointer to the jb_state in shared memory
  // Identifier for this process on the board
  // Used to identify which process is doing what on the board
  // Cant use pids for that as they are only unique per host
  // TODO we could combine host_name and pid ?
  // TODO We could use a BitVec as we can only have at most 16 concurrent
  // connections That could then also be used for reference counting and so on
  uint32_t instance_board_id;
  // TODO these are inside the job board jb_state so why are they here as well?
  Job* tasks;    // Pointer to the array of tasks in shared memory
  int is_owner;  // Flag to indicate if this process created the segment
  bool is_valid;

  // --- RB
  struct RingBearer* rb_state;
  size_t ring_idx;  // This handles owned ring or 0

  // -- Integrity
  uint16_t org_version;
  uint16_t org_job_slots;
} JobBoardHandle;

// ------------------------
// --- Queue Structures ---
// ------------------------

#include "ring.h"

typedef struct DoubleRing {
  ShmObjRing sub;
  ShmObjRing comp;
} DoubleRing;

typedef struct RingPair {
  _Atomic(bool) in_use;
  uint32_t owner_id;  // Executor id
  uint64_t flags;

  DoubleRing doubleRing;

  _Atomic(bool) valid;
  char _pad[26];
} RingPair __attribute__((aligned(CACHE_LINE_SIZE)));
// static_assert(sizeof(RingPair) == CACHE_LINE_SIZE , "sizeof(RingPair) must be
// one PCIe TX size");

// This must not match exactly
// though... could be more than
// 1 exec per host!
#define QUEUE_PAIRS MAX_CONCURRENT_HOSTS

// The ring bearer struct holds meta data information about active queue pairs
// and other related data
typedef struct RingBearer {
  // START MAGIC
  uint32_t _start_magic;

  // Counter for assigning unique handle IDs
  _Atomic(uint32_t) next_handle_id;
  // Counter for assigning unique job IDs
  _Atomic(uint32_t) next_job_id;

  RingPair _pairs[QUEUE_PAIRS];

  // END MAGIC
  uint32_t _end_magic;
} RingBearer;
static_assert(sizeof(RingBearer) <= FRAME_SIZE,
              "RingBearer must be at most FRAME_SIZE");

// The DispatchableJob
typedef struct DispatchaBatch {
  Job** j;
  size_t counter;
  // exec id? -> if found submit all to that queue
  // Other
} DispatchaBatch;

typedef struct {
  shm_obj_t job;
  size_t pair_idx;
} DispatchResult;

// ---------------------------------
// --- Helper  ---
// ---------------------------------

void _mprotect(void* p, size_t s, int f);

void _sleep();

bool _integrity_check(const JobBoardHandle* jbh);

shm_obj_t job_get_input(Job* j);

// ---------------------------------
// --- API Helper  ---
// ---------------------------------

JobBoardHandle create_empty_handle();
PostableJobTree create_empty_job_tree();

#ifdef __cplusplus
}
#endif

#endif  // COMMON_H
