#include "internal/job_board.h"

#include <stdio.h>

#include "internal/common.h"
#include "internal/debug_helper.h"
#include "smm/allocator.h"

int job_board_init(JobBoardHandle* handle, bool should_create) {
  TRACY_ZONE_START

  // TODO this is not guarded against concurrent init!!!

  // -- JobBoad
  shm_obj_t jbm_obj = handle->alloc_handle.base_addr->user_data1;
  if (jbm_obj == INVALID_SHM_OBJ || should_create) {
    // Allocate space on the board to place our data structures
    jbm_obj = shm_malloc(&handle->alloc_handle, sizeof(JobBoardState));
    // Update base structure
    handle->alloc_handle.base_addr->user_data1 = jbm_obj;
  } else {
    if (shm_map_read(&handle->alloc_handle, jbm_obj) == NULL) {
      return LIB_ERROR;
    }
  }

  JobBoardState* jbm =
      shm_obj_ptr_data(handle->alloc_handle.base_addr, jbm_obj);

  if (!(jbm->_start_magic == JBM_VALID_MAGIC &&
        jbm->_end_magic == JBM_VALID_MAGIC)) {
    if (!should_create) {
      return -2;
    }
    // We need to init everything as the data is not valid
    jbm->_start_magic = JBM_VALID_MAGIC;

    jbm->version = 1;
    jbm->job_slots = 126;

    jbm->active_slots = 0;
    jbm->next_instance_id = 1;
    jbm->next_task_id = 1;

    jbm->alive =
        true;  // TODO have an option to turn it back on if we want that?

    jbm->_end_magic = JBM_VALID_MAGIC;
  }
  // Now we have a valid JobBoardState
  handle->instance_board_id = atomic_fetch_add(&jbm->next_instance_id, 1);
  handle->jb_state = jbm;
  handle->tasks = (Job*)((char*)jbm + sizeof(JobBoardState));
  if (should_create) {
    memset(handle->tasks, 0, jbm->job_slots * sizeof(Job));
  }
  handle->is_owner = should_create;
  handle->is_valid = true;
  handle->alloc_handle.instance_id = handle->instance_board_id;

  // -- RingBearer
  shm_obj_t rb_obj = handle->alloc_handle.base_addr->user_data2;
  if (rb_obj == INVALID_SHM_OBJ || should_create) {
    // Allocate space on the board to place our data structures
    rb_obj = shm_malloc(&handle->alloc_handle, sizeof(RingBearer));
    // Update base structure
    handle->alloc_handle.base_addr->user_data2 = rb_obj;
  } else {
    if (shm_map_read(&handle->alloc_handle, rb_obj) == NULL) {
      return -3;
    }
  }

  RingBearer* rb = shm_obj_ptr_data(handle->alloc_handle.base_addr, rb_obj);
  // If magic values are broken we need to recreate everything
  if (!(rb->_start_magic == JBM_VALID_MAGIC &&
        rb->_end_magic == JBM_VALID_MAGIC_2)) {
    if (!should_create) {
      return -4;
    }

    rb->_start_magic = JBM_VALID_MAGIC;

    memset(&rb->_pairs, 0, MAX_CONCURRENT_HOSTS * sizeof(RingPair));

    rb->_end_magic = JBM_VALID_MAGIC_2;
  }
  handle->rb_state = rb;

  print_jobBoardHandle(handle);

  TRACY_ZONE_END

  return LIB_SUCCESS;
}

void job_board_close(JobBoardHandle* handle) {
  handle->jb_state->alive = false;
}