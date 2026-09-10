#include "internal/ring.h"

#include <assert.h>

inline bool ring_is_full(ShmObjRing* r) {
  // This goes wrong for writer 0 and reader 1
  int16_t a = (r->writer_idx - r->reader_idx);
  int16_t res = a & (RING_SLOTS - 1);
  bool is_full = res == (RING_SLOTS - 1);
  return is_full;
}

inline bool ring_is_empty(ShmObjRing* r) {
  return r->writer_idx == r->reader_idx;
}

int ring_put(ShmObjRing* r, shm_obj_t o) {
  assert(r != NULL);

  // make sure that reader does not overtake writer
  if (ring_is_full(r)) {
    return LIB_ERROR;
  }

  r->_slots[r->writer_idx] = o;
  r->writer_idx = ((r->writer_idx + 1) & RING_SLOTS - 1);

  return LIB_SUCCESS;
}

shm_obj_t ring_pop(ShmObjRing* r) {
  assert(r != NULL);

  // Check if there are any items to pop?
  if (ring_is_empty(r)) {
    return INVALID_JOB_ID;
  }

  shm_obj_t ret = r->_slots[r->reader_idx];
  r->reader_idx = ((r->reader_idx + 1) & RING_SLOTS - 1);
  return ret;
}