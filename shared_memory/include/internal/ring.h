#ifndef RING_H
#define RING_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#include "types.h"

// TODO really this value?
#define RING_SLOTS 16

typedef struct ShmObjRing {
  shm_obj_t _slots[RING_SLOTS];
  int16_t reader_idx;
  int16_t writer_idx;
} ShmObjRing;

// --- ring API

int ring_put(ShmObjRing* r, shm_obj_t o);
shm_obj_t ring_pop(ShmObjRing* r);

bool ring_is_full(ShmObjRing* r);
bool ring_is_empty(ShmObjRing* r);

#ifdef __cplusplus
}
#endif

#endif  // RING_H