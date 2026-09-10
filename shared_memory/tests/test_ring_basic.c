#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include "internal/ring.h"

// -- Does it work correct when a second handle is created?

int main() {
  printf("--- Running ring tests ---\n");

  ShmObjRing sor = {};

  shm_obj_t res = ring_pop(&sor);
  assert(res == INVALID_JOB_ID && "Empty ring should not return valid obj");

  //
  shm_obj_t o1 = 100;
  int put_res = ring_put(&sor, o1);
  assert(put_res == LIB_SUCCESS &&
         "Empty ring should be able to store new entry");
  shm_obj_t p1 = ring_pop(&sor);
  assert(p1 == o1 && "Pop should now return previously put element");
  assert(ring_pop(&sor) == INVALID_JOB_ID &&
         "there should be no element present");

  // Now we should be able to fill completly
  // TODO for some reason the logic currently needs at least a gap of 1 between
  // writer and reader
  for (size_t i = 0; i < RING_SLOTS - 1; i++) {
    assert(ring_put(&sor, i + 1) == LIB_SUCCESS && "should work");
  }
  assert(ring_put(&sor, 69) == LIB_ERROR &&
         "should not work due to queue being full");

  // Pop em all again
  for (size_t i = 0; i < RING_SLOTS - 1; i++) {
    assert(ring_pop(&sor) != INVALID_JOB_ID && "should work");
  }
  assert(ring_pop(&sor) == INVALID_JOB_ID && "");

  printf("--- Running ring tests ---\n  -> ☑️\n");

  return 0;
}
