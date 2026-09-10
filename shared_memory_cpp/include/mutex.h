/*
Different Mutex implementations for CXL 2.0 shared memory
*/

#pragma once

#include <stdint.h>

#include "primitive.h"

// API
void (*lock)(void *shmBase, uint8_t id);
void (*unlock)(void *shmBase, uint8_t id);

//--------------
// Dekker
//--------------
typedef struct dekkers_lock_t {
  CXLCacheLine_t wants_to_enter[2];
  CXLCacheLine_t turn;
} dekkers_lock_t;
void dekkers_lock(void *shmBase, uint8_t id) {
  dekkers_lock_t *lock_ = (dekkers_lock_t *)shmBase;

  uint8_t other_id = id ^ 1ULL;

  cxl_write_b1(&lock_->wants_to_enter[id], 1);

  while (cxl_read_b1(&lock_->wants_to_enter[other_id])) {
    if (cxl_read_b1(&lock_->turn) == other_id) {
      cxl_write_b1(&lock_->wants_to_enter[id], 0);

      while (cxl_read_b1(&lock_->turn) == other_id) {
      }

      cxl_write_b1(&lock_->wants_to_enter[id], 1);
    }
  }
}
void dekkers_unlock(void *shmBase, uint8_t id) {
  dekkers_lock_t *lock_ = (dekkers_lock_t *)shmBase;

  cxl_write_b1(&lock_->wants_to_enter[id], 0);

  cxl_write_b1(&lock_->turn, id ^ 1ULL);
}

void select_dekker(void *shmBase, uint8_t id) {
  lock = &dekkers_lock;
  unlock = &dekkers_unlock;

  dekkers_lock_t *lock_ = (dekkers_lock_t *)shmBase;

  cxl_write_b1(&lock_->wants_to_enter[id & 1], 0);

  cxl_write_b1(&lock_->turn, 0);
}

//--------------
// Peterson
//--------------
typedef struct petersons_lock_t {
  CXLCacheLine_t flags[2];
  CXLCacheLine_t turn;
} petersons_lock_t;

void peterson_lock(void *shmBase, uint8_t id) {
  petersons_lock_t *lock = (petersons_lock_t *)shmBase;

  uint8_t other_id = id ^ 1ULL;

  cxl_write_b1(&lock->flags[id]._data[0], 1);

  cxl_write_b1(&lock->turn._data[0], other_id);

  while (cxl_read_b1(&lock->flags[other_id]._data[0]) &&
         cxl_read_b1(&lock->turn._data[0]) == other_id) {
  }
}
void peterson_unlock(void *shmBase, uint8_t id) {
  petersons_lock_t *lock = (petersons_lock_t *)shmBase;

  cxl_write_b1(&lock->flags[id], 0);
}
void select_peterson(void *shmBase, uint8_t id) {
  lock = &peterson_lock;
  unlock = &peterson_unlock;

  petersons_lock_t *lock = (petersons_lock_t *)shmBase;

  cxl_write_b1(&lock->flags[id]._data[0], 0);

  cxl_write_b1(&lock->turn._data[0], 0);
}
