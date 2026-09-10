#include "internal/common.h"

#include <execinfo.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "internal/logging.h"

LibErr* new_err(err_t code, char* message, ...) {
  LibErr* e = malloc(sizeof(LibErr));
  e->err_code = code;
  e->msg = NULL;
  va_list argptr;
  va_start(argptr, message);
  int res = vsprintf(&e->msg, message, argptr);
  if (res == -1) {
    e->msg = message;
  };
  va_end(argptr);
  return e;
}

void print_trace(int skip) {
  void* array[10];
  char** strings;
  int size, i;

  size = backtrace(array, 10);
  strings = backtrace_symbols(array, size);
  if (strings != NULL) {
    if (skip < size) {
      i = skip;
    } else {
      i = 0;
    }
    for (; i < size; i++) printf("%s\n", strings[i]);
  }

  free(strings);
}

void _mprotect(void* p, size_t s, int f) {
  // print_trace(2);
  // LOG_INF("MPROT" "(%p, %lu, %u)\n", p,s,f);
#ifdef ENABLE_MPROTECT
  int res = mprotect(p, s, f);
  if (res == -1) {
    perror("mprotect failed with");
    assert(false && "mprotect for AllocState failed");
    exit(1);
  }
#else
  (void)p;
  (void)s;
  (void)f;
#endif
}

void _sleep() {
#ifdef SLOWMODE_10ms
  struct timespec ts = {.tv_sec = 0, .tv_nsec = 10000000};  // 10ms
  nanosleep(&ts, NULL);
#endif
#ifdef SLOWMODE_1s
  sleep(1);
#endif
}

shm_obj_t job_get_input(Job* j) { return j->meta._params; }

JobBoardHandle create_empty_handle() {
  JobBoardHandle j = {};
  return j;
}

PostableJobTree create_empty_job_tree() {
  PostableJobTree p = {};
  return p;
}

bool _integrity_check(const JobBoardHandle* jbh) {
  return jbh->org_version == jbh->jb_state->version &&
         jbh->org_job_slots == jbh->jb_state->job_slots;
}