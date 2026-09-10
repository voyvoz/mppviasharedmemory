#ifndef DEBUG_HELPER_H
#define DEBUG_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

// Use these to force sleeps between shared memory accesses in some functions
// Should be treated mutually exclusive
// -> _sleep()
#define SLOWMODE_10ms //TODO disable?
// #define SLOWMODE_1s

#include "common.h"

void print_job(const JobBoardHandle* handle, size_t i, Job* t);
void print_board_state(const JobBoardHandle* handle);
void print_board_state_max_jobs(const JobBoardHandle* handle,
                                uint16_t num_jobs);
void print_jobBoardHandle(const JobBoardHandle* handle);

#ifdef __cplusplus
}
#endif

#endif  // DEBUG_HELPER_H
