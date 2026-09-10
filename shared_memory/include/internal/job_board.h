#ifndef JOB_BOARD_H
#define JOB_BOARD_H

#ifdef __cplusplus
extern "C" {
#endif

#include "internal/common.h"

int job_board_init(JobBoardHandle* handle, bool should_create);

void job_board_close(JobBoardHandle* handle);

#ifdef __cplusplus
}
#endif

#endif  // JOB_BOARD_H
