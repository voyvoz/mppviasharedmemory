#ifndef DISPATCHER_API_H
#define DISPATCHER_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include "internal/debug_helper.h"
#include "internal/utils.h"
#include "smm/allocator.h"

DispatchResult dispatch_job(const JobBoardHandle* handle, PostableJobTree* job);

shm_obj_t dispatch_await(const JobBoardHandle* handle, DispatchResult res);

#ifdef __cplusplus
}
#endif

#endif  // DISPATCHER_API_H
