#ifndef TYPES_H
#define TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>  
#include <stdbool.h>

// --- 

typedef uint32_t job_id_t;
typedef uint32_t job_status_t;
typedef uint8_t job_deps_t;
typedef uint16_t err_t;
typedef int8_t result_t;

#define LIB_SUCCESS 0
#define LIB_ERROR -1

// --- SMM

// This strucuture is used to allow sharing of objects between hosts
// because pointers are only valid within a process.
//
// The offset is valid within the shared memory.
typedef uint64_t shm_obj_t;

#define INVALID_JOB_ID 0

#ifdef __cplusplus
}
#endif

#endif  // TYPES_H