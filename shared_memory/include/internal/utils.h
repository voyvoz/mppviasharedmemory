#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "internal/common.h"

// Helper to get the number of available job slots
size_t shm_get_job_num(const JobBoardHandle* handle);

// --- Internal Helper Functions (platform-specific?) ---

/**
 * @brief Initializes the connection to the named shared memory.
 *
 *  Helper to open/create and map shared memory
 *  NOTE This is the common entry point for both APIs.
 *
 * @param handle Must be a valid pointer when calling this function. Will be
 * filled with data by this function. Can be used to check wether this function
 * has failed.
 * @param creator If true, forces re/creation of the shared memory data
 * structures.
 * @return wether this call newly initialized the shared segment or not.
 */
bool shm_create_or_open(shared_mem_allocator_handle* handle,
                        shared_mem_allocator_init_params* params, bool create);

// Helper to unmap shared memory and also optionally unlink it
void shm_close(JobBoardHandle* handle, bool unlink_on_close);

#ifdef __cplusplus
}
#endif

#endif  // UTILS_H
