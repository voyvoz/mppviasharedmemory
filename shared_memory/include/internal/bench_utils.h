#ifndef BENCH_UTILS_H
#define BENCH_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

// https://stackoverflow.com/questions/9642732/parsing-command-line-arguments-in-c
// TODO

#include <stddef.h>

// IDEA switch case that uses cmd params: --name v
// then automatically assigns v to the varaible if --name is present
// and registers the help for the -h or empty cmd
void register_param(const char* name, size_t* v_ptr) {}

#define DECLARE_BENCH_PARAM(name, default_value, help) \
  size_t name = default_value;

#ifdef __cplusplus
}
#endif

#endif  // BENCH_UTILS_H