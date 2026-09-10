
#pragma once

#include <stdint.h>
#include <x86intrin.h>

// A single cacheline
typedef struct alignas(64) CXLCacheLine_t {
  uint64_t _data[8];
} CXLCacheLine_t;
static_assert(sizeof(CXLCacheLine_t) == 64,
              "CXLCacheLine_t must be size of CXL cacheline (64b)");
static_assert(alignof(CXLCacheLine_t) == 64,
              "CXLCacheLine_t must have an alignment of 64");

// Read/Write full cacheline
CXLCacheLine_t cxl_read(void *cachline_addr) {
  _mm_clflushopt(cachline_addr);
  _mm_lfence();
  return *(CXLCacheLine_t *)cachline_addr;
}
void cxl_write(void *cachline_addr, CXLCacheLine_t data) {
  *(CXLCacheLine_t *)cachline_addr = data;
  _mm_clflushopt(cachline_addr);
  _mm_mfence();
}

// Read/Write first 8 byte of cacheline
uint64_t cxl_read_b1(void *cachline_addr) {
  _mm_clflushopt(cachline_addr);
  _mm_lfence();
  return ((CXLCacheLine_t *)cachline_addr)->_data[0];
}
void cxl_write_b1(void *cachline_addr, uint64_t data) {
  ((CXLCacheLine_t *)cachline_addr)->_data[0] = data;
  _mm_clflushopt(cachline_addr);
  _mm_mfence();
}
