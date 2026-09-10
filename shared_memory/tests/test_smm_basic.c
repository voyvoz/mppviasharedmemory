#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/wait.h>

#define TEST_BOARD_NAME "/test_smm_allcator"

#include "internal/logging.h"
#include "smm/allocator.h"
#include "smm/test_helper.h"

shared_mem_allocator_init_params alloc_params = {
    .path = TEST_BOARD_NAME,
    .size_bytes = 4194304,  // 4MB
};

// -- Does init work?
void test_init(shared_mem_allocator_handle* handle,
               shared_mem_allocator_handle* handle2) {
  int res = shm_alloctor_init(handle, &alloc_params);
  assert(shm_alloc_handle_valid(handle) == true && "Invalid handle");
  assert(res == 1 && "Should have created a new base state");

  time_t created = handle->base_addr->created;

  res = shm_alloctor_init(handle2, &alloc_params);
  assert(shm_alloc_handle_valid(handle2) == true && "Invalid handle_2");
  assert(res == 0 && "Should have found a state in the shared mem");

  // this never works as calling shm_open && mmap on the same location will
  // create two mappings with distinct address spaces which actually are backed
  // by the same pyhsical memory. Without platform specific code we cant verify
  // that the pointed to locations are actually the same.
  // assert(handle->base_addr == handle2->base_addr && "State pointer should
  // point to the same location");
  assert(created == handle->base_addr->created &&
         created == handle2->base_addr->created &&
         "Created time should be equal from both handles");

  shm_print_state(handle);
  shm_print_state(handle2);
  printf("✔ Init test\n");
}

// -- Does single alloc work?
void test_malloc(shared_mem_allocator_handle* handle,
                 shared_mem_allocator_handle* handle2) {
  uint8_t* min_addr =
      (uint8_t*)(uint64_t)
          handle->base_addr;  // First addressable byte within shared segment
  uint8_t* max_addr = (uint8_t*)(uint64_t)handle->base_addr +
                      handle->base_addr->creation_size_bytes +
                      3;  // Last addressable byte within shared segment
  shm_obj_t addr_1_obj = shm_malloc(handle, 1);
  assert(addr_1_obj != 0 && "Invalid malloc return value");
  void* addr_1_meta = shm_obj_ptr_meta(handle->base_addr, addr_1_obj);
  assert(addr_1_meta != NULL && "Meta field pointer cant be NULL");
  void* addr_1 = shm_obj_ptr_data(handle->base_addr, addr_1_obj);
  assert(addr_1 != NULL && "Invalid shm_obj_t to data pointer conversion");
  assert(addr_1_meta != addr_1 &&
         "Meta and Data field of a shared mem object cant be at the same "
         "location");
  assert((uint64_t)addr_1 - (uint64_t)addr_1_meta ==
             sizeof(allocated_block_metadata) &&
         "Meta and Data part of a shared mem object must be consecutive");
  assert((uint64_t)min_addr < (uint64_t)addr_1 &&
         (uint64_t)addr_1 < (uint64_t)max_addr &&
         "Pointer must point within shared memory");
  *(uint8_t*)addr_1 = 0x11;

  uint8_t* min_addr_h2 =
      (uint8_t*)(uint64_t)
          handle2->base_addr;  // First addressable byte within shared segment
  uint8_t* max_addr_h2 = (uint8_t*)(uint64_t)handle2->base_addr +
                         handle2->base_addr->creation_size_bytes +
                         3;  // Last addressable byte within shared segment
  assert(min_addr_h2 != min_addr && max_addr_h2 != max_addr &&
         "Different handles that map the same segment to different offset must "
         "have different boundarys");
  shm_obj_t addr_2_obj = shm_malloc(handle2, 1);
  assert(addr_2_obj != 0 && "Invalid malloc return value");
  void* addr_2 = shm_obj_ptr_data(handle2->base_addr, addr_2_obj);
  assert(addr_2 != NULL && "Invalid shm_obj_t to pointer conversion");
  assert((uint64_t)min_addr_h2 < (uint64_t)addr_2 &&
         (uint64_t)addr_2 < (uint64_t)max_addr_h2 &&
         "Pointer must point within shared memory h2");
  assert(addr_1 != addr_2 &&
         "Different allocation should return different addresses");
  assert(addr_1_obj != addr_2_obj &&
         "Different allocations should be at different offsets");
  *(uint8_t*)addr_2 = 0x22;

  assert(*(uint8_t*)addr_1 == 0x11 && "addr_1 Readback failed");
  assert(*(uint8_t*)addr_2 == 0x22 && "addr_1 Readback failed");

  assert(shm_free(handle, addr_1_obj) != -1);
  assert(shm_free(handle2, addr_2_obj) != -1);

  // Double free fails with EXC_BAD_ACCESS dues to mprotect having unmapped the
  // segment already

  printf("✔ Malloc test\n");
}

void test_mask_creation() {
  assert(create_alloc_mask(0, 1) == 1 && "Mask(0,1) failed");
  assert(create_alloc_mask(1, 1) == 2 && "Mask(1,1) failed");

  uint64_t bm_1 = 0b10001;
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  int idx = find_consecutive_space_then_apply_mask(&bm_1, 3);
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  LOG_INF("-> %i \n", idx);
  assert(idx == 1 && "find_consecutive_space_then_apply_mask failed");

  bm_1 = 0b11111111;
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  idx = find_consecutive_space_then_apply_mask(&bm_1, 3);
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  LOG_INF("-> %i \n", idx);
  assert(idx == 8 && "find_consecutive_space_then_apply_mask failed");

  bm_1 = ~0ULL;
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  idx = find_consecutive_space_then_apply_mask(&bm_1, 3);
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  LOG_INF("-> %i \n", idx);
  assert(idx == -1 && "find_consecutive_space_then_apply_mask failed");

  bm_1 = 0b10010010001;
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  idx = find_consecutive_space_then_apply_mask(&bm_1, 4);
  LOG_INF("Map: " PRINTF_BINARY_PATTERN_INT64 "\n",
          PRINTF_BYTE_TO_BINARY_INT64(bm_1));
  LOG_INF("-> %i \n", idx);
  assert(idx == 11 && "find_consecutive_space_then_apply_mask failed");
}

// -- Does it work correct when a second handle is created?

int main() {
  printf("--- Running Basic shared Allocator Test ---\n");

  test_mask_creation();

  shm_unlink(TEST_BOARD_NAME);

  shared_mem_allocator_handle handle;
  shared_mem_allocator_handle handle_2;
  test_init(&handle, &handle_2);

  test_malloc(&handle, &handle_2);

  shm_unlink(TEST_BOARD_NAME);

  printf("--- Running Basic shared Allocator Test Finished ☑️ ---\n");

  return 0;
}
