#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "internal/common.h"
#include "internal/logging.h"
#include "smm/allocator.h"

#define SHM_ALLOC_PRINT_PREFIX "[SMM] "

#define DAX_PREFIX "/dev/dax"

// ----------
// Helper

// Create a mask of bit_count consecutive bits starting at start_index
uint64_t create_alloc_mask(size_t start_index, size_t bit_count) {
  return (bit_count >= 64) ? ~0ULL : ((1ULL << bit_count) - 1) << start_index;
}

// Create a mask of size count, find consecutive zeroes and then flip them
int find_consecutive_space_then_apply_mask(uint64_t* bitmap, size_t count) {
  if (count > 64 || count == 0) {
    return -1;
  }

  uint64_t value = *bitmap;

  // Create the mask
  uint64_t mask = create_alloc_mask(0, count);

  // Find enough consecutive zeroes and flip them
  for (size_t shift = 0; shift <= 64 - count; shift++) {
    uint64_t shifted_mask = mask << shift;
    if ((value & shifted_mask) == 0) {
      (*bitmap) |= shifted_mask;
      return shift;
    }
  }
  return -1;  // Not enough consecutive zeros found
}

bool has_valid_start_magic(shared_mem_allocator_handle* handle) {
  return *(uint32_t*)handle->base_addr == VALID_MAGIC;
}

void write_valid_start_block(shared_mem_allocator_handle* handle,
                             const shared_mem_allocator_init_params* params) {
  assert(handle != NULL);
  assert(params != NULL);

  LOG_INF(SHM_ALLOC_PRINT_PREFIX
          "[%u] "
          "write_valid_start_block() \n",
          handle->instance_id);

  // We already have FRAME_SIZE mprotect read/write accessable.
  // Now initilaize them with the shared state that is required.
  // We know the total requested size, so we create correctly sized data
  // structures

  // --- Simple state data
  AllocatorState* s = (AllocatorState*)handle->base_addr;
  atomic_store(&s->magic, VALID_MAGIC);
  s->creation_size_bytes =
      params->size_bytes;  // How many bytes the inital allocation used
  s->used_bytes = 0;       // We start with an empty shared segment
  s->created = time(NULL);
  atomic_store(&s->_lifetime_allocations, 0);
  atomic_store(&s->_shm_id_counter, 1);

  // v2
  // TODO this must be correctly aligned to TX_SIZE
  // So we start on the next page with these structures
  s->start_bitvec = FRAME_SIZE;

  // Calculate how many consecutive bit vector pages we need
  // Skip first page as that will always hold the allocator state info
  uint32_t required_bytes =
      ((params->size_bytes - FRAME_SIZE) / sizeof(bit_vec_page)) / 8;
  uint32_t required_consecutive_bitvec_pages =
      required_bytes / sizeof(bit_vec_page);
  if (required_consecutive_bitvec_pages == 0) {
    required_consecutive_bitvec_pages = 1;  // We need at least a single page
  }
  s->bitvec_pages = required_consecutive_bitvec_pages;

  // Get write priv on the memory and init it
  uint32_t bit_vec_byte_len =
      sizeof(bit_vec_page) * required_consecutive_bitvec_pages;
  bit_vec_page* start_bitvec_ptr =
      (bit_vec_page*)shm_generic_offset_to_ptr(s, s->start_bitvec);
  _mprotect(start_bitvec_ptr, bit_vec_byte_len, PROT_READ | PROT_WRITE);

  memset(start_bitvec_ptr, 0, bit_vec_byte_len);

  // Now we know where the shared segment starts
  s->start_mem = s->start_bitvec + bit_vec_byte_len;

  s->max_bytes = params->size_bytes -
                 (FRAME_SIZE + FRAME_SIZE * required_consecutive_bitvec_pages);
}

bool shm_alloc_handle_valid(shared_mem_allocator_handle* handle) {
  if (handle == NULL) {
    return false;
  }
  if (handle->base_addr == NULL) {
    return false;
  }
  return true;
}

void shm_print_state(shared_mem_allocator_handle* handle) {
#ifdef ENABLE_LOGS
  LOG_INF(SHM_ALLOC_PRINT_PREFIX
          "[%u] "
          "shm_handle[%p] = { .magic=%x .created=%ld .bv_pages=%u "
          ".total_bytes=%llu }\n",
          handle->instance_id, handle->base_addr, handle->base_addr->magic,
          handle->base_addr->created, handle->base_addr->bitvec_pages,
          handle->base_addr->creation_size_bytes);
#else
  (void)handle;
#endif
}

bool memTragetIsDaxDevice(const shared_mem_allocator_init_params* params) {
  return strncmp(params->path, DAX_PREFIX, sizeof(DAX_PREFIX)) == 0;
}

// ----------
// Implementaitons
int shm_alloctor_init(shared_mem_allocator_handle* h,
                      const shared_mem_allocator_init_params* params) {
  assert(params != NULL);
  assert(params->path != NULL);
  assert(params->size_bytes != 0);
  assert(params->size_bytes >= FRAME_SIZE * 10);

  // What about concurrent inits?

  int flags = O_RDWR;
  mode_t mode = S_IRUSR | S_IWUSR;
  int initial_mapped_size = params->size_bytes;
  int was_created_new = 0;

  // https://stackoverflow.com/questions/74243583/memory-map-file-with-growing-size

  int fd;
  // -- Open the shared memory file
  // Detect type of shared memory
  bool daxMode = memTragetIsDaxDevice(params);
  if (daxMode) {
    fd = open(params->path, flags);
    if (fd == -1) {
      perror("open failed");
      exit(1);
    }
  } else {
    fd = shm_open(params->path, flags, mode);
    if (fd == -1) {
      if (errno == ENOENT) {
        fd = shm_open(params->path, O_CREAT | O_EXCL | flags, mode);
        if (fd == -1) {
          perror("shm_open failed to create");
          exit(1);
        }
        if (ftruncate(fd, initial_mapped_size) == -1) {
          perror("ftruncate failed");
          if (shm_unlink(params->path) == -1) {
            if (errno != ENOENT) {
              perror("shm_unlink failed");
            }
          }
          close(fd);
          exit(1);
        }
        was_created_new = 1;
      } else {
        perror("shm_open failed to open existing");
        exit(1);
      }
    }
  }

  // Linux:
  // TODO check if MAP_SYNC might be usefull here
  // TODO MAP_NORESERVE?
#ifdef ENABLE_MPROTECT
  int flags_for_mmap = PROT_NONE;
#else
  int flags_for_mmap = PROT_READ | PROT_WRITE;
#endif
  h->base_addr =
      mmap(NULL, initial_mapped_size, flags_for_mmap, MAP_SHARED, fd, 0);
  if (h->base_addr == MAP_FAILED) {
    close(fd);
    perror("mmap failed");
    exit(1);
  }
  close(fd);

  // Make sure that all memory for the shared alloc state is accessable
  _mprotect(h->base_addr, FRAME_SIZE, PROT_READ | PROT_WRITE);

  LOG_INF(SHM_ALLOC_PRINT_PREFIX
          "[%u] "
          "init(bytes=%u,path='%s',create=%u,daxMode=%u) -> %p\n",
          h->instance_id, params->size_bytes, params->path, params->create,
          daxMode, h->base_addr);

  // TODO if should create then kill state block?
  if (daxMode && params->create) {
    h->base_addr->magic = 0x0;
  }

  // -- Check if there is already a valid State structure
  if (!has_valid_start_magic(h)) {
    // -- Create valid state structure
    write_valid_start_block(h, params);
  } else {
    void* bitvec_ptr =
        shm_generic_offset_to_ptr(h->base_addr, h->base_addr->start_bitvec);
    _mprotect(bitvec_ptr, h->base_addr->bitvec_pages * FRAME_SIZE,
              PROT_READ | PROT_WRITE);

    // TODO check for page size config and print warning when they dont match!
  }

  shm_print_state(h);

  return was_created_new;
}

// Should enforce 64byte alignment!
// TODO new strat idea:
// Just split the whole range by 2 until the reach the desired size. Then one
// cacheline for meta data and return a pointer to the rest All API functions
// know that given pointer -64 is where the meta data lives
//
shm_obj_t shm_malloc(const shared_mem_allocator_handle* handle,
                     uint64_t requested_bytes) {
  assert(handle != NULL);
  assert(requested_bytes > 0);

  TRACY_ZONE_START

  // TODO we actually need a guard here when we need more than 64x16384
  if (requested_bytes >= (64 * 16384) - sizeof(allocated_block_metadata)) {
    // Currently not supported
    TRACY_ZONE_END
    return INVALID_SHM_OBJ;
  }

  // We need more bytes than requested as the first few are used for metadata
  requested_bytes += sizeof(allocated_block_metadata);

  LOG_INF(SHM_ALLOC_PRINT_PREFIX
          "[%u] "
          "malloc(%llu): { prev_allocs=%llu } \n",
          handle->instance_id, requested_bytes,
          handle->base_addr->_lifetime_allocations);

  // Get correct amount of pages to satisfy request
  int required_frames =
      1 + ((requested_bytes + sizeof(allocated_block_metadata)) / FRAME_SIZE);

  size_t page_idx = 0;
  size_t byte_idx = 0;
  size_t bit_idx = 0;
  uint64_t new_bitvec_v = 0;
  for (; page_idx < handle->base_addr->bitvec_pages; page_idx++) {
    byte_idx = 0;
    bit_vec_page* page =
        ((bit_vec_page*)shm_generic_offset_to_ptr(
            handle->base_addr, handle->base_addr->start_bitvec)) +
        page_idx;
    for (; byte_idx < FRAME_SIZE; byte_idx++) {
      _Atomic(uint64_t)* v = &page->_bitvec[byte_idx];
      uint64_t current_bitvec_v = atomic_load(v);
      bit_idx = 0;
      switch (current_bitvec_v) {
        case 0xFFFFFFFFFFFFFFFF:
          // All used
          continue;
        case 0:
          // All free
          new_bitvec_v = create_alloc_mask(bit_idx, required_frames);
        default:
          // Check if the bitvec has enough space
          uint16_t used_pages = POPCOUNT(current_bitvec_v);
          if (used_pages + required_frames > 64) {
            continue;
          }
          // Check if the available space in consecutive
          new_bitvec_v = current_bitvec_v;
          bit_idx = find_consecutive_space_then_apply_mask(&new_bitvec_v,
                                                           required_frames);
          if (bit_idx == -1UL) {
            // Available space is not consecutive
            continue;
          }
      }
      if (atomic_compare_exchange_strong(v, &current_bitvec_v, new_bitvec_v)) {
        LOG_INF(
            "[%u] (req_frames: %u , idx: %zu) \n"
            "BitVec:  " PRINTF_BINARY_PATTERN_INT64
            "Prev\n"
            "BitVec:  " PRINTF_BINARY_PATTERN_INT64 " Now\n",
            handle->instance_id, required_frames, bit_idx,
            PRINTF_BYTE_TO_BINARY_INT64(current_bitvec_v),
            PRINTF_BYTE_TO_BINARY_INT64(new_bitvec_v));
        goto alloced_mem;
      }
      // Failed to allocate so we need to try again
    }
  }
  // Failed to find space!
  TRACY_ZONE_END
  return INVALID_SHM_OBJ;

alloced_mem:

  // Calc offset to the now reserved memory segment
  char* mem_seg_start_ptr =
      (char*)handle->base_addr + handle->base_addr->start_mem;
  uint32_t page_offset =
      ((page_idx * FRAME_SIZE) + (byte_idx * 64) + bit_idx) * FRAME_SIZE;
  allocated_block_metadata* shm_obj_meta =
      (allocated_block_metadata*)(mem_seg_start_ptr + page_offset);

  uint64_t allocated_bytes = required_frames * FRAME_SIZE;

  shm_obj_t ret = shm_ptr_to_generic_offset(handle->base_addr, shm_obj_meta);

  void* shm_obj_ptr = shm_generic_offset_to_ptr(handle->base_addr, ret);
  _mprotect(shm_obj_ptr, allocated_bytes, PROT_READ | PROT_WRITE);

  // Init shared mem object meta data
  memset(shm_obj_ptr, 0, sizeof(allocated_block_metadata));
  shm_obj_meta->size_bytes = allocated_bytes;
  shm_obj_meta->_refs = 1;
  shm_obj_meta->_lead_magic = LEAD_MAGIC;
  shm_obj_meta->_end_maigc = END_MAGIC;

  shm_obj_meta->_id = atomic_fetch_add(&handle->base_addr->_shm_id_counter, 1);
  atomic_fetch_add(&handle->base_addr->_lifetime_allocations, 1);

  LOG_INF(SHM_ALLOC_PRINT_PREFIX
          "[%u] "
          "malloc(%llu, base=%p): [shm_obj: %llu {%llu @ %p}] <- bv_p=%zu "
          "bv_byte=%zu bv_bit=%zu "
          "allocated_bytes=%llu frames=%u\n",
          handle->instance_id, requested_bytes, handle->base_addr,
          shm_obj_meta->_id, ret, shm_obj_ptr, page_idx, byte_idx, bit_idx,
          allocated_bytes, required_frames);

  TRACY_ZONE_END
  TRACY_ALLOC(shm_obj_ptr, allocated_bytes);

  return ret;
}

void* shm_map_read(const shared_mem_allocator_handle* handle, shm_obj_t obj) {
  assert(handle != NULL);
  assert(obj != INVALID_SHM_OBJ);

  TRACY_ZONE_START

  allocated_block_metadata* meta = shm_obj_ptr_meta(handle->base_addr, obj);

  // Allow us to look at the metadata which is fixed size
  _mprotect(meta, FRAME_SIZE, PROT_READ | PROT_WRITE);
  LOG_INF(SHM_ALLOC_PRINT_PREFIX
          "[%u] "
          "map_read(%llu): [obj_id:%llu] base=%p meta=%p size=%llu\n",
          handle->instance_id, obj, meta->_id, handle->base_addr, meta,
          meta->size_bytes);

  assert(meta->_lead_magic == LEAD_MAGIC &&
         "Invalid metadata start_magic value");
  assert(meta->_end_maigc == END_MAGIC && "Invalid metadata end_magic value");

  // In case the object is bigger than what we already used in the previous
  // mprotect call we need to call it again with the correct size
  if (meta->size_bytes > FRAME_SIZE) {
    _mprotect(meta, meta->size_bytes, PROT_READ | PROT_WRITE);
  }

  atomic_fetch_add(&meta->_refs, 1);

  // Flip bit of the host that called this func?
  // CXL2.0 allows at max 16 hosts. so we could just use a single bit map to
  // track which host has read mapped this allocation?

  TRACY_ZONE_END

  return shm_obj_ptr_data(handle->base_addr, obj);
}

int shm_free(const shared_mem_allocator_handle* handle, shm_obj_t obj) {
  assert(handle != NULL);

  TRACY_ZONE_START

  allocated_block_metadata* meta = shm_obj_ptr_meta(handle->base_addr, obj);
try_free:
  uint16_t refs = atomic_load(&meta->_refs);
  uint16_t decreased_refs = refs - 1;
  if (refs == 0) {
    fprintf(stderr,
            "[%u] "
            "cant double free\n",
            handle->instance_id);
    TRACY_ZONE_END
    return -1;
  }
  if (!atomic_compare_exchange_strong(&meta->_refs, &refs, decreased_refs)) {
    goto try_free;
  }

  LOG_INF(
      SHM_ALLOC_PRINT_PREFIX
      "[%u] "
      "free(%llu, base=%p): [obj_id:%llu] size=%llu refs=%u -> dec_refs=%u\n",
      handle->instance_id, obj, handle->base_addr, meta->_id, meta->size_bytes,
      meta->_refs, decreased_refs);

  if (decreased_refs == 0) {
    obj -= handle->base_addr->start_mem;
    // Find uint64_t that holds flags for the given shared obj
    size_t page_idx = 0;  // TODO
    // 16384*64 -> 1059498
    size_t byte_idx = obj >= 1059498 ? obj / 1059498 : 0;
    size_t bit_idx = (obj % 1059498) / FRAME_SIZE;

    // uint16_t used_bits = meta->size_bytes / FRAME_SIZE;

    bit_vec_page* page =
        ((bit_vec_page*)shm_generic_offset_to_ptr(
            handle->base_addr, handle->base_addr->start_bitvec)) +
        page_idx;
    _Atomic(uint64_t)* v = &page->_bitvec[byte_idx];
  free_mask:
    uint64_t current_mask = atomic_load(v);

    size_t frames = (meta->size_bytes / FRAME_SIZE);

    LOG_INF(
        SHM_ALLOC_PRINT_PREFIX
        "[%u] "
        "free(%llu, base=%p): [obj_id:%llu] bv_p=%zu bv_byte=%zu bv_bit=%zu "
        "consecutive_frames=%zu\n%p=" PRINTF_BINARY_PATTERN_INT64 "\n",
        handle->instance_id, obj, handle->base_addr, meta->_id, page_idx,
        byte_idx, bit_idx, frames, page,
        PRINTF_BYTE_TO_BINARY_INT64(current_mask));

    // Create a mask that has exactly the bits we want to free turned to 0
    uint64_t new_mask = current_mask & ~create_alloc_mask(bit_idx, frames);
    // Replace the mask we know with the new one that has the frames marked as
    // free
    if (!atomic_compare_exchange_strong(v, &current_mask, new_mask)) {
      goto free_mask;
    }

    LOG_INF(SHM_ALLOC_PRINT_PREFIX
            "[%u] "
            "free(%llu, base=%p): "
            "[obj_id:%llu]\nnew_used_mask=" PRINTF_BINARY_PATTERN_INT64 "\n",
            handle->instance_id, obj, handle->base_addr, meta->_id,
            PRINTF_BYTE_TO_BINARY_INT64(new_mask));

    TRACY_FREE(meta);
  }

  _mprotect(meta, meta->size_bytes, PROT_NONE);

  TRACY_ZONE_END
  return 0;
}
