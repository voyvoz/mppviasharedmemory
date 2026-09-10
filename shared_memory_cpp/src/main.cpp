/**
 * Using bits from
 *  https://vgatherps.github.io/2018-09-02-nontemporal/
 */

#include <emmintrin.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <x86intrin.h>

#include <chrono>
#include <iostream>
#include <thread>

#include <perfcpp/analyzer/memory_access.h>
#include <perfcpp/event_counter.h>
#include <perfcpp/hardware_info.h>
#include <perfcpp/sampler.h>

#include "mutex.h"
#include <assert.h>

typedef struct shared_t {
  alignas(64) volatile uint64_t flag;
} shared_t;
static_assert(sizeof(shared_t) == 64, "shared_t should be size of cacheline");

uint8_t store_variant = 0;
uint8_t load_variant = 0;
uint64_t iter_target = 1000000;
uint64_t spins = 0;
uint64_t warmup_target = 1000;

// ====================
// Common init
// ====================

void *init_dax(size_t size) {
  int fd = open("/dev/dax0.0", O_RDWR);
  if (fd == -1) {
    perror("opening dax dev failed");
    exit(-1);
  }
  void *base_addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (base_addr == MAP_FAILED) {
    perror("failed to mmap");
    exit(-1);
  }
  close(fd);
  return base_addr;
}

// ------------------------------------
// Load/Store variations
// ------------------------------------

// REGULAR STORES
static inline void write_mov_sfence(shared_t *s, uint64_t v) {
  __asm__ volatile("\n mov     %1, (%0)"
                   "\n sfence  "
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_mov_clwb(shared_t *s, uint64_t v) {
  __asm__ volatile("\n mov     %1, (%0)"
                   "\n clwb    (%0)"
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_mov_clwb_sfence(shared_t *s, uint64_t v) {
  __asm__ volatile("\n mov     %1, (%0)"
                   "\n clwb    (%0)"
                   "\n sfence  "
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_mov_clflush(shared_t *s, uint64_t v) {
  __asm__ volatile("\n mov     %1, (%0)"
                   "\n clflush (%0)"
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_mov_clflush_sfence(shared_t *s, uint64_t v) {
  __asm__ volatile("\n mov     %1, (%0)"
                   "\n clflush (%0)"
                   "\n sfence  "
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_mov_clflushopt(shared_t *s, uint64_t v) {
  __asm__ volatile("\n mov         %1, (%0)"
                   "\n clflushopt  (%0)"
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_mov_clflushopt_sfence(shared_t *s, uint64_t v) {
  __asm__ volatile("\n mov         %1, (%0)"
                   "\n clflushopt  (%0)"
                   "\n sfence  "
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}

alignas(64) uint64_t buf[8] = {0, 0, 0, 0, 0, 0, 0, 0};

// NON TEMPORAL HINT
static inline void write_movnti(shared_t *s, uint64_t v) {
  __asm__ volatile("\n movnti  %1, (%0)" : : "r"(&s->flag), "r"(v) : "memory");
}
static inline void write_movnti_sfence(shared_t *s, uint64_t v) {
  __asm__ volatile("\n movnti  %1, (%0)"
                   "\n sfence  "
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_vmovntdq(shared_t *s, uint64_t v) {
  buf[0] = v;
  __asm__ volatile("\n vmovdqa64   (%1), %%zmm0"
                   "\n vmovntdq    %%zmm0, (%0)"
                   :
                   : "r"(&s->flag), "r"(buf)
                   : "zmm0", "memory");
}
static inline void write_vmovntdq_sfence(shared_t *s, uint64_t v) {
  buf[0] = v;
  __asm__ volatile("\n vmovdqa64   (%1), %%zmm0"
                   "\n vmovntdq    %%zmm0, (%0)"
                   "\n sfence      "
                   :
                   : "r"(&s->flag), "r"(buf)
                   : "zmm0", "memory");
}
// DIRECT STORE
static inline void write_movdiri(shared_t *s, uint64_t v) {
  __asm__ volatile("\n movdiri   %1, (%0)"
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_movdiri_sfence(shared_t *s, uint64_t v) {
  __asm__ volatile("\n movdiri   %1, (%0)"
                   "\n sfence      "
                   :
                   : "r"(&s->flag), "r"(v)
                   : "memory");
}
static inline void write_movdir64b(shared_t *s, uint64_t v) {
  buf[0] = v;
  __asm__ volatile("\n movdir64b   (%1), %0"
                   :
                   : "r"(&s->flag), "r"(buf)
                   : "memory");
}
static inline void write_movdir64b_sfence(shared_t *s, uint64_t v) {
  buf[0] = v;
  __asm__ volatile("\n movdir64b   (%1), %0"
                   "\n sfence      "
                   :
                   : "r"(&s->flag), "r"(buf)
                   : "memory");
}
// ATOMIC
static inline void write_xchg(shared_t *s, uint64_t v) {
  __asm__ volatile("\n xchg    %0, (%1)" : "+r"(v) : "r"(&s->flag) : "memory");
}
static inline void write_xchg_flush(shared_t *s, uint64_t v) {
  __asm__ volatile("\n xchg    %0, (%1)" : "+r"(v) : "r"(&s->flag) : "memory");
  _mm_clflush(s);
}
static inline void write_xchg_flushopt(shared_t *s, uint64_t v) {
  __asm__ volatile("\n xchg    %0, (%1)" : "+r"(v) : "r"(&s->flag) : "memory");
  _mm_clflushopt(s);
}

// STORE
void write_to_cxl(shared_t *s, uint64_t v) {
  switch (store_variant) {
  case 0:
    s->flag = v;
    break;
  // Regular
  case 1:
    write_mov_sfence(s, v);
    break;
  case 2:
    write_mov_clwb(s, v);
    break;
  case 3:
    write_mov_clwb_sfence(s, v);
    break;
  case 4:
    write_mov_clflush(s, v);
    break;
  case 5:
    write_mov_clflush_sfence(s, v);
    break;
  case 6:
    write_mov_clflushopt(s, v);
    break;
  case 7:
    write_mov_clflushopt_sfence(s, v);
    break;
  // Non-temporal
  case 10:
    write_movnti(s, v);
    break;
  case 11:
    write_movnti_sfence(s, v);
    break;
  case 12:
    write_vmovntdq(s, v);
    break;
  case 13:
    write_vmovntdq_sfence(s, v);
    break;
  // Direct
  case 20:
    write_movdiri(s, v);
    break;
  case 21:
    write_movdiri_sfence(s, v);
    break;
  case 22:
    write_movdir64b(s, v);
    break;
  case 23:
    write_movdir64b_sfence(s, v);
    break;
  // Atomic
  case 30:
    write_xchg(s, v);
    break;
  case 31:
    write_xchg_flush(s, v);
    break;
  case 32:
    write_xchg_flushopt(s, v);
    break;
  default:
    fprintf(stderr, "invalid store_variant: '%u'\n", store_variant);
    exit(-1);
  }
}

// LOAD
static inline void await_flag_clflush(shared_t *s, uint64_t v) {
  uint64_t f = s->flag;
  while (f != v) {
    _mm_clflush(s);
    f = s->flag;
    spins++;
  }
}
static inline void await_flag_clflushopt(shared_t *s, uint64_t v) {
  uint64_t f = s->flag;
  while (f != v) {
    _mm_clflushopt(s);
    f = s->flag;
    spins++;
  }
}
static inline void await_flag_clwb(shared_t *s, uint64_t v) {
  // DOES NOT WORK
  // https://www.felixcloutier.com/x86/clwb
  // "The line may be retained in the cache hierarchy in non-modified state"
  // -> We never see the change that the other host makes so we just wait
  // forever.
  uint64_t f = s->flag;
  while (f != v) {
    _mm_clwb(s);
    f = s->flag;
    spins++;
  }
}
// LOAD - FENCE
static inline void await_flag_clflush_fence(shared_t *s, uint64_t v) {
  uint64_t f = s->flag;
  while (f != v) {
    _mm_clflush(s);
    _mm_lfence();
    f = s->flag;
    spins++;
  }
}
static inline void await_flag_clflushopt_fence(shared_t *s, uint64_t v) {
  uint64_t f = s->flag;
  while (f != v) {
    _mm_clflushopt(s);
    _mm_lfence();
    f = s->flag;
    spins++;
  }
}
static inline void await_flag_clwb_fence(shared_t *s, uint64_t v) {
  // DOES NOT WORK
  // https://www.felixcloutier.com/x86/clwb
  // "The line may be retained in the cache hierarchy in non-modified state"
  // -> We never see the change that the other host makes so we just wait
  // forever. LFENCE does not change that.
  uint64_t f = s->flag;
  while (f != v) {
    _mm_clwb(s);
    _mm_lfence();
    f = s->flag;
    spins++;
  }
}

void await_flag(shared_t *s, uint64_t v) {
  switch (load_variant) {
  case 0:
    do {
    } while (s->flag != v);
    break;
  case 1:
    await_flag_clflush(s, v);
    break;
  case 2:
    await_flag_clflushopt(s, v);
    break;
  case 3:
    await_flag_clwb(s, v);
    break;
  case 10:
    await_flag_clflush_fence(s, v);
    break;
  case 11:
    await_flag_clflushopt_fence(s, v);
    break;
  case 12:
    await_flag_clwb_fence(s, v);
    break;
  default:
    fprintf(stderr, "invalid load_variant: '%u'\n", load_variant);
    exit(-1);
  }
}

// ------------------
// Loops
// ------------------

// Infinite loop to toggle the flag
void cxl_write(shared_t *s) {
  while (1) {
    write_to_cxl(s, 1);

    await_flag(s, 0);
  }
}

// Infinite loop to toggle the flag that also tracks how long it took
void cxl_poll_read(shared_t *s, const uint64_t iterTarget) {
  uint64_t runCounter = 0;
  while (runCounter < iterTarget) {
    // -- Flag toggle
    write_to_cxl(s, 0);

    // -- wait until flagged other host toggeld the flag
    await_flag(s, 1);

    runCounter++;
  }
}

void cxl_poll_read_perfcpp(shared_t *s) {
  /// Initialize the counter
  auto event_counter = perf::EventCounter{};

  /// Specify hardware events to count
  event_counter.add({
      "milliseconds",
      "ls_l1_d_tlb_miss.tlb_reload_1g_l2_hit",
      "ls_l1_d_tlb_miss.tlb_reload_1g_l2_miss",
      "ls_l1_d_tlb_miss.tlb_reload_2m_l2_hit",
      "ls_l1_d_tlb_miss.tlb_reload_2m_l2_miss",
      //"ls_l1_d_tlb_miss.tlb_reload_4k_l2_hit","ls_l1_d_tlb_miss.tlb_reload_4k_l2_miss",
      // Counter that might give insight into actual CXL latency -> They seem to
      // not work
      "l3_xi_sampled_latency.ext_far",
      "l3_xi_sampled_latency_requests.ext_far",
      "l3_xi_sampled_latency.ext_near",
      "l3_xi_sampled_latency_requests.ext_near",
      // Other maybe interesting counter
      //"l3_xi_sampled_latency.dram_far",
      //"l3_xi_sampled_latency_requests.dram_far",
      //"l3_xi_sampled_latency.dram_near",
      //"l3_xi_sampled_latency_requests.dram_near",
      //"l3_xi_sampled_latency.far_cache",
      //"l3_xi_sampled_latency_requests.far_cache",
      //"l3_xi_sampled_latency.near_cache",
      //"l3_xi_sampled_latency_requests.near_cache",
      //"l3_xi_sampled_latency.all", "l3_xi_sampled_latency_requests.all"
  });

  printf("cxl_poll_read: load = %u, store = %u, iter_taget = %lu"
         " warmup = %lu"
         "\n",
         load_variant, store_variant, iter_target, warmup_target);

  // Do a little warming before actually measuring
  if (warmup_target != 0) {
    cxl_poll_read(s, warmup_target);
  }

  /// Run the workload
  event_counter.start();

  cxl_poll_read(s, iter_target);

  event_counter.stop();

  /// Print the result to the console
  double xi_lat_all, xi_lat_all_req;
  double xi_lat_dram, xi_lat_dram_req;
  double xi_lat_ext_all, xi_lat_ext_all_req;
  const auto result = event_counter.result();
  for (const auto [event_name, value] : result) {
    std::cout << event_name << ": " << uint64_t(value);
    if (event_name == "milliseconds") {
      std::cout << " -> " << (value * 1000000) / iter_target << " ns/iter";
    }
    if (event_name == "l3_xi_sampled_latency.all") {
      xi_lat_all = value;
    }
    if (event_name == "l3_xi_sampled_latency_requests.all") {
      xi_lat_all_req = value;
    }
    if (event_name == "l3_xi_sampled_latency.dram_near") {
      xi_lat_dram = value;
    }
    if (event_name == "l3_xi_sampled_latency_requests.dram_near") {
      xi_lat_dram_req = value;
    }
    std::cout << std::endl;
  }

  // std::cout << "All: " << xi_lat_all / xi_lat_all_req * 10 <<
  // "ns"<<std::endl; std::cout << "DRAM: " << xi_lat_dram / xi_lat_dram_req *
  // 10 << "ns"<<std::endl;
  std::cout << "spins_per_iter: " << spins / iter_target << std::endl;
}

static inline uint64_t now_nsec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * ((uint64_t)1000 * 1000 * 1000) + ts.tv_nsec;
}

uint64_t sync_me(void *cxl_base_addr) {
  uint64_t whoAmI = 0;     // global for ease of use
  uint64_t s = now_nsec(); // used as random starting point
  s += whoAmI * 1000;
  if (s == 0) {
    s = 1; // don't use 0
  }
  printf("Waiting for sync... (Token %lu)\n", s);

  // We need three cachelines
  uint64_t *mem_cl0 = (uint64_t *)cxl_base_addr;
  uint64_t *mem_cl1 = ((uint64_t *)cxl_base_addr) + 16;
  uint64_t *mem_cl2 = ((uint64_t *)cxl_base_addr) + 32;

  cxl_write_b1(mem_cl1, 0);
  cxl_write_b1(mem_cl2, 0);
  cxl_write_b1(mem_cl0, s);

  uint64_t t;
  // Wait until "other" changes the value
  do {
    t = cxl_read_b1(mem_cl0);
  } while (t == s);

  // Now we know who "we" are
  if (t != 0) {
    // write 0 to release the "other"
    cxl_write_b1(mem_cl0, 0);
    whoAmI = 0;
  } else {
    whoAmI = 1;
  }
  printf("Synced: i am > %lu < (t = %lu)\n", whoAmI, t);

  // Set "our" cacheline to 1
  cxl_write_b1(whoAmI ? mem_cl1 : mem_cl2, 1);
  // Wait until "other" is also 1
  while (!(cxl_read_b1(mem_cl1) & cxl_read_b1(mem_cl2))) {
  }

  printf("Sync complete\n");

  return whoAmI;
}

void (*lock_init)(void *shmBase, uint8_t id) = NULL;
void cxl_lock_test(shared_t *s) {
  // Params
  char *setting = getenv("N");
  ;
  size_t N = iter_target;
  if (setting) {
    N = atoi(setting);
  }
  uint64_t steps[N / 2];
  printf("cxl_lock_test(N=%lu)\n", N);

  uint64_t *mem = (uint64_t *)(((char *)s) + (2 << 20)); // 2MB offset

  uint64_t id = sync_me(mem);

  // Lock init
  assert(lock_init != NULL);
  lock_init(s, id);
  assert(lock != NULL);
  assert(unlock != NULL);

  // Bench
  printf("Bench loop...\n");
  uint64_t ts = now_nsec();
  for (size_t i = 0; i < N / 2; ++i) {
    lock(s, id);

    steps[i] = cxl_read_b1(mem);
    cxl_write_b1(mem, steps[i] + 1);

    unlock(s, id);
  }
  uint64_t te = now_nsec();

  usleep(200000);
  uint64_t final = cxl_read_b1(mem);
  usleep(200000);

  printf("time to %lu : %lu ns (lower is better)\n", N, te - ts);
  printf("%f ns / per increment \n", ((double)(te - ts)) / ((double)(N)));
  printf("final %lu , %s \n", final, (final == N) ? "OK   " : "ERROR");

  if (final != N) {
    for (size_t i = 0; i < N / 2 && i < 50; ++i) {
      printf("%lu ", steps[i]);
    }
    printf("\n");
    if (final != N) {
      exit(-1);
    }
  }
}

// ====================
// Main
// ====================

int main(int argc, char *argv[]) {
  char *p;

  void *mem = init_dax(4 * 1024 * 1024);
  shared_t *ptr = (shared_t *)mem;

  // pin
  cpu_set_t my_set;
  CPU_ZERO(&my_set);
  CPU_SET(42, &my_set);
  sched_setaffinity(0, sizeof(cpu_set_t), &my_set);

  // Parse options
  int opt;
  void (*fn)(shared_t *) = NULL;
  while ((opt = getopt(argc, argv, "abs:l:t:w:m:")) != -1) {
    switch (opt) {
    // Parameter
    case 's':
      store_variant = strtoul(optarg, &p, 0);
      break;
    case 'l':
      load_variant = strtoul(optarg, &p, 0);
      break;
    case 'w':
      warmup_target = strtoul(optarg, &p, 0);
      break;
    case 't':
      iter_target = strtoul(optarg, &p, 0);
      break;
    // Variant
    case 'a':
      fn = cxl_write;
      break;
    case 'b':
      fn = cxl_poll_read_perfcpp;
      break;
    case 'm':
      int v = strtoul(optarg, &p, 0);
      if (v == 1) {
        lock_init = select_dekker;
      } else {
        lock_init = select_peterson;
      }
      fn = cxl_lock_test;
    }
  }
  if (ptr == NULL) {
    perror("No function selected to run.");
    exit(-1);
  } else {
    fn(ptr);
  }

  return 0;
}
