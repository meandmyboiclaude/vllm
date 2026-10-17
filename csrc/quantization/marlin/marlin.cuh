#pragma once

#ifndef _marlin_cuh
  #define _marlin_cuh
  #include <cuda.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
  #include <iostream>

  #ifndef MARLIN_NAMESPACE_NAME
    #define MARLIN_NAMESPACE_NAME marlin
  #endif

  // [0045 L2PF] 1 ONLY in a twin TU (generate_kernels.py L2PF_FAMILIES -> sm80_kernel_l2pf*_<c_type>.cu), defined before it
  // includes kernel.h: it defines cp_async4_stream_l2pf below, which marlin_template.h's B-stream copy then calls. With
  // 0 (every other TU) the token stream of this header is unchanged.
  #ifndef MARLIN_L2PF_TWIN
    #define MARLIN_L2PF_TWIN 0
  #endif

namespace MARLIN_NAMESPACE_NAME {

// Marlin params

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages =
    4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int32_t*>(smem_ptr)[0] =
        reinterpret_cast<const int32_t*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  if (pred) {
    reinterpret_cast<int64_t*>(smem_ptr)[0] =
        reinterpret_cast<const int64_t*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  if (pred) {
    reinterpret_cast<int4*>(smem_ptr)[0] =
        reinterpret_cast<const int4*>(glob_ptr)[0];
  }
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  reinterpret_cast<int4*>(smem_ptr)[0] =
      reinterpret_cast<const int4*>(glob_ptr)[0];
}

__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  cp_async4(smem_ptr, glob_ptr);
}

    #if MARLIN_L2PF_TWIN
__device__ inline void cp_async4_stream_l2pf(void* smem_ptr, const void* glob_ptr) {
  cp_async4(smem_ptr, glob_ptr);
}
    #endif

__device__ inline void cp_async_fence() {}

template <int n>
__device__ inline void cp_async_wait() {}

  #else

__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 8;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

// Read-once global->shared copy for the weight stream: the line is filled
// L2::evict_first, so streaming a layer's weights through L2 evicts the weight
// lines already consumed instead of lines other kernels still hold (a dirty
// line evicted mid-GEMM costs a DRAM write-back inside the GEMM). The values
// copied are exactly cp_async4's; only the L2 replacement priority differs.
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
      "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

    #if MARLIN_L2PF_TWIN
// [0045 L2PF] cp_async4_stream with the .L2::256B prefetch-size qualifier (PTX ISA cp.async .level::prefetch_size, sm_80+):
// a hint that L2 fetch 256 B around the copied 16 B when it misses, so the weight stream reaches DRAM in larger requests.
// Same copy, same evict_first policy, same commit group: the 16 bytes that land in shared memory are cp_async4_stream's.
// Called ONLY by marlin_template.h's B-stream copy in L2PF twin TUs (SASS: LDGSTS.E.BYPASS.LTC256B.128 ... desc[]).
__device__ inline void cp_async4_stream_l2pf(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .b64 p;\n"
      "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
      "   cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], %2, p;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}
    #endif

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

// Invalidate one 128-byte L2 line WITHOUT writing it back (sm_80+). The line's
// contents become indeterminate, so call it only on data nothing reads before
// rewriting it.
__device__ inline void l2_discard_line(const void* p) {
  asm volatile("discard.global.L2 [%0], 128;\n" ::"l"(p) : "memory");
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

  #endif

}  // namespace MARLIN_NAMESPACE_NAME

#endif
