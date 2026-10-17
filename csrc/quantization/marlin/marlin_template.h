/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#ifndef MARLIN_IRED
  #error "[0056 IRED] MARLIN_IRED undefined: include kernel.h before marlin_template.h"
#endif
#ifndef MARLIN_IRED_V3_HELPERS
  #define MARLIN_IRED_V3_HELPERS 1
// [0056 IRED v3] int4 lanes by VALUE and float-register bit casts only (v2p's punned unsigned* into an int4 was miscompiled: RID 835).
__device__ __forceinline__ unsigned ired_lane(const int4 v, const int f) {
  return static_cast<unsigned>(f == 0 ? v.x : f == 1 ? v.y : f == 2 ? v.z : v.w);
}
template <typename FragCT>
__device__ __forceinline__ void ired_add4(FragCT& acc, const int4 a) {
  #pragma unroll
  for (int f = 0; f < 4; f++) acc[f] = __uint_as_float(__float_as_uint(acc[f]) + ired_lane(a, f));
}
template <typename FragCT>
__device__ __forceinline__ void ired_add4(FragCT& acc, const int4 a, const int4 b) {
  #pragma unroll
  for (int f = 0; f < 4; f++) acc[f] = __uint_as_float(__float_as_uint(acc[f]) + (ired_lane(a, f) + ired_lane(b, f)));
}
#endif
#include "marlin_dtypes.cuh"
#include "dequant.h"
#include "marlin_mma.h"
#include "core/scalar_type.hpp"
// [0055 KU3S] the LOWBIT int3 unpack (lowbit_unpack.cuh 61472f0052c54b60, verbatim): forceinline device functions only
#include "lowbit_unpack.cuh"

// [0036 XS-5] XS3: the cross-slice prefetch (XS) is compiled ONLY into the twin kernel MarlinXS<...>, defined when a
// translation unit sets MARLIN_XS_TWIN 1 before including this header (generate_kernels.py writes exactly one such TU,
// sm80_kernel_xs_<c_type>.cu, for the allow-listed configs). Every other TU compiles MARLIN_XS_TWIN 0: the template below
// is then the served Marlin<...> token for token (every XS line sits under `#if MARLIN_XS_TWIN`), so its SASS equals the
// served build. XS2 instead compiled XS into every instantiation (MARLIN_XSLICE 1) and switched it at run time, which
// changed 1164 of 1175 served Marlin bodies and slowed the OFF launches (XS2 card K3 FAIL-REGRESSION, 10-03).
#ifndef MARLIN_XS_TWIN
  #define MARLIN_XS_TWIN 0
#endif
#if MARLIN_XS_TWIN
  #define MARLIN_XS_KNAME MarlinXS
#else
  #define MARLIN_XS_KNAME Marlin
#endif

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

// [0040 OPR] The ordered-parallel-reduce twin MarlinOPR<...> is compiled ONLY by the TU that defines MARLIN_OPR_TWIN 1 before
// including this header (generate_kernels.py OPR_TWINS -> sm80_kernel_opr_<c_type>.cu). Every OPR line sits under
// `#if MARLIN_OPR_TWIN`, so the served Marlin<...> and the MarlinXS twins keep their exact token streams. The OPR code
// does not use sh_red and waits on no cp.async group, so a TU may combine it with MARLIN_XS_TWIN (that TU names its kernel).
#ifndef MARLIN_OPR_TWIN
  #define MARLIN_OPR_TWIN 0
#endif
#if MARLIN_OPR_TWIN && !MARLIN_XS_TWIN
  #undef MARLIN_XS_KNAME
  #define MARLIN_XS_KNAME MarlinOPR
#elif MARLIN_OPR_TWIN
  // [0039 XSOPR] A TU that sets both MARLIN_XS_TWIN 1 and MARLIN_OPR_TWIN 1 (generate_kernels.py XSOPR_TWINS ->
  // sm80_kernel_xsopr_<c_type>.cu) compiles the combined twin MarlinXSOPR<...>: the XS cross-slice prefetch plus the 0040 OPR.
  #undef MARLIN_XS_KNAME
  #define MARLIN_XS_KNAME MarlinXSOPR
#endif

namespace MARLIN_NAMESPACE_NAME {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

template <typename scalar_t,  // compute dtype, half or nv_float16
          const vllm::ScalarTypeId b_type_id,  // weight MarlinScalarType id
          const vllm::ScalarTypeId s_type_id,  // weight scale ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void MARLIN_XS_KNAME(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    int prob_m,                           // batch dimension m
    int prob_n,                           // output dimension n
    int prob_k,                           // reduction dimension k
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {}

}  // namespace marlin

#else

// Instruction for loading a full 16x16 matrix fragment of operand A from shared
// memory, directly in tensor core layout.
template <int count, vllm::ScalarTypeId type_id>
__device__ inline void ldsm(typename MarlinScalarType<type_id>::FragA& frag_a,
                            const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (count == 4) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));
  } else if constexpr (count == 2) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(a[0]), "=r"(a[1])
                 : "r"(smem));
  } else if constexpr (count == 1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(a[0])
                 : "r"(smem));
  } else {
    static_assert(count == 1 || count == 2 || count == 4, "invalid count");
  }
}

// Multiply dequantized values by the corresponding quantization scale; used
// only for grouped quantization.
template <vllm::ScalarTypeId type_id>
__device__ inline void scale(typename MarlinScalarType<type_id>::FragB& frag_b,
                             typename MarlinScalarType<type_id>::FragS& frag_s,
                             int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_s)[i]);
  frag_b[0] = __hmul2(frag_b[0], s);
  frag_b[1] = __hmul2(frag_b[1], s);
}

template <vllm::ScalarTypeId type_id>
__device__ inline void scale_and_sub(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t s,
    typename MarlinScalarType<type_id>::scalar_t zp) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 s2 = MarlinScalarType<type_id>::num2num2(s);
  scalar_t2 zp2 = MarlinScalarType<type_id>::num2num2(zp);
  frag_b[0] = __hfma2(frag_b[0], s2, __hneg2(zp2));
  frag_b[1] = __hfma2(frag_b[1], s2, __hneg2(zp2));
}

template <vllm::ScalarTypeId type_id>
__device__ inline void sub_zp(
    typename MarlinScalarType<type_id>::FragB& frag_b,
    typename MarlinScalarType<type_id>::scalar_t2& frag_zp, int i) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<type_id>::scalar_t2;
  scalar_t2 zp = MarlinScalarType<type_id>::num2num2(
      reinterpret_cast<scalar_t*>(&frag_zp)[i]);
  frag_b[0] = __hsub2(frag_b[0], zp);
  frag_b[1] = __hsub2(frag_b[1], zp);
}

// Given 2 floats multiply by 2 scales (halves)
template <vllm::ScalarTypeId type_id>
__device__ inline void scale_float(
    float* c, typename MarlinScalarType<type_id>::FragS& s) {
  using scalar_t = typename MarlinScalarType<type_id>::scalar_t;
  scalar_t* s_ptr = reinterpret_cast<scalar_t*>(&s);
  c[0] = __fmul_rn(c[0], MarlinScalarType<type_id>::num2float(s_ptr[0]));
  c[1] = __fmul_rn(c[1], MarlinScalarType<type_id>::num2float(s_ptr[1]));
}

#if MARLIN_G2
// [0071 G2 cuda 2026-10-12, BUG-562 guard] Split-K lock-leftover DETECTOR. Contract (stream order; DESIGN-G2 §2): every lock word
// of a layer workspace is 0 when a launch starts (zeroed at allocation, and each split column's final slice resets its word), and
// inside a launch exactly one column chain moves a given word: 0 -> 1 -> ... -> slice_count - 1 (+1 per non-final slice, in slice
// order) -> 0 (the final's reset). A slice waiting for `count` (its slice_idx; slice_count - 1 for the OPR final) therefore reads only
// 0 .. count. A first read above count (unsigned, so a negative word counts as above) is a value its own chain never wrote: the leftover of
// an aborted chain, or a second launch on the same workspace from another stream (BUG-562, RID 849/850: every later call of that layer
// silently WRONG, then a hang). The hit path writes only this module's hit count and the mapped host page, never a lock or an output.
// Per module (kernel TU): the host writes mode + slot once, at the module's first Marlin launch on a device (marlin.cu
// kt_g2_launch, through a private stream: never into a capture). Variable templates give the three words the same symbol name in
// every module (_ZN<ns>14marlin_g2_modeILi0EEE ...), which is how the host finds them in the module a launch uses.
template <int I>
__constant__ unsigned int marlin_g2_mode;  // 0 OFF / not armed yet (the served spin loop) | 1 DETECT | 2 TRAP
template <int I>
__constant__ unsigned long long marlin_g2_slot;  // this module's 16-B slot in the mapped host page (page = slot & ~4095)
template <int I>
__device__ unsigned int marlin_g2_hits;  // hits in this module (marlin_g2_read_reset)

static __device__ __forceinline__ void marlin_g2_hit(const int* lock, int state, int count) {
  const unsigned long long slot_addr = marlin_g2_slot<0>;  // written before the mode, in stream order: 0 is a defensive guard only
  if (atomicAdd(&marlin_g2_hits<0>, 1u) == 0u && slot_addr != 0ull) {  // the module's first hit fills its slot
    volatile unsigned int* slot = reinterpret_cast<volatile unsigned int*>(slot_addr);
    slot[1] = static_cast<unsigned int>(state);
    slot[2] = static_cast<unsigned int>(count);
    slot[3] = static_cast<unsigned int>(reinterpret_cast<unsigned long long>(lock));
    __threadfence_system();
    slot[0] = 0x4B4C3247u;  // "G2LK": the slot is valid
    __threadfence_system();
  }
  if (slot_addr != 0ull) {
    *reinterpret_cast<volatile unsigned int*>(slot_addr & ~4095ull) = 1u;  // the page's "any hit" word: one host load per marlin_gemm call
    __threadfence_system();
  }
  if (marlin_g2_mode<0> == 2u) __trap();  // VLLM_MARLIN_G2=2: stop loudly instead of leaving the layer silently wrong
}
#endif

// Wait until barrier reaches `count`, then lock for current threadblock.
__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
#if MARLIN_G2
    // [0071 G2 v6] runtime switch: one uniform constant-bank compare per acquire; 0 (OFF, or the module not armed yet) runs the
    // served loop in the else branch, verbatim and with that branch as its only entry (v5's goto gave the served loop a second
    // predecessor, and ptxas then reloaded spilled lock pointer/count at the loop head on every read). Armed: ONE check, on the
    // acquire's first lock read (a leftover of an earlier launch, or a second launch on this workspace from another stream, is already
    // in the word then), then its own plain spin with no G2 instruction per read; a word already at `count` skips it.
    if (marlin_g2_mode<0> != 0u) {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
      if (static_cast<unsigned int>(state) > static_cast<unsigned int>(count)) marlin_g2_hit(lock, state, count);
      if (state != count)
        do
          asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        while (state != count);
    } else
#endif
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}

// Release barrier and increment visitation count.
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    // Make sure that all writes since acquiring this barrier are visible
    // globally, while releasing the barrier.
    asm volatile("fence.acq_rel.gpu;\n");
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(val));
  }
}

// Wait until value of lock to be negative, and then add 1
__device__ inline void wait_negative_and_add(int* lock) {
  if (threadIdx.x == 0) {
    int state = 0;
    do
      // Guarantee that subsequent writes by this threadblock will be visible
      // globally.
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    while (state >= 0);
    atomicAdd(lock, 1);
  }
  __syncthreads();
}

// [0041 MR14] Group-scale L2 prefetch twins. A TU that defines MARLIN_SP_TWIN 1 before including this header
// (generate_kernels.py SP_TWINS / SP_FAMILIES -> sm80_kernel_sp*_<c_type>.cu) compiles the kernel of its family (the served
// Marlin, MarlinXS, MarlinOPR, MarlinXSOPR) with ONE addition: its first statement calls sp_prefetch_stripe, which issues
// prefetch.global.L2 for the group-scale and grouped zero-point rows of every k-group the CTA's stripe will read, so the
// late groups' cp.async hit L2 instead of DRAM. A prefetch is a cache hint (nothing reaches registers or smem, no cp.async
// group, no barrier), so the twin computes exactly what its family kernel computes. Every MR14 line sits under
// `#if MARLIN_SP_TWIN`; with 0 (every other TU) the token stream is unchanged. The flag composes with every family macro:
// the kernel name below is the family name + SP, so a family needs one SP TU and no new template branch (no 2^n twins).
#ifndef MARLIN_SP_TWIN
  #define MARLIN_SP_TWIN 0
#endif
#if MARLIN_SP_TWIN
  #undef MARLIN_XS_KNAME
  #if MARLIN_XS_TWIN && MARLIN_OPR_TWIN
    #define MARLIN_XS_KNAME MarlinXSOPRSP
  #elif MARLIN_XS_TWIN
    #define MARLIN_XS_KNAME MarlinXSSP
  #elif MARLIN_OPR_TWIN
    #define MARLIN_XS_KNAME MarlinOPRSP
  #else
    #define MARLIN_XS_KNAME MarlinSP
  #endif
// One 128-B L2 line. evict_last (host mode 1) ranks the line behind evict_normal and evict_first victims (the served B
// stream is evict_first); evict_normal (mode 2) leaves no high-priority line behind once the GEMM is done.
__device__ __forceinline__ void sp_prefetch_l2(const int4* p, bool evict_normal) {
  if (evict_normal)
    asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(p));
  else
    asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(p));
}
// The prefetch, out of line and called first in the kernel: nothing of the kernel is live across the call, so the kernel
// body after it compiles as its family's (ptxas allocates the callee on its own). It recomputes the kernel's partition from
// the launch arguments with the kernel's own expressions (parallel, k_tiles, n_tiles, part2 / part1, iters with the group
// rounding, the [0033] part-1 map on the same tile_major test) and walks the CTA's stripe as init_slice does: part-1 full-k
// tiles blockIdx.x + j * gridDim.x below part1_tiles, then the part-2 flat k-tile range [iters * b, iters * (b + 1)) of the
// part2_mn_tiles column tiles, cut at column ends. Per (k-group row g, column tile col) it prefetches the 128-B lines of
// scales_ptr[g * s_gl_stride + col * s_sh_stride + 0 .. s_sh_stride) and of the grouped zero points, the int4s
// fetch_to_shared's cp.async read for that group. Mode: tile_major >> 1, bit 1 = evict_normal (else evict_last).
template <int threads, int thread_m_blocks, int thread_n_blocks, int thread_k_blocks, int group_blocks,
          bool is_8bit_scale, int pack_factor, bool has_zp, bool is_zp_float>
// [0041 MR14-3] plain Marlin family: INLINE (the out-of-line CALL made ptxas save 2-3 kernel values across it on the 255-reg
// MarlinSP<kU4 g32 tmb4> / <kU4B8 g64 tmb2>); the XS / OPR / XSOPR twins keep the call (inline made the XS ones spill LESS than
// their siblings and did not fix OPR; spfix/mr14-3/all36). sp_twin_check v2 per family. [0041 MR14-5] The plain OPR
// twin no longer calls this function: it runs sp_walk_opr (below), so the switch matters for XS / XSOPR only.
  #if MARLIN_XS_TWIN || MARLIN_OPR_TWIN
    #define MARLIN_SP_FN_ATTR __noinline__
  #else
    #define MARLIN_SP_FN_ATTR __forceinline__
  #endif
__device__ MARLIN_SP_FN_ATTR void sp_prefetch_stripe(const int4* __restrict__ scales_ptr, const int4* __restrict__ zp_ptr,
                                                int prob_m, int prob_n, int prob_k, int tile_major) {
  static_assert(group_blocks > 0, "[0041 MR14] grouped scales only");
  const bool evict_normal = ((tile_major >> 1) & 2) != 0;
  constexpr int m_block_size = 16 * thread_m_blocks;
  int parallel = 1;
  if (prob_m > m_block_size) parallel = prob_m / m_block_size;
  const int k_tiles = prob_k / 16 / thread_k_blocks;
  const int n_tiles = prob_n / 16 / thread_n_blocks;
  const int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
  }
  const int part1_tiles = global_mn_tiles - part2_mn_tiles;
  const int p1_q = part1_tiles / n_tiles;
  const int p1_r = part1_tiles % n_tiles;
  int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);
  if (group_blocks >= thread_k_blocks)
    iters = (group_blocks / thread_k_blocks) * div_ceil(iters, (group_blocks / thread_k_blocks));
  constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
  constexpr int zp_sh_stride = is_zp_float ? 16 * thread_n_blocks / 8 : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int s_lines = div_ceil(s_sh_stride, 8);  // 128-B lines per (row, column tile)
  constexpr int zp_lines = has_zp ? div_ceil(zp_sh_stride, 8) : 0;
  const int s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);
  const int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  const int rows = prob_k / 16 / group_blocks;
  // k-tiles [r_lo, r_hi) of column tile col read the rows (thread_k_blocks * r) / group_blocks (+ the next s_tb_groups - 1
  // when a tile holds several groups) = [floor(tkb * r_lo / gb), ceil(tkb * r_hi / gb))
  auto col_rows = [&](int col, int r_lo, int r_hi) {
    int g_end = div_ceil(thread_k_blocks * r_hi, group_blocks);
    if (g_end > rows) g_end = rows;
  #pragma unroll 1
    for (int g = (thread_k_blocks * r_lo) / group_blocks + threadIdx.x; g < g_end; g += threads) {
      const int4* s = scales_ptr + g * s_gl_stride + col * s_sh_stride;
  #pragma unroll
      for (int l = 0; l < s_lines; l++) sp_prefetch_l2(s + 8 * l, evict_normal);
      if constexpr (zp_lines > 0) {
        const int4* z = zp_ptr + g * zp_gl_stride + col * zp_sh_stride;
  #pragma unroll
        for (int l = 0; l < zp_lines; l++) sp_prefetch_l2(z + 8 * l, evict_normal);
      }
    }
  };
  #pragma unroll 1
  for (int t = blockIdx.x; t < part1_tiles; t += gridDim.x) {
    int col;
    if (!tile_major)
      col = t % n_tiles;
    else if (t < p1_r * (p1_q + 1))
      col = t / (p1_q + 1);
    else  // reached only when p1_q >= 1 (as in init_part1_slice)
      col = p1_r + (t - p1_r * (p1_q + 1)) / p1_q;
    col_rows(col, 0, k_tiles);
  }
  const int b = blockIdx.x;
  int end = iters * (b + 1);
  if (end > k_tiles * part2_mn_tiles) end = k_tiles * part2_mn_tiles;
  #pragma unroll 1
  for (int f = iters * b; f < end;) {
    const int c = f / k_tiles;
    const int r = f - c * k_tiles;
    const int r_end = (r + (end - f) < k_tiles) ? r + (end - f) : k_tiles;
    col_rows((c + global_mn_tiles - part2_mn_tiles) % n_tiles, r, r_end);
    f += r_end - r;
  }
}
#endif
#if MARLIN_SP_TWIN && MARLIN_OPR_TWIN && !MARLIN_XS_TWIN
// [0041 MR14-5] The plain OPR family's SP twin (MarlinOPRSP) walks its stripe with this copy of sp_prefetch_stripe's body,
// inline as the kernel's first statement, with every input passed through an empty asm first. An empty asm emits no
// instruction and changes no value; it stops NVVM from sharing the walk's arithmetic (prob_n / 8, prob_n / 32, the partition
// terms, the special-register reads) with the kernel's, so the kernel body after the walk compiles as MarlinOPR's.
// MarlinOPRSP<kS8 kU4 bf16 128 4 4 8 g2> is then 255 regs / 0 B stack like its sibling (the out-of-line call: 255/16 with 3
// STL/LDL; inline without the asm: 255/8 with 2 -- ptxas spilled 2 of the 4 epilogue C-row offsets it hoists across the main
// loop), and every MarlinOPRSP passes sp_twin_check v2 against its MarlinOPR. Same lines, same order, same thread mapping as
// sp_prefetch_stripe (the body text is that function's, blockIdx / gridDim / threadIdx read into bx / gx / tx).
template <int threads, int thread_m_blocks, int thread_n_blocks, int thread_k_blocks, int group_blocks,
          bool is_8bit_scale, int pack_factor, bool has_zp, bool is_zp_float>
__device__ __forceinline__ void sp_walk_opr(const int4* __restrict__ scales_ptr, const int4* __restrict__ zp_ptr,
                                            int prob_m, int prob_n, int prob_k, int tile_major) {
  unsigned bx = blockIdx.x, gx = gridDim.x, tx = threadIdx.x;
  asm volatile("" : "+r"(prob_m), "+r"(prob_n), "+r"(prob_k), "+r"(tile_major));
  asm volatile("" : "+r"(bx), "+r"(gx), "+r"(tx));
  static_assert(group_blocks > 0, "[0041 MR14] grouped scales only");
  const bool evict_normal = ((tile_major >> 1) & 2) != 0;
  constexpr int m_block_size = 16 * thread_m_blocks;
  int parallel = 1;
  if (prob_m > m_block_size) parallel = prob_m / m_block_size;
  const int k_tiles = prob_k / 16 / thread_k_blocks;
  const int n_tiles = prob_n / 16 / thread_n_blocks;
  const int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  if (global_mn_tiles > gx) {
    part2_mn_tiles = global_mn_tiles % gx;
    if (part2_mn_tiles * 3 <= gx) part2_mn_tiles += gx;
  }
  const int part1_tiles = global_mn_tiles - part2_mn_tiles;
  const int p1_q = part1_tiles / n_tiles;
  const int p1_r = part1_tiles % n_tiles;
  int iters = div_ceil(k_tiles * part2_mn_tiles, gx);
  if (group_blocks >= thread_k_blocks)
    iters = (group_blocks / thread_k_blocks) * div_ceil(iters, (group_blocks / thread_k_blocks));
  constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
  constexpr int zp_sh_stride = is_zp_float ? 16 * thread_n_blocks / 8 : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int s_lines = div_ceil(s_sh_stride, 8);  // 128-B lines per (row, column tile)
  constexpr int zp_lines = has_zp ? div_ceil(zp_sh_stride, 8) : 0;
  const int s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);
  const int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  const int rows = prob_k / 16 / group_blocks;
  // k-tiles [r_lo, r_hi) of column tile col read the rows (thread_k_blocks * r) / group_blocks (+ the next s_tb_groups - 1
  // when a tile holds several groups) = [floor(tkb * r_lo / gb), ceil(tkb * r_hi / gb))
  auto col_rows = [&](int col, int r_lo, int r_hi) {
    int g_end = div_ceil(thread_k_blocks * r_hi, group_blocks);
    if (g_end > rows) g_end = rows;
  #pragma unroll 1
    for (int g = (thread_k_blocks * r_lo) / group_blocks + tx; g < g_end; g += threads) {
      const int4* s = scales_ptr + g * s_gl_stride + col * s_sh_stride;
  #pragma unroll
      for (int l = 0; l < s_lines; l++) sp_prefetch_l2(s + 8 * l, evict_normal);
      if constexpr (zp_lines > 0) {
        const int4* z = zp_ptr + g * zp_gl_stride + col * zp_sh_stride;
  #pragma unroll
        for (int l = 0; l < zp_lines; l++) sp_prefetch_l2(z + 8 * l, evict_normal);
      }
    }
  };
  #pragma unroll 1
  for (int t = bx; t < part1_tiles; t += gx) {
    int col;
    if (!tile_major)
      col = t % n_tiles;
    else if (t < p1_r * (p1_q + 1))
      col = t / (p1_q + 1);
    else  // reached only when p1_q >= 1 (as in init_part1_slice)
      col = p1_r + (t - p1_r * (p1_q + 1)) / p1_q;
    col_rows(col, 0, k_tiles);
  }
  const int b = bx;
  int end = iters * (b + 1);
  if (end > k_tiles * part2_mn_tiles) end = k_tiles * part2_mn_tiles;
  #pragma unroll 1
  for (int f = iters * b; f < end;) {
    const int c = f / k_tiles;
    const int r = f - c * k_tiles;
    const int r_end = (r + (end - f) < k_tiles) ? r + (end - f) : k_tiles;
    col_rows((c + global_mn_tiles - part2_mn_tiles) % n_tiles, r, r_end);
    f += r_end - r;
  }
}
#endif

// [0042 FOLD] Remainder-fold twins. A TU that defines MARLIN_FOLD_TWIN 1 before including this header (generate_kernels.py
// FOLD_TWINS -> sm80_kernel_fold_<c_type>.cu) compiles MarlinFOLD<...>: the served kernel whose LAST 64-row problem copy may
// start at prob_m - 64 instead of (parallel - 1) * 64, so marlin_mm can fold a GEMM's < 64-row remainder into the launch of
// its 64-row copies (VLLM_MARLIN_REMAINDER_FOLD_NK) instead of giving it a second launch that streams the whole weight
// matrix again. The folded copy overlaps its predecessor: it computes the overlap rows, but its write_result skips them,
// so every output row is written once, by the copy that owns it. For a prob_m that is a multiple of 64 or <= 64 (every
// launch the served host makes) row_off(p) == p * 64 and nothing is skipped. Every FOLD line sits under
// `#if MARLIN_FOLD_TWIN`; with 0 (every other TU) the token stream is unchanged. A fold has >= 2 problem copies and the
// XS, OPR and SP twins run exactly one, so the macro combines with none of them.
#ifndef MARLIN_FOLD_TWIN
  #define MARLIN_FOLD_TWIN 0
#endif
#if MARLIN_FOLD_TWIN
  #if MARLIN_XS_TWIN || MARLIN_OPR_TWIN || MARLIN_SP_TWIN
    #error "[0042 FOLD] MarlinFOLD combines with no other twin family (a fold has >= 2 problem copies, XS/OPR/SP exactly 1)"
  #endif
  #undef MARLIN_XS_KNAME
  #define MARLIN_XS_KNAME MarlinFOLD
#endif

// [0048 FOLDS] Served-partition remainder fold. A TU that defines MARLIN_FOLDS_TWIN 1 before including this header
// (generate_kernels.py FOLDS_TWINS -> sm80_kernel_folds_<c_type>.cu) compiles MarlinFOLDS<...>: the served kernel of the launch
// marlin_mm makes for par_count 64-row copies (launch A), which ALSO computes the < 64-row remainder that the served host gives a
// second launch (launch B). prob_m = 64 * p + r (0 < r < 64): every CTA runs two ITEMS, each a walk of the served partition with
// blockIdx.x replaced by the item's virtual block fs_vb:
//   A item = launch A's stripe fs_vb (parallel p, the served lock slots [0, gridDim.x)): every copy tile gets launch A's split-K
//            slices, slice_idx order and partials, so rows [0, 64 * p) are launch A's bit for bit (the arithmetic is untouched);
//   R item = the stripe fs_vb of ONE 64-row copy at rows [prob_m - 64, prob_m) (parallel 1, lock / C_tmp slots shifted by
//            gridDim.x); its write_result skips the 64 - r rows copy p - 1 owns.
// Schedule (host, tile_major bits 1-2): 0 AR = A item then R item (fs_vb = blockIdx.x for both); 1 RA = R then A; 2 PAIR (p == 1
// only, even grid): CTAs [0, G/2) run A stripes x and x + G/2, CTAs [G/2, G) the R stripes of the same two, in lockstep with
// their partner, so every B tile is streamed once and read twice. Each item's lock chain is the served one (its waits target
// first segments of higher virtual blocks of the same item kind), so with every CTA resident no wait can close a cycle.
// Every FOLDS line sits under `#if MARLIN_FOLDS_TWIN`; with 0 (every other TU) the token stream is unchanged.
#ifndef MARLIN_FOLDS_TWIN
  #define MARLIN_FOLDS_TWIN 0
#endif
#if MARLIN_FOLDS_TWIN
  #if MARLIN_XS_TWIN || MARLIN_OPR_TWIN || MARLIN_SP_TWIN || MARLIN_FOLD_TWIN
    #error "[0048 FOLDS] MarlinFOLDS combines with no other twin family"
  #endif
  #undef MARLIN_XS_KNAME
  #define MARLIN_XS_KNAME MarlinFOLDS
#endif

// [0043 TAILPF] Next-launch prefetch twins ("software PDL": PDL / griddepcontrol overlap needs sm_90; we serve on sm_89). A TU
// that defines MARLIN_TPF_TWIN 1 before including this header (generate_kernels.py TPF_TWINS / TPF_FAMILIES ->
// sm80_kernel_tpf*_<c_type>.cu) compiles the kernel of its family (the served Marlin, MarlinXS, MarlinOPR, MarlinXSOPR) with
// a by-value FIRST parameter `tpf` (MarlinTpfDesc, tailpf_desc.h) and ONE addition, its last statement: after the main loop
// and the CTA's last epilogue every CTA issues prefetch.global.L2 for its share of the NEXT Marlin launch's group scales and
// grouped zero points and (host mode 2) for that launch's first B stages, as marlin.cu describes them in `tpf`. A prefetch is
// a cache hint (nothing reaches registers or smem, no cp.async group, no barrier, no write), so the twin computes exactly
// what its family kernel computes. Every TAILPF line sits under `#if MARLIN_TPF_TWIN`; with 0 (every other TU) the token
// stream is unchanged. The flag composes with every family macro except MR14's SP (the alternative for the same lines) and
// FOLD (prefill only): the name below is the family name + TPF, one TU per family, no new template branch.
#ifndef MARLIN_TPF_TWIN
  #define MARLIN_TPF_TWIN 0
#endif
// [0063 KU3STPF cuda 2026-10-11] kU3S TPF twins. A TU that defines MARLIN_TPF_KU3S 1 next to MARLIN_TPF_TWIN 1 (generate_kernels.py
// generate_ku3stpf_twins -> sm80_kernel_tpf{,_xs}_u3s_<c_type>.cu; with MARLIN_CPF_TWIN 1 also sm80_kernel_cpftpf{,_xs}_u3s_s<k>_...)
// instantiates the TPF twin of the kS8 x kU3S family kernel (Marlin -> MarlinTPF, MarlinXS -> MarlinXSTPF; their CPF twins ->
// MarlinCPFTPF / MarlinXSCPFTPF): the 0055 body and TAILPF's tail, both unchanged. The flag does exactly two things: it exempts the
// TU from 0055's "kU3S has no twin" assert and asserts the opposite (the TU holds kU3S instances only). Every other TU defines 0,
// so its token stream is unchanged.
// MARLIN_TPF_KU3S_GC 1 (the TUs of the configs generate_kernels.py KU3STPF_GC lists): the twin takes its descriptor as a
// __grid_constant__ parameter and runs the same tail out of line (tpf_prefetch_next_gc below: __noinline__, placed after the kernel's
// last EXIT, reading the descriptor in place through a pointer). For those bodies the inline tail made ptxas allocate the body's
// registers differently (REG 189 -> 193, 178 -> 176, 177 -> 178); out of line the body keeps its sibling's REG / STACK.
#ifndef MARLIN_TPF_KU3S
  #define MARLIN_TPF_KU3S 0
#endif
#ifndef MARLIN_TPF_KU3S_GC
  #define MARLIN_TPF_KU3S_GC 0
#endif
#if (MARLIN_TPF_KU3S && !MARLIN_TPF_TWIN) || (MARLIN_TPF_KU3S_GC && !MARLIN_TPF_KU3S)
  #error "[0063 KU3STPF] MARLIN_TPF_KU3S needs MARLIN_TPF_TWIN; MARLIN_TPF_KU3S_GC needs MARLIN_TPF_KU3S"
#endif
#if MARLIN_TPF_TWIN
  #if MARLIN_SP_TWIN || MARLIN_FOLD_TWIN
    #error "[0043 TAILPF] a TPF twin wraps Marlin / MarlinXS / MarlinOPR / MarlinXSOPR only (not an SP or FOLD twin)"
  #endif
  #include "tailpf_desc.h"
  // The kernel definition below reads `__global__ void MARLIN_XS_KNAME(<the served parameters>) {`. For a TPF twin the name
  // expands to `<family>TPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST`, and the function-like MARLIN_TPF_REST then takes the
  // served parameter list (the source tokens that follow) as its arguments and gives it back with the closing parenthesis:
  // the twin's parameters are (tpf, <the served parameters>) and the served signature line is not edited.
  #define MARLIN_TPF_REST(...) __VA_ARGS__)
  #undef MARLIN_XS_KNAME
  #if MARLIN_XS_TWIN && MARLIN_OPR_TWIN
    #define MARLIN_XS_KNAME MarlinXSOPRTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
  #elif MARLIN_XS_TWIN && MARLIN_TPF_KU3S_GC
    #define MARLIN_XS_KNAME MarlinXSTPF(const __grid_constant__ MarlinTpfDesc tpf, MARLIN_TPF_REST
  #elif MARLIN_TPF_KU3S_GC
    #define MARLIN_XS_KNAME MarlinTPF(const __grid_constant__ MarlinTpfDesc tpf, MARLIN_TPF_REST
  #elif MARLIN_XS_TWIN
    #define MARLIN_XS_KNAME MarlinXSTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
  #elif MARLIN_OPR_TWIN
    #define MARLIN_XS_KNAME MarlinOPRTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
  #else
    #define MARLIN_XS_KNAME MarlinTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
  #endif
// One 128-B L2 line. evict_normal (default) leaves no high-priority line behind; evict_last (VLLM_MARLIN_TAIL_PREFETCH_EVICT=last,
// or auto for a next launch with > 48 MiB of B) keeps it ahead of the next GEMM's own B stream (plain LDGSTS, normal class).
__device__ __forceinline__ void tpf_prefetch_l2(const int4* p, int evict_last) {
  if (evict_last)
    asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(p));
  else
    asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(p));
}
// The twin's tail. (1) d.b != nullptr: for the CTAs yb = blockIdx.x, blockIdx.x + gridDim.x, ... of the NEXT launch, the
// first d.stages k-tiles of B that CTA reads (marlin_tpf_first_tile + marlin_tpf_b_line: rows [row * rows, (row + n) * rows)
// of column tile col, d.row_int4 int4s per row at stride d.gl_stride, the int4s fetch_to_shared's cp.async read). (2) the
// next launch's group-scale and zero-point lines, grid-strided over this launch's threads (thread blockIdx.x * threads +
// threadIdx.x takes lines l, l + gridDim.x * threads, ...: a warp issues 32 consecutive lines). Integer index math only:
// every quotient is marlin_tpf_q with a host-computed multiplier (no division / reciprocal sequence in the twin).
// The tail reads %ctaid.x / %nctaid.x / %tid.x through asm volatile, not blockIdx / gridDim / threadIdx: NVVM folds every
// blockIdx.x read into the kernel's entry read, so a tail use could keep that register live across the main loop; a volatile
// asm keeps its place after the main loop's volatile asm and is never merged with another read. (It does not rescue the four
// 255-register OPR / XSOPR tmb-4 bodies of generate_kernels.py TPF_EXCLUDE: there ONE prefetch instruction after the loop
// already moves ptxas's allocation, whatever the tail's form; those configs have no TPF twin.)
__device__ __forceinline__ int tpf_ctaid_x() {
  int v;
  asm volatile("mov.u32 %0, %%ctaid.x;\n" : "=r"(v));
  return v;
}
__device__ __forceinline__ int tpf_nctaid_x() {
  int v;
  asm volatile("mov.u32 %0, %%nctaid.x;\n" : "=r"(v));
  return v;
}
__device__ __forceinline__ int tpf_tid_x() {
  int v;
  asm volatile("mov.u32 %0, %%tid.x;\n" : "=r"(v));
  return v;
}
template <int threads>
__device__ __forceinline__ void tpf_prefetch_next(const MarlinTpfDesc& d) {
  const int bx = tpf_ctaid_x(), gx = tpf_nctaid_x(), tx = tpf_tid_x();
  if (d.b != nullptr) {
    const int lines_per_tile = d.rows * d.div_lpr.d;
  #pragma unroll 1
    for (int yb = bx; yb < d.grid; yb += gx) {
      const MarlinTpfTile t = marlin_tpf_first_tile(d, yb);
      const int lines = t.n * lines_per_tile;
  #pragma unroll 1
      for (int i = tx; i < lines; i += threads) tpf_prefetch_l2(d.b + marlin_tpf_b_line(d, t, i), d.evict_last);
    }
  }
  const int first = bx * threads + tx, step = gx * threads;
  if (d.s != nullptr) {
  #pragma unroll 1
    for (int l = first; l < d.s_lines; l += step) tpf_prefetch_l2(d.s + 8LL * l, d.evict_last);
  }
  if (d.zp != nullptr) {
  #pragma unroll 1
    for (int l = first; l < d.zp_lines; l += step) tpf_prefetch_l2(d.zp + 8LL * l, d.evict_last);
  }
}
  #if MARLIN_TPF_KU3S_GC
// [0063 KU3STPF] the same tail out of line (MARLIN_TPF_KU3S_GC), reading the __grid_constant__ descriptor in place (no copy)
template <int threads>
__device__ __noinline__ void tpf_prefetch_next_gc(const MarlinTpfDesc* d) {
  tpf_prefetch_next<threads>(*d);
}
  #endif
#endif

// [0061 COLFPF v2] out-of-line prefetch functions of the CPF sites that call one; each text is the screened variant's
// verbatim ([exp <variant>] tags = the placement experiments of REPORT-COLFPF-V2.md)
#if MARLIN_CPF_TWIN && (MARLIN_CPF_SITE == 1)  // [0061 COLFPF v2] out-of-line prefetch of site(s) 1
// [exp f_asm]
template <int threads>
__device__ __noinline__ void colf_pf_f(const float* global_scale_ptr, int prob_n) {
  asm volatile(
        "{\n"
        ".reg .u32 %%cpf0, %%cpf1, %%cpf2;\n"
        ".reg .u64 %%cpfa;\n"
        ".reg .pred %%cpfp, %%cpfq;\n"
        "mov.u32 %%cpf0, %%ctaid.x;\n"
        "mov.u32 %%cpf1, %%tid.x;\n"
        "mad.lo.u32 %%cpf0, %%cpf0, %2, %%cpf1;\n"
        "add.u32 %%cpf2, %1, 31;\n"
        "shr.u32 %%cpf2, %%cpf2, 5;\n"
        "setp.lt.u32 %%cpfp, %%cpf0, %%cpf2;\n"
        "setp.ne.and.u64 %%cpfq, %0, 0, %%cpfp;\n"
        "mul.wide.u32 %%cpfa, %%cpf0, 128;\n"
        "add.u64 %%cpfa, %%cpfa, %0;\n"
        "@%%cpfq prefetch.global.L2::evict_normal [%%cpfa];\n"
        "}\n" ::"l"(global_scale_ptr), "r"(prob_n), "n"(threads));
}
#endif
#if MARLIN_CPF_TWIN && (MARLIN_CPF_SITE == 4 || MARLIN_CPF_SITE == 10)  // [0061 COLFPF v2] out-of-line prefetch of site(s) 4, 10
// [exp ent_noinl] out-of-line prologue prefetch
template <int threads>
__device__ __noinline__ void colf_pf_noinl(const float* p, int n) {
  int bx, tx;
  asm volatile("mov.u32 %0, %%ctaid.x;\n" : "=r"(bx));
  asm volatile("mov.u32 %0, %%tid.x;\n" : "=r"(tx));
  const int l = bx * threads + tx;
  if (p != nullptr && l < (n + 31) / 32) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(p + 32 * l));
}
#endif
#if MARLIN_CPF_TWIN && (MARLIN_CPF_SITE == 6)  // [0061 COLFPF v2] out-of-line prefetch of site(s) 6
// [exp f_char]
template <int threads>
__device__ __noinline__ void colf_pf_f(const float* p, int n) {
  const int l = blockIdx.x * threads + threadIdx.x;
  if (p != nullptr && 32 * l < n) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(reinterpret_cast<const char*>(p) + 128 * l));
}
#endif
#if MARLIN_CPF_TWIN && (MARLIN_CPF_SITE == 9)  // [0061 COLFPF v2] out-of-line prefetch of site(s) 9
// [exp f_asmlines]
template <int threads>
__device__ __noinline__ void colf_pf_f(const float* global_scale_ptr, int lines) {
  asm volatile(
        "{\n"
        ".reg .u32 %%cpf0, %%cpf1, %%cpf2;\n"
        ".reg .u64 %%cpfa;\n"
        ".reg .pred %%cpfp, %%cpfq;\n"
        "mov.u32 %%cpf0, %%ctaid.x;\n"
        "mov.u32 %%cpf1, %%tid.x;\n"
        "mad.lo.u32 %%cpf0, %%cpf0, %2, %%cpf1;\n"
                        "setp.lt.u32 %%cpfp, %%cpf0, %1;\n"
        "setp.ne.and.u64 %%cpfq, %0, 0, %%cpfp;\n"
        "mul.wide.u32 %%cpfa, %%cpf0, 128;\n"
        "add.u64 %%cpfa, %%cpfa, %0;\n"
        "@%%cpfq prefetch.global.L2::evict_normal [%%cpfa];\n"
        "}\n" ::"l"(global_scale_ptr), "r"(lines), "n"(threads));
}
#endif
template <const vllm::ScalarTypeId a_type_id,  // A ScalarType id
          const vllm::ScalarTypeId b_type_id,  // B ScalarType id
          const vllm::ScalarTypeId c_type_id,  // C ScalarType id
          const vllm::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
// [0045 L2PF] B-stream L2 prefetch-size twins. A TU that defines MARLIN_L2PF_TWIN 1 before including kernel.h
// (generate_kernels.py L2PF_FAMILIES -> sm80_kernel_l2pf*_<c_type>.cu) compiles the kernel of its family (Marlin, MarlinXS,
// MarlinOPR, MarlinXSOPR or their SP twins) with ONE change: fetch_to_shared's B-stream copy calls cp_async4_stream_l2pf
// (marlin.cuh: the same cp.async + the .L2::256B prefetch-size hint), so the twin computes exactly what its family kernel
// computes. This name block sits right before the kernel, after every other name block, so it names the family the
// macros above chose (+ SP) + L2PF; with MARLIN_L2PF_TWIN 0 (every other TU) the token stream is unchanged.
#if MARLIN_L2PF_TWIN
  #if (defined(MARLIN_FOLD_TWIN) && MARLIN_FOLD_TWIN) || (defined(MARLIN_TPF_TWIN) && MARLIN_TPF_TWIN)
    #error "[0045 L2PF] no L2PF twin of a MarlinFOLD / TAILPF kernel (not in L2PF_FAMILIES)"
  #endif
  #undef MARLIN_XS_KNAME
  #if MARLIN_XS_TWIN && MARLIN_OPR_TWIN && MARLIN_SP_TWIN
    #define MARLIN_XS_KNAME MarlinXSOPRSPL2PF
  #elif MARLIN_XS_TWIN && MARLIN_OPR_TWIN
    #define MARLIN_XS_KNAME MarlinXSOPRL2PF
  #elif MARLIN_XS_TWIN && MARLIN_SP_TWIN
    #define MARLIN_XS_KNAME MarlinXSSPL2PF
  #elif MARLIN_XS_TWIN
    #define MARLIN_XS_KNAME MarlinXSL2PF
  #elif MARLIN_OPR_TWIN && MARLIN_SP_TWIN
    #define MARLIN_XS_KNAME MarlinOPRSPL2PF
  #elif MARLIN_OPR_TWIN
    #define MARLIN_XS_KNAME MarlinOPRL2PF
  #elif MARLIN_SP_TWIN
    #define MARLIN_XS_KNAME MarlinSPL2PF
  #else
    #define MARLIN_XS_KNAME MarlinL2PF
  #endif
#endif
// [0051 ZPPACK] Zero-point-in-scale-word twins. A TU that defines MARLIN_ZPPACK_TWIN 1 before including kernel.h
// (generate_kernels.py ZPP_FAMILIES -> sm80_kernel_zpp*_<c_type>.cu) compiles the kernel of its family (Marlin or MarlinXS) for
// int8 activations x asymmetric uint4 weights (kS8 x kU4, g32 / g64) reading a PACKED scale tensor: every 16-bit scale word is
// [zp:4][code:12], the zero points ride in the top nibbles (marlin.cu kt_zpp_pack: the nibbles a lane group needs are the 4
// words its 8-byte load reads), and the one code 12 bits cannot hold, 4096, is stored as u, a code the tensor does not use
// (host census; the kernel gets u in tile_major bits 4..15 and restores code == u ? 4096 : code). Changes v its family:
// fetch_to_shared issues no zero-point cp.async (the zp array is never read), fetch_zp_to_registers loads an int2 of the scale
// stage instead of an int of the zp stage, matmul_a8 takes the zp nibbles and the scale codes out of the words. Same smem
// layout (the zp stage stays allocated, unused), same values: the int8 operands and the int32 scale products are the
// family's bit for bit. With MARLIN_ZPPACK_TWIN 0 (every other TU) the token stream is unchanged.
#ifndef MARLIN_ZPPACK_TWIN
  #define MARLIN_ZPPACK_TWIN 0
#endif
#if MARLIN_ZPPACK_TWIN
  #if MARLIN_OPR_TWIN || MARLIN_SP_TWIN || MARLIN_FOLD_TWIN || \
      (defined(MARLIN_L2PF_TWIN) && MARLIN_L2PF_TWIN)
    #error "[0051 ZPPACK] a ZPP twin wraps Marlin / MarlinXS only (no OPR / SP / FOLD / L2PF twin; TPF: [0052 ZPPTPF])"
  #endif
  #undef MARLIN_XS_KNAME
  #if MARLIN_TPF_TWIN
  // [0052 ZPPTPF] ZPPACK x TAILPF: a TU that defines MARLIN_ZPPACK_TWIN 1 AND MARLIN_TPF_TWIN 1 (generate_kernels.py
  // ZPPTPF_TWINS -> sm80_kernel_zpptpf*_<c_type>.cu) compiles the ZPP twin of its family (packed scale words) with TAILPF's
  // descriptor parameter and its tail (tpf_prefetch_next, the kernel's last statement): the ZPP body is not touched, so the
  // twin computes exactly what its ZPP sibling computes. The name repeats the TAILPF block's function-like form.
    #if MARLIN_XS_TWIN
      #define MARLIN_XS_KNAME MarlinXSZPPTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
    #else
      #define MARLIN_XS_KNAME MarlinZPPTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
    #endif
  #elif MARLIN_XS_TWIN
    #define MARLIN_XS_KNAME MarlinXSZPP
  #else
    #define MARLIN_XS_KNAME MarlinZPP
  #endif
#endif
// [0061 COLFPF v2 cuda 2026-10-11] Column-factor L2-prefetch twins. A TU that defines MARLIN_CPF_TWIN 1 (generate_kernels.py
// generate_cpf_twins -> sm80_kernel_cpf*_<c_type>.cu) compiles its sibling's family -- MarlinZPP / MarlinXSZPP / MarlinZPPTPF /
// MarlinXSZPPTPF (kS8 x kU4 g32 / g64, packed scale words) or Marlin / MarlinXS (kS8 x kU3S) -- with ONE change: the kernel issues
// prefetch.global.L2 for the launch's column factor (fp32 [prob_n] in global_scale_ptr, read by write_result's __ldg after the main
// loop) at ONE prologue position, the MARLIN_CPF_SITE block the TU selects (generate_kernels.py CPF_SITES). The host runs a CPF twin only for a launch that carries a factor with
// VLLM_MARLIN_COLFPF=1 (marlin.cu). With MARLIN_CPF_TWIN 0 (every other TU) the token stream is unchanged, so no existing body
// changes. The name is the sibling's + CPF, before TPF for a TPF twin (0047 declares a name ending in TPF with the descriptor).
#ifndef MARLIN_CPF_TWIN
  #define MARLIN_CPF_TWIN 0
#endif
#if MARLIN_CPF_TWIN
  #if MARLIN_OPR_TWIN || MARLIN_SP_TWIN || MARLIN_FOLD_TWIN || MARLIN_FOLDS_TWIN || \
      (defined(MARLIN_L2PF_TWIN) && MARLIN_L2PF_TWIN) || (MARLIN_TPF_TWIN && !MARLIN_ZPPACK_TWIN && !MARLIN_TPF_KU3S)
    #error "[0061 COLFPF] a CPF twin wraps MarlinZPP / MarlinXSZPP / MarlinZPPTPF / MarlinXSZPPTPF / Marlin / MarlinXS only"
  #endif
  #undef MARLIN_XS_KNAME
  #if MARLIN_ZPPACK_TWIN && MARLIN_TPF_TWIN
    #if MARLIN_XS_TWIN
      #define MARLIN_XS_KNAME MarlinXSZPPCPFTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
    #else
      #define MARLIN_XS_KNAME MarlinZPPCPFTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
    #endif
  #elif MARLIN_TPF_TWIN
    // [0063 KU3STPF] the CPF twin of a kU3S TPF twin (MARLIN_TPF_KU3S): the sibling's name + CPF before TPF, as 0061 names ZPPCPFTPF
    // (the name ends in TPF, so 0047 declares it with the descriptor parameter)
    #if MARLIN_XS_TWIN && MARLIN_TPF_KU3S_GC
      #define MARLIN_XS_KNAME MarlinXSCPFTPF(const __grid_constant__ MarlinTpfDesc tpf, MARLIN_TPF_REST
    #elif MARLIN_XS_TWIN
      #define MARLIN_XS_KNAME MarlinXSCPFTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
    #elif MARLIN_TPF_KU3S_GC
      #define MARLIN_XS_KNAME MarlinCPFTPF(const __grid_constant__ MarlinTpfDesc tpf, MARLIN_TPF_REST
    #else
      #define MARLIN_XS_KNAME MarlinCPFTPF(const MarlinTpfDesc tpf, MARLIN_TPF_REST
    #endif
  #elif MARLIN_ZPPACK_TWIN
    #if MARLIN_XS_TWIN
      #define MARLIN_XS_KNAME MarlinXSZPPCPF
    #else
      #define MARLIN_XS_KNAME MarlinZPPCPF
    #endif
  #elif MARLIN_XS_TWIN
    #define MARLIN_XS_KNAME MarlinXSCPF
  #else
    #define MARLIN_XS_KNAME MarlinCPF
  #endif
#endif
__global__ void MARLIN_XS_KNAME(
    const int4* __restrict__ A0,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,   // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C0,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,     // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ b_bias_ptr,
    // float scales of input matrix, only used when is_a_8bit == true.
    // shape (m,)
    const float* __restrict__ a_scales_ptr,
    // fp16 quantization scales. shape (k/groupsize, n)
    const int4* __restrict__ scales_ptr,
    // float global scale (for nvfp4// only)
    const float* __restrict__ global_scale_ptr,
    // 4bit packed zero-points of shape
    // (k/groupsize, n/pack_factor)
    const int4* __restrict__ zp_ptr,
    int prob_m,  // batch dimension m
    int prob_n,  // output dimension n
    int prob_k,  // reduction dimension k
    int lda,     // A.stride(0), equal to prob_k is A is contiguous
    int* locks,  // extra global storage for barrier synchronization
    bool has_bias,
    bool use_atomic_add,   // whether to use atomic add to reduce
    bool use_fp32_reduce,  // whether to use fp32 global reduce
    int max_shared_mem,
    int tile_major) {  // [0033] host-gated 0013 part-1 order (0 = the served par-major order, byte for byte)
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  // FP8 computation is only supported for Ada Lovelace or newer architectures.
  if constexpr (a_type_id == vllm::kFE4M3fn.id()) return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  // Turing TensorCore only supports fp16 and int8
  if constexpr (a_type_id != vllm::kFloat16.id() && a_type_id != vllm::kS8.id())
    return;
  #endif

  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
  constexpr auto num_bits = vllm::ScalarType::from_id(b_type_id).size_bits();
  // Disable use_fp16_accum for NVFP4 and cases when group_size == -1 &&
  // num_bits == 4
  constexpr bool use_fp16_accum =
      a_type_id == vllm::kFloat16.id() &&
      (!(b_type_id == vllm::kFE2M1f.id() && s_type_id == vllm::kFE4M3fn.id()) &&
       !(group_blocks == -1 && num_bits == 4));
  #else
  constexpr bool use_fp16_accum = false;
  #endif
  using Adtype = MarlinScalarType<a_type_id>;
  using Cdtype = MarlinScalarType<c_type_id>;
  const int4* A = A0;
  int4* C = C0;

  using scalar_t = typename MarlinScalarType<a_type_id>::scalar_t;
  using scalar_t2 = typename MarlinScalarType<a_type_id>::scalar_t2;
  using scalar_32bit_t = typename MarlinScalarType<a_type_id>::scalar_32bit_t;

  using c_scalar_t = typename MarlinScalarType<c_type_id>::scalar_t;
  using c_scalar_t2 = typename MarlinScalarType<c_type_id>::scalar_t2;

  using FragA = typename MarlinScalarType<a_type_id>::FragA;
  using FragB = typename MarlinScalarType<a_type_id>::FragB;
  using FragC = typename MarlinScalarType<a_type_id>::FragC;
  using FragS = typename MarlinScalarType<c_type_id>::FragS;
  using FragZP = typename MarlinScalarType<c_type_id>::FragZP;

  static constexpr auto a_type = vllm::ScalarType::from_id(a_type_id);
  static constexpr auto b_type = vllm::ScalarType::from_id(b_type_id);
  static constexpr auto c_type = vllm::ScalarType::from_id(c_type_id);
  static constexpr auto s_type = vllm::ScalarType::from_id(s_type_id);
  if constexpr (b_type == vllm::kFE2M1f) {
    static_assert(s_type == vllm::kFE4M3fn && group_blocks == 1 ||
                  s_type == vllm::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (s_type == vllm::kFE8M0fnu) {
    // MXFP8: FP8 weights with e8m0 microscaling block scales
    static_assert(b_type == vllm::kFE4M3fn && group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == vllm::kBFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == vllm::kFloat16);
  }

  constexpr bool is_a_8bit = a_type.size_bits() == 8;
  constexpr bool is_8bit_scale = s_type.size_bits() == 8;
  // [0056 IRED] int32 split-K partials + one fp32 conversion in the final slice (kernel.h); kS8 grouped only
  constexpr bool ired = MARLIN_IRED && a_type == vllm::kS8 && group_blocks != -1;
  if constexpr (!is_a_8bit) {
    static_assert(std::is_same<scalar_t, c_scalar_t>::value);
  }
  constexpr bool has_zp = b_type == vllm::kU4 || b_type == vllm::kU8;
  // [0055 KU3S] symmetric int3, LOWBIT words (3 u32 per lane per k32 step), int8 activations only. Every tile constexpr below
  // keeps the int4 (virtual) geometry; only the B global->shared copy, the B shared read and the unpack differ.
  constexpr bool is_u3 = b_type == vllm::kU3S;
  // [0059 W4RCOLF cuda 2026-10-10] the 0055 KU3S-4 per-output-column power-of-two factor, widened to the int4 grouped int8-activation
  // instances (kU4 / kU4B8, group_blocks != -1): W4R headroom codes (quant 08:21) keep every code * |c| sum inside int32 for ANY input,
  // and the factor restores the column's magnitude exactly (x * 2^l is exact in fp32). Every twin writes through write_result(), whose
  // column map does not depend on the twin (FOLD/FOLDS remap rows only). nullptr (no factor) = arithmetic unchanged.
  constexpr bool colf_a8 = is_a_8bit && group_blocks != -1 && (b_type == vllm::kU4 || b_type == vllm::kU4B8);
  #if MARLIN_CPF_TWIN
  // [0061 COLFPF v2] a CPF twin exists only for a column-factor body; its prefetch is its MARLIN_CPF_SITE block
  static_assert(is_u3 || colf_a8, "[0061 COLFPF] a CPF twin is a column-factor body (kS8 x kU4 / kU4B8 grouped, kS8 x kU3S)");
  static_assert(MARLIN_CPF_SITE >= 1, "[0061 COLFPF] a CPF TU defines its prefetch site (generate_kernels.py CPF_SITES)");
  #endif
  #if MARLIN_OPR_TWIN || MARLIN_SP_TWIN || MARLIN_FOLD_TWIN || MARLIN_FOLDS_TWIN || (MARLIN_TPF_TWIN && !MARLIN_TPF_KU3S) || \
      MARLIN_ZPPACK_TWIN || (defined(MARLIN_L2PF_TWIN) && MARLIN_L2PF_TWIN)
  static_assert(!is_u3, "[0055 KU3S] kU3S has the plain Marlin and the MarlinXS family only");
  #endif
  #if MARLIN_TPF_KU3S
  static_assert(is_u3, "[0063 KU3STPF] a KU3STPF TU instantiates kS8 x kU3S only (its TPF twins; CPFTPF with MARLIN_CPF_TWIN)");
  #endif
  static_assert(!is_u3 || (a_type_id == vllm::kS8.id() && (group_blocks == 2 || group_blocks == 4 || group_blocks == 8) && !m_block_size_8 &&
                           !is_zp_float),
                "[0055 KU3S] kU3S = int8 activations, g64 / g128, 16-row m blocks");
  constexpr bool is_int_type = b_type == vllm::kU4 || b_type == vllm::kU8 ||
                               b_type == vllm::kS4 || b_type == vllm::kS8 ||
                               b_type == vllm::kU4B8 || b_type == vllm::kU8B128;
  // see comments of dequant.h for more details
  constexpr bool dequant_skip_flop =
      is_a_8bit || (b_type == vllm::kFE4M3fn && !(s_type == vllm::kFE8M0fnu)) ||
      b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn ||
      has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
      has_zp && !is_zp_float && !(b_type == vllm::kU8);

  float global_scale_f32 = 1.0f;

  if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
    global_scale_f32 = global_scale_ptr[0];
  }
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 6  // [0061 COLFPF v2] prefetch site 6 (f_char_ent)
  if constexpr (is_u3 || colf_a8) { colf_pf_f<threads>(global_scale_ptr, prob_n); }  // [exp form]
  #endif
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 4  // [0061 COLFPF v2] prefetch site 4 (ent_noinl)
  if constexpr (is_u3 || colf_a8) colf_pf_noinl<threads>(global_scale_ptr, prob_n);  // [exp ent_noinl]
  #endif
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 3  // [0061 COLFPF v2] prefetch site 3 (ent_cpp_nonull)
  if constexpr (is_u3 || colf_a8) {  // [exp ent_cpp_nonull]
    const int colf_l = blockIdx.x * threads + threadIdx.x;
    if (colf_l < (prob_n + 31) / 32) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(global_scale_ptr + 32 * colf_l));
  }
  #endif
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 1  // [0061 COLFPF v2] prefetch site 1 (f_asm_ent)
  if constexpr (is_u3 || colf_a8) { colf_pf_f<threads>(global_scale_ptr, prob_n); }  // [exp form]
  #endif

  static_assert(group_blocks != 0,
                "runtime activation ordering is not supported");
  constexpr int m_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);
  #if MARLIN_ZPPACK_TWIN
  // [0051 ZPPACK] u (the code that stands for 4096) arrives in tile_major bits 4..15; bits 0..3 keep their family meaning
  // ([0033] order, the host passes 0 or 1), restored before any use below.
  static_assert(a_type_id == vllm::kS8.id() && b_type_id == vllm::kU4.id() && !is_zp_float && !m_block_size_8 &&
                    (group_blocks == 2 || group_blocks == 4),
                "[0051 ZPPACK] ZPP twins cover kS8 x kU4 g32 / g64 only");
  const uint32_t zpp_u = (static_cast<uint32_t>(tile_major) >> 4) & 0xFFFu;
  tile_major &= 0xF;
  // Packed-state guard: when the host packed this weight it wrote 0x05100000 | u into the first int of its zp array (never
  // read by a ZPP twin; 0x0510 has zero nibbles, which no served zero-point int has: zp = stored + 1 >= 1). Thread 0 loads it
  // here and checks it after the first stages are in flight (below start_pipes): a ZPP launch on a weight that is not packed
  // with this u traps (a loud failure) instead of decoding the family's words wrong.
  int zpp_seen = 0;
  if (threadIdx.x == 0) zpp_seen = __ldg(reinterpret_cast<const int*>(zp_ptr));
  #endif

  #if MARLIN_SP_TWIN
  // [0041 MR14] First statement of the kernel: the out-of-line stripe prefetch (sp_prefetch_stripe above). The host passes
  // the mode in tile_major >> 1 and selects a twin only for one problem copy (parallel == 1) with the [0033] order off, so
  // the nonzero tile_major below changes no tile: with parallel == 1, p1_q == 0 and init_part1_slice's tile-major map is the
  // par-major one (slice_col = slice_col_par, par_id 0).
  static_assert(group_blocks != -1 && !m_block_size_8,
                "[0041 MR14] SP twins cover grouped scales on 16-row m blocks only");
  #if MARLIN_OPR_TWIN && !MARLIN_XS_TWIN
  sp_walk_opr<threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks, is_8bit_scale,
              32 / b_type.size_bits(), has_zp, is_zp_float>(scales_ptr, zp_ptr, prob_m, prob_n, prob_k, tile_major);
  #else
  sp_prefetch_stripe<threads, thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks, is_8bit_scale,
                     32 / b_type.size_bits(), has_zp, is_zp_float>(scales_ptr, zp_ptr, prob_m, prob_n, prob_k, tile_major);
  #endif
  #endif
  if constexpr (is_u3) {
    // [0055 KU3S] device engage tell: a symmetric type reads no zero points, so the host passes nullptr there unless the caller
    // handed marlin_gemm a 1-int b_zeros counter; then every CTA of every kU3S launch adds 1 (the KAT's no-fallback proof).
    if (zp_ptr != nullptr && threadIdx.x == 0) atomicAdd(reinterpret_cast<unsigned int*>(const_cast<int4*>(zp_ptr)), 1u);
  }
  extern __shared__ int4 sh[];
  float* sh_a_s = reinterpret_cast<float*>(sh);
  int4* sh_new = sh + (is_a_8bit ? (4 * thread_m_blocks) : 0);
  constexpr int pack_factor = is_u3 ? 8 : 32 / b_type.size_bits();  // [0055 KU3S] int4-virtual geometry
  static_assert(thread_m_blocks == 1 || !m_block_size_8);
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 8  // [0061 COLFPF v2] prefetch site 8 (p_shared)
  if constexpr (is_u3 || colf_a8) {  // [exp p_shared]
    const int colf_l = blockIdx.x * threads + threadIdx.x;
    if (colf_l < (prob_n + 31) / 32) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(global_scale_ptr + 32 * colf_l));
  }
  #endif

  // For larger GEMMs we run multiple batchsize 64 versions in parallel for a
  // better partitioning with less reductions
  #if MARLIN_XS_TWIN
  // [0036 XS-5] This is the twin MarlinXS: the host launches it only for a launch its gate selected (VLLM_MARLIN_XSLICE=1,
  // rows in [_MIN_M, _MAX_M], N in _N, parallel == 1, no bias, fp32 reduce, no atomic add), so XS is on by construction.
  // tile_major keeps its [0033] meaning (the host passes 0 or 1; no bit is borrowed any more).
  constexpr bool xs_env = true;
  #endif
  int parallel = 1;
  #if MARLIN_FOLD_TWIN
  // [0042 FOLD] parallel = ceil(prob_m / 64) copies; the last one starts at last_off = prob_m - 64, i.e. at (parallel - 1) * 64
  // unless the host folded a remainder. row_off(p) = min(p * 64, last_off) is p * 64 for every other copy (last_off >
  // (parallel - 2) * 64) and last_off for the last; write_result skips the p * 64 - row_off(p) leading rows of copy p (0 except
  // in a folded last copy: the rows its predecessor owns). The host folds only at the 64-row tile with the fp32 reduce and
  // no atomic-add reduce, where split-K partials live in C_tmp, never in C, and only the final slice writes C: the overlap
  // rows then reach no output. A launch that breaks that traps instead of writing them.
  static_assert(thread_m_blocks == 4 && !m_block_size_8, "[0042 FOLD] MarlinFOLD covers the 64-row tile only");
  if (use_atomic_add || !use_fp32_reduce) __trap();
  int last_off = 0;
  if (prob_m > m_block_size) {
    parallel = div_ceil(prob_m, m_block_size);
    last_off = prob_m - m_block_size;
    prob_m = m_block_size;
  }
  auto row_off = [&](int p) { return p * m_block_size < last_off ? p * m_block_size : last_off; };
  #elif MARLIN_FOLDS_TWIN
  // [0048 FOLDS] p served copies + the remainder window [fs_last_off, prob_m). The host folds only at the 64-row tile with the
  // fp32 reduce and no atomic add (a remainder row computed twice would otherwise reach C through an fp16 / atomic partial).
  static_assert(thread_m_blocks == 4 && !m_block_size_8, "[0048 FOLDS] MarlinFOLDS covers the 64-row tile only");
  // The whole FOLDS state is two registers: fs_m = the launch's rows (64 * p + r; p = fs_m / 64, the window = fs_m - 64) and
  // fs_st = the current item's virtual block | R << 10 | item << 11 | schedule << 12 (each read back with a shift / mask).
  const int fs_m = prob_m;
  int fs_st = ((tile_major >> 1) & 3) << 12;
  tile_major &= 1;
  if (use_atomic_add || !use_fp32_reduce || fs_m <= m_block_size || fs_m % m_block_size == 0 || (fs_st >> 12) == 3 ||
      gridDim.x > 1024 || ((fs_st >> 12) == 2 && (fs_m >= 2 * m_block_size || (gridDim.x & 1))))
    __trap();
  auto fs_vb = [&]() { return fs_st & 1023; };          // the current item's virtual block (the served blockIdx.x)
  auto fs_R = [&]() { return ((fs_st >> 10) & 1) != 0; };  // the current item is the remainder copy
  auto fs_pick = [&](int item) {
    const int sched = fs_st >> 12;
    const int h = gridDim.x >> 1;
    bool r;
    int vb;
    if (sched == 2) {
      r = blockIdx.x >= h;
      vb = (r ? blockIdx.x - h : blockIdx.x) + (item ? h : 0);
    } else {
      r = (item == 0) == (sched == 1);
      vb = blockIdx.x;
    }
    fs_st = (sched << 12) | (item << 11) | (int(r) << 10) | vb;
  };
  // first row of copy p_: launch A's copies at p_ * 64, the remainder copy at the window
  auto fs_row = [&](int p_) { return fs_R() ? fs_m - m_block_size : p_ * m_block_size; };
  fs_pick(0);
  parallel = fs_R() ? 1 : fs_m / m_block_size;
  prob_m = m_block_size;
  #else
  if (prob_m > m_block_size) {
    parallel = prob_m / m_block_size;
    prob_m = m_block_size;
  }
  #endif

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;

  int global_mn_tiles = parallel * n_tiles;
  int part2_mn_tiles = global_mn_tiles;
  int part1_mn_iters = 0;
  bool in_part2 = false;
  #if MARLIN_XS_TWIN
  // Set once the first slice's per-row A scales are loaded. With parallel == 1 (an XS precondition) every
  // later reload copies the same par_id-0 values, so XS skips it: its cp.async would otherwise join a
  // group that the XS ordering no longer waits for before the epilogue reads sh_a_s.
  bool xs_skip_as_reload = false;
  #endif

  if (global_mn_tiles > gridDim.x) {
    part2_mn_tiles = global_mn_tiles % gridDim.x;
    if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
    part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
  }

  // TILE-MAJOR part 1. For prob_m > 64 the problem runs as `parallel` 64-row copies and tile
  // g = par * n_tiles + col, so a round of gridDim.x blocks covers consecutive columns of ONE copy and
  // every copy streams the whole weight matrix again (a layer's weights do not fit L2). Part 1 (the
  // first part1_tiles tiles, full k per block, no locks) is enumerated column-major instead, so the
  // copies of a column run in the same round and their B reads can hit L2. The map is a bijection onto
  // the same tile set (columns < p1_r carry p1_q + 1 copies, the rest p1_q); part 2 is unchanged, and
  // parallel == 1 (every decode shape) is the identity. Each tile's arithmetic is untouched, so the
  // output is bit-identical to the par-major order.
  #if !MARLIN_FOLDS_TWIN  // [0048 FOLDS] derives them per part-1 tile in init_part1_slice (they change with the item)
  const int part1_tiles = global_mn_tiles - part2_mn_tiles;
  const int p1_q = part1_tiles / n_tiles;
  const int p1_r = part1_tiles % n_tiles;
  #endif

  int iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);

  if constexpr (group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) *
              div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }
  #if MARLIN_OPR_TWIN
  // [0040 OPR] opr_smax() = ceil(k_tiles / iters) + 1 bounds every slice_count of this launch (slice_count =
  // ceil((k_tiles - col_off) / iters) + (col_off > 0)); the fold below is unrolled for at most opr_max_slices slices. The
  // host selects a twin only when opr_smax() <= opr_max_slices and the fp32 reduce is on; the epilogue traps otherwise.
  // Evaluated in the epilogue only (k_tiles and iters are live there anyway), so the main loop carries no extra register.
  static_assert(!m_block_size_8, "[0040 OPR] MarlinOPR covers the 16-row m blocks only");
  constexpr int opr_max_slices = 4;
  auto opr_smax = [&]() { return div_ceil(k_tiles, iters) + 1; };
  #endif

  int slice_row = 0;
  #if MARLIN_FOLDS_TWIN
  int slice_col_par = fs_vb();
  #else
  int slice_col_par = blockIdx.x;
  #endif
  int slice_col;
  int slice_iters =
      k_tiles;  // number of threadblock tiles in the current slice
  // total number of active threadblocks in the current slice
  int slice_count = 1;
  // index of threadblock in current slice; numbered bottom to top
  int slice_idx = 0;

  int par_id = 0;
  int locks_off = 0;

  #if MARLIN_FOLDS_TWIN
  // [0048 FOLDS] the served lock / C_tmp slot of virtual block fs_vb; the remainder item's slots follow launch A's gridDim.x
  if (part2_mn_tiles >= gridDim.x) {
    locks_off = fs_vb() + (fs_R() ? gridDim.x : 0);
  } else {
    locks_off = (iters * fs_vb()) / k_tiles - 1 + (fs_R() ? gridDim.x : 0);
  }
  #else
  if (part2_mn_tiles >= gridDim.x) {
    // when part2_mn_tiles >= sms
    // then there are at most $sms$ conflict tile blocks
    locks_off = blockIdx.x;
  } else {
    locks_off = (iters * blockIdx.x) / k_tiles - 1;
  }
  #endif

  // Compute all information about the current slice which is required for
  // synchronization.
  bool first_init = true;
  #if MARLIN_FOLDS_TWIN
  // [0048 FOLDS] the second item: its partition from scratch, exactly as the kernel prologue derives the first one
  auto fs_setup = [&]() {
    global_mn_tiles = (fs_R() ? 1 : fs_m / m_block_size) * n_tiles;
    part2_mn_tiles = global_mn_tiles;
    part1_mn_iters = 0;
    in_part2 = false;
    if (global_mn_tiles > gridDim.x) {
      part2_mn_tiles = global_mn_tiles % gridDim.x;
      if (part2_mn_tiles * 3 <= gridDim.x) part2_mn_tiles += gridDim.x;
      part1_mn_iters = (global_mn_tiles - part2_mn_tiles) / gridDim.x;
    }
    iters = div_ceil(k_tiles * part2_mn_tiles, gridDim.x);
    if constexpr (group_blocks != -1) {
      if (group_blocks >= thread_k_blocks) {
        iters = (group_blocks / thread_k_blocks) * div_ceil(iters, (group_blocks / thread_k_blocks));
      }
    }
    slice_row = 0;
    slice_col_par = fs_vb();
    slice_iters = k_tiles;
    slice_count = 1;
    slice_idx = 0;
    par_id = 0;
    if (part2_mn_tiles >= gridDim.x) {
      locks_off = fs_vb() + (fs_R() ? gridDim.x : 0);
    } else {
      locks_off = (iters * fs_vb()) / k_tiles - 1 + (fs_R() ? gridDim.x : 0);
    }
    first_init = true;
  };
  #endif
  auto init_part2_slice = [&]() {
  #if MARLIN_FOLDS_TWIN
    slice_iters =
        iters * (fs_vb() + 1) - (k_tiles * slice_col_par + slice_row);
  #else
    slice_iters =
        iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
  #endif
    if (slice_iters < 0 || slice_col_par >= part2_mn_tiles) slice_iters = 0;
    if (slice_iters == 0) return;
    if (slice_row + slice_iters > k_tiles) slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * div_ceil(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = div_ceil(k_tiles - col_off, iters);
      if (col_off > 0) slice_count++;
  #if MARLIN_FOLDS_TWIN
      int delta_first = iters * fs_vb() - col_first;
  #else
      int delta_first = iters * blockIdx.x - col_first;
  #endif
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0) slice_idx--;
      }
    }
    if (part2_mn_tiles >= gridDim.x) {
      if (slice_count > 1 && slice_idx == slice_count - 1) {
        locks_off++;
      }
    } else {
      locks_off++;
    }

    if (first_init && use_atomic_add && slice_count > 1 && slice_idx == 0) {
      constexpr int threads_per_m = 16 * thread_n_blocks / 8;
      int m_per_thread =
          div_ceil(thread_m_blocks * 16, threads / threads_per_m);
      if (m_block_size_8) m_per_thread = div_ceil(8, threads / threads_per_m);
      for (int i = 0; i < m_per_thread; i++) {
        int row = threads / threads_per_m * i + threadIdx.x / threads_per_m;
        if (row < prob_m) {
          int col = slice_col * 16 * thread_n_blocks / 8 +
                    threadIdx.x % threads_per_m;
          C[row * prob_n / 8 + col] = {0, 0, 0, 0};
        }
      }
      // After write zero to output, write a negative value to lock.
      // Every SM that processes the same slice would wait for
      // the negative value, and then atomicAdd 1 to it.
      // After all SMs are processed, the lock value would back to 0 again.
      __syncthreads();
      if (threadIdx.x == 0) locks[locks_off] = 1 - slice_count;
    }

    if (slice_col == n_tiles) {
  #if MARLIN_FOLD_TWIN
      slice_col = 0;
      par_id++;
      A = A0 + row_off(par_id) * lda / (is_a_8bit ? 16 : 8);  // [0042 FOLD] was A += one m-block
      C = C0 + row_off(par_id) * prob_n / 8;
  #else
      A += 16 * thread_m_blocks * lda / (is_a_8bit ? 16 : 8);
      C += 16 * thread_m_blocks * prob_n / 8;
      slice_col = 0;
      par_id++;
  #endif
    }
    if (is_a_8bit && (first_init || slice_col == 0)
  #if MARLIN_XS_TWIN
        && !xs_skip_as_reload
  #endif
    ) {
      __syncthreads();
  #if MARLIN_FOLD_TWIN
      int a_s_gl_rd = row_off(par_id) + threadIdx.x;  // [0042 FOLD]
  #elif MARLIN_FOLDS_TWIN
      int a_s_gl_rd = fs_row(par_id) + threadIdx.x;  // [0048 FOLDS]
  #else
      int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
  #endif
      cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                        threadIdx.x < prob_m);
    }
  };

  auto init_part1_slice = [&]() {
    if (part1_mn_iters) {
      part1_mn_iters--;
  #if MARLIN_FOLDS_TWIN
      const int p1_q = (global_mn_tiles - part2_mn_tiles) / n_tiles;  // [0048 FOLDS] this item's part-1 map
      const int p1_r = (global_mn_tiles - part2_mn_tiles) % n_tiles;
  #endif
      if (!tile_major) {  // [0033] gate off: the served par-major enumeration
        par_id = slice_col_par / n_tiles;
        slice_col = slice_col_par % n_tiles;
      } else if (slice_col_par < p1_r * (p1_q + 1)) {
        slice_col = slice_col_par / (p1_q + 1);
        par_id = slice_col_par % (p1_q + 1);
      } else {  // reached only when p1_q >= 1
        const int h = slice_col_par - p1_r * (p1_q + 1);
        slice_col = p1_r + h / p1_q;
        par_id = h % p1_q;
      }
      slice_iters = k_tiles;
  #if MARLIN_FOLD_TWIN
      A = A0 + row_off(par_id) * lda / (is_a_8bit ? 16 : 8);  // [0042 FOLD]
      C = C0 + row_off(par_id) * prob_n / 8;
  #elif MARLIN_FOLDS_TWIN
      A = A0 + fs_row(par_id) * lda / (is_a_8bit ? 16 : 8);  // [0048 FOLDS]
      C = C0 + fs_row(par_id) * prob_n / 8;
  #else
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
  #endif
      if (is_a_8bit
  #if MARLIN_XS_TWIN
          && !xs_skip_as_reload
  #endif
      ) {
        __syncthreads();
  #if MARLIN_FOLD_TWIN
        int a_s_gl_rd = row_off(par_id) + threadIdx.x;  // [0042 FOLD]
  #elif MARLIN_FOLDS_TWIN
        int a_s_gl_rd = fs_row(par_id) + threadIdx.x;  // [0048 FOLDS]
  #else
        int a_s_gl_rd = par_id * 16 * thread_m_blocks + threadIdx.x;
  #endif
        cp_async1_ca_pred(&sh_a_s[threadIdx.x], &a_scales_ptr[a_s_gl_rd],
                          threadIdx.x < prob_m);
      }
    }
  };

  auto init_slice = [&]() {
  #if MARLIN_FOLDS_TWIN
    // [0048 FOLDS] the first item exhausted (or empty for this CTA): switch to the second one and walk it from its start
    for (;;) {
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * fs_vb()) / k_tiles;
      slice_row = (iters * fs_vb()) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
      A = A0 + fs_row(par_id) * lda / (is_a_8bit ? 16 : 8);
      C = C0 + fs_row(par_id) * prob_n / 8;
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
    if (slice_iters != 0 || ((fs_st >> 11) & 1)) break;
    fs_pick(1);
    fs_setup();
    }
  #else
    if (!in_part2 && !part1_mn_iters) {
      in_part2 = true;
      slice_col_par = (iters * blockIdx.x) / k_tiles;
      slice_row = (iters * blockIdx.x) % k_tiles;
      slice_col = (slice_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
      par_id = (slice_col_par + global_mn_tiles - part2_mn_tiles) / n_tiles;
  #if MARLIN_FOLD_TWIN
      A = A0 + row_off(par_id) * lda / (is_a_8bit ? 16 : 8);  // [0042 FOLD]
      C = C0 + row_off(par_id) * prob_n / 8;
  #else
      A = A0 + 16 * thread_m_blocks / (is_a_8bit ? 16 : 8) * par_id * lda;
      C = C0 + 16 * thread_m_blocks / 8 * par_id * prob_n;
  #endif
    }
    if (!in_part2) {
      init_part1_slice();
    } else {
      init_part2_slice();
      first_init = false;
    }
  #endif
  };

  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 7  // [0061 COLFPF v2] prefetch site 7 (p_beforeinit)
  if constexpr (is_u3 || colf_a8) {  // [exp p_beforeinit]
    const int colf_l = blockIdx.x * threads + threadIdx.x;
    if (colf_l < (prob_n + 31) / 32) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(global_scale_ptr + 32 * colf_l));
  }
  #endif
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 10  // [0061 COLFPF v2] prefetch site 10 (n_beforeinit)
  if constexpr (is_u3 || colf_a8) colf_pf_noinl<threads>(global_scale_ptr, prob_n);  // [exp noinl]
  #endif
  init_slice();

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = lda / (is_a_8bit ? 16 : 8);
  // stride of an A matrix tile in shared memory
  constexpr int a_sh_stride = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / (is_a_8bit ? 16 : 8);
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // within a shared memory tile
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * m_block_size;
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // B sizes/strides
  int b_gl_stride = 16 * prob_n / (pack_factor * (is_a_8bit ? 2 : 4));
  // [0055 KU3S] a k32 row of the LOWBIT B array is N/32 tiles of 24 int4
  if constexpr (is_u3) b_gl_stride = prob_n * 3 / 4;
  constexpr int b_sh_stride =
      ((thread_n_blocks * 16) * 16 / pack_factor) / (is_a_8bit ? 2 : 4);
  constexpr int b_thread_vecs = (b_type.size_bits() == 4 || is_u3) ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage =
      b_sh_stride * thread_k_blocks / (is_a_8bit ? 2 : 1);
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides
  int s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);
  constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
  constexpr int s_tb_groups =
      group_blocks != -1 && group_blocks < thread_k_blocks
          ? thread_k_blocks / group_blocks
          : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  constexpr int tb_n_warps = thread_n_blocks / (is_a_8bit ? 2 : 4);

  // Zero-points sizes/strides
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float
                                   ? 16 * thread_n_blocks / 8
                                   : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) +
                (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd =
      a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
      (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / tb_n_warps) * b_sh_wr_iters;

  int b_gl_rd;
  if (threads <= b_sh_stride) {
    b_gl_rd = threadIdx.x;
  } else {
    b_gl_rd =
        b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  }

  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;
  b_sh_rd += b_sh_rd / b_sh_stride * (b_sh_stride * (b_sh_wr_iters - 1));
  // [0055 KU3S] The int3 stage of a CTA tile is u3_stage int4, written COMPACT into the first 3/4 of the int4 stage slot: k32 row r,
  // int4 c of the row (u3_row = 24 per n32 tile) at r * u3_row + c, the global row stride b_gl_stride. b_gl_rd is the CTA-uniform
  // int4 base of the current stage (no thread part); thread t copies elements t and t + threads (the second only below
  // u3_stage: 1.5 int4 per thread on every generated tile). The shared read of virtual chunk ci = (n32 tile g = ci / 32, lane):
  // int2 [48 g + lane] = {W0, W1}, u32 [96 g + 64 + lane] = W2.
  constexpr int u3_row = b_sh_stride * 3 / 4;
  constexpr int u3_stage = u3_row * thread_k_blocks / (is_a_8bit ? 2 : 1);
  int u3_gl_off0 = 0, u3_gl_off1 = 0, u3_sh_rd = 0;
  if constexpr (is_u3) {
    static_assert(b_sh_stride % 32 == 0 && u3_stage > threads && u3_stage <= 2 * threads,
                  "[0055 KU3S] the int3 stage copy assumes 32-chunk rows and threads < u3_stage <= 2 * threads");
    b_gl_rd = u3_row * slice_col + b_gl_rd_delta_o * slice_row;
    const int e0 = threadIdx.x, e1 = threadIdx.x + threads;
    u3_gl_off0 = (e0 / u3_row) * b_gl_stride + e0 % u3_row;
    u3_gl_off1 = (e1 / u3_row) * b_gl_stride + e1 % u3_row;
    u3_sh_rd = static_cast<int>(b_sh_rd / 32) * 48 + static_cast<int>(threadIdx.x % 32);
  }

  int s_gl_rd;
  if constexpr (group_blocks == -1) {
    s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
  } else if constexpr (group_blocks >= thread_k_blocks) {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
  } else {
    s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stage;

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
                 zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                                 threadIdx.x / zp_sh_stride) +
                 zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = zp_sh_stage > 0 && threadIdx.x < zp_sh_stage;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (is_a_8bit) {
    s_sh_rd = 4 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 4);
  } else if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 &&
                     (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) % 4;

  int bias_sh_rd;
  if constexpr (m_block_size_8) {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 8;
  } else {
    bias_sh_rd = (is_a_8bit ? 4 : 8) * ((threadIdx.x / 32) % tb_n_warps) +
                 (threadIdx.x % 32) % 4;
  }

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd =
            8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
      }
    } else if (is_a_8bit) {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps / 2) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads *
                     ((threadIdx.x / 32) % tb_n_warps) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }
  #if MARLIN_ZPPACK_TWIN
  // [0051 ZPPACK] the int2 of the scale stage holding this lane group's 4 zero-point nibbles: the warp's 32-word block starts
  // at int4 4 * (warp % tb_n_warps) (= s_sh_rd's), i.e. int2 8 * (warp % tb_n_warps); lane group lane / 4 takes int2 lane / 4.
  const int zpp_sh_rd = 8 * ((threadIdx.x / 32) % tb_n_warps) + (threadIdx.x % 32) / 4;
  #endif

  // Precompute which thread should not read memory in which iterations; this is
  // needed if there are more threads than required for a certain tilesize or
  // when the batchsize is not a multiple of 16.
  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
  #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(2 * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.

  // Shared memory storage for global fetch pipelines.
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;
  constexpr int sh_size_b_red_min =
      (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_size_b_red_max =
      (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  constexpr int sh_b_red_bias_size =
      sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size)
          ? sh_size_b_red_max
          : (sh_size_b_red_min + sh_bias_size);

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_zp = sh_new + sh_b_red_bias_size;
  constexpr int sh_s_size = stages * s_sh_stage;
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  int4* sh_a = sh_s + sh_s_size;

  #if MARLIN_XS_TWIN
  // [0035 XS] Scope: grouped scales (no epilogue cp.async), no bias, fp32 global reduce (the fp16 reduce
  // stages through sh_red with cp.async), no atomic-add reduce, one 64-row problem copy (parallel == 1, so
  // A, C and the per-row A scales never change between segments), and the separated layout must fit the
  // dynamic smem the host granted (the A16 qkv/o_proj tiles do not fit; they keep the served path).
  constexpr bool xs_compiled = group_blocks != -1 && !m_block_size_8 &&
                               !use_fp16_accum && !is_zp_float;
  constexpr int xs_sh_red_off = (is_a_8bit ? 4 * thread_m_blocks : 0) +
                                sh_b_red_bias_size + stages * zp_sh_stage +
                                sh_s_size + stages * a_sh_stage;
  constexpr int xs_sh_bytes = 16 * (xs_sh_red_off + sh_red_size);
  // [0036 XS-5] A twin exists only for an XS-capable config whose separated layout fits the 99 KiB a 4090 (sm_89)
  // grants one block, so a selected twin always runs XS (the runtime xs_run check below stays as a guard).
  static_assert(xs_compiled, "[0036 XS-5] MarlinXS instantiated for a config XS does not cover");
    #if MARLIN_OPR_TWIN
  // [0039 XSOPR] The combined twin also covers the W8A16 o_proj tile, whose served layout (98,816 B) leaves no room for
  // sh_red past the stages within 99 KiB. During the epilogue the prefetch has filled A stage slots 0 .. stages-2 only, and
  // no cp.async of this segment is outstanding, so A slot stages-1 is free: sh_red goes there whenever it fits one A stage
  // (W8A16: 9/16 of a slot at every thread_m_blocks; W4A8: 9/8, so the A8 tiles keep XS's past-the-stages placement).
  // Reads of that slot by a warp still in its last k-step load a discarded register prefetch (as for the served sh_red
  // aliasing sh_b); the next write to the slot is the next segment's first in-loop fetch_to_shared(stages - 1, ...),
  // issued after write_result's closing __syncthreads and the next segment's wait_for_stage(). In A-slot mode the twin's
  // layout is the served layout, so it runs XS whatever dynamic smem the host granted (0038 TRIM sizes it as served).
  constexpr bool xso_red_in_a = sh_red_size <= a_sh_stage;
  static_assert(xso_red_in_a || xs_sh_bytes <= 99 * 1024,
                "[0039 XSOPR] MarlinXSOPR: sh_red fits neither one A stage nor past the stages within 99 KiB");
  const bool xs_run = xs_compiled && xs_env && parallel == 1 && !has_bias &&
                      !use_atomic_add && use_fp32_reduce &&
                      (xso_red_in_a || xs_sh_bytes <= max_shared_mem);
  if (xs_run) {
    sh_red = xso_red_in_a ? sh_a + (stages - 1) * a_sh_stage : sh_a + stages * a_sh_stage;
    xs_skip_as_reload = true;
  }
    #else
  static_assert(xs_sh_bytes <= 99 * 1024, "[0036 XS-5] MarlinXS separated smem layout exceeds 99 KiB");
  const bool xs_run = xs_compiled && xs_env && parallel == 1 && !has_bias &&
                      !use_atomic_add && use_fp32_reduce &&
                      xs_sh_bytes <= max_shared_mem;
  if (xs_run) {
    sh_red = sh_a + stages * a_sh_stage;  // == sh + xs_sh_red_off: past every stage buffer
    xs_skip_as_reload = true;  // the first slice's A scales are already in flight with its first stage
  }
    #endif
  bool xs_prefetched = false;  // the current segment end issued the next segment's first stages
    #if MARLIN_XSLICE_CHECK
  int xs_n_col = -1, xs_n_row = -1, xs_n_iters = -1;
    #endif
  #endif

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragC frag_c_tmp[thread_m_blocks][is_a_8bit ? 2 : 4][2];
  FragS frag_s[2][4];
  FragS frag_bias[2][4];
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16
  FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ
  #if MARLIN_ZPPACK_TWIN
  int2 frag_zz[2];  // [0051 ZPPACK] the packed words carrying this lane's zero points (k % 2 double buffer, as frag_qzp)
  // frag_qzp is dead here (matmul(), the A16 path, is compiled but returns at once for int8 activations): zeroed so the front
  // end does not see it read before it is set (#549-D); the stores are dead and leave no SASS.
  #pragma unroll
  for (int i = 0; i < 2; i++)
  #pragma unroll
    for (int j = 0; j < num_ints_per_thread; j++) frag_qzp[i][j] = 0;
  #endif

  if constexpr (is_a_8bit) {
  #pragma unroll
    for (int j = 0; j < 2; j++) {
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][0][g] = 0.0f;
        }

  #pragma unroll
        for (int g = 0; g < 4; g++) {
          frag_c_tmp[i][j][1][g] = 0.0f;
        }
      }
    }
  }

  // Zero accumulators.
  auto zero_accums = [&]() {
  #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
            &sh_a_stage[a_sh_wr_trans[i]],
            &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
            a_sh_wr_pred[i]);
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
      if constexpr (is_u3) {
        // [0055 KU3S] the compact int3 stage (see u3_row above): 2 copies per thread, the second for threadIdx.x < u3_stage - threads
        cp_async4_stream(&sh_b_stage[threadIdx.x], &B[b_gl_rd + u3_gl_off0]);
        if (threadIdx.x < u3_stage - threads)
          cp_async4_stream(&sh_b_stage[threads + threadIdx.x], &B[b_gl_rd + u3_gl_off1]);
      } else {
  #pragma unroll
      for (int i = 0; i < (b_sh_wr_iters * b_thread_vecs); i++) {
        constexpr int count = div_ceil(b_sh_stride, threads);
        int b_gl_idx =
            b_gl_rd + (i % count) * threads +
            b_gl_stride * (i / count) * div_ceil(threads, b_sh_stride);

  #if MARLIN_L2PF_TWIN
        // [0045 L2PF] the twin's only change: the same copy with the .L2::256B prefetch-size hint
        cp_async4_stream_l2pf(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
  #else
        cp_async4_stream(&sh_b_stage[threads * i + threadIdx.x], &B[b_gl_idx]);
  #endif
      }
      }

      b_gl_rd += b_gl_rd_delta_o;

      if constexpr (group_blocks != -1) {
        int4* sh_s_stage = sh_s + s_sh_stage * pipe;

        // Only fetch scales if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          s_gl_rd += s_gl_rd_delta * s_tb_groups;
        }
      }

  #if MARLIN_ZPPACK_TWIN
      if constexpr (false) {  // [0051 ZPPACK] no zero-point cp.async: the zero points ride in the scale words
  #else
      if constexpr (has_zp && group_blocks != -1) {
  #endif
        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        // Only fetch zero points if this tile starts a new group
        if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
          if (zp_sh_wr_pred) {
            cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
          }
          zp_gl_rd += zp_gl_rd_delta * zp_tb_groups;
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe) {
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;
  #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, a_type_id>(
          frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

    if constexpr (is_u3) {
      // [0055 KU3S] tile g = (b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd) / 32: LDS.64 {W0, W1} + LDS.32 W2
      const int i2 = (b_sh_stride / 32) * 48 * (k % b_sh_wr_iters) + u3_sh_rd;
      const int2 w01 = reinterpret_cast<const int2*>(sh_b_stage)[i2];
      frag_b_quant[k % 2][0][0] = w01.x;
      frag_b_quant[k % 2][0][1] = w01.y;
      frag_b_quant[k % 2][0][2] = reinterpret_cast<const int*>(sh_b_stage)[2 * i2 - static_cast<int>(threadIdx.x % 32) + 64];
    } else {
  #pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(
          &sh_b_stage[b_sh_stride * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
    }
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;
    using IT1 = typename std::conditional_t<is_a_8bit, int2, int4>;
    using IT0 = typename std::conditional_t<is_a_8bit, int, int2>;
    constexpr int group_blocks2 = div_ceil(group_blocks, is_a_8bit ? 2 : 1);

    if constexpr (group_blocks == -1) {
      // Load only when starting a new slice.
      if (k == 0 && full_pipe == 0 && dequant_skip_flop) {
        reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
        reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
      }
    } else if constexpr (group_blocks >= thread_k_blocks) {
      constexpr int g = group_blocks / thread_k_blocks;
      if (pipe % g == 0) {
        if (k % b_sh_wr_iters == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
          reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
        } else {
          reinterpret_cast<int4*>(&frag_s[1])[0] =
              reinterpret_cast<int4*>(&frag_s[0])[0];
        }
      }
    } else if (group_blocks2 < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
      auto warp_id = threadIdx.x / 32;
      int warp_row = warp_id / tb_n_warps;

      int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
      int cur_group_id = k_blocks / group_blocks2;

      int4* sh_s_stage = sh_s + s_sh_stage * pipe;

      if constexpr (!is_8bit_scale) {
        reinterpret_cast<int4*>(&frag_s[k % 2])[0] =
            sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
      } else {
        reinterpret_cast<int2*>(&frag_s[k % 2])[0] = reinterpret_cast<int2*>(
            sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
      }
    } else if (group_blocks >= b_sh_wr_iters) {
      if constexpr (!is_8bit_scale) {
        reinterpret_cast<int4*>(&frag_s[1])[0] =
            reinterpret_cast<int4*>(&frag_s[0])[0];
      } else {
        reinterpret_cast<int2*>(&frag_s[1])[0] =
            reinterpret_cast<int2*>(&frag_s[0])[0];
      }
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0 || is_a_8bit) {
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }
      } else if constexpr (group_blocks >= thread_k_blocks) {
        constexpr int g = group_blocks / thread_k_blocks;
        if (pipe % g == 0 && k % b_sh_wr_iters == 0 || is_a_8bit) {
  #if MARLIN_ZPPACK_TWIN
          // [0051 ZPPACK] the same k-group row of the SCALE stage (the cp.async group that carried the family's zp row)
          int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
          frag_zz[k % 2] = reinterpret_cast<int2*>(sh_s_stage)[zpp_sh_rd];
  #else
          int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
  #pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] =
                (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
  #endif
        }
      } else {
        auto warp_id = threadIdx.x / 32;

        int warp_row = warp_id / tb_n_warps;

        int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
        int cur_group_id = k_blocks / div_ceil(group_blocks, is_a_8bit ? 2 : 1);

  #if MARLIN_ZPPACK_TWIN
        // [0051 ZPPACK] the k-group row cur_group_id of the scale stage (fetch_scales_to_registers' row for this k)
        int4* sh_s_stage = sh_s + s_sh_stage * pipe + cur_group_id * s_sh_stride;
        frag_zz[k % 2] = reinterpret_cast<int2*>(sh_s_stage)[zpp_sh_rd];
  #else
        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

  #pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] =
              (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
  #endif
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0 && k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage = sh_zp + zp_sh_stage * (g * (pipe / g));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
                sh_zp_stage[zp_sh_rd];
          }
        } else if (group_blocks < b_sh_wr_iters || k % b_sh_wr_iters == 0) {
          auto warp_id = threadIdx.x / 32;

          int warp_row = warp_id / tb_n_warps;
          int k_blocks = b_sh_wr_iters * warp_row + k % b_sh_wr_iters;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] =
              sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_32bit_t* frag_b_ptr, int zp = 0) {
    if constexpr (a_type.size_bits() != b_type.size_bits() && !is_u3) {  // [0055 KU3S] unpacks in matmul_a8
      if constexpr (is_a_8bit && has_zp) {
        sub_zp_and_dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(
            q, frag_b_ptr, zp);
      } else {
        dequant<scalar_32bit_t, b_type_id, dequant_skip_flop>(q, frag_b_ptr);
      }
    }
  };

  // Execute the actual tensor core matmul of a sub-tile.
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k, int pipe) {
    if (is_a_8bit) return;
    int k2 = k % 2;
    constexpr int g =
        group_blocks > 0 ? div_ceil(group_blocks, thread_k_blocks) : 1;
    const bool is_new_zp =
        (group_blocks > 0 && (group_blocks < b_sh_wr_iters || k == 0)) &&
            (pipe % g == 0) ||
        (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        int zp_quant_0, zp_quant_1;

        if constexpr (b_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(b_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_zp));
        dequant_data(zp_quant_1,
                     reinterpret_cast<scalar_32bit_t*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] =
            reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    if constexpr (s_type == vllm::kFE4M3fn || s_type == vllm::kFE8M0fnu) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_0, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<c_scalar_t2, s_type_id>(
          s_quant_1, reinterpret_cast<c_scalar_t2*>(&frag_s[k2]) + 2);
    }

  // We have the m dimension as the inner loop in order to encourage overlapping
  // dequantization and matmul operations.
  #pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (b_type_id == vllm::kFE2M1f.id()) {
        b_quant_1 = frag_b_quant[k2][0][j];
        b_quant_0 = b_quant_1 << 8;
      } else if constexpr (b_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(b_type.size_bits() == 8 || is_u3);  // [0055 KU3S] matmul() is dead for int8 activations
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      dequant_data(b_quant_0, reinterpret_cast<scalar_32bit_t*>(&frag_b0));
      dequant_data(b_quant_1, reinterpret_cast<scalar_32bit_t*>(&frag_b1));

      if constexpr (dequant_skip_flop && has_zp && !is_zp_float && !is_a_8bit) {
        sub_zp<a_type_id>(frag_b0, frag_zp[j], 0);
        sub_zp<a_type_id>(frag_b1, frag_zp[j], 1);
      }

      if constexpr (!dequant_skip_flop && has_zp && !is_zp_float &&
                    group_blocks == -1 && !is_a_8bit) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Adtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_and_sub<a_type_id>(frag_b0, s2.x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, s2.y, frag_zp[j].y);
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1 &&
                           !is_a_8bit) {
        if (is_new_zp)
          frag_zp[j] = __hmul2(frag_zp[j],
                               *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_and_sub<a_type_id>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<a_type_id>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
      } else if constexpr (group_blocks != -1 && !is_a_8bit) {
        scale<a_type_id>(frag_b0, frag_s[k2][j], 0);
        scale<a_type_id>(frag_b1, frag_s[k2][j], 1);
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0, frag_b1,
                                               frag_c[i][j][0]);
        } else {
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b0,
                                         frag_c[i][j][0]);
          mma<a_type_id, use_fp16_accum>(frag_a[k2][i], frag_b1,
                                         frag_c[i][j][1]);
        }
      }
    }
  };

  auto matmul_a8 = [&](int k) {
    // [club-cuda-lane 2026-09-12] 8-bit B on the int8-A path is fed to the MMA as
    // packed bytes with NO zero-point step (the `else` branch below); an
    // asymmetric 8-bit B (kU8) would decode plausibly and wrongly. Refuse at
    // compile time so a future generator entry cannot instantiate it silently.
    static_assert(!(is_a_8bit && b_type.size_bits() == 8 && has_zp),
                  "int8-activation x 8-bit ASYMMETRIC weights has no zero-point "
                  "path in matmul_a8; only uint8b128 (symmetric, ^0x80 at load) "
                  "is supported");
    int k2 = k % 2;
  #pragma unroll
    for (int j = 0; j < 2; j++) {
      FragB frag_b[2];

      if constexpr (is_u3) {
        // [0055 KU3S] registers R[4 j .. 4 j + 3] (frag_b[h] = {R[4 j + 2 h], R[4 j + 2 h + 1]}): j = 0 reads W0 + W1, j = 1 W0 + W2
        uint32_t* fb = reinterpret_cast<uint32_t*>(&frag_b);
        const uint32_t w0 = static_cast<uint32_t>(frag_b_quant[k2][0][0]);
        if (j == 0)
          lowbit::unpack3_j<0>(w0, static_cast<uint32_t>(frag_b_quant[k2][0][1]), fb);
        else
          lowbit::unpack3_j<1>(w0, static_cast<uint32_t>(frag_b_quant[k2][0][2]), fb);
      } else
      if (is_a_8bit && b_type.size_bits() == 4 && !has_zp) {
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b));
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2);
      } else if (is_a_8bit && b_type.size_bits() == 4 && has_zp) {
  #if MARLIN_ZPPACK_TWIN
        // [0051 ZPPACK] word 2 j + h of the lane group's int2 carries the family's nibble 4 * (warp % 2) + 2 j + h
        const int zpp_word = j == 0 ? frag_zz[k2].x : frag_zz[k2].y;
        int zp = (zpp_word >> 12) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b), zp);
        zp = static_cast<int>(static_cast<uint32_t>(zpp_word) >> 28);
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2, zp);
  #else
        int off = (threadIdx.x / 32) % 2 * 2 + j;
        int zp = (frag_qzp[k2][0] >> (off * 8)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b), zp);
        zp = (frag_qzp[k2][0] >> (off * 8 + 4)) & 0xF;
        dequant_data(frag_b_quant[k2][0][j * 2 + 1],
                     reinterpret_cast<scalar_32bit_t*>(&frag_b) + 2, zp);
  #endif
      } else {
        reinterpret_cast<int2*>(&frag_b)[0] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[0];
        reinterpret_cast<int2*>(&frag_b)[1] =
            reinterpret_cast<int2*>(&frag_b_quant[k2][j])[1];
      }

  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[0],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][0]);
        mma<a_type_id, false, 32>(
            frag_a[k2][i], frag_b[1],
            (group_blocks == -1 ? frag_c : frag_c_tmp)[i][j][1]);
      }

      if constexpr (group_blocks != -1) {
        if (group_blocks == 2 || k == 1) {
          if constexpr (a_type == vllm::kS8) {
            int2 s_vals[2];
  #if MARLIN_ZPPACK_TWIN
            // [0051 ZPPACK] code = the low 12 bits of the word, u -> 4096 (the family reads the int16 word, 0 .. 4096)
            auto zpp_code = [&](uint32_t f) { return static_cast<int>(f == zpp_u ? 4096u : f); };
            const uint32_t zpp_w0 = *reinterpret_cast<uint32_t*>(&frag_s[k2][j * 2][0]);
            const uint32_t zpp_w1 = *reinterpret_cast<uint32_t*>(&frag_s[k2][j * 2 + 1][0]);
            s_vals[0] = {zpp_code(zpp_w0 & 0xFFFu), zpp_code((zpp_w0 >> 16) & 0xFFFu)};
            s_vals[1] = {zpp_code(zpp_w1 & 0xFFFu), zpp_code((zpp_w1 >> 16) & 0xFFFu)};
  #else
            s_vals[0] = {
                (int)reinterpret_cast<int16_t*>(&frag_s[k2][j * 2][0])[0],
                (int)reinterpret_cast<int16_t*>(&frag_s[k2][j * 2][0])[1]};
            s_vals[1] = {
                (int)reinterpret_cast<int16_t*>(&frag_s[k2][j * 2 + 1][0])[0],
                (int)reinterpret_cast<int16_t*>(&frag_s[k2][j * 2 + 1][0])[1]};
  #endif

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[0])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][0][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][0][g]) *
                    scale;
                frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                int scale = reinterpret_cast<int*>(&s_vals[1])[g % 2];
                *reinterpret_cast<int32_t*>(&frag_c[i][j][1][g]) +=
                    *reinterpret_cast<int32_t*>(&frag_c_tmp[i][j][1][g]) *
                    scale;
                frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          } else {
            float2 s_vals[2];
            if constexpr (s_type_id != vllm::kFE8M0fnu.id()) {
              static_assert(a_type.size_bits() == 16 ||
                            s_type.size_bits() == 16);
              s_vals[0] = Cdtype::num22float2(frag_s[k2][j * 2][0]);
              s_vals[1] = Cdtype::num22float2(frag_s[k2][j * 2 + 1][0]);
            } else {
              int32_t* s_vals_int = reinterpret_cast<int32_t*>(&s_vals[0]);
              int32_t s_vals_e8m0 =
                  *reinterpret_cast<int32_t*>(&frag_s[k2][j][0]);

              s_vals_int[0] = (s_vals_e8m0 & 0xFF) << 23;
              s_vals_int[1] = (s_vals_e8m0 & 0xFF00) << 15;
              s_vals_int[2] = (s_vals_e8m0 & 0xFF0000) << 7;
              s_vals_int[3] = (s_vals_e8m0 & 0xFF000000) >> 1;
            }

  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[0])[g % 2];
                frag_c[i][j][0][g] += frag_c_tmp[i][j][0][g] * scale;
                frag_c_tmp[i][j][0][g] = 0.0f;
              }

  #pragma unroll
              for (int g = 0; g < 4; g++) {
                float scale = reinterpret_cast<float*>(&s_vals[1])[g % 2];
                frag_c[i][j][1][g] += frag_c_tmp[i][j][1][g] * scale;
                frag_c_tmp[i][j][1][g] = 0.0f;
              }
            }
          }
        }
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride =
          b_sh_stride_threads * (is_a_8bit ? 2 : 4) * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) +
                      (threadIdx.x % b_sh_stride_threads);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

  #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
  #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
  #pragma unroll
            for (int j = 0; j < (is_a_8bit ? 2 : 4) * 2;
                 j += (m_block_size_8 ? 2 : 1)) {
              int red_sh_wr =
                  red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                if constexpr (ired) {  // [0056 IRED v3] int32 partials by int4 VALUE, unsigned adds: exact mod 2^32, order-free
                  ired_add4(frag_c[m_block][j / 2][j % 2], sh_red[red_sh_delta * j + red_sh_rd], sh_red[red_sh_wr]);
                } else {
                float* c_rd = reinterpret_cast<float*>(
                    &sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
  #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(
                      frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j][k] +=
                      c_rd[k] + c_wr[k];
                }
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
  #pragma unroll
          for (int i = 0; i < (is_a_8bit ? 2 : 4) * 2;
               i += (m_block_size_8 ? 2 : 1)) {
            if constexpr (ired) {  // [0056 IRED v3] int4 VALUE, unsigned adds: exact mod 2^32
              ired_add4(frag_c[m_block][i / 2][i % 2], sh_red[red_sh_delta * i + red_sh_rd]);
            } else {
            float* c_rd =
                reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
  #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(
                  frag_c)[(is_a_8bit ? 2 : 4) * 2 * m_block + i][j] += c_rd[j];
            }
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * tb_n_warps;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride * (is_a_8bit ? 2 : 1);
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr;
      if constexpr (m_block_size_8) {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 4) * 2) +
                  4 * (threadIdx.x / 32) + (threadIdx.x % 32) / 8;
        c_gl_wr += (2 * thread_n_blocks) * slice_col;
      } else {
        c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) * (is_a_8bit ? 2 : 1) +
                  4 * (threadIdx.x / 32) + threadIdx.x % 4;
        c_gl_wr += (2 * thread_n_blocks) * slice_col * (is_a_8bit ? 2 : 1);
      }
      constexpr int c_sh_wr_delta = active_threads;
      auto c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
  // Interestingly, doing direct global accesses here really seems to mess up
  // the compiler and lead to slowdowns, hence we also use async-copies even
  // though these fetches are not actually asynchronous.
  #pragma unroll
        for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
          if constexpr (m_block_size_8) {
            cp_async4_pred(&sh_red[c_sh_wr + c_sh_wr_delta * i],
                           &C[c_gl_wr + i * c_gl_stride +
                              (threadIdx.x % 8) / 4 * c_gl_wr_delta_i],
                           (threadIdx.x % 4) * 2 + i < prob_m);
          } else if constexpr (is_a_8bit) {
            int2* sh_red_int2 = reinterpret_cast<int2*>(sh_red);
            int2* c_int2 = reinterpret_cast<int2*>(C);
            cp_async2_ca_pred(
                &sh_red_int2[c_sh_wr + c_sh_wr_delta * i],
                &c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                        c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          } else {
            cp_async4_pred(
                &sh_red[c_sh_wr + c_sh_wr_delta * i],
                &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                   c_gl_wr_delta_i * (i % 2)],
                i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m);
          }
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

  #pragma unroll
      for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
        bool mask = (!m_block_size_8) && (i < (thread_m_blocks - 1) * 4 ||
                                          8 * (i / 2) + row < prob_m) ||
                    (m_block_size_8) && ((threadIdx.x % 4) * 2 + i < prob_m);
        if (mask) {
          if (!first) {
            c_scalar_t* c_red_f16;
            if constexpr (is_a_8bit) {
              int2 tmp =
                  reinterpret_cast<int2*>(sh_red)[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            } else {
              int4 tmp = sh_red[c_sh_wr + i * c_sh_wr_delta];
              c_red_f16 = reinterpret_cast<c_scalar_t*>(&tmp);
            }
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta] += Cdtype::num2float(c_red_f16[j]);
            }
          }
          if (!last) {
            c_scalar_t c_f16[is_a_8bit ? 4 : 8];
  #pragma unroll
            for (int j = 0; j < 2 * (is_a_8bit ? 2 : 4); j++) {
              int delta = 0;
              if constexpr (m_block_size_8) {
                delta = j % 2 == 1 ? -2 : 0;
              }
              c_f16[j] = Cdtype::float2num(reinterpret_cast<float*>(
                  &frag_c)[(is_a_8bit ? 2 : 4) * 2 * 4 * (i / 4) + 4 * j +
                           (i % 4) + delta]);
            }
            if constexpr (m_block_size_8) {
              C[c_gl_wr + i * c_gl_stride +
                (threadIdx.x % 8) / 4 * c_gl_wr_delta_i] =
                  *reinterpret_cast<int4*>(c_f16);
            } else if constexpr (is_a_8bit) {
              int2* c_int2 = reinterpret_cast<int2*>(C);
              c_int2[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                     c_gl_wr_delta_i * (i % 2)] =
                  *reinterpret_cast<int2*>(c_f16);
            } else {
              C[c_gl_wr + c_gl_wr_delta_o * (i / 2) +
                c_gl_wr_delta_i * (i % 2)] = *reinterpret_cast<int4*>(c_f16);
            }
          }
        }
      }
    }
  };

  #if MARLIN_OPR_TWIN
  // [0040 OPR] Ordered parallel reduce of one split column; replaces the served chain's barrier_acquire /
  // global_reduce_fp32 / barrier_release. The served chain makes slice i wait for slice i-1, load the running sum, add,
  // store, fence and release, so the slices that finish together at kernel end (each CTA's last piece) serialise one
  // link each. Here no slice waits for another except the column's FINAL slice (slice_idx == slice_count - 1: always
  // its CTA's last piece, and the slice the served chain already makes write C):
  //   * every other slice stores its OWN fp32 partial p_i to its CTA's slot (below) with the served store
  //     pattern and releases one arrival (barrier_release: __syncthreads, fence.acq_rel.gpu, red.relaxed.gpu +1);
  //   * the final slice waits for slice_count - 1 arrivals (barrier_acquire), folds the slots in slice_idx order
  //     v = p_0, v = p_i + v (i = 1 .. slice_count - 2) and adds its own registers last, C = p_{S-1} + v: the served
  //     chain's grouping p_{S-1} + (... + (p_1 + p_0)) operand for operand, so the output is the served output bitwise.
  //     It writes no slot and issues no fence; it drops the lines it read (0024) and resets the counter.
  // Slots are per CTA: a non-final slice starts mid-column, so it is its CTA's FIRST part-2 piece and a CTA holds at most one;
  // it writes slot blockIdx.x (C_tmp + blockIdx.x * c_size). The CTAs covering a column are consecutive, lowest rows first,
  // and slice_idx counts from the top, so slice i < S-1 of the column whose final slice runs in CTA b sits in slot
  // b + (S-1-i). gridDim.x slots of c_size fp32 fit the served persistent c_tmp at every tile (qkv's 64-KB partials: 8 MiB
  // exactly); the host checks it. No slot or counter is reused inside a launch and the next launch on the stream is ordered
  // after this one, hence the discards need no trailing fence (the served chain's 0024 fence guards its one shared slot).
  auto opr_reduce_fp32 = [&]() {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;
    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;
    constexpr int active_threads = 32 * tb_n_warps;
    constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;
    const bool is_th_active = threadIdx.x < active_threads;

    if (slice_idx != slice_count - 1) {
      if (is_th_active) {
        int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
        const int dst = blockIdx.x * c_size + threadIdx.x;
    #pragma unroll
        for (int k = 0; k < th_size; k++) {
          C_tmp[dst + active_threads * k] = frag_c_ptr[k];
        }
      }
      barrier_release(&locks[locks_off]);
      return;
    }

    barrier_acquire(&locks[locks_off], slice_count - 1);
    if (is_th_active) {
      const int n_prev = slice_count - 1;  // 1 .. opr_max_slices - 1 partials to fold
      const int p0_slot = (blockIdx.x + n_prev) * c_size + threadIdx.x;  // slice i at p0_slot - i * c_size
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
    #pragma unroll
      for (int k = 0; k < th_size; k++) {
        const int e = p0_slot + active_threads * k;
        int4 p[opr_max_slices - 1];
        p[0] = C_tmp[e];
    #pragma unroll
        for (int i = 1; i < opr_max_slices - 1; i++) {
          if (i < n_prev) p[i] = C_tmp[e - i * c_size];
        }
    #pragma unroll
        for (int f = 0; f < 4; f++) {
          if constexpr (ired) {  // [0056 IRED] int32 slots: exact, order-free
            unsigned v = ired_lane(p[0], f);  // [0056 IRED v3] int4 lanes by VALUE
    #pragma unroll
            for (int i = 1; i < opr_max_slices - 1; i++) {
              if (i < n_prev) v = ired_lane(p[i], f) + v;
            }
            frag_c_ptr[k * 4 + f] = __uint_as_float(__float_as_uint(frag_c_ptr[k * 4 + f]) + v);
          } else {
          float v = reinterpret_cast<float*>(&p[0])[f];
    #pragma unroll
          for (int i = 1; i < opr_max_slices - 1; i++) {
            if (i < n_prev) v = reinterpret_cast<float*>(&p[i])[f] + v;
          }
          frag_c_ptr[k * 4 + f] += v;
          }
        }
      }
      // 0024: the slots read above are dead until a later launch rewrites them; drop their lines instead of letting an
      // eviction write them back. Every load was consumed by the adds; __syncwarp orders the lanes' reads before the
      // discard of the line's first lane (c_size and active_threads are multiples of 8 int4s).
      __syncwarp();
      for (int i = 0; i < n_prev; i++) {
    #pragma unroll
        for (int k = 0; k < th_size; k++) {
          const int e = p0_slot - i * c_size + active_threads * k;
          if ((e & 7) == 0) l2_discard_line(&C_tmp[e]);
        }
      }
    }
    barrier_release(&locks[locks_off], true);
  };
  #endif

  // Globally reduce over threadblocks that compute the same column block.
  // We use a tmp C buffer to reduce in full fp32 precision.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

    constexpr int active_threads = 32 * tb_n_warps;
    bool is_th_active = threadIdx.x < active_threads;

    constexpr int num_floats = thread_m_blocks * (is_a_8bit ? 2 : 4) * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = locks_off * c_size;

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        sh_red[threadIdx.x] =
            C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        if constexpr (ired) {  // [0056 IRED] int32 running sum: exact, order-free
          const int4 v = sh_red[threadIdx.x];  // [0056 IRED v3] the partial by VALUE (v2p's punned pointer: .x added 4x, RID 835)
  #pragma unroll
          for (int f = 0; f < 4; f++) frag_c_ptr[k * 4 + f] = __uint_as_float(__float_as_uint(frag_c_ptr[k * 4 + f]) + ired_lane(v, f));
        } else {
        float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
  #pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
        }
      }
      if constexpr (!m_block_size_8) {
        if (last) {
          // The slice's last block has now consumed every partial in this slot.
          // Nothing reads the slot again before a later slice's FIRST block (or
          // the next launch) rewrites it, so drop its lines instead of letting a
          // later eviction write them back to DRAM. c_cur_offset and
          // active_threads are multiples of 8 int4s, so the lane whose element
          // starts a 128-byte line owns it, and its 7 neighbours in the line are
          // lanes of the same warp.
          __syncwarp();
  #pragma unroll
          for (int k = 0; k < th_size; k++) {
            int e = c_cur_offset + active_threads * k + threadIdx.x;
            if ((e & 7) == 0) l2_discard_line(&C_tmp[e]);
          }
          // A discard is a weak write. barrier_release(reset=true) frees the
          // slot with a plain store and no fence, so without this fence
          // another block could see the lock free and write new partials
          // before the discard lands, and lose them.
          __threadfence();
        }
      }
    }

    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
  #pragma unroll
      for (int k = 0; k < th_size; k += (m_block_size_8 ? 2 : 1)) {
        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&](bool last) {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta =
        c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) +
                (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr =
          (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += (is_a_8bit ? 16 : 32) * (threadIdx.x / 32);
    }

    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) +
                  (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;
  #if MARLIN_FOLD_TWIN
    // [0042 FOLD] rows [0, par_id * 64 - row_off(par_id)) of this copy belong to the previous copy (folded last copy only)
    const int c_gl_wr_begin = c_gl_stride * (par_id * m_block_size - row_off(par_id));
  #elif MARLIN_FOLDS_TWIN
    // [0048 FOLDS] the remainder copy's first 64 - r rows are copy p - 1's (launch A's) rows: skip them
    const int c_gl_wr_begin = c_gl_stride * (fs_R() ? m_block_size - fs_m % m_block_size : 0);
  #endif
    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s, FragS& b_bias) {
      if constexpr (b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn) {
        c0 *= global_scale_f32;
        c1 *= global_scale_f32;
      }
      c_scalar_t2 res =
          Cdtype::nums2num2(Cdtype::float2num(c0), Cdtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (group_blocks == -1 && !is_a_8bit &&
                    b_type.size_bits() == 4 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        c_scalar_t2 tmp_scale = s[0];
        if constexpr (m_block_size_8) {
          tmp_scale = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&s[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hmul2(res, tmp_scale);
      }
      if (has_bias && last) {
        c_scalar_t2 tmp_bias = b_bias[0];
        if constexpr (m_block_size_8) {
          tmp_bias = Cdtype::num2num2(
              reinterpret_cast<scalar_t*>(&b_bias[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hadd2(res, tmp_bias);
      }

      if constexpr (m_block_size_8) {
        ((c_scalar_t*)sh_red)[idx] = res.x;
        ((c_scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((c_scalar_t2*)sh_red)[idx] = res;
      }
    };

    if (threadIdx.x / 32 < tb_n_warps) {
      if constexpr (is_u3 || colf_a8) {  // [0059 W4RCOLF] + kU4 / kU4B8 grouped int8-activation instances
        // [0055 KU3S-4] optional per-output-column power-of-two factor f[n] = 2^(L_n - L) (fp32 [N], host: global_scale_or_none),
        // applied to the reduced fp32 frag_c before write() rounds it to bf16: x * 2^l is exact in fp32 and commutes with every
        // rounding of the int -> fp32 -> reduce chain, so the result equals the same GEMM on codes * 2^l (no int16 / int32 range
        // needed). nullptr (no column factor) = the per-tensor path, arithmetic unchanged. Column of frag_c[i][j][h][g] (a8, 16-row
        // m blocks; write() below puts it in sh_red slot c_sh_wr + 8 j + 4 h (+ 8 rows for g >= 2) as value g % 2):
        // 16 * thread_n_blocks * slice_col + 2 * (lane % 4 + 16 * warp + 8 * j + 4 * h) + g % 2. One factor pair per (j, h).
        if (global_scale_ptr != nullptr) {
          const int u3_n0 = 16 * thread_n_blocks * slice_col + 2 * ((threadIdx.x % 32) % 4 + 16 * (threadIdx.x / 32));
  #pragma unroll
          for (int j = 0; j < 2; j++) {
  #pragma unroll
            for (int h = 0; h < 2; h++) {
              const float2 u3_f = __ldg(reinterpret_cast<const float2*>(global_scale_ptr) + (u3_n0 + 16 * j + 8 * h) / 2);
  #pragma unroll
              for (int i = 0; i < thread_m_blocks; i++) {
                frag_c[i][j][h][0] *= u3_f.x;
                frag_c[i][j][h][1] *= u3_f.y;
                frag_c[i][j][h][2] *= u3_f.x;
                frag_c[i][j][h][3] *= u3_f.y;
              }
            }
          }
        }
      }
  #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
        for (int j = 0; j < (is_a_8bit ? 2 : 4); j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(wr, frag_c[i][j][0][0], frag_c[i][j][0][1],
                  frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + 8, frag_c[i][j][0][2], frag_c[i][j][0][3],
                  frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          } else {
            int wr = c_sh_wr + 8 * j;
            write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0],
                  frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2],
                  frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0],
                  frag_bias[j / 2][2 * (j % 2) + 0]);
            write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0],
                  frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
            write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2],
                  frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1],
                  frag_bias[j / 2][2 * (j % 2) + 1]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

  #pragma unroll
    for (int i = 0;
         i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks));
         i++) {
      if (c_gl_wr < c_gl_wr_end) {
  #if MARLIN_FOLD_TWIN || MARLIN_FOLDS_TWIN
        if (c_gl_wr < c_gl_wr_begin) {
          // [0042 FOLD] overlap row: owned and written by the previous copy
        } else
  #endif
        if (use_atomic_add && slice_count > 1) {
          c_scalar_t2* C_half2 = reinterpret_cast<c_scalar_t2*>(&C[c_gl_wr]);
          c_scalar_t2* sh_red_half2 =
              reinterpret_cast<c_scalar_t2*>(&sh_red[c_sh_rd]);
  #pragma unroll
          for (int a = 0; a < 4; a++) {
            atomicAdd(&C_half2[a], sh_red_half2[a]);
          }
        } else {
          C[c_gl_wr] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 5  // [0061 COLFPF v2] prefetch site 5 (p_spdef)
  if constexpr (is_u3 || colf_a8) {  // [exp p_spdef]
    const int colf_l = blockIdx.x * threads + threadIdx.x;
    if (colf_l < (prob_n + 31) / 32) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(global_scale_ptr + 32 * colf_l));
  }
  #endif
  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {

  #pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
        if (i == 0) {
          fetch_col_zp_to_shared();
          if constexpr (!dequant_skip_flop) {
            fetch_col_scale_to_shared();
          }
        }
      }
      fetch_to_shared(i, i, i < slice_iters);
    }

    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };
  if (slice_iters) {
    start_pipes();
  }
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 9  // [0061 COLFPF v2] prefetch site 9 (f_asmlines_pp)
  if constexpr (is_u3 || colf_a8) { colf_pf_f<threads>(global_scale_ptr, (prob_n + 31) / 32); }  // [exp form]
  #endif
  #if MARLIN_CPF_TWIN && MARLIN_CPF_SITE == 2  // [0061 COLFPF v2] prefetch site 2 (postpipes)
  if constexpr (is_u3 || colf_a8) {  // [exp postpipes]
    if (global_scale_ptr != nullptr) {
      const int colf_l = blockIdx.x * threads + threadIdx.x;
      if (colf_l < (prob_n + 31) / 32) asm volatile("prefetch.global.L2::evict_normal [%0];\n" ::"l"(global_scale_ptr + 32 * colf_l));
    }
  }
  #endif
  #if MARLIN_ZPPACK_TWIN
  if (threadIdx.x == 0 && zpp_seen != static_cast<int>(0x05100000u | zpp_u)) __trap();  // [0051 ZPPACK] guard (see top)
  #endif

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.

  #pragma unroll
    for (int pipe = 0; pipe < stages;) {
  #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        fetch_scales_to_registers(k + 1, pipe);
        fetch_zp_to_registers(k + 1, pipe);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe,
                          slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }

        if constexpr (!is_a_8bit) {
          matmul(k, pipe - (k >= b_sh_wr_iters - 2 ? 1 : 0));
        } else {
          static_assert(group_blocks != 0 && group_blocks != 1);
          matmul_a8(k);
        }
      }
      slice_iters--;
      if (slice_iters == 0) {
        break;
      }
    }

    a_gl_rd += a_gl_rd_delta_o * stages;

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
  #if MARLIN_XS_TWIN
      // [0035 XS] Predict the next segment exactly as the transition below + init_slice() will set it
      // (parallel == 1, so par_id stays 0 and the tile-major part-1 map is the identity), point the stage
      // pointers at it exactly as the post-init_slice() reset does, and issue the same `stages - 1`
      // fetch_to_shared() calls start_pipes() would issue, in the same order and with the same predicates.
      // They now overlap this segment's epilogue (block reduce, lock-chain reduce, C write) instead of
      // following it. The cp.async group count per segment is unchanged, so every wait_for_stage()
      // still waits for exactly the tile it waited for before.
      if constexpr (xs_compiled) {
        if (xs_run) {
          int n_col_par, n_col, n_row, n_iters;
          if (!in_part2 && part1_mn_iters) {  // next = another whole part-1 column
            n_col_par = slice_col_par + gridDim.x;
            n_col = n_col_par % n_tiles;
            n_row = 0;
            n_iters = k_tiles;
          } else {
            if (!in_part2) {  // part 1 is done: init_slice() switches to the part-2 stripe
              n_col_par = (iters * blockIdx.x) / k_tiles;
              n_row = (iters * blockIdx.x) % k_tiles;
              n_col = (n_col_par + global_mn_tiles - part2_mn_tiles) % n_tiles;
            } else {  // next column of the part-2 stripe
              n_col_par = slice_col_par + 1;
              n_row = 0;
              n_col = slice_col + 1;
            }
            n_iters = iters * (blockIdx.x + 1) - (k_tiles * n_col_par + n_row);
            if (n_iters < 0 || n_col_par >= part2_mn_tiles) n_iters = 0;
            if (n_iters != 0 && n_row + n_iters > k_tiles) n_iters = k_tiles - n_row;
          }
    #if MARLIN_XSLICE_CHECK
          xs_n_col = n_col;
          xs_n_row = n_row;
          xs_n_iters = n_iters;
    #endif
          if (n_iters) {
            // Every warp has finished the last k-step of this segment (its fetch_to_registers reads a
            // stage slot this prefetch is about to overwrite; the value is discarded, but no race is left).
            __syncthreads();
            a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                      (threadIdx.x % a_gl_rd_delta_o);
            a_gl_rd += a_gl_rd_delta_o * n_row;
            b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                      (threadIdx.x % b_sh_stride);
            b_gl_rd += b_sh_stride * n_col + b_gl_rd_delta_o * n_row;
            if constexpr (is_u3) b_gl_rd = u3_row * n_col + b_gl_rd_delta_o * n_row;  // [0055 KU3S] CTA-uniform stage base
            if constexpr (group_blocks >= thread_k_blocks) {
              s_gl_rd = s_gl_stride * ((thread_k_blocks * n_row) / group_blocks) +
                        s_sh_stride * n_col + threadIdx.x;
              zp_gl_rd =
                  zp_gl_stride * ((thread_k_blocks * n_row) / group_blocks) +
                  zp_sh_stride * n_col + threadIdx.x;
            } else {
              s_gl_rd =
                  s_gl_stride * ((thread_k_blocks * n_row) / group_blocks +
                                 threadIdx.x / s_sh_stride) +
                  s_sh_stride * n_col + threadIdx.x % s_sh_stride;
              zp_gl_rd =
                  zp_gl_stride * ((thread_k_blocks * n_row) / group_blocks +
                                  threadIdx.x / zp_sh_stride) +
                  zp_sh_stride * n_col + threadIdx.x % zp_sh_stride;
            }
    #pragma unroll
            for (int i = 0; i < stages - 1; i++) {
              fetch_to_shared(i, i, i < n_iters);
            }
            xs_prefetched = true;
          }
        }
      }
  #endif
      // convert fp16 accum to fp32 for reduction
      if constexpr (use_fp16_accum) {
  #pragma unroll
        for (int i = 0; i < (thread_m_blocks * (is_a_8bit ? 2 : 4) * 2); i++) {
          float* frag_c_part_float = reinterpret_cast<float*>(frag_c) + i * 4;
          scalar_t* frag_c_part_half =
              reinterpret_cast<scalar_t*>(frag_c_part_float);

  #pragma unroll
          for (int i = 3; i >= 0; i--) {
            frag_c_part_float[i] = Cdtype::num2float(frag_c_part_half[i]);
          }
        }
      }

      // [0056 IRED] the row act scales, loaded where the served code loads them, applied after the reduce (below)
      [[maybe_unused]] float ired_a_s[ired ? 2 * thread_m_blocks : 1];
      if constexpr (ired) {
        for (int i = 0; i < 2 * thread_m_blocks; i++)
          ired_a_s[i] = sh_a_s[i * 8 + (threadIdx.x % 32) / 4];
      } else if constexpr (is_a_8bit) {
        float frag_a_s[2 * thread_m_blocks];

        for (int i = 0; i < 2 * thread_m_blocks; i++)
          frag_a_s[i] = sh_a_s[i * 8 + (threadIdx.x % 32) / 4];

  #pragma unroll
        for (int j = 0; j < 2; j++) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][0][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][0][g] = c_val * s_val;
            }
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float c_val = frag_c[i][j][1][g];

              if constexpr (a_type == vllm::kS8) {
                c_val = __int2float_rn(*reinterpret_cast<int32_t*>(&c_val));
              }
              float s_val = frag_a_s[i * 2 + g / 2];
              frag_c[i][j][1][g] = c_val * s_val;
            }
          }
        }
      }

  #if MARLIN_XS_TWIN
      // [0035 XS] With the prefetch in flight, the only outstanding groups are the next segment's stages
      // (this segment's tiles were all waited for in the main loop, and XS excludes every config that
      // issues a cp.async in the epilogue). sh_red no longer aliases sh_b, so nothing here needs them.
      if (!xs_prefetched) cp_async_wait<0>();
  #else
      cp_async_wait<0>();
  #endif
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp)) {
        if (b_type.size_bits() == 8 || (last || use_atomic_add) || is_a_8bit) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();

      if (has_bias && last) {
        __syncthreads();
        cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd],
                       threadIdx.x < 16 * thread_n_blocks / 8);
        cp_async_fence();
      }

      if constexpr (group_blocks == -1 &&
                    (has_zp && dequant_skip_flop || !has_zp || is_a_8bit)) {
        if constexpr (is_a_8bit) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
          }
        } else if (b_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < tb_n_warps) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              c_scalar_t2* frag_s_half2 =
                  reinterpret_cast<c_scalar_t2*>(frag_s);
  #pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Cdtype::num2num2(
                    reinterpret_cast<c_scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (group_blocks == -1 && is_a_8bit) {
  #pragma unroll
        for (int j = 0; j < 2; j++) {
          float2 aa[2];
          aa[0] = Cdtype::num22float2(frag_s[0][j * 2][0]);
          aa[1] = Cdtype::num22float2(frag_s[0][j * 2 + 1][0]);

  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[0])[g % 2];
              frag_c[i][j][0][g] *= scale;
            }

  #pragma unroll
            for (int g = 0; g < 4; g++) {
              float scale = reinterpret_cast<float*>(&aa[1])[g % 2];
              frag_c[i][j][1][g] *= scale;
            }
          }
        }
      } else if (group_blocks == -1 && b_type.size_bits() == 8 &&
                 (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < tb_n_warps) {
  #pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][0]),
                  frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<c_type_id>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]),
                  frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][0]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<c_type_id>(
                    reinterpret_cast<float*>(&frag_c[i][j][1][2]),
                    frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if (slice_count > 1 && !use_atomic_add) {
        // only globally reduce if there is more than one block in a slice
  #if MARLIN_OPR_TWIN
        // [0040 OPR] The host selects a MarlinOPR twin only for a launch with the fp32 reduce and opr_smax() <= 4, so
        // the twin carries no served-chain fallback (a second reduce path costs registers in the 255-register A16 tiles);
        // a launch that breaks the precondition traps instead of reducing in a different order.
        if (!use_fp32_reduce || opr_smax() > opr_max_slices) __trap();
        opr_reduce_fp32();  // `last` keeps its served meaning: the final slice writes C below
  #else
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          if constexpr (ired) __trap();  // [0056 IRED] int32 partials: the fp16 chain would read them as halves
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
  #endif
      }
      if constexpr (ired) {
        // [0056 IRED] an atomic-add reduce would add int32 bits as floats
        if (use_atomic_add && slice_count > 1) __trap();
        // Only the column's FINAL slice holds the whole-K int32 sum: convert it ONCE (__int2float_rn, the served conversion) and
        // apply the row's act scale -- the served per-slice arithmetic, moved after the reduce. Non-final slices stored int32.
        if (last) {
  #pragma unroll
          for (int j = 0; j < 2; j++) {
  #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
  #pragma unroll
              for (int h = 0; h < 2; h++) {
  #pragma unroll
                for (int g = 0; g < 4; g++) {
                  float c_val = __int2float_rn(__float_as_int(frag_c[i][j][h][g]));  // [0056 IRED v3] bit cast, no pointer pun
                  frag_c[i][j][h][g] = c_val * ired_a_s[i * 2 + g / 2];
                }
              }
            }
          }
        }
      }

      if (has_bias && last) {
        cp_async_wait<0>();
        __syncthreads();
        reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
        if constexpr (!is_a_8bit)
          reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
        __syncthreads();
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0)
        wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result(last);
      slice_row = 0;
      if (!in_part2) {
        slice_col_par += gridDim.x;
      } else {
        slice_col_par++;
        slice_col++;
      }
      is_first_matmul_in_slice = true;
      init_slice();

  #if MARLIN_XS_TWIN
      if (xs_prefetched) {
    #if MARLIN_XSLICE_CHECK
        if (slice_col != xs_n_col || slice_row != xs_n_row ||
            slice_iters != xs_n_iters)
          __trap();
    #endif
        // [0035 XS] The stage pointers already sit where start_pipes() leaves them (fetch_to_shared
        // advanced b/s/zp per issued tile; a_gl_rd still holds the segment base). Do start_pipes()'s
        // second half: the first wait_for_stage() waits for tile 0 exactly as before.
        xs_prefetched = false;
        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        fetch_scales_to_registers(0, 0);
        fetch_zp_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
      } else
  #endif
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) +
                  (threadIdx.x % a_gl_rd_delta_o);
        a_gl_rd += a_gl_rd_delta_o * slice_row;
        b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) +
                  (threadIdx.x % b_sh_stride);
        b_gl_rd += b_sh_stride * slice_col + b_gl_rd_delta_o * slice_row;
        if constexpr (is_u3) b_gl_rd = u3_row * slice_col + b_gl_rd_delta_o * slice_row;  // [0055 KU3S] CTA-uniform stage base

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        if constexpr (group_blocks == -1) {
          s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
        } else if constexpr (group_blocks >= thread_k_blocks) {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              s_sh_stride * slice_col + threadIdx.x;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) +
              zp_sh_stride * slice_col + threadIdx.x;
        } else {
          s_gl_rd =
              s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                             threadIdx.x / s_sh_stride) +
              s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
          zp_gl_rd =
              zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks +
                              threadIdx.x / zp_sh_stride) +
              zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
        }
        start_pipes();
      }
    }
  }
  #if MARLIN_TPF_TWIN
  // [0043 TAILPF] The kernel's last statement (tpf_prefetch_next above). Every CTA has left the main loop and finished its
  // last epilogue, so nothing of the kernel is live here: the prefetch adds no register to the body. The CTAs that end first
  // (no work, or a split column's non-final slices) issue first, while the final slices still reduce and write.
    #if MARLIN_TPF_KU3S_GC
  tpf_prefetch_next_gc<threads>(&tpf);  // [0063 KU3STPF] out of line (MARLIN_TPF_KU3S_GC)
    #else
  tpf_prefetch_next<threads>(tpf);
    #endif
  #endif
}

}  // namespace MARLIN_NAMESPACE_NAME

#endif
