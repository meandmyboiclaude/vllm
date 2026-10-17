
#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

// [0035 XS] Cross-slice prefetch. Build switch: 0 (default) = the served kernel and host, byte for byte (every XS
// line sits under `#if MARLIN_XSLICE`). In an XS build the RUN switch is the env VLLM_MARLIN_XSLICE (default 0 = OFF). 1 = issue the next column segment's first `stages - 1` global->shared fetches
// at the END of the current segment (before its epilogue) instead of after it, with the reduce scratch
// (sh_red) moved off the B stage buffers so the epilogue no longer drains the cp.async pipeline. Only load
// timing moves; tiles, k order, partition, slice reduce order and writes are unchanged.
#ifndef MARLIN_XSLICE
  #define MARLIN_XSLICE 1
#endif
// 1 = trap if the XS next-segment prediction ever differs from init_slice() (KAT builds only).
#ifndef MARLIN_XSLICE_CHECK
  #define MARLIN_XSLICE_CHECK 0
#endif
// [0036 XS-5] Since XS3, MARLIN_XSLICE builds only the HOST side (marlin.cu: the per-launch gate + the twin selector) and
// lets generate_kernels.py's twin TU instantiate MarlinXS<...>; the kernel bodies are switched by MARLIN_XS_TWIN (marlin_template.h),
// which only that TU sets. Every served Marlin<...> instantiation therefore compiles exactly as without XS.
// [0040 OPR] Build switch for the ordered-parallel-reduce twins (default 1): generate_kernels.py's TU sm80_kernel_opr_<c_type>.cu
// instantiates MarlinOPR<...> (MARLIN_OPR_TWIN 1) under it and marlin.cu carries the host gate. The RUN switch is the env
// VLLM_MARLIN_OPR (unset/0 = OFF = the served launch). 0 = neither the twins nor the gate are compiled.
#ifndef MARLIN_OPR
  #define MARLIN_OPR 1
#endif
// [0041 MR14] Build switch for the group-scale L2 prefetch twins (default 1): generate_kernels.py's TUs
// sm80_kernel_sp*_<c_type>.cu instantiate <family>SP<...> (MARLIN_SP_TWIN 1) under it and marlin.cu carries the host gate.
// The RUN switch is the env VLLM_MARLIN_SCALE_PREFETCH (unset/0 = OFF = the launch the earlier selections chose).
// 0 = neither the twins nor the gate are compiled.
#ifndef MARLIN_SCALE_PF
  #define MARLIN_SCALE_PF 1
#endif
// [0045 L2PF] Build switch for the B-stream L2 prefetch-size twins (default 1): generate_kernels.py's TUs
// sm80_kernel_l2pf*_<c_type>.cu instantiate <family>L2PF<...> (MARLIN_L2PF_TWIN 1) under it and marlin.cu carries the host
// gate. The RUN switch is the env VLLM_MARLIN_L2PF (unset/0 = OFF = the launch the earlier selections chose). 0 = neither
// the twins nor the gate are compiled.
#ifndef MARLIN_L2PF
  #define MARLIN_L2PF 1
#endif
// [0042 FOLD] Build switch for the remainder-fold twins (default 1): generate_kernels.py's TU sm80_kernel_fold_<c_type>.cu
// instantiates MarlinFOLD<...> (MARLIN_FOLD_TWIN 1) under it and marlin.cu carries the host fold. The RUN switch is the env
// VLLM_MARLIN_REMAINDER_FOLD_NK (unset/empty = never fold = the served launches). 0 = neither the twins nor the fold are compiled.
#ifndef MARLIN_FOLD
  #define MARLIN_FOLD 1
#endif
// [0048 FOLDS] Build switch for the served-partition remainder-fold twins (default 1): generate_kernels.py's TU
// sm80_kernel_folds_<c_type>.cu instantiates MarlinFOLDS<...> (MARLIN_FOLDS_TWIN 1) under it and marlin.cu carries the host side.
// The RUN switch is VLLM_MARLIN_REMAINDER_FOLD_MODE=S|SAR|SRA with the 0042 N x K list (unset = the 0042 fold, or no fold at all).
// It needs MARLIN_FOLD (the N x K list and the fold condition live there). 0 = neither the twins nor the host side are compiled.
#ifndef MARLIN_FOLDS
  #define MARLIN_FOLDS 1
#endif
#if MARLIN_FOLDS && !MARLIN_FOLD
  #error "[0048 FOLDS] MARLIN_FOLDS needs MARLIN_FOLD"
#endif
// [0043 TAILPF] Build switch for the next-launch prefetch twins (default 1): generate_kernels.py's TUs
// sm80_kernel_tpf*_<c_type>.cu instantiate <family>TPF<...> (MARLIN_TPF_TWIN 1) under it and marlin.cu carries the host gate
// and the successor registry. The RUN switch is the env VLLM_MARLIN_TAIL_PREFETCH (unset/0 = OFF = the launch the earlier
// selections chose). 0 = neither the twins nor the gate are compiled.
#ifndef MARLIN_TAILPF
  #define MARLIN_TAILPF 1
#endif
// [0051 ZPPACK] Build switch for the zero-point-in-scale-word twins (default 1): generate_kernels.py's TUs
// sm80_kernel_zpp*_<c_type>.cu instantiate MarlinZPP / MarlinXSZPP<...> (MARLIN_ZPPACK_TWIN 1) under it and marlin.cu carries the
// host gate, the per-tensor census + load-time repack and the twin map. The RUN switch is the env VLLM_MARLIN_ZPPACK (unset/0 =
// OFF = the served launches, nothing packed). 0 = neither the twins nor the host side are compiled.
#ifndef MARLIN_ZPPACK
  #define MARLIN_ZPPACK 1
#endif
// [ZPW13 cuda-zpw13 2026-10-15] Build switch for the 13-bit ZPP group-word twins (default 1): generate_kernels.py's TUs
// sm80_kernel_zpw13*_<c_type>.cu / sm80_kernel_cpf_zpw13*_<c_type>.cu instantiate MarlinZPW13 / MarlinXSZPW13 (+ TPF / CPF forms,
// MARLIN_ZPW13_TWIN 1) under it and marlin.cu carries the L64-13 repack and the twin maps (DESIGN-ZPW13.v1). The RUN switch is the env
// VLLM_MARLIN_ZPW13 (unset/0 = OFF = the 0051 ZPPACK packs and launches, byte for byte). Needs MARLIN_ZPPACK.
#ifndef MARLIN_ZPW13
  #define MARLIN_ZPW13 1
#endif
#if MARLIN_ZPW13 && !MARLIN_ZPPACK
  #error "[ZPW13] MARLIN_ZPW13 needs MARLIN_ZPPACK"
#endif
// [0056 IRED cuda 10-10, BUG-559 Marlin site] Build switch (default 1 in an IRED build): int8-activation GROUPED instances
// (a_type kS8, group_blocks != -1) reduce split-K partials in int32 and convert to fp32 once, in the column's final slice.
// The group scales are int16 codes (marlin_act_int8_process_scales: code = round(s / s.max() * 4096), the one fp factor per tensor
// is folded into the act scale), so the whole-K sum is an integer: exact and order-free -> bitwise for any pass split, thread
// config or striping (batch-invariant). Every other instance compiles the served token stream (each IRED line is `if constexpr`).
// The host refuses unless the engine declares VLLM_MARLIN_IRED=1 (salted) and refuses int8 channelwise (group_size -1).
#ifndef MARLIN_IRED
  #define MARLIN_IRED 1
#endif
// [0061 COLFPF v2 cuda 2026-10-11] Build switch for the column-factor L2-prefetch twins (default 1): generate_kernels.py's TUs
// sm80_kernel_cpf*_<c_type>.cu instantiate the CPF twins (MARLIN_CPF_TWIN 1) under it and marlin.cu carries the host swap. The RUN
// switch is the env VLLM_MARLIN_COLFPF (unset/0 = OFF = the kernel every other selection chose, byte for byte). 0 = neither the
// twins nor the swap are compiled.
#ifndef MARLIN_COLFPF
  #define MARLIN_COLFPF 1
#endif
// [0063 KU3STPF cuda 2026-10-11] Build switch for the kU3S (0055 int3) TAILPF twins (default 1): generate_kernels.py's TUs
// sm80_kernel_tpf{,_xs}_u3s_<c_type>.cu (and, with MARLIN_COLFPF, sm80_kernel_cpftpf{,_xs}_u3s_s<site>_<c_type>.cu) instantiate them
// (MARLIN_TPF_TWIN 1 + MARLIN_TPF_KU3S 1) under it and marlin.cu carries the host map. The RUN switch is the env
// VLLM_MARLIN_TAILPF_KU3S=1 together with VLLM_MARLIN_TAIL_PREFETCH (unset/0 = OFF = the kernel every other selection chose, byte
// for byte). Needs MARLIN_TAILPF. 0 = neither the twins nor the map are compiled.
#ifndef MARLIN_KU3STPF
  #define MARLIN_KU3STPF 1
#endif
#if MARLIN_KU3STPF && !MARLIN_TAILPF
  #error "[0063 KU3STPF] MARLIN_KU3STPF needs MARLIN_TAILPF"
#endif
// [0071 G2 cuda 2026-10-12, BUG-562 guard] Build switch for the split-K lock-leftover DETECTOR (default 1): barrier_acquire
// (marlin_template.h) checks each acquire's FIRST lock read against the reading slice's index (v6: once, before the
// served spin); a read above it (impossible under the stream-order contract) is recorded per module and in a mapped host
// page, and marlin.cu prints "[marlin-G2] LOCK-LEFTOVER" once per process.
// The RUN switch is the env VLLM_MARLIN_G2 (compose: ${VLLM_MARLIN_G2:-1}; 1 = DETECT, 2 = TRAP, explicit 0 = OFF). A module's
// __constant__ marlin_g2_mode<0> stays 0 until the host arms that module at its first Marlin launch (marlin.cu kt_g2_launch), so
// OFF and every module never launched run the served spin loop after one uniform constant-bank compare per acquire, outside the
// k-loop; no module is loaded for G2. 0 = the served token stream, byte for byte.
#ifndef MARLIN_G2
  #define MARLIN_G2 1
#endif
#if MARLIN_G2
struct CUstream_st;
namespace MARLIN_NAMESPACE_NAME {
// marlin.cu: arm the module that holds `kernel` (once per module and device; nothing is enqueued on `stream`, which may be
// capturing), then tell the first hit. Called right before every Marlin launch.
void kt_g2_launch(const void* kernel, CUstream_st* stream);
}  // namespace MARLIN_NAMESPACE_NAME
#endif

#define MARLIN_KERNEL_PARAMS                                               \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                  \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                      \
      const int4 *__restrict__ b_bias_ptr,                                 \
      const float *__restrict__ a_scales_ptr,                              \
      const int4 *__restrict__ scales_ptr,                                 \
      const float *__restrict__ global_scale_ptr,                          \
      const int4 *__restrict__ zp_ptr, int prob_m, int prob_n, int prob_k, \
      int lda, int *locks, bool has_bias, bool use_atomic_add,             \
      bool use_fp32_reduce, int max_shared_mem, int tile_major

namespace MARLIN_NAMESPACE_NAME {
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
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
