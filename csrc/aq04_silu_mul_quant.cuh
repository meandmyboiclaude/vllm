// [0044 AQ04 cuda-lane 2026-10-06] SiLU*mul + per-token int8 act-quant for the W4A8 down_proj input, ONE launch.
// [0055 AQ04HQ cuda-lane 2026-10-08] RS tail: the foreign ws_o slices are loaded first, in one burst (SITE3 hoist); the
// rounding is QF1 = copysign(floor.ftz(|v| + 0.5), v) (RS and RR), equal to 0044's form on every fp32: codes unchanged.
//
// Replaces, at the 64 MLP-mid sites, the served pair
//   vllm::act_and_mul_kernel<bf16, ..., silu, act_first=1, use_vec=1, HAS_CLAMP=0, use_256b=0>  (this file's includer)
//   + the int8 quant of the bf16 result: Inductor triton_red_fused_..._reciprocal_sub_where_2/_3 (W5) or ACTQ site 3
//     (actq_ops._actq_two_launch_kernel PH=2, W6), which are EXACT to each other,
// bitwise: the int8 codes and the fp32 per-row scale are the served bits. Design + proof plan: cuda-lane
// wip/aq04/AQ04-DESIGN.md; the down Marlin is not touched.
//
// Arithmetic, per element (gate g, up u, bf16 -> fp32 exact):
//   o = bf16_rn(float(bf16_rn(g / (1 + expf(-g * alpha)))) * (u + beta))   = vllm::packed_compute + packed_silu_kernel,
//       called here, so the SiLU*mul is the served act_and_mul's own device code (same TU, same nvcc flags)
// per row:
//   amax = max.NaN |o|  (order free),  am = max.NaN(amax, 1e-10f),  r = div.full(1, am) * 127
//   v = min.NaN(max.NaN(o * r, -127), 127);  q = v >= 0 ? floor.ftz(v + 0.5) : ceil.ftz(v - 0.5);  int8 = cvt.rzi.s16 low byte
//       (0055: q computed as copysign(floor.ftz(|v| + 0.5), v): RN addition is sign-symmetric and ceil(-a) = -floor(a), so
//       the two forms agree on every fp32 v, NaN and +-0 included; proved over all 2^32 inputs)
//   scale = (am * fp32(1/127)) * input_global_scale
// i.e. the served Inductor kernel (P1-ARITH-SPEC section 2): the reciprocal is div.full (NOT the eager PN447 div.rn) and
// the scale multiplies by fp32(1/127) (NOT a division). Every quant op is an _rn intrinsic or inline PTX, so nvcc cannot
// contract or reassociate any of them.
//
// Two forms of one kernel, selected by S (CTAs per row):
//   S = 1 (RR, any M): one CTA per row, kThreads threads x kRRVecs vectors; the row stays in registers; no workspace.
//   S = 4 (RS, M <= kRSMaxRows): kRSSplit CTAs per row, one vector per thread. Each CTA stores its bf16 o slice to ws_o and
//       its partial max to ws_mx, then thread 0 counts in (fence.acq_rel.gpu + atom.acq_rel.gpu on ws_cnt[row]); the LAST
//       CTA of the row quantises the whole row (its own slice from registers, the others from ws_o through ld.relaxed.gpu,
//       never .nc), stores the scale and resets ws_cnt[row] to 0 (graph-replay safe, the ACTQ two-launch pattern). No
//       spin, no co-residency assumption (the B5 drafter side stream may share the SMs).
#pragma once

namespace vllm {
namespace aq04 {

constexpr int kD = 17408;                  // the served MLP intermediate width (Qwen3.8-27B); the host refuses any other
constexpr int kElts = 8;                   // bf16 per 16-byte vector = PackedVec<__nv_bfloat16, false>
constexpr int kNumVecs = kD / kElts;       // 2176 = act_and_mul's num_vecs at d = 17408
constexpr int kThreads = 544;              // 17 warps
constexpr int kRRVecs = kNumVecs / kThreads;       // 4 vectors per thread at S = 1
constexpr int kRSSplit = 4;                // CTAs per row at S > 1 (one vector per thread)
constexpr int kRSMaxRows = 64;             // RS workspace rows (the decode capture sizes are <= 64)
static_assert(kRRVecs * kThreads == kNumVecs, "AQ04 RR tiling");
static_assert(kThreads * kRSSplit == kNumVecs, "AQ04 RS tiling");
static_assert(kThreads % 32 == 0, "AQ04: whole warps only (full-mask shuffles)");

// ---- exact scalar helpers (inline PTX: identical instructions to the served quant's PTX) ----
__device__ __forceinline__ float aq_max_nan(float a, float b) {
  float r;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float aq_min_nan(float a, float b) {
  float r;
  asm("min.NaN.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float aq_div_full(float a, float b) {
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float aq_floor_ftz(float a) {
  float r;
  asm("cvt.rmi.ftz.f32.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ float aq_ceil_ftz(float a) {
  float r;
  asm("cvt.rpi.ftz.f32.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}
__device__ __forceinline__ uint32_t aq_i8(float v) {  // cvt.rzi.s16.f32, low byte (NaN -> 0; |v| <= 127 after the clamp)
  unsigned short s;
  asm("cvt.rzi.s16.f32 %0, %1;" : "=h"(s) : "f"(v));
  return static_cast<uint32_t>(s) & 0xffu;
}

// ---- memory-model helpers (RS only) ----
__device__ __forceinline__ void aq_fence_acq_rel_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ unsigned aq_atom_add_acq_rel_gpu(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ float aq_ld_relaxed_f32(const float* p) {
  float v;
  asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint4 aq_ld_relaxed_v4(const void* p) {
  uint4 u;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(u.x), "=r"(u.y), "=r"(u.z), "=r"(u.w)
               : "l"(p)
               : "memory");
  return u;
}
__device__ __forceinline__ void aq_st_relaxed_u32(unsigned* p, unsigned v) {
  asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

// ---- the served quant, per row and per element ----
__device__ __forceinline__ void aq_row_scalars(float amax, float g, float& r, float& sc) {
  const float am = aq_max_nan(amax, 1e-10f);                  // 0x2EDBE6FF
  r = __fmul_rn(aq_div_full(1.0f, am), 127.0f);                 // div.full.f32 then mul 0x42FE0000
  sc = __fmul_rn(__fmul_rn(am, 0.007874015718698502f), g);      // 0x3C010204 = fp32(1/127), then * input_global_scale
}
__device__ __forceinline__ uint32_t aq_quant1(float x, float r) {
  float v = __fmul_rn(x, r);
  v = aq_max_nan(v, -127.0f);
  v = aq_min_nan(v, 127.0f);
  const float t = copysignf(aq_floor_ftz(__fadd_rn(fabsf(v), 0.5f)), v);  // [0055] QF1 == the 0044 branch form, every fp32
  return aq_i8(t);
}
// 8 elements (one 16-byte bf16 vector) -> 8 int8 codes, element 0 in the lowest byte (the served byte order)
template <typename pvec_t>
__device__ __forceinline__ uint2 aq_quant_vec(const pvec_t& o, float r) {
  static_assert(pvec_t::NUM_ELTS == 4, "AQ04 expects 4 packed pairs per vector");
  uint32_t w[2] = {0u, 0u};
#pragma unroll
  for (int j = 0; j < pvec_t::NUM_ELTS; j++) {
    const float2 f = cast_to_float2(o.elts[j]);
    w[j >> 1] |= (aq_quant1(f.x, r) | (aq_quant1(f.y, r) << 8)) << ((j & 1) * 16);
  }
  return make_uint2(w[0], w[1]);
}

template <typename packed_t, packed_t (*PACKED_ACT_FN)(const packed_t&, const float), int S>
__global__ void __launch_bounds__(kThreads, 2) aq04_silu_mul_quant_kernel(
    int8_t* __restrict__ q,                    // [M, kD] int8 codes (the Marlin A operand)
    float* __restrict__ scale,                 // [M] fp32 per-row A scale, input_global_scale folded in
    const __nv_bfloat16* __restrict__ input,   // [M, 2 * kD] = [gate | up] (the gate_up GEMM output)
    const float* __restrict__ gscale,          // [1] the down_proj input_global_scale
    __nv_bfloat16* __restrict__ ws_o,          // S > 1: [kRSMaxRows, kD]
    float* __restrict__ ws_mx,                 // S > 1: [kRSMaxRows, S]
    unsigned* __restrict__ ws_cnt,             // S > 1: [kRSMaxRows], 0 at rest
    const float limit, const float alpha, const float beta) {
  using pvec_t = PackedVec<__nv_bfloat16, false>;
  constexpr int VPT = kNumVecs / (S * kThreads);
  static_assert(VPT * S * kThreads == kNumVecs, "AQ04 tiling");
  const int64_t row = blockIdx.x;
  const int sp = (S > 1) ? static_cast<int>(blockIdx.y) : 0;
  const pvec_t* x_vec = reinterpret_cast<const pvec_t*>(input + row * 2 * kD);   // gate half
  const pvec_t* y_vec = reinterpret_cast<const pvec_t*>(input + row * 2 * kD + kD);  // up half

  // phase 1: o = the served SiLU*mul (two bf16 roundings), kept in registers; running max.NaN of |o|
  pvec_t o[VPT];
  float mx = __int_as_float(0xff800000);  // -inf, the served init
#pragma unroll
  for (int v = 0; v < VPT; v++) {
    const int i = (sp * VPT + v) * kThreads + static_cast<int>(threadIdx.x);
    pvec_t x, y;
    ld128(x, &x_vec[i]);
    ld128(y, &y_vec[i]);
#pragma unroll
    for (int j = 0; j < pvec_t::NUM_ELTS; j++) {
      o[v].elts[j] = packed_compute<packed_t, PACKED_ACT_FN, true, false>(x.elts[j], y.elts[j], limit, alpha, beta);
      const float2 f = cast_to_float2(o[v].elts[j]);
      mx = aq_max_nan(mx, aq_max_nan(fabsf(f.x), fabsf(f.y)));
    }
    if constexpr (S > 1) st128(o[v], reinterpret_cast<pvec_t*>(ws_o + row * kD) + i);
  }

  // CTA max (max.NaN is exact in any order)
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) mx = aq_max_nan(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  __shared__ float red[kThreads / 32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  float amax = red[0];
#pragma unroll
  for (int w = 1; w < kThreads / 32; w++) amax = aq_max_nan(amax, red[w]);

  int8_t* q_row = q + row * kD;
  if constexpr (S == 1) {
    float r, sc;
    aq_row_scalars(amax, __ldg(gscale), r, sc);
#pragma unroll
    for (int v = 0; v < VPT; v++) {
      const int i = v * kThreads + static_cast<int>(threadIdx.x);
      *reinterpret_cast<uint2*>(q_row + i * kElts) = aq_quant_vec(o[v], r);
    }
    if (threadIdx.x == 0) scale[row] = sc;
    return;
  } else {
    // phase 2 (RS): publish the partial max; the last CTA of the row quantises the row
    __shared__ int last_cta;
    if (threadIdx.x == 0) ws_mx[row * S + sp] = amax;
    __syncthreads();  // every thread's ws_o store and the ws_mx store precede thread 0's release below
    if (threadIdx.x == 0) {
      aq_fence_acq_rel_gpu();
      last_cta = (aq_atom_add_acq_rel_gpu(&ws_cnt[row], 1u) == static_cast<unsigned>(S - 1));
    }
    __syncthreads();
    if (!last_cta) return;
    // [0055 AQ04HQ] SITE3 hoist: every foreign slice's ws_o load first (one burst; 0044 ran load -> quant -> store per slice,
    // serialised by the asm memory clobbers), then the ws_mx burst, then the quantisation. sp is CTA-uniform; fw[sp] is unused.
    const pvec_t* o_ws = reinterpret_cast<const pvec_t*>(ws_o + row * kD);
    uint4 fw[S][VPT];
#pragma unroll
    for (int s = 0; s < S; s++) {
      if (s == sp) continue;
#pragma unroll
      for (int v = 0; v < VPT; v++) fw[s][v] = aq_ld_relaxed_v4(o_ws + (s * VPT + v) * kThreads + static_cast<int>(threadIdx.x));
    }
    float am_raw = aq_ld_relaxed_f32(&ws_mx[row * S]);
#pragma unroll
    for (int s = 1; s < S; s++) am_raw = aq_max_nan(am_raw, aq_ld_relaxed_f32(&ws_mx[row * S + s]));
    float r, sc;
    aq_row_scalars(am_raw, __ldg(gscale), r, sc);
#pragma unroll
    for (int s = 0; s < S; s++) {
#pragma unroll
      for (int v = 0; v < VPT; v++) {
        const int i = (s * VPT + v) * kThreads + static_cast<int>(threadIdx.x);
        uint2 c;
        if (s == sp) {
          c = aq_quant_vec(o[v], r);
        } else {
          c = aq_quant_vec(*reinterpret_cast<const pvec_t*>(&fw[s][v]), r);
        }
        *reinterpret_cast<uint2*>(q_row + i * kElts) = c;
      }
    }
    if (threadIdx.x == 0) {
      scale[row] = sc;
      aq_st_relaxed_u32(&ws_cnt[row], 0u);  // all S arrivals are in; the next launch is stream-ordered after this one
    }
  }
}

}  // namespace aq04
}  // namespace vllm
