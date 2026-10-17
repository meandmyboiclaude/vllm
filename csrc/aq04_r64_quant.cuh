// [0056 AQ04R64 cuda-lane 2026-10-08] arm B, site DN: SiLU*mul + block-64 FWHT (x * D * H64, the build basis of the int3 down_proj
// codes) + per-token int8 act-quant (QF1), ONE launch, the RR form at EVERY M (RID 737: H64RR = the cheapest rotated form at M64,
// +0.0592 ms/step v RSHQ; RRQ == RSHQ). Ported VERBATIM from the RID 731/737 KAT library (fwhtfuse/kat/v4/src/aq04h_rot64_quant.cuh,
// aq04h 11cb1f2c4e4834b5: dm64, quant1_qf1, quant8_qf1, rot64_half, unpack_half, cta_max and the aq04h_rr_kernel ROT 1 body, NEG 0 only),
// which 731 + 737 proved bitwise == kv fwht_ref 9b33ea8b rotate(x, D, 64) + the served AQ04 quant on every M 1..128.
// Reference order: scale-first x * (D * 2^-3) (exact: a power of two), butterflies h = 1..32 increasing, (a+b, a-b) at the lower index.
// D = the packed sign bits of fwht_ref sign_vector(17408) (bit e of byte i set <=> D[8i+e] = -1), built and sha-pinned by the host.
// Dark: only the GENESIS_ARMB_FWHT64_DN model path calls it; no other kernel of this TU changes.
#pragma once

namespace vllm {
namespace aq04r64 {

using aq04::kD;
using aq04::kElts;
using aq04::kNumVecs;
using aq04::kThreads;
using pvec_t = PackedVec<__nv_bfloat16, false>;

static_assert(kNumVecs == 4 * kThreads, "AQ04R64: 4 contiguous vectors (one 32-element half block) per thread");
static_assert(kD % 64 == 0, "AQ04R64: d is a multiple of the Hadamard block");

// D * 2^-3 for element e of a sign word (bit e set <=> D = -1). Exact power of two: x * dm is exact for every bf16 x.
__device__ __forceinline__ float dm64(uint32_t sb, int e) {
  return __uint_as_float(0x3E000000u | (((sb >> e) & 1u) << 31));
}

// QF1 rounding: q = copysign(floor.ftz(|v| + 0.5), v), the clamp and cvt as AQ04 (aq_quant1); == AQ04's rounding on every fp32.
__device__ __forceinline__ uint32_t quant1_qf1(float x, float r) {
  float v = __fmul_rn(x, r);
  v = aq04::aq_max_nan(v, -127.0f);
  v = aq04::aq_min_nan(v, 127.0f);
  const float t = copysignf(aq04::aq_floor_ftz(__fadd_rn(fabsf(v), 0.5f)), v);
  return aq04::aq_i8(t);
}
// 8 fp32 values -> 8 int8 codes, element 0 in the lowest byte (the served byte order)
__device__ __forceinline__ uint2 quant8_qf1(const float (&z)[8], float r) {
  uint32_t w0 = 0u, w1 = 0u;
#pragma unroll
  for (int e = 0; e < 4; e++) w0 |= quant1_qf1(z[e], r) << (8 * e);
#pragma unroll
  for (int e = 0; e < 4; e++) w1 |= quant1_qf1(z[4 + e], r) << (8 * e);
  return make_uint2(w0, w1);
}

// a 32-element half block in registers (h = 1..16) + h = 32 across lane bit 0 (lower lane a + b, upper lane a - b, one rounding)
__device__ __forceinline__ void rot64_half(float (&v)[32], int lane) {
#pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 32; i++) {
      if (!(i & h)) {
        const float a = v[i], b = v[i + h];
        v[i] = __fadd_rn(a, b);
        v[i + h] = __fsub_rn(a, b);
      }
    }
  }
  const float sg = (lane & 1) ? -1.0f : 1.0f;
#pragma unroll
  for (int e = 0; e < 32; e++) {
    const float p = __shfl_xor_sync(0xffffffffu, v[e], 1);
    v[e] = __fmaf_rn(sg, v[e], p);
  }
}
// 4 packed bf16 vectors (uint4 words) + the 32 sign bits of the half block -> 32 scaled fp32 values (element 8k + e <-> bit 8k + e)
__device__ __forceinline__ void unpack_half(const uint4 (&w)[4], uint32_t sw, float (&v)[32]) {
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const pvec_t& o = *reinterpret_cast<const pvec_t*>(&w[k]);
#pragma unroll
    for (int j = 0; j < pvec_t::NUM_ELTS; j++) {
      const float2 f = cast_to_float2(o.elts[j]);
      v[8 * k + 2 * j] = __fmul_rn(f.x, dm64(sw, 8 * k + 2 * j));
      v[8 * k + 2 * j + 1] = __fmul_rn(f.y, dm64(sw, 8 * k + 2 * j + 1));
    }
  }
}
__device__ __forceinline__ float cta_max(float mx) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) mx = aq04::aq_max_nan(mx, __shfl_xor_sync(0xffffffffu, mx, off));
  __shared__ float red[kThreads / 32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  float amax = red[0];
#pragma unroll
  for (int w = 1; w < kThreads / 32; w++) amax = aq04::aq_max_nan(amax, red[w]);
  return amax;
}

// H64RR (any M): one CTA per row; thread t = half block t (elements 32t .. 32t+31 = vectors 4t .. 4t+3); the row stays in registers;
// no workspace. Body == aq04h_rr_kernel<..., NEG 0, ROT 1> of the KAT library.
template <typename packed_t, packed_t (*PACKED_ACT_FN)(const packed_t&, const float)>
__global__ void __launch_bounds__(kThreads, 2) aq04r64_rr_kernel(
    int8_t* __restrict__ q, float* __restrict__ scale, const __nv_bfloat16* __restrict__ input, const float* __restrict__ gscale,
    const uint8_t* __restrict__ sgn, const float limit, const float alpha, const float beta) {
  const int64_t row = blockIdx.x;
  const int t = static_cast<int>(threadIdx.x);
  const pvec_t* x_vec = reinterpret_cast<const pvec_t*>(input + row * 2 * kD);
  const pvec_t* y_vec = reinterpret_cast<const pvec_t*>(input + row * 2 * kD + kD);
  const int lane = t & 31;
  const uint32_t sw = __ldg(reinterpret_cast<const uint32_t*>(sgn) + t);   // sign bytes 4t .. 4t+3
  uint4 w[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    pvec_t x, y, o;
    ld128(x, &x_vec[4 * t + k]);
    ld128(y, &y_vec[4 * t + k]);
#pragma unroll
    for (int j = 0; j < pvec_t::NUM_ELTS; j++)
      o.elts[j] = packed_compute<packed_t, PACKED_ACT_FN, true, false>(x.elts[j], y.elts[j], limit, alpha, beta);
    w[k] = *reinterpret_cast<const uint4*>(&o);
  }
  float z[32];
  unpack_half(w, sw, z);
  rot64_half(z, lane);
  float mx = __int_as_float(0xff800000);
#pragma unroll
  for (int e = 0; e < 32; e++) mx = aq04::aq_max_nan(mx, fabsf(z[e]));
  const float amax = cta_max(mx);
  float r, sc;
  aq04::aq_row_scalars(amax, __ldg(gscale), r, sc);
  int8_t* q_row = q + row * kD;
  uint32_t c[8];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    float a[8];
#pragma unroll
    for (int e = 0; e < 8; e++) a[e] = z[8 * k + e];
    const uint2 u = quant8_qf1(a, r);
    c[2 * k] = u.x;
    c[2 * k + 1] = u.y;
  }
  uint4* dst = reinterpret_cast<uint4*>(q_row + 32 * t);
  dst[0] = make_uint4(c[0], c[1], c[2], c[3]);
  dst[1] = make_uint4(c[4], c[5], c[6], c[7]);
  if (threadIdx.x == 0) scale[row] = sc;
}

}  // namespace aq04r64
}  // namespace vllm
