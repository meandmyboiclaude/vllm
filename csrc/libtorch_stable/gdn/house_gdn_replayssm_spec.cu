/*
 * house_gdn_replayssm_spec.cu  (cuda-lane, 2026-09-12)
 *
 * ONE CUDA launch replacing the served Triton pair `gdn_replayssm_spec_circular_kernel` (verify + flush,
 * vllm/third_party/flash_linear_attention/ops/gdn_replayssm_spec_decode.py) for Qwen3.8's GDN spec-decode:
 * cached-history closed form over an int8 checkpoint (per-(row, 32-col group) fp32 scale) and a circular
 * d/k/g ring.  Same tensors, same side effects (out, d/k/g spec rows, checkpoint + scale on flush rows),
 * device-side per-row routing on is_flush[slot].  Math and masks follow the Triton kernel op for op
 * (see ~/shared/cuda-lane-20260912/gdn-cuda/CONTRACT.md); the tensor-core products round their operands to
 * tf32 exactly like Triton's input_precision="tf32", accumulation order differs (fp32 roundoff class).
 *
 * Geometry fixed at compile time (the record): K = V = 128, spec tile BS = 16, history tile BC = 16, v-tile 64,
 * scale group 32.  Grid (V/64, B, HV), 128 threads, ~46 KB static shared memory (2 CTAs/SM on sm_89).
 * "Less = more": no software pipelining, tiles staged once, tensor cores only for the three [.,128]x[128,32]
 * products, everything else plain fp32 FMA on shared memory.
 */
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace house_gdn {

constexpr int K = 128, V = 128, BV = 64, BS = 16, BC = 16, SG = 32, NG = K / SG;
constexpr int THREADS = 128, QKR = 2 * BS;  // qk rows: 0..15 = q, 16..31 = k
constexpr int KP = K + 4, HP = K + 16, DP = BC + 1;  // padded rows: fp32 +4, int8 +16 B (16-B aligned), dsc +1 (bank spread)
constexpr float SOFTPLUS_THRESHOLD = 20.f;

struct Strides {
  int64_t mqkv_t, a_t, b_t, o_t, state_slot, scale_slot, d_slot, k_slot, g_slot;
};

struct Smem {
  union {
    float qk[QKR][KP];  // phase 0/1: normalised q | k rows (fp32)
    struct {            // phase 2: dead qk region reused
      float R[BV][BS], D[BV][BS], A[BS][BS], T[BS][BS], F[BS][BS];
    } s2;
  } u1;
  union {
    float khist[BC][KP];  // phase 0/1
    float hw[BV][QKR];    // phase 2: hw_q (cols 0..15) | hw_k (cols 16..31)
  } u2;
  int8_t h0[BV][HP];
  float h0s[BV][NG];
  float dsc[BV][DP];   // d_cache history rows * replay decay   [v][c]
  float vin[BV][BS];   // v input                                [v][s]
  float p2[BS][QKR];   // k . qk^T : kq (cols 0..15) | kk (cols 16..31)
  float p3[BC][QKR];   // khist . qk^T : scores_q | scores_k
  float g[BS], beta[BS], G[BS], expG[BS], rdec[BC], total_decay;
};
static_assert(sizeof(Smem) <= 48 * 1024, "static shared memory budget");

__device__ __forceinline__ uint32_t tf32(float x) {
  uint32_t u;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u) : "f"(x));
  return u;
}
__device__ __forceinline__ void mma_tf32(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bf2f(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
// inclusive scan over the first 16 lanes (lanes >= 16 carry zeros)
__device__ __forceinline__ float scan16(float v, int lane) {
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) {
    const float t = __shfl_up_sync(0xffffffffu, v, o);
    if (lane >= o) v += t;
  }
  return v;
}

// C[16][32] = A[16][128] . qk^T with A rows given by a loader; fragments stay in c[nt][4] (mma C layout)
template <typename LoadA>
__device__ __forceinline__ void product_16x32(const Smem& sm, LoadA loadA, float (&c)[4][4], int lane) {
  const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
#pragma unroll 4
  for (int kk = 0; kk < K; kk += 8) {
    const uint32_t a0 = tf32(loadA(gid, kk + tig)), a1 = tf32(loadA(gid + 8, kk + tig)),
                   a2 = tf32(loadA(gid, kk + tig + 4)), a3 = tf32(loadA(gid + 8, kk + tig + 4));
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int n = nt * 8 + gid;
      mma_tf32(c[nt], a0, a1, a2, a3, tf32(sm.u1.qk[n][kk + tig]), tf32(sm.u1.qk[n][kk + tig + 4]));
    }
  }
}
__device__ __forceinline__ void store_frag(float (&c)[4][4], float (*dst)[QKR], int lane) {
  const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    dst[gid][nt * 8 + 2 * tig] = c[nt][0];
    dst[gid][nt * 8 + 2 * tig + 1] = c[nt][1];
    dst[gid + 8][nt * 8 + 2 * tig] = c[nt][2];
    dst[gid + 8][nt * 8 + 2 * tig + 1] = c[nt][3];
  }
}

__global__ void __launch_bounds__(THREADS, 2) replayssm_spec_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ out, int8_t* __restrict__ ckpt, float* __restrict__ ckpt_scale,
    __nv_bfloat16* __restrict__ d_cache, __nv_bfloat16* __restrict__ k_cache, float* __restrict__ g_cache,
    const int* __restrict__ query_start_loc, const int* __restrict__ ssm_state_indices,
    const int* __restrict__ write_pos, const int* __restrict__ cache_base, const int8_t* __restrict__ is_flush,
    float scale, int H, int HV, int buf, int null_block_id, Strides st) {
  __shared__ Smem sm;
  const int iv = blockIdx.x, n = blockIdx.y, hv = blockIdx.z;
  const int HVpH = HV / H, ih = hv / HVpH;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int bos = query_start_loc[n], eos = query_start_loc[n + 1];
  const int S = min(eos - bos, BS);
  const int slot = ssm_state_indices[n];
  if (slot <= null_block_id) {  // verify launch zero-fills; flush launch returns -> one zero-fill here
    for (int i = tid; i < S * BV; i += THREADS)
      out[(int64_t)(bos + i / BV) * st.o_t + hv * V + iv * BV + (i % BV)] = __float2bfloat16(0.f);
    return;
  }
  const bool flush = is_flush[slot] != 0;
  const int wp = write_pos[slot], C = min(wp, BC), base = cache_base[slot], bm = buf - 1;
  const bool write_k = (iv == 0) && (hv == ih * HVpH);
  const int8_t* h0g = ckpt + (int64_t)slot * st.state_slot + (int64_t)hv * V * K + (int64_t)iv * BV * K;
  const float* h0sg = ckpt_scale + (int64_t)slot * st.scale_slot + (int64_t)hv * V * NG + (int64_t)iv * BV * NG;
  const __nv_bfloat16* dg = d_cache + (int64_t)slot * st.d_slot + (int64_t)hv * buf * V;
  const __nv_bfloat16* kg = k_cache + (int64_t)slot * st.k_slot + (int64_t)ih * buf * K;
  float* gg = g_cache + (int64_t)slot * st.g_slot + (int64_t)hv * buf;

  // ---- phase 0a: gates, cumsums, history decay (warp 0, lanes = spec position / history position) ----
  if (warp == 0) {
    const int s = lane & 15;
    float gs = 0.f, bs = 0.f;
    if (lane < 16 && s < S) {
      const float x = bf2f(a[(int64_t)(bos + s) * st.a_t + hv]) + dt_bias[hv];
      const float sp = x <= SOFTPLUS_THRESHOLD ? logf(1.f + expf(x)) : x;
      gs = -expf(A_log[hv]) * sp;
      bs = 1.f / (1.f + expf(-bf2f(b[(int64_t)(bos + s) * st.b_t + hv])));
    }
    const float Gs = scan16(lane < 16 ? gs : 0.f, lane);
    float ga = 0.f;
    if (lane < 16 && lane < C) ga = gg[(base + lane) & bm];
    const float pre = scan16(lane < 16 ? ga : 0.f, lane);
    const float total = __shfl_sync(0xffffffffu, pre, 15);
    if (lane < 16) {
      sm.g[s] = gs; sm.beta[s] = bs; sm.G[s] = Gs; sm.expG[s] = expf(Gs);
      sm.rdec[s] = s < C ? expf(total - pre) : 0.f;
      if (lane == 0) sm.total_decay = expf(total);
    }
  }
  // ---- phase 0b: q/k rows -> l2norm -> qk[32][K]; warp w owns spec rows 4w..4w+3, lane owns 4 columns ----
  {
    const int kb = lane * 4;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int s = warp * 4 + r;
      float q4[4] = {}, k4[4] = {};
      if (s < S) {
        const __nv_bfloat16* row = mixed_qkv + (int64_t)(bos + s) * st.mqkv_t;
        const __nv_bfloat16* qp = row + ih * K + kb;
        const __nv_bfloat16* kp = row + H * K + ih * K + kb;
#pragma unroll
        for (int i = 0; i < 4; ++i) { q4[i] = bf2f(qp[i]); k4[i] = bf2f(kp[i]); }
      }
      float qs = warp_sum(q4[0] * q4[0] + q4[1] * q4[1] + q4[2] * q4[2] + q4[3] * q4[3]);
      float ks = warp_sum(k4[0] * k4[0] + k4[1] * k4[1] + k4[2] * k4[2] + k4[3] * k4[3]);
      const float qrn = (s < S ? 1.f / sqrtf(qs + 1e-6f) : 0.f) * scale;  // Triton: (q_rnorm * scale) then row multiply
      const float krn = s < S ? 1.f / sqrtf(ks + 1e-6f) : 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { sm.u1.qk[s][kb + i] = q4[i] * qrn; sm.u1.qk[BS + s][kb + i] = k4[i] * krn; }
      if (write_k && s < S && wp + s < buf) {  // spec key row -> ring (bf16 of the normalised fp32 row)
        __nv_bfloat16* kd = k_cache + (int64_t)slot * st.k_slot + ((int64_t)ih * buf + ((base + wp + s) & bm)) * K + kb;
#pragma unroll
        for (int i = 0; i < 4; ++i) kd[i] = __float2bfloat16(k4[i] * krn);
      }
    }
  }
  // ---- phase 0c: history keys, checkpoint tile, v tile (independent of 0a) ----
  for (int i = tid; i < BC * K; i += THREADS) {
    const int c = i / K, k = i % K;
    sm.u2.khist[c][k] = c < C ? bf2f(kg[((int64_t)((base + c) & bm)) * K + k]) : 0.f;
  }
  for (int i = tid; i < BV * K / 16; i += THREADS)  // 16-byte int8 copies into the padded rows
    *reinterpret_cast<int4*>(&sm.h0[i >> 3][(i & 7) * 16]) = reinterpret_cast<const int4*>(h0g)[i];
  for (int i = tid; i < BV * NG; i += THREADS) (&sm.h0s[0][0])[i] = h0sg[i];
  for (int i = tid; i < BV * BS; i += THREADS) {
    const int v = i / BS, s = i % BS;
    sm.vin[v][s] = s < S ? bf2f(mixed_qkv[(int64_t)(bos + s) * st.mqkv_t + 2 * H * K + hv * V + iv * BV + v]) : 0.f;
  }
  __syncthreads();
  // ---- phase 0d: d history rows scaled by the replay decay (needs rdec) ----
  for (int i = tid; i < BV * BC; i += THREADS) {
    const int v = i / BC, c = i % BC;
    sm.dsc[v][c] = c < C ? bf2f(dg[((int64_t)((base + c) & bm)) * V + iv * BV + v]) * sm.rdec[c] : 0.f;
  }
  __syncthreads();
  // ---- phase 1: tensor-core products. P1 = h0 . qk^T (all warps, 16 rows each) -> hw; P2 = k . qk^T; P3 = khist . qk^T ----
  {
    const int r0 = warp * 16;
    float c1[4][4], cs[4][4];
    product_16x32(sm, [&](int r, int k) { return (float)sm.h0[r0 + r][k] * sm.h0s[r0 + r][k >> 5]; }, c1, lane);
    if (warp == 0) { product_16x32(sm, [&](int r, int k) { return sm.u1.qk[BS + r][k]; }, cs, lane); store_frag(cs, sm.p2, lane); }
    if (warp == 1) { product_16x32(sm, [&](int r, int k) { return sm.u2.khist[r][k]; }, cs, lane); store_frag(cs, sm.p3, lane); }
    __syncthreads();
    const float td = sm.total_decay;
    if (flush) {  // checkpoint <- requant(total_decay * h0 + dsc . khist), thread owns (row, 2 groups)
      const int v = tid >> 1, g0 = (tid & 1) * 2;
      float sv[2 * SG];
#pragma unroll
      for (int gi = 0; gi < 2; ++gi) {
        float amax = 0.f;
#pragma unroll
        for (int x = 0; x < SG; ++x) {
          const int k = (g0 + gi) * SG + x;
          float s = td * (float)sm.h0[v][k] * sm.h0s[v][g0 + gi];
#pragma unroll
          for (int c = 0; c < BC; ++c) s = fmaf(sm.dsc[v][c], sm.u2.khist[c][k], s);
          sv[gi * SG + x] = s; amax = fmaxf(amax, fabsf(s));
        }
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        ckpt_scale[(h0sg - ckpt_scale) + v * NG + g0 + gi] = sc;
        int8_t* qd = ckpt + (h0g - ckpt) + v * K + (g0 + gi) * SG;
#pragma unroll
        for (int x = 0; x < SG; ++x) {
          const float y = sv[gi * SG + x] / sc;
          float q = y >= 0.f ? floorf(y + 0.5f) : -floorf(-y + 0.5f);
          q = fminf(fmaxf(q, -127.f), 127.f);
          qd[x] = (int8_t)q;
        }
      }
    }
    // hw = total_decay * P1 + dsc . P3  (verify order; the flush's materialised form is the same sum), in the C layout
    {
      const int gid = lane >> 2, tig = lane & 3;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int v = r0 + gid + (e >> 1) * 8, j = nt * 8 + 2 * tig + (e & 1);
          float acc = td * c1[nt][e];
#pragma unroll
          for (int c = 0; c < BC; ++c) acc = fmaf(sm.dsc[v][c], sm.p3[c][j], acc);
          c1[nt][e] = acc;
        }
    }
    __syncthreads();  // khist/qk dead from here: u1/u2 become the phase-2 buffers
    store_frag(c1, &sm.u2.hw[r0], lane);
  }
  // ---- phase 2: A, T = (I + A)^-1, R, D = R T^T, F, O ----
  for (int i = tid; i < BS * BS; i += THREADS) {
    const int r = i / BS, c = i % BS;
    const bool in = r < S && c < S;
    sm.u1.s2.A[r][c] = (in && r > c) ? sm.beta[r] * expf(sm.G[r] - sm.G[c]) * sm.p2[r][BS + c] : 0.f;
    sm.u1.s2.F[r][c] = (in && r <= c) ? expf(sm.G[c] - sm.G[r]) * sm.p2[r][c] : 0.f;
  }
  __syncthreads();
  if (warp == 0 && lane < BS) {  // column j of T by forward substitution: (I + A) T = I, A strictly lower
    const int j = lane;
    for (int i = 0; i < BS; ++i) {
      float t = (i == j) ? 1.f : 0.f;
      for (int m = 0; m < i; ++m) t = fmaf(-sm.u1.s2.A[i][m], sm.u1.s2.T[m][j], t);
      sm.u1.s2.T[i][j] = t;
    }
  }
  for (int i = tid; i < BV * BS; i += THREADS) {
    const int v = i / BS, j = i % BS;
    sm.u1.s2.R[v][j] = sm.beta[j] * (sm.vin[v][j] - sm.expG[j] * sm.u2.hw[v][BS + j]);
  }
  __syncthreads();
  for (int i = tid; i < BV * BS; i += THREADS) {
    const int v = i / BS, j = i % BS;
    float d = 0.f;
#pragma unroll
    for (int m = 0; m < BS; ++m) d = fmaf(sm.u1.s2.R[v][m], sm.u1.s2.T[j][m], d);
    sm.u1.s2.D[v][j] = d;
  }
  __syncthreads();
  for (int i = tid; i < BV * BS; i += THREADS) {
    const int v = i / BS, j = i % BS;
    if (j >= S) continue;
    float o = sm.expG[j] * sm.u2.hw[v][j];
#pragma unroll
    for (int m = 0; m < BS; ++m) o = fmaf(sm.u1.s2.D[v][m], sm.u1.s2.F[m][j], o);
    out[(int64_t)(bos + j) * st.o_t + hv * V + iv * BV + v] = __float2bfloat16(o);
    if (wp + j < buf)
      d_cache[(int64_t)slot * st.d_slot + ((int64_t)hv * buf + ((base + wp + j) & bm)) * V + iv * BV + v] =
          __float2bfloat16(sm.u1.s2.D[v][j]);
  }
  if (iv == 0 && tid < S && wp + tid < buf) gg[(base + wp + tid) & bm] = sm.g[tid];
}

void launch(const __nv_bfloat16* mixed_qkv, const __nv_bfloat16* a, const __nv_bfloat16* b, const float* A_log,
            const float* dt_bias, __nv_bfloat16* out, int8_t* ckpt, float* ckpt_scale, __nv_bfloat16* d_cache,
            __nv_bfloat16* k_cache, float* g_cache, const int* query_start_loc, const int* ssm_state_indices,
            const int* write_pos, const int* cache_base, const int8_t* is_flush, float scale, int B, int H, int HV,
            int buf, int null_block_id, Strides st, cudaStream_t stream) {
  const dim3 grid(V / BV, B, HV);
  replayssm_spec_kernel<<<grid, THREADS, 0, stream>>>(mixed_qkv, a, b, A_log, dt_bias, out, ckpt, ckpt_scale, d_cache,
                                                       k_cache, g_cache, query_start_loc, ssm_state_indices, write_pos,
                                                       cache_base, is_flush, scale, H, HV, buf, null_block_id, st);
}

}  // namespace house_gdn

#ifndef HOUSE_GDN_NO_TORCH
#include "../torch_utils.h"
#include "../../cuda_compat.h"

void house_gdn_replayssm_spec(torch::stable::Tensor const& mixed_qkv, torch::stable::Tensor const& a,
                              torch::stable::Tensor const& b, torch::stable::Tensor const& A_log,
                              torch::stable::Tensor const& dt_bias, torch::stable::Tensor& ckpt,
                              torch::stable::Tensor& ckpt_scale, torch::stable::Tensor& d_cache,
                              torch::stable::Tensor& k_cache, torch::stable::Tensor& g_cache,
                              torch::stable::Tensor& out, torch::stable::Tensor const& query_start_loc,
                              torch::stable::Tensor const& ssm_state_indices, torch::stable::Tensor const& write_pos,
                              torch::stable::Tensor const& cache_base, torch::stable::Tensor const& is_flush,
                              int64_t null_block_id, double scale) {
  using torch::headeronly::ScalarType;
  using namespace house_gdn;
  auto is = [](torch::stable::Tensor const& t, ScalarType s) { return t.is_cuda() && t.scalar_type() == s; };
  STD_TORCH_CHECK(is(mixed_qkv, ScalarType::BFloat16) && is(a, ScalarType::BFloat16) && is(b, ScalarType::BFloat16) &&
                      is(out, ScalarType::BFloat16) && is(d_cache, ScalarType::BFloat16) && is(k_cache, ScalarType::BFloat16),
                  "house_gdn: mixed_qkv/a/b/out/d_cache/k_cache must be CUDA bf16");
  STD_TORCH_CHECK(is(A_log, ScalarType::Float) && is(dt_bias, ScalarType::Float) && is(g_cache, ScalarType::Float) &&
                      is(ckpt_scale, ScalarType::Float),
                  "house_gdn: A_log/dt_bias/g_cache/ckpt_scale must be CUDA fp32");
  STD_TORCH_CHECK(is(ckpt, ScalarType::Char) && is(is_flush, ScalarType::Char), "house_gdn: ckpt/is_flush must be CUDA int8");
  STD_TORCH_CHECK(is(query_start_loc, ScalarType::Int) && is(ssm_state_indices, ScalarType::Int) &&
                      is(write_pos, ScalarType::Int) && is(cache_base, ScalarType::Int),
                  "house_gdn: index/cursor tensors must be CUDA int32");
  STD_TORCH_CHECK(ckpt.dim() == 4 && ckpt.size(2) == V && ckpt.size(3) == K && ckpt.stride(3) == 1 && ckpt.stride(2) == K &&
                      ckpt.stride(1) == (int64_t)V * K,
                  "house_gdn: ckpt must be [slots, HV, 128, 128] with contiguous slot contents");
  const int HV = (int)ckpt.size(1);
  STD_TORCH_CHECK(ckpt_scale.dim() == 4 && ckpt_scale.size(1) == HV && ckpt_scale.size(2) == V && ckpt_scale.size(3) == NG &&
                      ckpt_scale.stride(3) == 1 && ckpt_scale.stride(2) == NG && ckpt_scale.stride(1) == (int64_t)V * NG,
                  "house_gdn: ckpt_scale must be [slots, HV, 128, 4] contiguous per slot");
  STD_TORCH_CHECK(d_cache.dim() == 4 && d_cache.size(1) == HV && d_cache.size(3) == V && d_cache.stride(3) == 1 &&
                      d_cache.stride(2) == V && k_cache.dim() == 4 && k_cache.size(3) == K && k_cache.stride(3) == 1 &&
                      k_cache.stride(2) == K && g_cache.dim() == 3 && g_cache.stride(2) == 1,
                  "house_gdn: d/k/g caches must be [slots, heads, buf, dim] with contiguous rows");
  const int buf = (int)d_cache.size(2);
  STD_TORCH_CHECK((buf & (buf - 1)) == 0 && k_cache.size(2) == buf && g_cache.size(2) == buf &&
                      d_cache.stride(1) == (int64_t)buf * V && k_cache.stride(1) == (int64_t)buf * K && g_cache.stride(1) == buf,
                  "house_gdn: ring buffer must be a power of two shared by d/k/g");
  const int H = (int)k_cache.size(1);
  STD_TORCH_CHECK(H > 0 && HV % H == 0, "house_gdn: HV must be a multiple of H");
  STD_TORCH_CHECK(mixed_qkv.dim() == 2 && mixed_qkv.stride(1) == 1 && mixed_qkv.size(1) == (int64_t)2 * H * K + (int64_t)HV * V,
                  "house_gdn: mixed_qkv must be [T, 2*H*128 + HV*128] channel-contiguous");
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.stride(1) == 1 && b.stride(1) == 1 && a.size(1) == HV && b.size(1) == HV,
                  "house_gdn: a/b must be [T, HV]");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(1) == HV && out.size(2) == V && out.stride(2) == 1 && out.stride(1) == V,
                  "house_gdn: out must be [T, HV, 128] with contiguous [HV, 128]");
  STD_TORCH_CHECK(A_log.is_contiguous() && dt_bias.is_contiguous() && A_log.numel() == HV && dt_bias.numel() == HV,
                  "house_gdn: A_log/dt_bias must be contiguous [HV]");
  STD_TORCH_CHECK(query_start_loc.is_contiguous() && ssm_state_indices.is_contiguous() && write_pos.is_contiguous() &&
                      cache_base.is_contiguous() && is_flush.is_contiguous() && query_start_loc.dim() == 1 &&
                      ssm_state_indices.dim() == 1 && query_start_loc.numel() == ssm_state_indices.numel() + 1,
                  "house_gdn: cursors/indices must be contiguous 1-D, query_start_loc = [B+1]");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(ckpt.data_ptr()) % 16 == 0 && ckpt.stride(0) % 16 == 0,
                  "house_gdn: ckpt slots must keep 16-byte alignment");
  const int B = (int)ssm_state_indices.numel();
  if (B == 0) return;
  torch::stable::accelerator::DeviceGuard const guard(mixed_qkv.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(mixed_qkv.get_device_index());
  Strides st{mixed_qkv.stride(0), a.stride(0), b.stride(0), out.stride(0), ckpt.stride(0), ckpt_scale.stride(0),
             d_cache.stride(0), k_cache.stride(0), g_cache.stride(0)};
  launch(static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()), static_cast<const __nv_bfloat16*>(a.data_ptr()),
         static_cast<const __nv_bfloat16*>(b.data_ptr()), static_cast<const float*>(A_log.data_ptr()),
         static_cast<const float*>(dt_bias.data_ptr()), static_cast<__nv_bfloat16*>(out.data_ptr()),
         static_cast<int8_t*>(ckpt.data_ptr()), static_cast<float*>(ckpt_scale.data_ptr()),
         static_cast<__nv_bfloat16*>(d_cache.data_ptr()), static_cast<__nv_bfloat16*>(k_cache.data_ptr()),
         static_cast<float*>(g_cache.data_ptr()), static_cast<const int*>(query_start_loc.data_ptr()),
         static_cast<const int*>(ssm_state_indices.data_ptr()), static_cast<const int*>(write_pos.data_ptr()),
         static_cast<const int*>(cache_base.data_ptr()), static_cast<const int8_t*>(is_flush.data_ptr()),
         static_cast<float>(scale), B, H, HV, buf, (int)null_block_id, st, stream);
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess, "house_gdn_replayssm_spec launch failed: ", cudaGetErrorString(err));
}
#endif
