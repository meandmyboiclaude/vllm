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

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "kernel.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "torch_utils.h"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 750

}  // namespace marlin

torch::stable::Tensor marlin_gemm(
    torch::stable::Tensor& a, std::optional<torch::stable::Tensor> c_or_none,
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& b_bias_or_none,
    torch::stable::Tensor& b_scales,
    std::optional<torch::stable::Tensor> const& a_scales_or_none,
    std::optional<torch::stable::Tensor> const& global_scale_or_none,
    std::optional<torch::stable::Tensor> const& b_zeros_or_none,
    torch::stable::Tensor& workspace, vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float) {
  STD_TORCH_CHECK_NOT_IMPLEMENTED(false,
                                  "marlin_gemm(..) requires CUDA_ARCH >= 7.5");
  return torch::stable::empty({1, 1});
}

#else

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority

    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  // Get max scale groups per thread-block
  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, int has_zp, bool is_zp_float,
                          bool is_a_8bit, int stages) {
  int pack_factor = 32 / num_bits;

  // Get B size
  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, prob_m, prob_n, prob_k,
                                        num_bits, group_size, stages);
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size = tmp_size + sh_a_size + sh_s_size + sh_zp_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, int has_zp, bool is_zp_float,
                     bool is_a_8bit, int stages, int max_shared_mem) {
  // Sanity
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  // Verify K/N are divisible by thread K/N
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  // Verify min for thread K/N
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  // num_threads must be at least 128 (= 4 warps)
  if (th_config.num_threads < 128) {
    return false;
  }

  // Check that pipeline fits into cache
  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  #include "kernel_selector.h"

  return kernel;
}

#if MARLIN_OPR
// [0040 OPR] OPR twins. MarlinOPR<...> has Marlin's template parameters and is the same kernel with the ordered parallel
// reduce of split columns compiled in (marlin_template.h under MARLIN_OPR_TWIN 1). generate_kernels.py instantiates it ONLY
// for OPR_TWINS (sm80_kernel_opr_<c_type>.cu) and writes this selector (kernel_selector_opr.h); any other config returns
// nullptr and keeps the served Marlin<...> kernel. Called only from the armed branch in marlin_mm.
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void MarlinOPR(MARLIN_KERNEL_PARAMS);

static MarlinFuncPtr get_marlin_opr_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, int group_blocks, int threads, bool is_zp_float,
    int stages) {
  MarlinFuncPtr kernel = nullptr;

  #include "kernel_selector_opr.h"

  return kernel;
}
#endif

#if MARLIN_OPR && MARLIN_XSLICE
// [0039 XSOPR] Combined twins. MarlinXSOPR<...> = the XS cross-slice prefetch + the 0040 OPR in one body (marlin_template.h
// with MARLIN_XS_TWIN 1 and MARLIN_OPR_TWIN 1), instantiated ONLY for XSOPR_TWINS (sm80_kernel_xsopr_<c_type>.cu); this
// selector is kernel_selector_xsopr.h, nullptr = no combined twin for the config. Called only from the armed OPR branch.
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void MarlinXSOPR(MARLIN_KERNEL_PARAMS);

static MarlinFuncPtr get_marlin_xsopr_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, int group_blocks, int threads, bool is_zp_float,
    int stages) {
  MarlinFuncPtr kernel = nullptr;

  #include "kernel_selector_xsopr.h"

  return kernel;
}
#endif

#if MARLIN_XSLICE
// [0036 XS-5] XS twins. MarlinXS<...> has Marlin's template parameters and is the same kernel with the cross-slice
// prefetch compiled in (marlin_template.h under MARLIN_XS_TWIN 1). generate_kernels.py instantiates it ONLY for the configs
// the deploy gate can select (sm80_kernel_xs_<c_type>.cu) and writes this selector (kernel_selector_xs.h); any other
// config returns nullptr and its launch keeps the served Marlin<...> kernel. Called only from the armed branch in marlin_mm.
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void MarlinXS(MARLIN_KERNEL_PARAMS);

static MarlinFuncPtr get_marlin_xs_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, int group_blocks, int threads, bool is_zp_float,
    int stages) {
  MarlinFuncPtr kernel = nullptr;

  #include "kernel_selector_xs.h"

  return kernel;
}
#endif

#if MARLIN_FOLD
// [0042 FOLD] Remainder-fold twins. MarlinFOLD<...> has Marlin's template parameters and is the same kernel whose last 64-row
// problem copy may start at prob_m - 64 and skip the rows it shares with the previous copy (marlin_template.h under
// MARLIN_FOLD_TWIN 1). generate_kernels.py instantiates it ONLY for FOLD_TWINS (sm80_kernel_fold_<c_type>.cu) and writes this
// selector (kernel_selector_fold.h); any other config returns nullptr and marlin_mm re-plans the rows on the served path.
// Called only from the armed fold branch in marlin_mm.
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void MarlinFOLD(MARLIN_KERNEL_PARAMS);

static MarlinFuncPtr get_marlin_fold_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, int group_blocks, int threads, bool is_zp_float,
    int stages) {
  MarlinFuncPtr kernel = nullptr;

  #include "kernel_selector_fold.h"

  return kernel;
}
#endif

#if MARLIN_FOLDS
// [0048 FOLDS] Served-partition remainder-fold twins (marlin_template.h under MARLIN_FOLDS_TWIN 1): MarlinFOLDS<...> has Marlin's
// template parameters and runs, in ONE launch, the served launch of par_count 64-row copies (bit for bit: its partition, slices,
// locks and reduction order) plus the < 64-row remainder as a second item per CTA. generate_kernels.py instantiates it ONLY for
// FOLDS_TWINS (sm80_kernel_folds_<c_type>.cu) and writes this selector (kernel_selector_folds.h); any other config returns
// nullptr and marlin_mm keeps the served two-launch plan.
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id,
          const int threads, const int thread_m_blocks, const int thread_n_blocks,
          const int thread_k_blocks, const bool m_block_size_8, const int stages,
          const int group_blocks, const bool is_zp_float>
__global__ void MarlinFOLDS(MARLIN_KERNEL_PARAMS);

static MarlinFuncPtr get_marlin_folds_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, int group_blocks, int threads, bool is_zp_float,
    int stages) {
  MarlinFuncPtr kernel = nullptr;

  #include "kernel_selector_folds.h"

  return kernel;
}
#endif

// [club-kt 2026-09-12] Sweep/trace knobs for the thread-config heuristic (host side, read once).
//   VLLM_MARLIN_CFG_INDEX=<i>   force the i-th entry of the (small|large)_batch table when valid
//   VLLM_MARLIN_A8_NO_SMFILL=1  skip the "fill more SMs with {128,64,128}" override for int8 A
//   VLLM_MARLIN_CFG_TRACE=1     print the chosen config per launch to stderr
// Measured 2026-09-11 (4090, M=48, K=6144, N=5120 = o_proj/out_proj): int8-A gets {128,64,128}
// through the SM-fill override and runs 45.6 us; bf16-A on the same shape runs 30.9 us.
static int kt_env_int(const char* name, int dflt) {
  const char* v = getenv(name);
  if (v == nullptr || *v == '\0') return dflt;
  return atoi(v);
}
static const int KT_CFG_INDEX = kt_env_int("VLLM_MARLIN_CFG_INDEX", -1);
static const int KT_A8_NO_SMFILL = kt_env_int("VLLM_MARLIN_A8_NO_SMFILL", 0);
static const int KT_CFG_TRACE = kt_env_int("VLLM_MARLIN_CFG_TRACE", 0);
#if MARLIN_XSLICE
// [0035 XS] VLLM_MARLIN_XSLICE=1 arms cross-slice prefetch in an XS-built .so; unset/0 = OFF (every launch takes the
// served path). Read once per process; reaches the kernel as bit 1 of the tile_major argument.
static const int KT_XSLICE = kt_env_int("VLLM_MARLIN_XSLICE", 0);
// [0038 TRIM cuda 10-05] VLLM_MARLIN_SMEM_TRIM=1 launches each Marlin with the dynamic shared memory its own layout uses
// instead of the whole opt-in grant (101,376 B on the 4090). Every W4A8 decode tile addresses 50-61 KB of it, so with the full
// request no other kernel can be co-resident with the 1-block-per-SM grid (e.g. the GENESIS_ENABLE_GDN_BA_BRANCH in_proj_ba
// GEMM forked beside the qkvz Marlin). HOST ONLY: no kernel body changes; the attribute call keeps the full grant (it is a
// ceiling, so a node captured earlier can never exceed it); only the launch's dynamic smem and the kernel's max_shared_mem
// argument (read by the XS twin's xs_run check alone) shrink. kt_marlin_smem_need() mirrors the kernel's constexpr layout
// (marlin_template.h l.352-354 sh_a_s/sh_new, 563-608 strides/stages, 755-773 B/red/bias/zp/s/A, 782-785 XS twin) 1:1.
// Unset/0: a load-time constant 0 and one untaken branch -- the served launch byte for byte. A need above the grant keeps
// the grant (tell KEEP). Tells (stderr): "[marlin-TRIM] ARMED VLLM_MARLIN_SMEM_TRIM=1" once, then one
// "[marlin-TRIM] ON|KEEP a<A> b<B> zp<0|1> thr=<t> tmb=<m> tnb=<n> tkb=<k> gb=<g> xs=<0|1> smem=<need> grant=<grant>" per config.
static const int KT_SMEM_TRIM = kt_env_int("VLLM_MARLIN_SMEM_TRIM", 0);
static int kt_marlin_smem_need(bool a8, int w_bits, bool scale_8bit, bool has_zp,
                               bool zp_float, int tmb, int tnb, int tkb,
                               bool mbs8, int stages, int group_blocks, bool xs) {
  const int pack = 32 / w_bits;
  const int m_block = mbs8 ? 8 : 16 * tmb;
  const int head = a8 ? 4 * tmb : 0;                          // sh_a_s (int4)
  const int a_sh_stride = 16 * tkb / (a8 ? 16 : 8);
  const int a_sh_stage = a_sh_stride * m_block;
  const int b_sh_stride = ((tnb * 16) * 16 / pack) / (a8 ? 2 : 4);
  const int b_sh_stage = b_sh_stride * tkb / (a8 ? 2 : 1);
  const int sh_b = stages * b_sh_stage;
  const int sh_red = (2 * tnb + 1) * 16 * tmb;
  const int sh_bias = tnb * 16 / 8;
  const int mn = sh_red < sh_b ? sh_red : sh_b;
  const int mx = sh_red > sh_b ? sh_red : sh_b;
  const int b_red_bias = mx > mn + sh_bias ? mx : mn + sh_bias;
  const int s_sh_stride = 16 * tnb / (scale_8bit ? 16 : 8);
  const int s_tb_groups = (group_blocks != -1 && group_blocks < tkb) ? tkb / group_blocks : 1;
  const int s_sh_stage = s_tb_groups * s_sh_stride;
  const int zp_sh_stride = zp_float ? 16 * tnb / 8 : ((16 * tnb) / pack) / 4;
  const int zp_sh_stage = has_zp ? s_tb_groups * zp_sh_stride : 0;
  int total = head + b_red_bias + stages * zp_sh_stage + stages * s_sh_stage +
              stages * a_sh_stage;
  if (xs) total += sh_red;                                    // XS twin: sh_red past every stage buffer
  return 16 * total;
}
static void kt_trim_tell(bool on, int a_bits, int w_bits, bool has_zp, int thr,
                         int tmb, int tnb, int tkb, int gb, bool xs, int need,
                         int launch, int grant) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[200];
  snprintf(key, sizeof key, "a%d b%d zp%d thr=%d tmb=%d tnb=%d tkb=%d gb=%d xs=%d",
           a_bits, w_bits, (int)has_zp, thr, tmb, tnb, tkb, gb, (int)xs);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  if (told.empty()) fprintf(stderr, "[marlin-TRIM] ARMED VLLM_MARLIN_SMEM_TRIM=1\n");
  told.emplace_back(key);
  fprintf(stderr, "[marlin-TRIM] %s %s smem=%d grant=%d need=%d\n", on ? "ON" : "KEEP", key, launch, grant, need);
}
// [0046 BPSM2 cuda 10-06] VLLM_MARLIN_BPSM2 launches a Marlin GEMM with 2 * sms CTAs (two co-resident CTAs per SM) where that
// is EXACT and fits. Read once per process: unset/""/0 = OFF (a load-time constant 0 and two untaken branches: every launch
// keeps its served grid and smem byte for byte); 1 = on; 2 = on, and an XS twin that the doubled grid leaves with nothing to
// prefetch is swapped for the served kernel (smaller layout); anything else is a hard error at the first Marlin GEMM (never a
// silent OFF). VLLM_MARLIN_BPSM2_N: comma list of N (unset, empty or "*" = every N), parsed strictly at the first armed launch.
// HOST ONLY: no kernel changes; the attribute call keeps the full grant (a ceiling).
// EXACT by construction: gridDim reaches the kernel only through the stripe partition (marlin_template.h part2 / iters /
// init_slice / init_part2_slice). A launch is taken only if NEITHER gridDim blocks NOR 2 * blocks splits any column tile
// across CTAs (kt_bpsm2_split), so at both grids every tile is computed by one CTA over its whole k range with the same k loop,
// and the reduce site (slice_count > 1: lock chain, c_tmp, OPR fold) is never reached: the served output bit for bit. A
// launch that splits at either grid keeps the served grid: its slices, hence its fp32 partials and their fold, would change.
// FIT: the launch smem becomes the layout's own size (kt_marlin_smem_need, 0038 TRIM; + sh_red for an XS twin) and the launch
// is taken only if cudaOccupancyMaxActiveBlocksPerMultiprocessor(kernel it runs, threads, that smem) >= 2 (registers AND smem;
// the CC 8.x driver sizes the carveout itself, no attribute is set). One problem copy only (prob_m_split <= 16 * tmb).
// Tells (stderr): "[marlin-BPSM2] ARMED ..." once; "[marlin-BPSM2] ON|KEEP N=<n> (first M=<m>) K=<k> a<> w<> tmb=.. ..." once
// per (N, K, a/w bits, tmb, outcome); KEEP names its reason (split | occupancy | smem).
static int kt_bpsm2_mode() {
  const char* v = getenv("VLLM_MARLIN_BPSM2");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if ((v[0] == '1' || v[0] == '2') && v[1] == '\0') return v[0] - '0';
  return -1;
}
static const int KT_BPSM2 = kt_bpsm2_mode();
struct KtBpsm2Gate {
  bool all_n = true;
  std::vector<int> ns;
};
static const KtBpsm2Gate& kt_bpsm2_gate() {
  static const KtBpsm2Gate g = [] {
    const char* mode = getenv("VLLM_MARLIN_BPSM2");
    STD_TORCH_CHECK(KT_BPSM2 > 0, "VLLM_MARLIN_BPSM2: malformed value '", mode != nullptr ? mode : "",
                    "' (want 0, 1 or 2)");
    KtBpsm2Gate r;
    const char* s = getenv("VLLM_MARLIN_BPSM2_N");
    const std::string list = s != nullptr ? s : "";
    if (!list.empty() && list != "*") {
      r.all_n = false;
      std::string cur;
      for (size_t i = 0; i <= list.size(); ++i) {
        const char ch = i < list.size() ? list[i] : ',';
        if (ch == ',') {
          char* end = nullptr;
          const long x = strtol(cur.c_str(), &end, 10);
          STD_TORCH_CHECK(!cur.empty() && end == cur.c_str() + cur.size() && x >= 1 && x <= (1L << 30),
                          "VLLM_MARLIN_BPSM2_N: malformed value '", cur, "' (want a positive integer)");
          r.ns.push_back(static_cast<int>(x));
          cur.clear();
        } else if (ch != ' ') {
          cur.push_back(ch);
        }
      }
    }
    fprintf(stderr, "[marlin-BPSM2] ARMED VLLM_MARLIN_BPSM2=%d n=%s\n", KT_BPSM2, r.all_n ? "*" : list.c_str());
    return r;
  }();
  return g;
}
// [0046 BPSM2] BEGIN kt_bpsm2_split (self-contained: bpsm2/proof_bpsm2.py compiles this block alone)
// The stripe partition of marlin_template.h for ONE problem copy (parallel == 1) at gridDim G: part2 / part1 (l.523-527), iters
// with the group rounding (l.541-551); CTA b holds the flat part-2 k-tile range [iters * b, iters * (b + 1)) cut at column ends
// (init_slice / init_part2_slice). A column tile is split across CTAs (slice_count > 1) iff a CTA boundary iters * b with
// 0 < iters * b < k_tiles * part2 falls inside a column, i.e. iff iters < k_tiles * part2 and iters % k_tiles != 0.
// *tiles_per_cta (if not null) = the most column tiles one CTA computes (part 1 + part 2) when no tile is split, else -1.
static bool kt_bpsm2_split(int k_tiles, int n_tiles, int group_blocks, int thread_k_blocks, int G, int* tiles_per_cta) {
  int part2 = n_tiles, part1_iters = 0;
  if (n_tiles > G) {
    part2 = n_tiles % G;
    if (part2 * 3 <= G) part2 += G;
    part1_iters = (n_tiles - part2) / G;
  }
  int iters = (k_tiles * part2 + G - 1) / G;
  if (group_blocks != -1 && group_blocks >= thread_k_blocks) {
    const int g = group_blocks / thread_k_blocks;
    iters = g * ((iters + g - 1) / g);
  }
  const long long total = static_cast<long long>(k_tiles) * part2;
  const bool split = iters < total && iters % k_tiles != 0;
  if (tiles_per_cta != nullptr)
    *tiles_per_cta = split ? -1 : part1_iters + static_cast<int>((iters < total ? iters : total) / k_tiles);
  return split;
}
// [0046 BPSM2] END kt_bpsm2_split
static void kt_bpsm2_tell(bool on, const char* reason, int m, int n, int k, int a_bits, int w_bits, int tmb, int tnb,
                          int tkb, int thr, int blocks, int smem, int occ, bool xs, bool xs_dropped) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[96];
  snprintf(key, sizeof key, "%d/%d/a%d/w%d/tmb%d/%s", n, k, a_bits, w_bits, tmb, on ? "ON" : reason);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  if (on)
    fprintf(stderr,
            "[marlin-BPSM2] ON N=%d (first M=%d) K=%d a%d w%d tmb=%d tnb=%d tkb=%d thr=%d blocks=%d smem=%d occ=%d xs=%d "
            "xs_dropped=%d\n",
            n, m, k, a_bits, w_bits, tmb, tnb, tkb, thr, blocks, smem, occ, (int)xs, (int)xs_dropped);
  else
    fprintf(stderr, "[marlin-BPSM2] KEEP N=%d (first M=%d) K=%d a%d w%d tmb=%d tnb=%d tkb=%d reason=%s smem=%d occ=%d\n", n,
            m, k, a_bits, w_bits, tmb, tnb, tkb, reason, smem, occ);
}
// Resident CTAs per SM of `kernel` at `smem` dynamic bytes, cached per (kernel, threads, smem): the driver query runs once per
// config; a failed query reads as 0 (KEEP) and its error is cleared, never left for a later launch check to find.
static int kt_bpsm2_occ(MarlinFuncPtr kernel, int threads, int smem) {
  static std::mutex mu;
  static std::map<std::pair<std::uintptr_t, long long>, int> cache;
  const auto key = std::make_pair(reinterpret_cast<std::uintptr_t>(kernel), (static_cast<long long>(threads) << 32) | smem);
  std::lock_guard<std::mutex> lk(mu);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  int occ = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, threads, static_cast<size_t>(smem)) != cudaSuccess) {
    (void)cudaGetLastError();
    occ = 0;
  }
  cache.emplace(key, occ);
  return occ;
}
// [0036 XS-4] per-launch gate (K3 k3-20261003T134531Z: XS gains at M64 on N 34816/16384, loses at M8 and on the other N).
// With VLLM_MARLIN_XSLICE=1, bit 1 is set only for a launch whose rows lie in [VLLM_MARLIN_XSLICE_MIN_M, VLLM_MARLIN_XSLICE_MAX_M]
// (unset = 0 = no bound) and whose N is listed in VLLM_MARLIN_XSLICE_N (comma list; "*" alone = every N; UNSET or EMPTY = NO
// launch, so an unlisted model stays on the served path). Parsed strictly once, at the first armed launch: a malformed value or
// MIN_M > MAX_M is a hard error at that GEMM, never a silent OFF. Tells (stderr): "[marlin-XS] ARMED ..." once per process and
// "[marlin-XS] ON N=<n> (first M=<m>)" once per N. XSLICE unset/0 returns before any of this: the served path byte for byte.
// [0036 XS-5] XS3: a selected launch runs the twin kernel MarlinXS<...> (no tile_major bit); "ON N=" fires when a twin is launched.
static int kt_xs_strict(const char* name, const std::string& v) {
  char* end = nullptr;
  const long x = strtol(v.c_str(), &end, 10);
  STD_TORCH_CHECK(!v.empty() && end == v.c_str() + v.size() && x >= 0 && x <= (1L << 30), name, ": malformed value '", v,
                  "' (want a non-negative integer)");
  return static_cast<int>(x);
}
struct KtXsGate {
  int min_m = 0, max_m = 0;
  bool all_n = false;
  std::vector<int> ns;
  bool all_n_u3 = false;   // [0069 XSI3DN] VLLM_MARLIN_XSLICE_N_KU3S: "*" (every N) for kU3S launches
  std::vector<int> ns_u3;  // [0069 XSI3DN] the listed N for kU3S launches (v2: unset = the baked 5120; "" / "0" = none)
};
// [0069 XSI3DN cuda 2026-10-12] VLLM_MARLIN_XSLICE_N_KU3S: a second N list with VLLM_MARLIN_XSLICE_N's grammar (comma list, spaces
// ignored, strict non-negative integers, "*" alone = every N, UNSET or EMPTY = no launch) consulted ONLY for b_type == kU3S launches the
// N list did not admit. [v2, route (a): UNSET = the XR9 baked "5120" (int3 down N 5120 K 17408); "" or "0" = OFF (no launch).]
// int3 down and W4R down share N 5120 AND K 17408, so b_type is the only key that separates them. Every other XS
// precondition (VLLM_MARLIN_XSLICE=1, the MIN_M/MAX_M window, one problem copy, no bias, fp32 reduce, no atomic-add reduce, a twin
// exists) is the same code. Tells: "[marlin-XS] ARMED-KU3S n=<list> (0069: kU3S launches only)" once, after the unchanged ARMED line,
// only when the list is non-empty (v2: + " src=default|env"), and "[marlin-XS] KNOB (0069) VLLM_MARLIN_XSLICE_N_KU3S
// state=on|off src=default|env value=<list|0>" once, even when off; "[marlin-XS] ON-KU3S N=<n> K=<k> (first M=<m>)" once per (N, K) for a launch this list admitted.
static void kt_xs_parse_list_u3(const std::string& list, KtXsGate& r) {
  if (list == "*") {
    r.all_n_u3 = true;
    return;
  }
  if (list.empty()) return;
  std::string cur;
  for (size_t i = 0; i <= list.size(); ++i) {
    const char ch = i < list.size() ? list[i] : ',';
    if (ch == ',') {
      r.ns_u3.push_back(kt_xs_strict("VLLM_MARLIN_XSLICE_N_KU3S", cur));
      cur.clear();
    } else if (ch != ' ') {
      cur.push_back(ch);
    }
  }
}
static const KtXsGate& kt_xs_gate() {
  static const KtXsGate g = [] {
    KtXsGate r;
    const char* a = getenv("VLLM_MARLIN_XSLICE_MIN_M");
    const char* b = getenv("VLLM_MARLIN_XSLICE_MAX_M");
    if (a != nullptr && *a != '\0') r.min_m = kt_xs_strict("VLLM_MARLIN_XSLICE_MIN_M", a);
    if (b != nullptr && *b != '\0') r.max_m = kt_xs_strict("VLLM_MARLIN_XSLICE_MAX_M", b);
    STD_TORCH_CHECK(r.max_m == 0 || r.min_m <= r.max_m, "VLLM_MARLIN_XSLICE_MIN_M ", r.min_m, " > VLLM_MARLIN_XSLICE_MAX_M ", r.max_m);
    const char* s = getenv("VLLM_MARLIN_XSLICE_N");
    const std::string list = s != nullptr ? s : "";
    if (list == "*") {
      r.all_n = true;
    } else if (!list.empty()) {
      std::string cur;
      for (size_t i = 0; i <= list.size(); ++i) {
        const char ch = i < list.size() ? list[i] : ',';
        if (ch == ',') {
          r.ns.push_back(kt_xs_strict("VLLM_MARLIN_XSLICE_N", cur));
          cur.clear();
        } else if (ch != ' ') {
          cur.push_back(ch);
        }
      }
    }
    const char* s3 = getenv("VLLM_MARLIN_XSLICE_N_KU3S");  // [0069 XSI3DN] parsed before any tell: malformed = hard error here
    const bool u3_env = s3 != nullptr;  // [0069 v2 route (a)] UNSET = the XR9 baked "5120"; "" or "0" = OFF
    std::string list3 = u3_env ? s3 : "5120";
    if (list3 == "0") list3.clear();
    kt_xs_parse_list_u3(list3, r);
    fprintf(stderr, "[marlin-XS] ARMED min_m=%d max_m=%d n=%s\n", r.min_m, r.max_m,
            list.empty() ? "EMPTY (no launch takes XS)" : list.c_str());
    if (!list3.empty())
      fprintf(stderr, "[marlin-XS] ARMED-KU3S n=%s (0069: kU3S launches only) src=%s\n", list3.c_str(), u3_env ? "env" : "default");
    fprintf(stderr, "[marlin-XS] KNOB (0069) VLLM_MARLIN_XSLICE_N_KU3S state=%s src=%s value=%s\n", list3.empty() ? "off" : "on",
            u3_env ? "env" : "default", list3.empty() ? "0" : list3.c_str());
    return r;
  }();
  return g;
}
// [0036 XS-5] kt_xslice_on() is the XS-4 M/N gate unchanged (strict parse + ARMED at the first armed launch); the
// "ON N=" tell moved to kt_xs_tell_on(), which marlin_mm calls only when it actually launches a twin (XS2 told on the gate
// alone, even for a launch the kernel then ran on its served path).
static bool kt_xslice_on(int m, int n, bool u3, bool* by_u3) {
  if (!KT_XSLICE) return false;
  const KtXsGate& g = kt_xs_gate();
  if (m < g.min_m || (g.max_m > 0 && m > g.max_m)) return false;
  bool hit = g.all_n;
  for (int x : g.ns)
    if (x == n) hit = true;
  if (!hit && u3) {  // [0069 XSI3DN] the KU3S list, kU3S launches only (empty list: never admits)
    hit = g.all_n_u3;
    for (int x : g.ns_u3)
      if (x == n) hit = true;
    if (hit && by_u3 != nullptr) *by_u3 = true;
  }
  return hit;
}
static void kt_xs_tell_on(int m, int n) {
  static std::mutex told_mu;
  static std::vector<int> told;
  std::lock_guard<std::mutex> lk(told_mu);
  bool seen = false;
  for (int x : told)
    if (x == n) seen = true;
  if (!seen) {
    told.push_back(n);
    fprintf(stderr, "[marlin-XS] ON N=%d (first M=%d)\n", n, m);
  }
}
// [0069 XSI3DN] the launch-side tell of a launch the KU3S list admitted: once per (N, K)
static void kt_xs_tell_on_u3(int m, int n, int k) {
  static std::mutex told_mu;
  static std::vector<std::pair<int, int>> told;
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& t : told)
    if (t.first == n && t.second == k) return;
  told.emplace_back(n, k);
  fprintf(stderr, "[marlin-XS] ON-KU3S N=%d K=%d (first M=%d)\n", n, k, m);
}
#endif

#if MARLIN_OPR
// [0040 OPR] Run switch VLLM_MARLIN_OPR, read once per process: unset/""/0 = OFF (every launch takes the served path; one
// untaken branch on a load-time constant), 1 = v1 (the thread_n_blocks 4 twins: W4A8 out_proj/down/drafter, W8A16 o_proj),
// 2 = v1 + v2 (also the W8A16 qkv twins; their per-CTA slots, 128 x 64 KB, fill the served 8 MiB c_tmp exactly: no VRAM
// change). Anything else reads as -1 here and is a hard error at the first Marlin GEMM (never a silent OFF).
// Per-launch gate, parsed strictly at the first armed launch:
//   VLLM_MARLIN_OPR_N         comma list of N; unset, empty or "*" = every N (the partition checks below still apply);
//   VLLM_MARLIN_OPR_MIN_LATE  minimum late links (default 2 = MARLIN-GAP-PLAN §10 scope; 1 = every split launch).
// A launch takes its MarlinOPR twin only if: one 16*tmb-row problem copy, fp32 reduce, no atomic-add reduce, the served
// kernel is still selected (XS and any later twin take precedence), a twin exists, the partition has <= 4 slices per
// column, its slots fit the c_tmp marlin_gemm passes, and some split column has >= MIN_LATE late slices (slices that are
// their CTA's last piece: the links the served chain serialises after the main loop).
// Tells (stderr): "[marlin-OPR] ARMED ..." once per process; "[marlin-OPR] ON N=<n> (first M=<m>) slices=<s> ..." once per
// (N, K, a/b type) when a twin is launched.
static int kt_opr_mode() {
  const char* v = getenv("VLLM_MARLIN_OPR");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if ((v[0] == '1' || v[0] == '2') && v[1] == '\0') return v[0] - '0';
  return -1;
}
static const int KT_OPR = kt_opr_mode();

static int kt_opr_strict(const char* name, const std::string& v, long lo, long hi) {
  char* end = nullptr;
  const long x = strtol(v.c_str(), &end, 10);
  STD_TORCH_CHECK(!v.empty() && end == v.c_str() + v.size() && x >= lo && x <= hi, name, ": malformed value '", v,
                  "' (want an integer in [", lo, ", ", hi, "])");
  return static_cast<int>(x);
}
struct KtOprGate {
  bool all_n = true;
  std::vector<int> ns;
  int min_late = 2;
};
static const KtOprGate& kt_opr_gate() {
  static const KtOprGate g = [] {
    const char* mode = getenv("VLLM_MARLIN_OPR");
    STD_TORCH_CHECK(KT_OPR > 0, "VLLM_MARLIN_OPR: malformed value '", mode != nullptr ? mode : "", "' (want 0, 1 or 2)");
    KtOprGate r;
    const char* s = getenv("VLLM_MARLIN_OPR_N");
    const std::string list = s != nullptr ? s : "";
    if (!list.empty() && list != "*") {
      r.all_n = false;
      std::string cur;
      for (size_t i = 0; i <= list.size(); ++i) {
        const char ch = i < list.size() ? list[i] : ',';
        if (ch == ',') {
          r.ns.push_back(kt_opr_strict("VLLM_MARLIN_OPR_N", cur, 1, 1L << 30));
          cur.clear();
        } else if (ch != ' ') {
          cur.push_back(ch);
        }
      }
    }
    const char* l = getenv("VLLM_MARLIN_OPR_MIN_LATE");
    if (l != nullptr && *l != '\0') r.min_late = kt_opr_strict("VLLM_MARLIN_OPR_MIN_LATE", l, 1, 64);
    fprintf(stderr, "[marlin-OPR] ARMED VLLM_MARLIN_OPR=%d n=%s min_late=%d twins=%s\n", KT_OPR,
            r.all_n ? "*" : list.c_str(), r.min_late, KT_OPR == 2 ? "v1+v2(qkv)" : "v1");
    return r;
  }();
  return g;
}

// [0040 OPR] BEGIN kt_opr_plan (self-contained: opr/test_opr_plan.cpp compiles this block alone)
// Port of marlin_template.h's part-2 partition for one problem copy (parallel == 1; init_slice / init_part2_slice), the
// same port as marlingap/opr_model.py: per split column the slice count and the late slices (a slice is late when it is its
// CTA's last piece), the kernel's slice bound smax = ceil(k_tiles / iters) + 1 and the slot floats the twin addresses,
// blocks * (16 tmb) * (16 tnb): one slot per CTA (a CTA holds at most one non-final slice, its first part-2 piece).
struct KtOprPlan {
  int k_tiles = 0, n_tiles = 0, part2 = 0, iters = 0, smax = 0;
  int split_cols = 0, max_slices = 1, late_max = 0;
  int64_t slot_floats = 0;
};
static KtOprPlan kt_opr_plan(int prob_n, int prob_k, int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
                             int group_blocks, int blocks) {
  KtOprPlan p;
  p.k_tiles = prob_k / 16 / thread_k_blocks;
  p.n_tiles = prob_n / 16 / thread_n_blocks;
  p.part2 = p.n_tiles;
  if (p.n_tiles > blocks) {
    p.part2 = p.n_tiles % blocks;
    if (p.part2 * 3 <= blocks) p.part2 += blocks;
  }
  if (p.k_tiles <= 0 || p.part2 <= 0 || blocks <= 0) return p;
  p.iters = div_ceil(p.k_tiles * p.part2, blocks);
  if (group_blocks != -1 && group_blocks >= thread_k_blocks)
    p.iters = (group_blocks / thread_k_blocks) * div_ceil(p.iters, group_blocks / thread_k_blocks);
  p.smax = div_ceil(p.k_tiles, p.iters) + 1;
  p.slot_floats = static_cast<int64_t>(blocks) * (16 * thread_m_blocks) * (16 * thread_n_blocks);
  std::vector<int> cnt(p.part2, 0), late(p.part2, 0);
  const int64_t total = static_cast<int64_t>(p.k_tiles) * p.part2;
  for (int b = 0; b < blocks; b++) {
    const int64_t lo = static_cast<int64_t>(p.iters) * b;
    const int64_t end = static_cast<int64_t>(p.iters) * (b + 1);
    const int64_t hi = end < total ? end : total;
    for (int64_t x = lo; x < hi;) {
      const int col = static_cast<int>(x / p.k_tiles);
      const int64_t r0 = x % p.k_tiles;
      const int64_t r1 = r0 + hi - x < p.k_tiles ? r0 + hi - x : p.k_tiles;
      cnt[col]++;
      x += r1 - r0;
      if (x == hi) late[col]++;
    }
  }
  for (int c = 0; c < p.part2; c++) {
    if (cnt[c] < 2) continue;
    p.split_cols++;
    if (cnt[c] > p.max_slices) p.max_slices = cnt[c];
    if (late[c] > p.late_max) p.late_max = late[c];
  }
  return p;
}
// [0040 OPR] END kt_opr_plan

// [0039 XSOPR] tag = "OPR" (MarlinOPR, the 0040 tell text unchanged) or "XSOPR" (MarlinXSOPR); one tell per (tag, N, K, a, w).
static void kt_opr_tell_on(const char* tag, int m, int n, int k, int a_bits, int b_bits, const KtOprPlan& p) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[96];
  snprintf(key, sizeof(key), "%s/%d/%d/a%d/w%d", tag, n, k, a_bits, b_bits);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  fprintf(stderr,
          "[marlin-%s] ON N=%d (first M=%d) slices=%d late=%d K=%d a%d w%d split_cols=%d smax=%d slots=%.2fMiB\n",
          tag, n, m, p.max_slices, p.late_max, k, a_bits, b_bits, p.split_cols, p.smax,
          p.slot_floats * 4.0 / (1024.0 * 1024.0));
}
#endif

#if MARLIN_SCALE_PF
// [0041 MR14] Run switch VLLM_MARLIN_SCALE_PREFETCH, read once per process: unset/""/0 = OFF (every launch runs the kernel the
// earlier selections chose; one untaken branch on a load-time constant). 1 or 2 = the launch runs the SP twin of that kernel
// (Marlin -> MarlinSP, MarlinXS -> MarlinXSSP, MarlinOPR -> MarlinOPRSP, MarlinXSOPR -> MarlinXSOPRSP; no twin = keep), whose
// first statement issues prefetch.global.L2 over the group-scale + grouped zero-point rows of the CTA's whole stripe:
//   1 = L2::evict_last   (the MR14 brief's form)
//   2 = L2::evict_normal (no high-priority lines left behind after the GEMM; the served B stream is evict_first, so these
//                         still outrank it inside the GEMM)
// Anything else is a hard error at the first Marlin GEMM (never a silent OFF). VLLM_MARLIN_SCALE_PREFETCH_N: comma list of N
// (unset, empty or "*" = every N), parsed strictly at the first armed launch. A launch takes a twin only with one 16*tmb-row
// problem copy and tile_major == 0 (the [0033] order off): the twin reads its mode from tile_major >> 1, and with one problem
// copy the kernel's part-1 tile-major map equals the par-major one, so the nonzero tile_major it then sees changes no tile.
// Tells (stderr): "[marlin-SP] ARMED ..." once per process; "[marlin-SP] ON N=<n> (first M=<m>) K=<k> a<bits> w<bits>
// g<group> twin=<name> mode=<1|2>" once per (N, K, a/w bits, twin) when a twin is launched.
static int kt_sp_mode() {
  const char* v = getenv("VLLM_MARLIN_SCALE_PREFETCH");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if ((v[0] == '1' || v[0] == '2') && v[1] == '\0') return v[0] - '0';
  return -1;
}
static const int KT_SP = kt_sp_mode();

static int kt_sp_strict(const char* name, const std::string& v) {
  char* end = nullptr;
  const long x = strtol(v.c_str(), &end, 10);
  STD_TORCH_CHECK(!v.empty() && end == v.c_str() + v.size() && x >= 1 && x <= (1L << 30), name, ": malformed value '", v,
                  "' (want a positive integer)");
  return static_cast<int>(x);
}
struct KtSpGate {
  bool all_n = true;
  std::vector<int> ns;
  int bits = 0;  // kernel bits (tile_major >> 1): 1 on, 2 evict_normal (else evict_last)
};
static const KtSpGate& kt_sp_gate() {
  static const KtSpGate g = [] {
    const char* mode = getenv("VLLM_MARLIN_SCALE_PREFETCH");
    STD_TORCH_CHECK(KT_SP > 0, "VLLM_MARLIN_SCALE_PREFETCH: malformed value '", mode != nullptr ? mode : "",
                    "' (want 0, 1 or 2)");
    KtSpGate r;
    r.bits = 1 | (KT_SP == 2 ? 2 : 0);
    const char* s = getenv("VLLM_MARLIN_SCALE_PREFETCH_N");
    const std::string list = s != nullptr ? s : "";
    if (!list.empty() && list != "*") {
      r.all_n = false;
      std::string cur;
      for (size_t i = 0; i <= list.size(); ++i) {
        const char ch = i < list.size() ? list[i] : ',';
        if (ch == ',') {
          r.ns.push_back(kt_sp_strict("VLLM_MARLIN_SCALE_PREFETCH_N", cur));
          cur.clear();
        } else if (ch != ' ') {
          cur.push_back(ch);
        }
      }
    }
    fprintf(stderr, "[marlin-SP] ARMED VLLM_MARLIN_SCALE_PREFETCH=%d n=%s prefetch=%s\n", KT_SP,
            r.all_n ? "*" : list.c_str(), (r.bits & 2) ? "evict_normal" : "evict_last");
    return r;
  }();
  return g;
}
static void kt_sp_tell_on(int m, int n, int k, int a_bits, int b_bits, int group_size, const char* twin) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[128];
  snprintf(key, sizeof(key), "%d/%d/a%d/w%d/%s", n, k, a_bits, b_bits, twin);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  fprintf(stderr, "[marlin-SP] ON N=%d (first M=%d) K=%d a%d w%d g%d twin=%s mode=%d\n", n, m, k, a_bits, b_bits,
          group_size, twin, KT_SP);
}

  #include "kernel_selector_sp.h"
#endif

#if MARLIN_FOLD
// [0042 FOLD] VLLM_MARLIN_REMAINDER_FOLD_NK = an N x K allow-list (e.g. "34816x5120,5120x17408"), read once per process: unset or
// empty = OFF (never fold: a load-time constant 0 and one untaken branch; every launch is the served one). For a listed GEMM
// (prob_n, prob_k), where marlin_mm would run par_count 64-row copies and then give the < 64-row remainder a launch of its own
// (0 < par_count < max_par, a remainder left, the 64-row tile, fp32 reduce, no atomic-add reduce: the 0035 condition), the
// remainder is folded into the copies' launch (prob_m_split = rest_m) and that launch runs the MarlinFOLD twin of the config
// marlin_mm chose for it. A config without a twin: the fold is dropped and the rows are re-planned on the served path.
// Parsed strictly at the first Marlin GEMM: comma-separated "<N>x<K>" items, N and K positive decimal integers (spaces
// ignored); anything else is a hard error there, never a silent OFF. A listed shape that never has a remainder never folds.
// Tells (stderr): "[marlin-FOLD] ARMED VLLM_MARLIN_REMAINDER_FOLD_NK=<list>" once per process; "[marlin-FOLD] ON N=<n> K=<k>
// (first M=<m>) parallel=<p> last_off=<o>" once per (N, K) when a twin is launched (M = the launch's rows); "[marlin-FOLD]
// NO-TWIN N=<n> K=<k> (first M=<m>) ..." once per (N, K) when a fold is dropped for want of a twin.
static int kt_fold_mode() {
  const char* e = getenv("VLLM_MARLIN_REMAINDER_FOLD_NK");
  return (e != nullptr && *e != '\0') ? 1 : 0;
}
static const int KT_FOLD = kt_fold_mode();

static int kt_fold_strict(const std::string& item, const std::string& v) {
  char* end = nullptr;
  const long x = strtol(v.c_str(), &end, 10);
  STD_TORCH_CHECK(!v.empty() && v.find_first_not_of("0123456789") == std::string::npos &&
                      end == v.c_str() + v.size() && x >= 1 && x <= (1L << 30),
                  "VLLM_MARLIN_REMAINDER_FOLD_NK: malformed item '", item, "' (want <N>x<K>, positive integers)");
  return static_cast<int>(x);
}
static const std::vector<std::pair<int, int>>& kt_fold_nk() {
  static const std::vector<std::pair<int, int>> g = [] {
    const char* e = getenv("VLLM_MARLIN_REMAINDER_FOLD_NK");
    const std::string list = e != nullptr ? e : "";
    std::vector<std::pair<int, int>> v;
    std::string cur;
    for (size_t i = 0; i <= list.size(); ++i) {
      const char ch = i < list.size() ? list[i] : ',';
      if (ch == ',') {
        const size_t x = cur.find('x');
        STD_TORCH_CHECK(x != std::string::npos, "VLLM_MARLIN_REMAINDER_FOLD_NK: malformed item '", cur,
                        "' (want <N>x<K>, positive integers)");
        v.emplace_back(kt_fold_strict(cur, cur.substr(0, x)), kt_fold_strict(cur, cur.substr(x + 1)));
        cur.clear();
      } else if (ch != ' ') {
        cur.push_back(ch);
      }
    }
    fprintf(stderr, "[marlin-FOLD] ARMED VLLM_MARLIN_REMAINDER_FOLD_NK=%s\n", list.c_str());
    return v;
  }();
  return g;
}
static void kt_fold_tell(bool on, int n, int k, int m, int parallel, int last_off, int a_bits, int w_bits, int thr, int tmb,
                         int tnb, int tkb, int gb) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[64];
  snprintf(key, sizeof(key), "%s/%d/%d", on ? "ON" : "NO-TWIN", n, k);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  if (on)
    fprintf(stderr, "[marlin-FOLD] ON N=%d K=%d (first M=%d) parallel=%d last_off=%d\n", n, k, m, parallel, last_off);
  else
    fprintf(stderr,
            "[marlin-FOLD] NO-TWIN N=%d K=%d (first M=%d) a%d w%d thr=%d tmb=%d tnb=%d tkb=%d gb=%d: fold dropped, served plan\n",
            n, k, m, a_bits, w_bits, thr, tmb, tnb, tkb, gb);
}
#endif

#if MARLIN_FOLDS
// [0048 FOLDS] VLLM_MARLIN_REMAINDER_FOLD_MODE selects HOW a listed GEMM (VLLM_MARLIN_REMAINDER_FOLD_NK, unchanged) folds:
// unset / empty / "0042" = the 0042 fold above (the reference); "S" = MarlinFOLDS with the PAIR schedule when par_count == 1 and
// the grid is even, AR otherwise; "SAR" = AR always; "SRA" = RA always (the schedules: marlin_template.h [0048 FOLDS]); "S1" = S
// only at par_count == 1 (par_count >= 2 keeps the served launch B); "SB" = S only where launch B would take launch A's tile, so
// the remainder rows are bitwise too (every row of the launch = the served plan's). Read once
// per process, strict: anything else is a hard error at the first Marlin GEMM. Mode S keeps the served plan's thread config,
// grid and partition (prob_m_split stays par_count * 64); the launch gets prob_m = rest_m, the schedule in tile_major bits 1-2, a
// private zeroed lock buffer of >= 2 * grid ints (marlin_gemm, persistent per stream) and needs C_tmp for 2 * grid slots
// (marlin_gemm doubles the persistent c_tmp when mode S is armed). Every unmet condition keeps the served two-launch plan and
// says why once per (N, K): "[marlin-FOLDS] ON N= K= (first M=) p= r= sched= grid=", "[marlin-FOLDS] SKIP N= K= (first M=)
// <reason>". A16 GEMMs and decode (M <= 64) never fold.
static int kt_folds_mode() {
  const char* e = getenv("VLLM_MARLIN_REMAINDER_FOLD_MODE");
  if (e == nullptr || *e == '\0' || std::string(e) == "0042") return 0;
  const std::string v(e);
  if (v == "S") return 1;
  if (v == "SAR") return 2;
  if (v == "SRA") return 3;
  if (v == "S1") return 4;
  if (v == "SB") return 5;
  return -1;
}
static const int KT_FOLDS_RAW = kt_folds_mode();
static int kt_folds_armed() {  // 0 = 0042 (or no fold at all), 1..3 = S / SAR / SRA; malformed -> hard error at first use
  STD_TORCH_CHECK(KT_FOLDS_RAW >= 0, "VLLM_MARLIN_REMAINDER_FOLD_MODE must be unset, \"0042\", \"S\", \"SAR\", \"SRA\", \"S1\" or \"SB\", got '",
                  getenv("VLLM_MARLIN_REMAINDER_FOLD_MODE"), "'");
  return KT_FOLD ? KT_FOLDS_RAW : 0;
}
static void kt_folds_tell(bool on, int n, int k, int m, const char* what, int p, int r, int sched, int grid) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[96];
  snprintf(key, sizeof(key), "%s/%d/%d", on ? "ON" : what, n, k);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  if (on)
    fprintf(stderr, "[marlin-FOLDS] ON N=%d K=%d (first M=%d) p=%d r=%d sched=%s grid=%d\n", n, k, m, p, r,
            sched == 2 ? "PAIR" : sched == 1 ? "RA" : "AR", grid);
  else
    fprintf(stderr, "[marlin-FOLDS] SKIP N=%d K=%d (first M=%d) %s: served plan\n", n, k, m, what);
}
#endif

// [0072 STRICTPARSE] the strict whole-string integer parse of the 0043 / 0068 knobs, moved here unchanged from inside
// #if MARLIN_TAILPF: the 0072 parses (MAX_TMB, TILE_MAJOR_NK, CTMP_PER_CALL, ZERO_CTMP) compile in every build.
static long kt_tpf_strict(const char* name, const std::string& v, long lo, long hi) {
  char* end = nullptr;
  const long x = strtol(v.c_str(), &end, 10);
  STD_TORCH_CHECK(!v.empty() && end == v.c_str() + v.size() && x >= lo && x <= hi, name, ": malformed value '", v,
                  "' (want an integer in [", lo, ", ", hi, "])");
  return x;
}

#if MARLIN_TAILPF
// [0043 TAILPF] Next-launch prefetch ("software PDL": PDL / griddepcontrol overlap needs sm_90; we serve on sm_89). Run switch
// VLLM_MARLIN_TAIL_PREFETCH, read once per process: unset/""/0 = OFF (two untaken branches on a load-time constant per launch;
// every launch is byte for byte the kernel the earlier selections chose). 1 = a launch whose kernel has a TPF twin (Marlin ->
// MarlinTPF, MarlinXS -> MarlinXSTPF, MarlinOPR -> MarlinOPRTPF, MarlinXSOPR -> MarlinXSOPRTPF) runs it; after its main loop
// and last epilogue every CTA of the twin issues prefetch.global.L2 for its share of the NEXT Marlin launch's group scales and
// grouped zero points. 2 = also the first VLLM_MARLIN_TAIL_PREFETCH_B_STAGES (1..3, default 2) k-tiles of B of every CTA of the
// next launch. Anything else: hard error at the first Marlin GEMM (never a silent OFF).
//   VLLM_MARLIN_TAIL_PREFETCH_EVICT   normal (default) | last | auto: the L2 eviction class of the prefetch. auto = evict_last
//                                     for a next launch whose B exceeds 48 MiB (gate_up / d.gate_up 89 MB, qkv 73 MB), else
//                                     normal: the served B stream is plain LDGSTS.E.BYPASS (not evict_first), so under LRU a
//                                     big GEMM's own B stream evicts normal-class scale lines before its late k-groups read them
//   VLLM_MARLIN_TAIL_PREFETCH_MAX_MB  1..64 (default 16): a next launch whose scales + zp exceed this gets no scale prefetch
//                                     (lm_head g32 carries ~99 MB, more than the 72 MB L2)
//   VLLM_MARLIN_TAIL_PREFETCH_N       comma list of the N of the launches that CARRY the prefetch (unset/empty/"*" = every N)
//   VLLM_MARLIN_TAIL_PREFETCH_A16FS   [0062 A16TPF] 0 (default) | 1: a 0057 A16FS launch (listed 16-bit-activation shape, pinned
//                                     config, one block per column) with a TPF twin runs it, with the A16FS grid. The twin is the
//                                     same template instance + the tail prefetch after the last epilogue, so every column is still
//                                     ONE block over the whole K and the output bits are the plain launch's. 0: A16FS launches keep
//                                     their kernel (they still report to the successor registry, as before). Tell (1 only):
//                                     "[marlin-TAILPF] A16FS ARMED (0062) ..." once.
//   VLLM_MARLIN_TAILPF_BFULL          [0068 TPFOUT v1.2, route (a)] UNSET = the XR9 baked pairs 16384x5120:5120x6144,5120x17408:
//                                     6144x5120 (GDN in_qkvz -> out_proj, MLP down -> drafter qkv); "" / "0" = off. Else a comma list
//                                     of PNxPK:SNxSK pairs: a
//                                     carrier shape and its successor's shape (16384x5120:5120x6144 = GDN in_qkvz -> out_proj).
//                                     When a launch of shape SN x SK follows a launch of shape PN x PK, the predecessor's tail
//                                     prefetches ALL of its B (not only mode 2's first stages) beside its scale / zp lines. The
//                                     descriptor is the 0043 B-stage form over a VIRTUAL partition of the successor's B (every
//                                     column tile, k-tiles in chunks of 4..16, no part 1), so the twins run unchanged code. The
//                                     carrier shape is part of the key because GDN out_proj and attention o_proj share 5120 x
//                                     6144 but not their predecessors' windows. Needs TAILPF armed (mode 1 or 2).
//   VLLM_MARLIN_TAILPF_BFULL_FRAC     1..100; UNSET = 50 (v1.2 baked), "" / "0" = 100: percent of every chunk's k-tiles
//                                     prefetched (its first ones).
//   VLLM_MARLIN_TAILPF_BFULL_MAX_MB   1..48; unset / "" / "0" = 24: a listed successor whose B exceeds this gets no whole-B part.
//   VLLM_MARLIN_TAILPF_BFULL_EVICT    normal | last; unset / "" / "0" = normal: the L2 class of every line of a whole-B descriptor.
//   VLLM_MARLIN_TAILPF_LINKTELL       unset / "" / "0" = off | 1: diagnostic tells, one "[marlin-TAILPF] LINK cap=<id> ..." per (capture,
//                                     carrier shape, successor shape) link a graph capture patches, and one per capture's last
//                                     twin left without a successor. Host-only, like every 0068 knob.
//                                     [v1.2] one "[marlin-TAILPF] KNOB (0068) <key> state=on|off src=default|env value=<effective>"
//                                     line per key at the gate parse, printed even when off ("" / "0" is the OFF sentinel).
// Which launch is next: (a) eager launches: a per-process registry keyed by the B pointer (one per layer and projection) learns
// every launch's successor from two consecutive eager launches on its stream; an eager launch carries the descriptor of its
// last learned successor (all-null while unknown: the twin's tail issues nothing). Graph replays are invisible to the host,
// so an eager successor learned across a replay can be the wrong launch (a cost in DRAM bytes, never in results).
// (b) CUDA graphs: a captured TPF twin is captured with an all-null descriptor; when the next Marlin launch on the same stream
// is captured in the same capture sequence, the twin's kernel node is re-set (cudaGraphKernelNodeSetParams; the runtime
// permits every operation except destroy and node removal on the graph while its capture is in progress) to that launch's
// exact descriptor. A graph therefore replays only true successors; its last Marlin node prefetches nothing. Arguments are
// baked into the kernel node at capture, and weights, scales and zero points never move while a graph lives.
// A prefetch changes no architectural state: a wrong or stale successor costs DRAM traffic, never a result. A failing capture
// API call clears its own error and disables (b) for the process (tell CAPTURE-PATCH OFF); no API call is made while another
// CUDA error is pending (it would be cleared with ours). Not together with VLLM_MARLIN_SCALE_PREFETCH (MR14): the two are
// alternatives for the same lines and SP twins have no TPF twin (hard error).
// Tells (stderr): "[marlin-TAILPF] ARMED ..." once; "[marlin-TAILPF] ON N=<n> (first M=<m>) K=<k> a<bits> w<bits> twin=<name>
// next=..." once per (N, K, a/w bits, twin); "[marlin-TAILPF] NEXT N=.. K=.. -> N=.. K=.. ..." once per learned (shape, next
// shape) pair; "[marlin-TAILPF] PATCH ..." at the first graph-node patch; "[marlin-TAILPF] NO-TWIN ..." once per (N, K, config)
// of an armed launch that has no twin.
  #include "tailpf_desc.h"
using MarlinTpfFuncPtr = void (*)(MarlinTpfDesc, MARLIN_KERNEL_PARAMS);
  #include "kernel_selector_tpf.h"

static int kt_tpf_mode() {
  const char* v = getenv("VLLM_MARLIN_TAIL_PREFETCH");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if ((v[0] == '1' || v[0] == '2') && v[1] == '\0') return v[0] - '0';
  return -1;
}
static const int KT_TPF = kt_tpf_mode();
// [0068 TPFOUT] a whole-B or link-tell knob without TAILPF armed does nothing: say so once, at load (no per-launch branch)
[[maybe_unused]] static const bool KT_TPF_BFULL_INERT = [] {
  const char* b = getenv("VLLM_MARLIN_TAILPF_BFULL");
  const char* l = getenv("VLLM_MARLIN_TAILPF_LINKTELL");
  const bool set = (b != nullptr && *b != '\0' && !(b[0] == '0' && b[1] == '\0')) ||
                   (l != nullptr && *l != '\0' && !(l[0] == '0' && l[1] == '\0'));
  if (KT_TPF == 0 && set)
    fprintf(stderr, "[marlin-TAILPF] BFULL INERT (0068): VLLM_MARLIN_TAILPF_BFULL / _LINKTELL set but VLLM_MARLIN_TAIL_PREFETCH "
                    "is off: no launch prefetches\n");
  return KT_TPF == 0 && set;
}();

struct KtTpfGate {
  bool all_n = true;
  std::vector<int> ns;
  int evict = 0;     // 0 normal, 1 last, 2 auto (last for a next launch with B > 48 MiB)
  int b_stages = 0;  // 0 = mode 1 (scales + zero points only)
  long long max_bytes = 0;
  bool a16fs = false;  // [0062 A16TPF] A16FS launches may run their TPF twin
  // [0068 TPFOUT] whole-B prefetch: carrier -> successor shape pairs, 4 ints each (pn, pk, sn, sk); empty = off
  std::vector<int> bfull;
  int bfull_frac = 100;
  long long bfull_max_bytes = 24LL << 20;
  int bfull_evict_last = 0;
  bool linktell = false;
};
// [0068 TPFOUT] true iff the gate names (sn x sk) as the successor of some listed pair (the pair test is
// KtTpfRegistry::bfull_listed)
static bool kt_tpf_bfull_succ(const std::vector<int>& v, int sn, int sk) {
  for (size_t i = 0; i + 3 < v.size(); i += 4)
    if (v[i + 2] == sn && v[i + 3] == sk) return true;
  return false;
}
static const KtTpfGate& kt_tpf_gate() {
  static const KtTpfGate g = [] {
    const char* mode = getenv("VLLM_MARLIN_TAIL_PREFETCH");
    STD_TORCH_CHECK(KT_TPF > 0, "VLLM_MARLIN_TAIL_PREFETCH: malformed value '", mode != nullptr ? mode : "",
                    "' (want 0, 1 or 2)");
  #if MARLIN_SCALE_PF
    STD_TORCH_CHECK(KT_SP == 0, "VLLM_MARLIN_TAIL_PREFETCH and VLLM_MARLIN_SCALE_PREFETCH are alternatives for the same ",
                    "group-scale lines (an SP twin has no TPF twin): arm one of them");
  #endif
    KtTpfGate r;
    const char* ev = getenv("VLLM_MARLIN_TAIL_PREFETCH_EVICT");
    const std::string e = ev != nullptr ? ev : "";
    STD_TORCH_CHECK(e.empty() || e == "normal" || e == "last" || e == "auto",
                    "VLLM_MARLIN_TAIL_PREFETCH_EVICT: malformed value '", e, "' (want normal, last or auto)");
    r.evict = e == "last" ? 1 : e == "auto" ? 2 : 0;
    const char* bs = getenv("VLLM_MARLIN_TAIL_PREFETCH_B_STAGES");
    const std::string b = bs != nullptr ? bs : "";
    if (KT_TPF == 2)
      r.b_stages = b.empty() ? 2 : static_cast<int>(kt_tpf_strict("VLLM_MARLIN_TAIL_PREFETCH_B_STAGES", b, 1, 3));
    else
      STD_TORCH_CHECK(b.empty(), "VLLM_MARLIN_TAIL_PREFETCH_B_STAGES is set but VLLM_MARLIN_TAIL_PREFETCH=1 prefetches ",
                      "no B stages (use 2)");
    const char* mb = getenv("VLLM_MARLIN_TAIL_PREFETCH_MAX_MB");
    const std::string m = mb != nullptr ? mb : "";
    r.max_bytes = (m.empty() ? 16LL : static_cast<long long>(kt_tpf_strict("VLLM_MARLIN_TAIL_PREFETCH_MAX_MB", m, 1, 64))) *
                  (1LL << 20);
    const char* s = getenv("VLLM_MARLIN_TAIL_PREFETCH_N");
    const std::string list = s != nullptr ? s : "";
    if (!list.empty() && list != "*") {
      r.all_n = false;
      std::string cur;
      for (size_t i = 0; i <= list.size(); ++i) {
        const char ch = i < list.size() ? list[i] : ',';
        if (ch == ',') {
          r.ns.push_back(static_cast<int>(kt_tpf_strict("VLLM_MARLIN_TAIL_PREFETCH_N", cur, 1, 1L << 30)));
          cur.clear();
        } else if (ch != ' ') {
          cur.push_back(ch);
        }
      }
    }
    fprintf(stderr, "[marlin-TAILPF] ARMED VLLM_MARLIN_TAIL_PREFETCH=%d evict=%s b_stages=%d max_mb=%lld n=%s\n", KT_TPF,
            r.evict == 1 ? "last" : r.evict == 2 ? "auto" : "normal", r.b_stages, r.max_bytes >> 20,
            r.all_n ? "*" : list.c_str());
    const char* a16 = getenv("VLLM_MARLIN_TAIL_PREFETCH_A16FS");  // [0062 A16TPF]
    const std::string a = a16 != nullptr ? a16 : "";
    STD_TORCH_CHECK(a.empty() || a == "0" || a == "1", "VLLM_MARLIN_TAIL_PREFETCH_A16FS: malformed value '", a, "' (want 0 or 1)");
    r.a16fs = a == "1";
    if (r.a16fs)
      fprintf(stderr, "[marlin-TAILPF] A16FS ARMED (0062): 0057 A16FS launches with a TPF twin run it (pinned config, A16FS grid)\n");
    // [0068 TPFOUT] whole-B pairs "PNxPK:SNxSK[,...]" (strict: every item exactly two NxK shapes), fraction, cap, class, link tells
    // [0068 v1.2 route (a)] an UNSET key takes its XR9 baked value (BFULL = the two pairs, FRAC = 50; MAX_MB / EVICT / LINKTELL their
    // built-in 24 / normal / off); "" or "0" = OFF (BFULL / LINKTELL off, FRAC / MAX_MB / EVICT their built-in 100 / 24 / normal).
    const auto kt_knob = [](const char* key, const char* baked, bool* from_env) {
      const char* v = getenv(key);
      *from_env = v != nullptr;
      return std::string(v != nullptr ? v : baked);
    };
    bool bf_env = false, fr_env = false, bm_env = false, be_env = false, ls_env = false;
    const std::string bl = kt_knob("VLLM_MARLIN_TAILPF_BFULL", "16384x5120:5120x6144,5120x17408:6144x5120", &bf_env);
    if (!bl.empty() && bl != "0") {
      std::string item;
      for (size_t i = 0; i <= bl.size(); ++i) {
        const char ch = i < bl.size() ? bl[i] : ',';
        if (ch == ' ') continue;
        if (ch != ',') {
          item.push_back(ch);
          continue;
        }
        const size_t c = item.find(':');
        STD_TORCH_CHECK(c != std::string::npos && item.find(':', c + 1) == std::string::npos,
                        "VLLM_MARLIN_TAILPF_BFULL: malformed item '", item, "' (want PNxPK:SNxSK)");
        for (const std::string& shape : {item.substr(0, c), item.substr(c + 1)}) {
          const size_t x = shape.find('x');
          STD_TORCH_CHECK(x != std::string::npos && shape.find('x', x + 1) == std::string::npos,
                          "VLLM_MARLIN_TAILPF_BFULL: malformed shape '", shape, "' in '", item, "' (want NxK)");
          r.bfull.push_back(static_cast<int>(kt_tpf_strict("VLLM_MARLIN_TAILPF_BFULL", shape.substr(0, x), 1, 1L << 30)));
          r.bfull.push_back(static_cast<int>(kt_tpf_strict("VLLM_MARLIN_TAILPF_BFULL", shape.substr(x + 1), 1, 1L << 30)));
        }
        item.clear();
      }
    }
    std::string fr = kt_knob("VLLM_MARLIN_TAILPF_BFULL_FRAC", "50", &fr_env);
    if (fr == "0") fr.clear();
    if (!fr.empty()) r.bfull_frac = static_cast<int>(kt_tpf_strict("VLLM_MARLIN_TAILPF_BFULL_FRAC", fr, 1, 100));
    std::string bm = kt_knob("VLLM_MARLIN_TAILPF_BFULL_MAX_MB", "", &bm_env);
    if (bm == "0") bm.clear();
    if (!bm.empty())
      r.bfull_max_bytes = static_cast<long long>(kt_tpf_strict("VLLM_MARLIN_TAILPF_BFULL_MAX_MB", bm, 1, 48)) * (1LL << 20);
    std::string be = kt_knob("VLLM_MARLIN_TAILPF_BFULL_EVICT", "", &be_env);
    if (be == "0") be.clear();
    STD_TORCH_CHECK(be.empty() || be == "normal" || be == "last", "VLLM_MARLIN_TAILPF_BFULL_EVICT: malformed value '", be,
                    "' (want normal or last)");
    r.bfull_evict_last = be == "last" ? 1 : 0;
    const std::string ls = kt_knob("VLLM_MARLIN_TAILPF_LINKTELL", "", &ls_env);
    STD_TORCH_CHECK(ls.empty() || ls == "0" || ls == "1", "VLLM_MARLIN_TAILPF_LINKTELL: malformed value '", ls, "' (want 0 or 1)");
    r.linktell = ls == "1";
    if (r.bfull.empty() && ((fr_env && !fr.empty()) || !bm.empty() || !be.empty()))  // [v1.2] env overrides only (FRAC has a baked 50)
      fprintf(stderr, "[marlin-TAILPF] BFULL IGNORED (0068): _BFULL_FRAC / _BFULL_MAX_MB / _BFULL_EVICT set but "
                      "VLLM_MARLIN_TAILPF_BFULL lists no pair\n");
    if (!r.bfull.empty())
      fprintf(stderr, "[marlin-TAILPF] BFULL ARMED (0068): pairs=%s frac=%d max_mb=%lld evict=%s src=%s\n", bl.c_str(), r.bfull_frac,
              r.bfull_max_bytes >> 20, r.bfull_evict_last ? "last" : "normal", bf_env ? "env" : "default");
    // [0068 v1.2] the effective state of every key, once, even when off
    fprintf(stderr, "[marlin-TAILPF] KNOB (0068) VLLM_MARLIN_TAILPF_BFULL state=%s src=%s value=%s\n", r.bfull.empty() ? "off" : "on",
            bf_env ? "env" : "default", r.bfull.empty() ? "0" : bl.c_str());
    fprintf(stderr, "[marlin-TAILPF] KNOB (0068) VLLM_MARLIN_TAILPF_BFULL_FRAC state=%s src=%s value=%d\n", fr.empty() ? "off" : "on",
            fr_env ? "env" : "default", r.bfull_frac);
    fprintf(stderr, "[marlin-TAILPF] KNOB (0068) VLLM_MARLIN_TAILPF_BFULL_MAX_MB state=%s src=%s value=%lld\n",
            bm.empty() ? "off" : "on", bm_env ? "env" : "default", r.bfull_max_bytes >> 20);
    fprintf(stderr, "[marlin-TAILPF] KNOB (0068) VLLM_MARLIN_TAILPF_BFULL_EVICT state=%s src=%s value=%s\n",
            be.empty() ? "off" : "on", be_env ? "env" : "default", r.bfull_evict_last ? "last" : "normal");
    fprintf(stderr, "[marlin-TAILPF] KNOB (0068) VLLM_MARLIN_TAILPF_LINKTELL state=%s src=%s value=%d\n", r.linktell ? "on" : "off",
            ls_env ? "env" : "default", r.linktell ? 1 : 0);
    if (r.linktell) fprintf(stderr, "[marlin-TAILPF] LINKTELL ARMED (0068)\n");
    return r;
  }();
  return g;
}

// [0043 TAILPF] BEGIN registry (host-only C++ on tailpf_desc.h; tailpf_host_check.cpp compiles this block verbatim)
static const unsigned long long KT_TPF_CAP_UNKNOWN = ~0ULL;  // capture status not known: never learn, never patch
struct KtTpfArgs {  // a launched TPF twin, enough to rebuild its kernel node: the kernel parameters after `tpf`, in order
  const void* func = nullptr;
  unsigned grid = 0, block = 0;
  unsigned smem = 0;
  const int4* A = nullptr;
  const int4* B = nullptr;
  int4* C = nullptr;
  int4* C_tmp = nullptr;
  const int4* bias = nullptr;
  const float* a_s = nullptr;
  const int4* b_s = nullptr;
  const float* g_s = nullptr;
  const int4* zp = nullptr;
  int prob_m = 0, prob_n = 0, prob_k = 0, lda = 0;
  int* locks = nullptr;
  bool has_bias = false, use_atomic_add = false, use_fp32_reduce = false;
  int max_shared_mem = 0, tile_major = 0;
};
struct KtTpfSelf {  // a launch as the prefetch of the launch BEFORE it sees it
  const void* weight = nullptr;  // its B pointer: one per layer and projection
  int n = 0, k = 0, m = 0;
  MarlinTpfDesc desc{};
  // [0068 TPFOUT] its whole-B form (kt_tpf_bfull_desc), used instead of desc when the predecessor's shape is a listed carrier
  MarlinTpfDesc desc_bfull{};
  bool has_bfull = false;
  bool desc_is_bfull = false;  // set on the registry's stored successor: its desc IS the whole-B form
};
struct KtTpfPatch {  // re-set the predecessor's kernel node `node` to `args` with descriptor `desc`
  void* node = nullptr;
  KtTpfArgs args;
  MarlinTpfDesc desc{};
  int xn = 0, xk = 0, yn = 0, yk = 0;
};
struct KtTpfLearn {  // the successor this launch taught (for the NEXT tell)
  bool fresh = false;  // first time this (shape -> next shape) pair is seen
  bool captured = false;
  int xn = 0, xk = 0, yn = 0, yk = 0, m = 0;
  MarlinTpfDesc desc{};
  bool bfull = false;          // [0068 TPFOUT] desc is the successor's whole-B form
  bool carried_bfull = false;  // [0068 TPFOUT] the descriptor on_launch RETURNED is a whole-B form (set on every call)
};
static bool kt_tpf_same(const MarlinTpfDesc& a, const MarlinTpfDesc& b) {
  return a.s == b.s && a.zp == b.zp && a.b == b.b && a.s_lines == b.s_lines && a.zp_lines == b.zp_lines &&
         a.evict_last == b.evict_last && a.grid == b.grid && a.k_tiles == b.k_tiles && a.n_tiles == b.n_tiles &&
         a.part1_tiles == b.part1_tiles && a.p2_flat == b.p2_flat && a.iters == b.iters && a.p1_q == b.p1_q &&
         a.p1_r == b.p1_r && a.tile_major == b.tile_major && a.row_int4 == b.row_int4 && a.rows == b.rows &&
         a.gl_stride == b.gl_stride && a.stages == b.stages;
}
// The descriptor of a launch (what its PREDECESSOR prefetches): its scale / zero-point byte ranges as 128-B line spans (none if
// together above max_bytes) and, mode 2, its B and partition (only when every B stage row is whole 128-B lines). evict: 0
// normal, 1 last, 2 auto = last when this launch streams more than 48 MiB of B (its own B stream would evict normal-class lines).
static MarlinTpfDesc kt_tpf_self_desc(int mode, int b_stages, long long max_bytes, int evict, const void* s,
                                      long long s_bytes, const void* zp, long long zp_bytes, const void* B, int prob_m,
                                      int prob_n, int prob_k, int m_block_size, bool fold, int thread_k_blocks,
                                      int thread_n_blocks, int group_blocks, int grid, int tile_major_bit, bool a8,
                                      int num_bits) {
  MarlinTpfDesc d{};
  const long long b_bytes = static_cast<long long>(prob_k) * prob_n * num_bits / 8;
  d.evict_last = evict == 1 || (evict == 2 && b_bytes > (48LL << 20)) ? 1 : 0;
  auto span = [](const void* p, long long bytes, const int4** base, int* lines) {
    if (p == nullptr || bytes <= 0) return;
    const unsigned long long lo = reinterpret_cast<unsigned long long>(p) & ~127ULL;
    const unsigned long long hi = reinterpret_cast<unsigned long long>(p) + static_cast<unsigned long long>(bytes);
    *base = reinterpret_cast<const int4*>(lo);
    *lines = static_cast<int>((hi - lo + 127) / 128);
  };
  if (s_bytes + zp_bytes <= max_bytes) {
    span(s, s_bytes, &d.s, &d.s_lines);
    span(zp, zp_bytes, &d.zp, &d.zp_lines);
  }
  if (mode == 2 && b_stages > 0 && B != nullptr && num_bits > 0 && thread_k_blocks > 0 && thread_n_blocks > 0 &&
      grid > 0) {
    const int pack = 32 / num_bits;
    const int row_int4 = ((thread_n_blocks * 16) * 16 / pack) / (a8 ? 2 : 4);
    const long long gl_stride = 16LL * prob_n / (pack * (a8 ? 2 : 4));
    const int rows = thread_k_blocks / (a8 ? 2 : 1);
    if ((reinterpret_cast<unsigned long long>(B) & 127ULL) == 0 && row_int4 > 0 && row_int4 % 8 == 0 &&
        gl_stride % 8 == 0 && gl_stride < (1LL << 31) && rows > 0 && prob_k % (16 * thread_k_blocks) == 0 &&
        prob_n % (16 * thread_n_blocks) == 0) {
      MarlinTpfDesc e = d;
      e.b = reinterpret_cast<const int4*>(B);
      e.row_int4 = row_int4;
      e.rows = rows;
      e.gl_stride = static_cast<int>(gl_stride);
      e.stages = b_stages;
      marlin_tpf_partition(e, prob_m, prob_n, prob_k, m_block_size, fold, thread_k_blocks, thread_n_blocks, group_blocks,
                           grid, tile_major_bit);
      // the device index math is 32-bit (tailpf_desc.h contract): f = iters * yb < 2^31 for every yb < grid, the column
      // c + part1_tiles <= p2_flat + part1_tiles < 2^31, the line index < stages * rows * lpr < 2^31; otherwise no B part
      // (the scale / zp part stays)
      const long long lim = 1LL << 31;
      if (e.k_tiles > 0 && e.n_tiles > 0 && e.p2_flat >= 0 && e.part1_tiles >= 0 &&
          static_cast<long long>(e.iters) * grid < lim && static_cast<long long>(e.part1_tiles) + e.p2_flat < lim &&
          static_cast<long long>(e.stages) * e.rows * (row_int4 / 8) < lim) {
        marlin_tpf_finish(e);
        d = e;
      }
    }
  }
  return d;
}
// [0068 TPFOUT] The whole-B form of a launch's descriptor: `base` (its scale / zp spans) plus a B part that covers EVERY 128-B
// line of its B through the UNCHANGED device walk (tpf_prefetch_next): a virtual partition with no part 1 whose stripes are
// chunks of `iters` k-tiles of one column tile (iters = k_tiles when k_tiles <= 16, else the largest divisor of k_tiles in
// [4, 16]), so marlin_tpf_first_tile never clips a stripe at a column end: vCTA yb = column yb / (k_tiles / iters), first
// k-tile (yb % (k_tiles / iters)) * iters, grid = k_tiles * n_tiles / iters vCTAs. stages = max(1, iters * frac / 100): the
// first `stages` k-tiles of every chunk (frac 100 = all of B). Row / line geometry and eligibility are mode 2's (row_int4,
// rows, gl_stride from the launch's own tile; 128-B aligned B, whole-line rows, exact tiles), plus B <= max_b_bytes and
// num_bits > 0 (an int3 word has no B-line formula). [0073] Every line of the descriptor gets class base.evict_last | evict_last
// (one class per descriptor: BFULL_EVICT=last also raises the s/zp lines; TAIL_PREFETCH_EVICT=last is never lowered). Ineligible: b ==
// nullptr, grid == 0 and *why names the failed test. The device index math stays 32-bit (p2_flat and stages * rows * lpr <
// 2^31, checked).
static MarlinTpfDesc kt_tpf_bfull_desc(const MarlinTpfDesc& base, long long max_b_bytes, int frac, int evict_last,
                                       const void* B, int prob_n, int prob_k, int thread_k_blocks, int thread_n_blocks,
                                       bool a8, int num_bits, const char** why) {
  MarlinTpfDesc d = base;
  d.b = nullptr;
  d.grid = 0;
  if (B == nullptr || num_bits <= 0 || thread_k_blocks <= 0 || thread_n_blocks <= 0 || frac < 1 || frac > 100) {
    *why = "no-b-line-formula";
    return d;
  }
  const long long b_bytes = static_cast<long long>(prob_k) * prob_n * num_bits / 8;
  if (b_bytes > max_b_bytes) {
    *why = "over-max-mb";
    return d;
  }
  const int pack = 32 / num_bits;
  const int row_int4 = ((thread_n_blocks * 16) * 16 / pack) / (a8 ? 2 : 4);
  const long long gl_stride = 16LL * prob_n / (pack * (a8 ? 2 : 4));
  const int rows = thread_k_blocks / (a8 ? 2 : 1);
  if ((reinterpret_cast<unsigned long long>(B) & 127ULL) != 0 || row_int4 <= 0 || row_int4 % 8 != 0 || gl_stride % 8 != 0 ||
      gl_stride >= (1LL << 31) || rows <= 0 || prob_k % (16 * thread_k_blocks) != 0 || prob_n % (16 * thread_n_blocks) != 0) {
    *why = "layout";
    return d;
  }
  const int k_tiles = prob_k / 16 / thread_k_blocks, n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = k_tiles <= 16 ? k_tiles : 0;
  for (int c = 16; iters == 0 && c >= 4; --c)
    if (k_tiles % c == 0) iters = c;
  if (iters <= 0) {
    *why = "no-chunk";
    return d;
  }
  int stages = iters * frac / 100;
  if (stages < 1) stages = 1;
  const long long p2_flat = static_cast<long long>(k_tiles) * n_tiles;
  if (p2_flat >= (1LL << 31) || static_cast<long long>(stages) * rows * (row_int4 / 8) >= (1LL << 31)) {
    *why = "index-32bit";
    return d;
  }
  d.b = reinterpret_cast<const int4*>(B);
  d.evict_last = base.evict_last | evict_last;   // [0073 BFEVICT] MRL-08: keep the base (TAIL_PREFETCH_EVICT) class of the successor's s/zp lines; BFULL_EVICT can only raise it
  d.grid = static_cast<int>(p2_flat / iters);
  d.k_tiles = k_tiles;
  d.n_tiles = n_tiles;
  d.part1_tiles = 0;
  d.p2_flat = static_cast<int>(p2_flat);
  d.iters = iters;
  d.p1_q = 0;
  d.p1_r = 0;
  d.tile_major = 0;
  d.row_int4 = row_int4;
  d.rows = rows;
  d.gl_stride = static_cast<int>(gl_stride);
  d.stages = stages;
  marlin_tpf_finish(d);
  *why = "ok";
  return d;
}
class KtTpfRegistry {
 public:
  // Called for EVERY Marlin launch while armed, before it is launched; `self` = this launch as its predecessor sees it, `cap` =
  // the capture id of its stream (0 = eager, KT_TPF_CAP_UNKNOWN = not queried). When the previous launch on this stream
  // belongs to the same capture sequence (or both are eager), self is its successor: learned for eager use (eager pairs only)
  // and, if that previous launch is a captured TPF twin whose node carries another descriptor, *patch is filled. Returns the
  // descriptor THIS launch carries: eager = its last learned (eager) successor, all-null while unknown; captured = all-null
  // until the next launch of the same capture patches it (a graph never replays a learned, possibly stale, successor: its last
  // Marlin node prefetches nothing). *next_n / *next_k = the shape of the successor carried (0 = none).
  // [0068 TPFOUT] The successor's descriptor is its whole-B form (self.desc_bfull) iff it has one and set_bfull listed the pair
  // (this predecessor's shape -> its shape); otherwise self.desc, exactly as 0043. learn->carried_bfull says whether the
  // RETURNED descriptor is a whole-B form.
  MarlinTpfDesc on_launch(const void* stream, unsigned long long cap, const KtTpfSelf& self, KtTpfPatch* patch,
                          KtTpfLearn* learn, int* next_n, int* next_k) {
    std::lock_guard<std::mutex> lk(mu_);
    Pending& p = prev_[stream];
    // [0068 TPFOUT] link tell: the last launch on this stream was a captured twin of ANOTHER capture that no launch of its own
    // capture followed = that capture's last Marlin node (it prefetches nothing)
    if (linktell_ && p.valid && p.twin && p.cap != cap && p.cap != 0 && p.cap != KT_TPF_CAP_UNKNOWN)
      link_tell(p.cap, p.n, p.k, 0, 0, nullptr, false, false);
    if (p.valid && p.cap == cap && cap != KT_TPF_CAP_UNKNOWN) {
      // [0068 TPFOUT] the successor's whole-B form iff the gate lists (this predecessor's shape -> its shape)
      const bool bfull = self.has_bfull && bfull_listed(p.n, p.k, self.n, self.k);
      const MarlinTpfDesc& sd = bfull ? self.desc_bfull : self.desc;
      if (cap == 0) {
        succ_[p.weight] = self;
        succ_[p.weight].desc = sd;
        succ_[p.weight].desc_is_bfull = bfull;
      }
      const std::pair<std::pair<int, int>, std::pair<int, int>> key{{p.n, p.k}, {self.n, self.k}};
      if (pairs_.emplace(key, true).second) {
        learn->fresh = true;
        learn->captured = cap != 0;
        learn->xn = p.n;
        learn->xk = p.k;
        learn->yn = self.n;
        learn->yk = self.k;
        learn->m = self.m;
        learn->desc = sd;
        learn->bfull = bfull;
      }
      if (p.twin && cap != 0 && p.node != nullptr && !kt_tpf_same(p.baked, sd)) {
        patch->node = p.node;
        patch->args = p.args;
        patch->desc = sd;
        patch->xn = p.n;
        patch->xk = p.k;
        patch->yn = self.n;
        patch->yk = self.k;
        p.baked = sd;
      }
      if (linktell_ && p.twin && cap != 0) link_tell(cap, p.n, p.k, self.n, self.k, &sd, bfull, p.node != nullptr);
    }
    MarlinTpfDesc d{};
    *next_n = 0;
    *next_k = 0;
    const auto it = cap == 0 ? succ_.find(self.weight) : succ_.end();
    learn->carried_bfull = false;
    if (it != succ_.end()) {
      d = it->second.desc;
      *next_n = it->second.n;
      *next_k = it->second.k;
      learn->carried_bfull = it->second.desc_is_bfull;
    }
    p = Pending{};
    p.valid = true;
    p.weight = self.weight;
    p.n = self.n;
    p.k = self.k;
    p.cap = cap;
    p.baked = d;
    return d;
  }
  // After a TPF twin was launched for the launch on_launch last saw on this stream: its args and (captured) kernel node.
  void on_twin_launched(const void* stream, const KtTpfArgs& args, void* node) {
    std::lock_guard<std::mutex> lk(mu_);
    Pending& p = prev_[stream];
    p.twin = true;
    p.args = args;
    p.node = node;
  }
  // [0068 TPFOUT] the gate's whole-B pairs (pn, pk, sn, sk per 4 ints) and link-tell switch; a registry never set behaves as 0043
  void set_bfull(const std::vector<int>& pairs) {
    std::lock_guard<std::mutex> lk(mu_);
    bfull_ = pairs;
  }
  void set_linktell(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    linktell_ = on;
  }

 private:
  bool bfull_listed(int pn, int pk, int sn, int sk) const {
    for (size_t i = 0; i + 3 < bfull_.size(); i += 4)
      if (bfull_[i] == pn && bfull_[i + 1] == pk && bfull_[i + 2] == sn && bfull_[i + 3] == sk) return true;
    return false;
  }
  // One tell per (capture, carrier shape, successor shape); d == nullptr: the carrier is its capture's last Marlin node.
  // Called with mu_ held.
  void link_tell(unsigned long long cap, int pn, int pk, int sn, int sk, const MarlinTpfDesc* d, bool bfull, bool node) {
    char key[96];
    snprintf(key, sizeof(key), "%llu/%d/%d/%d/%d", cap, pn, pk, sn, sk);
    if (!links_.emplace(key, true).second) return;
    if (d == nullptr) {
      fprintf(stderr, "[marlin-TAILPF] LINK cap=%llu %dx%d -> none (last Marlin node of its capture: prefetches nothing)\n", cap,
              pn, pk);
      return;
    }
    char b[64];
    const double b_mb = d->b == nullptr ? 0.0
                                        : static_cast<double>(d->grid) * d->stages * d->rows * d->div_lpr.d * 128.0 / (1 << 20);
    if (d->b == nullptr)
      snprintf(b, sizeof(b), "off");
    else if (bfull)
      snprintf(b, sizeof(b), "full %.2fMB (%d/%d k-tiles per chunk)", b_mb, d->stages, d->iters);
    else
      snprintf(b, sizeof(b), "stages %d (<= %.2fMB)", d->stages, b_mb);
    fprintf(stderr, "[marlin-TAILPF] LINK cap=%llu %dx%d -> %dx%d s+zp=%.2fMB b=%s evict=%s%s\n", cap, pn, pk, sn, sk,
            (d->s_lines + d->zp_lines) * 128.0 / (1 << 20), b, d->evict_last ? "last" : "normal",
            node ? "" : " NOT PATCHED (no single kernel node)");
  }
  struct Pending {  // the last launch on a stream
    bool valid = false;
    const void* weight = nullptr;
    int n = 0, k = 0;
    unsigned long long cap = 0;
    bool twin = false;
    void* node = nullptr;
    KtTpfArgs args;
    MarlinTpfDesc baked{};  // the descriptor its launch (or last patch) carries
  };
  std::mutex mu_;
  std::map<const void*, KtTpfSelf> succ_;
  std::map<const void*, Pending> prev_;
  std::map<std::pair<std::pair<int, int>, std::pair<int, int>>, bool> pairs_;
  std::vector<int> bfull_;               // [0068 TPFOUT]
  bool linktell_ = false;                // [0068 TPFOUT]
  std::map<std::string, bool> links_;    // [0068 TPFOUT] link tells already printed
};
// [0043 TAILPF] END registry
static KtTpfRegistry& kt_tpf_registry() {
  static KtTpfRegistry r;
  // [0068 TPFOUT] the gate's whole-B pairs and link-tell switch, once (every caller runs with TAILPF armed)
  static const bool kt_tpf_registry_init = [] {
    const KtTpfGate& g = kt_tpf_gate();
    r.set_bfull(g.bfull);
    r.set_linktell(g.linktell);
    return true;
  }();
  (void)kt_tpf_registry_init;
  return r;
}
static bool kt_tpf_first(const std::string& key) {  // true the first time `key` is seen (tells once)
  static std::mutex mu;
  static std::map<std::string, bool> told;
  std::lock_guard<std::mutex> lk(mu);
  return told.emplace(key, true).second;
}
static std::mutex kt_tpf_api_mu;
static bool kt_tpf_api_on = true;  // (b) capture-node patching; off after the first failing capture API call
static bool kt_tpf_api() {
  std::lock_guard<std::mutex> lk(kt_tpf_api_mu);
  return kt_tpf_api_on;
}
static void kt_tpf_api_fail(const char* api, cudaError_t err) {
  (void)cudaGetLastError();  // clear OUR error: callers only reach this with no other error pending before their call
  std::lock_guard<std::mutex> lk(kt_tpf_api_mu);
  if (kt_tpf_api_on) {
    kt_tpf_api_on = false;
    fprintf(stderr, "[marlin-TAILPF] CAPTURE-PATCH OFF: %s failed (%s); learned successors only\n", api,
            cudaGetErrorString(err));
  }
}
static double kt_tpf_mb(const MarlinTpfDesc& d) { return (d.s_lines + d.zp_lines) * 128.0 / (1 << 20); }
static void kt_tpf_next_str(char* out, size_t len, const MarlinTpfDesc& d, int nn, int nk, bool bfull) {
  if (nn == 0) {
    snprintf(out, len, "unknown");
    return;
  }
  char b[64];
  if (d.b != nullptr && bfull)  // [0068 TPFOUT] whole-B form: every vCTA prefetches `stages` of its `iters` k-tiles
    snprintf(b, sizeof(b), "full(%.2fMB,%d/%d)",
             static_cast<double>(d.grid) * d.stages * d.rows * d.div_lpr.d * 128.0 / (1 << 20), d.stages, d.iters);
  else if (d.b != nullptr)
    snprintf(b, sizeof(b), "%d", d.stages);
  else
    snprintf(b, sizeof(b), "off");
  snprintf(out, len, "%dx%d s+zp=%.2fMB%s b_stages=%s", nn, nk, kt_tpf_mb(d), d.s != nullptr ? "" : "(over max_mb: none)",
           b);
}
// One armed launch: registry, graph-node patch of the predecessor, the twin launch (true) or nothing (false: the caller runs
// the served launch). attr_smem = the attribute value (the full grant), smem = the launch's dynamic smem ([0038 TRIM]).
// Resident CTAs per SM of `fn` at `threads` threads and `smem` dynamic bytes (cudaOccupancyMaxActiveBlocksPerMultiprocessor),
// cached per (fn, threads, smem); a failed query clears its own error and counts 0.
static int kt_tpf_occ(const void* fn, int threads, int smem) {
  static std::mutex mu;
  static std::map<std::pair<std::uintptr_t, long long>, int> cache;
  const auto key = std::make_pair(reinterpret_cast<std::uintptr_t>(fn), (static_cast<long long>(threads) << 32) | smem);
  std::lock_guard<std::mutex> lk(mu);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  int occ = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, static_cast<size_t>(smem)) != cudaSuccess) {
    (void)cudaGetLastError();
    occ = 0;
  }
  cache.emplace(key, occ);
  return occ;
}
// Whether this launch runs its TPF twin: the twin exists, its N is listed, and it keeps the residency of the kernel it replaces
// at this launch's block size and dynamic smem ([0046] BPSM2 doubles the grid only when that kernel keeps two CTAs per SM).
// Equal registers / stack are the twin screen's bar (tpf_twin_check), so on a screened build this never declines; it is the
// guard, not the plan. A decline keeps the kernel and tells once per (N, K, twin, smem).
static bool kt_tpf_runs(MarlinTpfFuncPtr twin, bool n_ok, MarlinFuncPtr kernel, int threads, int attr_smem, int smem, int n,
                        int k) {
  if (twin == nullptr || !n_ok) return false;
  cudaFuncSetAttribute(twin, cudaFuncAttributeMaxDynamicSharedMemorySize, attr_smem);
  const int occ_t = kt_tpf_occ(reinterpret_cast<const void*>(twin), threads, smem);
  const int occ_k = kt_tpf_occ(reinterpret_cast<const void*>(kernel), threads, smem);
  if (occ_t >= occ_k) return true;
  char key[160];
  snprintf(key, sizeof(key), "keep/%d/%d/%p/%d", n, k, reinterpret_cast<void*>(twin), smem);
  if (kt_tpf_first(key))
    fprintf(stderr, "[marlin-TAILPF] KEEP N=%d K=%d: twin occupancy %d < kernel %d at %d threads, smem %d: launch keeps its kernel\n",
            n, k, occ_t, occ_k, threads, smem);
  return false;
}
static bool kt_tpf_launch(MarlinTpfFuncPtr twin, const char* twin_name, bool n_ok, cudaStream_t stream, int blocks,
                          int threads, int attr_smem, int smem, const KtTpfSelf& self, KtTpfArgs a, int a_bits, int w_bits,
                          int tmb, int tnb, int tkb, int gb) {
  unsigned long long cap = KT_TPF_CAP_UNKNOWN;
  if (cudaPeekAtLastError() == cudaSuccess) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    const cudaError_t e = cudaStreamGetCaptureInfo(stream, &st, &id);
    if (e != cudaSuccess)
      kt_tpf_api_fail("cudaStreamGetCaptureInfo", e);
    else if (st == cudaStreamCaptureStatusNone)
      cap = 0;
    else if (st == cudaStreamCaptureStatusActive)
      cap = id;
  }
  KtTpfPatch patch;
  KtTpfLearn learn;
  int nn = 0, nk = 0;
  const MarlinTpfDesc d = kt_tpf_registry().on_launch(stream, cap, self, &patch, &learn, &nn, &nk);
  if (learn.fresh) {
    char nx[160];
    kt_tpf_next_str(nx, sizeof(nx), learn.desc, learn.yn, learn.yk, learn.bfull);
    fprintf(stderr, "[marlin-TAILPF] NEXT N=%d K=%d -> %s (M=%d) via=%s\n", learn.xn, learn.xk, nx, learn.m,
            learn.captured ? "capture" : "eager");
  }
  if (patch.node != nullptr && kt_tpf_api() && cudaPeekAtLastError() == cudaSuccess) {
    KtTpfArgs& q = patch.args;
    MarlinTpfDesc pd = patch.desc;
    void* kp[] = {&pd,        &q.A,     &q.B,     &q.C,      &q.C_tmp,    &q.bias,           &q.a_s,
                  &q.b_s,     &q.g_s,   &q.zp,    &q.prob_m, &q.prob_n,   &q.prob_k,         &q.lda,
                  &q.locks,   &q.has_bias, &q.use_atomic_add, &q.use_fp32_reduce, &q.max_shared_mem, &q.tile_major};
    cudaKernelNodeParams np = {};
    np.func = const_cast<void*>(q.func);
    np.gridDim = dim3(q.grid);
    np.blockDim = dim3(q.block);
    np.sharedMemBytes = q.smem;
    np.kernelParams = kp;
    np.extra = nullptr;
    const cudaError_t e = cudaGraphKernelNodeSetParams(static_cast<cudaGraphNode_t>(patch.node), &np);
    if (e != cudaSuccess)
      kt_tpf_api_fail("cudaGraphKernelNodeSetParams", e);
    else if (kt_tpf_first("patch"))
      fprintf(stderr, "[marlin-TAILPF] PATCH first graph-node patch (capture %llu): N=%d K=%d -> next N=%d K=%d\n", cap,
              patch.xn, patch.xk, patch.yn, patch.yk);
  }
  if (twin == nullptr || !n_ok) {
    char key[160];
    snprintf(key, sizeof(key), "no-twin/%d/%d/%d/%d/%d/%d/%d/%d", self.n, self.k, a_bits, w_bits, threads, tmb, tnb, tkb);
    if (twin == nullptr && kt_tpf_first(key))
      fprintf(stderr,
              "[marlin-TAILPF] NO-TWIN N=%d K=%d (first M=%d) a%d w%d thr=%d tmb=%d tnb=%d tkb=%d gb=%d: launch keeps its "
              "kernel\n",
              self.n, self.k, self.m, a_bits, w_bits, threads, tmb, tnb, tkb, gb);
    return false;
  }
  cudaFuncSetAttribute(twin, cudaFuncAttributeMaxDynamicSharedMemorySize, attr_smem);
#if MARLIN_G2
  kt_g2_launch(reinterpret_cast<const void*>(twin), stream);  // [0071 G2] arm the twin's module (end of file); #line: served numbering
#endif
#line 1657
  twin<<<blocks, threads, smem, stream>>>(d, a.A, a.B, a.C, a.C_tmp, a.bias, a.a_s, a.b_s, a.g_s, a.zp, a.prob_m, a.prob_n,
                                          a.prob_k, a.lda, a.locks, a.has_bias, a.use_atomic_add, a.use_fp32_reduce,
                                          a.max_shared_mem, a.tile_major);
  void* node = nullptr;
  if (cap != 0 && cap != KT_TPF_CAP_UNKNOWN && kt_tpf_api() && cudaPeekAtLastError() == cudaSuccess) {
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    size_t nd = 0;
    cudaError_t e = cudaStreamGetCaptureInfo(stream, &st, &id, nullptr, &deps, &edges, &nd);
    if (e != cudaSuccess) {
      kt_tpf_api_fail("cudaStreamGetCaptureInfo(deps)", e);
    } else if (st == cudaStreamCaptureStatusActive && id == cap && nd == 1) {
      const cudaGraphNode_t dn = deps[0];  // copied before any further API call on the stream
      cudaGraphNodeType ty = cudaGraphNodeTypeEmpty;
      e = cudaGraphNodeGetType(dn, &ty);
      if (e != cudaSuccess)
        kt_tpf_api_fail("cudaGraphNodeGetType", e);
      else if (ty == cudaGraphNodeTypeKernel)
        node = static_cast<void*>(dn);
    }
  }
  a.func = reinterpret_cast<const void*>(twin);
  a.grid = static_cast<unsigned>(blocks);
  a.block = static_cast<unsigned>(threads);
  a.smem = static_cast<unsigned>(smem);
  kt_tpf_registry().on_twin_launched(stream, a, node);
  char key[160];
  snprintf(key, sizeof(key), "on/%d/%d/%d/%d/%s", self.n, self.k, a_bits, w_bits, twin_name);
  if (kt_tpf_first(key)) {
    char nx[160];
    kt_tpf_next_str(nx, sizeof(nx), d, nn, nk, learn.carried_bfull);
    fprintf(stderr, "[marlin-TAILPF] ON N=%d (first M=%d) K=%d a%d w%d twin=%s next=%s via=%s\n", self.n, self.m, self.k,
            a_bits, w_bits, twin_name, nx, cap == 0 ? "eager" : cap == KT_TPF_CAP_UNKNOWN ? "unknown" : "capture");
  }
  return true;
}
#endif

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_zp, bool is_zp_float, int is_a_8bit,
    int stages, int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    if (KT_CFG_INDEX >= 0 && i != KT_CFG_INDEX) continue;  // [club-kt] sweep: one entry only
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_zp, is_zp_float, is_a_8bit,
                         stages, max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(
        th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits,
        group_size, has_zp, is_zp_float, is_a_8bit, stages);

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks,
        th_config.thread_n / 16, th_config.thread_k / 16, m_block_size_8,
        has_zp, group_blocks, th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

// [cuda-lane 2026-09-19] THE ONE-PASS CEILING, in ONE place because it has TWO consumers.
// 4 m-blocks = 64 rows; above that marlin_mm splits M and every extra pass repeats the dequant for
// the whole weight matrix (measured: sub-L2 units still cost 1.57-1.70x at M=88 despite fitting L2).
// ⚠ THE SECOND CONSUMER: the fp32-reduce scratch c_tmp is sized from "the largest M block (64 rows)"
// -- min(..., 64) and sms*64*max_thread_n. Raising the ceiling without raising that overruns the
// buffer by 16*(tmb-4) rows per SM, and USE_FP32_REDUCE_DEFAULT is True, so it is a LIVE overflow.
// Capped at 6: tmb=8 spills 2502 B/thread inside the k-loop. Unset / empty leaves the served 4; any other value must be an
// integer in [1, 6] or the first GEMM refuses ([0072]), so VLLM_MARLIN_MAX_TMB='' cannot silently become 1.
inline int kt_max_thread_m_blocks() {
  static const int v = [] {
    const char* e = std::getenv("VLLM_MARLIN_MAX_TMB");
    if (e != nullptr && *e != '\0') return static_cast<int>(kt_tpf_strict("VLLM_MARLIN_MAX_TMB", e, 1, 6));   // [0072 STRICTPARSE] MRL-09: was atoi (1e1 -> 1, 16 -> silent 4)
    return 4;
  }();
  return v;
}

#if MARLIN_L2PF
// [0045 L2PF] Run switch VLLM_MARLIN_L2PF, read once per process: unset/""/0 = OFF (every launch runs the kernel the earlier
// selections chose; one untaken branch on a load-time constant). 1 = the launch runs the L2PF twin of that kernel (Marlin ->
// MarlinL2PF, MarlinXS -> MarlinXSL2PF, MarlinOPR -> MarlinOPRL2PF, MarlinXSOPR -> MarlinXSOPRL2PF, MarlinSP -> MarlinSPL2PF,
// ... MarlinXSOPRSP -> MarlinXSOPRSPL2PF; no twin = keep): the same kernel with the .L2::256B prefetch-size hint on its
// B-stream cp.async (marlin.cuh cp_async4_stream_l2pf), so it computes exactly what that kernel computes. Anything else is a
// hard error at the first Marlin GEMM (never a silent OFF). VLLM_MARLIN_L2PF_N: comma list of N (unset, empty or "*" = every
// N; each entry 1..2^30 in plain decimal), parsed strictly at the first armed launch.
// Tells (stderr), printed after every grid / smem decision of the launch: "[marlin-L2PF] ARMED ..." once per process;
// "[marlin-L2PF] ON N=<n> K=<k> M=<m> a<bits> w<bits> twin=<name>" once per (N, K, a/w bits, twin), M = the rows of the first
// such launch; "[marlin-L2PF] NO-TWIN N=<n> K=<k> M=<m> ..." once per (N, K, a/w bits, config) for a listed launch whose kernel
// has no twin: it keeps its kernel (expected for every config outside L2PF_TWINS, i.e. M <= 16 and M > 64, and for
// MarlinFOLD); with 0043 TAILPF in the tree "[marlin-L2PF] TAILPF-RUNS N=<n> K=<k> M=<m> twin=<TPF twin>" once per (N, K,
// twin) for a listed launch that TAILPF runs as its TPF twin (the alternative on that launch: no L2PF).
static int kt_l2pf_mode() {
  const char* v = getenv("VLLM_MARLIN_L2PF");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if (v[0] == '1' && v[1] == '\0') return 1;
  return -1;
}
static const int KT_L2PF = kt_l2pf_mode();

struct KtL2pfGate {
  bool all_n = true;
  std::vector<int> ns;
};
static int kt_l2pf_strict(const std::string& v) {
  bool digits = !v.empty() && v.size() <= 10;
  for (char c : v)
    if (c < '0' || c > '9') digits = false;
  const long long x = digits ? std::stoll(v) : 0;
  STD_TORCH_CHECK(digits && x >= 1 && x <= (1LL << 30), "VLLM_MARLIN_L2PF_N: malformed value '", v,
                  "' (want a positive decimal integer <= 2^30)");
  return static_cast<int>(x);
}
static const KtL2pfGate& kt_l2pf_gate() {
  static const KtL2pfGate g = [] {
    const char* mode = getenv("VLLM_MARLIN_L2PF");
    STD_TORCH_CHECK(KT_L2PF > 0, "VLLM_MARLIN_L2PF: malformed value '", mode != nullptr ? mode : "", "' (want 0 or 1)");
    KtL2pfGate r;
    const char* s = getenv("VLLM_MARLIN_L2PF_N");
    const std::string list = s != nullptr ? s : "";
    if (!list.empty() && list != "*") {
      r.all_n = false;
      std::string cur;
      for (size_t i = 0; i <= list.size(); ++i) {
        const char ch = i < list.size() ? list[i] : ',';
        if (ch == ',') {
          r.ns.push_back(kt_l2pf_strict(cur));
          cur.clear();
        } else if (ch != ' ') {
          cur.push_back(ch);
        }
      }
    }
    fprintf(stderr, "[marlin-L2PF] ARMED VLLM_MARLIN_L2PF=%d n=%s b_stream=cp.async.L2::256B\n", KT_L2PF,
            r.all_n ? "*" : list.c_str());
    return r;
  }();
  return g;
}
// twin == nullptr: the NO-TWIN tell (keyed by the launch config).
static void kt_l2pf_tell(const char* twin, int m, int n, int k, int a_bits, int b_bits, int threads, int tmb, int tnb,
                         int tkb, int gb) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[128];
  if (twin != nullptr)
    snprintf(key, sizeof(key), "on/%d/%d/a%d/w%d/%s", n, k, a_bits, b_bits, twin);
  else
    snprintf(key, sizeof(key), "no/%d/%d/a%d/w%d/%d/%d/%d/%d/%d", n, k, a_bits, b_bits, threads, tmb, tnb, tkb, gb);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  if (twin != nullptr)
    fprintf(stderr, "[marlin-L2PF] ON N=%d K=%d M=%d a%d w%d twin=%s\n", n, k, m, a_bits, b_bits, twin);
  else
    fprintf(stderr,
            "[marlin-L2PF] NO-TWIN N=%d K=%d M=%d a%d w%d thr=%d tmb=%d tnb=%d tkb=%d gb=%d: launch keeps its kernel\n", n,
            k, m, a_bits, b_bits, threads, tmb, tnb, tkb, gb);
}
  #if defined(MARLIN_TAILPF) && MARLIN_TAILPF
// [0045 L2PF x 0043 TAILPF] a launch the TAILPF launch block runs as its TPF twin is not an L2PF launch: once per (N, K, twin).
static void kt_l2pf_tell_tpf(int m, int n, int k, const char* tpf_twin) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[128];
  snprintf(key, sizeof(key), "%d/%d/%s", n, k, tpf_twin);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  fprintf(stderr, "[marlin-L2PF] TAILPF-RUNS N=%d K=%d M=%d twin=%s: the TAILPF twin runs this launch (no L2PF)\n", n, k, m,
          tpf_twin);
}
  #endif

  #include "kernel_selector_l2pf.h"
#endif

// [0047 ENGAGE-TELL cuda 10-06] BEGIN static (self-contained: w8so/e47/test_0047_host.sh compiles this block alone).
// VLLM_MARLIN_ENGAGE_TELL=1 prints, once per (kernel, N, K, M), the exact template of the kernel a Marlin launch RUNS -- read at
// the launch line, after every twin selection (XS, BPSM2, OPR / XSOPR, SP, FOLD, and any later one), so it names what the GPU
// executes:  "[marlin-ENGAGE] <Family><a,b,c,s,threads,tmb,tnb,tkb,m8,stages,gb,zpf> N=<n> K=<k> M=<m>"  (M = the GEMM's size_m).
// The strings come from kernel_engage_names.h, which generate_kernels.py writes from the kernel expressions of every generated
// kernel_selector*.h (each selector's own text, under the same MARLIN_* conditions). A kernel without an entry prints
// "[marlin-ENGAGE] UNNAMED kernel=<ptr> N= K= M=" (a KAT reader counts it as a mismatch). Read once per process:
// unset/""/0 = OFF -- a load-time constant 0 and one untaken branch per launch, nothing else (no lookup, no lock); 1 = on;
// anything else is a hard error at the first Marlin GEMM (never a silent OFF). HOST ONLY: no kernel changes. A KAT diagnostic:
// never set on a record boot.
static int kt_engage_mode() {
  const char* v = getenv("VLLM_MARLIN_ENGAGE_TELL");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if (v[0] == '1' && v[1] == '\0') return 1;
  return -1;
}
static const int KT_ENGAGE_TELL = kt_engage_mode();
struct KtEngageName {
  MarlinFuncPtr fn;
  const char* name;
};
#include "kernel_engage_names.h"
struct KtEngageKey {
  MarlinFuncPtr fn;
  int n, k, m;
};
static void kt_engage_tell(MarlinFuncPtr kernel, int n, int k, int m) {
  const char* mode = getenv("VLLM_MARLIN_ENGAGE_TELL");
  STD_TORCH_CHECK(KT_ENGAGE_TELL == 1, "VLLM_MARLIN_ENGAGE_TELL: malformed value '", mode != nullptr ? mode : "",
                  "' (want 0 or 1)");
  static std::mutex told_mu;
  static std::vector<KtEngageKey> told;
  std::lock_guard<std::mutex> lk(told_mu);
  for (const KtEngageKey& t : told)
    if (t.fn == kernel && t.n == n && t.k == k && t.m == m) return;
  told.push_back(KtEngageKey{kernel, n, k, m});
  for (const KtEngageName& e : kt_engage_names)
    if (e.fn != nullptr && e.fn == kernel) {
      fprintf(stderr, "[marlin-ENGAGE] %s N=%d K=%d M=%d\n", e.name, n, k, m);
      return;
    }
  fprintf(stderr, "[marlin-ENGAGE] UNNAMED kernel=%p N=%d K=%d M=%d\n", reinterpret_cast<void*>(kernel), n, k, m);
}
// [0047 ENGAGE-TELL] END static

#if MARLIN_ZPPACK
// [0051 ZPPACK cuda-zppack 10-07] Zero points in the scale words (int8 activations x asymmetric uint4 weights, g32 / g64).
// Run switch VLLM_MARLIN_ZPPACK, read once per process: unset/""/0 = OFF (nothing is packed, every launch is the served one: one
// untaken branch per marlin_mm on a load-time constant); 1 = ON; anything else is a hard error at the first Marlin GEMM.
// VLLM_MARLIN_ZPPACK_N: comma list of N ("*" = every N; unset or empty = NO N, i.e. armed but nothing packed), strict.
// ON, at the FIRST marlin_mm that sees a weight's scale tensor (the profile run: weights are loaded, no graph is capturing) and
// only if the launch is kS8 x kU4 -> bf16 with group_blocks 2 or 4, its N listed and N % 64 == 0: the census reads the tensor's
// int16 codes and its zero points to the host (no device memory: host buffers only), and the tensor is PACKED in place iff every
// code is in 0..4096 and some u in 0..4095 is unused (u = the largest such code; deterministic): word = [zp:4][code:12] with
// 4096 stored as u, the zero-point nibbles moved into the words the twin's 8-byte load reads (kt_zpp_pack). Else, or when the
// first sight happens inside a graph capture, the tensor stays SERVED for the life of the process (never packed later, so a
// captured served launch never meets packed words). A PACKED tensor runs the ZPP twin of the kernel its launch selected
// (Marlin -> MarlinZPP, MarlinXS -> MarlinXSZPP; get_marlin_zpp_twin), with u in tile_major bits 4..15; its zp array stays
// allocated and is never read again: its first int becomes the packed-state guard 0x05100000 | u, which the twin checks at
// launch (a ZPP launch on a weight not packed with that u traps). No ZPP twin for the selected kernel = hard error (no fallback).
// Refused at arming (hard error): VLLM_MARLIN_OPR / SCALE_PREFETCH / REMAINDER_FOLD_NK / L2PF / BPSM2 armed (no
// ZPP twin of those families) and VLLM_MARLIN_MAX_TMB > 4 (no twin above tmb 4). [0052 ZPPTPF] VLLM_MARLIN_TAIL_PREFETCH
// composes: a packed launch runs the ZPPTPF twin of its ZPP twin (get_marlin_zpptpf_twin; none = the ZPP twin, no tail) and
// its predecessor's tail prefetches its scale lines only (the zero-point array of a packed weight is never read).
// Tells (stderr): "[marlin-ZPPACK] ARMED n=<list>" once; per tensor "[marlin-ZPPACK] PACK N= K= G= u= c4096= distinct=
// zpk64=<16 hex> (#<i>, zp bytes not read <total>)" or "[marlin-ZPPACK] SKIP N= K= <reason>"; "[marlin-ZPPACK] ON N= K= M= twin="
// once per (N, K, twin). zpk64 = sum_i (word_i + 1) * (splitmix64(i) | 1) mod 2^64 over the packed row-major words: the same
// value every boot for the same checkpoint (zppack/scripts/zppack_proof1.py computes it on the CPU).
static int kt_zpp_mode() {
  const char* v = getenv("VLLM_MARLIN_ZPPACK");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if (v[0] == '1' && v[1] == '\0') return 1;
  return -1;
}
static const int KT_ZPP = kt_zpp_mode();
// [0060 ZPPGUARD cuda 10-11] VLLM_MARLIN_ZPPACK_ZPGUARD = unset / 0 (OFF: the 0051 behaviour, no map, no digest) | 1 (ON) | else a hard
// error at the ZPPACK gate. Why: kt_zpp_state writes the packed-state guard 0x05100000 | u into zp[0] when it packs a weight, so two LIVE
// scale tensors sharing one zp array corrupt one of them in every order that contains a PACK (a later census reads guard bits as zero
// points; a served launch of the other reads them; a later pack overwrites zp[0] under an earlier served tensor). ON: on a registry miss,
// a per-device map zp ptr -> {owning scales ptr, numel, N/K/G, first-sight FNV-64 of the zp bytes, zp[0], state} refuses a zp array met
// under a second LIVE scales ptr (live = the bytes are still the owner's, or the owner's packed bytes with its guard in zp[0]); a freed
// address now holding other bytes or another size re-registers ("ZPGUARD reuse" tell). Fail-closed residuals: identical bytes re-used at
// the same address, and a zp first seen inside a graph capture (no census bytes), both refuse. Served weights live for the process.
static int kt_zpg_mode() {
  const char* v = getenv("VLLM_MARLIN_ZPPACK_ZPGUARD");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if (v[0] == '1' && v[1] == '\0') return 1;
  return -1;
}
static const int KT_ZPG = kt_zpg_mode();
struct KtZppGate {
  bool all_n = false;
  std::vector<int> ns;
};
static const KtZppGate& kt_zpp_gate() {
  static const KtZppGate g = [] {
    const char* mode = getenv("VLLM_MARLIN_ZPPACK");
    STD_TORCH_CHECK(KT_ZPP == 1, "VLLM_MARLIN_ZPPACK: malformed value '", mode != nullptr ? mode : "", "' (want 0 or 1)");
    const char* refuse = nullptr;
  #if MARLIN_OPR
    if (KT_OPR) refuse = "VLLM_MARLIN_OPR";
  #endif
  #if MARLIN_SCALE_PF
    if (KT_SP) refuse = "VLLM_MARLIN_SCALE_PREFETCH";
  #endif
  #if MARLIN_FOLD
    if (KT_FOLD) refuse = "VLLM_MARLIN_REMAINDER_FOLD_NK";
  #endif
  // [0052 ZPPTPF] VLLM_MARLIN_TAIL_PREFETCH is no longer refused: marlin_mm swaps a packed launch's TPF twin for its ZPPTPF twin
  #if MARLIN_L2PF
    if (KT_L2PF) refuse = "VLLM_MARLIN_L2PF";
  #endif
    if (KT_BPSM2) refuse = "VLLM_MARLIN_BPSM2";
    STD_TORCH_CHECK(refuse == nullptr, "[marlin-ZPPACK] VLLM_MARLIN_ZPPACK=1 with ", refuse,
                    " armed: no ZPP twin of that family (ZPPACK composes with the served kernel and XSLICE only)");
    STD_TORCH_CHECK(kt_max_thread_m_blocks() <= 4, "[marlin-ZPPACK] VLLM_MARLIN_ZPPACK=1 with VLLM_MARLIN_MAX_TMB=",
                    kt_max_thread_m_blocks(), ": ZPP twins exist for thread_m_blocks <= 4 only");
    KtZppGate r;
    const char* s = getenv("VLLM_MARLIN_ZPPACK_N");
    std::string list = s != nullptr ? s : "";
    if (list == "*") {
      r.all_n = true;
    } else if (!list.empty()) {
      size_t i = 0;
      while (i <= list.size()) {
        size_t j = list.find(',', i);
        if (j == std::string::npos) j = list.size();
        const std::string cur = list.substr(i, j - i);
        bool digits = !cur.empty() && cur.size() <= 10;
        for (char c : cur) digits = digits && c >= '0' && c <= '9';
        STD_TORCH_CHECK(digits, "VLLM_MARLIN_ZPPACK_N: entry '", cur, "' is not a plain decimal N (list '", list, "')");
        const long v = std::stol(cur);
        STD_TORCH_CHECK(v >= 1 && v <= (1L << 30), "VLLM_MARLIN_ZPPACK_N: entry ", v, " out of 1..2^30");
        r.ns.push_back(static_cast<int>(v));
        i = j + 1;
      }
    }
    fprintf(stderr, "[marlin-ZPPACK] ARMED VLLM_MARLIN_ZPPACK=1 n=%s\n",
            r.all_n ? "*" : (list.empty() ? "(none: nothing is packed)" : list.c_str()));
  #if MARLIN_TAILPF
    // [0052 ZPPTPF] the one literal only a 0052 build carries: the compose chain guard probes the mounted .so for it
    // (a capability probe, not a sha), so TAIL_PREFETCH stays a ZPPACK conflict on any .so without the ZPPTPF twins
    if (KT_TPF)
      fprintf(stderr, "[marlin-ZPPACK] ZPPTPF ARMED (0052): VLLM_MARLIN_TAIL_PREFETCH=%d composes; a packed launch runs "
                      "the ZPPTPF twin of its ZPP twin (none: the ZPP twin, no tail)\n", KT_TPF);
  #endif
    // [0060 ZPPGUARD] strict switch, read with the ZPPACK gate (only while ZPPACK is armed); OFF prints nothing
    const char* zg = getenv("VLLM_MARLIN_ZPPACK_ZPGUARD");
    STD_TORCH_CHECK(KT_ZPG >= 0, "VLLM_MARLIN_ZPPACK_ZPGUARD: malformed value '", zg != nullptr ? zg : "", "' (want 0 or 1)");
    if (KT_ZPG == 1)
      fprintf(stderr, "[marlin-ZPPACK] ZPGUARD ARMED VLLM_MARLIN_ZPPACK_ZPGUARD=1 (a zp array shared by two live scale tensors refuses)\n");
    return r;
  }();
  return g;
}
static inline uint64_t kt_zpp_splitmix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// The repack, host side. codes [G, N] int16 (0..4096, no code == u), zp [G, N / 8] int32 (marlin a8 layout) -> words [G, N]:
// word x of a row = (nibble 4 * ((x / 32) % 2) + x % 4 of zp int 8 * (x / 64) + (x % 32) / 4) << 12 | (code == 4096 ? u : code),
// i.e. the 4 nibbles marlin_template.h's lane group q of warp wm reads (zp int 8 * (wm / 2) + q, nibbles 4 * (wm % 2) + 0..3)
// sit in words 32 * wm + 4 q + 0..3 of the tile: the int2 at 8 * wm + q of the scale stage (MARLIN_ZPPACK_TWIN zpp_sh_rd).
static uint64_t kt_zpp_pack(const int16_t* codes, const int32_t* zp, int G, int N, int u, uint16_t* out) {
  uint64_t h = 0;
  for (int g = 0; g < G; g++) {
    const int16_t* c = codes + (size_t)g * N;
    const uint32_t* z = reinterpret_cast<const uint32_t*>(zp) + (size_t)g * (N / 8);
    uint16_t* o = out + (size_t)g * N;
    for (int x = 0; x < N; x++) {
      const int code = c[x];
      const uint32_t fld = code == 4096 ? (uint32_t)u : (uint32_t)code;
      const uint32_t nib = (z[8 * (x / 64) + (x % 32) / 4] >> (4 * (4 * ((x / 32) % 2) + (x % 4)))) & 0xFu;
      o[x] = (uint16_t)((nib << 12) | fld);
      const uint64_t i = (uint64_t)g * N + x;
      h += ((uint64_t)o[x] + 1) * (kt_zpp_splitmix64(i) | 1ull);
    }
  }
  return h;
}
// [0060 ZPPGUARD] one registration per zero-point array per device (kt_zpp_state's lock guards the map)
struct KtZpgReg {
  const void* s;
  long long numel;
  int n, k, g;
  bool known;   // census bytes seen (false: first sight inside a graph capture)
  uint64_t d0;  // FNV-64 of the zp bytes at first sight
  int32_t zp0;  // zp[0] at first sight
  int st;       // the owner's 0051 state: u >= 0 PACKED, -1 SERVED
};
static std::map<int, std::map<const void*, KtZpgReg>>& kt_zpg_map() {
  static std::map<int, std::map<const void*, KtZpgReg>> m;
  return m;
}
static uint64_t kt_zpg_fnv(const int32_t* w, size_t n32, bool sub0, int32_t w0) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n32; i++) {
    const uint32_t v = static_cast<uint32_t>(i == 0 && sub0 ? w0 : w[i]);
    for (int b = 0; b < 4; b++) {
      h ^= (v >> (8 * b)) & 0xFFu;
      h *= 0x100000001b3ull;
    }
  }
  return h;
}
// On a registry MISS, before any decision or H2D. zps = the census copy (nullptr inside a capture), nz its int32 count. Refuses a LIVE
// share or one scale tensor under two geometries; re-registers a reused address with a tell. Returns the digest of zps (0 if nullptr).
static uint64_t kt_zpg_check(const void* b_s, const void* zp, int G, int N, int K, const int32_t* zps, size_t nz) {
  int dev = -1;
  STD_TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "[marlin-ZPPACK] ZPGUARD: cudaGetDevice failed");
  const uint64_t d = zps != nullptr ? kt_zpg_fnv(zps, nz, false, 0) : 0;
  auto& m = kt_zpg_map()[dev];
  auto it = m.find(zp);
  if (it == m.end()) return d;
  const KtZpgReg r = it->second;
  char msg[640];
  if (r.s == b_s) {
    snprintf(msg, sizeof msg, "[marlin-ZPPACK] ZPGUARD REFUSED: scale tensor %p with zp array %p (dev %d) launched with two geometries: "
             "N %d K %d G %d, then N %d K %d G %d", b_s, zp, dev, r.n, r.k, r.g, N, K, G);
    STD_TORCH_CHECK(false, msg);
  }
  bool live = !r.known || zps == nullptr;
  if (!live && r.numel == static_cast<long long>(nz))
    live = d == r.d0 || (r.st >= 0 && zps[0] == static_cast<int32_t>(0x05100000u | static_cast<uint32_t>(r.st)) &&
                         kt_zpg_fnv(zps, nz, true, r.zp0) == r.d0);
  if (live) {
    snprintf(msg, sizeof msg, "[marlin-ZPPACK] ZPGUARD REFUSED: zp array %p (numel %lld, dev %d) is shared by LIVE scale tensors %p "
             "(N %d K %d G %d, %s) and %p (N %d K %d G %d): ZPPACK writes its guard into zp[0], so sharing corrupts one of them. Give each "
             "scale tensor its own zero-point array, or set VLLM_MARLIN_ZPPACK_ZPGUARD=0 (the 0051 behaviour)", zp, static_cast<long long>(nz),
             dev, r.s, r.n, r.k, r.g, r.st >= 0 ? "PACKED" : (r.known ? "SERVED" : "first seen inside a capture"), b_s, N, K, G);
    STD_TORCH_CHECK(false, msg);
  }
  fprintf(stderr, "[marlin-ZPPACK] ZPGUARD reuse zp=%p dev %d scales %p -> %p N/K/G %d/%d/%d -> %d/%d/%d\n", zp, dev, r.s, b_s, r.n, r.k,
          r.g, N, K, G);
  m.erase(it);
  return d;
}
static void kt_zpg_record(const void* b_s, const void* zp, int G, int N, int K, const int32_t* zps, size_t nz, uint64_t d, int st) {
  int dev = -1;
  STD_TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess, "[marlin-ZPPACK] ZPGUARD: cudaGetDevice failed");
  kt_zpg_map()[dev][zp] = KtZpgReg{b_s, static_cast<long long>(nz), N, K, G, zps != nullptr, d, zps != nullptr ? zps[0] : 0, st};
}
struct KtZppKey {
  const void* s;
  const void* zp;
  int g, n;
  bool operator<(const KtZppKey& o) const {
    return s != o.s ? s < o.s : zp != o.zp ? zp < o.zp : g != o.g ? g < o.g : n < o.n;
  }
};
// The tensor's state, decided at its first sight and never changed: u >= 0 = PACKED with u, -1 = SERVED.
static int kt_zpp_state(void* b_s, void* zp, int G, int N, int K, cudaStream_t stream) {
  static std::mutex mu;
  static std::map<KtZppKey, int> reg;
  static long long zp_bytes_off = 0;
  static int npacked = 0;
  std::lock_guard<std::mutex> lk(mu);
  const KtZppKey key{b_s, zp, G, N};
  auto it = reg.find(key);
  if (it != reg.end()) return it->second;
  int st = -1;
  const char* why = nullptr;
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  const cudaError_t ce = cudaStreamIsCapturing(stream, &cs);
  if (ce != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
    why = "first seen inside a graph capture (kept served)";
    if (KT_ZPG == 1) {  // [0060 ZPPGUARD] no census bytes: any other-scales sight of this zp refuses
      kt_zpg_check(b_s, zp, G, N, K, nullptr, (size_t)G * N / 8);
      kt_zpg_record(b_s, zp, G, N, K, nullptr, (size_t)G * N / 8, 0, -1);
    }
  } else {
    const size_t nw = (size_t)G * N;
    std::vector<int16_t> codes(nw);
    std::vector<int32_t> zps(nw / 8);
    std::vector<uint16_t> words(nw);
    STD_TORCH_CHECK(cudaMemcpyAsync(codes.data(), b_s, nw * 2, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                        cudaMemcpyAsync(zps.data(), zp, nw / 2, cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
                        cudaStreamSynchronize(stream) == cudaSuccess,
                    "[marlin-ZPPACK] census copy failed N=", N, " K=", K);
    const uint64_t zgd = KT_ZPG == 1 ? kt_zpg_check(b_s, zp, G, N, K, zps.data(), nw / 8) : 0;  // [0060 ZPPGUARD] before any decision
    std::vector<long long> hist(4097, 0);
    long long bad = 0;
    for (size_t i = 0; i < nw; i++) {
      const int c = codes[i];
      if (c < 0 || c > 4096) bad++;
      else hist[c]++;
    }
    int u = -1, distinct = 0;
    for (int c = 0; c <= 4096; c++) distinct += hist[c] > 0;
    for (int c = 4095; c >= 0 && u < 0; c--)
      if (hist[c] == 0) u = c;
    if (bad) {
      why = "a code outside 0..4096 (kept served)";
    } else if (u < 0) {
      why = "no unused code u in 0..4095 (kept served)";
    } else {
      const uint64_t h = kt_zpp_pack(codes.data(), zps.data(), G, N, u, words.data());
      // the packed-state guard the twin checks (marlin_template.h zpp_seen): the zp array is never read again for this weight
      const int32_t magic = static_cast<int32_t>(0x05100000u | static_cast<uint32_t>(u));
      STD_TORCH_CHECK(cudaMemcpyAsync(b_s, words.data(), nw * 2, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                          cudaMemcpyAsync(zp, &magic, 4, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
                          cudaStreamSynchronize(stream) == cudaSuccess,
                      "[marlin-ZPPACK] pack copy failed N=", N, " K=", K);
      st = u;
      npacked++;
      zp_bytes_off += (long long)nw / 2;
      fprintf(stderr,
              "[marlin-ZPPACK] PACK N=%d K=%d G=%d u=%d c4096=%lld distinct=%d zpk64=%016llx guard=%08x (#%d, zp bytes not read %lld)\n",
              N, K, G, u, hist[4096], distinct, (unsigned long long)h, (unsigned)magic, npacked, zp_bytes_off);
    }
    if (KT_ZPG == 1) kt_zpg_record(b_s, zp, G, N, K, zps.data(), nw / 8, zgd, st);  // [0060 ZPPGUARD] the owner, after its decision
  }
  if (why != nullptr) fprintf(stderr, "[marlin-ZPPACK] SKIP N=%d K=%d G=%d: %s\n", N, K, G, why);
  reg.emplace(key, st);
  return st;
}
// u >= 0: this launch's weight is PACKED (run the ZPP twin with u); -1: served.
static int kt_zpp_prepare(vllm::ScalarType const& a_type, vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
                          vllm::ScalarType const& s_type, bool has_zp, bool is_zp_float, int group_blocks, void* b_s,
                          void* zp, int num_groups, int prob_n, int prob_k, cudaStream_t stream) {
  const KtZppGate& zg = kt_zpp_gate();  // strict parse + refusals + ARMED tell; a malformed value fails here
  if (!(a_type == vllm::kS8 && b_type == vllm::kU4 && c_type == vllm::kBFloat16 && s_type == vllm::kBFloat16 && has_zp &&
        !is_zp_float && (group_blocks == 2 || group_blocks == 4) && prob_n % 64 == 0 && num_groups > 1))
    return -1;
  bool n_ok = zg.all_n;
  for (int x : zg.ns)
    if (x == prob_n) n_ok = true;
  if (!n_ok) return -1;
  return kt_zpp_state(b_s, zp, num_groups, prob_n, prob_k, stream);
}
static void kt_zpp_tell_on(int m, int n, int k, const char* twin) {
  static std::mutex told_mu;
  static std::vector<std::string> told;
  char key[96];
  snprintf(key, sizeof(key), "%d/%d/%s", n, k, twin);
  std::lock_guard<std::mutex> lk(told_mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  fprintf(stderr, "[marlin-ZPPACK] ON N=%d K=%d M=%d twin=%s\n", n, k, m, twin);
}
  #include "kernel_selector_zpp.h"
  #if MARLIN_TAILPF
    #include "kernel_selector_zpptpf.h"  // [0052 ZPPTPF] get_marlin_zpptpf_twin: the TAILPF twin of a ZPP twin
  #endif
#endif
#if MARLIN_COLFPF
// [0061 COLFPF v2 cuda 2026-10-11] VLLM_MARLIN_COLFPF: unset / "" / "0" -> off (every launch exactly as before); "1" -> a launch that
// carries a column factor (fp32 [N] in global_scale: 0059 W4RCOLF int4 grouped int8-activation, 0055 KU3S-4 kU3S) runs the CPF twin
// of the kernel (and of the TPF twin) the other selections chose: the same body + an L2 prefetch of the factor lines in its
// prologue, so the epilogue's factor __ldg hits L2 (a cache hint: results unchanged). A launch whose kernel has no CPF twin keeps
// it (tell NO-TWIN once per config; nothing is refused). Any other value refuses at the first factor launch. Tells: "[marlin-COLFPF]
// ARMED (0061) ..." once per process, "[marlin-COLFPF] ON N= K= M= twin= tpf=" / "NO-TWIN ..." once per (N, K, kernel, TPF twin).
// v1 (0061 v1x) put the prefetch into the shared template under a tile_major bit: every is_u3 || colf_a8 body's main loop moved
// (kat968: +0.026..+0.039 ms/step on every launch of those bodies, knob or not). v2 changes no existing body.
static int kt_colfpf_mode() {
  const char* v = getenv("VLLM_MARLIN_COLFPF");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if (v[0] == '1' && v[1] == '\0') return 1;
  return -1;
}
static const int KT_COLFPF = kt_colfpf_mode();
static void kt_colfpf_tell(int m, int n, int k, const char* twin, const char* tpf, bool any) {
  static std::mutex told_mu;
  static bool armed = false;
  static std::vector<std::string> told;
  std::lock_guard<std::mutex> lk(told_mu);
  if (!armed) {
    armed = true;
    fprintf(stderr, "[marlin-COLFPF] ARMED (0061): column-factor launches run their CPF twins (factor lines -> L2 in the prologue); "
                    "first N=%d K=%d\n", n, k);
  }
  char key[160];
  snprintf(key, sizeof(key), "%d/%d/%s/%s", n, k, twin, tpf);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  if (any)
    fprintf(stderr, "[marlin-COLFPF] ON N=%d K=%d M=%d twin=%s tpf=%s\n", n, k, m, twin, tpf);
  else
    fprintf(stderr, "[marlin-COLFPF] NO-TWIN N=%d K=%d M=%d: the launch keeps its kernel (no factor prefetch)\n", n, k, m);
}
  #include "kernel_selector_cpf.h"  // [0061 COLFPF v2] get_marlin_cpf_twin / get_marlin_cpf_tpf_twin
#endif
#if MARLIN_TAILPF && MARLIN_KU3STPF
// [0063 KU3STPF cuda 2026-10-11] VLLM_MARLIN_TAILPF_KU3S: unset / "" / "0" -> off: every kS8 x kU3S launch runs exactly the kernel it ran
// at 0062 (get_marlin_tpf_twin has no kU3S entry, so TAILPF keeps the kernel and tells NO-TWIN). "1" with VLLM_MARLIN_TAIL_PREFETCH armed
// (1 or 2) -> a kU3S launch (0055 int3: gate_up / down / in_qkvz / out_proj of the int3 arms) whose kernel has a KU3STPF twin
// (get_marlin_ku3stpf_twin: Marlin -> MarlinTPF, MarlinXS -> MarlinXSTPF, kU3S configs) runs it through TAILPF's own path (registry,
// residency guard, graph-node patch, ON / KEEP tells): the same 0055 body + the tail prefetch of the NEXT Marlin launch's scale / zp /
// factor lines after its last epilogue (a cache hint: results unchanged). "1" with TAIL_PREFETCH off: inert (tell INERT once). Anything
// else refuses at the first kU3S launch, armed or not. With VLLM_MARLIN_COLFPF=1 a column-factor launch whose TPF twin has a CPF twin
// (get_marlin_ku3s_cpftpf_twin: MarlinCPFTPF / MarlinXSCPFTPF) runs that; none -> the TPF twin (TAILPF precedence, as 0061; the factor
// still rides in the predecessor's tail, 0061 Part C). Tells: "[marlin-TAILPF] KU3S ARMED (0063) ..." once; "[marlin-TAILPF] KU3S ON N=
// K= M= kernel= twin=" / "KU3S NO-TWIN ..." / "KU3S NOT-RUN ..." once per (N, K, kernel config, twin).
static int kt_tpf_ku3s_mode() {
  const char* v = getenv("VLLM_MARLIN_TAILPF_KU3S");
  if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if (v[0] == '1' && v[1] == '\0') return 1;
  return -1;
}
static const int KT_TPF_KU3S = kt_tpf_ku3s_mode();
// Malformed value -> refuse; TAIL_PREFETCH off -> INERT once; armed -> ARMED once. Returns true when kU3S twins may be taken.
static bool kt_tpf_ku3s_gate(int n, int k) {
  const char* v = getenv("VLLM_MARLIN_TAILPF_KU3S");
  STD_TORCH_CHECK(KT_TPF_KU3S >= 0, "[marlin-TAILPF] VLLM_MARLIN_TAILPF_KU3S: malformed value '", v != nullptr ? v : "",
                  "' (want 0 or 1)");
  static std::mutex mu;
  static bool told = false;
  std::lock_guard<std::mutex> lk(mu);
  if (!told) {
    told = true;
  #if MARLIN_COLFPF
    const int colf = KT_COLFPF;
  #else
    const int colf = 0;
  #endif
    if (KT_TPF)
      fprintf(stderr, "[marlin-TAILPF] KU3S ARMED (0063): kS8 x kU3S launches run their TPF twins (VLLM_MARLIN_TAIL_PREFETCH=%d, "
                      "COLFPF=%d); first N=%d K=%d\n", KT_TPF, colf, n, k);
    else
      fprintf(stderr, "[marlin-TAILPF] KU3S INERT (0063): VLLM_MARLIN_TAILPF_KU3S=1 but VLLM_MARLIN_TAIL_PREFETCH is off: kU3S launches "
                      "keep their kernels\n");
  }
  return KT_TPF != 0;
}
static void kt_tpf_ku3s_tell(const char* what, int m, int n, int k, int thr, int tmb, int tnb, int tkb, int gb, bool xs,
                             const char* twin) {
  static std::mutex mu;
  static std::vector<std::string> told;
  char key[200];
  snprintf(key, sizeof(key), "%s/%d/%d/%d/%d/%d/%d/%d/%d/%s", what, n, k, thr, tmb, tnb, tkb, gb, (int)xs, twin);
  std::lock_guard<std::mutex> lk(mu);
  for (const auto& s : told)
    if (s == key) return;
  told.emplace_back(key);
  fprintf(stderr, "[marlin-TAILPF] KU3S %s N=%d K=%d M=%d kernel=%s<kS8,kU3S> thr=%d tmb=%d tnb=%d tkb=%d gb=%d twin=%s\n", what, n, k,
          m, xs ? "MarlinXS" : "Marlin", thr, tmb, tnb, tkb, gb, twin);
}
  #include "kernel_selector_ku3stpf.h"  // [0063 KU3STPF] get_marlin_ku3stpf_twin / get_marlin_ku3s_cpftpf_twin
#endif

// [0057 A16FS cuda 2026-10-10] batch-invariant 16-bit-activation launches (DESIGN-A16FS.md; the A16 twin of 0056 IRED): for a listed
// (N, K) every launch takes ONE pinned thread config and a grid that divides its tile count, so the kernel's own part1/part2 math gives
// every column one block over the whole K (no split-K, no cross-block reduce, no lock): the row's fp32 chain is the same for every M.
// VLLM_MARLIN_A16FS unset/empty/0 = off (served). 1 = on, for VLLM_MARLIN_A16FS_NK = "NxK:TKxTNxTHR,..." (unset/empty = the served
// attention shapes 14336x5120 and 5120x6144 at {128,64,128}). Host-only: every kernel is the carrier's.
struct KtA16fsCfg { int n, k, tk, tn, th; };
struct KtA16fsGate { bool on = false; std::vector<KtA16fsCfg> nk; std::string spec; };
static const KtA16fsGate& kt_a16fs_gate() {
  static const KtA16fsGate g = [] {
    KtA16fsGate r;
    const char* e = std::getenv("VLLM_MARLIN_A16FS");
    if (e == nullptr || e[0] == '\0' || std::string(e) == "0") return r;
    STD_TORCH_CHECK(std::string(e) == "1", "VLLM_MARLIN_A16FS must be 0 or 1, got '", e, "'");
    r.on = true;
    const char* l = std::getenv("VLLM_MARLIN_A16FS_NK");
    r.spec = (l == nullptr || l[0] == '\0') ? std::string("14336x5120:128x64x128,5120x6144:128x64x128") : std::string(l);
    size_t i = 0;
    while (i < r.spec.size()) {
      size_t j = r.spec.find(',', i);
      if (j == std::string::npos) j = r.spec.size();
      const std::string t = r.spec.substr(i, j - i);
      KtA16fsCfg c{};
      char extra = 0;
      const int got = sscanf(t.c_str(), "%dx%d:%dx%dx%d%c", &c.n, &c.k, &c.tk, &c.tn, &c.th, &extra);
      STD_TORCH_CHECK(got == 5 && c.n > 0 && c.k > 0 && c.n % c.tn == 0 && c.k % c.tk == 0 &&
                          (c.tk == 64 || c.tk == 128) && (c.tn == 64 || c.tn == 128 || c.tn == 256) && (c.th == 128 || c.th == 256),
                      "[marlin-A16FS] VLLM_MARLIN_A16FS_NK item '", t, "' is not NxK:TKxTNxTHR with TK 64|128, TN 64|128|256, THR 128|256 "
                      "dividing N and K");
      r.nk.push_back(c);
      i = j + 1;
    }
    STD_TORCH_CHECK(!r.nk.empty(), "[marlin-A16FS] VLLM_MARLIN_A16FS=1 with an empty shape list");
    fprintf(stderr, "[marlin-A16FS] ARMED (0057): 16-bit-activation launches of %s run one pinned config, one block per column "
                    "(no split-K)\n", r.spec.c_str());
    return r;
  }();
  return g;
}
static const KtA16fsCfg* kt_a16fs_cfg(const vllm::ScalarType& a_type, int n, int k) {
  const KtA16fsGate& g = kt_a16fs_gate();
  if (!g.on || a_type.size_bits() != 16) return nullptr;
  for (const auto& c : g.nk)
    if (c.n == n && c.k == k) return &c;
  return nullptr;
}

void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, int prob_m,
               int prob_n, int prob_k, int lda, void* workspace,
               vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
               vllm::ScalarType const& c_type, vllm::ScalarType const& s_type,
               bool has_bias, bool has_zp, int num_groups, int group_size,
               int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float,
               int64_t c_tmp_numel = 0,  // [0040 OPR] fp32 floats behind C_tmp (OPR slot capacity)
               int* folds_locks = nullptr, int folds_locks_n = 0) {  // [0048 FOLDS] private zeroed lock buffer
  bool is_a_8bit = a_type.size_bits() == 8;
  STD_TORCH_CHECK(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [",
                  prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    STD_TORCH_CHECK(prob_k % group_blocks == 0, "prob_k = ", prob_k,
                    " is not divisible by group_blocks = ", group_blocks);
  }

  // [0055 KU3S] kU3S runs the int4 sibling's tile geometry (config tables, validity, SM-fill / narrow tile, TRIM, BPSM2): every
  // host formula below sees 4 bits. Byte counts of the real B array use kt_b_bits (the TAILPF successor descriptor).
  int num_bits = b_type == vllm::kU3S ? 4 : b_type.size_bits();
  const int kt_b_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  STD_TORCH_CHECK(max_shared_mem > 0);

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 75,
                  "marlin kernel only support Turing or newer GPUs.");
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    STD_TORCH_CHECK(a_type == vllm::kFloat16 || a_type == vllm::kS8,
                    "Turing only support FP16 or INT8 activation.");
  }
  if (a_type == vllm::kFE4M3fn) {
    STD_TORCH_CHECK(major_capability * 10 + minor_capability >= 89,
                    "FP8 only support Ada Lovelace or newer GPUs.");
    STD_TORCH_CHECK(
        major_capability * 10 + minor_capability == 89 ||
            major_capability == 12,
        "Marlin W4A8-FP8 only support SM89 or SM12x device (It is slower than "
        "Marlin W4A16 on other devices).");
  }

  const KtA16fsCfg* kt_a16 = kt_a16fs_cfg(a_type, prob_n, prob_k);  // [0057 A16FS] nullptr = served launch plan
#if MARLIN_IRED
  // [0056 IRED] declaration == reality: this .so reduces int8-activation grouped split-K partials in int32. The engine must declare
  // it (VLLM_MARLIN_IRED=1, salted); int8 channelwise (group_size -1) is not covered and refuses; 16-bit activations stay fp32 (told once).
  {
    static const bool kt_ired_decl = [] {
      const char* e = std::getenv("VLLM_MARLIN_IRED");
      STD_TORCH_CHECK(e != nullptr && std::string(e) == "1",
                      "[marlin-IRED] REFUSED: this .so is built with 0056 IRED (int32 split-K reduce for int8-activation grouped GEMMs); "
                      "declare it with VLLM_MARLIN_IRED=1 (salted)");
      // [0056 IRED v2, corruption 10-10 03:34Z] A16 launches are NOT covered (fp32 split-K reduce kept); the engine must acknowledge it
      const char* a16 = std::getenv("VLLM_MARLIN_IRED_A16_ACK");
      // [0057 A16FS] the listed A16 shapes are covered by A16FS; any other A16 launch still needs the acknowledgement (checked per launch)
      STD_TORCH_CHECK((a16 != nullptr && std::string(a16) == "1") || kt_a16fs_gate().on,
                      "[marlin-IRED] REFUSED: 16-bit-activation launches are not covered by 0056 IRED; acknowledge with "
                      "VLLM_MARLIN_IRED_A16_ACK=1 (salted) or cover them with VLLM_MARLIN_A16FS=1 (0057)");
      fprintf(stderr, "[marlin-IRED] ARMED (0056): int8-activation grouped Marlin GEMMs reduce split-K partials in int32, one fp32 "
                      "conversion in the final slice\n");
      return true;
    }();
    (void)kt_ired_decl;
    if (a_type == vllm::kS8) {
      STD_TORCH_CHECK(group_blocks != -1, "[marlin-IRED] REFUSED: int8 activations with channelwise scales (group_size -1) are not "
                                          "covered by 0056 IRED (N=", prob_n, " K=", prob_k, ")");
    } else if (kt_a16 == nullptr) {
      static const bool kt_ired_a16_ack = [] {
        const char* a = std::getenv("VLLM_MARLIN_IRED_A16_ACK");
        return a != nullptr && std::string(a) == "1";
      }();
      STD_TORCH_CHECK(kt_ired_a16_ack, "[marlin-IRED] REFUSED: 16-bit-activation launch N=", prob_n, " K=", prob_k,
                      " is not in VLLM_MARLIN_A16FS_NK and VLLM_MARLIN_IRED_A16_ACK is not 1");
      static bool kt_ired_a16_told = false;
      if (!kt_ired_a16_told) {
        kt_ired_a16_told = true;
        fprintf(stderr, "[marlin-IRED] NOT COVERED: 16-bit-activation launches keep the fp32 split-K reduce (first N=%d K=%d)\n",
                prob_n, prob_k);
      }
    }
  }
#endif

#if MARLIN_ZPPACK
  // [0051 ZPPACK] BEGIN state (once per GEMM, before the row loop): -1 = served; u >= 0 = the weight is packed, every launch below
  // runs a ZPP twin. VLLM_MARLIN_ZPPACK unset/0: one untaken branch on a load-time constant.
  int kt_zpp_u = -1;
  if (KT_ZPP)
    kt_zpp_u = kt_zpp_prepare(a_type, b_type, c_type, s_type, has_zp, is_zp_float, group_blocks, b_s, zp, num_groups, prob_n,
                              prob_k, stream);
  // [0051 ZPPACK] END state
#endif
  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  // [0032 cuda 2026-10-01] the raised ceiling applies ONLY when the whole problem fits ONE pass (prob_m <= 16 * ceiling, e.g. a
  // 12-seat verify step of 96 rows). Anything larger (prefill chunks, >12 seats) keeps the served 64-row plan byte for byte:
  // 96-row passes would re-split prefill (different rounding; stage t7 measured M=128 up to 1.8x slower as 96+32).
  const int kt_ceil = kt_max_thread_m_blocks();
  int max_thread_m_blocks = (kt_ceil > 4 && prob_m <= 16 * kt_ceil) ? kt_ceil : (kt_ceil < 4 ? kt_ceil : 4);
#if MARLIN_FOLD
  bool kt_fold_drop = false;  // [0042 FOLD] set when a fold's config has no MarlinFOLD twin: re-plan that rest_m served
#endif
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;
#if MARLIN_FOLD
    // [0042 FOLD] BEGIN fold decision (the 0035 condition). BEFORE everything below that reads prob_m_split, so the thread
    // config, the [0033] tile-major test and the twin selections all see the folded value: the config chosen for rest_m is
    // the one chosen for (par_count + 1) * 64 rows (validity and the table pick do not read prob_m_split, the SM-fill override
    // reads div_ceil(prob_m_split, 64) = par_count + 1, the A8 narrow tile needs prob_m_split <= 64: false for both), so the
    // folded launch is that served launch's partition with rows moved; tile_major reads prob_m_split / 64 = par_count, the
    // unfolded launch's value (MarlinFOLD runs either order); XS, OPR and SP need prob_m_split <= 16 * thread_m_blocks = 64 and
    // a folded prob_m_split is > 64, so no other twin meets a fold.
    bool kt_fold = false;
  #if MARLIN_FOLDS
    bool kt_folds = false;  // [0048 FOLDS] this launch is a mode-S fold candidate (prob_m_split untouched)
  #endif
    if (KT_FOLD && kt_a16 == nullptr) {  // [0057 A16FS] no fold: the launch keeps its own rows
      const std::vector<std::pair<int, int>>& fold_nk = kt_fold_nk();  // strict parse + ARMED tell at the first Marlin GEMM
      if (!kt_fold_drop && par_count > 0 && par_count < max_par && rest_m > prob_m_split &&
          max_thread_m_blocks == 4 && use_fp32_reduce && !use_atomic_add) {
        for (const auto& nk : fold_nk)
          if (nk.first == prob_n && nk.second == prob_k) kt_fold = true;
  #if MARLIN_FOLDS
        // [0048 FOLDS] mode S / SAR / SRA: the same condition and list, but prob_m_split stays the served par_count * 64, so the
        // config, grid, tile_major and every twin selection below are the served launch's; the twin selection decides.
        if (kt_fold && kt_folds_armed()) {
          kt_fold = false;
          kt_folds = true;
        }
  #endif
        if (kt_fold) prob_m_split = rest_m;
      }
    }
    // [0042 FOLD] END fold decision
#endif

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    // [0033 cuda 2026-10-02, CONSENSUS-8b] 0013 tile-major part-1 order ONLY for allow-listed GEMMs (N x K pairs from
    // VLLM_MARLIN_TILE_MAJOR_NK, e.g. "34816x5120,248320x5120" = TC gate_up + lm_head, the >= 64 MB units) at par >= 2.
    // Unset/empty = never (the served order everywhere). The remap is a bijection onto the same tiles with each tile's arithmetic
    // untouched (0013, card-proven bit-identical 30/30), so the output is bitwise either way; only L2 reuse of B changes.
    static const std::vector<std::pair<int, int>> kt_tm_nk = [] {
      std::vector<std::pair<int, int>> v;
      const char* e = std::getenv("VLLM_MARLIN_TILE_MAJOR_NK");
      if (e == nullptr) return v;
      std::string s(e); size_t i = 0;
      while (i < s.size()) {
        size_t j = s.find(',', i); if (j == std::string::npos) j = s.size();
        std::string t = s.substr(i, j - i); size_t x = t.find('x');
        STD_TORCH_CHECK(x != std::string::npos && x > 0 && x + 1 < t.size(), "VLLM_MARLIN_TILE_MAJOR_NK item '", t, "' is not NxK");
        v.emplace_back(static_cast<int>(kt_tpf_strict("VLLM_MARLIN_TILE_MAJOR_NK", t.substr(0, x), 1, 1 << 20)),
                       static_cast<int>(kt_tpf_strict("VLLM_MARLIN_TILE_MAJOR_NK", t.substr(x + 1), 1, 1 << 20)));   // [0072 STRICTPARSE] MRL-09: was stoi (trailing junk accepted)
        i = j + 1;
      }
      return v;
    }();
    int tile_major = 0;
    if (!kt_tm_nk.empty() && prob_m_split / (16 * thread_m_blocks) >= 2)
      for (const auto& nk : kt_tm_nk) if (nk.first == prob_n && nk.second == prob_k) { tile_major = 1; break; }
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;
    if (kt_a16 != nullptr) m_block_size_8 = 0;  // [0057 A16FS] one kernel instance for every M

    // Set thread config
    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (kt_a16 != nullptr) {  // [0057 A16FS] the pinned config for every M: no table pick, no SM-fill / narrow override
      thread_tfg = thread_config_t{kt_a16->tk, kt_a16->tn, kt_a16->th};
      exec_cfg = exec_config_t{1, thread_tfg};
      STD_TORCH_CHECK(is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n, prob_k, num_bits, group_size, has_zp,
                                      is_zp_float, is_a_8bit, stages, max_shared_mem_new),
                      "[marlin-A16FS] REFUSED: pinned config {", kt_a16->tk, ",", kt_a16->tn, ",", kt_a16->th, "} is not valid for N=",
                      prob_n, " K=", prob_k, " thread_m_blocks=", thread_m_blocks);
    } else if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      STD_TORCH_CHECK(prob_n % thread_n == 0, "prob_n = ", prob_n,
                      " is not divisible by thread_n = ", thread_n);
      STD_TORCH_CHECK(prob_k % thread_k == 0, "prob_k = ", prob_k,
                      " is not divisible by thread_k = ", thread_k);
    } else {
      // Auto config
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_zp,
          is_zp_float, is_a_8bit, stages, max_shared_mem, sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1 && !(KT_A8_NO_SMFILL && is_a_8bit) &&
          KT_CFG_INDEX < 0) {  // [club-kt] the sweep knobs bypass the SM-fill override
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size, has_zp,
                              is_zp_float, is_a_8bit, stages,
                              max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }
      // [cuda lane 2026-09-23] VLLM_MARLIN_A8_NARROW_MIN_N=<n> (unset = today's selection, unchanged): for int8
      // activations at more than one m-block, serve every GEMM with prob_n >= n on the narrow {128,64,128} tile that
      // the SM-fill override above already gives the N=5120 projections. In situ (8-seat CoR nsys) the narrow tile
      // streams ~885-900 GB/s (down, out_proj) against ~786-806 for the wide {64,256,256} on gate_up/in_proj, after a
      // common ~4.8 us launch intercept; this knob is how a microbench and a paired leg measure that directly.
      // Read ONCE: this is the per-launch path.
      static const int a8_narrow_min_n = [] {
        const char* e = std::getenv("VLLM_MARLIN_A8_NARROW_MIN_N");
        if (e == nullptr || e[0] == '\0') return 0;
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        STD_TORCH_CHECK(end != e && *end == '\0' && v > 0 && v <= (1L << 30),
                        "VLLM_MARLIN_A8_NARROW_MIN_N must be a positive integer, got '", e, "'");
        return static_cast<int>(v);
      }();
      // [0029 cuda 2026-10-01] VLLM_MARLIN_A8_NARROW_MAX_M=<rows> (unset = 64 = the 0027 scope, unchanged): the
      // largest prob_m_split the narrow tile may take. A 12-seat verify step (12 x 8 = 96 rows) runs as ONE launch
      // under 0010 (tmb 6) or 0015 (folded copies) and would otherwise drop the big-N GEMMs back to the wide tile.
      static const int a8_narrow_max_m = [] {
        const char* e = std::getenv("VLLM_MARLIN_A8_NARROW_MAX_M");
        if (e == nullptr || e[0] == '\0') return 64;
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        STD_TORCH_CHECK(end != e && *end == '\0' && v >= 16 && v <= 1024,
                        "VLLM_MARLIN_A8_NARROW_MAX_M must be an integer in [16, 1024], got '", e, "'");
        return static_cast<int>(v);
      }();
      // [0027] decode-size problems only (one m-chunk, <= 64 rows): a prefill chunk runs as par x 64 rows at
      // thread_m_blocks 4 too, and the wide tile was built for that compute-heavier shape; stage 186 timed M=64/56 only.
      if (a8_narrow_min_n > 0 && is_a_8bit && thread_m_blocks > 1 && prob_m_split <= a8_narrow_max_m && prob_n >= a8_narrow_min_n &&
          thread_tfg.thread_n != -1 &&
          !(thread_tfg.thread_k == 128 && thread_tfg.thread_n == 64) &&
          is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split, prob_n,
                          prob_k, num_bits, group_size, has_zp, is_zp_float,
                          is_a_8bit, stages, max_shared_mem_new)) {
        thread_tfg = {128, 64, 128};
        exec_cfg = {1, thread_tfg};
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    if (KT_CFG_TRACE) {
      fprintf(stderr,
              "[marlin-cfg] a_bits=%d b_bits=%d M=%d(split %d) N=%d K=%d -> "
              "m_blocks=%d thread_k=%d thread_n=%d threads=%d\n",
              (int)a_type.size_bits(), num_bits, prob_m, prob_m_split, prob_n,
              prob_k, thread_m_blocks, thread_k, thread_n, num_threads);
    }
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    STD_TORCH_CHECK(
        is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                        prob_k, num_bits, group_size, has_zp, is_zp_float,
                        is_a_8bit, stages, max_shared_mem_new),
        "Invalid thread config: thread_m_blocks = ", thread_m_blocks,
        ", thread_k = ", thread_tfg.thread_k,
        ", thread_n = ", thread_tfg.thread_n,
        ", num_threads = ", thread_tfg.num_threads, " for MKN = [", prob_m,
        ", ", prob_k, ", ", prob_n, "] and num_bits = ", num_bits,
        ", prob_m_split = ", prob_m_split, ", group_size = ", group_size,
        ", has_zp = ", has_zp, ", is_zp_float = ", is_zp_float,
        ", stages = ", stages, ", max_shared_mem_new = ", max_shared_mem_new);

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
        is_zp_float, stages);

    if (kernel == MarlinDefault) {
      STD_TORCH_CHECK(
          false, "Unsupported shapes: MNK = [", prob_m, ", ", prob_n, ", ",
          prob_k, "]", ", num_groups = ", num_groups,
          ", group_size = ", group_size, ", prob_m_split = ", prob_m_split,
          ", thread_m_blocks = ", thread_m_blocks,
          ", thread_n_blocks = ", thread_n_blocks,
          ", thread_k_blocks = ", thread_k_blocks,
          ", num_threads = ", num_threads, ", num_bits = ", num_bits);
    }

    bool kt_xs_selected = false;  // [0038 TRIM] the launch runs the MarlinXS twin (its layout adds sh_red)
#if MARLIN_XSLICE
    // [0036 XS-5] Twin selection, BEFORE the attribute call so the launch sets the attribute of the kernel it runs (no
    // second driver call). VLLM_MARLIN_XSLICE unset/0: KT_XSLICE is a load-time constant 0, this is one untaken branch,
    // and everything below is the served code (same kernel pointer, same attribute call, same launch, tile_major untouched).
    // Armed: the XS-4 M/N gate first, then the kernel-side XS preconditions (one 16*tmb-row problem copy = parallel 1, no
    // bias, fp32 reduce, no atomic-add reduce = !part_use_atomic_add below), then the twin lookup; no twin = served kernel.
    bool kt_xs_by_u3 = false;  // [0069 XSI3DN] admitted by VLLM_MARLIN_XSLICE_N_KU3S (not by the N list)
    if (KT_XSLICE && kt_a16 == nullptr && kt_xslice_on(prob_m_split, prob_n, b_type == vllm::kU3S, &kt_xs_by_u3) &&
        prob_m_split <= 16 * thread_m_blocks && !has_bias && use_fp32_reduce &&
        !(use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048)) {
      MarlinFuncPtr xs_kernel = get_marlin_xs_kernel(
          a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
          thread_k_blocks, m_block_size_8, group_blocks, num_threads,
          is_zp_float, stages);
      if (xs_kernel != nullptr) {
        if (kt_xs_by_u3)
          kt_xs_tell_on_u3(prob_m_split, prob_n, prob_k);
        else
          kt_xs_tell_on(prob_m_split, prob_n);
        kernel = xs_kernel;
        kt_xs_selected = true;
      }
    }
#endif

    // [0046 BPSM2] BEGIN plan (+ the mode-2 XS choice). After the XS selection, before OPR / SP / FOLD: a launch this plan
    // admits splits no column at either grid, so OPR (split columns only) and FOLD (> 16 * tmb rows) never select it, and the
    // SP selection maps whatever kernel stays here. VLLM_MARLIN_BPSM2 unset/0: one untaken branch on a load-time constant.
    bool kt_bpsm2 = false, kt_bpsm2_xs_dropped = false;
    if (KT_BPSM2 && kt_a16 == nullptr) {
      const KtBpsm2Gate& bg = kt_bpsm2_gate();  // strict parse + ARMED tell; a malformed value fails here
      bool n_ok = bg.all_n;
      for (int x : bg.ns)
        if (x == prob_n) n_ok = true;
      if (n_ok && prob_m_split <= 16 * thread_m_blocks) {
        const int k_tiles = prob_k / 16 / thread_k_blocks, n_tiles = prob_n / 16 / thread_n_blocks;
        int tiles2 = -1;
        kt_bpsm2 = !kt_bpsm2_split(k_tiles, n_tiles, group_blocks, thread_k_blocks, blocks, nullptr) &&
                   !kt_bpsm2_split(k_tiles, n_tiles, group_blocks, thread_k_blocks, 2 * blocks, &tiles2);
        if (!kt_bpsm2) {
          kt_bpsm2_tell(false, "split", prob_m_split, prob_n, prob_k, (int)a_type.size_bits(), num_bits, thread_m_blocks,
                        thread_n_blocks, thread_k_blocks, num_threads, blocks, 0, 0, kt_xs_selected, false);
        } else if (KT_BPSM2 == 2 && kt_xs_selected && tiles2 <= 1) {
          // No CTA holds a second tile at 2 * blocks, so the XS twin has no next slice to prefetch: take the served kernel
          // (bitwise the same output, layout without sh_red) when IT keeps two CTAs per SM; else keep XS (and the served grid).
          MarlinFuncPtr served = get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                                   thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
                                                   is_zp_float, stages);
          const int need = kt_marlin_smem_need(a_type.size_bits() == 8, num_bits, s_type.size_bits() == 8, has_zp,
                                               is_zp_float, thread_m_blocks, thread_n_blocks, thread_k_blocks,
                                               m_block_size_8, stages, group_blocks, false);
          if (served != MarlinDefault && need > 0 && need <= max_shared_mem_new &&
              kt_bpsm2_occ(served, num_threads, need) >= 2) {
            kernel = served;
            kt_xs_selected = false;
            kt_bpsm2_xs_dropped = true;
          }
        }
      }
    }
    // [0046 BPSM2] END plan

#if MARLIN_OPR
    // [0040 OPR] BEGIN twin selection. After every other twin selection (XS and any later one keep precedence: OPR re-selects
    // only a launch whose kernel is still the served Marlin<...>), before the attribute call so the launch sets the attribute
    // of the kernel it runs. VLLM_MARLIN_OPR unset/0: one untaken branch on a load-time constant; the served code follows.
    // The twin needs no smem beyond the served layout, so the attribute and the launch below are unchanged.
    if (KT_OPR && kt_a16 == nullptr) {
      const KtOprGate& og = kt_opr_gate();  // strict parse + ARMED tell; a malformed VLLM_MARLIN_OPR fails here
      if (prob_m_split <= 16 * thread_m_blocks && use_fp32_reduce &&
          !(use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048) &&
          (KT_OPR == 2 || thread_n_blocks != 16)) {
        bool n_ok = og.all_n;
        for (int x : og.ns)
          if (x == prob_n) n_ok = true;
        MarlinFuncPtr opr_kernel = nullptr;
        const char* opr_tag = "OPR";
  #if MARLIN_XSLICE
        // [0039 XSOPR] A launch the XS gate ALSO claims (VLLM_MARLIN_XSLICE=1, its M range and N list, no bias; the other XS
        // preconditions are the enclosing ones) takes the combined twin MarlinXSOPR when one exists, replacing the MarlinXS
        // twin the XS block may have selected (same smem layout for the W4A8 tiles; for the W8A16 o_proj, which has no
        // MarlinXS twin, the combined twin keeps the served layout). No combined twin: the 0040 rule below (OPR only if the
        // kernel is still the served one, so a MarlinXS selection keeps precedence). No new knob.
        if (n_ok && KT_XSLICE && !has_bias && kt_xslice_on(prob_m_split, prob_n, b_type == vllm::kU3S, nullptr)) {  // [0069]
          opr_kernel = get_marlin_xsopr_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                               thread_k_blocks, m_block_size_8, group_blocks, num_threads, is_zp_float,
                                               stages);
          if (opr_kernel != nullptr) opr_tag = "XSOPR";
        }
  #endif
        if (opr_kernel == nullptr && n_ok) {
          opr_kernel = get_marlin_opr_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                             thread_k_blocks, m_block_size_8, group_blocks, num_threads, is_zp_float,
                                             stages);
          if (opr_kernel != nullptr &&
              kernel != get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                          thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
                                          is_zp_float, stages))
            opr_kernel = nullptr;
        }
        if (opr_kernel != nullptr) {
          const KtOprPlan p = kt_opr_plan(prob_n, prob_k, thread_m_blocks, thread_n_blocks, thread_k_blocks,
                                          group_blocks, blocks);
          // p.smax <= 4 is the kernel's opr_max_slices: a selected twin always runs OPR, never its served-chain guard
          if (p.split_cols > 0 && p.smax <= 4 && p.slot_floats <= c_tmp_numel && p.late_max >= og.min_late) {
            kt_opr_tell_on(opr_tag, prob_m_split, prob_n, prob_k, (int)a_type.size_bits(), num_bits, p);
            kernel = opr_kernel;
          }
        }
      }
    }
    // [0040 OPR] END twin selection
#endif
#if MARLIN_SCALE_PF
    // [0041 MR14] BEGIN twin selection. LAST of the twin selections: it swaps the kernel the earlier ones chose (served, XS,
    // OPR, XSOPR) for that kernel's SP twin, so it stacks on each of them; before the attribute call, so the launch sets the
    // attribute of the kernel it runs (a twin uses its family's smem layout: same value). VLLM_MARLIN_SCALE_PREFETCH unset/0:
    // one untaken branch on a load-time constant; the code below is unchanged.
    if (KT_SP && kt_a16 == nullptr) {
      const KtSpGate& sg = kt_sp_gate();  // strict parse + ARMED tell; a malformed value fails here
      bool n_ok = sg.all_n;
      for (int x : sg.ns)
        if (x == prob_n) n_ok = true;
      if (n_ok && tile_major == 0 && prob_m_split <= 16 * thread_m_blocks) {
        const char* sp_name = "";
        MarlinFuncPtr sp_kernel = get_marlin_sp_twin(kernel, &sp_name);
        if (sp_kernel != nullptr) {
          kt_sp_tell_on(prob_m_split, prob_n, prob_k, (int)a_type.size_bits(), num_bits, group_size, sp_name);
          kernel = sp_kernel;
          tile_major = sg.bits << 1;
        }
      }
    }
    // [0041 MR14] END twin selection
#endif
#if MARLIN_FOLD
    // [0042 FOLD] BEGIN twin selection. A folded launch runs the MarlinFOLD twin of its config (the template arguments of the
    // served kernel chosen above: same smem layout, attribute and launch). No twin: drop the fold and re-plan these rows on the
    // served path (the loop re-enters with rest_m unchanged and kt_fold_drop set, so this rest_m gets the served prob_m_split
    // and its own config).
    if (kt_fold) {
      STD_TORCH_CHECK(kernel == get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                                  thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
                                                  is_zp_float, stages),
                      "[marlin-FOLD] a folded launch (prob_m_split ", prob_m_split, ") reached another twin selection");
      MarlinFuncPtr fold_kernel = get_marlin_fold_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                                         thread_k_blocks, m_block_size_8, group_blocks, num_threads,
                                                         is_zp_float, stages);
      const int fold_parallel = div_ceil(prob_m_split, 16 * thread_m_blocks);
      const int fold_last_off = prob_m_split - 16 * thread_m_blocks;
      kt_fold_tell(fold_kernel != nullptr, prob_n, prob_k, prob_m_split, fold_parallel, fold_last_off,
                   (int)a_type.size_bits(), num_bits, num_threads, thread_m_blocks, thread_n_blocks, thread_k_blocks,
                   group_blocks);
      if (fold_kernel == nullptr) {
        kt_fold_drop = true;
        continue;
      }
      kernel = fold_kernel;
    }
    // [0042 FOLD] END twin selection
#endif
#if MARLIN_FOLDS
    // [0048 FOLDS] BEGIN twin selection (mode S / SAR / SRA). The launch stays the served launch of prob_m_split = par_count * 64
    // rows (config, grid, tile_major) and, when every condition holds, runs that kernel's MarlinFOLDS twin over rest_m rows: the
    // served kernel or its XS twin (bitwise the same) only (an OPR / SP / BPSM2 launch keeps the served plan), a twin, a lock
    // buffer of 2 * blocks ints and C_tmp for 2 * blocks slots of this tile; PAIR needs par_count == 1 and an even grid.
    int kt_folds_rows = prob_m_split;  // rows this launch covers
    int* kt_folds_locks = locks;
    int kt_folds_tm = tile_major;
    if (kt_folds) {
      const char* why = nullptr;
      MarlinFuncPtr served_k = get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                                 thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
                                                 is_zp_float, stages);
      bool k_ok = kernel == served_k;
  #if MARLIN_XSLICE
      k_ok = k_ok || (kt_xs_selected && kernel == get_marlin_xs_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                                                                        thread_n_blocks, thread_k_blocks, m_block_size_8,
                                                                        group_blocks, num_threads, is_zp_float, stages));
  #endif
      MarlinFuncPtr folds_kernel = get_marlin_folds_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                                           thread_k_blocks, m_block_size_8, group_blocks, num_threads,
                                                           is_zp_float, stages);
      const int mode = kt_folds_armed();
      const int sched = (mode == 2) ? 0 : (mode == 3) ? 1 : (par_count == 1 && (blocks % 2) == 0) ? 2 : 0;
      const int64_t slot = static_cast<int64_t>(16 * thread_m_blocks) * (16 * thread_n_blocks);
      // [0048 FOLDS] mode SB: fold only when the served launch B of the remainder would take launch A's tile (thread_k,
      // thread_n, threads, CTAs per SM; thread_m_blocks may differ): the remainder item then walks launch B's own partition
      // (one copy, same k / n tiles, same grid, same K-warp split), so every row of the launch is bitwise the served plan's.
      // The selection below repeats this loop's own for rest_m - prob_m_split rows (auto config, SM-fill, A8 narrow tile);
      // a forced config (thread_k / thread_n set) is the same for both launches; no valid config = not the same.
      bool b_same = false;
      if (mode == 5) {
        const int r_rows = rest_m - prob_m_split;
        const int b_tmb = min(div_ceil(r_rows, 16), max_thread_m_blocks);
        const bool b_mbs8 = r_rows <= 8 && a_type.size_bits() == 16;
        thread_config_t bt{-1, -1, -1};
        int b_bps = 1;
        if (thread_k_init != -1 && thread_n_init != -1) {
          bt = thread_config_t{thread_k_init, thread_n_init, default_threads};
        } else {
          exec_config_t be = determine_exec_config(a_type, b_type, c_type, s_type, r_rows, prob_n, prob_k, b_tmb, b_mbs8,
                                                   num_bits, group_size, has_zp, is_zp_float, is_a_8bit, stages,
                                                   max_shared_mem, sms);
          bt = be.tb_cfg;
          b_bps = be.blocks_per_sm;
          if (bt.thread_n != -1 && !(KT_A8_NO_SMFILL && is_a_8bit) && KT_CFG_INDEX < 0 &&
              prob_n / bt.thread_n * div_ceil(r_rows, b_tmb * 16) * 4 <= sms &&
              is_valid_config({128, 64, 128}, b_tmb, r_rows, prob_n, prob_k, num_bits, group_size, has_zp, is_zp_float,
                              is_a_8bit, stages, max_shared_mem_new)) {
            bt = {128, 64, 128};
            b_bps = 1;
          }
          static const int fs_narrow_min_n = [] {
            const char* e = std::getenv("VLLM_MARLIN_A8_NARROW_MIN_N");
            return (e == nullptr || e[0] == '\0') ? 0 : static_cast<int>(std::strtol(e, nullptr, 10));
          }();
          static const int fs_narrow_max_m = [] {
            const char* e = std::getenv("VLLM_MARLIN_A8_NARROW_MAX_M");
            return (e == nullptr || e[0] == '\0') ? 64 : static_cast<int>(std::strtol(e, nullptr, 10));
          }();
          if (fs_narrow_min_n > 0 && is_a_8bit && b_tmb > 1 && r_rows <= fs_narrow_max_m && prob_n >= fs_narrow_min_n &&
              bt.thread_n != -1 && !(bt.thread_k == 128 && bt.thread_n == 64) &&
              is_valid_config({128, 64, 128}, b_tmb, r_rows, prob_n, prob_k, num_bits, group_size, has_zp, is_zp_float,
                              is_a_8bit, stages, max_shared_mem_new)) {
            bt = {128, 64, 128};
            b_bps = 1;
          }
        }
        b_same = bt.thread_k == thread_k && bt.thread_n == thread_n && bt.num_threads == num_threads &&
                 b_bps == exec_cfg.blocks_per_sm;
      }
      if (!k_ok) why = "kernel is an OPR/SP twin";
      else if (mode == 4 && par_count != 1) why = "S1: par_count >= 2 keeps the served launch B";
      else if (mode == 5 && !b_same) why = "SB: launch B would take another tile (remainder not bitwise)";
      else if (kt_bpsm2) why = "BPSM2 grid";
      else if (thread_m_blocks != 4 || m_block_size_8 || prob_m_split != par_count * 64) why = "not a 64-row tile launch";
      else if (folds_kernel == nullptr) why = "NO-TWIN";
      else if (folds_locks == nullptr || folds_locks_n < 2 * blocks) why = "no FOLDS lock buffer (2 * grid ints)";
      else if (c_tmp_numel < 2 * blocks * slot) why = "C_tmp below 2 * grid slots";
      if (why == nullptr) {
        kt_folds_tell(true, prob_n, prob_k, rest_m, "", par_count, rest_m - prob_m_split, sched, blocks);
        kernel = folds_kernel;
        kt_xs_selected = false;  // the twin has the served (non-XS) smem layout: 0038 TRIM sizes it as such
        kt_folds_rows = rest_m;
        kt_folds_locks = folds_locks;
        kt_folds_tm = tile_major | (sched << 1);
      } else {
        kt_folds_tell(false, prob_n, prob_k, rest_m, why, 0, 0, 0, blocks);
      }
    }
    // [0048 FOLDS] END twin selection
#endif
#if MARLIN_TAILPF
    // [0043 TAILPF] BEGIN twin selection. LAST of the selections (after FOLD): the TPF twin of the kernel the earlier ones chose
    // (served, XS, OPR, XSOPR; an SP or FOLD kernel has none). The launch block right before the served launch below runs it.
    // VLLM_MARLIN_TAIL_PREFETCH unset/0: one untaken branch on a load-time constant.
    MarlinTpfFuncPtr tpf_kernel = nullptr;
    const char* tpf_name = "";
    bool tpf_n_ok = false;
    if (KT_TPF) {
      const KtTpfGate& tg = kt_tpf_gate();  // strict parse + ARMED tell; a malformed value fails here
      tpf_n_ok = tg.all_n;
      for (int x : tg.ns)
        if (x == prob_n) tpf_n_ok = true;
      tpf_kernel = get_marlin_tpf_twin(kernel, &tpf_name);
    }
    // [0043 TAILPF] END twin selection
  #if MARLIN_KU3STPF
    // [0063 KU3STPF] BEGIN twin selection. A kS8 x kU3S launch has no 0043 TPF twin (tpf_kernel stays nullptr above); with
    // VLLM_MARLIN_TAILPF_KU3S=1 and TAILPF armed it takes the KU3STPF twin of its kernel (Marlin / MarlinXS; nullptr = none: the launch
    // keeps its kernel and still reports to the successor registry). Unset/0: one untaken branch on a load-time constant.
    bool kt_ku3s_tpf = false;
    if (KT_TPF_KU3S != 0 && b_type == vllm::kU3S) {
      kt_ku3s_tpf = kt_tpf_ku3s_gate(prob_n, prob_k);  // malformed -> refuse; TAIL_PREFETCH off -> false (INERT)
      if (kt_ku3s_tpf && tpf_kernel == nullptr) tpf_kernel = get_marlin_ku3stpf_twin(kernel, &tpf_name);
    }
    // [0063 KU3STPF] END twin selection
  #endif
#endif
#if MARLIN_ZPPACK
    // [0051 ZPPACK] BEGIN twin selection. After every other selection (the gate refused OPR / SP / FOLD / TAILPF / L2PF / BPSM2 at
    // arming, so the kernel here is the served Marlin<...> or its MarlinXS twin), before the attribute call so the launch sets the
    // attribute of the kernel it runs. The ZPP twin has its family's smem layout: 0038 TRIM's size and the launch are unchanged.
    if (kt_zpp_u >= 0) {
      const char* zpp_name = "";
      MarlinFuncPtr zpp_kernel = get_marlin_zpp_twin(kernel, &zpp_name);
      STD_TORCH_CHECK(zpp_kernel != nullptr, "[marlin-ZPPACK] packed weight N=", prob_n, " K=", prob_k, ": the launch (M ",
                      prob_m_split, ", thread_m_blocks ", thread_m_blocks, ", thread_n ", thread_n, ", thread_k ", thread_k,
                      ", threads ", num_threads, ", group_blocks ", group_blocks, ", XS ", (int)kt_xs_selected,
                      ") selected a kernel with no ZPP twin");
      kt_zpp_tell_on(prob_m_split, prob_n, prob_k, zpp_name);
      kernel = zpp_kernel;
      tile_major = (tile_major & 0xF) | (kt_zpp_u << 4);
  #if MARLIN_TAILPF
      // [0052 ZPPTPF] the TPF twin chosen above wraps the FAMILY kernel, which would read these packed words as the family's
      // words: take the TPF twin of the ZPP twin instead (nullptr = none: the launch keeps its ZPP twin and still reports
      // itself to the successor registry). TAILPF unarmed: tpf_kernel stays nullptr (one untaken branch).
      if (KT_TPF) tpf_kernel = get_marlin_zpptpf_twin(zpp_kernel, &tpf_name);
  #endif
  #if MARLIN_FOLDS
      // [0051 ZPPACK-4 BUG-554] the 0048 FOLDS launch passes kt_folds_tm, snapshotted BEFORE the ZPP u bits above; without this
      // every packed launch of a FOLDS-built .so reads u = 0 and traps on the zpp_seen guard (Xid 43, B1 638)
      kt_folds_tm = (kt_folds_tm & 0xF) | (kt_zpp_u << 4);
  #endif
    }
    // [0051 ZPPACK] END twin selection
#endif
#if MARLIN_COLFPF
    bool kt_colf = false;  // a column-factor launch with VLLM_MARLIN_COLFPF=1 (read by the TAILPF self-descriptor below)
    // [0061 COLFPF v2] BEGIN twin swap. After every other selection (ZPPACK included: a packed launch's kernel is its ZPP twin, its
    // TPF twin the ZPPTPF twin), before the attribute call so the launch sets the attribute of the kernel it runs. The CPF twin has
    // its sibling's parameters, smem layout and registers (the screen's bar), so 0038 TRIM, 0046 BPSM2, TAILPF's residency guard,
    // the 0047 tell and the launch below see the kernel that runs. Unset/0: one untaken branch on a load-time constant.
    if (g_s_ptr != nullptr && is_a_8bit &&
        (b_type == vllm::kU3S || ((b_type == vllm::kU4 || b_type == vllm::kU4B8) && group_blocks != -1))) {
      STD_TORCH_CHECK(KT_COLFPF >= 0, "[marlin-COLFPF] VLLM_MARLIN_COLFPF: malformed value (want 0 or 1)");
      if (KT_COLFPF == 1) {
        kt_colf = true;
        const char* cpf_name = "none";
        MarlinFuncPtr cpf_kernel = get_marlin_cpf_twin(kernel, &cpf_name);
        if (cpf_kernel != nullptr) kernel = cpf_kernel;
        const char* cpf_tpf_name = "none";
        bool cpf_tpf_on = false;
  #if MARLIN_TAILPF
        if (tpf_kernel != nullptr) {
          MarlinTpfFuncPtr cpf_tpf = get_marlin_cpf_tpf_twin(tpf_kernel, &cpf_tpf_name);
    #if MARLIN_KU3STPF
          // [0063 KU3STPF] a kU3S TPF twin's CPF twin (none: the launch keeps its TPF twin, TAILPF precedence; Part C below still puts
          // the factor in the predecessor's tail). Knob off: tpf_kernel is nullptr for every kU3S launch, this line is never reached.
          if (cpf_tpf == nullptr && kt_ku3s_tpf) cpf_tpf = get_marlin_ku3s_cpftpf_twin(tpf_kernel, &cpf_tpf_name);
    #endif
          if (cpf_tpf != nullptr) {
            tpf_kernel = cpf_tpf;
            tpf_name = cpf_tpf_name;
            cpf_tpf_on = true;
          }
        }
  #endif
        kt_colfpf_tell(prob_m_split, prob_n, prob_k, cpf_name, cpf_tpf_name, cpf_kernel != nullptr || cpf_tpf_on);
      }
    }
    // [0061 COLFPF v2] END twin swap
#endif

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

#if MARLIN_L2PF
    // [0045 L2PF] BEGIN twin selection. LAST of the twin selections: it swaps the kernel the earlier ones chose (served, XS,
    // OPR, XSOPR and their SP twins; a MarlinFOLD has no twin) for that kernel's L2PF twin and gives the twin the same
    // attribute (a twin has its family's smem layout), so 0038 TRIM, 0046 BPSM2's occupancy check, the [0047] tell and the
    // launch below all see the kernel that runs. tile_major and every launch argument are unchanged. The tell waits for the
    // grid / smem decisions (END launch of 0046 BPSM2). VLLM_MARLIN_L2PF unset/0: one untaken branch on a load-time constant.
    bool l2pf_listed = false;         // armed and this N listed: the tell below reports this launch
    const char* l2pf_name = nullptr;  // the L2PF twin this launch runs (nullptr: no twin, the kernel stays)
    if (KT_L2PF) {
      const KtL2pfGate& lg = kt_l2pf_gate();  // strict parse + ARMED tell; a malformed value fails here
      l2pf_listed = lg.all_n;
      for (int x : lg.ns)
        if (x == prob_n) l2pf_listed = true;
      if (l2pf_listed) {
        const char* l2pf_twin_name = "";
        MarlinFuncPtr l2pf_kernel = get_marlin_l2pf_twin(kernel, &l2pf_twin_name);
        if (l2pf_kernel != nullptr) {
          cudaFuncSetAttribute(l2pf_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new);
          kernel = l2pf_kernel;
          l2pf_name = l2pf_twin_name;
        }
      }
    }
    // [0045 L2PF] END twin selection
#endif

    // [0038 TRIM] served: the whole grant. Armed: the layout's own size, never above the grant.
    int launch_smem = max_shared_mem_new;
    if (KT_SMEM_TRIM) {
      const int need = kt_marlin_smem_need(
          a_type.size_bits() == 8, num_bits, s_type.size_bits() == 8, has_zp,
          is_zp_float, thread_m_blocks, thread_n_blocks, thread_k_blocks,
          m_block_size_8, stages, group_blocks, kt_xs_selected);
      // Keep the served one-Marlin-CTA-per-SM property (two CTAs of one grid on one SM would double that SM's share): never
      // request half the SM's shared memory or less. Binds only below M 49 (A8 M 32: 34,176 B -> the floor); every M 64 tile
      // needs more than the floor anyway (50,432-51,968 B v 50,304 on the 4090: 102,400 / 2 - 1,024 + 1, rounded to 128).
      int per_sm = 0, rsv = 0;
      cudaDeviceGetAttribute(&per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
      cudaDeviceGetAttribute(&rsv, cudaDevAttrReservedSharedMemoryPerBlock, dev);
      const int one_cta_floor = ((per_sm / 2 - rsv + 1) + 127) / 128 * 128;
      const int launch = need < one_cta_floor ? one_cta_floor : need;
      const bool fits = need > 0 && per_sm > 0 && launch <= max_shared_mem_new;
      if (fits) launch_smem = launch;
      kt_trim_tell(fits, (int)a_type.size_bits(), num_bits, has_zp, num_threads,
                   thread_m_blocks, thread_n_blocks, thread_k_blocks,
                   group_blocks, kt_xs_selected, need, fits ? launch : max_shared_mem_new,
                   max_shared_mem_new);
    }

    // [0046 BPSM2] BEGIN launch. The plan held: take 2 * blocks CTAs at the layout's own smem when the kernel this launch runs
    // (after every twin selection) keeps two CTAs resident per SM; else the served grid and the served / TRIM smem above.
    if (kt_bpsm2) {
      const int need = kt_marlin_smem_need(a_type.size_bits() == 8, num_bits, s_type.size_bits() == 8, has_zp, is_zp_float,
                                           thread_m_blocks, thread_n_blocks, thread_k_blocks, m_block_size_8, stages,
                                           group_blocks, kt_xs_selected);
      const bool smem_ok = need > 0 && need <= max_shared_mem_new;
      const int occ = smem_ok ? kt_bpsm2_occ(kernel, num_threads, need) : 0;
      const bool on = occ >= 2;
      kt_bpsm2_tell(on, smem_ok ? "occupancy" : "smem", prob_m_split, prob_n, prob_k, (int)a_type.size_bits(), num_bits,
                    thread_m_blocks, thread_n_blocks, thread_k_blocks, num_threads, on ? 2 * blocks : blocks, need, occ,
                    kt_xs_selected, kt_bpsm2_xs_dropped);
      if (on) {
        blocks = 2 * blocks;
        launch_smem = need;
      }
    }
    // [0046 BPSM2] END launch
    if (kt_a16 != nullptr) {
      // [0057 A16FS] the grid divides the tile count: tiles <= blocks gives part2 == tiles, else tiles % blocks == 0 gives part2 ==
      // blocks; either way iters == k_tiles (marlin_template.h part1/part2), so every column is ONE block over the whole K. Backstop:
      // no twin may have claimed the launch (their gates exclude A16FS shapes). [0062 A16TPF] A TPF twin is chosen after this check
      // (kernel stays the plain instance) and launches the same instance + tail prefetch with these blocks.
      STD_TORCH_CHECK(!kt_bpsm2 && kernel == get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
                                                                thread_k_blocks, m_block_size_8, has_zp, group_blocks, num_threads,
                                                                is_zp_float, stages),
                      "[marlin-A16FS] REFUSED: a twin claimed the A16FS launch N=", prob_n, " K=", prob_k);
      const int a16_par = prob_m_split > 16 * thread_m_blocks ? prob_m_split / (16 * thread_m_blocks) : 1;
      const int a16_tiles = a16_par * (prob_n / 16 / thread_n_blocks);
      int a16_blocks = a16_tiles < sms * exec_cfg.blocks_per_sm ? a16_tiles : sms * exec_cfg.blocks_per_sm;
      while (a16_tiles % a16_blocks != 0) a16_blocks--;
      static std::vector<std::pair<int, int>> kt_a16_told;
      bool told = false;
      for (const auto& t : kt_a16_told) told = told || (t.first == prob_n && t.second == prob_k);
      if (!told) {
        kt_a16_told.emplace_back(prob_n, prob_k);
        fprintf(stderr, "[marlin-A16FS] ON N=%d K=%d cfg {%d,%d,%d} (first M=%d: %d blocks for %d tiles, one block per column)\n",
                prob_n, prob_k, kt_a16->tk, kt_a16->tn, kt_a16->th, prob_m_split, a16_blocks, a16_tiles);
      }
      blocks = a16_blocks;
    }

#if MARLIN_L2PF
    // [0045 L2PF] BEGIN tell, after every grid / smem decision (0038 TRIM, 0046 BPSM2): ON for a launch that runs its L2PF twin,
    // NO-TWIN for a listed launch whose kernel has none (it keeps its kernel). With 0043 TAILPF in the tree, TAILPF-RUNS for a
    // listed launch that TAILPF's decision below runs as its TPF twin: the same kt_tpf_runs call on the same arguments (TAILPF
    // and L2PF are alternatives on one launch, no TPF x L2PF twin; its residency guard compares against the kernel here, the
    // L2PF twin, whose registers / smem are its family kernel's), so the tell names what the launch block runs.
    if (l2pf_listed) {
  #if defined(MARLIN_TAILPF) && MARLIN_TAILPF
      if (KT_TPF && kt_tpf_runs(tpf_kernel, tpf_n_ok, kernel, num_threads, max_shared_mem_new, launch_smem, prob_n, prob_k))
        kt_l2pf_tell_tpf(prob_m_split, prob_n, prob_k, tpf_name);
      else
  #endif
        kt_l2pf_tell(l2pf_name, prob_m_split, prob_n, prob_k, (int)a_type.size_bits(), num_bits, num_threads,
                     thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks);
    }
    // [0045 L2PF] END tell
#endif

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    MarlinFuncPtr kt_engage_kernel = kernel;
#if MARLIN_TAILPF
    // [0043 TAILPF] BEGIN decision (after every grid / smem change: 0038 TRIM, 0046 BPSM2). The TPF twin runs iff it exists,
    // its N is listed and it keeps the residency of the kernel it replaces at this grid / smem (kt_tpf_runs). Decided before
    // the [0047] tell, which then names the kernel the GPU executes.
    const bool kt_tpf_run = KT_TPF && (kt_a16 == nullptr || kt_tpf_gate().a16fs) && kt_tpf_runs(tpf_kernel, tpf_n_ok, kernel, num_threads, max_shared_mem_new,
                                                  launch_smem, prob_n, prob_k);
    if (kt_tpf_run) kt_engage_kernel = reinterpret_cast<MarlinFuncPtr>(tpf_kernel);
    // [0043 TAILPF] END decision
  #if MARLIN_KU3STPF
    // [0063 KU3STPF] per-config tell after TAILPF's decision: ON = the twin runs; NO-TWIN = the kernel has none (it runs, no tail);
    // NOT-RUN = it has one but TAILPF's own guard (VLLM_MARLIN_TAIL_PREFETCH_N list / residency, kt_tpf_runs) kept the kernel
    if (kt_ku3s_tpf)
      kt_tpf_ku3s_tell(kt_tpf_run ? "ON" : tpf_kernel == nullptr ? "NO-TWIN" : "NOT-RUN", prob_m_split, prob_n, prob_k, num_threads,
                       thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks, kt_xs_selected,
                       tpf_kernel == nullptr ? "none" : tpf_name);
  #endif
#endif
    // [0047 ENGAGE-TELL] the kernel this launch runs (after every twin selection); OFF = one untaken branch
    if (KT_ENGAGE_TELL) kt_engage_tell(kt_engage_kernel, prob_n, prob_k, prob_m);
#if MARLIN_G2
    // [0071 G2] arm the module of the kernel launched below (end of this file). A TPF launch runs its twin instead, armed in
    // kt_tpf_launch; `kernel` itself is loaded either way (cudaFuncSetAttribute above). #line: the served numbering (__LINE__).
    kt_g2_launch(reinterpret_cast<const void*>(kernel), stream);
#endif
#line 3155

    // avoid ">>>" being formatted to "> > >"
    // clang-format off
#if MARLIN_TAILPF
    // [0043 TAILPF] BEGIN launch. Armed: EVERY launch reports itself to the successor registry (it is the successor of the
    // launch before it on this stream); a launch with a TPF twin and a listed N runs the twin, with the descriptor of its own
    // successor as last learned, and the served launch below is skipped. Unset/0: one untaken branch, the served launch runs.
    // The twin launches exactly as the served launch would: [0038 TRIM]'s launch_smem as its dynamic smem and its
    // max_shared_mem argument, the full grant max_shared_mem_new as its attribute (a ceiling). 0043 sits above 0038.
    bool kt_tpf_done = false;
    if (KT_TPF) {
      const KtTpfGate& tg = kt_tpf_gate();
      KtTpfSelf tpf_self;
      tpf_self.weight = B_ptr;
      tpf_self.n = prob_n;
      tpf_self.k = prob_k;
      tpf_self.m = prob_m_split;
      const int tpf_pack = 32 / num_bits;
      const long long tpf_s_bytes = (long long)num_groups * prob_n * (long long)s_type.size_bits() / 8;
      const long long tpf_zp_bytes = !has_zp ? 0
                                     : is_zp_float ? (long long)num_groups * prob_n * 2
                                                   : (long long)num_groups * (prob_n / tpf_pack) * 4;
  #if MARLIN_ZPPACK
      // [0052 ZPPTPF] a PACKED weight's launch never reads its zero-point array (the nibbles ride in the scale words): its
      // predecessor's tail prefetches its scale lines only
      const long long tpf_zp_bytes_eff = kt_zpp_u >= 0 ? 0 : tpf_zp_bytes;
  #else
      const long long tpf_zp_bytes_eff = tpf_zp_bytes;
  #endif
      // [0061 COLFPF v2] a column-factor launch (knob on) that reads no zero points -- packed ZPP, symmetric kU4B8 / kU3S -- puts its
      // factor (fp32 [N]) in the zp span its PREDECESSOR's TPF tail prefetches: the factor reaches L2 before this launch starts.
      // A launch that reads zero points keeps them in the span (its CPF twin still prefetches the factor itself). Only within the
      // gate's budget: kt_tpf_self_desc drops the scale AND zp spans together when they exceed max_bytes, so a factor that does
      // not fit beside the scales is left out rather than costing the scale prefetch.
      const void* tpf_zp_src = zp_ptr;
      long long tpf_zp_span = tpf_zp_bytes_eff;
  #if MARLIN_COLFPF
      if (kt_colf && tpf_zp_bytes_eff == 0 && tpf_s_bytes + 4LL * prob_n <= tg.max_bytes) {
        tpf_zp_src = g_s_ptr;
        tpf_zp_span = 4LL * prob_n;
      }
  #endif
      bool tpf_fold = false;
  #if MARLIN_FOLD
      tpf_fold = kt_fold;
  #endif
      tpf_self.desc = kt_tpf_self_desc(KT_TPF, tg.b_stages, tg.max_bytes, tg.evict, b_s_ptr, tpf_s_bytes, tpf_zp_src,
                                       tpf_zp_span, B_ptr, prob_m_split, prob_n, prob_k,
                                       m_block_size_8 ? 8 : 16 * thread_m_blocks, tpf_fold, thread_k_blocks,
                                       thread_n_blocks, group_blocks, blocks, tile_major & 1, is_a_8bit,
                                       kt_b_bits == 3 ? 0 : num_bits);  // [0055 KU3S] no int4 B-line formula on int3 words
      // [0068 TPFOUT] a shape the gate names as a whole-B successor also gets its whole-B descriptor; the registry uses it only
      // when the launch before it has the listed carrier shape. Off (no pairs): one untaken branch on the gate's empty list.
      if (!tg.bfull.empty() && kt_tpf_bfull_succ(tg.bfull, prob_n, prob_k)) {
        const char* why = "";
        tpf_self.desc_bfull = kt_tpf_bfull_desc(tpf_self.desc, tg.bfull_max_bytes, tg.bfull_frac, tg.bfull_evict_last, B_ptr,
                                                prob_n, prob_k, thread_k_blocks, thread_n_blocks, is_a_8bit,
                                                kt_b_bits == 3 ? 0 : num_bits, &why);
        tpf_self.has_bfull = tpf_self.desc_bfull.b != nullptr;
        char key[96];
        snprintf(key, sizeof(key), "bfull/%d/%d/%d/%d/%d", prob_n, prob_k, thread_n_blocks, thread_k_blocks,
                 tpf_self.has_bfull ? 1 : 0);
        if (kt_tpf_first(key)) {
          const MarlinTpfDesc& e = tpf_self.desc_bfull;
          if (tpf_self.has_bfull)
            fprintf(stderr,
                    "[marlin-TAILPF] BFULL ON (0068) N=%d K=%d (first M=%d) tnb=%d tkb=%d b=%.2fMB vgrid=%d iters=%d stages=%d "
                    "s+zp=%.2fMB evict=%s\n",
                    prob_n, prob_k, prob_m_split, thread_n_blocks, thread_k_blocks,
                    static_cast<double>(e.grid) * e.stages * e.rows * e.div_lpr.d * 128.0 / (1 << 20), e.grid, e.iters,
                    e.stages, (e.s_lines + e.zp_lines) * 128.0 / (1 << 20), e.evict_last ? "last" : "normal");
          else
            fprintf(stderr, "[marlin-TAILPF] BFULL SKIP (0068) N=%d K=%d (first M=%d) tnb=%d tkb=%d: %s\n", prob_n, prob_k,
                    prob_m_split, thread_n_blocks, thread_k_blocks, why);
        }
      }
      KtTpfArgs tpf_args;
      tpf_args.A = A_ptr;
      tpf_args.B = B_ptr;
      tpf_args.C = C_ptr;
      tpf_args.C_tmp = C_tmp_ptr;
      tpf_args.bias = bias_ptr;
      tpf_args.a_s = a_s_ptr;
      tpf_args.b_s = b_s_ptr;
      tpf_args.g_s = g_s_ptr;
      tpf_args.zp = zp_ptr;
      tpf_args.prob_m = prob_m_split;
      tpf_args.prob_n = prob_n;
      tpf_args.prob_k = prob_k;
      tpf_args.lda = lda;
      tpf_args.locks = locks;
      tpf_args.has_bias = has_bias;
      tpf_args.use_atomic_add = part_use_atomic_add;
      tpf_args.use_fp32_reduce = use_fp32_reduce;
      tpf_args.max_shared_mem = launch_smem;
      tpf_args.tile_major = tile_major;
      kt_tpf_done = kt_tpf_launch(tpf_kernel, tpf_name, kt_tpf_run, stream, blocks, num_threads, max_shared_mem_new,
                                  launch_smem, tpf_self, tpf_args, (int)a_type.size_bits(), num_bits, thread_m_blocks,
                                  thread_n_blocks, thread_k_blocks, group_blocks);
    }
    if (!kt_tpf_done)
    // [0043 TAILPF] END launch
#endif
#if MARLIN_FOLDS
    // [0048 FOLDS] a FOLDS launch covers rest_m rows with its own lock buffer and the schedule in tile_major bits 1-2; any other
    // launch passes exactly the served arguments (kt_folds_* hold them)
    kernel<<<blocks, num_threads, launch_smem, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        kt_folds_rows, prob_n, prob_k, lda, kt_folds_locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, launch_smem, kt_folds_tm);
    prob_m_split = kt_folds_rows;
#else
    kernel<<<blocks, num_threads, launch_smem, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, launch_smem, tile_major);
#endif
    // clang-format on

    bool is_a_8bit = a_type.size_bits() == 8;
    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace marlin

#if MARLIN_ZPPACK
// [0051 ZPPACK-3 HOLD] The ZPPACK registry (kt_zpp_state) keys raw (scales, zp) pointers for the life of the process. Holding a
// reference to every (scales, zp) pair a ZPPACK-armed GEMM has seen means neither allocation is ever freed and handed to another
// tensor, so a key can never stand for a different tensor (closes design R3: reload / freed temporary -> trap or silent no-pack).
// Served weights live for the process anyway, so the hold costs no memory there. Heap-allocated and never destroyed: tensor
// destructors must not run at static-destruction time, after the CUDA context may be gone.
static void kt_zpp_hold(const torch::stable::Tensor& s, const torch::stable::Tensor& zp) {
  static std::mutex mu;
  static auto* held = new std::map<const void*, std::pair<torch::stable::Tensor, torch::stable::Tensor>>();
  std::lock_guard<std::mutex> lk(mu);
  const void* key = s.const_data_ptr();
  if (held->find(key) == held->end()) held->emplace(key, std::make_pair(s, zp));
}
#endif

torch::stable::Tensor marlin_gemm(
    torch::stable::Tensor& a, std::optional<torch::stable::Tensor> c_or_none,
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& b_bias_or_none,
    torch::stable::Tensor& b_scales,
    std::optional<torch::stable::Tensor> const& a_scales_or_none,
    std::optional<torch::stable::Tensor> const& global_scale_or_none,
    std::optional<torch::stable::Tensor> const& b_zeros_or_none,
    torch::stable::Tensor& workspace, vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float) {
  vllm::ScalarTypeId a_type_id, c_type_id, s_type_id;

  auto c_scalar_type = a.scalar_type();
  if (a.scalar_type() == torch::headeronly::ScalarType::Half) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_scalar_type = b_scales.scalar_type();
    if (b_scales.scalar_type() == torch::headeronly::ScalarType::Half) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() ==
               torch::headeronly::ScalarType::BFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();

      STD_TORCH_CHECK(c_or_none.has_value(), "c must be passed for W4A8-FP4");
      torch::stable::Tensor c = c_or_none.value();
      c_scalar_type = c.scalar_type();

      if (c.scalar_type() == torch::headeronly::ScalarType::Half) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        STD_TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == torch::headeronly::ScalarType::Char) {
      a_type_id = vllm::kS8.id();
    } else {
      STD_TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() ==
        torch::headeronly::ScalarType::Float8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() ==
               torch::headeronly::ScalarType::Float8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      STD_TORCH_CHECK(
          false, "When b_type = float4_e2m1f, b_scale scalar type must be",
          "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
    }
  } else if (b_type_id == vllm::kFE4M3fn.id() &&
             b_scales.scalar_type() ==
                 torch::headeronly::ScalarType::Float8_e8m0fnu) {
    s_type_id = vllm::kFE8M0fnu.id();
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
#if MARLIN_IRED
  if (a_type == vllm::kS8 && (use_atomic_add || !use_fp32_reduce)) {
    // [0056 IRED v2] int32 partials need the lock-chain reduce through the fp32-sized C_tmp: an atomic-add reduce would add int32 bits
    // as floats, the fp16 chain would read them as halves. Forced here, before C and C_tmp are allocated (the served compose sets
    // VLLM_MARLIN_USE_ATOMIC_ADD=1); the result is the exact integer reduce, order-free. Told once.
    static bool kt_ired_forced_told = false;
    if (!kt_ired_forced_told) {
      kt_ired_forced_told = true;
      fprintf(stderr, "[marlin-IRED] FORCED: int8-activation launch asked use_atomic_add=%d use_fp32_reduce=%d -> 0 / 1 (lock-chain int32 "
                      "reduce) (first M=%ld N=%ld K=%ld)\n", (int)use_atomic_add, (int)use_fp32_reduce, (long)size_m, (long)size_n,
              (long)size_k);
    }
    use_atomic_add = false;
    use_fp32_reduce = true;
  }
#endif
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);

  int pack_factor = 32 / b_type.size_bits();

  // Verify A
  STD_TORCH_CHECK(a.size(0) == size_m,
                  "Shape mismatch: a.size(0) = ", a.size(0),
                  ", size_m = ", size_m);
  STD_TORCH_CHECK(a.size(1) == size_k,
                  "Shape mismatch: a.size(1) = ", a.size(1),
                  ", size_k = ", size_k);

  // Verify B
  STD_TORCH_CHECK(
      size_k % MARLIN_NAMESPACE_NAME::tile_size == 0, "size_k = ", size_k,
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  STD_TORCH_CHECK(
      (size_k / MARLIN_NAMESPACE_NAME::tile_size) == b_q_weight.size(0),
      "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
      ", size_k = ", size_k,
      ", tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  STD_TORCH_CHECK(
      b_q_weight.size(1) % MARLIN_NAMESPACE_NAME::tile_size == 0,
      "b_q_weight.size(1) = ", b_q_weight.size(1),
      " is not divisible by tile_size = ", MARLIN_NAMESPACE_NAME::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / MARLIN_NAMESPACE_NAME::tile_size) * pack_factor;
  if (b_type == vllm::kU3S) {
    // [0055 KU3S] LOWBIT int3 words: b_q_weight int32 [K/16, 3N/2] (24 int4 per k32 x n32 tile), int8 activations only
    STD_TORCH_CHECK(a_type == vllm::kS8, "[0055 KU3S] b_type uint3b4 (kU3S) needs int8 activations, got ", a_type.str());
    STD_TORCH_CHECK(b_q_weight.size(1) % 3 == 0, "[0055 KU3S] b_q_weight.size(1) = ", b_q_weight.size(1),
                    " is not 3N/2");
    actual_size_n = static_cast<int>(b_q_weight.size(1) * 2 / 3);
  }
  STD_TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
                  ", actual_size_n = ", actual_size_n);

  // Verify device and strides
  STD_TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  STD_TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");
  // We use int4 (16 bytes) to load A, so A must aligned to 16 bytes
  STD_TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  STD_TORCH_CHECK(((uint64_t)a.const_data_ptr()) % 16 == 0,
                  "A must aligned to 16 bytes");

  STD_TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  STD_TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  STD_TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  STD_TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  torch::stable::Tensor a_scales;
  const auto device = a.device();

  if (a_scales_or_none.has_value()) {
    a_scales = a_scales_or_none.value();
    STD_TORCH_CHECK(a_type.size_bits() == 8,
                    "a_scales can only be used for 8bit activation.");
  } else {
    a_scales = torch::stable::empty({0}, torch::headeronly::ScalarType::Float,
                                    std::nullopt, device);
    STD_TORCH_CHECK(
        a_type.size_bits() != 8,
        "the a_scales parameter must be passed for 8bit activation.");
  }

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel
  int sms = -1;
  const int32_t device_index = a.get_device_index();
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_index);

  // Alloc buffers
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  torch::stable::Tensor c;
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    STD_TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    STD_TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
    STD_TORCH_CHECK(c.size(0) == size_m,
                    "Shape mismatch: c.size(0) = ", c.size(0),
                    ", size_m = ", size_m);
    STD_TORCH_CHECK(c.size(1) == size_n,
                    "Shape mismatch: c.size(1) = ", c.size(1),
                    ", size_n = ", size_n);
  } else {
    c = torch::stable::empty({size_m, size_n}, c_scalar_type, std::nullopt,
                             device);
    // cudagraph-replay safety: with use_atomic_add the kernel atomic-accumulates
    // into c, so it must start at the additive identity (0). An uninitialized c is
    // reused dirty across CUDA graph replays -> accumulation onto stale data -> NaN.
    if (use_atomic_add) {
      torch::stable::zero_(c);
    }
  }
  if (size_m == 0) return c;

  // Alloc C tmp buffer that is going to be used for the global reduce
  torch::stable::Tensor c_tmp;
  if (use_fp32_reduce) {
    int max_m_block_size = (size_m + 16 - 1) / 16 * 16;
    // sized by the ACTUAL ceiling, not a literal 64 -- see kt_max_thread_m_blocks()
    max_m_block_size = min(max_m_block_size,
                           16 * MARLIN_NAMESPACE_NAME::kt_max_thread_m_blocks());
    int max_c_tmp_size =
        sms * max_m_block_size * MARLIN_NAMESPACE_NAME::max_thread_n;
#if MARLIN_FOLDS
    // [0048 FOLDS] mode S armed: a FOLDS launch's remainder item uses C_tmp slots [grid, 2 * grid) (marlin_mm checks the size)
    const int kt_folds_ctmp_x = MARLIN_NAMESPACE_NAME::kt_folds_armed() ? 2 : 1;
    max_c_tmp_size *= kt_folds_ctmp_x;
#endif
    // [club-cuda-lane 2026-09-12] PERSISTENT c_tmp. A per-call empty() under CUDA
    // graph capture lands in each captured graph's private pool: with ~257 Marlin
    // launches per forward and a capture set of 6+ graphs the fp32-reduce scratch
    // alone raised the capture-phase peak by ~1 GiB (quant's paired boot on
    // p2070e8, 22,896 -> 23,876 MiB). The kernel never reads a slot before writing
    // it and two launches on one stream are ordered, so ONE buffer per
    // (device, stream), sized once at the largest M block (64 rows), serves every
    // call; the first use happens in eager warm-up, outside any graph pool, and the
    // captured graphs reference that fixed address. Different streams get their
    // own buffer (no cross-stream aliasing). VLLM_MARLIN_CTMP_PER_CALL=1 restores
    // the per-call allocation for bisection.
    static const bool kt_ctmp_per_call = [] {
      const char* v = getenv("VLLM_MARLIN_CTMP_PER_CALL");
      if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return false;   // [0072 STRICTPARSE] MRL-09: was *v=='1' ('1x' = on)
      STD_TORCH_CHECK(v[0] == '1' && v[1] == '\0', "VLLM_MARLIN_CTMP_PER_CALL: malformed value '", v, "' (want 0 or 1)");
      return true;
    }();
    if (kt_ctmp_per_call) {
      c_tmp = torch::stable::empty({max_c_tmp_size},
                                   torch::headeronly::ScalarType::Float,
                                   std::nullopt, device);
    } else {
      const int64_t persistent_size =
          static_cast<int64_t>(sms) *
          (16 * MARLIN_NAMESPACE_NAME::kt_max_thread_m_blocks()) *
#if MARLIN_FOLDS
          kt_folds_ctmp_x *  // [0048 FOLDS]
#endif
          MARLIN_NAMESPACE_NAME::max_thread_n;
      const cudaStream_t cur_stream = get_current_cuda_stream(device_index);
      static std::mutex ctmp_mutex;
      static std::map<std::pair<int, cudaStream_t>, torch::stable::Tensor>
          ctmp_cache;
      std::lock_guard<std::mutex> guard(ctmp_mutex);
      auto key = std::make_pair(static_cast<int>(device_index), cur_stream);
      auto it = ctmp_cache.find(key);
      if (it == ctmp_cache.end() || it->second.numel() < persistent_size) {
        it = ctmp_cache
                 .insert_or_assign(
                     key, torch::stable::empty(
                              {persistent_size},
                              torch::headeronly::ScalarType::Float,
                              std::nullopt, device))
                 .first;
      }
      c_tmp = it->second;
    }
    // [club-cuda-lane 2026-09-12] NO memset here. The cross-CTA fp32 reduce never
    // reads a c_tmp slot before writing it: the slice's first CTA (slice_idx == 0)
    // only WRITES (global_reduce_fp32(first=true)), every later CTA acquires the
    // lock the previous one released after its write, and the last CTA only READS.
    // Stale c_tmp is therefore unreachable while the lock protocol holds; the
    // upstream-PR-#45660 zero_() (never merged) cost sms*ceil16(M)*256 fp32 =
    // 6.0 MiB (M=48) / 8.0 MiB (M=1032) of memset PER LAUNCH -- ~257 launches per
    // decode step on Qwen3.8 -> ~1.5 GiB/step -- and, when the locks are broken,
    // turned a loud NaN into a silently WRONG partial sum. A broken lock is fixed
    // by locking the workspace after graph capture (upstream #55341), not here.
    // VLLM_MARLIN_ZERO_CTMP=1 restores the memset for bisection only.
    static const bool kt_zero_ctmp = [] {
      const char* v = getenv("VLLM_MARLIN_ZERO_CTMP");
      if (v == nullptr || *v == '\0' || (v[0] == '0' && v[1] == '\0')) return false;   // [0072 STRICTPARSE] MRL-09: was *v=='1' ('1x' = on)
      STD_TORCH_CHECK(v[0] == '1' && v[1] == '\0', "VLLM_MARLIN_ZERO_CTMP: malformed value '", v, "' (want 0 or 1)");
      return true;
    }();
    if (kt_zero_ctmp) torch::stable::zero_(c_tmp);
  } else {
    c_tmp = torch::stable::empty({0}, torch::headeronly::ScalarType::Float,
                                 std::nullopt, device);
  }

  // Detect group size.
  int num_groups = -1;
  int group_size = -1;

  int rank = b_scales.dim();
  STD_TORCH_CHECK(rank == 2, "b_scales rank = ", rank, " is not 2");
  STD_TORCH_CHECK(b_scales.size(1) == size_n,
                  "b_scales dim 1 = ", b_scales.size(1),
                  " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  if (num_groups > 1) {
    STD_TORCH_CHECK(
        size_k % num_groups == 0, "size_k = ", size_k,
        ", is not divisible by b_scales.size(0) = ", b_scales.size(0));
    group_size = size_k / num_groups;
  } else {
    group_size = -1;
  }

  torch::stable::Tensor global_scale;
  // [0059 W4RCOLF] W4R (int4 weights, int8 activations, grouped) takes the same optional fp32 [size_n] column factor as kU3S
  const bool w4r_colf = a_type == vllm::kS8 && (b_type == vllm::kU4 || b_type == vllm::kU4B8) && b_scales.size(0) > 1;
  if (global_scale_or_none.has_value() && (b_type == vllm::kU3S || w4r_colf)) {
    // [0055 KU3S-4] for kU3S global_scale_or_none is the optional per-output-column power-of-two factor: fp32 [size_n] on the device
    global_scale = global_scale_or_none.value();
    STD_TORCH_CHECK(global_scale.device().is_cuda() && global_scale.is_contiguous() && global_scale.dim() == 1 &&
                        global_scale.size(0) == size_n && global_scale.scalar_type() == torch::headeronly::ScalarType::Float,
                    "[0055 KU3S-4] the kU3S column factor must be a contiguous fp32 [size_n] cuda tensor");
    STD_TORCH_CHECK(!b_bias_or_none.has_value(), "[0055 KU3S-4] bias is not supported with a kU3S column factor");
    if (w4r_colf) {  // [0059 W4RCOLF] one tell per process
      static bool told = false;
      if (!told) { told = true; fprintf(stderr, "[marlin-W4RCOLF] column factor ON (0059): first N=%lld K=%lld\n", (long long)size_n, (long long)size_k); }
    }
  } else if (global_scale_or_none.has_value()) {
    global_scale = global_scale_or_none.value();
    STD_TORCH_CHECK(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn,
                    "global_scale can only be used for nvfp4 format.");
  } else {
    global_scale = torch::stable::empty(
        {0}, torch::headeronly::ScalarType::Float, std::nullopt, device);
    STD_TORCH_CHECK(
        !(b_type == vllm::kFE2M1f && s_type == vllm::kFE4M3fn),
        "the global_scale parameter must be passed for nvfp4 format.");
  }

  bool has_bias = b_bias_or_none.has_value();
  torch::stable::Tensor b_bias;
  if (has_bias) {
    b_bias = b_bias_or_none.value();
    STD_TORCH_CHECK(b_bias.device().is_cuda(), "b_bias is not on GPU");
    STD_TORCH_CHECK(b_bias.is_contiguous(), "b_bias is not contiguous");
    STD_TORCH_CHECK(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    STD_TORCH_CHECK(b_bias.stride(0) == 1, "b_bias.stride(0) != 1");
  } else {
    b_bias = torch::stable::empty({0}, c_scalar_type, std::nullopt, device);
  }

  torch::stable::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    STD_TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    STD_TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::stable::empty({0}, c_scalar_type, std::nullopt, device);
  }
  bool has_zp = b_zeros.size(-1) > 0;
  // [0055 KU3S] a symmetric int3 weight has no zero points. b_zeros_or_none may carry the KAT's device engage counter (one int32 on
  // the device; every CTA of every kU3S launch adds 1), accepted only with VLLM_MARLIN_KU3S_COUNT=1 so a served / speed launch never
  // carries it. The kernel gets nullptr otherwise.
  void* kt_zp_arg = b_zeros.mutable_data_ptr();
  if (b_type == vllm::kU3S) {
    static const bool kt_ku3s_count = [] {
      const char* v = getenv("VLLM_MARLIN_KU3S_COUNT");
      return v != nullptr && v[0] == '1' && v[1] == '\0';
    }();
    if (has_zp) {
      STD_TORCH_CHECK(kt_ku3s_count, "[0055 KU3S] b_zeros passed for kU3S without VLLM_MARLIN_KU3S_COUNT=1 (KAT-only counter)");
      STD_TORCH_CHECK(b_zeros.numel() == 1 && b_zeros.scalar_type() == torch::headeronly::ScalarType::Int,
                      "[0055 KU3S] the engage counter must be one int32");
    }
    STD_TORCH_CHECK(group_size == 32 || group_size == 64 || group_size == 128, "[0055 KU3S-5] group_size must be 32, 64 or 128, got ", group_size);
    kt_zp_arg = has_zp ? b_zeros.mutable_data_ptr() : nullptr;
    has_zp = false;
  }
  if (has_zp) {
    STD_TORCH_CHECK(
        b_type == vllm::kU4 || b_type == vllm::kU8,
        "b_type must be u4 or u8 when has_zp = True. Got = ", b_type.str());
  } else {
    STD_TORCH_CHECK(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 || b_type == vllm::kU3S ||
                        b_type == vllm::kS4 || b_type == vllm::kS8 ||
                        b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                    "b_type must be uint4b8, uint8b128, int4, int8, "
                    "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                    b_type.str());
  }

  if (has_zp && is_zp_float) {
    STD_TORCH_CHECK(
        a.scalar_type() == torch::headeronly::ScalarType::Half,
        "Computation type must be float16 (half) when using float zero "
        "points.");
  }

  // Verify b_zeros
  if (has_zp) {
    int rank = b_zeros.dim();
    STD_TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    if (is_zp_float) {
      STD_TORCH_CHECK(b_zeros.size(1) == size_n,
                      "b_zeros dim 1 = ", b_zeros.size(1),
                      " is not size_n = ", size_n);
      STD_TORCH_CHECK(num_groups == b_zeros.size(0),
                      "b_zeros dim 0 = ", b_zeros.size(0),
                      " is not num_groups = ", num_groups);
      STD_TORCH_CHECK(num_groups != -1, "num_groups must be != -1");
    } else {
      STD_TORCH_CHECK(b_zeros.size(0) == num_groups,
                      "b_zeros dim 0 = ", b_zeros.size(0),
                      " is not num_groups = ", num_groups);
      STD_TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                      "b_zeros dim 1 = ", b_zeros.size(1),
                      " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  // Verify workspace size
  STD_TORCH_CHECK(size_n % MARLIN_NAMESPACE_NAME::min_thread_n == 0,
                  "size_n = ", size_n, ", is not divisible by min_thread_n = ",
                  MARLIN_NAMESPACE_NAME::min_thread_n);

  int min_workspace_size = sms;
  STD_TORCH_CHECK(workspace.numel() >= min_workspace_size,
                  "workspace.numel = ", workspace.numel(),
                  " is below min_workspace_size = ", min_workspace_size);

  STD_TORCH_CHECK(
      a_scales.scalar_type() == torch::headeronly::ScalarType::Float,
      "scalar type of a_scales must be float");
  STD_TORCH_CHECK(
      global_scale.scalar_type() == torch::headeronly::ScalarType::Float,
      "scalar type of global_scale must be float");
  if (a_type.size_bits() == 16) {
    STD_TORCH_CHECK(
        a.scalar_type() == c.scalar_type(),
        "scalar type of a must be the same with c for 16 bit activation");
  }

#if MARLIN_FOLDS
  // [0048 FOLDS] mode S armed: a private lock buffer for FOLDS launches (served copies' locks in [0, grid), the remainder's in
  // [grid, 2 * grid)), persistent per (device, stream) like c_tmp and zeroed once at creation (the first Marlin GEMM, eager
  // warm-up); every lock returns to 0 when its column's final slice releases it. 4 * sms ints covers 2 * grid at <= 2 CTAs/SM.
  int* kt_folds_lk = nullptr;
  int kt_folds_lk_n = 0;
  if (use_fp32_reduce && MARLIN_NAMESPACE_NAME::kt_folds_armed()) {
    const int64_t n_lk = 4 * static_cast<int64_t>(sms);
    const cudaStream_t cur_stream = get_current_cuda_stream(device_index);
    static std::mutex folds_lk_mutex;
    static std::map<std::pair<int, cudaStream_t>, torch::stable::Tensor> folds_lk_cache;
    std::lock_guard<std::mutex> guard(folds_lk_mutex);
    auto key = std::make_pair(static_cast<int>(device_index), cur_stream);
    auto it = folds_lk_cache.find(key);
    if (it == folds_lk_cache.end()) {
      torch::stable::Tensor t = torch::stable::empty({n_lk}, torch::headeronly::ScalarType::Int, std::nullopt, device);
      torch::stable::zero_(t);
      it = folds_lk_cache.insert_or_assign(key, t).first;
    }
    kt_folds_lk = static_cast<int*>(it->second.mutable_data_ptr());
    kt_folds_lk_n = static_cast<int>(it->second.numel());
  }
#endif
#if MARLIN_ZPPACK
  if (marlin::KT_ZPP && has_zp) kt_zpp_hold(b_scales, b_zeros);  // [0051 ZPPACK-3 HOLD] before the registry can key them
#endif
  marlin::marlin_mm(
      a.const_data_ptr(), b_q_weight.const_data_ptr(), c.mutable_data_ptr(),
      c_tmp.mutable_data_ptr(), b_bias.mutable_data_ptr(),
      a_scales.mutable_data_ptr(), b_scales.mutable_data_ptr(),
      ((b_type == vllm::kU3S || w4r_colf) && !global_scale_or_none.has_value()) ? nullptr : global_scale.mutable_data_ptr(),  // [0055 KU3S-4] [0059 W4RCOLF]
      kt_zp_arg, size_m,
      size_n, size_k, a.stride(0), workspace.mutable_data_ptr(), a_type, b_type,
      c_type, s_type, has_bias, has_zp, num_groups, group_size, device_index,
      get_current_cuda_stream(device_index), thread_k, thread_n, sms,
#if MARLIN_FOLDS
      use_atomic_add, use_fp32_reduce, is_zp_float, c_tmp.numel(), kt_folds_lk, kt_folds_lk_n);
#else
      use_atomic_add, use_fp32_reduce, is_zp_float, c_tmp.numel());
#endif

  return c;
}

#endif

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("marlin_gemm", TORCH_BOX(&marlin_gemm));
}

#if MARLIN_G2
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <cuda.h>  // CUmodule / CUfunction / CUdeviceptr types; the two driver calls come through cudaGetDriverEntryPointByVersion

// [0071 G2 cuda 2026-10-12, BUG-562 guard] Host half of the split-K lock-leftover detector (device half: marlin_template.h
// barrier_acquire / marlin_g2_hit). VLLM_MARLIN_G2 (compose: ${VLLM_MARLIN_G2:-1}) = 1 DETECT | 2 TRAP | 0 OFF (explicit; "" also
// reads OFF) | unset = 1 | else a hard error at the first Marlin launch. OFF arms nothing: every module's __constant__
// marlin_g2_mode<0> stays 0, so barrier_acquire runs the served spin loop.
// DETECT / TRAP arm each kernel module at its first Marlin launch on a device (kt_g2_launch, called right before every launch): the
// module is found from the kernel (cudaGetFuncBySymbol + cuFuncGetModule) and its three G2 words by their template symbol names
// (cuModuleGetGlobal), so only modules a launch already loaded are touched (the served CUDA_MODULE_LOADING=LAZY loads no extra one).
// Once per process: one 4 KiB mapped pinned HOST page (byte 0 = "any hit", the module armed m-th gets the 16-B slot at 64 + 16 m;
// host memory, so no device memory and nothing in vLLM's memory profile). Per armed module: slot then mode, by cudaMemcpyAsync on a
// private non-blocking stream of that device + one sync of that stream, under a relaxed capture mode for this thread: nothing is
// enqueued on the launch stream, so a module armed while that stream captures never puts a node into the graph, and its kernels
// in the graph read the armed words when they run. Every launch then costs one lock-free table probe (dev, kernel) and one host
// load of the "any hit" word; the first hit is printed ONCE per process (or by the atexit hook if the process ends first: TRAP, or
// a hit during graph-only decode). marlin_g2_read_reset (extern "C", for KAT harnesses through ctypes) sums the device hit counts of
// the current device's armed modules and, with reset != 0, zeroes them + the page.
static int kt_g2_mode() {
  const char* v = getenv("VLLM_MARLIN_G2");
  if (v == nullptr) return 1;
  if (*v == '\0' || (v[0] == '0' && v[1] == '\0')) return 0;
  if ((v[0] == '1' || v[0] == '2') && v[1] == '\0') return v[0] - '0';
  return -1;
}
static const int KT_G2 = kt_g2_mode();
static const char* kt_g2_mode_name(int m) { return m == 2 ? "TRAP" : m == 1 ? "DETECT" : "OFF"; }
static constexpr unsigned int KT_G2_TAG = 0x4B4C3247u;  // "G2LK", written last into a valid slot (marlin_g2_hit)
static constexpr size_t KT_G2_PAGE = 4096, KT_G2_MAX_MODULES = (KT_G2_PAGE - 64) / 16;
static constexpr size_t KT_G2_TAB = 4096;  // (device, kernel) pairs already handled: insert-only, read lock-free
#define KT_G2_STR2(x) #x
#define KT_G2_STR(x) KT_G2_STR2(x)
struct KtG2Mod {
  int dev;
  CUmodule mod;
  CUdeviceptr hits;
};
static std::mutex kt_g2_mu;                         // guards everything below except the atomics
static std::vector<KtG2Mod> kt_g2_mods;              // armed modules; position = slot index
static std::atomic<volatile unsigned int*> kt_g2_page{nullptr};  // host view of the mapped page (set once)
static std::atomic<uintptr_t> kt_g2_tab[KT_G2_TAB];  // keys of handled (device, kernel) pairs; 0 = empty
static std::atomic<bool> kt_g2_told{false};
static cudaStream_t kt_g2_streams[64];               // private arm stream per device (created at its first arm)
using KtCuFuncGetModule = CUresult (*)(CUmodule*, CUfunction);
using KtCuModuleGetGlobal = CUresult (*)(CUdeviceptr*, size_t*, CUmodule, const char*);
static KtCuFuncGetModule kt_g2_cu_func_get_module = nullptr;
static KtCuModuleGetGlobal kt_g2_cu_module_get_global = nullptr;
// the device-side names of marlin_template.h's marlin_g2_<word><0> (namespace MARLIN_NAMESPACE_NAME)
static std::string kt_g2_sym(const char* word) {
  const std::string ns = KT_G2_STR(MARLIN_NAMESPACE_NAME), w = std::string("marlin_g2_") + word;
  return "_ZN" + std::to_string(ns.size()) + ns + std::to_string(w.size()) + w + "ILi0EEE";
}
static uintptr_t kt_g2_key(const void* kernel, int dev) {
  return reinterpret_cast<uintptr_t>(kernel) ^ (static_cast<uintptr_t>(dev + 1) << 56);
}
static size_t kt_g2_slot_of(uintptr_t key) { return static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 52); }  // 12 bits
static bool kt_g2_seen(uintptr_t key) {
  for (size_t i = kt_g2_slot_of(key), n = 0; n < KT_G2_TAB; i = (i + 1) & (KT_G2_TAB - 1), n++) {
    const uintptr_t v = kt_g2_tab[i].load(std::memory_order_acquire);
    if (v == key) return true;
    if (v == 0) return false;
  }
  return false;
}
static void kt_g2_insert(uintptr_t key) {  // under kt_g2_mu; a full table only means this pair takes the slow path again
  for (size_t i = kt_g2_slot_of(key), n = 0; n < KT_G2_TAB; i = (i + 1) & (KT_G2_TAB - 1), n++) {
    const uintptr_t v = kt_g2_tab[i].load(std::memory_order_relaxed);
    if (v == key) return;
    if (v == 0) {
      kt_g2_tab[i].store(key, std::memory_order_release);
      return;
    }
  }
}
struct KtG2Relaxed {  // like c10::cuda::CUDAStreamCaptureModeGuard: this thread may be inside a capture while it arms
  cudaStreamCaptureMode m = cudaStreamCaptureModeRelaxed;
  KtG2Relaxed() { (void)cudaThreadExchangeStreamCaptureMode(&m); }
  ~KtG2Relaxed() { (void)cudaThreadExchangeStreamCaptureMode(&m); }
};
static void kt_g2_print(const char* when) {
  volatile unsigned int* page = kt_g2_page.load(std::memory_order_acquire);
  for (size_t m = 0; m < KT_G2_MAX_MODULES; m++) {
    volatile unsigned int* s = page + 16 + 4 * m;
    if (s[0] != KT_G2_TAG) continue;
    fprintf(stderr,
            "[marlin-G2] LOCK-LEFTOVER (0071)%s: a split-K lock read %u > slice index %u (module %zu, lock 0x%08x, mode %s). The lock held a "
            "value its own chain never wrote: one Marlin workspace used by two unordered streams, or a launch aborted mid-chain. That "
            "layer's outputs are WRONG until its workspace is zeroed (BUG-562). Told once per process.\n",
            when, s[1], s[2], m, s[3], kt_g2_mode_name(KT_G2));
    return;
  }
  fprintf(stderr, "[marlin-G2] LOCK-LEFTOVER (0071)%s: a hit is flagged but no slot is visible yet (mode %s; BUG-562). Told once per "
                  "process.\n", when, kt_g2_mode_name(KT_G2));
}
static void kt_g2_atexit() {
  volatile unsigned int* page = kt_g2_page.load(std::memory_order_acquire);
  if (page != nullptr && page[0] != 0u && !kt_g2_told.exchange(true)) kt_g2_print(" at exit");
}
static void kt_g2_not_covered(int dev, const void* kernel, const char* why) {  // under kt_g2_mu; once per (device, kernel)
  fprintf(stderr, "[marlin-G2] NOT-COVERED (0071): dev=%d kernel=%p: %s; this kernel's lock reads are not checked\n", dev, kernel, why);
}
static void kt_g2_arm(const void* kernel, int dev, uintptr_t key) {
  std::lock_guard<std::mutex> lk(kt_g2_mu);
  if (kt_g2_seen(key)) return;
  const char* v = getenv("VLLM_MARLIN_G2");
  STD_TORCH_CHECK(KT_G2 > 0, "VLLM_MARLIN_G2: malformed value '", v != nullptr ? v : "", "' (want 0, 1 or 2)");
  KtG2Relaxed relaxed;  // nothing below is enqueued on the launch stream
  volatile unsigned int* page = kt_g2_page.load(std::memory_order_relaxed);
  if (page == nullptr) {
    fprintf(stderr, "[marlin-G2] KNOB (0071) VLLM_MARLIN_G2 state=on src=%s value=%d\n", v == nullptr ? "default" : "env", KT_G2);
    cudaDriverEntryPointQueryResult q1 = cudaDriverEntryPointSymbolNotFound, q2 = cudaDriverEntryPointSymbolNotFound;
    void* f1 = nullptr;
    void* f2 = nullptr;
    STD_TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuFuncGetModule", &f1, 12000, cudaEnableDefault, &q1) == cudaSuccess &&
                        q1 == cudaDriverEntryPointSuccess && f1 != nullptr &&
                        cudaGetDriverEntryPointByVersion("cuModuleGetGlobal", &f2, 12000, cudaEnableDefault, &q2) == cudaSuccess &&
                        q2 == cudaDriverEntryPointSuccess && f2 != nullptr,
                    "[marlin-G2] ARM: driver entry points cuFuncGetModule / cuModuleGetGlobal unavailable (", q1, "/", q2, ")");
    kt_g2_cu_func_get_module = reinterpret_cast<KtCuFuncGetModule>(f1);
    kt_g2_cu_module_get_global = reinterpret_cast<KtCuModuleGetGlobal>(f2);
    void* h = nullptr;
    STD_TORCH_CHECK(cudaHostAlloc(&h, KT_G2_PAGE, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess,
                    "[marlin-G2] ARM: cudaHostAlloc of the mapped page failed");
    memset(h, 0, KT_G2_PAGE);
    page = static_cast<volatile unsigned int*>(h);
    kt_g2_page.store(page, std::memory_order_release);
    std::atexit(kt_g2_atexit);
  }
  cudaFunction_t fn = nullptr;
  CUmodule mod = nullptr;
  if (cudaGetFuncBySymbol(&fn, kernel) != cudaSuccess || fn == nullptr) {
    (void)cudaGetLastError();
    kt_g2_not_covered(dev, kernel, "cudaGetFuncBySymbol failed");
  } else if (kt_g2_cu_func_get_module(&mod, reinterpret_cast<CUfunction>(fn)) != CUDA_SUCCESS || mod == nullptr) {
    kt_g2_not_covered(dev, kernel, "cuFuncGetModule failed");
  } else {
    bool armed = false;
    for (const auto& m : kt_g2_mods) armed |= m.dev == dev && m.mod == mod;
    if (!armed) {
      CUdeviceptr p_mode = 0, p_slot = 0, p_hits = 0;
      size_t b_mode = 0, b_slot = 0, b_hits = 0;
      if (kt_g2_cu_module_get_global(&p_mode, &b_mode, mod, kt_g2_sym("mode").c_str()) != CUDA_SUCCESS ||
          kt_g2_cu_module_get_global(&p_slot, &b_slot, mod, kt_g2_sym("slot").c_str()) != CUDA_SUCCESS ||
          kt_g2_cu_module_get_global(&p_hits, &b_hits, mod, kt_g2_sym("hits").c_str()) != CUDA_SUCCESS || b_mode != 4 ||
          b_slot != 8 || b_hits != 4) {
        kt_g2_not_covered(dev, kernel, "its module has no marlin_g2 words");
      } else if (kt_g2_mods.size() >= KT_G2_MAX_MODULES) {
        kt_g2_not_covered(dev, kernel, "the hit page is full");
      } else {
        void* d = nullptr;
        STD_TORCH_CHECK(cudaHostGetDevicePointer(&d, const_cast<unsigned int*>(page), 0) == cudaSuccess,
                        "[marlin-G2] ARM: no device address for the mapped page (dev ", dev, ")");
        if (kt_g2_streams[dev] == nullptr)
          STD_TORCH_CHECK(cudaStreamCreateWithFlags(&kt_g2_streams[dev], cudaStreamNonBlocking) == cudaSuccess,
                          "[marlin-G2] ARM: private stream (dev ", dev, ")");
        const cudaStream_t s = kt_g2_streams[dev];
        const unsigned long long slot = reinterpret_cast<unsigned long long>(d) + 64 + 16 * kt_g2_mods.size();
        const unsigned int mode = static_cast<unsigned int>(KT_G2);
        STD_TORCH_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(p_slot), &slot, 8, cudaMemcpyHostToDevice, s) == cudaSuccess &&
                            cudaMemcpyAsync(reinterpret_cast<void*>(p_mode), &mode, 4, cudaMemcpyHostToDevice, s) == cudaSuccess &&
                            cudaStreamSynchronize(s) == cudaSuccess,
                        "[marlin-G2] ARM: writing the G2 words failed (dev ", dev, ")");
        kt_g2_mods.push_back(KtG2Mod{dev, mod, p_hits});
        fprintf(stderr, "[marlin-G2] ARMED (0071): mode=%s dev=%d module=%zu\n", kt_g2_mode_name(KT_G2), dev, kt_g2_mods.size() - 1);
      }
    }
  }
  kt_g2_insert(key);
}
void MARLIN_NAMESPACE_NAME::kt_g2_launch(const void* kernel, CUstream_st* stream) {
  (void)stream;  // the launch stream is never used: arming goes through the private stream
  if (KT_G2 == 0) {  // OFF: nothing armed, every module's mode stays 0 (the served spin loop)
    static std::atomic<bool> knob_told{false};
    if (!knob_told.load(std::memory_order_relaxed) && !knob_told.exchange(true)) {
      const char* v = getenv("VLLM_MARLIN_G2");
      fprintf(stderr, "[marlin-G2] KNOB (0071) VLLM_MARLIN_G2 state=off src=env value=%s\n", v != nullptr ? v : "");
    }
    return;
  }
  int dev = -1;
  STD_TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64, "[marlin-G2] cudaGetDevice failed");
  const uintptr_t key = kt_g2_key(kernel, dev);
  if (!kt_g2_seen(key)) kt_g2_arm(kernel, dev, key);
  volatile unsigned int* page = kt_g2_page.load(std::memory_order_acquire);
  if (page != nullptr && page[0] != 0u && !kt_g2_told.exchange(true)) kt_g2_print("");
}
// For KAT harnesses (ctypes on this .so): the hits of every armed module on the current device, summed; reset != 0 also zeroes them
// and the page (and re-arms the once-per-process tell). -1 = nothing armed on this device (OFF or no Marlin launch yet); -2 = the
// device sync failed (a TRAP hit poisons the context); -3 = a copy failed.
extern "C" __attribute__((visibility("default"))) long long marlin_g2_read_reset(int reset) {
  std::lock_guard<std::mutex> lk(kt_g2_mu);
  int dev = -1;
  volatile unsigned int* page = kt_g2_page.load(std::memory_order_acquire);
  if (KT_G2 <= 0 || page == nullptr || cudaGetDevice(&dev) != cudaSuccess) return -1;
  size_t on_dev = 0;
  for (const auto& m : kt_g2_mods) on_dev += m.dev == dev;
  if (on_dev == 0) return -1;
  if (cudaDeviceSynchronize() != cudaSuccess) return -2;
  long long total = 0;
  for (const auto& m : kt_g2_mods) {
    if (m.dev != dev) continue;
    unsigned int h = 0;
    if (cudaMemcpy(&h, reinterpret_cast<const void*>(m.hits), sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) return -3;
    total += h;
    if (reset) {
      const unsigned int z = 0;
      if (cudaMemcpy(reinterpret_cast<void*>(m.hits), &z, sizeof z, cudaMemcpyHostToDevice) != cudaSuccess) return -3;
    }
  }
  if (reset) {
    for (size_t i = 0; i < KT_G2_PAGE / 4; i++) page[i] = 0u;
    kt_g2_told.store(false);
  }
  return total;
}
#endif
