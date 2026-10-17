// [0055 KU3S-1] ku3s_repack_ct: compressed-tensors pack-quantized SYMMETRIC int3 (the IST-DASLab GSQ checkpoints) -> the kU3S
// Marlin B layout (marlin_template.h is_u3 path, LOWBIT words: lowbit_unpack.cuh / lowbit_pack.py 1d7e2c30f997a4a5).
//
// Input, AS STORED in the checkpoint: int32 [N, K*3/32], one dense LSB-first 3-bit stream per output row n (value k at bits
// 3k .. 3k+2 of the row, crossing word edges), stored code c = w + 4, w in -4..3.
// Output: int32 [K/16, 3N/2] = K/32 x N/32 tiles of 96 words in Marlin's tile order (tile (kt, nt) at word (kt * N/32 + nt) * 96):
// plane A = 32 lanes x {W0, W1}, plane B = 32 lanes x W2. Lane `lane` of tile (kt, nt) holds the 32 weights of the served kS8
// fragment map (byte i of register R[o] = W[32 kt + 4 (lane % 4) + 16 (o % 2) + i, 32 nt + 16 (o / 4) + 8 ((o / 2) % 2) + lane / 4]),
// encoded with LOWBIT's PLAN3 bit plan and ENC3 tables (brute-forced from the device unpack's PTX emulation, so this is the exact
// inverse of lowbit::unpack3). Integer only, one thread per (tile, lane), no shared state: deterministic and idempotent.
#include "marlin.cuh"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "torch_utils.h"

namespace marlin {

// ENC3[jx][c] = (s << 2) | (t << 1) | l for stored code c (w = c - 4); lowbit_pack.ENC3 re-indexed by c
__constant__ uint8_t ku3s_enc3[2][8] = {{2, 3, 7, 6, 4, 5, 1, 0}, {2, 7, 3, 6, 4, 1, 5, 0}};

__global__ void ku3s_repack_ct_kernel(uint32_t const* __restrict__ ct, uint32_t* __restrict__ out, int size_k,
                                      int size_n, int row_words) {
  const long long gid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n_tiles = size_n / 32;
  const long long tiles = static_cast<long long>(size_k / 32) * n_tiles;
  if (gid >= tiles * 32) return;
  const long long tile = gid >> 5;
  const int lane = static_cast<int>(gid & 31);
  const int kt = static_cast<int>(tile / n_tiles);
  const int nt = static_cast<int>(tile % n_tiles);
  // LOWBIT PLAN3 per register o: PRMT word, view offset, XOR word, XOR shift, jx
  constexpr int pw[8] = {0, 0, 1, 1, 0, 0, 2, 2};
  constexpr int voff[8] = {0, 16, 0, 16, -2, 14, 0, 16};
  constexpr int xw[8] = {1, 1, 1, 1, 2, 2, 2, 2};
  constexpr int xsh[8] = {0, 0, 4, 4, 0, 0, 4, 4};
  constexpr int jx[8] = {0, 1, 0, 1, 0, 1, 0, 1};
  uint32_t w[3] = {0u, 0u, 0u};
#pragma unroll
  for (int o = 0; o < 8; o++) {
    const int n = 32 * nt + 16 * (o / 4) + 8 * ((o / 2) % 2) + lane / 4;
    uint32_t const* row = ct + static_cast<size_t>(n) * row_words;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int k = 32 * kt + 4 * (lane % 4) + 16 * (o % 2) + i;
      const int p = 3 * k, wi = p >> 5, sh = p & 31;
      uint32_t c = row[wi] >> sh;
      if (sh > 29) c |= row[wi + 1] << (32 - sh);
      c &= 7u;
      const uint32_t e = ku3s_enc3[jx[o]][c];
      w[pw[o]] |= ((e >> 2) << (4 * i + 3 + voff[o])) | (((e >> 1) & 1u) << (4 * i + 2 + voff[o]));
      w[xw[o]] |= (e & 1u) << (8 * i + xsh[o] + jx[o]);
    }
  }
  const long long base = tile * 96;
  out[base + 2 * lane] = w[0];
  out[base + 2 * lane + 1] = w[1];
  out[base + 64 + lane] = w[2];
}

}  // namespace marlin

torch::stable::Tensor ku3s_repack_ct(torch::stable::Tensor& ct_packed, int64_t size_k, int64_t size_n) {
  STD_TORCH_CHECK(size_k > 0 && size_n > 0 && size_k % 32 == 0 && size_n % 32 == 0,
                  "[0055 KU3S] ku3s_repack_ct: size_k = ", size_k, ", size_n = ", size_n, " must be positive multiples of 32");
  STD_TORCH_CHECK(ct_packed.is_cuda(), "[0055 KU3S] ct_packed is not on GPU");
  STD_TORCH_CHECK(ct_packed.is_contiguous(), "[0055 KU3S] ct_packed is not contiguous");
  STD_TORCH_CHECK(ct_packed.scalar_type() == torch::headeronly::ScalarType::Int, "[0055 KU3S] ct_packed is not int32");
  STD_TORCH_CHECK(ct_packed.dim() == 2 && ct_packed.size(0) == size_n && ct_packed.size(1) == size_k * 3 / 32,
                  "[0055 KU3S] ct_packed must be int32 [N, K*3/32] (dense compressed-tensors int3) = [", size_n, ", ",
                  size_k * 3 / 32, "], got [", ct_packed.size(0), ", ", ct_packed.dim() == 2 ? ct_packed.size(1) : -1, "]");
  STD_TORCH_CHECK(size_k * size_n / 32 < (int64_t{1} << 31), "[0055 KU3S] tensor too large for the 32-bit tile index");
  const int32_t device_index = ct_packed.get_device_index();
  torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  torch::stable::Tensor out = torch::stable::empty({size_k / 16, size_n * 3 / 2}, torch::headeronly::ScalarType::Int,
                                                   std::nullopt, ct_packed.device());
  const long long threads_total = size_k * size_n / 32;  // (K/32) (N/32) tiles x 32 lanes
  const int block = 256;
  const unsigned grid = static_cast<unsigned>((threads_total + block - 1) / block);
  marlin::ku3s_repack_ct_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<uint32_t const*>(ct_packed.const_data_ptr()), reinterpret_cast<uint32_t*>(out.mutable_data_ptr()),
      static_cast<int>(size_k), static_cast<int>(size_n), static_cast<int>(size_k * 3 / 32));
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "[0055 KU3S] ku3s_repack_ct launch failed");
  return out;
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) { m.impl("ku3s_repack_ct", TORCH_BOX(&ku3s_repack_ct)); }
