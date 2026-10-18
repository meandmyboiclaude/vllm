#include <torch/csrc/stable/tensor.h>

#include <cmath>

#include "../../../dispatch_utils.h"
#include "../../../torch_utils.h"
#include "../../vectorization_utils.cuh"
#include "../../../cub_helpers.h"

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // dst = std::clamp(dst, i8_min, i8_max);
  dst = (dst < i8_min) ? i8_min : (dst > i8_max) ? i8_max : dst;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

static inline __device__ int32_t float_to_int32_rn(float x) {
#ifdef USE_ROCM
  // int32_max is not exactly representable as float.
  // Therefore, we need to be careful and manually return int32_max on overflow.
  // For symmetry, we also do the same for int32_min, even though it is exactly
  // representable as float and the conversion should be exact.
  static constexpr auto i32_min = std::numeric_limits<int32_t>::min();
  static constexpr auto i32_min_f = static_cast<float>(i32_min);
  static constexpr auto i32_max = std::numeric_limits<int32_t>::max();
  static constexpr auto i32_max_f = static_cast<float>(i32_max);

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate on the higher end.
  if (dst >= i32_max_f) {
    return i32_max;
  }
  // saturate on the lower end.
  if (dst <= i32_min_f) {
    return i32_min;
  }

  return static_cast<int32_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int32_t&>(dst);
#endif
}

static inline __device__ int8_t int32_to_int8(int32_t x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<int32_t>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<int32_t>(std::numeric_limits<int8_t>::max());

  // saturate
  // See https://github.com/pytorch/pytorch/issues/127666
  // See https://github.com/llvm/llvm-project/issues/95183
  // hip-clang std::clamp __glibcxx_assert_fail host function when building on
  // Arch/gcc14. The following replaces std::clamp usage with similar logic
  // int32_t dst = std::clamp(x, i8_min, i8_max);
  int32_t dst = (x < i8_min) ? i8_min : (x > i8_max) ? i8_max : x;
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(dst) : "r"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_t>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    const scale_t* scale_ptr, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;
  const float scale = *scale_ptr;

  // Must be performed using 64-bit math to avoid integer overflow.
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        dst = float_to_int8_rn(static_cast<float>(src) / scale);
      });
}

template <typename scalar_t, typename scale_t, typename azp_t>
__global__ void static_scaled_int8_azp_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    const scale_t* scale_ptr, const azp_t* azp_ptr, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;
  const float scale = *scale_ptr;
  const azp_t azp = *azp_ptr;
  const float inv_s = 1.0f / scale;

  // Must be performed using 64-bit math to avoid integer overflow.
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        const auto v = static_cast<float>(src) * inv_s;
        dst = int32_to_int8(float_to_int32_rn(v) + azp);
      });
}

template <typename scalar_t, typename scale_t>
__global__ void dynamic_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    scale_t* scale_out, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;

  // Must be performed using 64-bit math to avoid integer overflow.
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  // calculate for absmax
  float thread_max = 0.f;
  vectorize_read_with_alignment<16>(
      row_in, hidden_size, tid, stride, [&] __device__(const scalar_t& src) {
        const float v = fabsf(static_cast<float>(src));
        thread_max = fmaxf(thread_max, v);
      });
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);
  __shared__ float absmax;
  if (tid == 0) {
    absmax = block_max;
    scale_out[blockIdx.x] = absmax / 127.f;
  }
  __syncthreads();

  float inv_s = (absmax == 0.f) ? 0.f : 127.f / absmax;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        dst = float_to_int8_rn(static_cast<float>(src) * inv_s);
      });
}

// MinMax structure to hold min and max values in one go
struct MinMax {
  float min, max;

  __host__ __device__ MinMax()
      : min(std::numeric_limits<float>::max()),
        max(std::numeric_limits<float>::lowest()) {}

  __host__ __device__ explicit MinMax(float v) : min(v), max(v) {}

  __host__ __device__ MinMax& operator+=(float v) {
    min = fminf(min, v);
    max = fmaxf(max, v);
    return *this;
  }

  // merge two MinMax objects
  __host__ __device__ MinMax& operator&=(const MinMax& other) {
    min = fminf(min, other.min);
    max = fmaxf(max, other.max);
    return *this;
  }
};

__host__ __device__ inline MinMax operator+(MinMax a, float v) {
  return a += v;
}
__host__ __device__ inline MinMax operator&(MinMax a, const MinMax& b) {
  return a &= b;
}

template <typename scalar_t, typename scale_t, typename azp_t>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ output,
    scale_t* scale_out, azp_t* azp_out, const int hidden_size) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;

  // Must be performed using 64-bit math to avoid integer overflow.
  const scalar_t* row_in = input + token_idx * hidden_size;
  int8_t* row_out = output + token_idx * hidden_size;

  MinMax thread_mm;
  vectorize_read_with_alignment<16>(row_in, hidden_size, tid, stride,
                                    [&] __device__(const scalar_t& src) {
                                      thread_mm += static_cast<float>(src);
                                    });

  using BlockReduce = cub::BlockReduce<MinMax, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;

  MinMax mm = BlockReduce(tmp).Reduce(
      thread_mm,
      [] __device__(MinMax a, const MinMax& b) {
        a &= b;
        return a;
      },
      blockDim.x);

  __shared__ float scale_sh;
  __shared__ azp_t azp_sh;
  if (tid == 0) {
    float s = (mm.max - mm.min) / 255.f;
    float zp = nearbyintf(-128.f - mm.min / s);  // round-to-even
    scale_sh = s;
    azp_sh = azp_t(zp);
    scale_out[blockIdx.x] = s;
    azp_out[blockIdx.x] = azp_sh;
  }
  __syncthreads();

  const float inv_s = 1.f / scale_sh;
  const azp_t azp = azp_sh;

  vectorize_with_alignment<16>(
      row_in, row_out, hidden_size, tid, stride,
      [=] __device__(int8_t& dst, const scalar_t& src) {
        const auto v = static_cast<float>(src) * inv_s;
        dst = int32_to_int8(float_to_int32_rn(v) + azp);
      });
}

}  // namespace vllm

void static_scaled_int8_quant(
    torch::stable::Tensor& out,          // [..., hidden_size]
    torch::stable::Tensor const& input,  // [..., hidden_size]
    torch::stable::Tensor const& scale,
    std::optional<torch::stable::Tensor> const& azp) {
  STD_TORCH_CHECK(input.is_contiguous());
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(scale.numel() == 1);
  STD_TORCH_CHECK(!azp || azp->numel() == 1);

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 256));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          vllm::static_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(input.const_data_ptr<scalar_t>(),
                                           out.mutable_data_ptr<int8_t>(),
                                           scale.const_data_ptr<float>(),
                                           hidden_size);
        } else {
          vllm::static_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(
                  input.const_data_ptr<scalar_t>(),
                  out.mutable_data_ptr<int8_t>(), scale.const_data_ptr<float>(),
                  azp->const_data_ptr<int32_t>(), hidden_size);
        }
      });
}

void dynamic_scaled_int8_quant(
    torch::stable::Tensor& out,          // [..., hidden_size]
    torch::stable::Tensor const& input,  // [..., hidden_size]
    torch::stable::Tensor& scales,
    std::optional<torch::stable::Tensor> const& azp) {
  STD_TORCH_CHECK(input.is_contiguous());
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(scales.is_contiguous());
  STD_TORCH_CHECK(!azp || azp->is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 256));
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          vllm::dynamic_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(input.const_data_ptr<scalar_t>(),
                                           out.mutable_data_ptr<int8_t>(),
                                           scales.mutable_data_ptr<float>(),
                                           hidden_size);
        } else {
          vllm::dynamic_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(input.const_data_ptr<scalar_t>(),
                                           out.mutable_data_ptr<int8_t>(),
                                           scales.mutable_data_ptr<float>(),
                                           azp->mutable_data_ptr<int32_t>(),
                                           hidden_size);
        }
      });
}


// ---------------------------------------------------------------------------------------------
// [C3 cuda-lane 2026-09-21] silu_and_mul + DYNAMIC per-token int8 quant, fused.
//
// The W4A8 Marlin path feeds down_proj with  silu(gate) * up  produced by the opaque
// `silu_and_mul` op, then quantises it with `dynamic_scaled_int8_quant_kernel` (per-row absmax)
// and finally multiplies a_scales by a per-layer scalar (marlin_utils.py:718).  Inductor cannot
// fuse across the opaque op (PN447 only reaches NORM-fed GEMMs), so at 8 seats that is
// 3 launches x 64 layers per decode step for one [M, d] activation.  This kernel does all three
// in ONE launch, reading the [M, 2d] gate_up tensor twice (absmax pass, quant pass -- the row is
// L2-resident on the second pass, exactly like dynamic_scaled_int8_quant_kernel) and writing
// int8 + one fp32 scale per row.  Bit-exact w.r.t. the SERVED sequence: _C.silu_and_mul's two bf16
// roundings, then PN447's quant (absmax floor 1e-10, clamp, round half away); the optional global_scale (the
// per-layer act-scale factor) is folded into the STORED scale, which is what the unfused path's
// `a_scales * input_global_scale` computes.
// ---------------------------------------------------------------------------------------------
namespace vllm {

template <typename T> __device__ __forceinline__ float c3_served_silu_mul(const T& x, const T& y) { const T s = static_cast<T>(static_cast<float>(x) / (1.0f + expf(-static_cast<float>(x)))); return static_cast<float>(static_cast<T>(static_cast<float>(s) * static_cast<float>(y))); }  // = _C.silu_and_mul: (T)silu, then bf16*bf16

template <typename scalar_t>
__global__ void silu_and_mul_dynamic_int8_quant_kernel(
    int8_t* __restrict__ out,            // [num_tokens, d]
    float* __restrict__ scale_out,       // [num_tokens]
    const scalar_t* __restrict__ input,  // [num_tokens, 2 * d]  (gate | up)
    const float* __restrict__ global_scale,  // nullptr or one float
    const int d) {
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  const int64_t token_idx = blockIdx.x;
  const scalar_t* __restrict__ x_ptr = input + token_idx * 2 * d;
  const scalar_t* __restrict__ y_ptr = x_ptr + d;
  int8_t* __restrict__ row_out = out + token_idx * d;

  // pass 1: absmax of silu(x) * y over the row
  float thread_max = 0.f;
  for (int i = tid; i < d; i += stride) {
    const float v = c3_served_silu_mul<scalar_t>(x_ptr[i], y_ptr[i]);
    thread_max = fmaxf(thread_max, fabsf(v));
  }
  using BlockReduce = cub::BlockReduce<float, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  const float block_max = BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);
  __shared__ float absmax;
  if (tid == 0) {
    absmax = fmaxf(block_max, 1e-10f);  // served: torch.clamp(amax, min=1e-10)
    const float g = (global_scale == nullptr) ? 1.0f : global_scale[0];
    scale_out[blockIdx.x] = (absmax / 127.f) * g;
  }
  __syncthreads();
  const float inv_s = 127.f / absmax;

  // pass 2: recompute and quantise (same arithmetic as the served _pn447_fused_int8_quant)
  for (int i = tid; i < d; i += stride) {
    const float v = c3_served_silu_mul<scalar_t>(x_ptr[i], y_ptr[i]);
    const float t = fminf(fmaxf(v * inv_s, -127.f), 127.f); row_out[i] = static_cast<int8_t>(t >= 0.f ? floorf(t + 0.5f) : ceilf(t - 0.5f));  // PN447: clamp, round half away
  }
}

}  // namespace vllm

void silu_and_mul_dynamic_int8_quant(
    torch::stable::Tensor& out,          // [..., d]  int8
    torch::stable::Tensor const& input,  // [..., 2 * d]
    torch::stable::Tensor& scales,       // [..., 1]  float32
    std::optional<torch::stable::Tensor> const& global_scale) {
  STD_TORCH_CHECK(input.is_contiguous());
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(scales.is_contiguous());
  STD_TORCH_CHECK(input.size(-1) % 2 == 0, "silu_and_mul input last dim must be even");
  STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Char, "out must be int8");
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float, "scales must be float32");
  const int d = input.size(-1) / 2;
  const int64_t num_tokens = input.numel() / input.size(-1);
  STD_TORCH_CHECK(out.numel() == num_tokens * d, "out shape mismatch");
  STD_TORCH_CHECK(scales.numel() == num_tokens, "scales must hold one value per token");
  const float* gptr = nullptr;
  if (global_scale.has_value()) {
    STD_TORCH_CHECK(global_scale->numel() == 1 &&
                        global_scale->scalar_type() == torch::headeronly::ScalarType::Float,
                    "global_scale must be one float32 element");
    gptr = global_scale->const_data_ptr<float>();
  }
  const dim3 grid(num_tokens);
  const dim3 block(256);   // matches cub::BlockReduce<float, 256> above
  const torch::stable::accelerator::DeviceGuard device_guard(input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_dynamic_int8_quant_kernel", [&] {
        vllm::silu_and_mul_dynamic_int8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.mutable_data_ptr<int8_t>(), scales.mutable_data_ptr<float>(),
            input.const_data_ptr<scalar_t>(), gptr, d);
      });
}
