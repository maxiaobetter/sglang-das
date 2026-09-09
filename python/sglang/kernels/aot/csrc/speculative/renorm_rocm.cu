// Copyright 2026 SGLang Team. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <hipcub/hipcub.hpp>
#include <limits>

#include "pytorch_extension_utils_rocm.h"

namespace {

constexpr int kThreads = 256;

// Select a probability threshold without sorting or allocating a vocabulary
// workspace. Nonnegative IEEE float bits have the same order as their values.
// The integer interval strictly shrinks, including for equal probabilities.
template <bool TOP_K>
__global__ void small_renorm_kernel(
    const float* probs,
    float* output,
    const int32_t* top_ks,
    const float* top_ps,
    int64_t top_k,
    float top_p,
    int vocab) {
  using Sum = hipcub::BlockReduce<double, kThreads>;
  using Max = hipcub::BlockReduce<uint32_t, kThreads>;
  __shared__ union {
    typename Sum::TempStorage sum;
    typename Max::TempStorage max;
  } storage;
  __shared__ uint32_t lower, upper;
  __shared__ double retained_mass;
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int64_t offset = static_cast<int64_t>(row) * vocab;
  const int64_t k = top_ks == nullptr ? top_k : top_ks[row];
  const double p = top_ps == nullptr ? top_p : top_ps[row];

  uint32_t local_max = 0;
  for (int i = tx; i < vocab; i += kThreads) {
    local_max = max(local_max, __float_as_uint(probs[offset + i]));
  }
  const auto max_bits = Max(storage.max).Reduce(local_max, hipcub::Max());
  if (tx == 0) {
    lower = 0;
    upper = (TOP_K ? k >= vocab : p >= 1.0) ? 0 : max_bits;
  }
  __syncthreads();

  while (lower < upper) {
    const uint32_t pivot = lower + (upper - lower + 1) / 2;
    double local_mass = 0;
    for (int i = tx; i < vocab; i += kThreads) {
      const float value = probs[offset + i];
      if (__float_as_uint(value) >= pivot) {
        local_mass += TOP_K ? 1.0 : static_cast<double>(value);
      }
    }
    const double mass = Sum(storage.sum).Sum(local_mass);
    if (tx == 0) {
      if (mass >= (TOP_K ? static_cast<double>(k) : p)) {
        lower = pivot;
      } else {
        upper = pivot - 1;
      }
    }
    __syncthreads();
  }

  double local_mass = 0;
  for (int i = tx; i < vocab; i += kThreads) {
    const float value = probs[offset + i];
    if (__float_as_uint(value) >= lower) {
      local_mass += value;
    }
  }
  const double mass = Sum(storage.sum).Sum(local_mass);
  if (tx == 0) {
    retained_mass = mass;
  }
  __syncthreads();
  for (int i = tx; i < vocab; i += kThreads) {
    const float value = probs[offset + i];
    output[offset + i] =
        __float_as_uint(value) >= lower && retained_mass > 0 ? static_cast<float>(value / retained_mass) : 0.0f;
  }
}

constexpr int kBins = 16;
constexpr int kChunk = 2048;

struct Histogram {
  double bins[kBins];
};

struct AddHistogram {
  __device__ Histogram operator()(const Histogram& a, const Histogram& b) const {
    Histogram result;
#pragma unroll
    for (int i = 0; i < kBins; ++i)
      result.bins[i] = a.bins[i] + b.bins[i];
    return result;
  }
};

// Radix selection spreads each vocabulary over many blocks. Eight fixed passes
// find the exact FP32 cutoff, without sorting or a data-dependent retry loop.
// Per-thread histograms avoid contended floating-point atomics on equal values.
template <bool TOP_K>
__global__ void
radix_histogram(const float* probs, const double* state, double* partial, int vocab, int parts, int shift) {
  using Reduce = hipcub::BlockReduce<Histogram, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  if (shift != 28 && state[row * 2 + 1] < 0) return;
  const uint32_t prefix = shift == 28 ? 0 : static_cast<uint32_t>(state[row * 2]);
  const uint32_t mask = shift == 28 ? 0 : ~((1u << (shift + 4)) - 1);
  Histogram local = {};
  for (int i = part * kChunk + threadIdx.x; i < min(vocab, (part + 1) * kChunk); i += kThreads) {
    const float value = probs[static_cast<int64_t>(row) * vocab + i];
    const uint32_t bits = __float_as_uint(value);
    const int bin = (bits >> shift) & (kBins - 1);
    const double weight = (bits & mask) == prefix ? (TOP_K ? 1.0 : static_cast<double>(value)) : 0.0;
#pragma unroll
    for (int j = 0; j < kBins; ++j)
      local.bins[j] += bin == j ? weight : 0.0;
  }
  const Histogram total = Reduce(storage).Reduce(local, AddHistogram());
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < kBins; ++j)
      partial[(row * parts + part) * kBins + j] = total.bins[j];
  }
}

template <bool TOP_K>
__global__ void radix_select(
    const double* partial,
    double* state,
    const int32_t* top_ks,
    const float* top_ps,
    int64_t top_k,
    float top_p,
    int vocab,
    int parts,
    int shift) {
  __shared__ double sums[kThreads];
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const double parameter = TOP_K ? static_cast<double>(top_ks ? top_ks[row] : top_k) : (top_ps ? top_ps[row] : top_p);
  if (parameter >= (TOP_K ? vocab : 1.0)) {
    if (tx == 0) {
      state[row * 2] = 0;
      state[row * 2 + 1] = -1;
    }
    return;
  }
  double sum = 0;
  for (int part = tx / kBins; part < parts; part += kThreads / kBins) {
    sum += partial[(row * parts + part) * kBins + tx % kBins];
  }
  sums[tx] = sum;
  __syncthreads();
#pragma unroll
  for (int stride = kThreads / 2; stride >= kBins; stride /= 2) {
    if (tx < stride) sums[tx] += sums[tx + stride];
    __syncthreads();
  }
  if (tx == 0) {
    double remaining = shift == 28 ? parameter : state[row * 2 + 1];
    int bin = kBins - 1;
    for (; bin > 0; --bin) {
      if (sums[bin] >= remaining) break;
      remaining -= sums[bin];
    }
    const uint32_t prefix = shift == 28 ? 0 : static_cast<uint32_t>(state[row * 2]);
    state[row * 2] = prefix | (static_cast<uint32_t>(bin) << shift);
    state[row * 2 + 1] = remaining;
  }
}

__global__ void retained_mass_kernel(const float* probs, const double* state, double* partial, int vocab, int parts) {
  using Reduce = hipcub::BlockReduce<double, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const uint32_t cutoff = static_cast<uint32_t>(state[row * 2]);
  double mass = 0;
  for (int i = part * kChunk + threadIdx.x; i < min(vocab, (part + 1) * kChunk); i += kThreads) {
    const float value = probs[static_cast<int64_t>(row) * vocab + i];
    mass += __float_as_uint(value) >= cutoff ? static_cast<double>(value) : 0.0;
  }
  const double total = Reduce(storage).Sum(mass);
  if (threadIdx.x == 0) partial[row * parts + part] = total;
}

__global__ void
apply_cutoff(const float* probs, float* output, const double* state, const double* partial, int vocab, int parts) {
  using Reduce = hipcub::BlockReduce<double, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ double mass;
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const uint32_t cutoff = static_cast<uint32_t>(state[row * 2]);
  double sum = 0;
  for (int i = threadIdx.x; i < parts; i += kThreads)
    sum += partial[row * parts + i];
  const double total = Reduce(storage).Sum(sum);
  if (threadIdx.x == 0) mass = total;
  __syncthreads();
  for (int i = part * kChunk + threadIdx.x; i < min(vocab, (part + 1) * kChunk); i += kThreads) {
    const int64_t offset = static_cast<int64_t>(row) * vocab + i;
    const float value = probs[offset];
    output[offset] = __float_as_uint(value) >= cutoff && mass > 0 ? static_cast<float>(value / mass) : 0.0f;
  }
}

template <bool TOP_K>
void launch_renorm(
    const at::Tensor& probs,
    const at::Tensor& output,
    const int32_t* top_ks,
    const float* top_ps,
    int64_t top_k,
    float top_p) {
  const int batch = probs.size(0);
  const int vocab = probs.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (vocab <= kChunk) {
    small_renorm_kernel<TOP_K><<<batch, kThreads, 0, stream>>>(
        probs.data_ptr<float>(), output.data_ptr<float>(), top_ks, top_ps, top_k, top_p, vocab);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int parts = (vocab - 1) / kChunk + 1;
  auto state = at::empty({batch, 2}, probs.options().dtype(at::kDouble));
  auto partial = at::empty({batch, parts, kBins}, probs.options().dtype(at::kDouble));
  const dim3 grid(batch, parts);
  for (int shift = 28; shift >= 0; shift -= 4) {
    radix_histogram<TOP_K><<<grid, kThreads, 0, stream>>>(
        probs.data_ptr<float>(), state.data_ptr<double>(), partial.data_ptr<double>(), vocab, parts, shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    radix_select<TOP_K><<<batch, kThreads, 0, stream>>>(
        partial.data_ptr<double>(), state.data_ptr<double>(), top_ks, top_ps, top_k, top_p, vocab, parts, shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  retained_mass_kernel<<<grid, kThreads, 0, stream>>>(
      probs.data_ptr<float>(), state.data_ptr<double>(), partial.data_ptr<double>(), vocab, parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  apply_cutoff<<<grid, kThreads, 0, stream>>>(
      probs.data_ptr<float>(),
      output.data_ptr<float>(),
      state.data_ptr<double>(),
      partial.data_ptr<double>(),
      vocab,
      parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_renorm_inputs(const at::Tensor& probs, const at::Tensor& output) {
  CHECK_INPUT(probs);
  CHECK_INPUT(output);
  CHECK_DIM(2, probs);
  TORCH_CHECK(
      probs.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat, "renorm probabilities must be float32");
  TORCH_CHECK(probs.sizes() == output.sizes() && probs.device() == output.device(), "renorm output must match probs");
  TORCH_CHECK(probs.size(1) > 0 && probs.size(1) <= std::numeric_limits<int>::max(), "invalid vocabulary size");
}

void check_parameter(const at::Tensor& parameter, const at::Tensor& probs, at::ScalarType dtype) {
  CHECK_INPUT(parameter);
  TORCH_CHECK(
      parameter.dim() == 1 && parameter.size(0) == probs.size(0), "sampling parameter batch size must match probs");
  TORCH_CHECK(
      parameter.device() == probs.device() && parameter.scalar_type() == dtype,
      "sampling parameter device or dtype mismatch");
}

}  // namespace

void top_k_renorm_probs(
    at::Tensor probs, at::Tensor renorm_probs, std::optional<at::Tensor> maybe_top_k_arr, int64_t top_k_val) {
  check_renorm_inputs(probs, renorm_probs);
  if (maybe_top_k_arr.has_value()) {
    check_parameter(*maybe_top_k_arr, probs, at::kInt);
  } else {
    TORCH_CHECK(top_k_val >= 1, "top_k must be positive");
  }
  if (probs.size(0) == 0) return;
  const at::cuda::CUDAGuard guard(probs.device());
  launch_renorm<true>(
      probs,
      renorm_probs,
      maybe_top_k_arr.has_value() ? maybe_top_k_arr->data_ptr<int32_t>() : nullptr,
      nullptr,
      top_k_val,
      1.0f);
}

void top_p_renorm_probs(
    at::Tensor probs, at::Tensor renorm_probs, std::optional<at::Tensor> maybe_top_p_arr, double top_p_val) {
  check_renorm_inputs(probs, renorm_probs);
  if (maybe_top_p_arr.has_value()) {
    check_parameter(*maybe_top_p_arr, probs, at::kFloat);
  } else {
    TORCH_CHECK(top_p_val > 0 && top_p_val <= 1, "top_p must be in (0, 1]");
  }
  if (probs.size(0) == 0) return;
  const at::cuda::CUDAGuard guard(probs.device());
  launch_renorm<false>(
      probs,
      renorm_probs,
      nullptr,
      maybe_top_p_arr.has_value() ? maybe_top_p_arr->data_ptr<float>() : nullptr,
      probs.size(1),
      static_cast<float>(top_p_val));
}
