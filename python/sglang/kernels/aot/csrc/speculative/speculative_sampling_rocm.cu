/*
 * Copyright (c) 2025-2026 by SGLang team.
 * Copyright (c) 2024-2025 by FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 *
 * HIP implementation of speculative_sampling.cuh's target-only tree sampler.
 * Preserves its acceptance thresholds, sibling traversal, residual sampling,
 * and caller-supplied random numbers without a FlashInfer runtime dependency.
 */

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
constexpr int kItems = 4;
constexpr int kChunk = 2048;

__global__ void tree_walk_kernel(
    int32_t* predicts,
    int32_t* accept_index,
    int32_t* accept_token_num,
    const int64_t* candidates,
    const int64_t* retrieve_index,
    const int64_t* retrieve_next_token,
    const int64_t* retrieve_next_sibling,
    const float* uniform_samples,
    const float* target_probs,
    float* draft_probs,
    int64_t* state,
    int steps,
    int nodes,
    int vocab,
    float threshold_single,
    float threshold_acc) {
  if (threadIdx.x != 0) return;
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * nodes;
  int64_t prob_offset, last_index;
  {
    float accumulated = 0.0f;
    float coin = uniform_samples[base];
    int64_t current = 0;
    int accepted = 0;
    prob_offset = base * vocab;
    last_index = retrieve_index[base];
    accept_index[row * steps] = last_index;
    for (int j = 1; j < steps; ++j) {
      current = retrieve_next_token[base + current];
      while (current != -1) {
        const int64_t token = candidates[base + current];
        const float probability = target_probs[prob_offset + token];
        accumulated += probability;
        if (coin <= accumulated / threshold_acc || probability >= threshold_single) {
          predicts[last_index] = token;
          ++accepted;
          last_index = retrieve_index[base + current];
          accept_index[row * steps + accepted] = last_index;
          prob_offset = (base + current) * vocab;
          coin = uniform_samples[base + current];
          accumulated = 0.0f;
          break;
        }
        draft_probs[prob_offset + token] = probability;
        current = retrieve_next_sibling[base + current];
      }
      if (current == -1) break;
    }
    accept_token_num[row] = accepted;
    state[row * 3] = prob_offset;
    state[row * 3 + 1] = last_index;
    state[row * 3 + 2] = accepted != steps - 1;
  }
}

// Reduce independent vocabulary chunks in parallel, then scan only the chunk
// containing the sampled token. This avoids a serial full-vocabulary block scan
// when the per-DP request batch is small.
__global__ void residual_mass_kernel(
    const float* target_probs, const float* draft_probs, const int64_t* state, double* partial, int vocab, int parts) {
  using Reduce = hipcub::BlockReduce<double, kThreads>;
  __shared__ typename Reduce::TempStorage storage;
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int64_t offset = state[row * 3];
  const bool subtract_draft = state[row * 3 + 2];
  double local_mass = 0.0;
  for (int i = part * kChunk + threadIdx.x; i < min(vocab, (part + 1) * kChunk); i += kThreads) {
    const float draft = subtract_draft ? draft_probs[offset + i] : 0.0f;
    local_mass += fmaxf(target_probs[offset + i] - draft, 0.0f);
  }
  const double total = Reduce(storage).Sum(local_mass);
  if (threadIdx.x == 0) partial[row * parts + part] = total;
}

__global__ void tree_draw_kernel(
    int32_t* predicts,
    const int64_t* state,
    const double* partial,
    const float* final_uniform_samples,
    const float* target_probs,
    const float* draft_probs,
    int vocab,
    int parts) {
  using Reduce = hipcub::BlockReduce<double, kThreads>;
  using Scan = hipcub::BlockScan<double, kThreads>;
  using Max = hipcub::BlockReduce<int, kThreads>;
  __shared__ union {
    typename Reduce::TempStorage reduce;
    typename Scan::TempStorage scan;
    typename Max::TempStorage max;
  } storage;
  __shared__ double residual_mass, chunk_u;
  __shared__ int selected_part, last_part, sampled_id, last_valid_id;
  const int tx = threadIdx.x;
  const int row = blockIdx.x;
  const int64_t prob_offset = state[row * 3];
  const int64_t last_index = state[row * 3 + 1];
  const bool subtract_draft = state[row * 3 + 2];
  double local_mass = 0;
  for (int part = tx; part < parts; part += kThreads)
    local_mass += partial[row * parts + part];
  const double total = Reduce(storage.reduce).Sum(local_mass);
  if (tx == 0) {
    residual_mass = total;
    selected_part = parts;
    last_part = -1;
    sampled_id = vocab;
    last_valid_id = -1;
  }
  __syncthreads();
  const double u = final_uniform_samples[row] * residual_mass;
  double aggregate = 0;
  for (int begin = 0; begin < parts; begin += kThreads) {
    const int part = begin + tx;
    const double mass = part < parts ? partial[row * parts + part] : 0.0;
    double prefix, group_mass;
    Scan(storage.scan).ExclusiveSum(mass, prefix, group_mass);
    __syncthreads();
    if (mass > 0) {
      atomicMax(&last_part, part);
      if (aggregate + prefix + mass > u) atomicMin(&selected_part, part);
    }
    __syncthreads();
    if (selected_part == part) chunk_u = u - (aggregate + prefix);
    __syncthreads();
    if (selected_part < parts) break;
    aggregate += group_mass;
  }
  if (tx == 0 && selected_part == parts) {
    selected_part = last_part >= 0 ? last_part : parts - 1;
    chunk_u = partial[row * parts + selected_part];
  }
  __syncthreads();

  aggregate = 0;
  const int64_t end = min(vocab, (selected_part + 1) * kChunk);
  for (int64_t begin = selected_part * kChunk; begin < end; begin += kThreads * kItems) {
    float values[kItems];
    double thread_sum = 0.0;
    int last_valid = -1;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      const int64_t i = begin + tx * kItems + j;
      values[j] = 0.0f;
      if (i < end) {
        const float draft = subtract_draft ? draft_probs[prob_offset + i] : 0.0f;
        values[j] = fmaxf(target_probs[prob_offset + i] - draft, 0.0f);
        if (values[j] > 0.0f) last_valid = i;
      }
      thread_sum += values[j];
    }
    double prefix, chunk_mass;
    Scan(storage.scan).ExclusiveSum(thread_sum, prefix, chunk_mass);
    __syncthreads();
    double cdf = aggregate + prefix;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      cdf += values[j];
      if (values[j] > 0.0f && cdf > chunk_u) {
        atomicMin(&sampled_id, static_cast<int>(begin + tx * kItems + j));
        break;
      }
    }
    const int chunk_last = Max(storage.max).Reduce(last_valid, hipcub::Max());
    if (tx == 0 && chunk_last >= 0) last_valid_id = chunk_last;
    __syncthreads();
    aggregate += chunk_mass;
    if (aggregate > chunk_u) break;
  }
  if (tx == 0) {
    predicts[last_index] = sampled_id != vocab ? sampled_id : (last_valid_id >= 0 ? last_valid_id : vocab - 1);
  }
}

void check_tensor(const at::Tensor& tensor, const at::Tensor& reference, int dimensions, at::ScalarType dtype) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(
      tensor.dim() == dimensions && tensor.scalar_type() == dtype, "tree sampling tensor shape or dtype mismatch");
  TORCH_CHECK(tensor.device() == reference.device(), "tree sampling tensors must be on the same device");
}

}  // namespace

void tree_speculative_sampling_target_only(
    at::Tensor predicts,
    at::Tensor accept_index,
    at::Tensor accept_token_num,
    at::Tensor candidates,
    at::Tensor retrive_index,
    at::Tensor retrive_next_token,
    at::Tensor retrive_next_sibling,
    at::Tensor uniform_samples,
    at::Tensor uniform_samples_for_final_sampling,
    at::Tensor target_probs,
    at::Tensor draft_probs,
    double threshold_single,
    double threshold_acc,
    bool deterministic) {
  check_tensor(target_probs, target_probs, 3, at::kFloat);
  check_tensor(draft_probs, target_probs, 3, at::kFloat);
  check_tensor(predicts, target_probs, 1, at::kInt);
  check_tensor(accept_index, target_probs, 2, at::kInt);
  check_tensor(accept_token_num, target_probs, 1, at::kInt);
  check_tensor(uniform_samples, target_probs, 2, at::kFloat);
  check_tensor(uniform_samples_for_final_sampling, target_probs, 1, at::kFloat);
  for (const auto& tensor : {candidates, retrive_index, retrive_next_token, retrive_next_sibling}) {
    check_tensor(tensor, target_probs, 2, at::kLong);
    TORCH_CHECK(
        tensor.size(0) == target_probs.size(0) && tensor.size(1) == target_probs.size(1),
        "tree layout must match target_probs batch and nodes");
  }
  const auto batch = target_probs.size(0);
  const auto nodes = target_probs.size(1);
  const auto vocab = target_probs.size(2);
  const auto steps = accept_index.size(1);
  TORCH_CHECK(
      nodes > 0 && steps > 0 && steps <= nodes && vocab > 0 && vocab <= std::numeric_limits<int>::max() &&
          batch * nodes <= std::numeric_limits<int32_t>::max(),
      "invalid tree sampling dimensions");
  TORCH_CHECK(draft_probs.sizes() == target_probs.sizes(), "draft_probs must match target_probs");
  TORCH_CHECK(
      predicts.numel() == batch * nodes && accept_index.size(0) == batch && accept_token_num.numel() == batch,
      "tree sampling output shape mismatch");
  TORCH_CHECK(
      uniform_samples.sizes() == candidates.sizes() && uniform_samples_for_final_sampling.numel() == batch,
      "tree sampling random number shape mismatch");
  TORCH_CHECK(
      threshold_single >= 0 && threshold_single <= 1 && threshold_acc >= 0 && threshold_acc <= 1,
      "acceptance thresholds must be in [0, 1]");
  if (batch == 0) return;
  const at::cuda::CUDAGuard guard(target_probs.device());
  // A fixed block scan is deterministic for both allowed flag values. Random
  // state is owned by the caller, so graph replay requires no host RNG access.
  (void)deterministic;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int parts = (vocab - 1) / kChunk + 1;
  auto state = at::empty({batch, 3}, target_probs.options().dtype(at::kLong));
  auto partial = at::empty({batch, parts}, target_probs.options().dtype(at::kDouble));
  tree_walk_kernel<<<batch, 32, 0, stream>>>(
      predicts.data_ptr<int32_t>(),
      accept_index.data_ptr<int32_t>(),
      accept_token_num.data_ptr<int32_t>(),
      candidates.data_ptr<int64_t>(),
      retrive_index.data_ptr<int64_t>(),
      retrive_next_token.data_ptr<int64_t>(),
      retrive_next_sibling.data_ptr<int64_t>(),
      uniform_samples.data_ptr<float>(),
      target_probs.data_ptr<float>(),
      draft_probs.data_ptr<float>(),
      state.data_ptr<int64_t>(),
      steps,
      nodes,
      vocab,
      static_cast<float>(threshold_single),
      fmaxf(static_cast<float>(threshold_acc), 1e-9f));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  residual_mass_kernel<<<dim3(batch, parts), kThreads, 0, stream>>>(
      target_probs.data_ptr<float>(),
      draft_probs.data_ptr<float>(),
      state.data_ptr<int64_t>(),
      partial.data_ptr<double>(),
      vocab,
      parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tree_draw_kernel<<<batch, kThreads, 0, stream>>>(
      predicts.data_ptr<int32_t>(),
      state.data_ptr<int64_t>(),
      partial.data_ptr<double>(),
      uniform_samples_for_final_sampling.data_ptr<float>(),
      target_probs.data_ptr<float>(),
      draft_probs.data_ptr<float>(),
      vocab,
      parts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
