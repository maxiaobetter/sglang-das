# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pure index math for decode context parallel (DCP): per-rank lengths and
the owner-rule local-index filter."""

import torch

from sglang.srt.runtime_context import get_parallel


def get_dcp_lens(
    lens: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-rank visible KV length under the owner rule pos % dcp_size == dcp_rank.

    Superset implementation (PR #25090): supports both start=None and a per-request
    `start` offset. update_local_kv_lens_for_dcp is the start=None special case.
    """
    if dcp_size == 1:
        return lens
    if start is None:
        return lens // dcp_size + (dcp_rank < lens % dcp_size)

    first = start + torch.remainder(dcp_rank - start, dcp_size)
    remaining = start + lens - first
    return torch.clamp((remaining + dcp_size - 1) // dcp_size, min=0)


def filter_dcp_local_kv_indices(kv_indices: torch.Tensor):
    parallel = get_parallel()
    if parallel.dcp_enabled:
        kv_indices = (
            kv_indices[kv_indices % parallel.dcp_size == parallel.dcp_rank]
            // parallel.dcp_size
        )
    return kv_indices


def translate_dcp_cache_write(
    loc: torch.Tensor,
    *values: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Translate virtual DCP cache slots without changing the input shape.

    DCP's token allocator exposes one widened, *logical* slot space to the
    scheduler.  Every physical cache (MLA KV and DSA Index-K included) stores
    only its owner slots, addressed as ``logical_slot // dcp_size``. Non-owner
    rows are redirected to the reserved padding slot 0 instead of being removed.
    Keeping the original shape is required by decode/verify CUDA graph replay;
    boolean indexing would create a data-dependent output shape.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return loc, values

    valid = torch.remainder(loc, parallel.attn_dcp_size) == parallel.attn_dcp_rank
    local_loc = torch.div(loc, parallel.attn_dcp_size, rounding_mode="floor")
    write_loc = torch.where(valid, local_loc, torch.zeros_like(local_loc))
    return write_loc, values


def remap_dsa_topk_indices_for_dcp(
    topk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map global DSA sparse slots to this rank's physical DCP slots.

    DSA TopK is produced in the allocator's widened logical-slot space. The
    main MLA KV pool stores only the slots owned by this DCP rank, so sparse
    attention must discard non-owned entries, divide the remaining entries by
    the DCP size, and compact them without changing their TopK order.

    The implementation is fixed-shape tensor algebra, so CUDA graph capture
    keeps the same [rows, topk] contract as eager execution.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return topk_indices, None
    if topk_indices.ndim != 2:
        raise ValueError(
            "DSA DCP TopK remap expects [rows, topk] indices, got "
            f"shape={tuple(topk_indices.shape)}."
        )

    valid = (topk_indices >= 0) & (
        torch.remainder(topk_indices, parallel.attn_dcp_size) == parallel.attn_dcp_rank
    )
    local_indices = torch.div(
        topk_indices,
        parallel.attn_dcp_size,
        rounding_mode="floor",
    )
    local_lengths = valid.sum(dim=1, dtype=torch.int32)

    # Stable compact without a dynamic index list: each valid input receives
    # its prefix-sum destination. Invalid inputs target the final padding slot;
    # if that slot is valid there cannot be any invalid inputs, so no valid
    # value is overwritten.
    width = topk_indices.shape[1]
    destination = torch.cumsum(valid.to(torch.int64), dim=1) - 1
    destination = destination.masked_fill(~valid, width - 1)
    compact = torch.full_like(topk_indices, -1)
    compact.scatter_(
        1,
        destination,
        local_indices.masked_fill(~valid, -1),
    )
    return compact, local_lengths


def update_local_kv_lens_for_dcp(kv_len_arr):
    """In-place per-rank KV length: the start=0 case of get_dcp_lens.

    floor((len - rank - 1) / N) + 1  ==  len // N + (rank < len % N)  for len >= 0
    (bit-identical; see test/registered/cp/test_dcp_layout_unit.py). Kept as an
    in-place mutation because callers (plan_dcp_decode_metadata, the FlashInfer-MLA
    cuda-graph replay path) rely on it.
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return
    kv_len_arr.copy_(get_dcp_lens(kv_len_arr, parallel.dcp_size, parallel.dcp_rank))
