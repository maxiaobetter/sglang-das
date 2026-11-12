"""Physical-row plans and order-preserving indices for eager BF16 sparse MLA.

Only metadata is shared across layers. Each layer must dequantize its own KV
with the original dequantize_k_cache_paged(raw, plan.physical_rows) operation.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class PhysicalKVPlan:
    logical_rows: torch.Tensor
    physical_rows: torch.Tensor
    inverse: torch.Tensor

    @property
    def has_duplicates(self) -> bool:
        return self.physical_rows.numel() < self.logical_rows.numel()


def build_physical_kv_plan(logical_rows: torch.Tensor) -> PhysicalKVPlan:
    """Build once per eager forward; dynamic unique can synchronize the host."""
    assert logical_rows.is_cuda and logical_rows.ndim == 1
    assert logical_rows.is_contiguous()
    assert logical_rows.dtype in (torch.int32, torch.int64)
    assert 0 < logical_rows.numel() <= torch.iinfo(torch.int32).max
    physical, inverse = torch.unique(logical_rows, sorted=True, return_inverse=True)
    return PhysicalKVPlan(logical_rows, physical, inverse.to(torch.int32))


@triton.jit
def _remap_physical_kv_indices(
    indices, inverse, output, count, logical_count, physical_count, BLOCK: tl.constexpr
):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(indices + x, x < count, other=-1)
    valid = (i >= 0) & (i < logical_count) & (x < count)
    j = tl.load(inverse + i, valid, other=0)
    # Preserve every slot and negative sentinel. An out-of-range positive index
    # remains out of range after compaction; never turn it into a valid row.
    mapped = tl.where(valid, j, tl.where(i < 0, i, physical_count))
    tl.store(output + x, mapped, x < count)


def remap_physical_kv_indices(indices: torch.Tensor, plan: PhysicalKVPlan):
    """Remap addresses only, without sorting or mutating the producer's TopK."""
    assert indices.is_cuda and indices.is_contiguous() and indices.dtype == torch.int32
    assert indices.device == plan.inverse.device
    result = torch.empty_like(indices)
    if indices.numel():
        _remap_physical_kv_indices[(triton.cdiv(indices.numel(), 512),)](
            indices,
            plan.inverse,
            result,
            indices.numel(),
            plan.logical_rows.numel(),
            plan.physical_rows.numel(),
            BLOCK=512,
        )
    return result
