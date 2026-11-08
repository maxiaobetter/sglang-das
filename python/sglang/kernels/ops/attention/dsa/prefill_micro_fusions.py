"""Optional prefill kernels preserving gate arithmetic and opaque KV bytes."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_k1(weights, q_scale, output, scale, m, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    weight = tl.load(weights + row, mask=row < m)
    scaled = weight * scale
    q = tl.load(q_scale + row, mask=row < m)
    result = scaled * q
    tl.store(output + row, result, mask=row < m)


def prefill_gate_k1(weights, q_scale, n_heads, softmax_scale):
    if q_scale.shape[-1] != 1:
        from sglang.kernels.ops.attention.dsa.triton_kernel import (
            fused_get_logits_head_gate_triton,
        )

        return fused_get_logits_head_gate_triton(
            weights, q_scale, n_heads, softmax_scale
        )
    weights, q_scale = weights.contiguous(), q_scale.contiguous()
    assert weights.numel() == q_scale.numel()
    output = torch.empty_like(
        q_scale, dtype=torch.promote_types(weights.dtype, q_scale.dtype)
    )
    scale = softmax_scale * (n_heads**-0.5)
    m = weights.numel()
    if m:
        _gate_k1[(triton.cdiv(m, 256),)](weights, q_scale, output, scale, m, BLOCK=256)
    return output


@triton.jit
def _packed_kv_store(
    gathered,
    loc,
    cache,
    n,
    physical_len,
    CP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Opaque 32-bit words: 656 bytes = 164 words, with each aligned group
    # of four words staying inside one row. No floating-point conversion.
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index // 164
    col = index % 164
    target = tl.load(loc + row, mask=row < n, other=0)
    source_row = row % CP * physical_len + row // CP
    source_index = tl.max_contiguous(tl.multiple_of(source_row * 164 + col, 4), 4)
    target_index = tl.max_contiguous(tl.multiple_of(target * 164 + col, 4), 4)
    value = tl.load(gathered + source_index, mask=row < n)
    tl.store(cache + target_index, value, mask=row < n)


def store_interleaved_packed_kv(gathered, loc, cache, cp_size, physical_rank_len):
    """Store gathered rows at allocator-owned, unique physical cache locations."""
    assert gathered.dtype == cache.dtype == torch.uint8
    assert gathered.ndim == cache.ndim == 2
    assert gathered.shape == (cp_size * physical_rank_len, 656)
    assert cache.shape[1] == 656
    assert gathered.is_contiguous() and cache.is_contiguous()
    assert gathered.data_ptr() % 16 == cache.data_ptr() % 16 == 0
    assert loc.dtype == torch.int64 and loc.ndim == 1 and loc.is_contiguous()
    assert gathered.device == cache.device == loc.device
    n = loc.numel()
    if n:
        _packed_kv_store[(triton.cdiv(n * 164, 512),)](
            gathered.view(torch.int32),
            loc,
            cache.view(torch.int32),
            n,
            physical_rank_len,
            CP=cp_size,
            BLOCK=512,
        )
