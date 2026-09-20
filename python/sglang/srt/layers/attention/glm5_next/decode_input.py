"""Fuse the small device-side copies before DCU NSA decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def can_copy_q_and_pad_indices(
    q: torch.Tensor,
    indices: torch.Tensor,
    padded_width: int,
) -> bool:
    """Check the exact layouts supported by the fused decode-input copy."""
    return bool(
        q.is_cuda
        and indices.is_cuda
        and q.device == indices.device
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.stride(2) == 1
        and indices.dtype == torch.int32
        and indices.ndim == 2
        and indices.stride(1) == 1
        and q.shape[0] == indices.shape[0]
        and q.shape[1] > 0
        and q.shape[2] > 0
        and indices.shape[1] > 0
        and padded_width > indices.shape[1]
    )


def copy_q_and_pad_indices(
    q: torch.Tensor,
    indices: torch.Tensor,
    padded_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make Q contiguous and pad indices with -1 in one GPU launch.

    This is a pure data-movement operation. BF16 values are copied without
    arithmetic and every int32 destination element is written once: source
    columns preserve their bits and padding columns receive the -1 sentinel.
    """
    if not can_copy_q_and_pad_indices(q, indices, padded_width):
        raise ValueError("unsupported Q/indices layout for fused decode-input copy")

    num_rows, num_heads, head_dim = q.shape
    q_out = (
        q
        if q.is_contiguous()
        else torch.empty(
            q.shape,
            dtype=q.dtype,
            device=q.device,
        )
    )
    indices_out = torch.empty(
        (num_rows, padded_width),
        dtype=indices.dtype,
        device=indices.device,
    )
    if num_rows == 0:
        return q_out, indices_out

    q_row_elems = num_heads * head_dim
    has_q_copy = not q.is_contiguous()
    block = 512
    # Padding alone is a narrow int32 copy. Two warps avoid the scheduling cost
    # of the wider Q-copy launch once Q already arrives in token-major layout.
    if has_q_copy:
        num_warps = 4 if q_row_elems <= 2048 else 8
    else:
        num_warps = 2
    row_width = max(q_row_elems if has_q_copy else 0, padded_width)
    _copy_q_and_pad_indices_kernel[(num_rows, triton.cdiv(row_width, block))](
        q,
        q_out,
        indices,
        indices_out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        indices.stride(0),
        Q_ROW_ELEMS=q_row_elems,
        HEAD_DIM=head_dim,
        LOGICAL_WIDTH=indices.shape[1],
        PADDED_WIDTH=padded_width,
        HAS_Q_COPY=has_q_copy,
        BLOCK=block,
        num_warps=num_warps,
    )
    return q_out, indices_out


@triton.jit
def _copy_q_and_pad_indices_kernel(
    q_ptr,
    q_out_ptr,
    indices_ptr,
    indices_out_ptr,
    q_stride_0: tl.constexpr,
    q_stride_1: tl.constexpr,
    q_stride_2: tl.constexpr,
    indices_stride_0: tl.constexpr,
    Q_ROW_ELEMS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOGICAL_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    HAS_Q_COPY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)

    if HAS_Q_COPY:
        q_mask = cols < Q_ROW_ELEMS
        head = cols // HEAD_DIM
        dim = cols - head * HEAD_DIM
        q_value = tl.load(
            q_ptr + row * q_stride_0 + head * q_stride_1 + dim * q_stride_2,
            mask=q_mask,
        )
        tl.store(q_out_ptr + row * Q_ROW_ELEMS + cols, q_value, mask=q_mask)

    indices_mask = cols < PADDED_WIDTH
    indices_value = tl.load(
        indices_ptr + row * indices_stride_0 + cols,
        mask=cols < LOGICAL_WIDTH,
        other=-1,
    )
    tl.store(
        indices_out_ptr + row * PADDED_WIDTH + cols,
        indices_value,
        mask=indices_mask,
    )
