"""Bit-preserving MLA Q concat for the validated gfx938 draft layouts."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def concat_two_dimensional(
    A,
    B,
    OUT,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    HEADS_PER_CTA: tl.constexpr,
):
    token = tl.program_id(0)
    heads = tl.program_id(1) * HEADS_PER_CTA + tl.arange(0, HEADS_PER_CTA)
    a_columns = tl.arange(0, 512)
    b_columns = tl.arange(0, 64)
    # Integer loads/stores preserve every BF16 bit, including NaN payloads.
    a = A.to(tl.pointer_type(tl.uint16))
    b = B.to(tl.pointer_type(tl.uint16))
    out = OUT.to(tl.pointer_type(tl.uint16))
    a_values = tl.load(a + token * A0 + heads[:, None] * A1 + a_columns[None, :])
    b_values = tl.load(b + token * B0 + heads[:, None] * B1 + b_columns[None, :])
    output_rows = (token * 64 + heads[:, None]) * 576
    tl.store(out + output_rows + a_columns[None, :], a_values)
    tl.store(out + output_rows + 512 + b_columns[None, :], b_values)


def try_concat_mla_draft_hcu(
    q_nope: torch.Tensor,
    q_rope: Optional[torch.Tensor],
    *,
    batch_size: int,
    is_draft_extend: bool,
) -> Optional[torch.Tensor]:
    """Return None for layouts outside the actual BS1-5 replay coverage.

    The caller must restrict this entry to a gfx938 draft worker's DECODE or
    DRAFT_EXTEND_V2 phase. In the latter phase, six speculative rows belong to
    each captured request. Shape/metadata disagreement uses the native path.
    """
    if q_rope is None or not isinstance(batch_size, int) or not 1 <= batch_size <= 5:
        return None
    rows = batch_size * (6 if is_draft_extend else 1)
    if not (
        q_nope.is_cuda
        and q_rope.is_cuda
        and q_nope.device == q_rope.device
        and q_nope.dtype == q_rope.dtype == torch.bfloat16
        and q_nope.shape == (rows, 64, 512)
        and q_rope.shape == (rows, 64, 64)
        and q_nope.stride() == ((32768, 512, 1) if rows == 1 else (512, rows * 512, 1))
        and q_rope.stride() == (16384, 256, 1)
        and q_nope.data_ptr() % 16 == q_rope.data_ptr() % 16 == 0
    ):
        return None
    heads_per_cta, num_warps = (8, 8) if is_draft_extend else (4, 4)
    with torch.cuda.device(q_nope.device):
        out = torch.empty((rows, 64, 576), dtype=q_nope.dtype, device=q_nope.device)
        concat_two_dimensional[(rows, 64 // heads_per_cta)](
            q_nope,
            q_rope,
            out,
            q_nope.stride(0),
            q_nope.stride(1),
            q_rope.stride(0),
            q_rope.stride(1),
            heads_per_cta,
            num_warps=num_warps,
        )
    return out
