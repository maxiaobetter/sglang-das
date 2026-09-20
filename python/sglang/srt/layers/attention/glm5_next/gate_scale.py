"""Prepare KPool indexer weights without intermediate scaling tensors."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _nsa_gate_scale_kernel(
    Weights,
    QScale,
    Output,
    NUMEL: tl.constexpr,
    INPUT_HEADS: tl.constexpr,
    OUTPUT_HEADS: tl.constexpr,
    HEAD_SCALE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // OUTPUT_HEADS
    heads = offsets % OUTPUT_HEADS
    input_heads = heads // (OUTPUT_HEADS // INPUT_HEADS)
    weights = tl.load(
        Weights + rows * INPUT_HEADS + input_heads, offsets < NUMEL, other=0
    )
    # Eager HIP rounds the head-count scaling to BF16 before q_scale promotes
    # the next multiplication to FP32. Preserve that intermediate rounding.
    scaled = (weights.to(tl.float32) * HEAD_SCALE).to(tl.bfloat16).to(tl.float32)
    q_scale = tl.load(QScale + offsets, offsets < NUMEL, other=0)
    result = (scaled * q_scale) * SOFTMAX_SCALE
    tl.store(Output + offsets, result, offsets < NUMEL)


@register_custom_op(mutates_args=["output"])
def _nsa_gate_scale_op(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    output: torch.Tensor,
    head_scale: float,
    softmax_scale: float,
) -> None:
    _nsa_gate_scale_kernel[(triton.cdiv(output.numel(), 256),)](
        weights,
        q_scale,
        output,
        output.numel(),
        weights.shape[1],
        output.shape[1],
        head_scale,
        softmax_scale,
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )


def nsa_gate_scale(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    n_heads: int,
    softmax_scale: float,
) -> torch.Tensor:
    """BF16 [rows, heads] and dense FP32 [rows, max(heads, 8), 1].

    The caller validates this layout and selects the optional optimization.
    """
    output = torch.empty_like(q_scale)
    if output.numel():
        _nsa_gate_scale_op(weights, q_scale, output, n_heads**-0.5, softmax_scale)
    return output
