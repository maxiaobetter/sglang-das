"""Optional GLM shared-expert prefill fusion and overlap."""

import logging
from functools import lru_cache

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _report_shared_quant():
    logger.info("GLM prefill selected shared SwiGLU + per-token FP8 quantization")


def try_shared_stream(moe, batch, hidden_states):
    shared = getattr(moe, "shared_experts", None)
    if not (
        envs.SGLANG_GLM_PREFILL_SHARED_OVERLAP.get()
        and getattr(shared, "_glm_prefill_shared_quant", False)
        and not moe.is_nextn
        and moe.num_fused_shared_experts == 0
        and not moe._fuse_shared_experts_inside_sbo
        and batch.forward_mode.is_extend_without_speculative()
        and not batch.can_run_tbo
        and getattr(batch, "tbo_parent_token_range", None) is None
        and not is_batch_invariant_mode_enabled()
        and not get_exec().deterministic.enable_deterministic_inference
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    ):
        return None
    from sglang.srt.models.deepseek_common.glm_prefill import prefill_compute_stream

    if not getattr(moe, "_glm_shared_overlap_reported", False):
        moe._glm_shared_overlap_reported = True
        if moe.layer_id == getattr(moe.config, "first_k_dense_replace", 0):
            logger.info("GLM prefill selected shared expert overlap")
    return (
        moe.alt_stream
        if moe.alt_stream is not None
        else prefill_compute_stream(hidden_states.device, "shared_expert")
    )


def try_shared_swiglu_quant(mlp, batch, gate_up):
    if not (
        envs.SGLANG_GLM_PREFILL_SHARED_QUANT.get()
        and getattr(mlp, "_glm_prefill_shared_quant", False)
        and batch is not None
        and batch.forward_mode.is_extend_without_speculative()
        and not batch.can_run_tbo
        and not is_batch_invariant_mode_enabled()
        and not get_exec().deterministic.enable_deterministic_inference
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
        and mlp.swiglu_limit is None
        and gate_up.is_cuda
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.shape[0] > 0
        and gate_up.shape[1] % 32 == 0
        and gate_up.is_contiguous()
    ):
        return None
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        supports_hcu_channelwise_prequant,
    )

    if not supports_hcu_channelwise_prequant(mlp.down_proj):
        return None
    from lightop.fuse_silu_mul_quant import fuse_silu_mul_fp8_quant

    # Use the existing HCU SwiGLU ordering: [gate, up], BF16 activation output,
    # then row-wise E4M3FN quantization. No clamp or Spark GELU substitution.
    with torch.profiler.record_function("glm_prefill_shared_swiglu_quant"):
        quant, scale = fuse_silu_mul_fp8_quant(gate_up, fp8type=0)
        _report_shared_quant()
        return mlp.down_proj((quant, scale, gate_up.dtype))[0]
