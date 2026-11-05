"""Opt-in GLM sparse prefill optimizations; preserve all fallback paths."""

import logging
from functools import lru_cache

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.lora.deepseek_mla_correction import is_kv_b_lora_active
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def prefill_compute_stream(device, purpose):
    """One compute queue per device/purpose, shared across all model layers."""
    return torch.cuda.Stream(device=device)


def eligible(attn, batch):
    """Only ordinary eager GLM prefill, with the original sparse BF16 ABI."""
    return (
        getattr(attn, "_hcu_glm_prefill", False)
        and batch.forward_mode.is_extend_without_speculative()
        and not is_batch_invariant_mode_enabled()
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
        and not getattr(batch, "can_run_tbo", False)
        and getattr(batch, "tbo_parent_token_range", None) is None
        and not get_exec().deterministic.enable_deterministic_inference
        and not get_parallel().dcp_enabled
        and not envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
        and not is_kv_b_lora_active(attn)
        and attn.use_dsa
        and not attn.use_deep_gemm_bmm
        and not attn.attn_mqa.is_cross_attention
        and not getattr(get_attn_backend(), "use_mha", True)
        and getattr(get_attn_backend(), "dsa_prefill_impl", None) == "flashmla_sparse"
    )


def report(attn, feature):
    reported = getattr(attn, "_glm_prefill_reported", None)
    if reported is None:
        reported = attn._glm_prefill_reported = set()
    if feature not in reported:
        reported.add(feature)
        if attn.layer_id == 0:
            logger.info(
                "GLM prefill selected %s (confirm execution in GPU trace)", feature
            )
