"""Optional GLM shared-expert prefill overlap."""

import logging

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


def try_shared_stream(moe, batch, hidden_states):
    shared = getattr(moe, "shared_experts", None)
    if not (
        envs.SGLANG_GLM_PREFILL_SHARED_OVERLAP.get()
        and getattr(shared, "_glm_prefill_shared_overlap", False)
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
