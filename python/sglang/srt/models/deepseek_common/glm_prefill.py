"""Opt-in GLM sparse prefill optimizations; preserve all fallback paths."""

import logging

import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
    is_dsa_prefill_cp_round_robin_split,
)
from sglang.srt.layers.cp.utils import (
    InterleaveCPStrategy,
    get_cp_strategy,
    is_cp_v2_active,
)
from sglang.srt.lora.deepseek_mla_correction import is_kv_b_lora_active
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_exec, get_parallel

logger = logging.getLogger(__name__)


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


def launch_cp_kv_gather(attn, batch, positions, q_pe, k_nope, k_pe):
    """Rotate Q/K before launching KV, leaving the Q absorb BMM independent."""
    from sglang.srt.distributed.parallel_state import (
        get_attn_cp_group,
        get_attn_cp_overlap_group,
    )
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.rope_variant import (
        DeepseekScalingRotaryEmbedding,
    )
    from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_launch

    if not (
        eligible(attn, batch)
        and dsa_use_prefill_cp(batch)
        and (
            isinstance(get_cp_strategy(), InterleaveCPStrategy)
            if is_cp_v2_active(batch)
            else is_dsa_prefill_cp_round_robin_split()
        )
        and attn.cp_size > 1
        and type(attn.rotary_emb) in (RotaryEmbedding, DeepseekScalingRotaryEmbedding)
        and not attn._fuse_rope_for_trtllm_mla(batch)
        and not attn._skip_rope_for_dsa_tilelang_fused()
        and not attn._skip_rope_for_aiter_fused_mla()
    ):
        return None, q_pe, k_pe
    if get_attn_cp_overlap_group() is get_attn_cp_group():
        raise RuntimeError("GLM prefill overlap requires the duplicate CP communicator")
    total_tokens = None
    if is_cp_v2_active(batch):
        metadata = batch.attn_cp_metadata
        total_tokens = int(metadata.total_seq_lens)
        if total_tokens == 0:
            return None, q_pe, k_pe
        logical_lens = metadata.per_rank_logical_token or metadata.per_rank_actual_token
        local_logical_len = logical_lens[get_parallel().attn_cp_rank]
        physical_rank_len = max(metadata.per_rank_actual_token)
        if not (0 <= local_logical_len <= k_nope.shape[0] <= physical_rank_len):
            raise RuntimeError("Unexpected GLM interleave CP local token count")
    q_pe, k_pe = attn.rotary_emb(positions, q_pe, k_pe)
    # Do not mutate latent_cache: its views may still be used by the indexer.
    packed_kv = torch.cat((k_nope.squeeze(1), k_pe.squeeze(1)), dim=-1)
    if total_tokens is not None:
        # Match InterleaveCPStrategy padding, including uneven final chunks.
        padded_kv = packed_kv.new_zeros((physical_rank_len, packed_kv.shape[-1]))
        padded_kv[:local_logical_len] = packed_kv[:local_logical_len]
        packed_kv = padded_kv
    stream = getattr(attn, "_glm_prefill_cp_stream", None)
    if stream is None:
        stream = attn._glm_prefill_cp_stream = torch.cuda.Stream(
            device=packed_kv.device
        )
    with torch.profiler.record_function("glm_prefill_cp_kv_launch"):
        handle = cp_all_gather_rerange_launch(
            packed_kv, attn.cp_size, stream, ("glm_prefill_kv", attn.layer_id)
        )
        # In addition to the handle references, preserve allocations on error
        # paths where Python unwinds before the normal finish call.
        handle[0].record_stream(stream)
        packed_kv.record_stream(stream)
    report(attn, "CP KV gather overlapping Q absorb BMM")
    return (handle, total_tokens), q_pe, k_pe


def finish_cp_kv_gather(handle, kv_lora_rank):
    from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_finish

    with torch.profiler.record_function("glm_prefill_cp_kv_join"):
        gather_handle, total_tokens = handle
        kv = cp_all_gather_rerange_finish(gather_handle)
        if total_tokens is not None:
            kv = kv[:total_tokens]
    return kv[..., :kv_lora_rank].unsqueeze(1), kv[..., kv_lora_rank:].unsqueeze(1)


def try_direct_q_buffer(attn, batch, q_nope, q_pe):
    if not (
        envs.SGLANG_GLM_PREFILL_DIRECT_Q.get()
        and eligible(attn, batch)
        and dsa_use_prefill_cp(batch)
        and q_nope.dtype == torch.bfloat16
        and attn.w_kc.dtype == torch.bfloat16
        and q_nope.ndim == q_pe.ndim == 3
        and q_nope.shape[0] > 0
        and q_nope.shape[1] == attn.num_local_heads
        and tuple(attn.w_kc.shape)
        == (attn.num_local_heads, q_nope.shape[-1], attn.kv_lora_rank)
        and tuple(q_pe.shape)
        == (q_nope.shape[0], attn.num_local_heads, attn.qk_rope_head_dim)
    ):
        return None
    report(attn, "Q absorb BMM writing final Q layout")
    return q_nope.new_empty(
        (
            q_nope.shape[0],
            attn.num_local_heads,
            attn.kv_lora_rank + attn.qk_rope_head_dim,
        )
    )


def try_q_norm_quant(attn, batch, q):
    """Fuse Q RMSNorm + per-token quant; retain BF16 Q for the indexer.

    No residual fusion: older LightOp kernels round the residual sum before
    normalization. Only the non-residual producer is used here.
    """
    if not (envs.SGLANG_GLM_PREFILL_NORM_QUANT.get() and eligible(attn, batch)):
        return None
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        CompressedTensorsW8A8Fp8,
        QuantizationStrategy,
        _use_aiter,
    )

    linear = attn.q_b_proj
    scheme = getattr(linear, "scheme", None)
    norm = attn.q_a_layernorm
    if not (
        not _use_aiter  # AITER uses shuffled weights with a different GEMM ABI.
        and isinstance(scheme, CompressedTensorsW8A8Fp8)
        and scheme.strategy == QuantizationStrategy.CHANNEL
        and scheme.weight_block_size is None
        and not scheme.is_static_input_scheme
        and getattr(linear, "input_scale", None) is None
        and linear.weight.dtype == torch.float8_e4m3fn
        and linear.weight_scale.dtype == torch.float32
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
        and not norm.fp32_residual
        and norm.override_orig_dtype is None
        and not getattr(norm, "x_pad_to_multiple", 0)
        and q.dtype == torch.bfloat16
        and q.ndim == 2
        and q.shape[0] > 0
        and q.shape[1] % 16 == 0
        and q.shape[1] <= 16384
        and norm.weight.dtype == q.dtype
        and norm.weight.device == q.device
        and norm.weight.is_contiguous()
        and norm.weight.shape == (q.shape[1],)
        and norm.weight.data_ptr() % 16 == 0
    ):
        return None
    from lightop.rmsnorm import rms_norm_per_token_fp8_quant

    # q is normally a view into fused Q/K projection; preserve its other users.
    normalized = q.clone(memory_format=torch.contiguous_format)
    with torch.profiler.record_function("glm_prefill_q_norm_quant"):
        quant, scale = rms_norm_per_token_fp8_quant(
            normalized,
            norm.weight,
            epsilon=norm.variance_epsilon,
            fp8type=0,
            update_input=True,
        )
    report(attn, "Q RMSNorm + per-token FP8 quantization")
    return (quant, scale, q.dtype), normalized
