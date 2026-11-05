"""Optional GLM prefix prefill: fused KV producer and byte-wise CP exchange."""

from typing import NamedTuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.cp.utils import (
    InterleaveCPStrategy,
    get_cp_strategy,
    is_cp_v2_active,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.models.deepseek_common.glm_prefill import eligible, report


class PrefillQWithStoredKV(NamedTuple):
    """Explicitly tell the core that Q is packed and global KV is already stored."""

    query: torch.Tensor


def use_fused_kv_producer(attn, batch, latent):
    if not (
        envs.SGLANG_GLM_PREFILL_KV_ROPE_STORE.get()
        and eligible(attn, batch)
        and dsa_use_prefill_cp(batch)
        and is_cp_v2_active(batch)
        and isinstance(get_cp_strategy(), InterleaveCPStrategy)
        and attn.cp_size > 1
        and attn.current_attention_backend in ("dsa", "nsa")
        and attn.kv_lora_rank == 512
        and attn.qk_rope_head_dim == 64
        and attn.qk_head_dim == 256
        and batch.extend_prefix_lens_cpu is not None
        and any(batch.extend_prefix_lens_cpu)
    ):
        return False
    from sglang.srt.layers.layernorm import RMSNorm, _use_aiter
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.rope_variant import (
        DeepseekScalingRotaryEmbedding,
    )
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    norm, rope = attn.kv_a_layernorm, attn.rotary_emb
    pool, backend = get_token_to_kv_pool(), get_attn_backend()
    loc = batch.out_cache_loc
    metadata = batch.attn_cp_metadata
    logical = metadata.per_rank_logical_token or metadata.per_rank_actual_token
    # A wrapped/compact pool owns additional location translations. Keep those
    # paths on their original writer rather than guessing physical cache IDs.
    if not (
        not _use_aiter
        and type(pool) is DSATokenToKVPool
        and pool.dsa_kv_cache_store_fp8
        and pool.kv_cache_dim == 656
        and pool.layer_transfer_counter is None
        and backend.hisparse_coordinator is None
        and type(norm) is RMSNorm
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
        and not norm.fp32_residual
        and norm.override_orig_dtype is None
        and not getattr(norm, "x_pad_to_multiple", 0)
        and type(rope) in (RotaryEmbedding, DeepseekScalingRotaryEmbedding)
        and not rope._force_native
        and rope.rotary_dim == 64
        and rope.cos_sin_cache.dtype == torch.bfloat16
        and rope.cos_sin_cache.is_contiguous()
        and not attn._fuse_rope_for_trtllm_mla(batch)
        and not attn._skip_rope_for_dsa_tilelang_fused()
        and not attn._skip_rope_for_aiter_fused_mla()
        and latent.is_cuda
        and latent.dtype == torch.bfloat16
        and latent.ndim == 2
        and latent.shape[1] == 576
        and latent.stride(1) == 1
        and latent.stride(0) % 8 == 0
        and latent.data_ptr() % 16 == 0
        and norm.weight.dtype == latent.dtype
        and norm.weight.is_contiguous()
        and norm.weight.shape == (512,)
        and norm.weight.data_ptr() % 16 == 0
        and isinstance(loc, torch.Tensor)
        and loc.dtype == torch.int64
        and loc.ndim == 1
        and loc.is_contiguous()
        and loc.device == latent.device
        and loc.numel() == int(metadata.total_seq_lens)
        and min(logical) > 0
    ):
        return False
    from lightop import rmsnorm

    if 256 not in getattr(rmsnorm.op, "MLA_ROPE_SUPPORTED_Q_HEAD_STRIDES", ()):
        raise RuntimeError("GLM prefill KV fusion requires LightOp Q head stride 256")
    return True


def fuse_kv_rope_store(attn, batch, positions, query_nope, query_rope, latent):
    from lightop.rmsnorm import fused_rms_norm_rope_contiguous_q_cat

    tokens = latent.shape[0]
    packed_kv = torch.empty((tokens, 656), dtype=torch.uint8, device=latent.device)
    local_slots = torch.arange(tokens, dtype=torch.int64, device=latent.device)
    normalized = latent.new_empty((tokens, 512))
    query = query_nope.new_empty((tokens, attn.num_local_heads, 576))
    with torch.profiler.record_function("glm_prefill_kv_norm_rope_pack"):
        fused_rms_norm_rope_contiguous_q_cat(
            positions,
            query_rope,
            latent[:, 512:],
            latent[:, :512],
            normalized,
            attn.kv_a_layernorm.weight,
            attn.rotary_emb.cos_sin_cache,
            local_slots,
            packed_kv,
            query_nope,
            query,
            "fp8_e4m3",
            1.0,
            is_neox_style=attn.rotary_emb.is_neox_style,
            epsilon=attn.kv_a_layernorm.variance_epsilon,
            flashmla_fp8_layout=True,
            kv_quant_epsilon=1e-6,
        )
    # Every rank needs every new cache row. Exchange the exact packed cache
    # bytes, including scales and BF16 RoPE, rather than re-quantizing remote K.
    # Interleave's existing gather handles logical lengths and physical padding.
    with torch.profiler.record_function("glm_prefill_packed_kv_cp_gather_store"):
        full_kv = get_cp_strategy().gather_kv_cache(packed_kv, batch)
        cache = get_token_to_kv_pool().get_key_buffer(attn.attn_mqa.layer_id)
        cache.view(torch.uint8).view(-1, 656).index_copy_(
            0, batch.out_cache_loc, full_kv
        )
    report(attn, "KV RMSNorm + Q/K RoPE + FP8 store; packed CP exchange")
    # Only the prefix path is eligible: its attention already reads quantized
    # cache rows. A cold prefill must keep the original unquantized BF16 K path.
    return PrefillQWithStoredKV(query)
