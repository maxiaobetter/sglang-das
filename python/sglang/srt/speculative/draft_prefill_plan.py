"""Experimental query compaction for single-layer GLM FP8 draft prefill.

The cache producers still see the original batch. Only query consumers and
the post-attention residual/MLP use this plan. In particular, an empty local
query set must not suppress either cache writes or another EP rank's work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsa_backend import DSAMetadata
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)
_reported_fallbacks = set()


@dataclass
class DraftPrefillPlan:
    complete: Tuple[bool, ...]
    next_prompt_tokens: Tuple[Optional[int], ...]
    total_tokens: int
    request_indices: torch.Tensor
    local_rows: torch.Tensor
    local_query_lens: Tuple[int, ...]
    rank_query_counts: Tuple[int, ...]
    gather_rows: torch.Tensor

    @property
    def num_queries(self) -> int:
        return self.request_indices.numel()

    @property
    def num_local_queries(self) -> int:
        return self.local_rows.numel()

    @classmethod
    def build(cls, extend_lens, complete, next_prompt_tokens, cp_size, cp_rank, device):
        # Every request must have a row on every CP rank. Keeping the original
        # per-request KV tables then also keeps ragged TopK bases identical.
        if not extend_lens or any(length < cp_size for length in extend_lens):
            raise ValueError("Draft prefill compaction requires extend_len >= CP size")
        if len(extend_lens) != len(complete) or len(complete) != len(
            next_prompt_tokens
        ):
            raise ValueError("Draft prefill completion snapshot does not match batch")
        request_indices, local_rows, gather_locations = [], [], []
        counts = [0] * cp_size
        local_query_lens = [0] * len(extend_lens)
        end = 0
        for request_index, (length, finished) in enumerate(zip(extend_lens, complete)):
            end += length
            if not finished:
                continue
            owner = (end - 1) % cp_size
            request_indices.append(request_index)
            gather_locations.append((owner, counts[owner]))
            counts[owner] += 1
            if owner == cp_rank:
                local_rows.append((end - 1) // cp_size)
                local_query_lens[request_index] = 1
        max_rows = max(counts)

        def tensor(values):
            return torch.tensor(values, dtype=torch.long, device=device)

        return cls(
            complete=tuple(complete),
            next_prompt_tokens=tuple(next_prompt_tokens),
            total_tokens=end,
            request_indices=tensor(request_indices),
            local_rows=tensor(local_rows),
            local_query_lens=tuple(local_query_lens),
            rank_query_counts=tuple(counts),
            gather_rows=tensor(
                [rank * max_rows + row for rank, row in gather_locations]
            ),
        )

    def select_local(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            # Some installed PyTorch builds do not implement FP8 index_select.
            # Gather payload bytes, without dequantization or another rounding.
            return (
                value.view(torch.uint8)
                .index_select(0, self.local_rows)
                .view(value.dtype)
            )
        return value.index_select(0, self.local_rows)

    def validate_attention_dispatch(self, backend, forward_mode) -> None:
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
            TopkTransformMethod,
        )

        # Called after this batch's init_forward_metadata, before embeddings
        # or cache producers. A capability flag does not describe this dispatch.
        metadata = backend.forward_metadata
        if (
            backend.enable_auto_select_prefill_impl
            or backend.use_mha
            or backend.dsa_prefill_impl != "flashmla_sparse"
            or backend.get_topk_transform_method(forward_mode)
            != TopkTransformMethod.RAGGED
            or metadata is None
            or metadata.topk_indices_offset is None
        ):
            raise RuntimeError(
                "Draft prefill tail rows require this batch's fixed "
                "flashmla_sparse MLA dispatch and RAGGED metadata; "
                "refusing to execute before cache writes"
            )

    def select_attention_metadata(self, metadata: DSAMetadata) -> DSAMetadata:
        # Do not rebuild K tables from a synthetic one-token extend: that would
        # change both ragged offsets and cold-prefill BF16-vs-cached-FP8 math.
        select = self.select_local
        lengths = select(metadata.dsa_cache_seqlens_int32)
        cu_k = lengths.new_zeros(self.num_local_queries + 1)
        cu_k[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
        query_lens = lengths.new_tensor(self.local_query_lens)
        cu_q = lengths.new_zeros(len(self.complete) + 1)
        cu_q[1:] = torch.cumsum(query_lens, dim=0, dtype=torch.int32)
        return replace(
            metadata,
            max_seq_len_q=1 if self.num_local_queries else 0,
            cu_seqlens_q=cu_q,
            dsa_cache_seqlens_int32=lengths,
            dsa_cu_seqlens_q=torch.arange(
                self.num_local_queries + 1, dtype=torch.int32, device=lengths.device
            ),
            dsa_cu_seqlens_k=cu_k,
            dsa_extend_seq_lens_list=list(self.local_query_lens),
            dsa_seqlens_expanded=select(metadata.dsa_seqlens_expanded),
            topk_indices_offset=select(metadata.topk_indices_offset),
            indexer_k_start_end=tuple(select(t) for t in metadata.indexer_k_start_end),
            token_to_batch_idx=select(metadata.token_to_batch_idx),
            # These plans belong to the original query rows, not the KV tables.
            flashmla_metadata=None,
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
            topk_v2_plan=None,
        )

    def gather(self, value: torch.Tensor) -> torch.Tensor:
        """Gather only selected rows, restoring completed-request order."""
        if value.shape[0] != self.num_local_queries:
            raise ValueError("Draft prefill output does not match local query plan")
        max_rows = max(self.rank_query_counts)
        if max_rows == 0:
            return value
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import (
            attn_cp_all_gather_into_tensor,
            is_allocation_symmetric,
        )
        from sglang.srt.runtime_context import get_parallel

        # Padding is only for this gather. It never reaches attention or MoE.
        padded = value.new_zeros((max_rows, *value.shape[1:]))
        padded[: value.shape[0]].copy_(value)
        with use_symmetric_memory(
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
        ):
            gathered = value.new_empty(
                (len(self.rank_query_counts) * max_rows, *value.shape[1:])
            )
        attn_cp_all_gather_into_tensor(gathered, padded)
        return gathered.index_select(0, self.gather_rows)

    def scatter_requests(self, value: torch.Tensor, fill_value=0) -> torch.Tensor:
        """Retain the B-row worker/PD contract; unfinished rows are invalid."""
        if value.shape[0] != self.num_queries:
            raise ValueError("Draft prefill result does not match completed requests")
        result = value.new_full((len(self.complete), *value.shape[1:]), fill_value)
        result.index_copy_(0, self.request_indices, value)
        return result


def get_draft_prefill_plan(forward_batch) -> Optional[DraftPrefillPlan]:
    return getattr(forward_batch.spec_info, "prefill_plan", None)


def _is_channel_fp8(model) -> bool:
    # GLM-5.3 uses compressed-tensors CHANNEL/TOKEN W8A8, not the generic
    # Fp8Config block/per-tensor route. Check the loaded consumer schemes too.
    quant = getattr(model.config, "quantization_config", None)
    if not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors":
        return False
    groups = quant.get("config_groups", {})
    if not groups:
        return False
    for group in groups.values():
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if not (
            weight.get("num_bits") == activation.get("num_bits") == 8
            and weight.get("type") == activation.get("type") == "float"
            and weight.get("strategy") == "channel"
            and activation.get("strategy") == "token"
            and activation.get("dynamic") is True
            and weight.get("block_structure") is None
            and activation.get("block_structure") is None
        ):
            return False
    from compressed_tensors.quantization import QuantizationStrategy

    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
        CompressedTensorsW8A8Fp8MoE,
    )

    decoder = model.model.decoder
    for name in ("fused_qkv_a_proj_with_mqa", "q_b_proj", "o_proj"):
        scheme = getattr(getattr(decoder.self_attn, name, None), "scheme", None)
        if not (
            isinstance(scheme, CompressedTensorsW8A8Fp8)
            and scheme.strategy == QuantizationStrategy.CHANNEL
            and scheme.weight_block_size is None
            and not scheme.is_static_input_scheme
        ):
            return False
    scheme = getattr(decoder.mlp.experts, "scheme", None)
    return (
        isinstance(scheme, CompressedTensorsW8A8Fp8MoE)
        and scheme.weight_quant.strategy == QuantizationStrategy.CHANNEL
        and scheme.input_quant.strategy == QuantizationStrategy.TOKEN
        and not scheme.static_input_scales
        and not scheme.block_quant
    )


def maybe_plan_draft_prefill(
    batch: ScheduleBatch, runner: ModelRunner, *, topk: int, hidden_before_norm: bool
) -> Optional[DraftPrefillPlan]:
    if not envs.SGLANG_GLM_DRAFT_PREFILL_TAIL_ROWS.get():
        return None

    # Resolve eligibility before any producer writes a cache. Never retry a
    # partially executed compact forward through the baseline path.
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
    from sglang.srt.layers.cp.utils import enable_cp_v2, get_cp_strategy
    from sglang.srt.layers.moe.utils import get_deepep_mode, get_moe_a2a_backend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
    from sglang.srt.utils import is_hcu

    parallel = get_parallel()
    model = runner.model
    config = model.config
    backend = runner.attn_backend
    complete = batch.draft_prefill_complete
    next_prompt_tokens = batch.draft_prefill_next_prompt_tokens
    checks = (
        (lambda: is_hcu(), "HCU only"),
        (
            lambda: runner.server_args.disaggregation_mode == "prefill",
            "PD prefill only",
        ),
        (
            lambda: type(model).__name__ == "GlmMoeDsaForCausalLMNextN",
            "GLM DSA NextN only",
        ),
        (
            lambda: config.hidden_size == 6144
            and config.num_hidden_layers == 78
            and runner.model_config.num_nextn_predict_layers == 1
            and runner.layer_info.num_effective_layers == 1,
            "single-layer GLM-5.3 only",
        ),
        (
            lambda: _is_channel_fp8(model),
            "channel FP8 only",
        ),
        (
            lambda: parallel.tp_size
            == parallel.attn_cp_size
            == parallel.moe_ep_size
            == 8
            and parallel.attn_tp_size == parallel.moe_tp_size == parallel.pp_size == 1
            and parallel.attn_dp_size == parallel.moe_dp_size == 1
            and not parallel.dcp_enabled
            and parallel.attn_cp_group.ranks == parallel.moe_ep_group.ranks,
            "matching CP8/EP8 groups with TP1 attention/experts only",
        ),
        (
            lambda: get_moe_a2a_backend().is_deepep()
            and get_deepep_mode().resolve(True).is_normal(),
            "DeepEP normal only",
        ),
        (
            lambda: batch.forward_mode == ForwardMode.EXTEND
            and enable_cp_v2()
            and getattr(get_cp_strategy(), "name", None) == "interleave"
            and parallel.dsa_prefill_cp_mode == "round-robin-split",
            "ordinary CP-v2 interleave extend only",
        ),
        (
            lambda: not runner.server_args.enable_torch_compile
            and (
                runner.prefill_cuda_graph_runner is None
                or isinstance(runner.prefill_cuda_graph_runner, EagerRunner)
            ),
            "eager prefill only",
        ),
        (
            lambda: type(backend).__name__ == "DeepseekSparseAttnBackend"
            # Auto dispatch is resolved later, during metadata initialization.
            # Its current implementation may describe a previous batch.
            and not backend.enable_auto_select_prefill_impl,
            "fixed DSA prefill backend required; flashmla_auto is unsupported",
        ),
        (
            lambda: backend.dsa_prefill_impl == "flashmla_sparse"
            and backend.dsa_topk_backend.is_sgl_kernel()
            and backend.get_topk_transform_method(ForwardMode.EXTEND)
            == TopkTransformMethod.RAGGED
            and backend.hisparse_coordinator is None
            and backend.dsa_kv_cache_store_fp8
            and backend.kv_cache_dim == 656
            and runner.dtype == torch.bfloat16,
            "ragged TopK with BF16 sparse FlashMLA only",
        ),
        (
            lambda: not runner.server_args.enable_lora
            and batch.input_embeds is None
            and all(r.multimodal_inputs is None for r in batch.reqs)
            and not batch.is_prefill_only
            and not get_exec().overlap.enable_two_batch_overlap
            and not model.logits_processor.do_tensor_parallel_all_gather_dp_attn
            and not model.logits_processor.use_tp_lm_head_all_to_all
            and not hidden_before_norm,
            "text generation without LoRA, TBO or auxiliary hidden capture only",
        ),
        (
            lambda: topk == 1
            and not get_spec().enable_multi_layer_eagle
            and not get_spec().speculative_use_rejection_sampling,
            "single-layer topk1 without rejection sampling only",
        ),
        (
            lambda: complete is not None
            and next_prompt_tokens is not None
            and len(complete) == len(batch.reqs) == len(next_prompt_tokens)
            and all(
                done or token is not None
                for done, token in zip(complete, next_prompt_tokens)
            ),
            "complete scheduling-time prompt snapshot required",
        ),
        (
            lambda: batch.extend_lens is not None
            and all(length >= 8 for length in batch.extend_lens),
            "each request must have at least eight new tokens",
        ),
    )
    for eligible, reason in checks:
        if not eligible():
            if reason not in _reported_fallbacks:
                logger.info("Draft prefill tail rows: baseline fallback (%s)", reason)
                _reported_fallbacks.add(reason)
            return None
    plan = DraftPrefillPlan.build(
        batch.extend_lens,
        complete,
        next_prompt_tokens,
        parallel.attn_cp_size,
        parallel.attn_cp_rank,
        batch.input_ids.device,
    )
    logger.debug(
        "Draft prefill tail rows: N=%d, R=%d, CP query counts=%s",
        plan.total_tokens,
        plan.num_queries,
        plan.rank_query_counts,
    )
    return plan
