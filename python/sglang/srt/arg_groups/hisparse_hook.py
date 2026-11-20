from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv", "flashinfer_sparse_mla"},
}
HISPARSE_ROCM_DSA_BACKENDS = {"tilelang", "aiter"}
HISPARSE_HCU_DSA_BACKENDS_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}
HISPARSE_KV_CACHE_DTYPES = ("bfloat16", "fp8_e4m3")


def _is_hip() -> bool:
    from sglang.srt.server_args import is_hip

    return is_hip()


def _is_hcu() -> bool:
    from sglang.srt.server_args import is_hcu

    return is_hcu()


def _hisparse_default_backend(kv_cache_dtype: str) -> str:
    if _is_hcu():
        return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"
    if _is_hip():
        return "tilelang"
    return "flashmla_kv" if kv_cache_dtype == "fp8_e4m3" else "flashmla_sparse"


def _hisparse_allowed_backends(kv_cache_dtype: str) -> set[str]:
    if _is_hcu():
        # HCU FlashMLA consumes packed FP8 KV (656 bytes/token for GLM-5.3).
        # The generic ROCm tilelang/aiter path uses a different KV layout.
        return HISPARSE_HCU_DSA_BACKENDS_BY_DTYPE.get(kv_cache_dtype, set())
    if _is_hip():
        return HISPARSE_ROCM_DSA_BACKENDS
    return HISPARSE_CUDA_DSA_BACKENDS_BY_DTYPE.get(
        kv_cache_dtype, {"flashmla_sparse", "flashmla_kv", "flashinfer_sparse_mla"}
    )


# The hisparse DSA backend defaults moved to the resolution pipeline
# (arg_groups/overrides.py: _dsa_split_backend_resolution, hisparse arm).


def validate_hisparse_dsa_backend(
    server_args: ServerArgs, attr: str, label: str
) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    # Invoked after the DSA kv-cache-dtype / split-backend declarations:
    # read the resolving state through the view.
    view = resolved_view(server_args)
    backend = getattr(view, attr)
    kv_cache_dtype = view.kv_cache_dtype
    allowed_backends = _hisparse_allowed_backends(kv_cache_dtype)
    if backend is not None and backend not in allowed_backends:
        raise ValueError(
            f"HiSparse supports DSA {label} backend(s) {sorted(allowed_backends)} "
            f"on this platform with --kv-cache-dtype={kv_cache_dtype}, "
            f"but got --dsa-{label}-backend={backend}. "
            f"Please use one of {sorted(allowed_backends)}, or omit the option "
            "to let SGLang pick a backend for this platform."
        )


def validate_hisparse_kv_cache_dtype(server_args: ServerArgs) -> None:
    from sglang.srt.arg_groups.overrides import resolved_view

    kv_cache_dtype = resolved_view(server_args).kv_cache_dtype
    if kv_cache_dtype in HISPARSE_KV_CACHE_DTYPES:
        return

    choices = " or ".join(
        f"--kv-cache-dtype={dtype}" for dtype in HISPARSE_KV_CACHE_DTYPES
    )
    raise ValueError(
        f"HiSparse requires one of {HISPARSE_KV_CACHE_DTYPES} KV cache dtypes, "
        f"but got --kv-cache-dtype={kv_cache_dtype}. Please use {choices}."
    )


def validate_hisparse(server_args: ServerArgs) -> None:
    """Validate --enable-hisparse constraints (model class, radix cache, DSA backend)."""
    if not server_args.enable_hisparse:
        return

    from sglang.srt.configs.model_config import (
        is_deepseek_dsa,
        is_deepseek_v4,
    )

    hf_config = server_args.get_model_config().hf_config
    is_v4_hisparse = is_deepseek_v4(hf_config)
    is_hip = _is_hip()
    assert is_deepseek_dsa(hf_config) or is_v4_hisparse, (
        "--enable-hisparse is only supported for DSA (DeepSeek Sparse Attention) "
        "models (e.g., DeepSeek V3.2, GLM-5) and DeepSeek V4 now. "
    )

    assert (
        server_args.disable_radix_cache
    ), "Hierarchical sparse attention currently requires --disable-radix-cache."

    # DSv4 hisparse handles its own dtype/backend pairing elsewhere; the dtype-
    # aware checks below only apply to the DSA hisparse path.
    if is_hip and is_v4_hisparse:
        # TEMPORARY GUARD: DSv4 HiSparse is not supported on the unified-KV path.
        # In unified-KV mode c4_kv_pool is None, so DeepSeekV4HiSparseTokenToKVPoolAllocator
        # cannot attach and pool init dies with a cryptic AssertionError. Fail fast
        # at startup with a clear message instead. Remove once unified-KV HiSparse lands.
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            raise ValueError(
                "--enable-hisparse is not supported with the unified-KV path on ROCm"
                "(SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton) for DeepSeek-V4: "
                "HiSparse currently requires the separate packed KV layout. "
                "Either set SGLANG_HACK_FLASHMLA_BACKEND=triton, or run without "
                "--enable-hisparse."
            )
        return

    from sglang.srt.arg_groups.overrides import resolved_view

    if _is_hcu():
        is_unified_glm_fp8 = (
            server_args.disaggregation_mode == "null"
            and "GlmMoeDsaForCausalLM" in (hf_config.architectures or [])
            and resolved_view(server_args).kv_cache_dtype == "fp8_e4m3"
        )
        if server_args.disaggregation_mode != "decode" and not is_unified_glm_fp8:
            raise ValueError(
                "HCU DSA HiSparse supports PD decode, and unified serving "
                "for GLM DSA with FP8 KV cache."
            )
        if server_args.page_size != 64:
            raise ValueError("HCU DSA HiSparse requires --page-size 64.")
        if server_args.speculative_algorithm is not None:
            _validate_hcu_dsa_mtp(server_args, hf_config)

    if resolved_view(server_args).kv_cache_dtype not in (
        "bfloat16",
        "auto",
        "fp8_e4m3",
    ):
        validate_hisparse_kv_cache_dtype(server_args)

    for attr, label in [
        ("dsa_prefill_backend", "prefill"),
        ("dsa_decode_backend", "decode"),
    ]:
        validate_hisparse_dsa_backend(server_args, attr, label)


def _validate_hcu_dsa_mtp(server_args: ServerArgs, hf_config) -> None:
    from sglang.srt.mem_cache.sparsity import parse_hisparse_config

    model_config = server_args.get_model_config()
    if (
        "GlmMoeDsaForCausalLM" not in (hf_config.architectures or [])
        or server_args.speculative_algorithm not in ("EAGLE", "NEXTN")
        or server_args.enable_multi_layer_eagle
        or model_config.num_nextn_predict_layers != 1
        or server_args.speculative_eagle_topk != 1
        or server_args.speculative_adaptive
    ):
        raise ValueError(
            "HCU DSA HiSparse MTP requires GLM NextN/EAGLE with one draft layer, "
            "--speculative-eagle-topk 1 and fixed speculative steps."
        )
    if server_args.pp_size != 1 or server_args.dcp_size != 1:
        raise ValueError("HCU DSA HiSparse MTP requires PP=1 and DCP=1.")
    if server_args.enable_two_batch_overlap or server_args.enable_pdmux:
        raise ValueError(
            "HCU DSA HiSparse MTP currently uses one verify transaction at a time; "
            "two-batch overlap and PDMux require separate transaction workspaces."
        )
    if server_args.disaggregation_mode == "decode" and (
        server_args.disaggregation_transfer_backend not in ("mooncake", "fake")
    ):
        raise ValueError(
            "HCU DSA HiSparse MTP requires Mooncake for PD transfer or fake "
            "for decode smoke testing."
        )
    width = server_args.speculative_num_draft_tokens
    cfg = parse_hisparse_config(server_args)
    top_k = getattr(model_config.hf_text_config, "index_topk", cfg.top_k)
    if not isinstance(cfg.device_buffer_size, int) or isinstance(
        cfg.device_buffer_size, bool
    ):
        raise ValueError("HiSparse MTP device_buffer_size must be an integer.")
    if (
        width is None
        or not 2 <= width <= 16
        or server_args.speculative_num_steps is None
        or width != server_args.speculative_num_steps + 1
        or width * top_k > 32768
        or top_k < 1024
        or top_k % 64
        or cfg.top_k != top_k
    ):
        raise ValueError(
            "HCU DSA HiSparse MTP requires verify width 2-16, width * index_topk "
            "<= 32768 and a matching HiSparse top_k >= 1024 divisible by 64."
        )
    if not top_k <= cfg.device_buffer_size <= 32768 or cfg.device_buffer_size % 64:
        raise ValueError(
            "HCU DSA HiSparse MTP requires a page-aligned device_buffer_size "
            "between index_topk and 32768."
        )
    if server_args.max_running_requests is None:
        raise ValueError(
            "Set --max-running-requests for HCU DSA HiSparse MTP so per-request "
            "verify metadata is included in the memory budget."
        )
    attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
    per_worker = server_args.max_running_requests // max(1, attn_dp_size)
    slots = per_worker + (server_args.disaggregation_decode_extra_slots or 0) + 1
    if per_worker < 1 or slots > cfg.device_buffer_size:
        raise ValueError("HCU DSA HiSparse MTP request slots exceed CLOCK capacity.")
