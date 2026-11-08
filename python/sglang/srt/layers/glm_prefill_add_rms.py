"""Default-off, real-input-validated GLM main-prefill Add RMS dispatch."""

import hashlib
import importlib.util
import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)
_EXTENSION_ENV = "SGLANG_GLM_PREFILL_ADD_RMS_ALIGN16_SO"
_EXTENSION_SHA256 = "11e6a0dff8d96913fb6a22ff12adee7179c004a3dab97c2443fa2bbb0ecab295"
_LIGHTOP_SHA256 = "f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c"
_reported = set()


def _channel_fp8(scheme):
    from compressed_tensors.quantization import QuantizationStrategy

    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        CompressedTensorsW8A8Fp8,
    )

    return (
        isinstance(scheme, CompressedTensorsW8A8Fp8)
        and scheme.strategy == QuantizationStrategy.CHANNEL
        and scheme.weight_block_size is None
        and not scheme.is_static_input_scheme
    )


def mark_decoder_norms(decoder):
    """Schemes already exist when the decoder's two norm owners are created."""
    attn = decoder.self_attn
    if decoder.is_nextn or not attn._hcu_glm_prefill:
        return
    # Reject mixed W4/FP8 and block checkpoints even if these two attention
    # projections alone happen to use channel FP8.
    quant = getattr(decoder.config, "quantization_config", None)
    if not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors":
        return
    groups = quant.get("config_groups", {})
    if not groups:
        return
    for group in groups.values():
        weight = group.get("weights") or {}
        activation = group.get("input_activations") or {}
        if (
            weight.get("num_bits") != 8
            or weight.get("type") != "float"
            or weight.get("strategy") != "channel"
            or weight.get("block_structure") is not None
            or activation.get("num_bits") != 8
            or activation.get("type") != "float"
            or activation.get("strategy") != "token"
            or activation.get("dynamic") is not True
            or activation.get("block_structure") is not None
        ):
            return
    schemes = tuple(
        getattr(getattr(attn, name, None), "scheme", None)
        for name in ("fused_qkv_a_proj_with_mqa", "q_b_proj")
    )
    if not all(_channel_fp8(scheme) for scheme in schemes):
        return
    for role in ("input_layernorm", "post_attention_layernorm"):
        norm = getattr(decoder, role)
        norm._glm_prefill_add_rms_owner = (decoder.layer_id, role)
        norm._glm_prefill_add_rms_schemes = schemes


@lru_cache(maxsize=1)
def _load_extension():
    # No JIT, search-path import, library override or replacement of LightOp.
    from lightop import op

    if (
        torch.__version__ != "2.11.0"
        or torch.version.hip != "6.3.26113"
        or not torch._C._GLIBCXX_USE_CXX11_ABI
        or hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() != _LIGHTOP_SHA256
    ):
        raise RuntimeError("Add RMS requires the validated Torch/HIP/LightOp ABI")
    configured_path = os.environ.get(_EXTENSION_ENV)
    if not configured_path:
        raise RuntimeError(f"Set {_EXTENSION_ENV} to the validated extension path")
    path = Path(configured_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != _EXTENSION_SHA256:
        raise RuntimeError(f"Unvalidated Add RMS extension at {path}")
    spec = importlib.util.spec_from_file_location("add_rms_align16_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the validated Add RMS extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("Add RMS extension resolved to a different binary")
    return module


def _record(norm, x, reason, scope):
    if not envs.SGLANG_GLM_PREFILL_ADD_RMS_ALIGN16_TRACE.get():
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    owner = getattr(norm, "_glm_prefill_add_rms_owner", None)
    key = (rank, reason, owner)
    if key in _reported:
        return
    _reported.add(key)
    parallel = get_parallel()
    logger.info(
        "GLM_ADD_RMS_SCOPE %s",
        json.dumps(
            dict(
                rank=rank,
                reason=reason,
                candidate_dispatched=reason == "dispatch",
                scope_enabled=scope,
                phase="ordinary_extend" if scope else "outside_validated_scope",
                owner=owner,
                rows=x.shape[0] if x.ndim else None,
                cp=parallel.attn_cp_size,
                tp=parallel.tp_size,
                ep=parallel.moe_ep_size,
            ),
            sort_keys=True,
        ),
    )


def try_glm_prefill_add_rms_align16(norm, x, residual, post_residual_addition):
    # Return before any tracing/import/device query inside compiled code.
    if torch.compiler.is_compiling():
        return False
    backend = get_forward_context().attn_backend if has_forward_context() else None
    scope = bool(getattr(backend, "_glm_prefill_ep_gather_k8_eligible", False))

    def reject(reason):
        _record(norm, x, reason, scope)
        return False

    if not envs.SGLANG_GLM_PREFILL_ADD_RMS_ALIGN16.get():
        return reject("flag_off")
    if (
        is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):
        return reject("graph")
    if not scope:
        return reject("outside_scope")
    if not getattr(norm, "_glm_prefill_add_rms_owner", None):
        return reject("not_decoder_owner")
    schemes = getattr(norm, "_glm_prefill_add_rms_schemes", ())
    if len(schemes) != 2 or not all(_channel_fp8(scheme) for scheme in schemes):
        return reject("quantization")
    parallel = get_parallel()
    if (parallel.attn_cp_size, parallel.tp_size, parallel.moe_ep_size) != (8, 8, 8):
        return reject("parallel_groups")
    if (
        post_residual_addition is not None
        or norm.variance_size_override is not None
        or norm.cast_x_before_out_mul
        or norm.fp32_residual
        or not norm.has_weight
        or norm.override_orig_dtype is not None
        or getattr(norm, "x_pad_to_multiple", 0)
        or not math.isfinite(norm.variance_epsilon)
        or norm.variance_epsilon <= 0
    ):
        return reject("norm_variant")
    weight = norm.weight.data
    tensors = (x, residual, weight)
    if (
        x.ndim != 2
        or x.shape[0] not in (704, 832, 1408, 3520, 4096)
        or x.shape[1] != 6144
        or residual.shape != x.shape
        or weight.shape != (6144,)
        or any(t.dtype != torch.bfloat16 for t in tensors)
    ):
        return reject("shape_dtype")
    if (
        torch.version.hip is None
        or not all(t.is_cuda and t.device == x.device for t in tensors)
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)
        or len({t.untyped_storage().data_ptr() for t in tensors}) != 3
    ):
        return reject("storage_device")
    if (
        torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0]
        != "gfx938"
    ):
        return reject("architecture")
    _load_extension().run(x, residual, weight, norm.variance_epsilon, 1)
    _record(norm, x, "dispatch", scope)
    return True
