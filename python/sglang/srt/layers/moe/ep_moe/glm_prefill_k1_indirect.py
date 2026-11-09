"""Default-off GLM FP8 main-prefill metadata scatter and indirect-A K1."""

import hashlib
import importlib.util
import json
import logging
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
_EXTENSION_ENV = "SGLANG_GLM_PREFILL_K1_INDIRECT_A_SO"
_EXTENSION_SHA256 = "e59ac60daff9ff6788bf05c91e48dbf4cf3e966d41f2079d6e652be4b30cac9d"
_DEEPGEMM_SHA256 = "79325b382a56de60f69cfaf8f75fe2b725d2092c002d1cf67bd1248f1537af7a"
_LIGHTOP_SHA256 = "f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c"
_DEEPEP_SHA256 = "7ee4f4edec4e6fa275b6d64d842e3925c4ac2ca4b72419046a03d0d103a16d8d"
_reported = set()


@lru_cache(maxsize=1)
def _load_extension():
    from deep_ep import deep_ep_cpp
    from deepgemm import op as deepgemm_op
    from lightop import op as lightop_op

    if (
        torch.__version__ != "2.11.0"
        or torch.version.hip != "6.3.26113"
        or not torch._C._GLIBCXX_USE_CXX11_ABI
        or hashlib.sha256(Path(deep_ep_cpp.__file__).read_bytes()).hexdigest()
        != _DEEPEP_SHA256
        or hashlib.sha256(Path(deepgemm_op.__file__).read_bytes()).hexdigest()
        != _DEEPGEMM_SHA256
        or hashlib.sha256(Path(lightop_op.__file__).read_bytes()).hexdigest()
        != _LIGHTOP_SHA256
    ):
        raise RuntimeError("Indirect K1 requires the validated Torch/HIP/operator ABI")
    configured_path = os.environ.get(_EXTENSION_ENV)
    if not configured_path:
        spec = importlib.util.find_spec("r09plus_wasp_indirect")
        if spec is None or spec.origin is None:
            raise RuntimeError(
                "Install the matching r09plus_wasp_indirect dependency or set "
                + _EXTENSION_ENV
            )
        configured_path = spec.origin
    path = Path(configured_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != _EXTENSION_SHA256:
        raise RuntimeError(f"Unvalidated indirect K1 extension at {path}")
    spec = importlib.util.spec_from_file_location("r09plus_wasp_indirect", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the validated indirect K1 extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("Indirect K1 resolved to a different binary")
    return module


def _record(layer, x, reason, scope):
    if not envs.SGLANG_GLM_PREFILL_K1_INDIRECT_A_TRACE.get():
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    key = (rank, getattr(layer, "layer_id", None), reason)
    if key in _reported:
        return
    _reported.add(key)
    logger.info(
        "GLM_K1_INDIRECT_SCOPE %s",
        json.dumps(
            dict(
                rank=rank,
                layer=key[1],
                reason=reason,
                candidate_dispatched=reason == "dispatch",
                scope_enabled=scope,
                recv_rows=x.shape[0] if x.ndim else None,
                extension_sha256=_EXTENSION_SHA256 if reason == "dispatch" else None,
            ),
            sort_keys=True,
        ),
    )


def _channel_fp8_scheme(layer):
    from compressed_tensors.quantization import QuantizationStrategy

    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
        CompressedTensorsW8A8Fp8MoE,
    )

    scheme = getattr(layer, "scheme", None)
    return (
        isinstance(scheme, CompressedTensorsW8A8Fp8MoE)
        and scheme.weight_quant.strategy == QuantizationStrategy.CHANNEL
        and scheme.input_quant.strategy == QuantizationStrategy.TOKEN
        and scheme.weight_block_size is None
        and not scheme.static_input_scales
        and scheme.use_hcu_fp8_w8a8_moe
        and scheme.use_deepep
    )


def try_glm_prefill_k1_indirect_a(
    layer, x, scale, topk, topk_weights, padded_counts, padded_gpu, weight, n
):
    """Return (BF16 K1, new expert ids, new token/k map), or use native path.

    The caller is exclusively the FP8 contiguous DeepEP normal-dispatch method.
    Its padded counts come from DeepEP expert_alignment=256, not routing guesses.
    All consumers must use these returned maps through K2 and ordered gather.
    """
    if (
        not envs.SGLANG_GLM_PREFILL_K1_INDIRECT_A.get()
        and not envs.SGLANG_GLM_PREFILL_K1_INDIRECT_A_TRACE.get()
    ):
        return None
    if torch.compiler.is_compiling():
        return None
    backend = get_forward_context().attn_backend if has_forward_context() else None
    scope = bool(getattr(backend, "_glm_prefill_ep_gather_k8_eligible", False))

    def reject(reason):
        _record(layer, x, reason, scope)
        return None

    if not envs.SGLANG_GLM_PREFILL_K1_INDIRECT_A.get():
        return reject("flag_off")
    if not scope:
        return reject("outside_scope")
    if (
        is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):
        return reject("graph")
    if not getattr(layer, "_glm_prefill_k1_main_owner", False):
        return reject("not_main_owner")
    parallel = get_parallel()
    if (parallel.attn_cp_size, parallel.tp_size, parallel.moe_ep_size) != (8, 8, 8):
        return reject("parallel_groups")
    if (
        not _channel_fp8_scheme(layer)
        or not layer.use_fp8_w8a8
        or layer.use_block_quant
        or layer.use_w4afp8
        or layer.use_w4a8_marlin
        or layer.use_w8a8_marlin
        or getattr(layer, "use_w4a16_marlin", False)
        or layer.moe_runner_config.activation != "silu"
    ):
        return reject("quantization")
    m = sum(padded_counts)
    if (
        len(padded_counts) != 32
        or any(type(c) is not int or c < 0 or c % 256 for c in padded_counts)
        or not 0 < m < 2**31
        or not 0 < m * n * 2 < 2**31
        or x.ndim != 2
        or x.shape[1] != 6144
        or not 0 < x.shape[0] * 6144 < 2**31
        or n != 4096
        or scale.shape != (x.shape[0], 1)
        or topk.shape != (x.shape[0], 8)
        or topk_weights.shape != topk.shape
        or padded_gpu.shape != (32,)
        or weight[0].shape != (32, 96, 128, 2, 4, 16, 16)
        or weight[1].shape != (32, 4096, 1)
        or x.dtype != torch.float8_e4m3fn
        or weight[0].dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or weight[1].dtype != torch.float32
        or topk_weights.dtype != torch.float32
        or topk.dtype != torch.int64
        or padded_gpu.dtype != torch.int32
    ):
        return reject("shape_dtype")
    tensors = (x, scale, topk, topk_weights, padded_gpu, *weight)
    if (
        torch.version.hip is None
        or not all(t.is_cuda and t.device == x.device for t in tensors)
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)
    ):
        return reject("storage_device")
    if (
        torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0]
        != "gfx938"
    ):
        return reject("architecture")
    module = _load_extension()
    src = torch.empty((m,), dtype=torch.int32, device=x.device)
    ids = torch.empty_like(src)
    out_index = torch.empty_like(topk, dtype=torch.int32)
    counters = torch.empty_like(padded_gpu)
    gateup = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    module.metadata(topk, padded_gpu, counters, src, ids, out_index)
    module.indirect(x, weight[0], gateup, scale, weight[1], ids, src)
    _record(layer, x, "dispatch", scope)
    return gateup, ids, out_index
