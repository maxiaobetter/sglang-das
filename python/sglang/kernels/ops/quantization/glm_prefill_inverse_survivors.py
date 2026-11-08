"""Default-off inverse broadcast survivors in the retained ordinary GLM scope."""

import hashlib
import importlib.util
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch

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

_BINARIES = {
    "token": (
        "SGLANG_GLM_PREFILL_TOKEN_QUANT_INV_SO",
        "e7870a15db717c3761e80725e238b8709ee1adb4ff92239121b7e4b8211f0361",
        "token_inverse_diag",
    ),
    "activation": (
        "SGLANG_GLM_PREFILL_ACTIVATION_INV_SO",
        "a13d3751d5cde8e9a4b66ea1cca769431dbb6051a1ef113bab6ef7b129d4bf9a",
        "activation_inverse_diag",
    ),
}
_LIGHTOP_SHA = "f4d69b478289cf93a2be4c0f8f92c7374877609276bd6ba40476ab5a10abee6c"
_ACTIVATION_ROWS = (14336, 84224, 84992)
_FLAGS = {
    "token": "SGLANG_GLM_PREFILL_TOKEN_QUANT_INV_BROADCAST",
    "activation": "SGLANG_GLM_PREFILL_ACTIVATION_INV_BROADCAST",
}
_LOGGED = set()
logger = logging.getLogger(__name__)


def _scope(kind):
    if (
        os.environ.get(_FLAGS[kind]) != "1"
        or torch.compiler.is_compiling()
        or not has_forward_context()
    ):
        return False
    if not getattr(
        get_forward_context().attn_backend, "_glm_prefill_ep_gather_k8_eligible", False
    ):
        return False
    if (
        is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    parallel = get_parallel()
    return (parallel.attn_cp_size, parallel.tp_size, parallel.moe_ep_size) == (8, 8, 8)


def _input_ok(value):
    return (
        value.is_cuda
        and value.dtype == torch.bfloat16
        and value.ndim == 2
        and value.is_contiguous()
        and value.storage_offset() == 0
        and value.data_ptr() % 16 == 0
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(value.device).gcnArchName.split(":", 1)[0]
        == "gfx938"
    )


@lru_cache(maxsize=2)
def _load(kind):
    from lightop import op

    if (
        torch.__version__ != "2.11.0"
        or torch.version.hip != "6.3.26113"
        or not torch._C._GLIBCXX_USE_CXX11_ABI
        or hashlib.sha256(Path(op.__file__).read_bytes()).hexdigest() != _LIGHTOP_SHA
    ):
        raise RuntimeError(
            "Inverse survivors require the validated Torch/HIP/LightOp ABI"
        )
    path_env, expected, name = _BINARIES[kind]
    configured_path = os.environ.get(path_env)
    if not configured_path:
        raise RuntimeError(f"Set {path_env} to the validated extension path")
    path = Path(configured_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Unvalidated inverse survivor extension: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load validated inverse survivor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path.resolve():
        raise RuntimeError("Inverse survivor resolved to another binary")
    return module


def _record(kind, value):
    if os.environ.get("SGLANG_GLM_PREFILL_INVERSE_TRACE") != "1":
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    key = (rank, kind, tuple(value.shape))
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(
            "GLM_INVERSE_SURVIVOR_SCOPE %s",
            json.dumps(
                dict(
                    rank=rank,
                    kind=kind,
                    shape=list(value.shape),
                    phase="ordinary_extend",
                    candidate_dispatched=True,
                    extension_sha256=_BINARIES[kind][1],
                ),
                sort_keys=True,
            ),
        )


def try_token_quant_inverse(input, output, scale):
    """Only the validated K16384 shape; K12288 keeps native quantization."""
    if os.environ.get(_FLAGS["token"]) != "1" or torch.compiler.is_compiling():
        return False
    if input.ndim != 2 or input.shape != (4096, 16384):
        return False
    if not _scope("token") or not _input_ok(input):
        return False
    if (
        output.shape != input.shape
        or output.dtype != torch.float8_e4m3fn
        or scale.shape != (4096, 1)
        or scale.dtype != torch.float32
        or any(
            not t.is_cuda
            or t.device != input.device
            or not t.is_contiguous()
            or t.storage_offset() != 0
            or t.data_ptr() % 16
            for t in (output, scale)
        )
        or len({t.untyped_storage().data_ptr() for t in (input, output, scale)}) != 3
    ):
        return False
    _load("token").run(input, output, scale, 1)
    _record("token", input)
    return True


def try_expert_activation_inverse(input):
    """Only the no-limit, no-topkcount caller and three actually tested M values."""
    if os.environ.get(_FLAGS["activation"]) != "1" or torch.compiler.is_compiling():
        return None
    if (
        input.ndim != 2
        or input.shape[1] != 4096
        or input.shape[0] not in _ACTIVATION_ROWS
    ):
        return None
    if not _scope("activation") or not _input_ok(input):
        return None
    rows = input.shape[0]
    # Match native Python API dtype; the extension ABI consumes its byte view.
    output = torch.empty((rows, 2048), dtype=torch.float8_e4m3fn, device=input.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=input.device)
    assert len({t.untyped_storage().data_ptr() for t in (input, output, scale)}) == 3
    assert output.data_ptr() % 16 == 0 and scale.data_ptr() % 16 == 0
    _load("activation").run(input, output.view(torch.uint8), scale, 1)
    _record("activation", input)
    return output, scale
