"""Default-off inverse broadcast survivors in the retained ordinary GLM scope."""

import json
import logging
import os

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.glm_prefill_native import lightop_glm_op
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

_ACTIVATION_ROWS = (14336, 84224, 84992)
_FLAGS = {
    "token": "SGLANG_GLM_PREFILL_TOKEN_QUANT_INV_BROADCAST",
    "activation": "SGLANG_GLM_PREFILL_ACTIVATION_INV_BROADCAST",
}
_LOGGED = set()
logger = logging.getLogger(__name__)


def _scope(kind):
    if (
        (not envs.SGLANG_GLM_OPT_PREFILL.get() and os.environ.get(_FLAGS[kind]) != "1")
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
                    broadcast_inverse=True,
                ),
                sort_keys=True,
            ),
        )


def try_token_quant_inverse(input, output, scale):
    """Only the validated K16384 shape; K12288 keeps native quantization."""
    if (
        not envs.SGLANG_GLM_OPT_PREFILL.get() and os.environ.get(_FLAGS["token"]) != "1"
    ) or torch.compiler.is_compiling():
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
    lightop_glm_op("per_token_quant_fp8", "broadcast_inverse")(
        output, input, scale, broadcast_inverse=True
    )
    _record("token", input)
    return True


def try_expert_activation_inverse(input):
    """Only the no-limit, no-topkcount caller and three actually tested M values."""
    if (
        not envs.SGLANG_GLM_OPT_PREFILL.get()
        and os.environ.get(_FLAGS["activation"]) != "1"
    ) or torch.compiler.is_compiling():
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
    lightop_glm_op("fuse_silu_mul_fp8_quant", "broadcast_inverse")(
        input,
        output.view(torch.uint8),
        scale,
        0,
        None,
        1,
        -1,
        None,
        broadcast_inverse=True,
    )
    _record("activation", input)
    return output, scale
