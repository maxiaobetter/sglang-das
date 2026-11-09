"""GLM quantization and phase guards for decode, verify, and draft."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.glm_prefill_native import lightop_glm_op

_phase = ContextVar("glm_decode_o_proj_phase", default=None)


def glm_decode_phase_enabled(batch, *, is_nextn=False):
    if (
        not envs.SGLANG_GLM_OPT_DECODE.get()
        or batch is None
        or torch.compiler.is_compiling()
    ):
        return False
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.attention.dsa.forward_batch_utils import (
        effective_forward_mode,
    )
    from sglang.srt.runtime_context import get_exec

    if (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    ):
        return False
    mode = effective_forward_mode(batch)
    return (
        mode.is_decode()
        or mode.is_target_verify()
        or mode.is_draft_extend_v2()
        or (is_nextn and mode.is_extend())
    )


def try_shared_silu_quant(mlp, gate_up, batch):
    if (
        not getattr(mlp, "_glm_decode_shared_quant", False)
        or mlp.swiglu_limit is not None
        or not glm_decode_phase_enabled(
            batch, is_nextn=getattr(mlp, "_glm_decode_is_nextn", False)
        )
        or gate_up.ndim != 2
        or gate_up.shape[0] not in (4, 6, 24, 30, 32)
        or gate_up.shape[1] != 4096
        or gate_up.dtype != torch.bfloat16
        or not gate_up.is_cuda
        or not gate_up.is_contiguous()
        or gate_up.data_ptr() % 16
        or torch.version.hip is None
    ):
        return None
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        supports_lightop_input_quant,
    )

    if not supports_lightop_input_quant(mlp.down_proj):
        return None
    props = torch.cuda.get_device_properties(gate_up.device)
    if (
        props.gcnArchName.split(":", 1)[0] != "gfx938"
        or props.multi_processor_count != 64
    ):
        return None
    q = torch.empty(
        (gate_up.shape[0], 2048), device=gate_up.device, dtype=torch.float8_e4m3fn
    )
    scale = torch.empty(
        (gate_up.shape[0], 1), device=gate_up.device, dtype=torch.float32
    )
    # Match fuse_silu_and_mul -> BF16 -> per_token_quant_fp8, including
    # zero/tiny rows. The ordinary fused API has a different scale floor.
    lightop_glm_op("fuse_silu_mul_fp8_quant", "per_token_scale")(
        gate_up,
        q.view(torch.uint8),
        scale,
        0,
        None,
        1,
        -1,
        None,
        per_token_scale=True,
    )
    return q, scale, gate_up.dtype


@contextmanager
def o_proj_quant_scope(attn, batch):
    phase = None
    if envs.SGLANG_GLM_OPT_DECODE.get() and attn._lightop_glm_fusions:
        from sglang.srt.layers.attention.dsa.forward_batch_utils import (
            effective_forward_mode,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
            supports_lightop_input_quant,
        )

        if supports_lightop_input_quant(attn.o_proj):
            phase = effective_forward_mode(batch)
    token = _phase.set(phase)
    try:
        yield
    finally:
        _phase.reset(token)


def try_o_proj_quant(x, output, scale):
    if not envs.SGLANG_GLM_OPT_DECODE.get() or torch.compiler.is_compiling():
        return False
    phase = _phase.get()
    if phase is None:
        return False
    if phase.is_decode():
        rows = (1, 2, 3, 4, 5)
    elif phase.is_target_verify() or phase.is_draft_extend_v2():
        rows = (6, 12, 18, 24, 30)
    else:
        return False
    if (
        x.ndim != 2
        or x.shape[0] not in rows
        or x.shape[1] != 16384
        or x.dtype != torch.bfloat16
        or output.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or output.shape != x.shape
        or scale.shape != (x.shape[0], 1)
        or x.stride() != (16384, 1)
        or output.stride() != (16384, 1)
        or scale.stride() != (1, 1)
        or not all(
            t.is_cuda and t.device == x.device and t.is_contiguous()
            for t in (x, output, scale)
        )
        or x.data_ptr() % 64
        or output.data_ptr() % 64
        or scale.data_ptr() % 4
        or torch.version.hip is None
        or torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0]
        != "gfx938"
    ):
        return False
    ranges = []
    for tensor in (x, output, scale):
        storage = tensor.untyped_storage()
        begin, size = storage.data_ptr(), storage.nbytes()
        offset, width = tensor.storage_offset(), tensor.element_size()
        if (
            begin == 0
            or size <= 0
            or offset < 0
            or offset * width + tensor.numel() * width > size
            or tensor.data_ptr() != begin + offset * width
        ):
            return False
        ranges.append((begin, begin + size))
    if any(
        a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]
    ):
        return False
    lightop_glm_op("per_token_quant_fp8", "broadcast_inverse")(
        output, x, scale, broadcast_inverse=True
    )
    return True
