"""Opt-in native replacement for paged FP8 prefill KV gather/dequant."""

import torch


def try_native_kv_gather(cache, indices):
    if not (
        cache.is_cuda
        and cache.dtype == torch.float8_e4m3fn
        and cache.ndim in (3, 4)
        and tuple(cache.shape[-2:]) == (1, 656)
        and cache.is_contiguous()
        and cache.data_ptr() % 16 == 0
        and indices.device == cache.device
        and indices.ndim == 1
        and indices.dtype == torch.int32
        and indices.is_contiguous()
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    ):
        return None
    from lightop import op

    # Legacy versions handle FP8 NaNs differently from the Triton baseline.
    # Do not silently substitute them even when the kernel symbol exists.
    if getattr(op, "PREFILL_GATHER_TRITON_NAN_COMPAT_VERSION", 0) < 1:
        raise RuntimeError(
            "GLM native KV gather requires LightOp's "
            "PREFILL_GATHER_TRITON_NAN_COMPAT_VERSION >= 1"
        )
    output = torch.empty(
        (indices.numel(), 1, 576), dtype=torch.bfloat16, device=cache.device
    )
    with torch.profiler.record_function("glm_prefill_native_kv_gather"):
        op.prefill_gather_and_upconvert_fp8_kv_cache(cache, indices, output)
    return output
