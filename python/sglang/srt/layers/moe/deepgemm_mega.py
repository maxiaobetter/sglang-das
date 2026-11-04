"""Adapter for DeepGEMM develop's deepgemm.mega API (not deep_gemm).

Pinned interface: a1efb4e619e232a40f3b13eb8f2c5b12a8867740.
Experimental gfx938 / HIP IPC integration; never selected by default.
"""

from pathlib import Path

import torch
import torch.distributed as dist


def transform_weights(w13, w13_scale, w2, w2_scale):
    from deepgemm.mega import transform_weights_for_mega_moe

    from sglang.srt.server_args import get_global_server_args

    if not get_global_server_args().disable_cuda_graph:
        raise ValueError("deepgemm_mega requires --disable-cuda-graph")
    arch = torch.cuda.get_device_properties(w13.device).gcnArchName.split(":")[0]
    if arch != "gfx938":
        raise ValueError(f"deepgemm.mega requires gfx938, got {arch}")
    for weight in (w13, w2):
        if weight.shape[1] % 16 or weight.shape[2] % 64:
            raise ValueError("deepgemm.mega w6 requires N % 16 == K % 64 == 0")
    # Preserve checkpoint FP8 values and per-channel scales. The alternative
    # transform_fp8_weights_for_mega_moe API would quantize weights again.
    return transform_weights_for_mega_moe(
        (w13.contiguous(), w13_scale.contiguous()),
        (w2.contiguous(), w2_scale.contiguous()),
    )


def create_buffer(group, *args, **kwargs):
    from deepgemm.mega import get_symm_buffer_for_mega_moe

    # This revision exchanges HIP IPC handles, not RDMA endpoints. Until
    # cross-host IPC is verified, reject it before opening any peer handles.
    # boot_id identifies the host even when ranks live in different containers.
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot_id = None
    hosts = [None] * group.size()
    dist.all_gather_object(hosts, boot_id, group=group)
    if not boot_id or len(set(hosts)) != 1:
        raise ValueError(
            "deepgemm_mega currently requires one verified HIP IPC host; "
            "cross-host/supernode IPC has not been validated. Use the existing "
            "DeepEP backend for multi-host CP/EP."
        )
    return get_symm_buffer_for_mega_moe(group, *args, **kwargs)


def run(hidden_states, topk_ids, topk_weights, moe, buf):
    from deepgemm.mega import fp8_mega_moe, mega_moe_preprocess

    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("deepgemm_mega expects BF16 activations")
    if getattr(moe.experts, "mega_moe_recv_stats", None) is not None:
        raise ValueError("deepgemm.mega does not expose EPLB receive statistics")
    num_tokens = hidden_states.shape[0]
    y = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    if num_tokens:
        mega_moe_preprocess(
            hidden_states.contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.to(torch.float32).contiguous(),
            buf,
            num_tokens,
            is_int8=False,
        )
    # Empty ranks must still participate in dispatch/combine barriers.
    limit = getattr(moe.config, "swiglu_limit", None)
    fp8_mega_moe(
        y,
        moe.experts.mega_l1_weights,
        moe.experts.mega_l2_weights,
        buf,
        activation_clamp=float("inf") if limit is None else limit,
        fast_math=False,
        is_int8=False,
    )
    return y
