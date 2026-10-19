"""HCU: HIPC W4A8 versus unpacked INT4, including graph replay and normal EP."""

from types import MethodType, SimpleNamespace

import torch
from lightop import fuse_silu_mul_clamp_quant, fuse_silu_mul_quant
from lightop.quant import per_token_quant_int8

# Registers the masked activation custom op and supplies normal EP scatter/gather.
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.quantization.hcu_deepgemm_w4a8_utils import (
    prepare_slimquant_w4a8_deepgemm_weights,
)
from sglang.srt.layers.quantization.slimquant_w4a8 import (
    SlimQuantW4A8Int8MoEMethod,
)

torch.manual_seed(20261016)
E, M, K, N = 2, 37, 512, 512
layer = torch.nn.Module()
w1 = torch.randint(-128, 128, (E, 2 * N, K // 2), device="cuda", dtype=torch.int8)
w2 = torch.randint(-128, 128, (E, K, N // 2), device="cuda", dtype=torch.int8)
s1 = torch.rand(E, 2 * N, 1, device="cuda") * 0.003 + 0.001
s2 = torch.rand(E, K, 1, device="cuda") * 0.003 + 0.001
for name, value in (
    ("w13_weight", w1),
    ("w2_weight", w2),
    ("w13_weight_scale", s1),
    ("w2_weight_scale", s2),
):
    layer.register_parameter(name, torch.nn.Parameter(value.clone(), requires_grad=False))
prepare_slimquant_w4a8_deepgemm_weights(layer)
prepare_slimquant_w4a8_deepgemm_weights(layer)
torch.testing.assert_close(layer.w13_weight_scale, s1 / 16, rtol=0, atol=0)


def unpack(weight):
    byte = weight.to(torch.int32) & 255
    nibble = torch.stack((byte // 16, byte % 16), dim=-1)
    return torch.where(nibble >= 8, nibble - 16, nibble).flatten(-2).float()


u1, u2 = unpack(w1), unpack(w2)


def assert_numerics(actual, reference):
    assert torch.isfinite(actual).all()
    if reference.numel() == 0:
        return
    relative_rmse = (
        (actual.float() - reference.float()).square().mean()
        / reference.float().square().mean()
    ).sqrt().item()
    assert relative_rmse < 0.02, relative_rmse
    torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.001)


def golden(qx, xs, limit):
    first = (
        torch.bmm(qx.float(), u1.transpose(1, 2)) * xs * s1.transpose(1, 2)
    ).to(torch.bfloat16)
    flat = first.reshape(-1, 2 * N)
    if limit is None:
        qa, sa = fuse_silu_mul_quant(flat)
    else:
        qa, sa = fuse_silu_mul_clamp_quant(flat, float(limit))
    return (
        torch.bmm(qa.view(E, -1, N).float(), u2.transpose(1, 2))
        * sa.view(E, -1, 1)
        * s2.transpose(1, 2)
    ).to(torch.bfloat16)


x = torch.randn(E, M, K, device="cuda", dtype=torch.bfloat16)
qx, xs = per_token_quant_int8(x.reshape(-1, K))
qx, xs = qx.view(E, M, K), xs.view(E, M, 1)
counts = torch.tensor([M, 3], device="cuda", dtype=torch.int32)
for limit in (None, 7.0):
    method = SimpleNamespace(
        use_deepgemm=True,
        moe_runner_config=SimpleNamespace(
            activation="silu", swiglu_limit=limit, apply_router_weight_on_input=False
        ),
    )
    dispatch = SimpleNamespace(
        hidden_states=qx, hidden_states_scale=xs, masked_m=counts, expected_m=M
    )

    def run():
        return SlimQuantW4A8Int8MoEMethod.apply_deepep_low_latency(
            method, layer, dispatch
        )
    reference = golden(qx, xs, limit)
    def check(actual):
        mask = torch.arange(M, device="cuda")[None, :] < counts[:, None]
        assert_numerics(actual[mask], reference[mask])
    check(run())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for new_counts in ([0, M], [1, 0], [0, 0], [M, M]):
        counts.copy_(torch.tensor(new_counts, device="cuda", dtype=torch.int32))
        graph.replay()
        check(captured)
    print("PASS masked HIPC / independent INT4 golden / CUDA graph, limit", limit, flush=True)

# Reference contiguous HIPC uses 256-aligned per-expert token counts.
M = 256
x = torch.randn(E, M, K, device="cuda", dtype=torch.bfloat16)
qx, xs = per_token_quant_int8(x.reshape(-1, K))
ids = torch.arange(E, device="cuda", dtype=torch.int64).repeat_interleave(M).view(-1, 1)
weights = torch.ones(E * M, 1, device="cuda")
for limit in (None, 7.0):
    layer.quant_method = object()
    layer.moe_runner_config = SimpleNamespace(
        activation="silu", swiglu_limit=limit, apply_router_weight_on_input=False
    )
    layer.forward_deepgemm_w4a8_contiguous = MethodType(
        DeepEPMoE.forward_deepgemm_w4a8_contiguous, layer
    )
    method = SimpleNamespace(use_deepgemm=True, moe_runner_config=layer.moe_runner_config)
    result = SlimQuantW4A8Int8MoEMethod.apply_deepep_normal(
        method, layer, (qx, xs, ids, weights, [M] * E)
    )
    reference = golden(qx.view(E, M, K), xs.view(E, M, 1), limit).view(-1, K)
    assert_numerics(result, reference)
    print("PASS contiguous HIPC / independent INT4 golden, limit", limit, flush=True)
