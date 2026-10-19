"""HCU regression for the 21,911-token prefill workspace overflow."""

import os
from unittest.mock import patch

import torch
from lightop.quant import per_token_quant_int8
from sglang.srt.layers.quantization.slimquant_w4a8 import fused_experts_impl_w4a8_triton


torch.manual_seed(42)
tokens, experts, hidden, intermediate, topk = 21911, 8, 256, 256, 8
x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
x *= ((torch.arange(tokens, device="cuda") % 31 + 1) * 0.1)[:, None]
qx, scale = per_token_quant_int8(x)
w1 = torch.randint(
    -128, 128, (experts, 2 * intermediate, hidden // 2), device="cuda", dtype=torch.int8
)
w2 = torch.randint(
    -128, 128, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.int8
)
ids = torch.arange(experts, device="cuda").expand(tokens, -1).clone()
ids[::3, 1::2] = -1
ids[::17] = -1
weights = torch.rand(tokens, topk, device="cuda")
kwargs = dict(
    activation="silu",
    apply_router_weight_on_input=False,
    global_num_experts=experts,
    expert_map=None,
    w1_scale=torch.full((experts, 2 * intermediate, 1), 0.001, device="cuda"),
    w2_scale=torch.full((experts, hidden, 1), 0.001, device="cuda"),
    routed_scaling_factor=1.0,
    shared_output=None,
    input_scale=scale,
    out_dtype=torch.bfloat16,
)


def run(capacity):
    cache = torch.full(
        (capacity * topk * max(2 * intermediate, hidden),),
        float("nan"),
        device="cuda",
        dtype=torch.bfloat16,
    )
    return fused_experts_impl_w4a8_triton(qx, w1, w2, weights, ids, cache, **kwargs)


with patch.dict(os.environ, {"LMSLIM_FUSED_MOE_CHUNK_SIZE": "32768"}):
    reference = run(32768)
    actual = run(16384)
torch.cuda.synchronize()
assert torch.isfinite(actual).all()
assert (actual[::17] == 0).all()
torch.testing.assert_close(actual, reference, rtol=0, atol=0)
print(
    "PASS: 21911 tokens, topk=8, workspace=16384; chunked output matches unchunked exactly",
    flush=True,
)
