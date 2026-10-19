"""Compare prequantized DeepEP input against the original W4A8 entry path."""

import torch
from lightop.quant import per_token_quant_int8

from sglang.srt.layers.quantization.slimquant_w4a8 import (
    fused_experts_impl_w4a8_triton,
)

torch.manual_seed(42)
device = "cuda"
tokens, experts, hidden, intermediate, topk = 9, 2, 256, 256, 2
x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device)
w1 = torch.randint(-128, 128, (experts, 2 * intermediate, hidden // 2),
                   dtype=torch.int8, device=device)
w2 = torch.randint(-128, 128, (experts, hidden, intermediate // 2),
                   dtype=torch.int8, device=device)
ids = torch.tensor([[0, 1], [1, -1], [-1, -1]] * 3, device=device)
weights = torch.rand(tokens, topk, dtype=torch.float32, device=device)
kwargs = dict(
    activation="silu", apply_router_weight_on_input=False,
    global_num_experts=experts, expert_map=None,
    w1_scale=torch.full((experts, 2 * intermediate, 1), 0.01, device=device),
    w2_scale=torch.full((experts, hidden, 1), 0.01, device=device),
    routed_scaling_factor=1.0, shared_output=None,
)

def workspace():
    # NaNs make accidental reads of untouched/invalid routing slots visible.
    return torch.full((tokens * topk * max(2 * intermediate, hidden),),
                      float("nan"), dtype=torch.bfloat16, device=device)

# Independent equivalent routing: send absent routes to expert 0 at zero weight.
reference = fused_experts_impl_w4a8_triton(
    x, w1, w2, weights.masked_fill(ids < 0, 0), ids.clamp_min(0), workspace(), **kwargs
)
qx, scale = per_token_quant_int8(x)
actual = fused_experts_impl_w4a8_triton(
    qx, w1, w2, weights, ids, workspace(), input_scale=scale,
    out_dtype=torch.bfloat16, **kwargs,
)
torch.cuda.synchronize()
assert actual.dtype == torch.bfloat16
assert torch.isfinite(actual).all()
torch.testing.assert_close(actual, reference, rtol=0, atol=0)
assert (actual[2::3] == 0).all(), "Invalid routes must contribute zero"
empty = fused_experts_impl_w4a8_triton(
    qx[:0], w1, w2, weights[:0], ids[:0], workspace(), input_scale=scale[:0],
    out_dtype=torch.bfloat16, **kwargs,
)
assert empty.shape == (0, hidden) and empty.dtype == torch.bfloat16
print("PASS: HCU W4A8 prequantized path equals original path exactly; invalid and empty routes pass")
