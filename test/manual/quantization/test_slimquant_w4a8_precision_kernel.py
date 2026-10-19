"""HCU regression: expert-major W4A8, independent unpack golden and graph replay."""

import torch
import torch.nn.functional as F
from lightop.quant import per_token_quant_int8
from sglang.srt.layers.quantization.slimquant_w4a8 import (
    fused_experts_impl_w4a8_low_latency,
    SlimQuantW4A8Int8Config,
)

torch.manual_seed(20260923)
torch.backends.cuda.matmul.allow_tf32 = False
E, M, K, N = 3, 37, 256, 256
device = "cuda"
x = torch.randn(E, M, K, device=device, dtype=torch.bfloat16)
qx, xs = per_token_quant_int8(x.reshape(-1, K))
qx, xs = qx.view(E, M, K), xs.view(E, M, 1)
raw1 = torch.randint(0, 256, (E, 2 * N, K // 2), device=device, dtype=torch.uint8)
raw2 = torch.randint(0, 256, (E, K, N // 2), device=device, dtype=torch.uint8)
def normalize(raw):
    return torch.stack([SlimQuantW4A8Int8Config.normalize_expert_weight(
        "model.layers.3.mlp.experts.0.gate_proj.qweight", item
    )[1] for item in raw])
w1, w2 = normalize(raw1), normalize(raw2)
# Large gate/up values ensure the clamp is actually exercised.
raw_s1 = torch.rand(E, 2 * N, 1, device=device) * 0.3 + 0.2
raw_s2 = torch.rand(E, K, 1, device=device) * 0.003 + 0.001
s1, s2 = raw_s1 / 16, raw_s2 / 16
counts = torch.tensor([M, 5, 0], device=device, dtype=torch.int32)


def run():
    return fused_experts_impl_w4a8_low_latency(
        qx,
        xs,
        w1,
        w2,
        s1,
        s2,
        counts,
        activation="silu",
        out_dtype=torch.bfloat16,
        swiglu_limit=10.0,
    )


def unpack(weight):
    # Decode the checkpoint independently: signed LOW nibble first, no x16.
    unsigned = weight.to(torch.int32) & 255
    nibbles = torch.stack((unsigned % 16, unsigned // 16), dim=-1)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.flatten(-2).float()


def golden():
    first = torch.bmm(qx.float(), unpack(raw1).transpose(1, 2))
    first = (first * xs * raw_s1.transpose(1, 2)).to(torch.bfloat16)
    gate, up = first.chunk(2, dim=-1)
    activated = F.silu(gate.clamp(max=10)) * up.clamp(-10,10)
    qa, sa = per_token_quant_int8(activated.reshape(-1, N))
    second = torch.bmm(qa.view(E, M, N).float(), unpack(raw2).transpose(1, 2))
    second = (second * sa.view(E, M, 1) * raw_s2.transpose(1, 2)).to(torch.bfloat16)
    mask = torch.arange(M, device=device)[None, :] < counts[:, None]
    return second.masked_fill(~mask[:, :, None], 0)


def check(actual):
    reference = golden()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    print("PASS exact provider decode + scale/16 + clamp10 golden; counts", counts.tolist(), flush=True)


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
for new_counts in ([0, 1, M], [2, 0, 17], [0, 0, 0], [M, M, M]):
    counts.copy_(torch.tensor(new_counts, device=device, dtype=torch.int32))
    graph.replay()
    check(captured)
print(
    "PASS: CUDA graph replay handles changing receive counts and empty experts",
    flush=True,
)
