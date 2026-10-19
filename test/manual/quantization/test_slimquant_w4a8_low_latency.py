"""HCU regression: expert-major W4A8, independent unpack golden and graph replay."""

import torch
import torch.nn.functional as F
from lightop.quant import per_token_quant_int8
from sglang.srt.layers.quantization.slimquant_w4a8 import (
    fused_experts_impl_w4a8_low_latency,
)

torch.manual_seed(20260923)
torch.backends.cuda.matmul.allow_tf32 = False
E, M, K, N = 3, 37, 256, 256
device = "cuda"
x = torch.randn(E, M, K, device=device, dtype=torch.bfloat16)
qx, xs = per_token_quant_int8(x.reshape(-1, K))
qx, xs = qx.view(E, M, K), xs.view(E, M, 1)
w1 = torch.randint(-128, 128, (E, 2 * N, K // 2), device=device, dtype=torch.int8)
w2 = torch.randint(-128, 128, (E, K, N // 2), device=device, dtype=torch.int8)
s1 = torch.rand(E, 2 * N, 1, device=device) * 0.003 + 0.001
s2 = torch.rand(E, K, 1, device=device) * 0.003 + 0.001
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
    )


def unpack(weight):
    # Independently decode two's-complement nibbles to [-8, 7], then apply x16.
    unsigned = weight.to(torch.int32) & 255
    nibbles = torch.stack((unsigned // 16, unsigned % 16), dim=-1)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.flatten(-2).float() * 16


def golden():
    first = torch.bmm(qx.float(), unpack(w1).transpose(1, 2))
    first = (first * xs * s1.transpose(1, 2)).to(torch.bfloat16)
    gate, up = first.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    qa, sa = per_token_quant_int8(activated.reshape(-1, N))
    second = torch.bmm(qa.view(E, M, N).float(), unpack(w2).transpose(1, 2))
    second = (second * sa.view(E, M, 1) * s2.transpose(1, 2)).to(torch.bfloat16)
    mask = torch.arange(M, device=device)[None, :] < counts[:, None]
    return second.masked_fill(~mask[:, :, None], 0)


def check(actual):
    reference = golden()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    print("PASS exact independent INT4 x16 golden; counts", counts.tolist(), flush=True)


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
