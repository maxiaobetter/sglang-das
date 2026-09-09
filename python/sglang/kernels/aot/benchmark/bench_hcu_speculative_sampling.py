"""Compare the official HIP MTP sampling chain with a vectorized Torch reference."""

import argparse

import sgl_kernel
import torch
import triton.testing


def torch_filter(probs, top_k, top_p):
    values, ids = probs.sort(dim=-1, descending=True)
    if top_k < probs.shape[-1]:
        values[:, top_k:] = 0
        values = values / values.sum(-1, keepdim=True)
    keep = values.cumsum(-1) - values < top_p
    values = values.masked_fill(~keep, 0)
    values = values / values.sum(-1, keepdim=True)
    return torch.zeros_like(probs).scatter(1, ids, values)


def torch_verify(probs, candidates, coins, final_coins):
    batch, nodes, vocab = probs.shape
    next_ids = candidates[:, 1:, None]
    p = probs[:, :-1].gather(-1, next_ids).squeeze(-1)
    count = (coins[:, :-1] <= p).int().cumprod(-1).sum(-1)
    rows = torch.arange(batch, device=probs.device)
    ranks = torch.arange(nodes, device=probs.device)
    weights = probs[rows, count]
    rejected = candidates.gather(1, (count + 1).clamp_max(nodes - 1)[:, None])
    mask = (count < nodes - 1)[:, None] & (
        torch.arange(vocab, device=probs.device)[None] == rejected
    )
    weights = weights.masked_fill(mask, 0)
    cdf = weights.double().cumsum(-1)
    final = torch.searchsorted(
        cdf, (final_coins.double() * cdf[:, -1])[:, None], right=True
    )
    predicts = torch.full((batch, nodes), -1, dtype=torch.int32, device=probs.device)
    predicts[:, :-1] = torch.where(ranks[:-1] < count[:, None], candidates[:, 1:], -1)
    predicts.scatter_(1, count[:, None], final.int())
    indices = torch.where(
        ranks <= count[:, None], rows[:, None] * nodes + ranks, -1
    ).int()
    return predicts.flatten(), indices, count.int()


def benchmark(batch, nodes, vocab, top_k, top_p):
    device = "cuda"
    torch.manual_seed(42)
    logits = torch.randn(batch * nodes, vocab, device=device)
    candidates = torch.arange(nodes, device=device).expand(batch, -1).contiguous()
    indices = torch.arange(batch * nodes, device=device).reshape(batch, nodes)
    children = torch.arange(1, nodes + 1, device=device).expand(batch, -1).clone()
    children[:, -1] = -1
    siblings = torch.full_like(children, -1)
    coins = torch.rand(batch, nodes, device=device)
    final_coins = torch.rand(batch, device=device)
    predicts = torch.empty(batch * nodes, device=device, dtype=torch.int32)
    accept_index = torch.empty(batch, nodes, device=device, dtype=torch.int32)
    accept_num = torch.empty(batch, device=device, dtype=torch.int32)
    ks = torch.full((batch * nodes,), top_k, device=device, dtype=torch.int32)
    ps = torch.full((batch * nodes,), top_p, device=device)

    def native():
        probs = logits.softmax(-1)
        if top_k < vocab:
            probs = sgl_kernel.top_k_renorm_prob(probs, ks)
        probs = sgl_kernel.top_p_renorm_prob(probs, ps).reshape(batch, nodes, vocab)
        predicts.fill_(-1)
        accept_index.fill_(-1)
        sgl_kernel.tree_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_num,
            candidates,
            indices,
            children,
            siblings,
            coins,
            final_coins,
            probs,
            torch.zeros_like(probs),
        )
        return predicts, accept_index, accept_num

    def reference():
        probs = torch_filter(logits.softmax(-1), top_k, top_p).reshape(
            batch, nodes, vocab
        )
        return torch_verify(probs, candidates, coins, final_coins)

    actual = native()
    expected = reference()
    for result, baseline in zip(actual, expected):
        torch.testing.assert_close(result, baseline, rtol=0, atol=0)
    native_ms = triton.testing.do_bench(native, warmup=50, rep=200)
    torch_ms = triton.testing.do_bench(reference, warmup=50, rep=200)
    print(
        f"batch={batch} nodes={nodes} vocab={vocab} top_k={top_k} top_p={top_p} "
        f"native_ms={native_ms:.4f} torch_ms={torch_ms:.4f} speedup={torch_ms / native_ms:.2f}x",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=154880)
    parser.add_argument("--nodes", type=int, default=6)
    args = parser.parse_args()
    for batch in (1, 2, 8):
        for top_k in (1 << 30, 128):
            benchmark(batch, args.nodes, args.vocab, top_k, 0.95)


if __name__ == "__main__":
    main()
