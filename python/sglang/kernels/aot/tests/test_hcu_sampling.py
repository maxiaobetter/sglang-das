"""HIP coverage for the public renorm and target-only tree sampling APIs."""

import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires the HIP sgl-kernel build",
)


def _torch_top_k(probs, top_ks):
    values = probs.sort(dim=-1, descending=True).values
    ks = torch.as_tensor(top_ks, device=probs.device).expand(probs.shape[0])
    cutoff = values.gather(1, (ks.clamp(1, probs.shape[1]) - 1)[:, None])
    filtered = probs.masked_fill(probs < cutoff, 0)
    return filtered / filtered.sum(-1, keepdim=True)


def _torch_top_p(probs, top_ps):
    values, ids = probs.sort(dim=-1, descending=True)
    ps = torch.as_tensor(top_ps, device=probs.device).expand(probs.shape[0])
    keep = values.double().cumsum(-1) - values.double() < ps[:, None]
    # p=1 disables filtering even when FP32 normalization sums slightly above 1.
    keep |= ps[:, None] >= 1
    keep[:, 0] = True
    # The public renorm operators retain all ties at the cutoff probability.
    cutoff = values.masked_fill(~keep, float("inf")).min(-1).values
    filtered = probs.masked_fill(probs < cutoff[:, None], 0)
    return filtered / filtered.sum(-1, keepdim=True)


@pytest.mark.parametrize("batch", [1, 2, 6, 12])
@pytest.mark.parametrize("vocab", [8, 257, 154880])
@pytest.mark.parametrize("per_row", [False, True])
def test_renorm_matches_torch(batch, vocab, per_row):
    torch.manual_seed(2026)
    probs = torch.randn(batch, vocab, device="cuda").softmax(-1)
    if per_row:
        ks = torch.tensor(
            ([1, 2, 8, 32, 128, 1 << 30] * 2)[:batch],
            dtype=torch.int32,
            device="cuda",
        )
        ps = torch.tensor(([0.05, 0.2, 0.5, 0.8, 0.95, 1.0] * 2)[:batch], device="cuda")
    else:
        ks, ps = min(vocab, 100), 0.95
    actual_k = sgl_kernel.top_k_renorm_prob(probs, ks)
    expected_k = _torch_top_k(probs, ks)
    actual = sgl_kernel.top_p_renorm_prob(actual_k, ps)
    expected = _torch_top_p(expected_k, ps)
    assert torch.equal(actual_k > 0, expected_k > 0)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual_k, expected_k, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(actual.sum(-1), torch.ones(batch, device="cuda"))


@pytest.mark.parametrize("temperature", [0.1, 0.6, 1.0, 2.0])
@pytest.mark.parametrize("top_p", [0.05, 0.5, 0.95, 1.0])
def test_unbounded_top_k_matches_torch(temperature, top_p):
    # Default MTP: top-k disabled, six verify rows for each of two requests.
    torch.manual_seed(13)
    logits = torch.randn(12, 154880, device="cuda") / temperature
    logits[:, ::7] = float("-inf")
    probs = logits.softmax(-1)
    expected = _torch_top_p(probs, top_p)
    actual = sgl_kernel.top_p_renorm_prob(probs, top_p)
    assert torch.equal(actual > 0, expected > 0)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)


def test_renorm_top_k_first_and_boundary():
    # The old joint filter kept rank 3 for this case; top-k-first keeps 0..2.
    logits = torch.tensor(
        [[0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5]], device="cuda"
    )
    actual = sgl_kernel.top_p_renorm_prob(
        sgl_kernel.top_k_renorm_prob(logits.softmax(-1), 4), 0.6
    )
    logits[:, 4:] = float("-inf")
    expected = logits.softmax(-1)
    expected[:, 3:] = 0
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)
    exact = torch.tensor([[0.5, 0.25, 0.125, 0.125]], device="cuda")
    torch.testing.assert_close(
        sgl_kernel.top_p_renorm_prob(exact, 0.5),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"),
    )
    # Ties follow the public FlashInfer/sgl-kernel probability threshold rule.
    equal = torch.full((2, 8), 0.125, device="cuda")
    torch.testing.assert_close(sgl_kernel.top_k_renorm_prob(equal, 2), equal)
    torch.testing.assert_close(sgl_kernel.top_p_renorm_prob(equal, 0.5), equal)


def _tree_reference(
    candidates,
    indices,
    children,
    siblings,
    coins,
    final_coins,
    probs,
    steps,
    thresholds,
):
    """CPU oracle following the upstream target-only tree acceptance algorithm."""
    batch, nodes = candidates.shape
    predicts = torch.full((batch * nodes,), -1, dtype=torch.int32)
    accepted_indices = torch.full((batch, steps), -1, dtype=torch.int32)
    lengths = torch.zeros(batch, dtype=torch.int32)
    draft = torch.zeros_like(probs)
    for row in range(batch):
        current = probability_node = count = 0
        last = int(indices[row, 0])
        accepted_indices[row, 0] = last
        coin = coins[row, 0]
        cumulative = torch.tensor(0.0)
        for _ in range(1, steps):
            current = int(children[row, current])
            while current != -1:
                token = int(candidates[row, current])
                probability = probs[row, probability_node, token]
                cumulative += probability
                if (
                    coin <= cumulative / max(thresholds[1], 1e-9)
                    or probability >= thresholds[0]
                ):
                    predicts[last] = token
                    count += 1
                    last = int(indices[row, current])
                    accepted_indices[row, count] = last
                    probability_node = current
                    coin = coins[row, current]
                    cumulative = torch.tensor(0.0)
                    break
                draft[row, probability_node, token] = probability
                current = int(siblings[row, current])
            if current == -1:
                break
        lengths[row] = count
        weights = probs[row, probability_node].clone()
        if count != steps - 1:
            weights = (weights - draft[row, probability_node]).clamp_min(0)
        cdf = weights.double().cumsum(0)
        token = int(
            torch.searchsorted(cdf, final_coins[row].double() * cdf[-1], right=True)
        )
        if token >= probs.shape[-1]:
            positive = torch.where(weights > 0)[0]
            token = int(positive[-1]) if positive.numel() else probs.shape[-1] - 1
        predicts[last] = token
    return predicts, accepted_indices, lengths, draft


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("vocab", [20, 257, 154880])
@pytest.mark.parametrize("thresholds", [(1.0, 1.0), (0.0, 0.0), (0.7, 0.5)])
def test_tree_sampling_matches_upstream_torch_oracle(batch, vocab, thresholds):
    torch.manual_seed(57)
    nodes, steps = 6, 4
    candidates = torch.arange(nodes).expand(batch, -1).contiguous()
    indices = torch.arange(batch * nodes).reshape(batch, nodes)
    children = torch.tensor([[1, 2, -1, 4, 5, -1]]).expand(batch, -1).contiguous()
    siblings = torch.tensor([[-1, 3, -1, -1, -1, -1]]).expand(batch, -1).contiguous()
    probs = torch.zeros(batch, nodes, vocab)
    for node in range(nodes):
        probs[:, node, (node + 1) % nodes] = 0.5
        probs[:, node, (node + 3) % nodes] = 0.25
        probs[:, node, vocab - 1] = 0.25
    coins = torch.rand(batch, nodes)
    final_coins = torch.tensor(([0.0, 1.0, 0.25, 0.95] * 2)[:batch])
    expected = _tree_reference(
        candidates,
        indices,
        children,
        siblings,
        coins,
        final_coins,
        probs,
        steps,
        thresholds,
    )
    gpu = [
        x.cuda()
        for x in (candidates, indices, children, siblings, coins, final_coins, probs)
    ]
    predicts = torch.full((batch * nodes,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((batch, steps), -1, dtype=torch.int32, device="cuda")
    accept_num = torch.zeros(batch, dtype=torch.int32, device="cuda")
    draft = torch.zeros_like(gpu[-1])
    sgl_kernel.tree_speculative_sampling_target_only(
        predicts, accept_index, accept_num, *gpu, draft, *thresholds
    )
    for actual, reference in zip((predicts, accept_index, accept_num, draft), expected):
        torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [257, 154880])
def test_official_sampling_chain_graph_replay(vocab):
    # Exercise the public renorm -> tree API, including graph capture/replay.
    batch, nodes = 2, 6
    torch.manual_seed(21)
    logits = torch.randn(batch * nodes, vocab, device="cuda")
    candidates = torch.arange(nodes, device="cuda").expand(batch, -1).contiguous()
    indices = torch.arange(batch * nodes, device="cuda").reshape(batch, nodes)
    children = (
        torch.tensor([[1, 2, 3, 4, 5, -1]], device="cuda")
        .expand(batch, -1)
        .contiguous()
    )
    siblings = torch.full_like(children, -1)
    ks = torch.full((batch * nodes,), 32, device="cuda", dtype=torch.int32)
    ps = torch.full((batch * nodes,), 0.95, device="cuda")
    coins = torch.rand(batch, nodes, device="cuda")
    final_coins = torch.rand(batch, device="cuda")
    predicts = torch.full((batch * nodes,), -1, dtype=torch.int32, device="cuda")
    accept_index = torch.full((batch, nodes), -1, dtype=torch.int32, device="cuda")
    accept_num = torch.zeros(batch, dtype=torch.int32, device="cuda")

    def run():
        predicts.fill_(-1)
        accept_index.fill_(-1)
        probs = sgl_kernel.top_p_renorm_prob(
            sgl_kernel.top_k_renorm_prob(logits.softmax(-1), ks), ps
        )
        target = probs.reshape(batch, nodes, vocab)
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
            target,
            torch.zeros_like(target),
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    expected_probs = (
        _torch_top_p(_torch_top_k(logits.softmax(-1), ks), ps)
        .reshape(batch, nodes, vocab)
        .cpu()
    )
    for _ in range(3):
        coins.uniform_()
        final_coins.uniform_()
        graph.replay()
        expected = _tree_reference(
            candidates.cpu(),
            indices.cpu(),
            children.cpu(),
            siblings.cpu(),
            coins.cpu(),
            final_coins.cpu(),
            expected_probs,
            nodes,
            (1.0, 1.0),
        )
        for actual, reference in zip((predicts, accept_index, accept_num), expected):
            torch.testing.assert_close(actual.cpu(), reference, rtol=0, atol=0)
