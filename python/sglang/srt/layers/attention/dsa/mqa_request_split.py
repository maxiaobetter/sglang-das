"""CPU planning for request-local HCU ragged MQA.

Adapted from sglang-model commit 5ba8f17fdd44eca781f260593ca5bd4f72701085.
"""


def plan_mqa_request_slices(q_lens, k_lens, min_saved_cells_per_launch=2_000_000):
    """Return request slices when saved K-prefix work amortizes extra launches.

    The threshold applies to avoided logits cells per additional request-level
    launch, not the size of the full logits matrix. Q chunking for the memory
    budget is applied separately by ``iter_mqa_chunks``.
    """
    if min_saved_cells_per_launch < 0:
        raise ValueError("MQA split threshold must be nonnegative")
    if len(q_lens) != len(k_lens):
        raise ValueError("MQA Q/K request counts differ")
    slices = []
    q_start = k_start = saved_cells = 0
    for q_len, k_len in zip(q_lens, k_lens):
        if q_len < 0 or k_len < 0:
            raise ValueError("MQA request lengths must be nonnegative")
        if q_len:
            slices.append((q_start, q_start + q_len, k_start, k_start + k_len))
            saved_cells += q_len * k_start
        q_start += q_len
        k_start += k_len
    extra_launches = len(slices) - 1
    if extra_launches <= 0 or saved_cells < min_saved_cells_per_launch * extra_launches:
        return ()
    return tuple(slices)


def iter_mqa_chunks(slices, budget_bytes, bytes_per_element=4):
    """Split each request along Q while keeping only that request's K."""
    for q_start, q_end, k_start, k_end in slices:
        # A zero budget is the caller's small-matrix fast path: no chunk needed.
        max_rows = (
            max(1, budget_bytes // max((k_end - k_start) * bytes_per_element, 1))
            if budget_bytes
            else max(1, q_end - q_start)
        )
        for start in range(q_start, q_end, max_rows):
            yield start, min(start + max_rows, q_end), k_start, k_end
