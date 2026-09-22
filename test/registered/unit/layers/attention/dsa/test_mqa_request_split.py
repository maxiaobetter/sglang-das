"""CPU checks for request-local MQA planning and ragged/paged TopK coordinates.

Run directly; no accelerator runtime is needed.
"""

import importlib.util
import runpy
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
# Load the registry without importing sglang and its accelerator dependencies.
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

SPEC = importlib.util.spec_from_file_location(
    "mqa_request_split",
    ROOT / "python/sglang/srt/layers/attention/dsa/mqa_request_split.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
plan = MODULE.plan_mqa_request_slices
chunks = MODULE.iter_mqa_chunks


class TestMQARequestSplit(unittest.TestCase):
    def test_threshold_and_override(self):
        self.assertEqual(plan([896], [512]), ())
        self.assertEqual(plan([256] * 16, [512] * 16), ())
        self.assertEqual(len(plan([896] * 4, [21760] * 4)), 4)
        self.assertEqual(plan([1, 1], [1999999, 1]), ())
        self.assertEqual(len(plan([1, 1], [2000000, 1])), 2)
        self.assertEqual(len(plan([1, 1], [1, 1], 0)), 2)
        self.assertEqual(plan([896] * 4, [21760] * 4, 10**12), ())

    def test_empty_requests_preserve_offsets(self):
        self.assertEqual(
            plan([0, 3, 0, 2, 0], [5, 7, 9, 4, 8], 0),
            ((0, 3, 5, 12), (3, 5, 21, 25)),
        )
        self.assertEqual(plan([], []), ())
        with self.assertRaises(ValueError):
            plan([1], [], 0)
        with self.assertRaises(ValueError):
            plan([1], [1], -1)

    def test_threshold_is_per_extra_launch_not_full_matrix(self):
        # Full logits exceed two million cells, but K-prefix savings do not.
        self.assertEqual(plan([1000, 1], [1000, 100000]), ())
        # Three requests require two extra launches: savings must reach 4M.
        self.assertEqual(plan([1, 1, 1], [1000000, 1999999, 1]), ())
        self.assertEqual(len(plan([1, 1, 1], [1000000, 2000000, 1])), 3)

    def test_budget_and_zero_budget_fast_path(self):
        slices = ((0, 5, 0, 10), (5, 8, 10, 30))
        self.assertEqual(
            list(chunks(slices, 80)),
            [
                (0, 2, 0, 10),
                (2, 4, 0, 10),
                (4, 5, 0, 10),
                (5, 6, 10, 30),
                (6, 7, 10, 30),
                (7, 8, 10, 30),
            ],
        )
        self.assertEqual(list(chunks(slices, 0)), list(slices))
        self.assertEqual(len(list(chunks(slices, 1))), 8)

    def test_logits_topk_and_cp_rank_local_lengths(self):
        rng = np.random.default_rng(7)
        # Different Q lengths also model the CPU lengths after CP sharding.
        q_lens, k_lens = [3, 2, 4], [9, 6, 12]
        q = rng.normal(size=(sum(q_lens), 3, 8))
        k = rng.normal(size=(sum(k_lens), 8))
        weights = rng.uniform(size=(sum(q_lens), 3))
        global_logits = (
            np.maximum(np.einsum("qhd,kd->qhk", q, k), 0) * weights[..., None]
        ).sum(axis=1)
        slices = plan(q_lens, k_lens, 0)
        page_table = rng.permutation(sum(k_lens)) + 100
        seen = []
        for start, end, k_start, k_end in chunks(slices, 96):
            local = (
                np.maximum(np.einsum("qhd,kd->qhk", q[start:end], k[k_start:k_end]), 0)
                * weights[start:end, :, None]
            ).sum(axis=1)
            np.testing.assert_allclose(local, global_logits[start:end, k_start:k_end])
            request = next(i for i, sl in enumerate(slices) if sl[0] <= start < sl[1])
            q_base = slices[request][0]
            for row in range(start, end):
                length = k_lens[request] - q_lens[request] + row - q_base + 1
                expected = np.argsort(-global_logits[row, k_start : k_start + length])[
                    :4
                ]
                actual = np.argsort(-local[row - start, :length])[:4]
                # Unfused output, ragged global offsets, and paged physical slots.
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual + k_start, expected + k_start)
                np.testing.assert_array_equal(
                    page_table[k_start + actual], page_table[k_start + expected]
                )
                seen.append(row)
        self.assertEqual(seen, list(range(sum(q_lens))))


if __name__ == "__main__":
    unittest.main()
