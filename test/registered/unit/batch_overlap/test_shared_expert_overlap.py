"""CPU regression tests for DeepEP dispatch layouts used by shared-expert SBO."""

import ast
import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
register_cpu_ci = runpy.run_path(str(ROOT / "python/sglang/test/ci/ci_register.py"))[
    "register_cpu_ci"
]
register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class TestSharedExpertOverlap(unittest.TestCase):
    def setUp(self):
        # Exercise the production function without importing accelerator libraries.
        source = ROOT / "python/sglang/srt/batch_overlap/single_batch_overlap.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "compute_overlap_args"
        )
        self.flags = SimpleNamespace(
            enable_combine_down_gemm_two_stream_overlap=Mock(return_value=True),
            enable_combine_shared_two_stream_overlap=Mock(return_value=False),
        )
        self.cuda = Mock()
        self.cuda.get_device_properties.return_value.multi_processor_count = 64
        self.zeros = Mock(return_value=object())
        namespace = dict(
            SboFlags=self.flags,
            torch=SimpleNamespace(cuda=self.cuda, zeros=self.zeros, int32="int32"),
            envs=SimpleNamespace(
                SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS=SimpleNamespace(
                    is_set=lambda: False
                )
            ),
            is_blackwell=lambda: False,
            CombineOverlapArgs=SimpleNamespace,
            DownGemmOverlapArgs=SimpleNamespace,
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self.compute = namespace["compute_overlap_args"]

    def test_normal_dispatch_skips_incompatible_combine_overlap(self):
        for shape in [(32, 7168), (0, 7168)]:
            with self.subTest(shape=shape):
                output = SimpleNamespace(
                    hidden_states=SimpleNamespace(shape=shape, dim=lambda: 2)
                )
                self.assertEqual(self.compute(output, None), (None, None, {}))
        self.cuda.get_device_properties.assert_not_called()
        self.cuda.Event.assert_not_called()
        self.zeros.assert_not_called()

    def test_expert_major_dispatch_preserves_overlap(self):
        output = SimpleNamespace(
            hidden_states=SimpleNamespace(
                shape=(4, 65, 7168), dim=lambda: 3, device="cuda:0"
            )
        )
        stream = object()
        combine, down, meta = self.compute(output, stream)
        self.assertTrue(combine.overlap)
        self.assertIs(combine.stream, stream)
        self.assertIs(combine.signal, down.signal)
        self.assertEqual(meta["compute_num_sms"], 61)
        self.assertEqual(combine.num_sms, 3)
        self.zeros.assert_called_once_with(8, dtype="int32", device="cuda:0")

    def test_disabled_overlap_does_not_access_dispatch_output(self):
        self.flags.enable_combine_down_gemm_two_stream_overlap.return_value = False
        self.assertEqual(self.compute(None, None), (None, None, {}))
        self.cuda.get_device_properties.assert_not_called()


if __name__ == "__main__":
    unittest.main()
