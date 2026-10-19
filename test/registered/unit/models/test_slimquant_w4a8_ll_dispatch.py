"""LL dispatch must follow each layer's dtype, not global target flags."""

import unittest
from unittest.mock import Mock, patch
from sglang.srt.layers.moe.token_dispatcher import deepep


class TestLowLatencyDispatch(unittest.TestCase):
    def test_mixed_layer_quantization(self):
        for config, expected in (
            ({"bf16_dispatch": True}, 0),
            ({"int8_dispatch": True}, 1),
        ):
            with self.subTest(config=config):
                dispatcher = deepep._DeepEPDispatcherImplLowLatency.__new__(
                    deepep._DeepEPDispatcherImplLowLatency
                )
                dispatcher.quant_config = config
                dispatcher.use_fp8 = False
                dispatcher.return_recv_hook = False
                dispatcher.num_max_dispatch_tokens_per_rank = 96
                dispatcher.num_experts = 256
                buffer = Mock()
                buffer.low_latency_dispatch.return_value = (
                    "x",
                    "counts",
                    "handle",
                    "event",
                    "hook",
                )
                with (
                    patch.object(dispatcher, "_get_buffer", return_value=buffer),
                    patch.object(deepep, "_deepep_precompile_tp_barrier"),
                    patch.object(deepep, "use_groupgemm", True),
                    patch.object(deepep, "_use_fp8_w8a8_moe", True),
                ):
                    result = dispatcher._dispatch_core("hidden", "ids", "weights")
                self.assertEqual(
                    buffer.low_latency_dispatch.call_args.kwargs["quant_type"], expected
                )
                self.assertEqual(result, ("x", "counts", "event", "hook"))


if __name__ == "__main__":
    unittest.main()
