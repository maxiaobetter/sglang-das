import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as w8a8_module,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)


class TestCompressedTensorsPrequantizedInput(unittest.TestCase):
    def test_wrapper_forwards_dynamic_w8a8_capability(self):
        scheme = CompressedTensorsW8A8Int8(
            strategy="channel",
            is_static_input_scheme=False,
            input_symmetric=True,
        )

        method = CompressedTensorsLinearMethod(None, scheme=scheme)

        self.assertTrue(scheme.supports_prequantized_input)
        self.assertTrue(method.supports_prequantized_input)

    def test_wrapper_does_not_enable_incompatible_w8a8_schemes(self):
        for is_static, is_symmetric in ((True, True), (False, False)):
            with self.subTest(
                is_static_input_scheme=is_static,
                input_symmetric=is_symmetric,
            ):
                scheme = CompressedTensorsW8A8Int8(
                    strategy="channel",
                    is_static_input_scheme=is_static,
                    input_symmetric=is_symmetric,
                )

                method = CompressedTensorsLinearMethod(None, scheme=scheme)

                self.assertFalse(scheme.supports_prequantized_input)
                self.assertFalse(method.supports_prequantized_input)

    def test_w8a8_scheme_consumes_prequantized_inputs_without_requantizing(self):
        scheme = CompressedTensorsW8A8Int8(
            strategy="channel",
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        scheme.w8a8_strategy = 1
        layer = SimpleNamespace(
            weight=torch.empty((4, 3), dtype=torch.int8),
            weight_scale=torch.ones(3),
        )
        x = torch.empty((2, 4), dtype=torch.bfloat16)
        x_q = torch.empty_like(x, dtype=torch.int8)
        x_scale = torch.ones((2, 1), dtype=torch.float32)
        expected = object()

        for quant_arg_name in ("input_quant_args", "silu_quant_args"):
            with self.subTest(quant_arg_name=quant_arg_name):
                with (
                    patch.object(
                        w8a8_module, "W8A8_TRITONJSON",
                        SimpleNamespace(triton_json_dict={}),
                    ),
                    patch.object(
                        w8a8_module,
                        "per_token_quant_int8",
                        side_effect=AssertionError("unexpected requantization"),
                    ),
                    patch.object(
                        w8a8_module.ops,
                        "triton_scaled_mm",
                        return_value=expected,
                    ) as scaled_mm,
                ):
                    actual = scheme.apply_weights(
                        layer,
                        x,
                        bias=None,
                        **{quant_arg_name: (x_q, x_scale)},
                    )

                self.assertIs(actual, expected)
                self.assertIs(scaled_mm.call_args.args[0], x_q)
                self.assertIs(scaled_mm.call_args.args[2], x_scale)


if __name__ == "__main__":
    unittest.main()
