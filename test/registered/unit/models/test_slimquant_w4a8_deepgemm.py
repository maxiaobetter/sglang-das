"""Regression checks for normalized INT4 packing and DeepGEMM selection."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.quantization import slimquant_w4a8 as sq
from sglang.srt.layers.quantization.hcu_deepgemm_w4a8_utils import (
    prepare_slimquant_w4a8_deepgemm_weights,
)


class TestSlimQuantW4A8DeepGEMM(unittest.TestCase):
    def test_backend_requires_hcu_deepep_and_environment_flag(self):
        for hcu, enabled, deepep, expected in (
            (True, True, True, True),
            (False, True, True, False),
            (True, False, True, False),
            (True, True, False, False),
        ):
            method = SimpleNamespace()
            with (
                patch.object(sq, "is_hcu", return_value=hcu),
                patch.object(sq, "get_bool_env_var", return_value=enabled),
                patch.object(sq, "W8a8GetCacheJSON"),
                patch.object(
                    sq, "get_moe_a2a_backend",
                    return_value=SimpleNamespace(is_deepep=lambda: deepep),
                ),
            ):
                sq.SlimQuantW4A8Int8MoEMethod.__init__(method, object())
            self.assertEqual(method.use_deepgemm, expected)

    def test_contiguous_rejects_incomplete_output_tiles(self):
        from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE

        layer = SimpleNamespace(
            quant_method=object(),
            moe_runner_config=SimpleNamespace(activation="silu"),
            w13_weight_scale=torch.ones(2, 512, 1),
        )
        dispatch = (
            torch.zeros(512, 256, dtype=torch.int8),
            torch.ones(512, 1), None, None, [256, 256],
        )
        with self.assertRaisesRegex(ValueError, "divisible by 512"):
            DeepEPMoE.forward_deepgemm_w4a8_contiguous(layer, dispatch)

    def test_provider_and_compressed_tensors_are_not_swapped_twice(self):
        # Both on-disk encodings represent signed INT4 [1, -2, 3, -4].
        for suffix, raw in (
            ("qweight", torch.tensor([[0xE1, 0xC3]], dtype=torch.uint8)),
            ("weight_packed", torch.tensor([[0x44444B69]], dtype=torch.int32)),
        ):
            with self.subTest(format=suffix):
                _, normalized = sq.SlimQuantW4A8Int8Config.normalize_expert_weight(
                    "model.layers.0.mlp.experts.0.gate_proj." + suffix, raw
                )
                self.assertEqual(
                    normalized.view(torch.uint8).flatten()[:2].tolist(), [0x1E, 0x3C]
                )
                layer = torch.nn.Module()
                for name in ("w13_weight", "w2_weight"):
                    layer.register_parameter(
                        name,
                        torch.nn.Parameter(
                            normalized.unsqueeze(0).clone(), requires_grad=False
                        ),
                    )
                for name in ("w13_weight_scale", "w2_weight_scale"):
                    layer.register_parameter(
                        name,
                        torch.nn.Parameter(torch.tensor([[[0.5]]]), requires_grad=False),
                    )
                pack = Mock(side_effect=lambda x: x.clone())
                with patch.dict(
                    sys.modules,
                    {"deepgemm": SimpleNamespace(pack_w4a8_moe_hipc_weight=pack)},
                ):
                    prepare_slimquant_w4a8_deepgemm_weights(layer)
                    prepare_slimquant_w4a8_deepgemm_weights(layer)
                self.assertEqual(pack.call_count, 2)
                torch.testing.assert_close(
                    pack.call_args_list[0].args[0], normalized.unsqueeze(0)
                )
                torch.testing.assert_close(
                    layer.w13_weight_scale, torch.tensor([[[0.5 / 16]]])
                )
                self.assertFalse(hasattr(layer, "w13_weight"))
                self.assertFalse(hasattr(layer, "w2_weight"))
                self.assertIn("w13_weight_deepgemm", dict(layer.named_buffers()))
                self.assertNotIn("w13_weight_deepgemm", layer.state_dict())


if __name__ == "__main__":
    unittest.main()
