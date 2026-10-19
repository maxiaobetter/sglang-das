"""CPU regression for shared workspace bounds, without launching HCU kernels."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers.quantization import slimquant_w4a8 as sq


class TestW4A8ScaleLoading(unittest.TestCase):
    def test_channelwise_scales_are_divided_by_16_after_loading(self):
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(
            torch.ones(2, 8, 2, dtype=torch.int8), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            torch.ones(2, 4, 2, dtype=torch.int8), requires_grad=False
        )
        s1 = torch.arange(1, 17, dtype=torch.float32).view(2, 8, 1)
        s2 = torch.arange(1, 9, dtype=torch.float32).view(2, 4, 1)
        layer.w13_weight_scale = torch.nn.Parameter(s1.clone(), requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(s2.clone(), requires_grad=False)
        method = SimpleNamespace(
            tritonsingleton=SimpleNamespace(
                moe_weight_shapes=[],
                topk=8,
                get_moeint8json_name=Mock(return_value="unused"),
                get_moeint8_triton_cache=Mock(return_value={}),
                triton_moejson_dict={},
            )
        )
        sq.SlimQuantW4A8Int8MoEMethod.process_weights_after_loading(method, layer)
        torch.testing.assert_close(layer.w13_weight_scale, s1 / 16, rtol=0, atol=0)
        torch.testing.assert_close(layer.w2_weight_scale, s2 / 16, rtol=0, atol=0)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.float32)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.float32)
        self.assertFalse(layer.w13_weight_scale.requires_grad)
        self.assertFalse(layer.w2_weight_scale.requires_grad)
        self.assertTrue((layer.w13_weight == 1).all())
        self.assertTrue((layer.w2_weight == 1).all())


class TestW4A8Workspace(unittest.TestCase):
    def run_case(self, tokens, capacity, *, hidden=4, n1=8, limit="32768"):
        topk, experts = 2, 2
        x = torch.ones(tokens, hidden, dtype=torch.int8)
        scales = torch.arange(1, tokens + 1, dtype=torch.float32).view(-1, 1)
        ids = torch.zeros(tokens, topk, dtype=torch.int64)
        weights = torch.ones(tokens, topk)
        workspace = torch.empty(capacity * topk * max(hidden, n1))
        calls = []

        def gemm(a, b, c, a_scale, *args, **kwargs):
            calls.append((tuple(c.shape), a_scale.clone()))
            c.fill_(1)

        with (
            patch.dict(os.environ, {"LMSLIM_FUSED_MOE_CHUNK_SIZE": limit}),
            patch.object(
                sq.w4a8_triton,
                "get_w8a8moe_json",
                return_value=({"BLOCK_SIZE_M": 16}, {"BLOCK_SIZE_M": 32}),
            ),
            patch.object(sq, "moe_align_block_size", return_value=(None, None, None)),
            patch.object(
                sq.w4a8_triton, "invoke_fused_moe_kernel_w4a8", side_effect=gemm
            ),
            patch.object(
                sq,
                "per_token_quant_int8",
                side_effect=lambda a: (a.to(torch.int8), torch.ones(a.shape[0], 1)),
            ),
        ):
            out = sq.fused_experts_impl_w4a8_triton(
                x,
                torch.empty(experts, n1, hidden // 2, dtype=torch.int8),
                torch.empty(experts, hidden, n1 // 4, dtype=torch.int8),
                weights,
                ids,
                workspace,
                activation="silu",
                apply_router_weight_on_input=False,
                global_num_experts=experts,
                expert_map=None,
                w1_scale=torch.ones(experts, n1, 1),
                w2_scale=torch.ones(experts, hidden, 1),
                routed_scaling_factor=1.0,
                shared_output=None,
                input_scale=scales,
                out_dtype=torch.float32,
            )
        self.assertEqual(out.shape, (tokens, hidden))
        self.assertTrue((out == topk).all())
        if tokens:
            torch.testing.assert_close(torch.cat([c[1] for c in calls[::2]]), scales)
        return [c[0][0] for c in calls[::2]]

    def test_reported_21911_tokens_with_16384_capacity(self):
        self.assertEqual(self.run_case(21911, 16384), [16384, 5527])

    def test_second_gemm_is_larger(self):
        self.assertEqual(self.run_case(13, 5, hidden=16, n1=8), [5, 5, 3])

    def test_explicit_smaller_chunk_is_honored(self):
        self.assertEqual(self.run_case(13, 8, limit="4"), [4, 4, 4, 1])

    def test_exact_capacity_and_empty(self):
        self.assertEqual(self.run_case(8, 8), [8])
        self.assertEqual(self.run_case(0, 0), [])

    def test_invalid_capacity_and_chunk_fail_clearly(self):
        for capacity, limit in ((0, "32"), (2, "0")):
            with self.assertRaisesRegex(ValueError, "workspace for at least one token"):
                self.run_case(1, capacity, limit=limit)


if __name__ == "__main__":
    unittest.main()
