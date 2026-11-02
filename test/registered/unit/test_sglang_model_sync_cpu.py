"""Portable behavior checks for the 2026-10-28 source-model sync.

Load production definitions without importing the serving package: these tests
exercise CPU orchestration with real torch tensors, without Triton/HCU imports.
They do not replace the serving-package or GPU integration tests.
"""

import ast
import contextlib
import io
import itertools
import os
import random
import sys
import threading
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def load_definition(relative, name, namespace, methods=None):
    tree = ast.parse((ROOT / "python/sglang" / relative).read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    if isinstance(node, ast.ClassDef):
        node.bases = []
        node.decorator_list = []
        node.body = [
            n for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (methods is None or n.name in methods)
        ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(relative), "exec"), namespace)
    return namespace[name]


class TestModelSyncCPU(unittest.TestCase):
    def quant_definitions(self):
        ns = {"torch": torch, "os": os, "bool": bool}
        wrapper = load_definition(
            "srt/layers/quantization/compressed_tensors/compressed_tensors.py",
            "CompressedTensorsLinearMethod", ns, {"__init__"},
        )
        config = load_definition(
            "srt/layers/quantization/compressed_tensors/compressed_tensors.py",
            "CompressedTensorsConfig", ns, {"get_linear_method"},
        )
        scheme = load_definition(
            "srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py",
            "CompressedTensorsW8A8Int8", ns, {"__init__", "apply_weights"},
        )
        return ns, wrapper, config, scheme

    def test_generic_linear_factory_stays_conservative(self):
        _, _, config, _ = self.quant_definitions()
        method = config().get_linear_method()
        self.assertFalse(method.supports_prequantized_input)

    def test_only_dynamic_symmetric_scheme_advertises_int8_input(self):
        _, wrapper, _, scheme = self.quant_definitions()
        for static, symmetric in itertools.product((False, True), repeat=2):
            with self.subTest(static=static, symmetric=symmetric):
                value = scheme("channel", static, symmetric)
                self.assertEqual(
                    wrapper(None, scheme=value).supports_prequantized_input,
                    not static and symmetric,
                )
        self.assertFalse(wrapper(None, scheme=object()).supports_prequantized_input)

    def test_fused_activations_are_consumed_without_requantization(self):
        ns, _, _, scheme = self.quant_definitions()
        quantize = Mock(side_effect=AssertionError("fused input was quantized twice"))
        gemm = Mock(return_value=object())
        ns.update(per_token_quant_int8=quantize,
                  W8A8_TRITONJSON=SimpleNamespace(triton_json_dict={}),
                  ops=SimpleNamespace(triton_scaled_mm=gemm))
        value = scheme("channel", False, True)
        value.w8a8_strategy = 1
        x = torch.randn(2, 4)
        q = torch.zeros(2, 4, dtype=torch.int8)
        scale = torch.ones(2, 1)
        layer = SimpleNamespace(weight=torch.zeros(4, 3, dtype=torch.int8),
                                weight_scale=torch.ones(3))
        for arg in ("input_quant_args", "silu_quant_args"):
            with self.subTest(arg=arg):
                value.apply_weights(layer, x, None, **{arg: (q, scale)})
                self.assertIs(gemm.call_args.args[0], q)
                self.assertIs(gemm.call_args.args[2], scale)
        quantize.assert_not_called()

    def test_hicache_reports_mamba_occupancy_and_clears_missing_pool(self):
        ns = {}
        reporter = load_definition(
            "srt/managers/scheduler_components/metrics_reporter.py",
            "SchedulerMetricsReporter", ns, {"_log_hicache_stats"},
        )()
        tree = SimpleNamespace(
            token_to_kv_pool_host=SimpleNamespace(logical_size=100, available_size=lambda: 60),
            mamba_pool_host=SimpleNamespace(size=20, available_size=lambda: 7),
        )
        reporter.scheduler = SimpleNamespace(enable_hierarchical_cache=True, tree_cache=tree)
        reporter.stats = SimpleNamespace()
        reporter._log_hicache_stats()
        self.assertEqual(reporter.stats.hicache_host_used_tokens, 40)
        self.assertEqual(reporter.stats.hicache_host_mamba_used_slots, 13)
        self.assertEqual(reporter.stats.hicache_host_mamba_total_slots, 20)
        del tree.mamba_pool_host
        reporter._log_hicache_stats()
        self.assertEqual(reporter.stats.hicache_host_mamba_used_slots, 0)
        self.assertEqual(reporter.stats.hicache_host_mamba_total_slots, 0)

    def test_async_staging_has_cpu_fallback_and_preserves_dtype(self):
        fn = load_definition(
            "srt/disaggregation/decode_schedule_batch_mixin.py", "_to_device_async",
            {"torch": torch, "is_pin_memory_available": lambda device: False},
        )
        self.assertTrue(torch.equal(fn([3, 7], torch.int64, "cpu"), torch.tensor([3, 7])))
        self.assertEqual(fn([], torch.int32, "cpu").dtype, torch.int32)

    def test_failed_peer_socket_waits_for_live_send(self):
        ns = {"threading": threading}
        manager = load_definition(
            "srt/disaggregation/common/conn.py", "CommonKVManager", ns,
            {"_send_multipart_locked", "_disconnect_manager_endpoint"},
        )()
        entered, release = threading.Event(), threading.Event()
        socket = Mock()
        def send(parts):
            entered.set()
            if not release.wait(2):
                raise AssertionError("send did not drain")
        socket.send_multipart.side_effect = send
        monitor = Mock()
        manager._socket_lock = threading.Lock()
        manager._socket_send_locks = {}
        manager._socket_cache = {"peer": socket}
        manager._monitor_cache = {"peer": monitor}
        manager._connect = lambda endpoint, **kw: manager._socket_cache[endpoint]
        sender = threading.Thread(target=manager._send_multipart_locked,
                                  args=("peer", [b"data"]))
        sender.start()
        self.assertTrue(entered.wait(2))
        closer = threading.Thread(target=manager._disconnect_manager_endpoint, args=("peer",))
        closer.start()
        try:
            # The sender owns the endpoint lock; cleanup may start but cannot
            # close its socket until that send releases it.
            self.assertTrue(manager._socket_send_locks["peer"].locked())
            socket.close.assert_not_called()
        finally:
            release.set()
            sender.join(2)
            closer.join(2)
        self.assertFalse(sender.is_alive())
        self.assertFalse(closer.is_alive())
        socket.close.assert_called_once_with(linger=0)
        monitor.close.assert_called_once_with()
        self.assertNotIn("peer", manager._socket_cache)
        manager._disconnect_manager_endpoint("peer")
        socket.close.assert_called_once()

    def test_tilelang_precompile_uses_capture_sizes_and_slot_pitches(self):
        compile_kernel = Mock()
        runner = object()
        def capture_sizes(model_runner, captured_req_width):
            self.assertIs(model_runner, runner)
            self.assertEqual(captured_req_width, 5)
            return [1, 3, 8], []
        module_name = "sglang.kernels.ops.attention.fla.hcu.kda_tilelang_verify"
        graph_name = "sglang.srt.model_executor.runner.base_cuda_graph_runner"
        ns = {"logger": Mock(), "envs": SimpleNamespace(
            SGLANG_KDA_USE_TILELANG_VERIFY=SimpleNamespace(get=lambda: True))}
        backend = load_definition(
            "srt/layers/attention/hybrid_linear_attn_backend.py",
            "MambaAttnBackendBase", ns, {"_precompile_tilelang_verify"},
        )()
        cache = SimpleNamespace(
            temporal=torch.empty(2, 4, 8, 16, 16),
            replayssm_g=torch.empty(2, 4, 8, 8, 16),
            replayssm_rawv=torch.empty(2, 4, 8, 8, 16),
            replayssm_rawk=torch.empty(2, 4, 8, 8, 16),
            replayssm_beta=torch.empty(2, 4, 8, 8),
        )
        backend.req_to_token_pool = SimpleNamespace(mamba_pool=SimpleNamespace(
            use_hcu_kda_replayssm=True, mamba_cache=cache))
        backend._model_runner = runner
        with patch.dict(sys.modules, {
            module_name: SimpleNamespace(precompile=compile_kernel),
            graph_name: SimpleNamespace(get_batch_sizes_to_capture=capture_sizes),
        }):
            backend._precompile_tilelang_verify(4, 5)
        compile_kernel.assert_called_once()
        self.assertEqual(compile_kernel.call_args.kwargs["n_list"], [1, 3])
        self.assertEqual(compile_kernel.call_args.kwargs["h0_stride"], cache.temporal[0].stride(0))
        ns["logger"].warning.assert_not_called()

    def generated_rows(self, **options):
        prompts = itertools.count()
        ns = dict(itertools=itertools, random=random, uuid=uuid, datetime=datetime, np=np,
                  DatasetRow=SimpleNamespace, tqdm=lambda values, **kw: values,
                  get_gen_prefix_cache_path=lambda *a, **kw: Path("/unused"),
                  compute_random_lens=lambda full_len, range_ratio, num: [full_len] * num,
                  gen_prompt=lambda *a: str(next(prompts)))
        load_definition("benchmark/datasets/generated_shared_prefix.py", "_zipf_group_probs", ns)
        fn = load_definition("benchmark/datasets/generated_shared_prefix.py",
                             "sample_generated_shared_prefix_requests", ns)
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(3, 3, 4, 2, 7, 0, None, 17,
                      send_routing_key=True, fast_prepare=True, **options)

    def test_default_uniform_rows_cover_all_groups_in_each_wave(self):
        rows = self.generated_rows()
        groups = [r.routing_key.rsplit("_", 1)[-1] for r in rows]
        self.assertEqual(groups, ["0", "1", "2"] * 3)
        for group in range(3):
            self.assertEqual(len({r.prompt for r in rows[group::3]}), 3)

    def test_target_ordered_and_zipf_modes_remain_available(self):
        rows = self.generated_rows(ordered=True)
        self.assertEqual([r.routing_key.rsplit("_", 1)[-1] for r in rows],
                         ["0"] * 3 + ["1"] * 3 + ["2"] * 3)
        rows = self.generated_rows(group_distribution="zipf", zipf_alpha=1.5)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(r.output_len == 7 for r in rows))


if __name__ == "__main__":
    unittest.main()
