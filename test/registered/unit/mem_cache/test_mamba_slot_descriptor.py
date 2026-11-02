"""GPU contracts for pool-stable fused Mamba slot descriptors."""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd-mi35x")


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestMambaSlotDescriptor(unittest.TestCase):
    def test_creation_does_not_wait_for_prior_gpu_work(self):
        from sglang.srt.mem_cache.mamba_slot_fused import build_conv_slot_descriptor

        tensor = torch.empty(2, 8, 3, 128, device="cuda", dtype=torch.bfloat16)
        # Warm the context and pinned allocator before testing synchronization.
        for _ in range(3):
            build_conv_slot_descriptor([tensor])
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        prior_work_done = torch.cuda.Event()
        try:
            with torch.cuda.stream(stream):
                torch.cuda._sleep(1_000_000_000)
                prior_work_done.record()
                descriptor = build_conv_slot_descriptor([tensor])
                waited = prior_work_done.query()
        finally:
            stream.synchronize()

        self.assertFalse(waited, "descriptor creation synchronized the serving stream")
        self.assertEqual(descriptor.ptr.item(), tensor.data_ptr())

    def test_clear_and_copy_preserve_slot_contents(self):
        from sglang.srt.mem_cache.mamba_slot_fused import (
            build_conv_slot_descriptor,
            fused_clear_conv_slots,
            fused_copy_conv_slots,
        )

        tensors = [
            torch.randn(2, 8, *shape, device="cuda", dtype=torch.bfloat16)
            for shape in ((3, 17), (2, 64), (4, 32))
        ]
        expected = [tensor.clone() for tensor in tensors]
        descriptor = build_conv_slot_descriptor(tensors)
        indices = torch.tensor([1, 5], device="cuda")
        fused_clear_conv_slots(descriptor, indices)
        for tensor in expected:
            tensor[:, indices] = 0
        source = torch.tensor([0, 2], device="cuda")
        destination = torch.tensor([3, 6], device="cuda")
        fused_copy_conv_slots(descriptor, source, destination)
        for tensor in expected:
            tensor[:, destination] = tensor[:, source]
        torch.cuda.synchronize()
        for actual, reference in zip(tensors, expected):
            self.assertTrue(torch.equal(actual, reference))


if __name__ == "__main__":
    unittest.main()
