"""Tests for the HCU wiring of the FlashKDA KDA prefill backend.

Covers the contract ``KDAAttnBackend`` relies on when
``--linear-attn-prefill-backend flashkda`` is selected on HCU
(``FlashKDAKernel(fallback_kernel=HcuKDAKernel())``):

- Fallback routing is correctness-only: with the backend switch on, EVERY
  extend length runs on the fused kernel (no <64 / >2048 length routing).
  Lengths here deliberately span the old routing boundaries (1, 63, 64,
  2048, 2049, ...).
- Outputs and updated SSM states of the fused kernel match the production
  HCU Triton ``chunk_kda`` reference at those lengths.
- The fused kernel's per-chunk h layout (chunk count and boundary states)
  matches the Triton reference -- ``track_ssm_h_src`` indexes h by chunk, so
  a batch that runs fused while another runs fallback must agree on layout.
- The two remaining fallback guards (unsafe gate, spec-decode draft-extend)
  return exactly the ``HcuKDAKernel`` reference, including the (o, h) tuple
  when intermediate states are requested.

Requires the ``flash_kda`` package and the HCU fla kernels; skips otherwise.
Note: the Triton reference and the fused kernel both run on the current GPU,
so this file also serves as the numeric regression for HCU platforms.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

# Same slot as test_kda_prefill_flashkda.py. Disabled in public CI: flash_kda
# is not in the public runner image.
register_cuda_ci(
    est_time=60,
    stage="base-b",
    runner_config="1-gpu-large",
    disabled="flash_kda not in public CI runner image (only on Ant-internal PyPI)",
)

if not torch.cuda.is_available():
    pytest.skip("FlashKDA requires a CUDA device.", allow_module_level=True)
try:
    import flash_kda  # noqa: F401
except ImportError:
    pytest.skip(
        "flash_kda not installed "
        "(pip install git+https://github.com/MoonshotAI/FlashKDA.git).",
        allow_module_level=True,
    )
try:
    from sglang.srt.layers.attention.linear.kernels.kda_hcu import (  # noqa: F401
        HcuKDAKernel,
    )
    from sglang.kernels.ops.attention.fla.hcu.kda import chunk_kda  # noqa: F401
except Exception:
    pytest.skip(
        "HCU fla kernels (sglang.kernels.ops.attention.fla.hcu) unavailable.",
        allow_module_level=True,
    )

from sglang.srt.layers.attention.linear.kernels.kda_flashkda import (  # noqa: E402
    FlashKDAKernel,
    _FLASHKDA_CHUNK_SIZE,
)
from sglang.srt.layers.attention.linear.kernels.kda_hcu import (  # noqa: E402
    HcuKDAKernel,
)

LOWER_BOUND = -5.0
H, K, V = 16, 128, 128  # FlashKDA requires K == V == 128, HV == H

# Lengths spanning the removed routing boundaries: below the 64-token chunk,
# exactly at both old boundaries, and above the removed 2048 cap.
ROUTING_LENGTHS = [1, 10, 63, 64, 333, 2048, 2049, 4096]


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _make_inputs(seq_lens):
    n = len(seq_lens)
    cu = torch.zeros(n + 1, device="cuda", dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lens, device="cuda").cumsum(0)
    total = int(cu[-1].item())
    idx = torch.arange(n, device="cuda", dtype=torch.int32)
    return dict(
        cu=cu,
        idx=idx,
        seq_lens=seq_lens,
        # RAW q/k (both kernels enable in-kernel qk l2norm).
        q=torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16) * 0.5,
        k=torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16) * 0.5,
        v=torch.randn(1, total, H, V, device="cuda", dtype=torch.bfloat16) * 0.5,
        g=torch.randn(1, total, H, K, device="cuda", dtype=torch.bfloat16) * 0.5,
        # post-sigmoid beta in [0.1, 0.9]; both paths treat beta as activated.
        beta=(torch.rand(1, total, H, device="cuda") * 0.8 + 0.1).to(torch.bfloat16),
        A_log=torch.randn(1, 1, H, 1, device="cuda", dtype=torch.float32) * 0.5,
        dt_bias=torch.randn(H * K, device="cuda", dtype=torch.float32) * 0.1,
        pool=torch.randn(n, H, V, K, device="cuda", dtype=torch.float32) * 0.1,
    )


def _run_hcu_fallback_ref(d, lower_bound=LOWER_BOUND, return_intermediate_state=False):
    """Run the production HCU Triton chunk_kda reference directly.

    Returns o, h_or_None, updated_state_slots (chunk_kda returns (o, h) only
    when intermediate states were requested)."""
    st = d["pool"].clone()
    out = HcuKDAKernel().extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        ssm_states=st,
        cache_indices=d["idx"],
        query_start_loc=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=lower_bound,
        return_intermediate_state=return_intermediate_state,
    )
    if return_intermediate_state:
        o, h = out
        return o, h, st[d["idx"]]
    return out, None, st[d["idx"]]


def _run_flashkda(d, kernel, **kwargs):
    st = d["pool"].clone()
    out = kernel.extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        ssm_states=st,
        cache_indices=d["idx"],
        query_start_loc=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=kwargs.pop("lower_bound", LOWER_BOUND),
        extend_seq_lens_cpu=d["seq_lens"],
        return_intermediate_states=kwargs.pop("return_intermediate_states", False),
        **kwargs,
    )
    if isinstance(out, tuple):
        return out[0], out[1], st[d["idx"]]
    return out, None, st[d["idx"]]


def test_should_fall_back_only_correctness_guards():
    """Length routing is gone: only the unsafe gate and spec-decode guards."""
    device = "cuda"
    for seq_lens in ([63], [64], [2048], [2049], [100000], [1, 63, 64, 2049]):
        cu = torch.zeros(len(seq_lens) + 1, device=device, dtype=torch.int32)
        cu[1:] = torch.tensor(seq_lens, device=device).cumsum(0)
        fused = FlashKDAKernel._should_fall_back(LOWER_BOUND, False, cu, seq_lens)
        assert fused is False, f"lengths {seq_lens} must run fused"

    cu = torch.tensor([0, 128], device=device, dtype=torch.int32)
    assert FlashKDAKernel._should_fall_back(None, False, cu, [128]) is True
    assert FlashKDAKernel._should_fall_back(LOWER_BOUND, True, cu, [128]) is True


@pytest.mark.parametrize("seq_lens", [[t] for t in ROUTING_LENGTHS])
def test_all_lengths_run_fused_when_switch_on(seq_lens):
    """With guards passing, the HCU-wrapped kernel must take the exact same
    fused path as the plain kernel -- routing length must not decide."""
    torch.manual_seed(seq_lens[0])
    d = _make_inputs(seq_lens)

    out_plain, h_plain, state_plain = _run_flashkda(d, FlashKDAKernel())
    out_wrapped, h_wrapped, state_wrapped = _run_flashkda(
        d, FlashKDAKernel(fallback_kernel=HcuKDAKernel())
    )

    assert torch.equal(out_plain, out_wrapped), "wrapped kernel bypassed the fused path"
    assert torch.equal(state_plain, state_wrapped)
    assert h_wrapped is None


def test_matches_hcu_triton_reference_at_old_fallback_boundaries():
    """Numeric equivalence across the boundaries that previously fell back."""
    for seq_lens in ([10], [63], [64], [2049], [3000]):
        torch.manual_seed(hash(tuple(seq_lens)) % (2**31))
        d = _make_inputs(seq_lens)

        ref_out, _, ref_state = _run_hcu_fallback_ref(d)
        out, _, state = _run_flashkda(d, FlashKDAKernel(fallback_kernel=HcuKDAKernel()))

        assert _cos(out, ref_out) >= 0.99, f"output mismatch at lens {seq_lens}"
        assert torch.allclose(
            state, ref_state, rtol=5e-2, atol=5e-2
        ), f"state mismatch at lens {seq_lens}"
        batched_err = (state - ref_state).abs().max().item()
        assert batched_err < 0.5, f"state max-abs err {batched_err} at lens {seq_lens}"


def test_chunk_h_layout_matches_hcu_triton_reference():
    """track_ssm_h_src indexes h by chunk: fused and Triton reference must
    produce the same chunk count and aligned boundary states."""
    torch.manual_seed(7)
    seq_lens = [200, 260]  # cross-sequence batch, non power-of-two lengths
    d = _make_inputs(seq_lens)

    ref_out, ref_h, _ = _run_hcu_fallback_ref(d, return_intermediate_state=True)
    wrapped = FlashKDAKernel(fallback_kernel=HcuKDAKernel())
    out, h, _ = _run_flashkda(d, wrapped, return_intermediate_states=True)

    assert ref_out.shape == out.shape
    assert h is not None and ref_h is not None
    assert h.shape == ref_h.shape, (
        f"chunk layout diverged: fused {tuple(h.shape)} vs triton {tuple(ref_h.shape)}"
    )
    assert _cos(h, ref_h) >= 0.99, "boundary states diverged between kernels"


def test_unsafe_gate_routes_to_hcu_fallback_exactly():
    """lower_bound=None (unsafe gate) must return the HcuKDAKernel result."""
    torch.manual_seed(11)
    d = _make_inputs([96, 320])

    ref_out, _, ref_state = _run_hcu_fallback_ref(d, lower_bound=None)
    out, _, state = _run_flashkda(
        d, FlashKDAKernel(fallback_kernel=HcuKDAKernel()), lower_bound=None
    )

    assert torch.equal(out, ref_out), "unsafe gate did not run the HCU fallback"
    assert torch.equal(state, ref_state)


def test_spec_decode_routes_to_hcu_fallback_with_h_tuple():
    """is_spec_decode (draft-extend-v2) must run the fallback and keep the
    (o, h) tuple contract so rollback bookkeeping sees intermediate states."""
    torch.manual_seed(13)
    d = _make_inputs([128])

    ref_out, ref_h, _ = _run_hcu_fallback_ref(
        d, lower_bound=LOWER_BOUND, return_intermediate_state=True
    )
    wrapped = FlashKDAKernel(fallback_kernel=HcuKDAKernel())
    out = wrapped.extend(
        d["q"].clone(),
        d["k"].clone(),
        d["v"].clone(),
        d["g"].clone(),
        d["beta"].clone(),
        ssm_states=d["pool"].clone(),
        cache_indices=d["idx"],
        query_start_loc=d["cu"],
        A_log=d["A_log"],
        dt_bias=d["dt_bias"],
        lower_bound=LOWER_BOUND,
        extend_seq_lens_cpu=d["seq_lens"],
        is_spec_decode=True,
        return_intermediate_states=True,
    )

    # The HCU fallback returns exactly what HcuKDAKernel.extend returns.
    assert isinstance(out, tuple) and len(out) == 2, (
        "spec path must return the (o, h) tuple for rollback bookkeeping"
    )
    out_tensor, out_h = out
    assert out_h is not None and ref_h is not None
    assert ref_h.shape == out_h.shape
    assert _cos(out_tensor, ref_out) >= 0.99
    assert _cos(out_h, ref_h) >= 0.99


def test_fallback_returns_plain_o_without_intermediate_states():
    """Normalizing: fallback must return bare o (not (o, h)) when h was not
    requested, mirroring the Triton kernel's contract."""
    torch.manual_seed(17)
    d = _make_inputs([64, 128])
    wrapped = FlashKDAKernel(fallback_kernel=HcuKDAKernel())

    out, h, _ = _run_flashkda(d, wrapped, lower_bound=None)
    assert isinstance(out, torch.Tensor)
    assert h is None

    assert _FLASHKDA_CHUNK_SIZE == 64  # intermediate-state sizing depends on it


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
