# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""BF16 absorb BMM for the two GLM gfx938 decode layouts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _absorb_bmm(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
):
    tile, head = tl.program_id(0), tl.program_id(1)
    mi = tile // tl.cdiv(N, 64) * 32 + tl.arange(0, 32)
    ni = tile % tl.cdiv(N, 64) * 64 + tl.arange(0, 64)
    ki = tl.arange(0, 64)
    acc = tl.zeros((32, 64), tl.float32)
    for start in range(tl.cdiv(K, 64)):
        kk = start * 64 + ki
        av = tl.load(
            A + head * A0 + mi[:, None] * A1 + kk[None, :] * A2,
            (mi[:, None] < M) & (kk[None, :] < K),
            other=0.0,
        )
        bv = tl.load(
            B + head * B0 + kk[:, None] * B1 + ni[None, :] * B2,
            (kk[:, None] < K) & (ni[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(av, bv, acc)
    tl.store(
        C + head * C0 + mi[:, None] * C1 + ni[None, :] * C2,
        acc,
        (mi[:, None] < M) & (ni[None, :] < N),
    )


def try_glm_absorb_bmm(a, b, out=None):
    """Return None before allocation or launch for an unsupported contract."""
    if (
        a.ndim != 3
        or b.ndim != 3
        or any(
            t.dtype != torch.bfloat16 or not t.is_cuda or t.device != a.device
            for t in (a, b)
        )
        or torch.version.hip is None
    ):
        return None
    heads, m, k = a.shape
    n = b.shape[2]
    if heads != 64 or not (m == 4 or 6 <= m <= 30 or m == 32):
        return None
    if (k, n) == (192, 512):
        a_stride, b_stride = (256, 16384, 1), (98304, 1, 192)
        c_stride = (m * 512, 512, 1)
    elif (k, n) == (512, 256) and out is not None:
        a_stride, b_stride = (512, 32768, 1), (131072, 1, 512)
        c_stride = (256, 16384, 1)
    else:
        return None
    if (
        b.shape != (heads, k, n)
        or a.stride() != a_stride
        or b.stride() != b_stride
        or a.data_ptr() % 16
        or b.data_ptr() % 16
    ):
        return None
    if out is not None and (
        out.shape != (heads, m, n)
        or out.stride() != c_stride
        or out.dtype != torch.bfloat16
        or out.device != a.device
        or out.data_ptr() % 16
        or any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in (a, b)
        )
    ):
        return None
    props = torch.cuda.get_device_properties(a.device)
    if (
        props.gcnArchName.split(":", 1)[0] != "gfx938"
        or props.multi_processor_count != 64
    ):
        return None
    if out is None:
        out = torch.empty((heads, m, n), dtype=a.dtype, device=a.device)
    _absorb_bmm[(triton.cdiv(m, 32) * triton.cdiv(n, 64), heads)](
        a,
        b,
        out,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        *out.stride(),
        num_warps=4,
        num_stages=2,
    )
    return out
