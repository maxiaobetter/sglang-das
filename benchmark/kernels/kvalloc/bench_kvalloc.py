"""Microbenchmark: DCU-native vs Triton KV slot allocation kernels.

Mirrors exactly how PagedTokenToKVPoolAllocator calls each kernel
(python/sglang/srt/mem_cache/allocator/paged.py), so the numbers reflect the
SGLANG_KVALLOC_KERNEL=1 (DCU) vs =0 (Triton) code paths.

Usage (inside the container, on an idle GPU):
    HIP_VISIBLE_DEVICES=0 python3 bench_kvalloc.py
"""

import json
import statistics
import sys
import time

import torch
from sgl_kernel.kvcacheio import hcu_alloc_decode_kernel, hcu_alloc_extend_kernel

from sglang.kernels.ops.memory.allocator import alloc_decode_kernel, alloc_extend_kernel
from sglang.srt.utils import next_power_of_2

DEV = "cuda"
PAGE = 64
NUM_PAGES = 2_502_784 // PAGE  # same KV pool size as the GLM5.3 D node (per DP rank)
WARMUP, ITERS = 3, 20
torch.manual_seed(0)
free_pages = (torch.randperm(NUM_PAGES - 1, device=DEV, dtype=torch.int64) + 1)


def gpu_time_ms(fn, iters=ITERS):
    """Median GPU time per call (events around a single launch)."""
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), max(ts)


def host_time_ms(fn, iters=ITERS):
    """Median host-side launch cost (no sync inside the timed region)."""
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    torch.cuda.synchronize()
    return statistics.median(ts)


def first_call_ms(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1e3


def make_extend(bs, extend_len, prefix_len):
    """Requests with `prefix_len` cached tokens extending by `extend_len`."""
    pre = torch.full((bs,), prefix_len, dtype=torch.int64, device=DEV)
    seq = pre + extend_len
    # last_loc: last token slot of the prefix (or -1); keep page alignment consistent.
    if prefix_len == 0:
        last = torch.full((bs,), -1, dtype=torch.int64, device=DEV)
    else:
        last_pages = free_pages[-bs:]  # disjoint from pages handed out below
        last = last_pages * PAGE + (prefix_len - 1) % PAGE
    return pre, seq, last


def bench_extend(bs, extend_len, prefix_len):
    pre, seq, last = make_extend(bs, extend_len, prefix_len)
    n = bs * extend_len
    out_dcu = torch.empty(n, dtype=torch.int64, device=DEV)
    out_tri = torch.empty(n, dtype=torch.int64, device=DEV)

    def dcu():  # allocator.py, SGLANG_KVALLOC_KERNEL=1
        hcu_alloc_extend_kernel(
            pre_lens_ptr=pre.to(torch.int64),
            seq_lens_ptr=seq.to(torch.int64),
            last_loc_ptr=last.to(torch.int64),
            free_page_ptr=free_pages.to(torch.int64),
            out_indices=out_dcu,
            bs=bs,
            page_size=PAGE,
        )

    def tri():  # allocator.py, SGLANG_KVALLOC_KERNEL=0
        alloc_extend_kernel[(bs,)](
            pre, seq, last, free_pages, out_tri, next_power_of_2(bs), PAGE
        )

    first_tri = first_call_ms(tri)
    first_dcu = first_call_ms(dcu)
    for _ in range(WARMUP):
        dcu(); tri()
    torch.cuda.synchronize()
    equal = torch.equal(out_dcu, out_tri)
    g_dcu, gmax_dcu = gpu_time_ms(dcu, iters=5 if n > 2_000_000 else ITERS)
    g_tri, gmax_tri = gpu_time_ms(tri)
    return dict(
        kind="extend", bs=bs, extend_len=extend_len, prefix_len=prefix_len,
        total_tokens=n, equal=equal,
        dcu_gpu_ms=g_dcu, dcu_gpu_max_ms=gmax_dcu,
        triton_gpu_ms=g_tri, triton_gpu_max_ms=gmax_tri,
        dcu_host_ms=host_time_ms(dcu, iters=5 if n > 2_000_000 else ITERS),
        triton_host_ms=host_time_ms(tri),
        dcu_first_call_ms=first_dcu, triton_first_call_ms=first_tri,
    )


def bench_decode(bs, seq_len_base=100_000):
    seq = torch.randint(seq_len_base, seq_len_base + PAGE, (bs,), device=DEV, dtype=torch.int64)
    # seq_len % PAGE == 1 means a new page is needed; mix both cases.
    seq[::2] = (seq[::2] // PAGE) * PAGE + 1
    last_pages = free_pages[-bs:]
    # Production dtype: alloc_for_decode reads last_loc from req_to_token (int32).
    last = (last_pages * PAGE + (seq - 2) % PAGE).to(torch.int32)
    out_dcu = torch.empty(bs, dtype=torch.int64, device=DEV)
    out_tri = torch.empty(bs, dtype=torch.int64, device=DEV)

    def dcu():
        hcu_alloc_decode_kernel(
            seq_lens_ptr=seq, last_loc_ptr=last, free_page_ptr=free_pages,
            out_indices=out_dcu, bs=bs, page_size=PAGE,
        )

    def tri():
        alloc_decode_kernel[(bs,)](
            seq, last, free_pages, out_tri, next_power_of_2(bs), PAGE
        )

    first_tri = first_call_ms(tri)
    first_dcu = first_call_ms(dcu)
    for _ in range(WARMUP):
        dcu(); tri()
    torch.cuda.synchronize()
    return dict(
        kind="decode", bs=bs, equal=torch.equal(out_dcu, out_tri),
        mismatch_rows=int((out_dcu != out_tri).sum()),
        dcu_gpu_ms=gpu_time_ms(dcu)[0], triton_gpu_ms=gpu_time_ms(tri)[0],
        dcu_host_ms=host_time_ms(dcu), triton_host_ms=host_time_ms(tri),
        dcu_first_call_ms=first_dcu, triton_first_call_ms=first_tri,
    )


def main():
    rows = []
    # 1) PD decode prealloc: one request, whole prompt allocated at once.
    for L in (16_384, 32_768, 65_536, 100_000, 500_000):
        rows.append(bench_extend(1, L, 0))
        rows.append(bench_extend(1, L, 100))  # unaligned prefix -> part1/part3 paths
    # 2) Batched long extend (several prealloc'd / prefill requests together).
    for L in (16_384, 100_000):
        rows.append(bench_extend(4, L, 0))
    # 3) Spec-v2 per-decode-step extend: each request grows by 2*alloc_len_per_decode
    #    (12 tokens for num_steps=5) on top of a ~100k context.
    for bs in (1, 10, 16, 64, 160):
        rows.append(bench_extend(bs, 12, 100_003))
    # 4) Non-spec decode (alloc_decode).
    for bs in (1, 10, 16, 64, 160):
        rows.append(bench_decode(bs))
    for r in rows:
        print(json.dumps(r))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
