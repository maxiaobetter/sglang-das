"""Prepare NSA replay metadata directly in persistent CUDA-graph buffers.

Sequence lengths and the request table remain GPU-owned.  The kernel derives
the integer lengths, prefix sums, NSA lengths and both page-table layouts
without temporary tensors, host reads or a subsequent metadata copy.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_decode_metadata_kernel(
    ReqToToken,
    ReqIndices,
    SeqLens,
    CacheLens,
    CuLens,
    NsaLens,
    NsaCuLens,
    ExpandedLens,
    PageTable,
    RealPageTable,
    req_stride,
    page_stride,
    real_stride,
    num_reqs,
    page_cols,
    real_cols,
    NSA_TOPK: tl.constexpr,
    KPOOL: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAS_REAL_TABLE: tl.constexpr,
    LENS_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)

    # This block owns all small metadata.  It reads only the input sequence
    # lengths, so it has no dependency on any other block in this launch.
    if (row == 0) & (block == 0):
        r = tl.arange(0, LENS_BLOCK)
        lens = tl.load(SeqLens + r, r < num_reqs, other=0).to(tl.int32)
        if KPOOL > 1:
            history = (lens // KPOOL) * KPOOL
            nsa = tl.minimum(history, NSA_TOPK) + lens - history
        else:
            nsa = tl.minimum(lens, NSA_TOPK)
        tl.store(CacheLens + r, lens, r < num_reqs)
        tl.store(ExpandedLens + r, lens, r < num_reqs)
        tl.store(NsaLens + r, nsa, r < num_reqs)
        tl.store(CuLens, 0)
        tl.store(NsaCuLens, 0)
        tl.store(CuLens + r + 1, tl.cumsum(lens), r < num_reqs)
        tl.store(NsaCuLens + r + 1, tl.cumsum(nsa), r < num_reqs)

    req = tl.load(ReqIndices + row).to(tl.int64)
    seq_len = tl.load(SeqLens + row)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    page_mask = cols < page_cols
    pages = tl.load(
        ReqToToken + req * req_stride + cols,
        page_mask & (cols < seq_len),
        other=0,
    )
    tl.store(PageTable + row * page_stride + cols, pages, page_mask)

    if HAS_REAL_TABLE:
        source_cols = cols * PAGE_SIZE
        real_mask = cols < real_cols
        real_pages = tl.load(
            ReqToToken + req * req_stride + source_cols,
            real_mask & (source_cols < page_cols) & (source_cols < seq_len),
            other=0,
        )
        tl.store(
            RealPageTable + row * real_stride + cols,
            real_pages // PAGE_SIZE,
            real_mask,
        )


def prepare_decode_metadata(
    req_to_token: torch.Tensor,
    req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    cache_lens: torch.Tensor,
    cu_lens: torch.Tensor,
    nsa_lens: torch.Tensor,
    nsa_cu_lens: torch.Tensor,
    expanded_lens: torch.Tensor,
    page_table: torch.Tensor,
    real_page_table: torch.Tensor | None,
    nsa_topk: int,
    kpool: int,
    page_size: int,
) -> None:
    """Generate one decode batch's final metadata in one GPU launch.

    ``req_indices`` and ``seq_lens`` describe the padded graph batch.  Callers
    order this launch after the GPU update of sequence lengths and before graph
    replay on the same stream (or with a stream event).  The complete page-table
    width is overwritten, including zeroing rows/tails invalidated by request
    replacement or a shorter sequence.  All destinations keep their addresses.
    """
    bs = req_indices.numel()
    if bs == 0:
        return
    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("req_to_token must be a matrix with contiguous columns")
    if req_to_token.dtype != torch.int32:
        raise ValueError("req_to_token must have dtype int32")
    if req_indices.ndim != 1 or not req_indices.is_contiguous():
        raise ValueError("req_indices must be a contiguous vector")
    if req_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("req_indices must have dtype int32 or int64")
    if seq_lens.ndim != 1 or not seq_lens.is_contiguous() or seq_lens.numel() < bs:
        raise ValueError(
            "seq_lens must be a contiguous vector with at least bs entries"
        )
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError("seq_lens must have dtype int32 or int64")
    if page_size < 1 or kpool < 1 or nsa_topk < 1:
        raise ValueError("page_size, kpool and nsa_topk must be positive")
    small_outputs = (cache_lens, cu_lens, nsa_lens, nsa_cu_lens, expanded_lens)
    for out, size in zip(small_outputs, (bs, bs + 1, bs, bs + 1, bs)):
        if out.ndim != 1 or not out.is_contiguous() or out.numel() < size:
            raise ValueError(
                "length outputs must be contiguous vectors of sufficient size"
            )
    tables = (page_table,) if real_page_table is None else (page_table, real_page_table)
    for table in tables:
        if table.ndim != 2 or table.stride(1) != 1 or table.shape[0] < bs:
            raise ValueError(
                "page tables must have contiguous columns and at least bs rows"
            )
    outputs = (*small_outputs, *tables)
    if any(out.dtype != torch.int32 for out in outputs):
        raise ValueError("metadata outputs must have dtype int32")
    tensors = (req_to_token, req_indices, seq_lens, *outputs)
    if not req_to_token.is_cuda or any(
        t.device != req_to_token.device for t in tensors
    ):
        raise ValueError("all metadata tensors must share a CUDA/ROCm device")
    width = page_table.shape[1]
    if width < 1 or width > req_to_token.shape[1]:
        raise ValueError("page-table width must fit in req_to_token")
    if real_page_table is not None and real_page_table.shape[1] < triton.cdiv(
        width, page_size
    ):
        raise ValueError("real_page_table is too narrow")
    real = page_table if real_page_table is None else real_page_table
    real_cols = real.shape[1] if real_page_table is not None else 0
    block = 1024
    _prepare_decode_metadata_kernel[(bs, triton.cdiv(max(width, real_cols), block))](
        req_to_token,
        req_indices,
        seq_lens,
        cache_lens,
        cu_lens,
        nsa_lens,
        nsa_cu_lens,
        expanded_lens,
        page_table,
        real,
        req_to_token.stride(0),
        page_table.stride(0),
        real.stride(0),
        bs,
        width,
        real_cols,
        NSA_TOPK=nsa_topk,
        KPOOL=kpool,
        PAGE_SIZE=page_size,
        HAS_REAL_TABLE=real_page_table is not None,
        LENS_BLOCK=triton.next_power_of_2(bs),
        BLOCK=block,
        num_warps=8,
    )
