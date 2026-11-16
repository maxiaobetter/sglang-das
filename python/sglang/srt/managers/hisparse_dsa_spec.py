"""Fixed-width GLM/DSA MTP transactions, following HiSparse bda41d6895.

The target keeps a hot cache plus one write page and a union scratch buffer.
The single MTP layer keeps dense KV in the allocator's logical location space.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.kvcache.hisparse_spec import (
    HiSparseSpecState,
    backup_accepted_mla,
    load_cache_to_device_buffer_spec_mla,
)


@triton.jit
def _map_verify_window(
    REQS,
    PREFIX,
    LOCS,
    ORIGINAL_LOCS,
    TOKENS,
    MAPPING,
    WIDTH: tl.constexpr,
    HOT: tl.constexpr,
    PAGE: tl.constexpr,
    BUFFER_STRIDE: tl.constexpr,
    TOKEN_LAYER_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid, layer = tl.program_id(0), tl.program_id(1)
    rid = tl.load(REQS + bid)
    prefix = tl.load(PREFIX + bid)
    offsets = tl.arange(0, BLOCK)
    tl.store(
        TOKENS + layer * TOKEN_LAYER_STRIDE + rid * BUFFER_STRIDE + HOT + offsets,
        tl.where(offsets < WIDTH, prefix + offsets, -1),
        offsets < PAGE,
    )
    if layer == 0:
        logical = tl.load(LOCS + bid * WIDTH + offsets, offsets < WIDTH, other=0)
        physical = tl.load(
            ORIGINAL_LOCS + rid * BUFFER_STRIDE + HOT + offsets,
            offsets < WIDTH,
            other=0,
        )
        tl.store(MAPPING + logical, physical, (offsets < WIDTH) & (logical > 0))


@triton.jit
def _clear_verify_mapping(LOCS, MAPPING, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(LOCS + offsets, offsets < N, other=0)
    # These addresses belong to the coordinator, not the logical allocator.
    tl.store(MAPPING + logical, 0, (offsets < N) & (logical > 0))


class DSAHiSparseSpeculativeState:
    def __init__(self, coordinator, width: int):
        self.child = coordinator
        self.width = width
        self.top_k = coordinator.top_k
        self.page = coordinator.mem_pool_device.page_size
        self.occurrences = width * self.top_k
        if not 2 <= width <= min(16, self.page):
            raise ValueError("DSA HiSparse MTP requires a verify width in [2, 16]")
        if self.top_k < 1024 or self.top_k % 64 or self.occurrences > 32768:
            raise ValueError("DSA HiSparse MTP exceeds the speculative swap capacity")
        self.capacity = triton.cdiv(self.occurrences, self.page) * self.page
        layers, slots, _ = coordinator.req_device_buffer_tokens.shape
        if slots > coordinator.device_buffer_size:
            raise ValueError("HiSparse MTP request slots exceed CLOCK control capacity")
        device = coordinator.req_to_device_buffer.device
        self.original_scratch = {}
        self.states = [
            HiSparseSpecState(
                torch.full(
                    (slots, 2, triton.next_power_of_2(coordinator.device_buffer_size)),
                    -1,
                    dtype=torch.int64,
                    device=device,
                ),
                torch.zeros(
                    (slots + 1, coordinator.device_buffer_size),
                    dtype=torch.int32,
                    device=device,
                ),
                torch.full(
                    (slots, self.capacity), -1, dtype=torch.int32, device=device
                ),
                torch.zeros(
                    (slots + 1, max(4 * slots, 5 * self.occurrences)),
                    dtype=torch.int32,
                    device=device,
                ),
            )
            for _ in range(layers)
        ]
        self.output = torch.full(
            (slots, width, self.top_k), -1, dtype=torch.int32, device=device
        )

    def admit(self, req):
        child = self.child
        rid = req.req_pool_idx
        if rid in self.original_scratch:
            raise RuntimeError("HiSparse MTP request already owns verify scratch")
        locs = child.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc(
            self.capacity
        )
        if locs is None:
            raise RuntimeError("HiSparse MTP cannot allocate verify scratch")
        self.original_scratch[rid] = locs
        for state in self.states:
            state.scratch_locs[rid].copy_(locs)
        n = req.extend_range.end
        labels = child.req_device_buffer_tokens[:, rid, : child.device_buffer_size]
        labels.masked_fill_(labels >= n, -1)
        child.req_device_buffer_tokens[:, rid, child.device_buffer_size :].fill_(-1)

    def release(self, req):
        rid = req.req_pool_idx
        locs = self.original_scratch.pop(rid, None)
        if locs is not None:
            self.child.token_to_kv_pool_allocator.free_hisparse_indices(locs)
        slots = self.child.req_to_device_buffer.shape[0]
        for state in self.states:
            state.cache_index[rid].fill_(-1)
            state.cache_policy[0, rid] = 0
            state.cache_policy[rid + 1].zero_()
            state.scratch_locs[rid].fill_(-1)
            state.scratch_state[0, rid : 4 * slots : slots] = 0
            state.scratch_state[rid + 1].zero_()

    def prepare(self, reqs, prefix, locs):
        child = self.child
        if locs.numel() != reqs.numel() * self.width:
            raise ValueError("HiSparse MTP requires fixed-width request-major verify")
        if reqs.numel() == 0:
            return
        _map_verify_window[(reqs.numel(), len(self.states))](
            reqs,
            prefix,
            locs,
            child.req_to_device_buffer,
            child.req_device_buffer_tokens,
            child.mem_pool_device.full_to_hisparse_device_index_mapping,
            self.width,
            child.device_buffer_size,
            self.page,
            child.padded_buffer_size,
            child.req_device_buffer_tokens.stride(0),
            triton.next_power_of_2(self.page),
        )

    def commit(self, reqs, prefix, locs, accepted):
        if reqs.numel() == 0:
            return
        child = self.child
        # Stay on the verify stream: host bytes are complete before the next
        # forward may overwrite the write page or retire a request's host pool.
        backup_accepted_mla(
            src_layers=child.mem_pool_device.data_ptrs,
            dst_layers=child.mem_pool_host.data_ptrs,
            reqs=reqs,
            prefix=prefix,
            accepted=accepted,
            write_locs=child.req_to_device_buffer,
            host_locs=child.req_to_host_pool,
            hot_size=child.device_buffer_size,
            width=self.width,
            top_k=self.top_k,
            item_size_bytes=child.item_size_bytes,
        )
        _clear_verify_mapping[(triton.cdiv(locs.numel(), 256),)](
            locs,
            child.mem_pool_device.full_to_hisparse_device_index_mapping,
            locs.numel(),
            256,
        )

    def swap(self, reqs, positions, topk, layer, **plan):
        child = self.child
        bs = topk.shape[0] // self.width
        if topk.shape != (bs * self.width, self.top_k) or reqs.numel() != bs:
            raise ValueError("HiSparse MTP requires request-major per-row Top-K")
        if positions.numel() != bs * self.width:
            raise ValueError("HiSparse MTP positions must cover every verify row")
        if self.output.shape[0] < bs:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("HiSparse MTP output must grow before graph capture")
            self.output = self.output.new_full((bs, self.width, self.top_k), -1)
        output = self.output[:bs]
        load_cache_to_device_buffer_spec_mla(
            top_k_tokens=topk.contiguous().view(bs, self.width, self.top_k),
            device_buffer_tokens=child.req_device_buffer_tokens[layer],
            host_cache_locs=child.req_to_host_pool,
            device_buffer_locs=child.req_device_buffer_token_locs[layer],
            host_cache=child.mem_pool_host.kv_buffer[layer],
            device_buffer=child.mem_pool_device.kv_buffer[layer],
            top_k_device_locs=output,
            req_pool_indices=reqs.to(torch.int64),
            seq_lens=(positions + 1).to(torch.int32),
            state=self.states[layer],
            num_real_reqs=child.num_real_reqs,
            **plan,
        )
        return output.view(bs * self.width, self.top_k)
