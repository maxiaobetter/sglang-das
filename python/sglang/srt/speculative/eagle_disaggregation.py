from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from sglang.srt.layers.attention.dsa.utils import (
    should_remap_pd_dsa_seed_to_local_slots,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


def _bytes_view(t: torch.Tensor) -> np.ndarray:
    """Raw-byte numpy view of a contiguous CPU tensor (works for bf16 too)."""
    return t.reshape(-1).view(torch.uint8).numpy()


def _stack_to_device_async(rows: list, dtype: torch.dtype, device) -> torch.Tensor:
    """Stack equally-shaped CPU rows straight into one pinned buffer."""
    rows = [torch.as_tensor(r, dtype=dtype).contiguous() for r in rows]
    shape = (len(rows), *rows[0].shape)
    use_pin = is_pin_memory_available(device)
    staging = torch.empty(shape, dtype=dtype, pin_memory=use_pin)
    if staging.numel():
        dst = staging.view(len(rows), -1).view(torch.uint8).numpy()
        for i, r in enumerate(rows):
            np.copyto(dst[i], _bytes_view(r))
    return staging.to(device, non_blocking=use_pin)


def build_eagle_disagg_draft_input(
    batch: ScheduleBatch,
    last_tokens_tensor: torch.Tensor,
) -> EagleDraftInput:
    # Adaptive spec moves the step count after publish, and this runs once per
    # prebuilt batch.
    spec = get_spec()
    num_states = spec.speculative_eagle_topk
    if spec.enable_multi_layer_eagle:
        num_states *= spec.speculative_num_steps

    topk_p = _stack_to_device_async(
        [req.output_topk_p[:num_states].cpu() for req in batch.reqs],
        torch.float32,
        batch.device,
    )
    topk_index = _stack_to_device_async(
        [req.output_topk_index[:num_states].cpu() for req in batch.reqs],
        torch.int64,
        batch.device,
    )
    hidden_states = _stack_to_device_async(
        [req.hidden_states_tensor.cpu() for req in batch.reqs],
        batch.reqs[0].hidden_states_tensor.dtype,
        batch.device,
    )

    dsa_topk_indices = None
    dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
    if dsa_indices_list and all(t is not None for t in dsa_indices_list):
        dsa_topk_indices = _stack_to_device_async(
            [row.cpu() for row in dsa_indices_list],
            dsa_indices_list[0].dtype,
            batch.device,
        )
        if should_remap_pd_dsa_seed_to_local_slots():
            # PD sends request-relative positions; fused TopK consumes
            # decode-local physical slots. Remap once before the draft loop/graph.
            req_to_token = batch.req_to_token_pool.req_to_token
            table_width = req_to_token.shape[1]
            valid_positions = dsa_topk_indices >= 0
            gather_positions = dsa_topk_indices.clamp(min=0, max=table_width - 1).to(
                torch.int64
            )
            local_slots = req_to_token[
                batch.req_pool_indices[:, None], gather_positions
            ]
            invalid_rows = torch.any(
                (dsa_topk_indices < -1)
                | (dsa_topk_indices >= batch.seq_lens[:, None])
                | (dsa_topk_indices >= table_width)
                # Slot 0 is the reserved padding sink; real KV allocations
                # start at 1, and untouched req-to-token entries remain 0.
                | (valid_positions & (local_slots <= 0)),
                dim=1,
            )
            local_slots.masked_fill_(~valid_positions, -1)
            local_slots.masked_fill_(invalid_rows[:, None], -1)
            dsa_topk_indices = local_slots
        if torch.any(torch.all(dsa_topk_indices < 0, dim=1)).item():
            dsa_topk_indices = None

    spec_info = EagleDraftInput(
        topk_p=topk_p,
        topk_index=topk_index,
        hidden_states=hidden_states,
        bonus_tokens=last_tokens_tensor,
        dsa_topk_indices=dsa_topk_indices,
    )
    spec_info.capture_hidden_mode = CaptureHiddenMode.LAST

    return spec_info
