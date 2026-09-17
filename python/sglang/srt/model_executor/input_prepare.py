"""Input metadata transfers with per-step ownership of the pinned snapshot."""

from typing import Optional, Sequence, Tuple

import torch


def copy_input_metadata_to_device(
    num_tokens: Optional[int],
    global_num_tokens: Sequence[int],
    global_num_tokens_for_logprob: Sequence[int],
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Pack the small CPU inputs into one asynchronous H2D transfer.

    The returned views retain their original dtypes and shapes. Each call owns
    a fresh pinned snapshot and GPU allocation, so subsequent CPU preparation
    and in-place MLP padding cannot overwrite another step's in-flight inputs.
    """
    values = []
    if num_tokens is not None:
        values.append(torch.tensor(num_tokens, dtype=torch.int32))
    values.extend(
        (
            torch.tensor(global_num_tokens, dtype=torch.int64),
            torch.tensor(global_num_tokens_for_logprob, dtype=torch.int64),
        )
    )
    offsets = []
    size = 0
    for value in values:
        # Preserve alignment for both the int32 scalar and int64 vectors.
        alignment = value.element_size()
        size = (size + alignment - 1) // alignment * alignment
        offsets.append(size)
        size += value.numel() * alignment
    snapshot = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    for value, offset in zip(values, offsets):
        nbytes = value.numel() * value.element_size()
        snapshot[offset : offset + nbytes].view(value.dtype).view(value.shape).copy_(
            value
        )
    packed = snapshot.to(device, non_blocking=True)
    outputs = []
    for value, offset in zip(values, offsets):
        nbytes = value.numel() * value.element_size()
        outputs.append(
            packed[offset : offset + nbytes].view(value.dtype).view(value.shape)
        )
    if num_tokens is None:
        return None, outputs[0], outputs[1]
    return outputs[0], outputs[1], outputs[2]
