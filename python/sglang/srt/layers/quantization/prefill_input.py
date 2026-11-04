"""Typed, explicit handoff from prefill normalization to its FP8 consumer."""

from typing import NamedTuple

import torch


class PrefillFP8Input(NamedTuple):
    quant: torch.Tensor
    scale: torch.Tensor
    bf16: torch.Tensor
