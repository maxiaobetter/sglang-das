"""Runner-local topology for replicated HCU DFlash drafts.

The target keeps its CP/DP/EP topology. Each attention rank owns a complete
draft, including its KV, so a draft step never waits on another DP request.
"""

from contextlib import contextmanager
from functools import wraps

from sglang.srt.distributed.parallel_state import patch_tensor_parallel_group
from sglang.srt.runtime_context import get_context, get_flags, get_parallel


class DFlashParallelContext:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.group = get_parallel().attn_tp_group if enabled else None
        self.depth = 0
        if enabled and self.group.world_size != 1:
            raise ValueError("HCU DFlash CP/DP currently requires attention TP=1")

    @contextmanager
    def scope(self):
        if not self.enabled or self.depth:
            yield
            return
        self.depth += 1
        try:
            with (
                patch_tensor_parallel_group(self.group),
                get_parallel().override(
                    tp_size=1,
                    tp_rank=0,
                    tp_group=self.group,
                    attn_tp_size=1,
                    attn_tp_rank=0,
                    attn_cp_size=1,
                    attn_cp_rank=0,
                    attn_dp_size=1,
                    attn_dp_rank=0,
                    moe_ep_size=1,
                    moe_ep_rank=0,
                    moe_tp_size=1,
                    moe_tp_rank=0,
                    moe_dp_size=1,
                    moe_dp_rank=0,
                ),
                get_context()
                .config_bag("parallel")
                .override(
                    tp_size=1,
                    dp_size=1,
                    attn_cp_size=1,
                    enable_dp_attention=False,
                    enable_prefill_cp=False,
                    enable_dsa_prefill_context_parallel=False,
                    enable_prefill_context_parallel=False,
                    enable_attn_tp_input_scattered=False,
                ),
                get_flags().dp.override(enabled=False),
            ):
                yield
        finally:
            self.depth -= 1


def draft_local(method):
    """Scope only draft work; target forwards must remain outside this scope."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.draft_parallel.scope():
            return method(self, *args, **kwargs)

    return wrapped
