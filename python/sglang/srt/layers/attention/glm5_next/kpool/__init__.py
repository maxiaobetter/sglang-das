"""Kpool indexer: pooled-history sparse indexer for DSA models.

Layered architecture:

    planner.py    -- build the per-forward plan (CPU planning + H2D)
    indexer.py    -- the IndexerKPool class (forward_cuda entry point)
    kernels.py    -- triton kernels (compress / topk / gather / scatter)

Public API:
    IndexerKPool
    KPoolExtendPlan, KPoolWritePlan, PoolWriteRows, TailWriteRows, KPoolCpInfo
    init_kpool_extend_metadata, init_kpool_write_plan,
    init_pooled_paged_mqa_metadata, update_pooled_paged_mqa_metadata
"""

from typing import TYPE_CHECKING

from sglang.srt.layers.attention.glm5_next.kpool.planner import (
    KPoolCpInfo,
    KPoolExtendPlan,
    KPoolWritePlan,
    PoolWriteRows,
    TailWriteRows,
    init_kpool_extend_metadata,
    init_kpool_write_plan,
    init_pooled_paged_mqa_metadata,
    update_kpool_write_plan_multi_decode,
    update_pooled_paged_mqa_metadata,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.glm5_next.kpool.indexer import IndexerKPool


__all__ = [
    "IndexerKPool",
    "KPoolCpInfo",
    "KPoolExtendPlan",
    "KPoolWritePlan",
    "PoolWriteRows",
    "TailWriteRows",
    "init_kpool_extend_metadata",
    "init_kpool_write_plan",
    "init_pooled_paged_mqa_metadata",
    "update_kpool_write_plan_multi_decode",
    "update_pooled_paged_mqa_metadata",
]


def __getattr__(name: str):
    # Importing planner must not pull indexer back into an unfinished dsa_backend.
    if name == "IndexerKPool":
        from sglang.srt.layers.attention.glm5_next.kpool.indexer import IndexerKPool

        globals()[name] = IndexerKPool
        return IndexerKPool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
