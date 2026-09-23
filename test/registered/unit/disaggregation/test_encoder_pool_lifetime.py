import copy
import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.disaggregation.encoder.receiver import (
    EmbeddingPool,
    GlmWaitingRDMARequest,
    WaitingMMRequestBase,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _received(kind="glm", slot=7):
    release = Mock()
    pool = SimpleNamespace(release=release)
    pool.release_on_gc = lambda obj, index: EmbeddingPool.release_on_gc(pool, obj, index)
    raw = MultimodalProcessorOutput(mm_items=[MultimodalDataItem(
        modality=Modality.IMAGE, hash=123, pad_value=1000123, offsets=[(0, 3)],
        precomputed_embeddings=torch.arange(8).reshape(4, 2).float(),
    )])
    receiver = SimpleNamespace(
        recv_req=SimpleNamespace(encoder_part_routes=[{}]),
        embedding_pool=pool, _pool_slot_id=slot,
        embeddings_buffer=torch.empty(8), _mm_finalizer=None,
        _part_buffers={0: (torch.empty(8), slot, 8)}, _part_finalizers=[],
    )
    if kind == "glm":
        GlmWaitingRDMARequest._bind_pool_slot_to_mm_inputs(receiver, raw)
        finalizer = receiver._part_finalizers[0]
    else:
        WaitingMMRequestBase._bind_pool_slot_to_mm_inputs(receiver, raw)
        finalizer = receiver._mm_finalizer
    converted = MultimodalInputs.from_processor_output(raw)
    return converted, release, finalizer, weakref.ref(raw)


@pytest.mark.parametrize("kind", ["generic", "glm"])
def test_pool_slot_survives_processor_output_conversion(kind):
    converted, release, finalizer, raw_ref = _received(kind)
    gc.collect()
    # The scheduler still holds zero-copy image views. Reusing this slot now
    # lets a concurrent encoder request overwrite those views before forward.
    release.assert_not_called()
    assert raw_ref() is not None
    assert torch.equal(converted.mm_items[0].precomputed_embeddings,
                       torch.arange(8).reshape(4, 2).float())
    del converted
    gc.collect()
    release.assert_called_once_with(7)
    assert raw_ref() is None
    assert not finalizer.alive


def test_pool_owners_survive_shallow_copy_and_merge():
    first, release_first, _, _ = _received(slot=1)
    second, release_second, _, _ = _received(slot=2)
    merged = copy.copy(first)
    merged.merge(second)
    del first, second
    gc.collect()
    release_first.assert_not_called()
    release_second.assert_not_called()
    assert len(merged.mm_items) == 2
    del merged
    gc.collect()
    release_first.assert_called_once_with(1)
    release_second.assert_called_once_with(2)


def test_abort_releases_pool_slot_only_once():
    converted, release, finalizer, _ = _received()
    release.assert_not_called()
    finalizer()
    release.assert_called_once_with(7)
    del converted
    gc.collect()
    release.assert_called_once_with(7)


def test_regular_embeddings_do_not_retain_processor_output():
    raw = MultimodalProcessorOutput(mm_items=[MultimodalDataItem(
        modality=Modality.IMAGE, hash=123, pad_value=1000123,
        precomputed_embeddings=torch.ones(4, 2),
    )])
    raw_ref = weakref.ref(raw)
    converted = MultimodalInputs.from_processor_output(raw)
    del raw
    gc.collect()
    assert raw_ref() is None
    assert not converted._encoder_buffer_owners
