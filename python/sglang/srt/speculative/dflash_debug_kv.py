"""Bounded, read-only DFlash prompt snapshots for P/D and CP comparisons.

Compare requests by their input digest and tokens by logical ``positions``.
``cache_locs`` identify local allocations and must never be compared across peers.
All indexing happens before the recorder copies the small snapshots to the CPU.
"""

import logging
from functools import wraps

import torch

from sglang.srt.speculative.dflash_debug import get_dflash_debug_recorder
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

logger = logging.getLogger(__name__)
_disabled_hooks = set()


def _best_effort_snapshot(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        if function.__name__ in _disabled_hooks:
            return
        try:
            return function(*args, **kwargs)
        except Exception as error:
            # Include eligibility, topology inspection and shape preparation in
            # the protection: debug failures must never fail an inference step.
            _disabled_hooks.add(function.__name__)
            logger.warning("Disabling DFlash %s: %s", function.__name__, error)
            try:
                recorder = get_dflash_debug_recorder()
                if recorder is not None:
                    recorder.disable(error)
            except Exception:
                pass

    return guarded


def _debug_metadata(worker):
    metadata = dict(getattr(worker, "_dflash_debug_context", {}))
    for name in (
        "tp_rank",
        "tp_size",
        "dp_rank",
        "dp_size",
        "attn_tp_rank",
        "attn_tp_size",
        "attn_cp_rank",
        "attn_cp_size",
        "attn_dp_rank",
        "attn_dp_size",
        "pp_rank",
        "pp_size",
    ):
        metadata[f"target_{name}"] = getattr(worker.ps, name)
    return metadata


def _capture_layers(worker):
    draft_config = worker.draft_model_runner.model_config.hf_config
    return parse_dflash_draft_config(
        draft_hf_config=draft_config
    ).resolve_target_layer_ids(
        target_num_layers=worker.model_runner.model_config.hf_text_config.num_hidden_layers,
        draft_num_layers=draft_config.num_hidden_layers,
    )


def _sample_positions(start, end, page_size, limit):
    if end <= start or limit <= 0:
        return []
    # Prioritize both ends and the first/last page transitions, including the
    # tokens on either side of the transition (63, 64, 65 for page size 64).
    first_boundary = max(1, (start + page_size - 1) // page_size) * page_size
    last_boundary = (end - 1) // page_size * page_size
    candidates = [start, end - 1]
    for boundary in (first_boundary, last_boundary):
        candidates.extend((boundary - 1, boundary, boundary + 1))
    candidates.append((start + end - 1) // 2)
    if limit > 1:
        candidates.extend(
            start + i * (end - start - 1) // (limit - 1) for i in range(limit)
        )
    selected = []
    for position in candidates:
        if start <= position < end and position not in selected:
            selected.append(position)
            if len(selected) == limit:
                break
    return sorted(selected)


def _prefill_transfer_owner(worker):
    # Every local draft has the complete prompt KV. Under target LayerSplit,
    # Mooncake actually sends the draft buffers from the last CP rank only.
    pool = worker.model_runner.token_to_kv_pool
    if getattr(pool, "layer_shard_enabled", False):
        if pool.layer_shard_rank != pool.layer_shard_size - 1:
            return False
    if getattr(pool, "requires_descriptor_matched_transfer", False):
        if pool.cp_rank != pool.cp_size - 1:
            return False
    return True


def _recorder_for(worker):
    if getattr(worker.server_args, "speculative_algorithm", None) != "DFLASH":
        return None
    recorder = get_dflash_debug_recorder()
    if recorder is not None and worker.use_compact_draft_cache:
        recorder.disable(
            "DFlash KV/aux snapshots do not support compact draft cache; "
            "target and draft request mappings can differ"
        )
        return None
    return recorder


def _selected_kv(pool, local_layer, cache_locs):
    # Read raw storage: get_key_buffer/get_value_buffer may wait for a layer
    # transfer or materialize/dequantize a whole pool. Neither belongs in debug.
    if getattr(pool, "is_quantized_kv_cache", False) or pool.store_dtype != pool.dtype:
        raise ValueError("Draft KV debug requires unquantized KV storage")
    key = pool.k_buffer[local_layer]
    value = pool.v_buffer[local_layer]
    layout = getattr(pool, "kv_cache_layout", "nhd")
    if key.ndim == value.ndim == 3 and layout == "nhd":
        return key.index_select(0, cache_locs), value.index_select(0, cache_locs)
    pages = cache_locs // pool.page_size
    offsets = cache_locs % pool.page_size
    if key.ndim == value.ndim == 4 and layout == "hnd":
        return key[pages, :, offsets, :], value[pages, :, offsets, :]
    if key.ndim == value.ndim == 4 and layout == "nhd":
        # HCU FA storage: K [page,H,page_size,D], V [page,H,D,page_size].
        if key.shape[2] == pool.page_size and value.shape[3] == pool.page_size:
            return key[pages, :, offsets, :], value[pages, :, :, offsets]
    raise ValueError(
        f"Unsupported draft KV debug layout {layout}: "
        f"K={tuple(key.shape)}, V={tuple(value.shape)}"
    )


def _record_error(recorder, req, phase, error):
    logger.warning("DFlash %s snapshot failed for %s: %s", phase, req.rid, error)
    recorder.write(req, phase, {"snapshot_error": str(error)}, {})


def _add_kv_tensors(pool, cache_locs, tensors):
    # Validate only sampled indices before dereferencing storage; a corrupted
    # mapping should produce a debug error, not an out-of-bounds GPU access.
    sampled_locs = cache_locs.detach().cpu().tolist()
    if any(loc < 0 or loc >= pool.size + pool.page_size for loc in sampled_locs):
        raise ValueError(f"Invalid sampled draft KV locations: {sampled_locs}")
    layer_ids = list(range(pool.start_layer, pool.start_layer + pool.layer_num))
    for local_layer, layer_id in enumerate(layer_ids):
        key, value = _selected_kv(pool, local_layer, cache_locs)
        tensors[f"layer_{layer_id}_k"] = key
        tensors[f"layer_{layer_id}_v"] = value
    return layer_ids


@_best_effort_snapshot
def record_dflash_prompt_kv(worker, batch, *, phase):
    """Call after P prompt materialization or before D's first draft forward.

    The caller must have completed the real transfer before the decode call.
    Only a complete original prompt is sampled; later decode steps are skipped.
    No KV getter, cache write, collective, or new device-wide barrier is used.
    """
    recorder = _recorder_for(worker)
    if recorder is None:
        return
    if phase not in ("prefill_kv", "decode_kv"):
        raise ValueError(f"Unexpected DFlash KV debug phase: {phase}")
    if phase == "prefill_kv" and not _prefill_transfer_owner(worker):
        return
    for index, req in enumerate(batch.reqs):
        if not recorder.enabled_for(req, phase):
            continue
        try:
            prompt_length = len(req.origin_input_ids)
            # The P final chunk ends here. On D the first sampled P token is a
            # bonus token; its KV has not yet been appended to this prompt.
            if int(batch.seq_lens[index].item()) != prompt_length:
                continue
            pool = worker.draft_model_runner.token_to_kv_pool
            selected = _sample_positions(
                0, prompt_length, pool.page_size, recorder.max_tokens
            )
            if not selected:
                continue
            mapping = worker.model_runner.req_to_token_pool.req_to_token
            positions = torch.tensor(selected, dtype=torch.int64, device=mapping.device)
            cache_locs = mapping[batch.req_pool_indices[index], positions].to(
                torch.int64
            )
            tensors = {
                "positions": positions,
                "cache_locs": cache_locs,
                "input_ids": torch.tensor(
                    [req.origin_input_ids[p] for p in selected], dtype=torch.int64
                ),
            }
            layer_ids = _add_kv_tensors(pool, cache_locs, tensors)
            recorder.write(
                req,
                phase,
                {
                    **_debug_metadata(worker),
                    "page_size": pool.page_size,
                    "kv_layout": getattr(pool, "kv_cache_layout", "nhd"),
                    "kv_dtype": str(pool.dtype),
                    "layer_ids": layer_ids,
                    "comparison_key": "input_ids_sha256, positions, layer_id",
                    "cache_locs_are_peer_local": True,
                },
                tensors,
            )
        except Exception as error:
            _record_error(recorder, req, phase, error)


@_best_effort_snapshot
def record_dflash_prefill_aux(worker, batch, *, target_hidden, positions):
    """Save original, gathered target features before P discards the tensor.

    Chunked prefill samples only the current extend range. Capture layers are
    checkpoint indices (zero-based outputs), in packed feature order.
    """
    recorder = _recorder_for(worker)
    if recorder is None or not _prefill_transfer_owner(worker):
        return
    phase = "prefill_aux"
    packed_start = 0
    for index, req in enumerate(batch.reqs):
        extend_length = int(batch.extend_lens[index])
        row_start = packed_start
        packed_start += extend_length
        if not recorder.enabled_for(req, phase):
            continue
        try:
            start = int(batch.prefix_lens[index])
            end = min(start + extend_length, len(req.origin_input_ids))
            pool = worker.draft_model_runner.token_to_kv_pool
            selected = _sample_positions(
                start, end, pool.page_size, recorder.max_tokens
            )
            if not selected:
                continue
            rows = torch.tensor(
                [row_start + p - start for p in selected],
                dtype=torch.int64,
                device=target_hidden.device,
            )
            recorder.write(
                req,
                phase,
                {
                    **_debug_metadata(worker),
                    "page_size": pool.page_size,
                    "capture_layers": _capture_layers(worker),
                    "capture_semantics": "zero_based_target_layer_output",
                    "extend_start": start,
                    "extend_end": start + extend_length,
                    "cache_locs_are_peer_local": True,
                },
                {
                    "positions": positions.index_select(0, rows),
                    "input_ids": torch.tensor(
                        [req.origin_input_ids[p] for p in selected], dtype=torch.int64
                    ),
                    "cache_locs": batch.out_cache_loc.index_select(0, rows),
                    "aux_hidden": target_hidden.index_select(0, rows),
                },
            )
        except Exception as error:
            _record_error(recorder, req, phase, error)


@_best_effort_snapshot
def record_dflash_verify_state(
    worker,
    batch,
    *,
    target_hidden,
    positions,
    cache_locs,
    candidates,
    commit_lens,
):
    """Snapshot committed verify inputs after their draft KV materialization.

    The committed prefix includes the anchor and accepted draft inputs, but
    excludes the newly sampled bonus token: that token has no target hidden
    state or committed KV yet. Never read the rejected suffix cache slots.
    """
    recorder = _recorder_for(worker)
    if recorder is None:
        return
    phase = "verify_state"
    selected_requests = [
        (index, req)
        for index, req in enumerate(batch.reqs)
        if recorder.enabled_for(req, phase)
    ]
    if not selected_requests:
        return
    committed = commit_lens.detach().reshape(-1).cpu().tolist()
    block_size = int(worker.block_size)
    batch_size = len(batch.reqs)
    hidden_rows = target_hidden.reshape(batch_size, block_size, -1)
    position_rows = positions.reshape(batch_size, block_size)
    cache_rows = cache_locs.reshape(batch_size, block_size)
    candidate_rows = candidates.reshape(batch_size, block_size)
    pool = worker.draft_model_runner.token_to_kv_pool
    for index, req in selected_requests:
        try:
            count = int(committed[index])
            if count == 0:
                continue
            if not 0 < count <= block_size:
                raise ValueError(f"Invalid verify commit length: {count}")
            start = int(position_rows[index, 0].item())
            selected = _sample_positions(
                start, start + count, pool.page_size, recorder.max_tokens
            )
            if not selected:
                continue
            offsets = torch.tensor(
                [position - start for position in selected],
                dtype=torch.int64,
                device=positions.device,
            )
            selected_locs = cache_rows[index].index_select(0, offsets).to(torch.int64)
            tensors = {
                "positions": position_rows[index].index_select(0, offsets),
                "input_ids": candidate_rows[index].index_select(0, offsets),
                "cache_locs": selected_locs,
                "aux_hidden": hidden_rows[index].index_select(0, offsets),
            }
            layer_ids = _add_kv_tensors(pool, selected_locs, tensors)
            recorder.write(
                req,
                phase,
                {
                    **_debug_metadata(worker),
                    "page_size": pool.page_size,
                    "kv_layout": getattr(pool, "kv_cache_layout", "nhd"),
                    "kv_dtype": str(pool.dtype),
                    "layer_ids": layer_ids,
                    "capture_layers": _capture_layers(worker),
                    "capture_semantics": "zero_based_target_layer_output",
                    "verify_input_start": start,
                    "commit_length": count,
                    "includes_anchor": True,
                    "includes_new_bonus": False,
                    "cache_locs_are_peer_local": True,
                },
                tensors,
            )
        except Exception as error:
            _record_error(recorder, req, phase, error)
