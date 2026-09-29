"""Opt-in, bounded DFlash diagnostics. Payloads contain request data and tensors.

This module never changes sampling or draws random numbers. Recording is
synchronous and intended for short correctness investigations, not benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _plain(value):
    """Keep metadata loadable with torch.load(weights_only=True)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        raise TypeError("Put diagnostic tensors in tensors, not metadata")
    return str(value)


class DFlashDebugRecorder:
    def __init__(self):
        self.directory = Path(envs.SGLANG_DFLASH_DEBUG_DIR.get()).expanduser()
        self.rid_prefix = envs.SGLANG_DFLASH_DEBUG_RID_PREFIX.get()
        self.max_requests = envs.SGLANG_DFLASH_DEBUG_MAX_REQUESTS.get()
        self.max_steps = envs.SGLANG_DFLASH_DEBUG_MAX_STEPS.get()
        self.max_tokens = envs.SGLANG_DFLASH_DEBUG_MAX_TOKENS.get()
        if min(self.max_requests, self.max_steps, self.max_tokens) <= 0:
            raise ValueError("DFlash debug limits must all be positive")
        self.disabled = False
        self._requests = set()
        self._steps = {}
        self._request_metadata = {}
        self._lock = threading.Lock()
        self._instance = uuid.uuid4().hex
        self._hostname = socket.gethostname()

    def _disable(self, exc: Exception) -> None:
        with self._lock:
            first_failure = not self.disabled
            self.disabled = True
        if first_failure:
            logger.warning(
                "DFlash diagnostics disabled after recording failure: %s", exc
            )

    def disable(self, error_or_message) -> None:
        """Disable diagnostics after a caller-side observation failure."""
        self._disable(error_or_message)

    def _eligible(self, rid: str, phase: str) -> bool:
        return (
            not self.disabled
            and rid.startswith(self.rid_prefix)
            and (rid in self._requests or len(self._requests) < self.max_requests)
            and self._steps.get((rid, phase), 0) < self.max_steps
        )

    def enabled_for(self, req, phase: str) -> bool:
        """Check the request/phase budget without reserving or consuming it."""
        try:
            if self.disabled or _capturing():
                return False
            with self._lock:
                return self._eligible(str(req.rid), phase)
        except Exception as exc:
            self._disable(exc)
            return False

    def _request_info(self, req) -> dict:
        rid = str(req.rid)
        cached = self._request_metadata.get(rid)
        if cached is not None:
            return cached
        token_ids = getattr(req, "origin_input_ids", None)
        if token_ids is not None:
            token_ids = [int(token) for token in token_ids]
        info = {
            "rid": rid,
            "bootstrap_room": _plain(getattr(req, "bootstrap_room", None)),
            "prompt_length": len(token_ids) if token_ids is not None else None,
            "input_ids_sha256": (
                hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode()
                ).hexdigest()
                if token_ids is not None
                else None
            ),
        }
        self._request_metadata[rid] = info
        return info

    @staticmethod
    def _cpu_tensors(tensors: dict) -> dict:
        # Copy only supplied slices. One event per device waits for these copies;
        # avoid a device-wide synchronize and avoid a blocking copy per tensor.
        copied = {}
        streams = {}
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Diagnostic tensor {name!r} is not a Tensor")
            source = tensor.detach()
            if source.device.type == "cuda":
                host = torch.empty_like(source, device="cpu", pin_memory=True)
                host.copy_(source, non_blocking=True)
                streams[source.device] = torch.cuda.current_stream(source.device)
                copied[name] = host
            else:
                copied[name] = source.to(device="cpu", copy=True)
        for stream in streams.values():
            event = torch.cuda.Event()
            event.record(stream)
            event.synchronize()
        return copied

    def write(self, req, phase: str, metadata: dict, tensors: dict) -> None:
        """Consume one budget slot and write an independent CPU-only .pt file.

        Failures disable subsequent diagnostics in this process and never escape
        to the inference caller. No directory is created until the first write.
        """
        try:
            if self.disabled or _capturing():
                return
            rid = str(req.rid)
            with self._lock:
                if not self._eligible(rid, phase):
                    return
                self._requests.add(rid)
                step = self._steps.get((rid, phase), 0)
                self._steps[(rid, phase)] = step + 1
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else None
            )
            try:
                from sglang.srt.runtime_context import get_disagg

                role = get_disagg().disaggregation_mode
            except (AttributeError, RuntimeError):
                role = None
            info = _plain(metadata)
            info.update(self._request_info(req))
            info.update(
                phase=phase,
                step=step,
                hostname=self._hostname,
                pid=os.getpid(),
                dist_rank=rank,
                role=_plain(role),
            )
            payload = {
                "format_version": 1,
                "metadata": info,
                "tensors": self._cpu_tensors(tensors),
            }
            identity = json.dumps(
                [self._instance, self._hostname, os.getpid(), rank, rid, phase, step]
            )
            filename = hashlib.sha256(identity.encode()).hexdigest() + ".pt"
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / filename
            created = False
            try:
                with path.open("xb") as output:
                    created = True
                    torch.save(payload, output)
            except Exception:
                if created:
                    path.unlink(missing_ok=True)
                raise
        except Exception as exc:
            self._disable(exc)

    def record_sampling(
        self,
        *,
        batch,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        q_rows: torch.Tensor,
        accept_len: torch.Tensor,
        bonus: torch.Tensor,
        prefix_lens: torch.Tensor,
        draft_input,
        target_probs: torch.Tensor | None = None,
        verify_uniforms: torch.Tensor | None = None,
        final_uniforms: torch.Tensor | None = None,
        actual_q_at_candidates: torch.Tensor | None = None,
        actual_selected_q: torch.Tensor | None = None,
        actual_q_sum: torch.Tensor | None = None,
        context: dict | None = None,
    ) -> None:
        """Record selected rows after real acceptance, without changing its inputs.

        Prefer p and q gathered from the actual acceptance call. Without that
        observer, p is explicitly labeled a reconstruction, not kernel evidence.
        """
        try:
            if self.disabled or _capturing():
                return
            with torch.no_grad():
                self._record_sampling(
                    batch=batch,
                    candidates=candidates,
                    target_logits=target_logits,
                    candidate_ids=candidate_ids,
                    q_rows=q_rows,
                    accept_len=accept_len,
                    bonus=bonus,
                    prefix_lens=prefix_lens,
                    draft_input=draft_input,
                    target_probs=target_probs,
                    verify_uniforms=verify_uniforms,
                    final_uniforms=final_uniforms,
                    actual_q_at_candidates=actual_q_at_candidates,
                    actual_selected_q=actual_selected_q,
                    actual_q_sum=actual_q_sum,
                    context=context,
                )
        except Exception as exc:
            self._disable(exc)

    def _record_sampling(
        self,
        *,
        batch,
        candidates,
        target_logits,
        candidate_ids,
        q_rows,
        accept_len,
        bonus,
        prefix_lens,
        draft_input,
        target_probs,
        verify_uniforms,
        final_uniforms,
        actual_q_at_candidates,
        actual_selected_q,
        actual_q_sum,
        context,
    ):
        from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs

        bs, block = candidates.shape
        gamma = block - 1
        slots = min(gamma, self.max_tokens)
        if slots <= 0:
            return
        sampling = batch.sampling_info
        logits = target_logits.reshape(bs, block, -1)
        for index, req in enumerate(batch.reqs):
            if not self.enabled_for(req, "sampling"):
                continue
            sliced = SimpleNamespace(
                temperatures=sampling.temperatures[index : index + 1].detach().clone(),
                top_ps=sampling.top_ps[index : index + 1].detach().clone(),
                top_ks=sampling.top_ks[index : index + 1].detach().clone(),
                need_top_k_sampling=sampling.need_top_k_sampling,
                need_top_p_sampling=sampling.need_top_p_sampling,
            )
            actual_p = target_probs is not None
            if actual_p:
                p = target_probs.reshape(bs, block, -1)[index]
            else:
                # Preserve batch-level top-k hints: they select the same sparse
                # or dense path as the real verifier even for a one-request slice.
                p = build_dflash_verify_target_probs(
                    next_token_logits=logits[index].detach().clone(),
                    sampling_info=sliced,
                    draft_token_num=block,
                    bs=1,
                    max_top_k=draft_input.max_top_k,
                    uniform_top_k_value=draft_input.uniform_top_k_value,
                )[0]
            ids = candidate_ids[index, :slots].to(torch.int64)
            q = (
                actual_q_at_candidates[index, :slots]
                if actual_q_at_candidates is not None
                else q_rows[index, :slots]
            ).float()
            p_candidates = p[:slots].gather(-1, ids)
            selected = candidates[index, 1 : slots + 1].to(torch.int64)
            selected_p = p[:slots].gather(-1, selected[:, None]).squeeze(-1)
            selected_q = (
                actual_selected_q[index, :slots]
                if actual_selected_q is not None
                else torch.where(ids == selected[:, None], q, 0.0).amax(-1)
            )
            accepted_count = accept_len[index : index + 1].to(torch.int64)
            slot_ids = torch.arange(slots, device=ids.device)
            min_ps = sampling.min_ps[index : index + 1]
            tensors = {
                "candidate_ids": ids,
                "q_rows": q,
                "proposal_q_rows": q_rows[index, :slots],
                "p_candidates": p_candidates,
                "selected_token_ids": selected,
                "selected_p": selected_p,
                "selected_q": selected_q,
                "selected_accept_prob": torch.where(
                    selected_q > 0,
                    (selected_p / selected_q).clamp(max=1.0),
                    (selected_p > 0).to(selected_q.dtype),
                ),
                "Cmass": p_candidates.sum(-1),
                "alpha": torch.minimum(p_candidates, q).sum(-1),
                "tail_q": torch.where(p_candidates == 0, q, 0.0).sum(-1),
                "q_sum": q.sum(-1),
                "p_sum": p[:slots].sum(-1),
                "candidate_ids_unique": (ids.sort(dim=-1).values.diff(dim=-1) != 0).all(
                    -1
                ),
                "selected_candidate_matches": (ids == selected[:, None]).sum(-1),
                "reached": slot_ids <= accepted_count,
                "accepted": slot_ids < accepted_count,
                "accept_len": accepted_count,
                "bonus": bonus[index : index + 1],
                "target_top1": p[:slots].argmax(-1),
                "proposal_positions": prefix_lens[index] + slot_ids + 1,
                "target_logit_positions": prefix_lens[index] + slot_ids,
                "prefix_len": prefix_lens[index : index + 1],
                "temperature": sliced.temperatures,
                "top_p": sliced.top_ps,
                "top_k": sliced.top_ks,
                "min_p": min_ps,
                # Logits already include penalties and grammar masks. This one
                # vocab row supports offline top-p A/B at an identical prefix.
                "first_target_logits": logits[index, 0],
            }
            if actual_q_sum is not None:
                tensors["actual_dense_q_sum"] = actual_q_sum[index, :slots]
            if verify_uniforms is not None:
                uniforms = verify_uniforms[index, :slots]
                expected = uniforms * selected_q < selected_p
                tensors["verify_uniforms"] = uniforms
                tensors["expected_accept"] = expected
                tensors["expected_prefix_accept_len"] = (
                    expected.to(torch.int64).cumprod(dim=0).sum().reshape(1)
                )
            if final_uniforms is not None:
                tensors["final_uniform"] = final_uniforms[index : index + 1]
                final_row = accepted_count.clamp(min=0, max=block - 1)
                q_row = accepted_count.clamp(min=0, max=gamma - 1)
                tensors["bonus_target_probs"] = p.index_select(0, final_row).squeeze(0)
                tensors["bonus_candidate_ids"] = (
                    candidate_ids[index].index_select(0, q_row).squeeze(0)
                )
                full_q = (
                    actual_q_at_candidates[index]
                    if actual_q_at_candidates is not None
                    else q_rows[index]
                )
                tensors["bonus_q_rows"] = full_q.index_select(0, q_row).squeeze(0)
                tensors["bonus_uses_p_only"] = accepted_count == gamma
            metadata = dict(context or {})
            metadata.update(
                p_source=(
                    "actual_kernel_input" if actual_p else "reconstructed_after_accept"
                ),
                q_source=(
                    "actual_kernel_input"
                    if actual_q_at_candidates is not None
                    else "selector_proposal"
                ),
                p_note=(
                    "Actual target probability tensor passed to the chain verifier"
                    if actual_p
                    else "Reconstructed after accept; not the actual kernel input"
                ),
                reconstructed_p_complete=False if not actual_p else None,
                block_size=block,
                proposal_slots=gamma,
                recorded_slots=slots,
                expected_prefix_is_capped=slots < gamma,
                need_top_k_sampling=bool(sampling.need_top_k_sampling),
                need_top_p_sampling=bool(sampling.need_top_p_sampling),
                need_min_p_sampling=bool(sampling.need_min_p_sampling),
                has_grammar=bool(getattr(req, "grammar", None) is not None),
                has_custom_logit_processor=bool(sampling.has_custom_logit_processor),
                logits_include_penalties_and_grammar=True,
                min_p_note="The current selector verifier does not apply min_p",
                batch_max_top_k=draft_input.max_top_k,
                batch_uniform_top_k_value=draft_input.uniform_top_k_value,
                candidate_metrics_note="Cmass/alpha/tail_q require unique candidate ids",
            )
            self.write(req, "sampling", metadata, tensors)


_recorder = None
_initialization_failed = False


def get_dflash_debug_recorder() -> DFlashDebugRecorder | None:
    """Return the process-local recorder only when explicitly enabled."""
    global _recorder, _initialization_failed
    if _initialization_failed:
        return None
    try:
        if not envs.SGLANG_DFLASH_DEBUG_DIR.get():
            return None
        if _recorder is None:
            _recorder = DFlashDebugRecorder()
        return None if _recorder.disabled else _recorder
    except Exception as exc:
        _initialization_failed = True
        logger.warning("DFlash diagnostics initialization failed; disabled: %s", exc)
        return None
