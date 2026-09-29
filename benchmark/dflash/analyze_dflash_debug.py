"""Inspect bounded DFlash debug dumps on CPU without loading model weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Record:
    path: Path
    metadata: dict
    tensors: dict

    @property
    def phase(self):
        return self.metadata.get("phase")


def load_records(roots):
    records, errors = [], []
    paths = set()
    for root in roots:
        if not root.is_dir():
            errors.append(f"Not a debug directory: {root}")
            continue
        paths.update(path.resolve() for path in root.rglob("*.pt"))
    for path in sorted(paths):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("format_version") != 1:
                raise ValueError("unsupported debug format_version")
            metadata, tensors = payload["metadata"], payload["tensors"]
            if not isinstance(metadata, dict) or not isinstance(tensors, dict):
                raise ValueError("metadata and tensors must be dictionaries")
            if metadata.get("snapshot_error"):
                raise ValueError(f"snapshot failed: {metadata['snapshot_error']}")
            if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
                raise ValueError("tensors must contain only tensors")
            records.append(Record(path, metadata, tensors))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    return records, errors


def request_identity(record):
    metadata = record.metadata
    room, rid = metadata.get("bootstrap_room"), metadata.get("rid")
    if room is None and not rid:
        raise ValueError("missing bootstrap_room and rid")
    return str(room) if room is not None else None, rid


def sampling_summary(records, errors):
    groups = {}
    seen = set()
    for record in records:
        if record.phase != "sampling":
            continue
        try:
            room, rid = request_identity(record)
            source = record.metadata.get("p_source", "unknown")
            # Keep different ranks/processes separate rather than silently merging
            # duplicate requests or counting a replicated observation twice.
            process = (
                record.metadata.get("hostname"),
                record.metadata.get("pid"),
                record.metadata.get("dist_rank"),
            )
            key = (room, rid, source, process)
            step = record.metadata["step"]
            if (key, step) in seen:
                raise ValueError("duplicate request/process/step sampling record")
            seen.add((key, step))
            tensors = record.tensors
            reached = tensors["reached"].reshape(-1).bool()
            accepted = tensors["accepted"].reshape(-1).bool()
            p = tensors["p_candidates"].double()
            q = tensors["q_rows"].double()
            ids = tensors["candidate_ids"]
            if (
                p.ndim != 2
                or p.shape != q.shape
                or ids.shape != p.shape
                or reached.numel() != p.shape[0]
                or accepted.shape != reached.shape
            ):
                raise ValueError("sampling shapes do not align")
            if (accepted & ~reached).any():
                raise ValueError("invalid accepted/unreached mask")
            unique_rows = [torch.unique(row).numel() == row.numel() for row in ids]
            finite_rows = torch.isfinite(p).all(-1) & torch.isfinite(q).all(-1)
            for name in ("p_sum", "actual_dense_q_sum"):
                if name in tensors:
                    mass = tensors[name].reshape(-1)
                    if mass.shape != reached.shape:
                        raise ValueError(f"{name} does not align with proposal slots")
                    finite_rows &= torch.isfinite(mass)
            nonnegative_rows = (p >= 0).all(-1) & (q >= 0).all(-1)
            group = groups.setdefault(
                key,
                {
                    "bootstrap_room": room,
                    "rid": rid,
                    "p_source": source,
                    "q_source": record.metadata.get("q_source", "unknown"),
                    "process": process,
                    "steps": 0,
                    "p_notes": [],
                    "positions": {},
                    "prefix_accept_mismatches": 0,
                    "prefix_accept_checks": 0,
                },
            )
            group["steps"] += 1
            note = record.metadata.get("p_note")
            if note and note not in group["p_notes"]:
                group["p_notes"].append(note)
            if "expected_prefix_accept_len" in tensors:
                group["prefix_accept_checks"] += 1
                actual_prefix = tensors["accept_len"].item()
                if record.metadata.get("expected_prefix_is_capped"):
                    actual_prefix = min(actual_prefix, reached.numel())
                group["prefix_accept_mismatches"] += int(
                    tensors["expected_prefix_accept_len"].item() != actual_prefix
                )
            for position in range(reached.numel()):
                if not reached[position]:
                    continue
                metrics = group["positions"].setdefault(
                    position + 1,
                    dict(
                        reached=0,
                        accepted=0,
                        theory_rows=0,
                        invalid_candidate_rows=0,
                        invalid_probability_rows=0,
                        finite_probability_rows=0,
                        alpha=0.0,
                        Cmass=0.0,
                        tail_q=0.0,
                        q_sum=0.0,
                        p_sum_min=None,
                        p_sum_max=None,
                        q_sum_min=None,
                        q_sum_max=None,
                        dense_q_sum=0.0,
                        dense_q_checks=0,
                        max_candidate_dense_q_gap=0.0,
                        kernel_decision_checks=0,
                        kernel_decision_mismatches=0,
                    ),
                )
                metrics["reached"] += 1
                metrics["accepted"] += int(accepted[position])
                candidate_q_sum = q[position].sum().item()
                if finite_rows[position]:
                    metrics["finite_probability_rows"] += 1
                    metrics["q_sum"] += candidate_q_sum
                    metrics["q_sum_min"] = (
                        candidate_q_sum
                        if metrics["q_sum_min"] is None
                        else min(metrics["q_sum_min"], candidate_q_sum)
                    )
                    metrics["q_sum_max"] = (
                        candidate_q_sum
                        if metrics["q_sum_max"] is None
                        else max(metrics["q_sum_max"], candidate_q_sum)
                    )
                if "p_sum" in tensors and torch.isfinite(tensors["p_sum"][position]):
                    p_sum = tensors["p_sum"][position].item()
                    metrics["p_sum_min"] = (
                        p_sum
                        if metrics["p_sum_min"] is None
                        else min(metrics["p_sum_min"], p_sum)
                    )
                    metrics["p_sum_max"] = (
                        p_sum
                        if metrics["p_sum_max"] is None
                        else max(metrics["p_sum_max"], p_sum)
                    )
                if (
                    unique_rows[position]
                    and finite_rows[position]
                    and nonnegative_rows[position]
                ):
                    metrics["theory_rows"] += 1
                    metrics["alpha"] += (
                        torch.minimum(p[position], q[position]).sum().item()
                    )
                    metrics["Cmass"] += p[position].sum().item()
                    metrics["tail_q"] += q[position][p[position] == 0].sum().item()
                if not unique_rows[position]:
                    metrics["invalid_candidate_rows"] += 1
                    errors.append(
                        f"{record.path}: duplicate candidate IDs at position {position + 1}; excluded theoretical metrics"
                    )
                if not finite_rows[position] or not nonnegative_rows[position]:
                    metrics["invalid_probability_rows"] += 1
                    errors.append(
                        f"{record.path}: nonfinite/negative probability at position {position + 1}; excluded theoretical metrics"
                    )
                if "actual_dense_q_sum" in tensors and torch.isfinite(
                    tensors["actual_dense_q_sum"][position]
                ):
                    dense_sum = tensors["actual_dense_q_sum"][position].item()
                    metrics["dense_q_checks"] += 1
                    metrics["dense_q_sum"] += dense_sum
                    if finite_rows[position]:
                        metrics["max_candidate_dense_q_gap"] = max(
                            metrics["max_candidate_dense_q_gap"],
                            abs(candidate_q_sum - dense_sum),
                        )
                if "expected_accept" in tensors:
                    metrics["kernel_decision_checks"] += 1
                    metrics["kernel_decision_mismatches"] += int(
                        bool(tensors["expected_accept"][position])
                        != bool(accepted[position])
                    )
        except Exception as exc:
            errors.append(f"{record.path}: sampling: {type(exc).__name__}: {exc}")
    for group in groups.values():
        for metrics in group["positions"].values():
            count = metrics["reached"]
            metrics["actual_accept_rate"] = metrics["accepted"] / count
            for metric in ("alpha", "Cmass", "tail_q"):
                metrics[metric] = (
                    metrics[metric] / metrics["theory_rows"]
                    if metrics["theory_rows"]
                    else None
                )
            metrics["q_sum"] = (
                metrics["q_sum"] / metrics["finite_probability_rows"]
                if metrics["finite_probability_rows"]
                else None
            )
            metrics["dense_q_sum"] = (
                metrics["dense_q_sum"] / metrics["dense_q_checks"]
                if metrics["dense_q_checks"]
                else None
            )
    return list(groups.values())


def tensor_difference(left, right):
    if left.shape != right.shape:
        raise ValueError(f"tensor shape mismatch: {left.shape} vs {right.shape}")
    delta = (left.double() - right.double()).abs()
    finite = torch.isfinite(delta)
    all_finite = bool(finite.all())
    return {
        "exact": bool(torch.equal(left, right)),
        "dtype_equal": left.dtype == right.dtype,
        "elements": left.numel(),
        "nonfinite_differences": int((~finite).sum()),
        "max_abs": delta.max().item() if all_finite and delta.numel() else None,
        "mean_abs": delta.mean().item() if all_finite and delta.numel() else None,
    }


def token_rows(record):
    positions = record.tensors["positions"].reshape(-1).tolist()
    input_ids = record.tensors["input_ids"].reshape(-1).tolist()
    if len(positions) != len(input_ids) or len(set(positions)) != len(positions):
        raise ValueError("positions must be unique and align with input_ids")
    return {
        int(pos): (index, token)
        for index, (pos, token) in enumerate(zip(positions, input_ids))
    }


def aligned_rows(left, right):
    left_rows, right_rows = token_rows(left), token_rows(right)
    shared = sorted(left_rows.keys() & right_rows.keys())
    mismatched = [pos for pos in shared if left_rows[pos][1] != right_rows[pos][1]]
    if mismatched:
        raise ValueError(f"input token mismatch at logical positions {mismatched}")
    return (
        shared,
        [left_rows[pos][0] for pos in shared],
        [right_rows[pos][0] for pos in shared],
    )


def compare_records(left, right, *, auxiliary):
    fingerprint = left.metadata.get("input_ids_sha256")
    if not fingerprint or fingerprint != right.metadata.get("input_ids_sha256"):
        raise ValueError("complete prompt input_ids_sha256 is missing or differs")
    positions, left_rows, right_rows = aligned_rows(left, right)
    if not positions:
        return []
    if auxiliary:
        left_layers = left.metadata["capture_layers"]
        right_layers = right.metadata["capture_layers"]
        left_hidden = left.tensors["aux_hidden"]
        right_hidden = right.tensors["aux_hidden"]
        if (
            left_hidden.ndim != 2
            or right_hidden.ndim != 2
            or not left_layers
            or not right_layers
            or left_hidden.shape[1] % len(left_layers)
            or right_hidden.shape[1] % len(right_layers)
        ):
            raise ValueError("aux_hidden does not match capture_layers")
        left_values = dict(zip(left_layers, left_hidden.chunk(len(left_layers), dim=1)))
        right_values = dict(
            zip(right_layers, right_hidden.chunk(len(right_layers), dim=1))
        )
    else:
        left_values = {
            key: value
            for key, value in left.tensors.items()
            if key.startswith("layer_")
        }
        right_values = {
            key: value
            for key, value in right.tensors.items()
            if key.startswith("layer_")
        }
    if not left_values or set(left_values) != set(right_values):
        raise ValueError("layer names differ or no layer tensors were recorded")
    results = []
    for layer in left_values:
        result = tensor_difference(
            left_values[layer][left_rows], right_values[layer][right_rows]
        )
        result.update(
            layer=layer,
            positions=positions,
            left_file=str(left.path),
            right_file=str(right.path),
            left_rank=left.metadata.get("dist_rank"),
            right_rank=right.metadata.get("dist_rank"),
            left_dtype=str(left_values[layer].dtype),
            right_dtype=str(right_values[layer].dtype),
        )
        results.append(result)
    return results


def kv_comparisons(records, errors):
    prefill, decode = defaultdict(list), defaultdict(list)
    for record in records:
        if record.phase not in ("prefill_kv", "decode_kv"):
            continue
        try:
            key = request_identity(record)
            (prefill if record.phase == "prefill_kv" else decode)[key].append(record)
        except Exception as exc:
            errors.append(f"{record.path}: KV identity: {exc}")
    results, no_overlap = [], 0
    for key in prefill.keys() & decode.keys():
        for left in prefill[key]:
            for right in decode[key]:
                try:
                    pairs = compare_records(left, right, auxiliary=False)
                    results.extend(pairs)
                    no_overlap += int(not pairs)
                except Exception as exc:
                    errors.append(f"{left.path} vs {right.path}: KV: {exc}")
    return {
        "layer_pairs": results,
        "prefill_requests_without_decode": len(prefill.keys() - decode.keys()),
        "decode_requests_without_prefill": len(decode.keys() - prefill.keys()),
        "record_pairs_without_shared_positions": no_overlap,
    }


def aux_comparisons(records, other_records, errors):
    groups = []
    for dataset in (records, other_records):
        grouped = defaultdict(list)
        for record in dataset:
            if record.phase != "prefill_aux":
                continue
            fingerprint = record.metadata.get("input_ids_sha256")
            if not fingerprint:
                errors.append(
                    f"{record.path}: cannot compare aux without input_ids_sha256"
                )
                continue
            grouped[fingerprint].append(record)
        groups.append(grouped)
    left_group, right_group = groups
    results, no_overlap = [], 0
    for fingerprint in left_group.keys() & right_group.keys():
        for left in left_group[fingerprint]:
            for right in right_group[fingerprint]:
                try:
                    pairs = compare_records(left, right, auxiliary=True)
                    no_overlap += int(not pairs)
                    for pair in pairs:
                        pair["input_ids_sha256"] = fingerprint
                    results.extend(pairs)
                except Exception as exc:
                    errors.append(f"{left.path} vs {right.path}: aux: {exc}")
    return {
        "layer_pairs": results,
        "left_unmatched_input_hashes": len(left_group.keys() - right_group.keys()),
        "right_unmatched_input_hashes": len(right_group.keys() - left_group.keys()),
        "record_pairs_without_shared_positions": no_overlap,
    }


def print_comparisons(title, comparison):
    pairs = comparison["layer_pairs"]
    exact = sum(pair["exact"] and pair["dtype_equal"] for pair in pairs)
    print(f"\n{title}: {len(pairs)} layer pairs, {exact} exact with same dtype")
    for key, value in comparison.items():
        if key != "layer_pairs":
            print(f"  {key}={value}")
    for pair in pairs:
        print(
            f"  layer={pair['layer']} ranks={pair['left_rank']}->{pair['right_rank']} "
            f"tokens={len(pair['positions'])} exact={pair['exact']} "
            f"max_abs={pair['max_abs']} mean_abs={pair['mean_abs']} "
            f"nonfinite={pair['nonfinite_differences']}"
        )


def metric_text(value):
    return f"{value:.6f}" if value is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories", nargs="+", type=Path, help="P/D dump directories from one run"
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        type=Path,
        help="Other run directory for same-input prefill aux comparison; repeatable",
    )
    parser.add_argument(
        "--json", type=Path, help="Write complete report, including compared file paths"
    )
    args = parser.parse_args()
    records, errors = load_records(args.directories)
    other_records, other_errors = load_records(args.compare)
    errors.extend(other_errors)
    report = {
        "record_count": len(records),
        "phase_counts": dict(Counter(record.phase for record in records)),
        "comparison_record_count": len(other_records),
        "sampling": sampling_summary(records, errors),
        "draft_kv": kv_comparisons(records, errors),
        "prefill_aux": (
            aux_comparisons(records, other_records, errors) if args.compare else None
        ),
        "errors": errors,
    }
    print(
        f"Loaded {len(records)} records; comparison run: {len(other_records)} records"
    )
    print(f"Phases: {report['phase_counts']}")
    print(
        "Acceptance is descriptive only: bounded diagnostic samples do not establish statistical significance."
    )
    print("alpha=sum(min(p_C,q)); Cmass=sum(p_C); tail_q=sum(q where p_C==0).")
    print(
        "Only p_source=actual_kernel_input describes the verifier's actual p; all other sources are references."
    )
    for group in report["sampling"]:
        print(
            f"\nrequest={group['rid']} room={group['bootstrap_room']} "
            f"process={group['process']} steps={group['steps']} p_source={group['p_source']}"
        )
        print(f"  q_source={group['q_source']}")
        for note in group["p_notes"]:
            print(f"  note: {note}")
        print(
            "  draft_position reached accepted actual_rate theory_rows alpha Cmass tail_q q_sum decision_mismatches/checks"
        )
        for position, metrics in sorted(group["positions"].items()):
            print(
                f"  {position:14d} {metrics['reached']:7d} {metrics['accepted']:8d} "
                f"{metrics['actual_accept_rate']:.6f} {metrics['theory_rows']:11d} "
                f"{metric_text(metrics['alpha'])} {metric_text(metrics['Cmass'])} "
                f"{metric_text(metrics['tail_q'])} {metric_text(metrics['q_sum'])} "
                f"{metrics['kernel_decision_mismatches']}/{metrics['kernel_decision_checks']}"
            )
            if metrics["dense_q_checks"]:
                print(
                    f"    actual_dense_q_sum={metric_text(metrics['dense_q_sum'])} "
                    f"max_candidate_dense_q_gap={metrics['max_candidate_dense_q_gap']:.6g} "
                    f"invalid_candidate_rows={metrics['invalid_candidate_rows']}"
                )
            print(
                f"    p_sum_range=[{metric_text(metrics['p_sum_min'])}, {metric_text(metrics['p_sum_max'])}] "
                f"q_sum_range=[{metric_text(metrics['q_sum_min'])}, {metric_text(metrics['q_sum_max'])}] "
                f"invalid_probability_rows={metrics['invalid_probability_rows']} "
                f"invalid_candidate_rows={metrics['invalid_candidate_rows']}"
            )
        print(
            f"  prefix_accept_mismatches/checks="
            f"{group['prefix_accept_mismatches']}/{group['prefix_accept_checks']}"
        )
    print_comparisons("P/D draft KV", report["draft_kv"])
    if report["prefill_aux"] is not None:
        print_comparisons("Same-input cross-run prefill aux", report["prefill_aux"])
    if not records:
        errors.append("No supported debug records were loaded")
    for error in errors:
        print(f"WARNING: {error}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        print(f"\nReport: {args.json}")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
