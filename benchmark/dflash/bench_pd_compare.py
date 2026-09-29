"""Replay fixed tokenized prompts through a real P/D router, retaining metadata."""

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path

import aiohttp


async def run(args):
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    rows = rows[: args.num_prompts]
    if not rows or args.concurrency < 1:
        raise ValueError("Need prompts and positive concurrency")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:

        async def request(row, index, output_tokens):
            payload = {
                "rid": f"{args.tag}-{index}",
                "input_ids": row["input_ids"],
                "stream": True,
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": not args.allow_eos,
                },
                "return_spec_tokens_details": True,
            }
            async with semaphore:
                start = time.perf_counter()
                first = None
                first_count = 0
                last = start
                events = []
                data = {}
                error = None
                try:
                    async with session.post(
                        args.url + "/generate", json=payload
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(
                                f"HTTP {response.status}: {await response.text()}"
                            )
                        async for raw in response.content:
                            raw = raw.strip()
                            if not raw or raw == b"data: [DONE]":
                                continue
                            if not raw.startswith(b"data:"):
                                continue
                            data = json.loads(raw[5:])
                            now = time.perf_counter()
                            count = data.get("meta_info", {}).get(
                                "completion_tokens", 0
                            )
                            if count and (not events or count > events[-1][1]):
                                if first is None:
                                    first, first_count = now, count
                                last = now
                                events.append([now - start, count])
                except Exception as exc:
                    error = str(exc)
                end = time.perf_counter()
                meta = data.get("meta_info", {})
                count = meta.get("completion_tokens", 0)
                result = {
                    "index": index,
                    "source": row.get("source"),
                    "input_sha256": hashlib.sha256(
                        json.dumps(row["input_ids"]).encode()
                    ).hexdigest(),
                    "input_tokens": len(row["input_ids"]),
                    "output_tokens": count,
                    "first_chunk_tokens": first_count,
                    "latency_s": end - start,
                    "ttft_s": first - start if first is not None else None,
                    "tpot_s": (
                        (last - first) / (count - first_count)
                        if first is not None and count > first_count
                        else None
                    ),
                    "events": events,
                    "meta_info": meta,
                    "text": data.get("text", ""),
                    "error": error,
                }
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in result.items()
                            if k not in {"text", "events", "meta_info"}
                        }
                    ),
                    flush=True,
                )
                return result

        if args.warmup:
            warm = await asyncio.gather(
                *(
                    request(rows[i % len(rows)], f"warm-{i}", 32)
                    for i in range(args.warmup)
                )
            )
            if any(r["error"] or not r["output_tokens"] for r in warm):
                raise RuntimeError("Warmup failed")
        start = time.perf_counter()
        results = await asyncio.gather(
            *(request(row, i, args.output_tokens) for i, row in enumerate(rows))
        )
        elapsed = time.perf_counter() - start
    args.output.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    good = [r for r in results if r["error"] is None and r["output_tokens"]]
    tpots = [r["tpot_s"] * 1000 for r in good if r["tpot_s"] is not None]
    verify = sum(r["meta_info"].get("spec_verify_ct", 0) for r in good)
    summary = {
        "tag": args.tag,
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "concurrency": args.concurrency,
        "requested": len(rows),
        "succeeded": len(good),
        "elapsed_s": elapsed,
        "output_tokens": sum(r["output_tokens"] for r in good),
        "throughput_tokens_s": sum(r["output_tokens"] for r in good) / elapsed,
        "tpot_mean_ms": statistics.mean(tpots) if tpots else None,
        "tpot_median_ms": statistics.median(tpots) if tpots else None,
        "ttft_mean_ms": (
            statistics.mean(r["ttft_s"] * 1000 for r in good) if good else None
        ),
        "spec_verify_ct": verify,
        "weighted_accept_length": (
            sum(r["output_tokens"] for r in good) / verify if verify else None
        ),
        "temperature": 1.0,
        "top_p": 0.95,
        "ignore_eos": not args.allow_eos,
        "output_limit": args.output_tokens,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if len(good) != len(rows):
        raise RuntimeError("Some requests failed; results retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://10.41.101.127:30005")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--allow-eos", action="store_true")
    asyncio.run(run(parser.parse_args()))
