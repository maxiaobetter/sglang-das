"""Run a bounded, sequential P/D benchmark matrix on an already running service."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests


def main(args):
    root = args.root
    result_dir = root / args.algorithm
    result_dir.mkdir(parents=True, exist_ok=True)
    client = root / "bench_pd_compare.py"
    jobs = [
        ("mixed_short", "short_c16_r1", 16, 128),
        ("mixed_short", "profile_short_c16", 16, 16),
        ("mixed_short", "short_c64_r1", 64, 128),
        ("mixed_short", "short_c1_r1", 1, 12),
        ("long_8k", "long_c16_r1", 16, 128),
        ("long_8k", "long_c64_r1", 64, 128),
        ("long_8k", "profile_long_c64", 64, 64),
        ("mixed_short", "short_c16_r2", 16, 128),
        ("long_8k", "long_c16_r2", 16, 128),
    ]
    if args.wait_for:
        deadline = time.monotonic() + 1800
        while not args.wait_for.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(args.wait_for)
            time.sleep(5)
    for dataset, tag, concurrency, count in jobs:
        if args.skip_profile and tag.startswith("profile"):
            continue
        output = result_dir / f"{tag}.jsonl"
        summary = output.with_suffix(".summary.json")
        if summary.exists():
            saved = json.loads(summary.read_text())
            if saved["succeeded"] != count:
                raise RuntimeError(f"Incomplete prior run: {summary}")
            print(f"Already complete: {tag}", flush=True)
            continue
        command = [
            sys.executable,
            str(client),
            "--dataset",
            str(root / f"{dataset}.jsonl"),
            "--output",
            str(output),
            "--tag",
            f"{args.algorithm}-{tag}",
            "--concurrency",
            str(concurrency),
            "--num-prompts",
            str(count),
            "--output-tokens",
            "512",
            "--warmup",
            "0" if tag.startswith("profile") else str(min(concurrency, 16)),
        ]
        print(f"Starting {tag}: {command}", flush=True)
        with output.with_suffix(".log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            if tag.startswith("profile"):
                # Profile only a dedicated run, never the throughput measurements.
                time.sleep(5)
                body = {
                    "output_dir": str(result_dir / "profiles"),
                    "num_steps": 8,
                    "activities": ["CPU", "GPU"],
                    "record_shapes": True,
                    "with_stack": False,
                    "profile_prefix": tag,
                    "detailed_annotations": True,
                }
                response = requests.post(
                    "http://10.41.101.127:30026/start_profile", json=body, timeout=180
                )
                output.with_suffix(".profile.json").write_text(
                    json.dumps(
                        {
                            "request": body,
                            "status": response.status_code,
                            "response": response.text,
                        },
                        indent=2,
                    )
                )
                response.raise_for_status()
            code = process.wait(timeout=1800)
        if code:
            raise RuntimeError(f"{tag} failed with code {code}")
        print(summary.read_text(), flush=True)
    (result_dir / "matrix_done.json").write_text(json.dumps({"completed": time.time()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--skip-profile", action="store_true")
    main(parser.parse_args())
