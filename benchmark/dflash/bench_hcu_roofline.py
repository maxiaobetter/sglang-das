"""Measure achieved memory/compute ceilings on one explicitly selected idle GPU."""

import argparse
import json
import statistics

import torch


def measure(fn, repeats):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values), values


def main(args):
    torch.cuda.set_device(0)
    print(
        json.dumps(
            {
                "device": str(torch.cuda.get_device_properties(0)),
                "torch": torch.__version__,
                "hip": torch.version.hip,
            }
        ),
        flush=True,
    )
    if args.layouts:
        torch.manual_seed(20260929)
        for m, n, k in [
            (7, 154880, 6144),
            (8, 24576, 6144),
            (8, 10240, 6144),
            (40, 154880, 6144),
        ]:
            a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda") * 0.01
            weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * 0.01
            column_ids = torch.linspace(0, n - 1, 64, device="cuda").long()
            reference = (
                a.cpu().float() @ weight.index_select(0, column_ids).cpu().float().T
            )
            for layout in ["NT", "NN_prepacked"]:
                b = weight.T if layout == "NT" else weight.T.contiguous()
                out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
                ms, samples = measure(
                    lambda a=a, b=b, out=out: torch.mm(a, b, out=out), 30
                )
                actual = out.index_select(1, column_ids).cpu().float()
                print(
                    json.dumps(
                        {
                            "op": "mm_bf16",
                            "layout": layout,
                            "mnk": [m, n, k],
                            "median_ms": ms,
                            "minimum_bytes_GB_s": (m * k + n * k + m * n)
                            * 2
                            / (ms * 1e6),
                            "TFLOP_s": 2 * m * n * k / (ms * 1e9),
                            "max_abs_error": (actual - reference).abs().max().item(),
                            "relative_l2_error": (
                                (actual - reference).norm() / reference.norm()
                            ).item(),
                            "samples_ms": samples,
                        }
                    ),
                    flush=True,
                )
                del b, out
            del a, weight
        return
    for mib in ([256] if args.counters else [256, 1024]):
        n = mib * 1024 * 1024 // 4
        x = torch.ones(n, device="cuda")
        y = torch.ones_like(x)
        z = torch.empty_like(x)
        ms, samples = measure(
            lambda x=x, y=y, z=z: torch.add(x, y, out=z), 1 if args.counters else 20
        )
        print(
            json.dumps(
                {
                    "op": "add_f32",
                    "bytes": n * 12,
                    "median_ms": ms,
                    "logical_GB_s": n * 12 / (ms * 1e6),
                    "samples_ms": samples,
                    "valid": z[0].item() == 2.0 and z[-1].item() == 2.0,
                }
            ),
            flush=True,
        )
        del x, y, z
    if args.counters:
        return
    for m, n, k in [
        (4096, 4096, 4096),
        (8192, 6144, 6144),
        (8, 154880, 6144),
        (40, 154880, 6144),
    ]:
        a = torch.full((m, k), 0.01, dtype=torch.bfloat16, device="cuda")
        b = torch.full((n, k), 0.01, dtype=torch.bfloat16, device="cuda")
        out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        ms, samples = measure(lambda a=a, b=b, out=out: torch.mm(a, b.T, out=out), 20)
        print(
            json.dumps(
                {
                    "op": "mm_bf16_nt",
                    "mnk": [m, n, k],
                    "median_ms": ms,
                    "TFLOP_s": 2 * m * n * k / (ms * 1e9),
                    "minimum_bytes_GB_s": (m * k + n * k + m * n) * 2 / (ms * 1e6),
                    "samples_ms": samples,
                    "finite": bool(torch.isfinite(out).all().item()),
                }
            ),
            flush=True,
        )
        del a, b, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--layouts", action="store_true")
    main(parser.parse_args())
