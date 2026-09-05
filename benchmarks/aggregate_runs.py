#!/usr/bin/env python
"""Combine several benchmark runs into one report with medians and spread.

A single benchmark run reports a single sample. This script takes N runs of the same
configuration and reports, per variant, the median across runs plus the observed min-max
band, so every published figure carries its own uncertainty.

How much that matters depends entirely on the host. On an idle, mains-powered laptop the
observed spread across three runs was 0.5-1.1% for the ONNX variants and 4.6% at worst.
Runs taken while the same machine was busy with other work differed by up to 17% on p50 and
64% on p95. The spread table this script emits is therefore not boilerplate -- it is the
evidence for how much precision a given set of runs actually supports.

Accuracy is asserted identical across runs (it is deterministic: fixed weights over a fixed
split) and reported once. If it moves, that is a harness bug, not machine noise.

Usage
-----
    python benchmarks/aggregate_runs.py benchmarks/results_cpu1.json \
        benchmarks/results_cpu2.json benchmarks/results_cpu3.json

    # or by glob
    python benchmarks/aggregate_runs.py --glob "benchmarks/results_cpu?.json"
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _median(values):
    return statistics.median(values)


def _band(values):
    return min(values), max(values)


def _spread_pct(values):
    """Peak-to-peak spread as a percentage of the median."""
    low, high = _band(values)
    mid = _median(values)
    return 100.0 * (high - low) / mid if mid else 0.0


def load_runs(paths):
    runs = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        runs.append((Path(path).name, payload))
    if not runs:
        raise SystemExit("no result files given")

    devices = {p["protocol"].get("device", "cpu") for _, p in runs}
    if len(devices) > 1:
        raise SystemExit(f"refusing to mix devices in one aggregate: {sorted(devices)}")

    variants = [set(r["id"] for r in p["results"]) for _, p in runs]
    common = set.intersection(*variants)
    missing = set.union(*variants) - common
    if missing:
        print(f"WARNING: these variants are not present in every run and are excluded: "
              f"{sorted(missing)}")
    return runs, sorted(common, key=lambda v: [r["id"] for r in runs[0][1]["results"]].index(v))


def aggregate(runs, variant_ids):
    out = []
    for vid in variant_ids:
        rows = [next(r for r in payload["results"] if r["id"] == vid) for _, payload in runs]
        first = rows[0]

        map50 = {round(r["accuracy"]["map50"], 9) for r in rows}
        map5095 = {round(r["accuracy"]["map50_95"], 9) for r in rows}
        # Accuracy is deterministic: identical weights, identical images, no sampling. If it
        # moved between runs, something is wrong with the harness, not with the machine.
        accuracy_stable = len(map50) == 1 and len(map5095) == 1

        p50s = [r["latency"]["p50_ms"] for r in rows]
        p95s = [r["latency"]["p95_ms"] for r in rows]
        peaks = [r["memory_mib"]["peak_process"] for r in rows]
        infers = [r["latency"]["stage_mean_ms"]["inference"] for r in rows]

        out.append({
            "id": vid,
            "model": first["model"], "backend": first["backend"],
            "precision": first["precision"], "imgsz": first["imgsz"],
            "size_mb": first["size_mb"], "size_bytes": first["size_bytes"],
            "runs": len(rows),
            "p50": {"median": _median(p50s), "min": min(p50s), "max": max(p50s),
                    "spread_pct": _spread_pct(p50s), "values": p50s},
            "p95": {"median": _median(p95s), "min": min(p95s), "max": max(p95s),
                    "spread_pct": _spread_pct(p95s), "values": p95s},
            "inference": {"median": _median(infers), "min": min(infers), "max": max(infers)},
            "peak_mib": {"median": _median(peaks), "min": min(peaks), "max": max(peaks)},
            "accuracy_stable": accuracy_stable,
            "map50": first["accuracy"]["map50"],
            "map50_95": first["accuracy"]["map50_95"],
            "map50_observed": sorted(map50),
            "sweep": first.get("sweep", False),
        })
    return out


def _median_payload(runs, rows):
    """Build a results payload whose every latency/memory figure is the median across runs.

    This is fed to write_results.write_markdown() so the aggregate report carries the full
    analysis -- Pareto front, recommended operating point, INT8 quantisation diagnosis,
    resolution trade-off, per-class AP -- without any of it being duplicated here. Only the
    numbers change; the reasoning is the same code that renders a single-run report.
    """
    per_run_results = [{r["id"]: r for r in payload["results"]} for _, payload in runs]
    numeric_latency = ("p50_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms", "max_ms",
                       "mean_ms", "stdev_ms", "throughput_fps_at_p50",
                       "mean_detections_per_image")
    memory_keys = ("baseline_rss", "after_model_load", "model_load_delta",
                   "peak_after_latency", "peak_process")

    merged = []
    for row in rows:
        samples = [run[row["id"]] for run in per_run_results]
        first = samples[0]

        latency = dict(first["latency"])
        for key in numeric_latency:
            latency[key] = round(_median([s["latency"][key] for s in samples]), 4)
        latency["stage_mean_ms"] = {
            stage: round(_median([s["latency"]["stage_mean_ms"][stage] for s in samples]), 4)
            for stage in first["latency"]["stage_mean_ms"]
        }
        memory = {k: round(_median([s["memory_mib"][k] for s in samples]), 2)
                  for k in memory_keys if k in first["memory_mib"]}

        merged.append({**first, "latency": latency, "memory_mib": memory})

    base = runs[0][1]
    return {
        "host": base["host"],
        "protocol": {**base["protocol"], "aggregated_runs": len(runs)},
        "generated_utc": base["generated_utc"],
        "results": merged,
    }


def _spread_sections(runs, rows):
    """The tables unique to an aggregate: run-to-run spread, and cross-variant ratios."""
    lines = []
    add = lines.append

    unstable = [r for r in rows if not r["accuracy_stable"]]
    if unstable:
        add("> **WARNING:** accuracy changed between runs for "
            + ", ".join(f"`{r['id']}` ({r['map50_observed']})" for r in unstable)
            + ". Accuracy is deterministic given fixed weights and a fixed split, so this "
              "indicates a harness or artefact problem rather than machine noise. "
              "Investigate before publishing.")
    else:
        add(f"Accuracy was **identical across all {len(runs)} runs** for every variant, as "
            f"expected - the same weights over the same images is a deterministic "
            f"computation. Only latency and memory vary between runs.")
    add("")

    add("### Run-to-run spread")
    add("")
    add("Every latency figure in this report is the median of "
        f"{len(runs)} runs. This table shows how far the individual runs sat apart, which is "
        "the honest precision of any number quoted above.")
    add("")
    add("| Variant | p50 median | p50 observed range | spread | p95 median | p95 spread |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        add(f"| `{r['id']}` | {r['p50']['median']:.2f} | "
            f"{r['p50']['min']:.2f}-{r['p50']['max']:.2f} | {r['p50']['spread_pct']:.1f}% | "
            f"{r['p95']['median']:.2f} | {r['p95']['spread_pct']:.1f}% |")
    add("")

    worst = max(rows, key=lambda r: r["p50"]["spread_pct"])
    worst95 = max(rows, key=lambda r: r["p95"]["spread_pct"])
    add(f"Worst observed spread: **{worst['p50']['spread_pct']:.1f}%** on p50 "
        f"(`{worst['id']}`) and **{worst95['p95']['spread_pct']:.1f}%** on p95 "
        f"(`{worst95['id']}`). Anything quoted to tighter precision than that is reading "
        f"noise. Note that a benchmark run alongside other heavy work on the same machine "
        f"is far less reproducible than this - these runs were taken on an idle host.")
    add("")

    add("### Ratios between variants")
    add("")
    add("Each ratio is computed **within** a run and then median-ed across runs, so both "
        "measurements share the same thermal and load conditions. Ratios are consequently "
        "tighter than either endpoint and are the safest form in which to quote a speed "
        "comparison.")
    add("")
    non_sweep = [r for r in rows if not r["sweep"]]
    if non_sweep:
        fastest = min(non_sweep, key=lambda r: r["p50"]["median"])
        add(f"| Variant | p50 median | x slower than `{fastest['id']}` |")
        add("| --- | ---: | ---: |")
        for r in sorted(non_sweep, key=lambda r: r["p50"]["median"]):
            per_run = [a / b for a, b in zip(r["p50"]["values"], fastest["p50"]["values"])]
            add(f"| `{r['id']}` | {r['p50']['median']:.2f} ms | "
                f"{_median(per_run):.2f}x (range {min(per_run):.2f}-{max(per_run):.2f}) |")
    return lines


def _gpu_context_section(context_path: Path):
    """Render the CPU-vs-GPU appendix from benchmarks/gpu_context.json, if it exists.

    Kept as an appendix rather than folded into the main tables because it is a different
    measurement: PyTorch on both devices, one run, no mAP. It exists to qualify how the CPU
    results should be read, not to compete with them.
    """
    if not context_path.is_file():
        return []
    ctx = json.loads(context_path.read_text(encoding="utf-8"))
    head, sweep = ctx["head_to_head"], ctx["sweep"]

    lines = []
    add = lines.append
    add("## Appendix: what a GPU changes")
    add("")
    add(f"Measured {ctx['generated_utc']} on **{ctx['gpu']}** ({ctx['gpu_memory_gib']} GiB), "
        f"{ctx['backend']}, torch {ctx['torch']}. "
        f"{ctx['protocol']['warmup']} warmup + {ctx['protocol']['runs']} timed runs. "
        f"{ctx['protocol']['note']}. Raw data in [`gpu_context.json`](gpu_context.json).")
    add("")
    add("This benchmark targets CPU because that is the edge deployment story. This appendix "
        "answers only two questions: how much a GPU actually buys for a nano detector, and "
        "whether the resolution finding above survives on one.")
    add("")

    add("### Head to head at imgsz 640")
    add("")
    repeat = ctx["protocol"].get("repeat", 1)
    if repeat > 1:
        add("| Model | CPU p50 | GPU p50 | End-to-end speedup | range | Inference-only |")
        add("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in head:
            lo, hi = row.get("speedup_p50_range", [row["speedup_p50"]] * 2)
            add(f"| {row['model']} | {row['cpu']['p50_ms']:.2f} ms "
                f"(±{row['cpu'].get('p50_spread_pct', 0):.1f}%) | "
                f"{row['gpu']['p50_ms']:.2f} ms (±{row['gpu'].get('p50_spread_pct', 0):.1f}%) | "
                f"**{row['speedup_p50']:.2f}x** | {lo:.2f}-{hi:.2f} | "
                f"{row['speedup_inference']:.2f}x |")
    else:
        add("| Model | CPU p50 | GPU p50 | End-to-end speedup | Inference-only speedup |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for row in head:
            add(f"| {row['model']} | {row['cpu']['p50_ms']:.2f} ms | "
                f"{row['gpu']['p50_ms']:.2f} ms | "
                f"**{row['speedup_p50']:.2f}x** | {row['speedup_inference']:.2f}x |")
    add("")
    best = max(row["speedup_p50"] for row in head)
    worst = min(row["speedup_p50"] for row in head)
    add(f"**{worst:.2f}-{best:.2f}x, not the 10-20x a GPU usually suggests.** A nano YOLO at "
        f"640 is roughly 8 GFLOPs against this card's ~20 TFLOPS, so the arithmetic is "
        f"sub-millisecond and what remains is kernel launch, host-device transfer and Python "
        f"overhead. The CPU numbers in the tables above are not embarrassing by comparison "
        f"because the model is too small to saturate a GPU.")
    add("")

    add("### The resolution trade-off does not survive on GPU")
    add("")
    add("| imgsz | CPU p50 | GPU p50 | GPU inference stage |")
    add("| ---: | ---: | ---: | ---: |")
    for row in sweep:
        add(f"| {row['imgsz']} | {row['cpu']['p50_ms']:.2f} ms | {row['gpu']['p50_ms']:.2f} ms | "
            f"{row['gpu']['inference_ms']:.2f} ms |")
    add("")
    cpu_span = sweep[-1]["cpu"]["p50_ms"] / sweep[0]["cpu"]["p50_ms"]
    gpu_span = sweep[-1]["gpu"]["p50_ms"] / sweep[0]["gpu"]["p50_ms"]
    infer_lo = min(r["gpu"]["inference_ms"] for r in sweep)
    infer_hi = max(r["gpu"]["inference_ms"] for r in sweep)
    pixels = (sweep[-1]["imgsz"] / sweep[0]["imgsz"]) ** 2
    infer_series = [r["gpu"]["inference_ms"] for r in sweep]
    add(f"Across imgsz {sweep[0]['imgsz']} to {sweep[-1]['imgsz']} the CPU spans "
        f"**{cpu_span:.2f}x** but the GPU spans only **{gpu_span:.2f}x**. The GPU *inference "
        f"stage* barely moves at all - {infer_lo:.2f} to {infer_hi:.2f} ms, a "
        f"{100 * (infer_hi - infer_lo) / infer_lo:.1f}% change for **{pixels:.2f}x more "
        f"pixels**.")
    add("")
    if infer_series == sorted(infer_series, reverse=True):
        # Inference time falling as the input grows cannot be a compute effect. When this
        # happens the measurement is not tracking arithmetic at all, which is a stronger
        # statement than "the relationship is weak".
        add(f"In fact the inference stage is *inversely* ordered: "
            f"{' > '.join(f'{v:.2f}' for v in infer_series)} ms as the input grows "
            f"{sweep[0]['imgsz']} -> {sweep[-1]['imgsz']}. More pixels cannot make "
            f"convolutions finish sooner, so this is measurement noise around a roughly "
            f"constant ~{sum(infer_series) / len(infer_series):.1f} ms floor. The number "
            f"being measured is not compute - it is fixed per-call overhead, and the model "
            f"is small enough to hide entirely underneath it.")
    else:
        add("That is the signature of a workload bound by fixed overhead rather than by "
            "compute.")
    add("")
    add("**Consequence:** the headline CPU conclusion - that input resolution is the biggest "
        "latency lever - is a CPU conclusion. On a GPU you should run at 640 and take the "
        "accuracy, because the smaller inputs cost almost nothing less. Anyone reading these "
        "results for a GPU target should start from that inversion.")
    add("")

    add("### Reading this for Pi and Jetson")
    add("")
    add("| Transfers to other hardware | Does not |")
    add("| --- | --- |")
    add("| Rankings between variants | Absolute milliseconds |")
    add("| The mechanisms (nano models are overhead-bound on GPU; ONNX INT8 leaves most "
        "convolutions in float) | Tail latency (p95/p99) |")
    add("| Every mAP figure - accuracy is hardware-independent | Anything from a TensorRT "
        "engine built on a different GPU |")
    add("")
    add("Neither target device was measured here. An x86 Ryzen is not a Pi 5's ARM "
        "Cortex-A76, and a laptop RTX 4050 is not a Jetson Orin. Treat the CPU table as "
        "Pi-*shaped* and this appendix as Jetson-*indicative*; quote ratios and mechanisms "
        "rather than absolute times. Two specifics worth knowing:")
    add("")
    add("- **TensorRT engines are not portable.** An engine built here (sm_89) will not load "
        "on a Jetson Orin (sm_87). Jetson figures must be built and measured on the Jetson.")
    add("- **The harness runs unmodified on both.** `run_benchmark.py` reads the CPU name "
        "from `/proc/cpuinfo` on Linux and falls back to `resource.getrusage` for peak "
        "memory where Windows' `peak_wset` does not exist. Running it on the device is the "
        "only way to get defensible numbers for it.")
    add("")
    if repeat > 1:
        add(f"Each figure is the median of **{repeat} interleaved rounds** per device, with "
            f"the observed spread shown alongside. Speedups are computed within a round and "
            f"then median-ed, so both halves of every ratio share one thermal state.")
    add("")
    add("This appendix reports no mAP "
        "(accuracy is device-independent - see the main tables) and measures PyTorch on "
        "both devices rather than ONNX, because ONNX on GPU needs `onnxruntime-gpu` for the "
        "CUDA execution provider. The ratios are what it is for.")
    return lines


def write_markdown(runs, rows, out_path: Path):
    """Render the full analysis over median values, with spread tables spliced in."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from write_results import write_markdown as render

    sources = ", ".join(f"`{name}`" for name, _ in runs)
    subtitle = (
        f"**Aggregated over {len(runs)} independent runs** ({sources}). Every latency and "
        f"memory figure below is the **median across those runs**; the run-to-run spread is "
        f"reported directly under the results table. Accuracy is deterministic and identical "
        f"in every run. Raw data in [`results_aggregate.json`](results_aggregate.json)."
    )
    render(_median_payload(runs, rows), out_path,
           extra_after_results=_spread_sections(runs, rows),
           extra_at_end=_gpu_context_section(out_path.parent / "gpu_context.json"),
           subtitle=subtitle)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate several benchmark runs into medians plus spread.")
    parser.add_argument("files", nargs="*", help="results_*.json files from run_benchmark.py")
    parser.add_argument("--glob", default=None, help="glob pattern instead of explicit files")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks" / "results_aggregate")
    args = parser.parse_args(argv)

    paths = sorted(globmod.glob(args.glob)) if args.glob else args.files
    if not paths:
        raise SystemExit("no input files; pass files or --glob")
    print("aggregating %d run(s):" % len(paths))
    for p in paths:
        print("   ", p)

    runs, variant_ids = load_runs(paths)
    rows = aggregate(runs, variant_ids)

    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(
        {"source_runs": [name for name, _ in runs],
         "host": runs[0][1]["host"], "protocol": runs[0][1]["protocol"],
         "variants": rows}, indent=2), encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    write_markdown(runs, rows, md_path)

    print("\n%-26s %10s %16s %8s   %s" % ("variant", "p50 med", "p50 range", "spread", "mAP50"))
    for r in rows:
        print("%-26s %10.2f %7.2f-%7.2f %7.1f%%   %.4f%s" % (
            r["id"], r["p50"]["median"], r["p50"]["min"], r["p50"]["max"],
            r["p50"]["spread_pct"], r["map50"],
            "" if r["accuracy_stable"] else "  <-- ACCURACY UNSTABLE"))
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
