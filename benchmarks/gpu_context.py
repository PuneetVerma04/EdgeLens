#!/usr/bin/env python
"""Measure CPU-vs-GPU context for the report, using the PyTorch checkpoints.

This is deliberately narrow. The main benchmark targets CPU because that is the edge
deployment story; this script exists only to answer two questions that shape how the CPU
results should be read:

  1. How much does a GPU actually buy for a nano detector?
  2. Does the resolution trade-off -- the headline CPU finding -- still hold on GPU?

It uses the .pt checkpoints, not the ONNX artefacts, because ONNX on GPU needs the CUDA
execution provider (`pip install onnxruntime-gpu`), while PyTorch already has CUDA here. The
architectural conclusions do not depend on the runtime.

**Both devices are measured back-to-back in a single process.** Absolute latency on a laptop
moves with thermal state and background load, but a ratio taken from two measurements
seconds apart shares those conditions and is far more robust. The ratios are the point; the
absolutes are context.

Every timed GPU iteration calls torch.cuda.synchronize(). CUDA kernels launch
asynchronously, so without it the timer measures launch overhead rather than execution.

Usage
-----
    python benchmarks/gpu_context.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs" / "neu_det"

# Repository root on sys.path so the shared helpers resolve when this script is run
# directly from any working directory.
sys.path.insert(0, str(REPO_ROOT))
from common.metrics import percentile_nearest_rank  # noqa: E402

CONF, IOU, MAX_DET = 0.25, 0.7, 300
IMAGE_POOL = 20


def bench(model, frames, imgsz, device, runs, warmup):
    import torch

    on_gpu = str(device) != "cpu"
    kwargs = dict(imgsz=imgsz, conf=CONF, iou=IOU, max_det=MAX_DET,
                  device=device, verbose=False)

    for i in range(warmup):
        model.predict(frames[i % len(frames)], **kwargs)
    if on_gpu:
        torch.cuda.synchronize()

    latencies, inference_total = [], 0.0
    for i in range(runs):
        started = time.perf_counter()
        result = model.predict(frames[i % len(frames)], **kwargs)[0]
        if on_gpu:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        inference_total += float(result.speed.get("inference", 0.0))

    latencies.sort()
    return {
        "p50_ms": round(percentile_nearest_rank(latencies, 50), 4),
        "p95_ms": round(percentile_nearest_rank(latencies, 95), 4),
        "mean_ms": round(statistics.fmean(latencies), 4),
        "inference_ms": round(inference_total / runs, 4),
    }


def bench_repeated(model, frames, imgsz, devices, runs, warmup, repeat):
    """Measure every device `repeat` times, interleaved.

    Rounds alternate CPU then GPU so the two are always seconds apart and share thermal and
    load state. Ratios are therefore computed within a round and median-ed across rounds,
    which is far more robust than dividing two independently-drifting medians. On a laptop
    this matters more than usual: CPU and GPU share one power and thermal budget, so GPU
    work heats the package and can down-clock the CPU measured just after it.
    """
    per_device = {name: [] for name in devices}
    for _ in range(repeat):
        for name, device in devices.items():
            per_device[name].append(bench(model, frames, imgsz, device, runs, warmup))

    summary = {}
    for name, samples in per_device.items():
        summary[name] = {
            key: round(statistics.median([s[key] for s in samples]), 4)
            for key in ("p50_ms", "p95_ms", "mean_ms", "inference_ms")
        }
        p50s = [s["p50_ms"] for s in samples]
        summary[name]["rounds"] = repeat
        summary[name]["p50_values"] = p50s
        summary[name]["p50_spread_pct"] = round(
            100.0 * (max(p50s) - min(p50s)) / statistics.median(p50s), 2) if p50s else 0.0
    return summary, per_device


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure CPU vs GPU context for the report.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "neu_det" / "data.yaml")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default="0", help="CUDA device index")
    parser.add_argument("--sweep-imgsz", type=int, nargs="+", default=(256, 320, 640))
    parser.add_argument("--repeat", type=int, default=3,
                        help="measurement rounds per device; medians and spread are reported")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "benchmarks" / "gpu_context.json")
    args = parser.parse_args(argv)

    import cv2
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device available; nothing to measure")

    image_paths = sorted((args.data.parent / "test" / "images").glob("*.jpg"))[:IMAGE_POOL]
    frames = [cv2.imread(str(p)) for p in image_paths]
    if not frames or any(f is None for f in frames):
        raise SystemExit("could not decode benchmark images")

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"protocol: {args.warmup} warmup + {args.runs} timed runs, batch 1, "
          f"CPU and GPU back-to-back in one process\n")

    models = {"yolov8n": RUNS / "yolov8n" / "weights" / "best.pt",
              "yolo26n": RUNS / "yolo26n" / "weights" / "best.pt"}

    devices = {"cpu": "cpu", "gpu": args.device}
    head_to_head = []
    for name, weights in models.items():
        model = YOLO(str(weights))
        summary, raw = bench_repeated(model, frames, 640, devices,
                                      args.runs, args.warmup, args.repeat)
        cpu, gpu = summary["cpu"], summary["gpu"]
        # Ratio per round, then median: both halves of each ratio share one thermal state.
        per_round = [c["p50_ms"] / g["p50_ms"] for c, g in zip(raw["cpu"], raw["gpu"])]
        row = {
            "model": name, "imgsz": 640, "cpu": cpu, "gpu": gpu,
            "speedup_p50": round(statistics.median(per_round), 4),
            "speedup_p50_range": [round(min(per_round), 4), round(max(per_round), 4)],
            "speedup_inference": round(cpu["inference_ms"] / gpu["inference_ms"], 4),
        }
        head_to_head.append(row)
        print(f"{name:<9} CPU p50 {cpu['p50_ms']:7.2f} (spread {cpu['p50_spread_pct']:4.1f}%)  "
              f"GPU p50 {gpu['p50_ms']:7.2f} (spread {gpu['p50_spread_pct']:4.1f}%)  "
              f"end-to-end {row['speedup_p50']:.2f}x "
              f"[{row['speedup_p50_range'][0]:.2f}-{row['speedup_p50_range'][1]:.2f}]  "
              f"inference-only {row['speedup_inference']:.2f}x")

    print()
    sweep = []
    model = YOLO(str(models["yolo26n"]))
    for imgsz in args.sweep_imgsz:
        summary, _ = bench_repeated(model, frames, imgsz, devices,
                                    args.runs, args.warmup, args.repeat)
        cpu, gpu = summary["cpu"], summary["gpu"]
        sweep.append({"model": "yolo26n", "imgsz": imgsz, "cpu": cpu, "gpu": gpu})
        print(f"sweep imgsz={imgsz:<4} CPU p50 {cpu['p50_ms']:7.2f}  "
              f"GPU p50 {gpu['p50_ms']:7.2f}  GPU inference {gpu['inference_ms']:6.2f}  "
              f"(GPU spread {gpu['p50_spread_pct']:.1f}%)")

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_name,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
        "torch": torch.__version__,
        "backend": "PyTorch (both devices)",
        "protocol": {"runs": args.runs, "warmup": args.warmup, "batch": 1,
                     "repeat": args.repeat,
                     "conf": CONF, "iou": IOU, "max_det": MAX_DET,
                     "image_pool": len(frames),
                     "note": f"{args.repeat} interleaved rounds per device in one process, "
                             f"medians reported; torch.cuda.synchronize() before stopping "
                             f"the clock"},
        "head_to_head": head_to_head,
        "sweep": sweep,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
