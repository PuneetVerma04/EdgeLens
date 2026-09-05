#!/usr/bin/env python
"""CPU benchmark for the exported NEU-DET detectors.

Measures, per variant: p50/p95 end-to-end latency, peak process memory, on-disk size, and
mAP@0.5 / mAP@0.5:0.95 on the held-out test split.

Each variant is measured in its **own subprocess**. Peak working set is a high-water mark
for the life of a process, so measuring several variants in one process would report the
maximum across all of them for each. One process per variant is the only way the memory
column means anything.

Usage
-----
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --runs 500 --warmup 50
    python benchmarks/run_benchmark.py --only yolo26n-onnx-fp32-640
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts" / "onnx"
RUNS = REPO_ROOT / "runs" / "neu_det"

# Repository root on sys.path so the shared helpers resolve when this script is run
# directly from any working directory.
sys.path.insert(0, str(REPO_ROOT))
from common.metrics import percentile_nearest_rank  # noqa: E402

# Inference settings held constant across every variant, so the only thing that varies is
# the artefact being measured. conf=0.25 is the deployment operating point; the mAP figures
# are computed by Ultralytics' validator at its own low threshold, independently of this.
BENCH_CONF = 0.25
BENCH_IOU = 0.7
BENCH_MAX_DET = 300

# Images are cycled during the timed loop rather than reusing one frame, so postprocessing
# cost reflects a realistic spread of detection counts instead of a single lucky image.
LATENCY_IMAGE_POOL = 20


def default_variants() -> list[dict]:
    """The variants to measure, in report order."""
    return [
        # PyTorch baselines: the reference the ONNX artefacts are derived from.
        {"id": "yolov8n-pytorch-640", "model": "yolov8n", "backend": "PyTorch",
         "precision": "FP32", "imgsz": 640, "path": str(RUNS / "yolov8n" / "weights" / "best.pt")},
        {"id": "yolo26n-pytorch-640", "model": "yolo26n", "backend": "PyTorch",
         "precision": "FP32", "imgsz": 640, "path": str(RUNS / "yolo26n" / "weights" / "best.pt")},
        # The four exported artefacts.
        {"id": "yolov8n-onnx-fp32-640", "model": "yolov8n", "backend": "ONNX Runtime",
         "precision": "FP32", "imgsz": 640, "path": str(ARTIFACTS / "yolov8n_fp32.onnx")},
        {"id": "yolov8n-onnx-int8-640", "model": "yolov8n", "backend": "ONNX Runtime",
         "precision": "INT8", "imgsz": 640, "path": str(ARTIFACTS / "yolov8n_int8.onnx")},
        {"id": "yolo26n-onnx-fp32-640", "model": "yolo26n", "backend": "ONNX Runtime",
         "precision": "FP32", "imgsz": 640, "path": str(ARTIFACTS / "yolo26n_fp32.onnx")},
        {"id": "yolo26n-onnx-int8-640", "model": "yolo26n", "backend": "ONNX Runtime",
         "precision": "INT8", "imgsz": 640, "path": str(ARTIFACTS / "yolo26n_int8.onnx")},
        # Resolution sweep on 200x200 native imagery.
        {"id": "yolo26n-onnx-fp32-256", "model": "yolo26n", "backend": "ONNX Runtime",
         "precision": "FP32", "imgsz": 256, "path": str(ARTIFACTS / "yolo26n_256_fp32.onnx"),
         "sweep": True},
        {"id": "yolo26n-onnx-fp32-320", "model": "yolo26n", "backend": "ONNX Runtime",
         "precision": "FP32", "imgsz": 320, "path": str(ARTIFACTS / "yolo26n_320_fp32.onnx"),
         "sweep": True},
    ]


# ---------------------------------------------------------------------------------------
# Host description
# ---------------------------------------------------------------------------------------


def cpu_name() -> str:
    """Human-readable CPU model. platform.processor() returns a family/model string on
    Windows, so prefer the registry's ProcessorNameString when available."""
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            with key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def host_info(threads: int, device: str = "cpu") -> dict:
    import onnxruntime
    import psutil
    import torch
    import ultralytics

    memory = psutil.virtual_memory()
    return {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "threads_used": threads,
        "ram_gib": round(memory.total / 1024 ** 3, 2),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "ultralytics": ultralytics.__version__,
        "device": device,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "gpu_memory_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
            if torch.cuda.is_available() else None),
    }


# ---------------------------------------------------------------------------------------
# Worker: measures exactly one variant
# ---------------------------------------------------------------------------------------


def measure(spec: dict, data: Path, runs: int, warmup: int, threads: int,
            device: str = "cpu") -> dict:
    """Measure one variant. Runs inside a dedicated subprocess."""
    import cv2
    import psutil
    import torch

    process = psutil.Process()

    def peak_mib() -> float:
        info = process.memory_info()
        # peak_wset is Windows-only; elsewhere fall back to current RSS (POSIX high-water
        # marks come from resource.getrusage instead).
        peak = getattr(info, "peak_wset", None)
        if peak is None:
            try:
                import resource

                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            except ImportError:
                peak = info.rss
        return peak / 1024 ** 2

    def rss_mib() -> float:
        return process.memory_info().rss / 1024 ** 2

    torch.set_num_threads(threads)
    baseline_rss = rss_mib()

    from ultralytics import YOLO

    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"artefact not found: {path}")

    on_gpu = str(device) != "cpu"
    if on_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={device} requested but torch reports no CUDA")
        if path.suffix == ".onnx":
            import onnxruntime
            # Without this check an ONNX model silently runs on the CPU EP and the run
            # is reported as a GPU measurement. That is the worst possible outcome.
            if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                raise RuntimeError(
                    "ONNX GPU run requested but onnxruntime has no CUDAExecutionProvider "
                    f"(available: {onnxruntime.get_available_providers()}). Install the GPU "
                    "build:  pip uninstall -y onnxruntime && pip install onnxruntime-gpu")

    load_started = time.perf_counter()
    model = YOLO(str(path), task="detect")
    load_seconds = time.perf_counter() - load_started
    after_load_rss = rss_mib()

    # --- latency ------------------------------------------------------------------------
    image_paths = sorted((data.parent / "test" / "images").glob("*.jpg"))[:LATENCY_IMAGE_POOL]
    if not image_paths:
        raise FileNotFoundError(f"no test images under {data.parent / 'test' / 'images'}")
    # Decoded up front: this benchmark measures the model, not the JPEG decoder or the disk.
    frames = [cv2.imread(str(p)) for p in image_paths]
    if any(f is None for f in frames):
        raise RuntimeError("cv2 failed to decode one or more benchmark images")

    predict_kwargs = dict(imgsz=spec["imgsz"], conf=BENCH_CONF, iou=BENCH_IOU,
                          max_det=BENCH_MAX_DET, device=device, verbose=False)

    for i in range(warmup):
        model.predict(frames[i % len(frames)], **predict_kwargs)
    if on_gpu:
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    stage_totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    detections = 0
    for i in range(runs):
        frame = frames[i % len(frames)]
        started = time.perf_counter()
        result = model.predict(frame, **predict_kwargs)[0]
        if on_gpu:
            # CUDA kernels are launched asynchronously; without a sync the timer
            # measures launch overhead rather than execution.
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        for key in stage_totals:
            stage_totals[key] += float(result.speed.get(key, 0.0))
        detections += 0 if result.boxes is None else len(result.boxes)

    latencies_ms.sort()

    def percentile(p: float) -> float:
        return percentile_nearest_rank(latencies_ms, p)

    latency = {
        "runs": runs,
        "warmup": warmup,
        "image_pool": len(frames),
        "p50_ms": round(percentile(50), 4),
        "p90_ms": round(percentile(90), 4),
        "p95_ms": round(percentile(95), 4),
        "p99_ms": round(percentile(99), 4),
        "min_ms": round(latencies_ms[0], 4),
        "max_ms": round(latencies_ms[-1], 4),
        "mean_ms": round(statistics.fmean(latencies_ms), 4),
        "stdev_ms": round(statistics.stdev(latencies_ms), 4) if len(latencies_ms) > 1 else 0.0,
        "throughput_fps_at_p50": round(1000.0 / percentile(50), 3),
        "stage_mean_ms": {k: round(v / runs, 4) for k, v in stage_totals.items()},
        "mean_detections_per_image": round(detections / runs, 3),
    }
    peak_after_latency = peak_mib()

    # --- accuracy -----------------------------------------------------------------------
    validation = model.val(data=str(data), split="test", imgsz=spec["imgsz"], batch=1,
                           device=device, rect=False, plots=False, verbose=False)
    accuracy = {
        "map50": round(float(validation.box.map50), 6),
        "map50_95": round(float(validation.box.map), 6),
        "mean_precision": round(float(validation.box.mp), 6),
        "mean_recall": round(float(validation.box.mr), 6),
        "per_class_ap50": {
            model.names[int(c)]: round(float(validation.box.ap50[i]), 6)
            for i, c in enumerate(validation.box.ap_class_index)
        },
    }

    return {
        **spec,
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024 ** 2, 4),
        "load_seconds": round(load_seconds, 3),
        "memory_mib": {
            "baseline_rss": round(baseline_rss, 2),
            "after_model_load": round(after_load_rss, 2),
            "model_load_delta": round(after_load_rss - baseline_rss, 2),
            "peak_after_latency": round(peak_after_latency, 2),
            "peak_process": round(peak_mib(), 2),
        },
        "gpu_memory_mib": (
            {"peak_allocated": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2),
             "peak_reserved": round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2)}
            if on_gpu else None),
        "device": str(device),
        "latency": latency,
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------------------
# Parent: orchestrates one subprocess per variant
# ---------------------------------------------------------------------------------------


def run_worker(spec: dict, args, out_path: Path) -> dict:
    payload = json.dumps({
        "spec": spec, "data": str(args.data), "runs": args.runs,
        "warmup": args.warmup, "threads": args.threads, "device": args.device,
    })
    env = dict(os.environ)
    # Pin the thread pools so every variant gets the same CPU budget. ONNX Runtime and
    # torch both size their pools from these unless told otherwise.
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", payload,
         "--worker-out", str(out_path)],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False,   # returncode is inspected below so the worker's stderr can be surfaced
    )
    if completed.returncode != 0 or not out_path.is_file():
        tail = "\n".join((completed.stderr or completed.stdout or "").splitlines()[-15:])
        raise RuntimeError(f"variant {spec['id']} failed (exit {completed.returncode}):\n{tail}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark exported NEU-DET detectors on CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "neu_det" / "data.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "benchmarks")
    parser.add_argument("--runs", type=int, default=200, help="timed inference runs per variant")
    parser.add_argument("--warmup", type=int, default=20, help="untimed runs before timing starts")
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU threads per variant (0 = physical core count)")
    parser.add_argument("--device", default="cpu",
                        help="cpu, or a CUDA index such as 0. ONNX on GPU needs onnxruntime-gpu.")
    parser.add_argument("--tag", default=None,
                        help="output suffix; defaults to cpu/gpu based on --device")
    parser.add_argument("--only", nargs="+", default=None, help="variant ids to run")
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- worker mode --------------------------------------------------------------------
    if args.worker:
        job = json.loads(args.worker)
        result = measure(job["spec"], Path(job["data"]), job["runs"], job["warmup"],
                         job["threads"], job.get("device", "cpu"))
        args.worker_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0

    # --- parent mode --------------------------------------------------------------------
    import psutil

    if args.threads <= 0:
        args.threads = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1

    variants = default_variants()
    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v["id"] in wanted]
        missing = wanted - {v["id"] for v in variants}
        if missing:
            raise SystemExit(f"unknown variant id(s): {sorted(missing)}")

    host = host_info(args.threads, args.device)
    tag = args.tag or ("cpu" if args.device == "cpu" else "gpu")
    print("=" * 78)
    print("BENCHMARK  device=%s  tag=%s" % (args.device, tag))
    print("=" * 78)
    print(f"CPU        : {host['cpu']}")
    print(f"Cores      : {host['physical_cores']} physical / {host['logical_cores']} logical, "
          f"using {args.threads} threads per variant")
    print(f"RAM        : {host['ram_gib']} GiB")
    if host["gpu"]:
        print(f"GPU        : {host['gpu']} ({host['gpu_memory_gib']} GiB)"
              f"{'  [in use]' if args.device != 'cpu' else '  [not used: --device cpu]'}")
    print(f"Runtime    : torch {host['torch']} | onnxruntime {host['onnxruntime']} "
          f"{host['onnxruntime_providers']} | ultralytics {host['ultralytics']}")
    print(f"Protocol   : {args.warmup} warmup + {args.runs} timed runs, batch 1, "
          f"conf={BENCH_CONF} iou={BENCH_IOU}, one subprocess per variant")
    print(f"Variants   : {len(variants)}")

    scratch = args.out / "_worker_results"
    scratch.mkdir(parents=True, exist_ok=True)

    results = []
    for index, spec in enumerate(variants, 1):
        print(f"\n[{index}/{len(variants)}] {spec['id']}  ({Path(spec['path']).name})")
        started = time.time()
        result = run_worker(spec, args, scratch / f"{tag}-{spec['id']}.json")
        results.append(result)
        latency, accuracy, memory = result["latency"], result["accuracy"], result["memory_mib"]
        print(f"    p50 {latency['p50_ms']:8.2f} ms | p95 {latency['p95_ms']:8.2f} ms | "
              f"peak {memory['peak_process']:7.1f} MiB | {result['size_mb']:6.2f} MB | "
              f"mAP50 {accuracy['map50']:.4f} | mAP50-95 {accuracy['map50_95']:.4f}"
              f"   [{time.time() - started:.0f}s]")

    payload = {
        "host": host,
        "protocol": {
            "runs": args.runs, "warmup": args.warmup, "threads": args.threads,
            "batch": 1, "conf": BENCH_CONF, "iou": BENCH_IOU, "max_det": BENCH_MAX_DET,
            "latency_image_pool": LATENCY_IMAGE_POOL,
            "split": "test",
            "isolation": "one subprocess per variant",
            "device": args.device,
            "percentile_method": "nearest-rank (no interpolation)",
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    json_path = args.out / f"results_{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {json_path}")

    from write_results import write_markdown  # noqa: E402  (same directory)

    markdown_path = args.out / f"results_{tag}.md"
    write_markdown(payload, markdown_path)
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
