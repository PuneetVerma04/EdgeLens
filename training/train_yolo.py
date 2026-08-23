#!/usr/bin/env python
"""Train a YOLO detector on the NEU-DET YOLO dataset under a fully pinned configuration.

The point of this script is *controlled comparison*. Every hyperparameter that can affect
the result is either

  * required on the command line (epochs, batch, imgsz, seed, patience, data), so two
    invocations are visibly identical in your shell history, or
  * pinned in CONTROLLED_HYPERPARAMETERS below, so neither Ultralytics' defaults nor its
    "auto" heuristics can quietly diverge between runs.

Anything else requires --set KEY=VALUE together with --allow-uncontrolled, and is recorded
in the run's config JSON under "uncontrolled_overrides" so it cannot be forgotten later.

After every run the script diffs this run's configuration against every other run in the
same --project directory and prints a table of differences. If two runs differ by anything
other than --model, you get a loud warning instead of an invalid comparison six weeks later.

Pipeline
--------
    1. Command line                       parse_args
    2. Resolve the run configuration      build_config
    3. Probe the environment              environment_info, check_compute_device/host_memory
    4. Fingerprint the dataset and repo   dataset_fingerprint, _git_state
    5. Guard comparability across runs    comparison_keys, compare_with_previous_runs
    6. Resolve the run directory          unique_run_dir
    7. Train, then write the config JSON  main

Usage
-----
    python training/train_yolo.py --model yolov8n.pt --data data/neu_det/data.yaml \
        --epochs 100 --batch 16 --imgsz 640 --seed 0 --patience 50

Outputs
-------
    <project>/<name>/edgelens_train_config.json   full config + environment + results
    <project>/<name>/weights/{best,last}.pt       Ultralytics checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Determinism knobs must be set before CUDA is initialised, i.e. before torch is imported
# by Ultralytics. Ultralytics' init_seeds() sets these too when deterministic=True, but by
# then the CUDA context may already exist, so we set them up front.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

CONFIG_FILENAME = "edgelens_train_config.json"
CONFIG_SCHEMA = "edgelens.train_config/2"

# --------------------------------------------------------------------------------------
# The pinned configuration.
#
# Ultralytics' defaults are NOT stable across releases and several of them are adaptive.
# Everything below is written out explicitly, including values that happen to equal the
# current default, so that a future `pip install -U ultralytics` cannot silently change the
# experiment underneath you.
#
# The five knobs the CLI requires (epochs, batch, imgsz, seed, patience) are deliberately
# absent here -- they must be typed out on every invocation.
# --------------------------------------------------------------------------------------
CONTROLLED_HYPERPARAMETERS = {
    # --- optimisation -----------------------------------------------------------------
    # optimizer MUST NOT be "auto". With "auto" Ultralytics picks the optimizer, lr0 and
    # momentum from the number of classes and the iteration count, which means two
    # architectures on the same dataset can end up on different optimizers. That is exactly
    # the kind of hidden confound this script exists to prevent.
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,
    "nbs": 64,              # loss is normalised to this nominal batch, so accumulate = nbs/batch

    # --- loss gains --------------------------------------------------------------------
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "cls_pw": 0.0,          # no inverse-frequency class weighting; NEU-DET is only mildly skewed

    # --- data handling -----------------------------------------------------------------
    "single_cls": False,
    "rect": False,          # rectangular batching would interact with imgsz; keep square
    "fraction": 1.0,        # use the whole training split
    "multi_scale": 0.0,     # a second image-size axis is the classic uncontrolled variable
    "cache": False,         # RAM/disk caching changes nothing but speed; pinned for honesty
    "close_mosaic": 10,     # last N epochs run without mosaic

    # --- augmentation ------------------------------------------------------------------
    # NEU-DET is 200x200 grayscale steel surface. Deviations from Ultralytics defaults are
    # annotated; they are applied identically to every model, so they do not affect the
    # fairness of the comparison, only its absolute numbers.
    "hsv_h": 0.0,           # deviation: hue is meaningless on grayscale input
    "hsv_s": 0.0,           # deviation: saturation is meaningless on grayscale input
    "hsv_v": 0.4,           # brightness jitter is a real lighting nuisance on a steel line
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.5,          # deviation: a steel surface has no canonical "up"
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,

    # --- runtime -----------------------------------------------------------------------
    "pretrained": True,     # only meaningful when --model is a .pt; recorded either way
    "cls_remap": True,      # no-op here (no NEU-DET class name exists in COCO) but pinned
    "deterministic": True,
    "amp": True,
    "compile": False,
    "channels_last": False,
    "val": True,
    "plots": True,
    "save": True,
    "save_period": -1,
    "profile": False,
    "freeze": None,         # train every layer
    "time": None,           # a wall-clock budget would override epochs -- must stay unset
    "resume": False,

    # --- validation during training ----------------------------------------------------
    "iou": 0.7,             # NMS IoU (ignored by NMS-free heads such as YOLO26's end2end)
    "max_det": 300,
    "conf": None,           # None -> Ultralytics uses 0.001 for val
}

# Keys that identify the run rather than configure it. These are expected to differ between
# runs and are not part of the "did anything drift?" comparison.
RUN_IDENTITY_KEYS = ("name", "project", "exist_ok")

# Keys that describe the machine rather than the experiment. Differences here are reported
# separately: they do not invalidate a comparison the way a different lr0 would, but they
# can explain small deltas, so they are worth seeing.
ENVIRONMENT_KEYS = ("device", "workers")

# end2end is deliberately NOT pinned. YOLO26 ships an NMS-free end-to-end head and YOLOv8
# does not; forcing either value would break one of the two architectures. It is left at
# Ultralytics' auto-detection and the resolved value is recorded per run.


# --------------------------------------------------------------------------------------
# STEP 1 - Command line
# --------------------------------------------------------------------------------------

def _parse_scalar(text: str):
    """Parse a --set value into bool / None / int / float / str."""
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null", ""):
        return None
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            pass
    return text

def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector on NEU-DET with every hyperparameter pinned.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Every knob below is REQUIRED on purpose. Two runs that differ only by --model\n"
            "are a controlled comparison; anything else is not, and the script will say so."
        ),
    )
    required = parser.add_argument_group("required (state these on every run)")
    required.add_argument("--model", required=True,
                          help="weights or architecture, e.g. yolov8n.pt / yolo26n.pt / yolo26n.yaml")
    required.add_argument("--data", required=True, type=Path,
                          help="path to data.yaml produced by prepare_neu_det.py")
    required.add_argument("--epochs", required=True, type=int)
    required.add_argument("--batch", required=True, type=int,
                          help="integer batch size; float AutoBatch is rejected (it is not reproducible)")
    required.add_argument("--imgsz", required=True, type=int)
    required.add_argument("--seed", required=True, type=int)
    required.add_argument("--patience", required=True, type=int,
                          help="early-stopping patience in epochs")

    runtime = parser.add_argument_group("run identity and machine")
    runtime.add_argument("--project", type=Path, default=repo_root / "runs" / "neu_det",
                         help="directory holding all comparable runs (default: runs/neu_det)")
    runtime.add_argument("--name", default=None,
                         help="run name (default: derived from the model stem)")
    runtime.add_argument("--device", default=None,
                         help="'0', 'cpu', '0,1'... default: Ultralytics auto-selection")
    runtime.add_argument("--workers", type=int, default=8, help="dataloader workers")

    escape = parser.add_argument_group("escape hatch")
    escape.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a pinned hyperparameter; requires --allow-uncontrolled")
    escape.add_argument("--allow-uncontrolled", action="store_true",
                        help="acknowledge that --set makes this run non-comparable to the others")
    escape.add_argument("--skip-fingerprint", action="store_true",
                        help="skip hashing every dataset file (faster startup, weaker provenance)")
    escape.add_argument("--dry-run", action="store_true",
                        help="resolve and print the configuration, then exit without training")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# STEP 2 - Resolve the run configuration
# --------------------------------------------------------------------------------------

class ConfigError(RuntimeError):
    """Raised when the requested configuration would produce an uncontrolled comparison."""

def build_config(args) -> dict:
    """Resolve the full Ultralytics argument set, validating the controlled ones."""
    if args.batch <= 0:
        raise ConfigError(
            "--batch must be a positive integer. Ultralytics accepts a float here to mean "
            "'AutoBatch: fill this fraction of GPU memory', which picks a different batch "
            "size per model and per machine -- the exact confound this script prevents."
        )
    for name, value in (("--epochs", args.epochs), ("--imgsz", args.imgsz)):
        if value <= 0:
            raise ConfigError("%s must be positive, got %s" % (name, value))
    if args.patience < 0:
        raise ConfigError("--patience must be >= 0, got %s" % args.patience)
    if not args.data.is_file():
        raise ConfigError("--data does not exist: %s (run prepare_neu_det.py first)" % args.data)

    explicit = {
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "patience": args.patience,
    }

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise ConfigError("--set expects KEY=VALUE, got %r" % item)
        key, _, raw = item.partition("=")
        key = key.strip()
        if key in explicit:
            raise ConfigError("%s is a required flag; pass --%s instead of --set" % (key, key))
        if key in RUN_IDENTITY_KEYS + ENVIRONMENT_KEYS:
            raise ConfigError("%s has a dedicated flag; use --%s" % (key, key))
        overrides[key] = _parse_scalar(raw)

    if overrides and not args.allow_uncontrolled:
        raise ConfigError(
            "--set %s changes a pinned hyperparameter. Re-run with --allow-uncontrolled if "
            "that is intentional; the override will be recorded in the run config so the "
            "comparison table can flag it." % sorted(overrides)
        )

    controlled = dict(CONTROLLED_HYPERPARAMETERS)
    unknown = sorted(set(overrides) - set(controlled))
    if unknown:
        print("NOTE: --set introduced key(s) not in CONTROLLED_HYPERPARAMETERS: %s\n"
              "      They are passed straight to Ultralytics and recorded, but they are not "
              "part of the pinned baseline." % unknown)
    controlled.update(overrides)

    if controlled["close_mosaic"] >= args.epochs:
        print("WARNING: close_mosaic=%s >= --epochs %s, so mosaic augmentation is disabled for "
              "the whole run." % (controlled["close_mosaic"], args.epochs))

    return {
        "explicit": explicit,
        "controlled": controlled,
        "uncontrolled_overrides": overrides,
    }


# --------------------------------------------------------------------------------------
# STEP 3 - Probe the environment (versions, GPU, RAM)
# --------------------------------------------------------------------------------------

def _torchvision_version():
    try:
        import torchvision

        return torchvision.__version__
    except ImportError:
        return None

def _physical_nvidia_gpus():
    """Ask the driver (not torch) whether an NVIDIA GPU is actually present.

    This is what distinguishes "no GPU in this machine" from "there is a GPU but torch
    cannot talk to it", which are very different problems with very different fixes.
    """
    try:
        import pynvml  # ships with ultralytics as nvidia-ml-py

        pynvml.nvmlInit()
        try:
            names = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                names.append(name.decode() if isinstance(name, bytes) else name)
            return names
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode == 0:
            return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    return []

def environment_info() -> dict:
    import torch
    import ultralytics

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "gpu_memory_gib": [
            round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 2)
            for i in range(torch.cuda.device_count())
        ],
        "torchvision": _torchvision_version(),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        import psutil  # ships with ultralytics

        memory = psutil.virtual_memory()
        info["host_ram_gib"] = round(memory.total / 1024 ** 3, 2)
        info["host_ram_available_gib"] = round(memory.available / 1024 ** 3, 2)
    except Exception:
        info["host_ram_gib"] = info["host_ram_available_gib"] = None
    # Ultralytics silently adds an extra augmentation pipeline when albumentations is
    # importable. Two machines with different extras installed therefore train differently
    # from identical commands, so its presence and version are part of the record.
    try:
        import albumentations

        info["albumentations"] = albumentations.__version__
    except ImportError:
        info["albumentations"] = None
    return info

def check_compute_device(args, environment: dict) -> None:
    """Refuse to start a silent CPU run when the machine clearly has a usable GPU.

    A nano YOLO on 1,260 images is roughly 20-40x slower on CPU than on a modern GPU --
    hours instead of minutes. That is not a failure mode worth discovering from a progress
    bar three hours in, so training aborts unless CPU was asked for explicitly.
    """
    wants_cpu = str(args.device or "").lower().startswith("cpu")
    if environment["cuda_available"]:
        if wants_cpu:
            print("\nNOTE: a CUDA GPU is available but --device cpu was requested. Proceeding "
                  "on CPU.")
        return
    if wants_cpu:
        print("\n--device cpu requested; torch has no CUDA support in this environment. "
              "Proceeding on CPU.")
        return

    physical = _physical_nvidia_gpus()
    cpu_only_build = environment["cuda_version"] is None
    lines = [
        "",
        "!" * 78,
        "REFUSING TO TRAIN ON CPU",
        "!" * 78,
        "torch.cuda.is_available() is False, so this run would use the CPU.",
        "    torch        : %s" % environment["torch"],
        "    torchvision  : %s" % environment["torchvision"],
        "    built w/ CUDA: %s" % (environment["cuda_version"] or "no -- this is a CPU-only build"),
        "    NVIDIA GPUs visible to the driver: %s" % (physical or "none"),
        "",
    ]
    if physical and cpu_only_build:
        lines += [
            "This machine HAS a GPU, but the installed torch is a CPU-only wheel, so no",
            "--device value can reach it.",
            "",
            "The usual cause: torchvision was installed from PyPI. Its Windows wheels are",
            "CPU-only and pin an exact torch version, so pip replaces a working +cuXXX torch",
            "with the matching +cpu build. Install both from the PyTorch CUDA index together:",
            "",
            "    pip install --index-url https://download.pytorch.org/whl/cu130 \\",
            "        torch==%s+cu130 torchvision==%s+cu130"
            % (str(environment["torch"]).split("+")[0],
               str(environment["torchvision"] or "").split("+")[0]),
            "",
            "(Pick the cuXXX that matches your driver; check `nvidia-smi`.)",
        ]
    elif physical:
        lines += [
            "torch was built with CUDA %s but cannot see the GPU. This is usually a driver or"
            % environment["cuda_version"],
            "toolkit mismatch: check `nvidia-smi` and that your driver is new enough for that",
            "CUDA version.",
        ]
    else:
        lines += [
            "No NVIDIA GPU was found on this machine. If that is expected, re-run with",
            "--device cpu to say so explicitly; the choice is then recorded in the run config.",
        ]
    lines += [
        "",
        "To train on CPU anyway, pass --device cpu.",
        "!" * 78,
    ]
    raise ConfigError("\n".join(lines))

# Rough host-RAM budget, measured on this project (1,260 images of 200x200, imgsz 640).
# Windows spawns DataLoader workers as fresh processes that each re-import torch, and the
# CUDA-enabled wheels are large, so a worker costs far more than the image data it carries.
# Ultralytics keeps a train loader and a val loader alive at the epoch boundary, hence the
# factor of two -- and the epoch boundary is exactly where memory exhaustion bites.
_TRAIN_PROCESS_GIB = 2.0
_PER_WORKER_GIB = 0.55

def check_host_memory(args, environment: dict) -> dict:
    """Warn when the requested worker count will not fit in available host RAM.

    Under memory exhaustion the failures are lurid and misleading -- a DataLoader worker
    exiting "unexpectedly", cuDNN returning CUDNN_STATUS_EXECUTION_FAILED, or torch.save
    reporting "I/O operation on closed file" because an allocation failed mid-serialisation.
    None of those name the real problem, so it is worth predicting up front.

    This is a heuristic, so it warns rather than aborts.
    """
    total = environment.get("host_ram_gib")
    available = environment.get("host_ram_available_gib")
    estimate = _TRAIN_PROCESS_GIB + 2 * args.workers * _PER_WORKER_GIB
    report = {"estimated_peak_gib": round(estimate, 2), "available_gib": available}
    if total is None or available is None:
        return report

    headroom = available - estimate
    report["headroom_gib"] = round(headroom, 2)
    print("\nhost RAM: %.1f GiB total, %.1f GiB available | estimated peak for %d workers: "
          "%.1f GiB" % (total, available, args.workers, estimate))
    if headroom >= 1.0:
        return report

    safe = max(0, int((available - _TRAIN_PROCESS_GIB - 1.0) / (2 * _PER_WORKER_GIB)))
    safe = min(safe, args.workers)
    report["recommended_workers"] = safe
    print("\n" + "!" * 78)
    print("LOW MEMORY WARNING -- %d dataloader workers may not fit" % args.workers)
    print("!" * 78)
    print("Estimated peak %.1f GiB against %.1f GiB available. Training will probably die a"
          % (estimate, available))
    print("few epochs in, and the error will NOT mention memory -- expect one of:")
    print("    RuntimeError: DataLoader worker (pid ...) exited unexpectedly")
    print("    RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED")
    print("    ValueError: I/O operation on closed file   (inside torch.save)")
    print("")
    print("Suggested: --workers %d" % safe)
    print("On small images this costs no throughput -- the GPU, not the loader, is the")
    print("bottleneck. Whatever you pick, use the SAME value for every model you compare;")
    print("--workers is recorded per run and the comparison check reports any difference.")
    print("!" * 78)
    return report


# --------------------------------------------------------------------------------------
# STEP 4 - Fingerprint the dataset and the repo
# --------------------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _git_state(repo_root: Path) -> dict:
    def run(*args):
        try:
            out = subprocess.run(
                args, cwd=str(repo_root), capture_output=True, text=True, timeout=15
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }

def dataset_fingerprint(data_yaml: Path, skip: bool = False) -> dict:
    """Hash the dataset so a config JSON proves which bytes the model actually saw.

    Comparing only the data.yaml path is not enough -- the same path can point at a
    regenerated dataset with a different split.
    """
    info = {"data_yaml": str(data_yaml.resolve()), "data_yaml_sha256": _sha256_file(data_yaml)}
    manifest = data_yaml.parent / "split_manifest.json"
    if manifest.is_file():
        info["split_manifest_sha256"] = _sha256_file(manifest)
    if skip:
        info["content_sha256"] = None
        info["note"] = "content hashing skipped via --skip-fingerprint"
        return info

    root = data_yaml.parent
    digest = hashlib.sha256()
    counts = {}
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            directory = root / split / kind
            if not directory.is_dir():
                continue
            files = sorted(directory.iterdir(), key=lambda p: p.name)
            counts["%s/%s" % (split, kind)] = len(files)
            for path in files:
                digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                digest.update(_sha256_file(path).encode("ascii"))
    info["content_sha256"] = digest.hexdigest()
    info["file_counts"] = counts
    return info


# --------------------------------------------------------------------------------------
# STEP 5 - Guard comparability across runs
# --------------------------------------------------------------------------------------

def comparison_keys(config: dict, fingerprint: dict, model: str) -> dict:
    """The dict that must be identical across runs for the comparison to mean anything."""
    keys = dict(config["controlled"])
    keys.update(config["explicit"])
    keys["dataset_content_sha256"] = fingerprint.get("content_sha256")
    keys["data_yaml_sha256"] = fingerprint.get("data_yaml_sha256")
    # Fine-tuning from COCO weights (.pt) vs training from scratch (.yaml) is a different
    # experiment, so the distinction is compared even though the model name itself is not.
    keys["weights_kind"] = "pretrained_checkpoint" if str(model).endswith(".pt") else "from_scratch"
    return keys

def compare_with_previous_runs(project: Path, this_run: dict) -> None:
    """Diff this run's comparison keys against every other run in the same project."""
    others = []
    if project.is_dir():
        for path in sorted(project.glob("*/" + CONFIG_FILENAME)):
            if path.parent.resolve() == Path(this_run["run"]["save_dir"]).resolve():
                continue
            try:
                others.append((path, json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError) as exc:
                print("    (could not read %s: %s)" % (path, exc))

    print("\n" + "=" * 78)
    print("CONTROLLED-COMPARISON CHECK")
    print("=" * 78)
    if not others:
        print("No other runs in %s yet -- nothing to compare against." % project)
        print("This run's config is the baseline for the next one.")
        return

    mine = this_run["comparison_keys"]
    clean = True
    for path, other in others:
        name = path.parent.name
        theirs = other.get("comparison_keys", {})
        diffs = [
            (key, mine.get(key, "<absent>"), theirs.get(key, "<absent>"))
            for key in sorted(set(mine) | set(theirs))
            if mine.get(key, "<absent>") != theirs.get(key, "<absent>")
        ]
        env_diffs = [
            (key, this_run["environment"].get(key), other.get("environment", {}).get(key))
            for key in ("ultralytics", "torch", "albumentations", "cuda_version")
            if this_run["environment"].get(key) != other.get("environment", {}).get(key)
        ]
        machine_diffs = [
            (key, this_run["machine"].get(key), other.get("machine", {}).get(key))
            for key in ENVIRONMENT_KEYS
            if this_run["machine"].get(key) != other.get("machine", {}).get(key)
        ]

        print("\nvs %-28s model %s -> %s"
              % (name, other.get("model"), this_run["model"]))
        if not diffs:
            print("    OK: identical hyperparameters, identical dataset bytes.")
            print("    The only difference is the model. This comparison is controlled.")
        else:
            clean = False
            print("    !! %d HYPERPARAMETER DIFFERENCE(S) -- THIS COMPARISON IS NOT CONTROLLED:"
                  % len(diffs))
            print("       %-28s %-22s %-22s" % ("key", "this run", name))
            for key, a, b in diffs:
                print("       %-28s %-22s %-22s" % (key, a, b))
        for key, a, b in env_diffs:
            print("    note: %s differs (%s vs %s) -- same config, different software stack"
                  % (key, a, b))
        for key, a, b in machine_diffs:
            print("    note: %s differs (%s vs %s) -- can shift results slightly" % (key, a, b))

    if not clean:
        print("\n" + "!" * 78)
        print("At least one comparison above is uncontrolled. Any mAP delta between those runs")
        print("mixes the architecture change with the hyperparameter change and cannot be")
        print("attributed to either. Re-run with matching settings before drawing conclusions.")
        print("!" * 78)


# --------------------------------------------------------------------------------------
# STEP 6 - Resolve the run directory
# --------------------------------------------------------------------------------------

def unique_run_dir(project: Path, name: str) -> Path:
    """Pick a run directory that does not exist yet, so runs are never silently merged."""
    candidate = project / name
    suffix = 2
    while candidate.exists():
        candidate = project / ("%s_%d" % (name, suffix))
        suffix += 1
    return candidate


# --------------------------------------------------------------------------------------
# STEP 7 - Entry point
# --------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    try:
        config = build_config(args)
    except ConfigError as exc:
        print("\nCONFIG ERROR: %s" % exc, file=sys.stderr)
        return 2

    # Probe the software stack before creating any directories, so a missing dependency
    # cannot leave an empty run directory behind to confuse the comparison scan later.
    try:
        environment = environment_info()
    except ImportError as exc:
        print("\nMISSING DEPENDENCY: %s\n"
              "Install with: pip install -r training/requirements.txt" % exc, file=sys.stderr)
        return 2

    try:
        check_compute_device(args, environment)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2
    memory_report = check_host_memory(args, environment)

    model_stem = Path(str(args.model)).stem
    run_name = args.name or model_stem
    save_dir = unique_run_dir(args.project, run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TRAINING RUN: %s" % save_dir)
    print("=" * 78)
    print("model    : %s" % args.model)
    print("data     : %s" % args.data.resolve())
    print("explicit : %s" % json.dumps(config["explicit"]))
    if config["uncontrolled_overrides"]:
        print("OVERRIDES: %s   <-- this run is not directly comparable to the others"
              % json.dumps(config["uncontrolled_overrides"]))

    print("\nfingerprinting dataset...")
    fingerprint = dataset_fingerprint(args.data, skip=args.skip_fingerprint)
    print("    data.yaml sha256 : %s" % fingerprint["data_yaml_sha256"][:16])
    print("    content   sha256 : %s"
          % (fingerprint["content_sha256"][:16] if fingerprint["content_sha256"] else "skipped"))

    print("\nultralytics %s | torch %s | cuda %s | gpus %s"
          % (environment["ultralytics"], environment["torch"],
             environment["cuda_version"], environment["gpus"] or "none"))
    if environment["albumentations"]:
        print("NOTE: albumentations %s is installed. Ultralytics will apply its extra "
              "augmentation\n      pipeline. Keep it installed (or not) consistently across "
              "the runs you compare." % environment["albumentations"])

    record = {
        "schema": CONFIG_SCHEMA,
        "model": str(args.model),
        "run": {
            "name": save_dir.name,
            "project": str(args.project.resolve()),
            "save_dir": str(save_dir.resolve()),
            "command": " ".join([Path(sys.executable).name, "training/train_yolo.py"] + (
                argv if argv is not None else sys.argv[1:])),
            "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_utc": None,
            "duration_seconds": None,
            "status": "started",
        },
        "explicit": config["explicit"],
        "controlled": config["controlled"],
        "uncontrolled_overrides": config["uncontrolled_overrides"],
        "comparison_keys": comparison_keys(config, fingerprint, str(args.model)),
        "dataset": fingerprint,
        "environment": environment,
        "machine": {"device": args.device, "workers": args.workers, "memory": memory_report},
        "git": _git_state(repo_root),
        "ultralytics_resolved_args": None,
        "results": None,
    }

    config_path = save_dir / CONFIG_FILENAME
    config_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print("\nwrote %s (before training, so it survives a crash)" % config_path)

    if args.dry_run:
        print("\n--dry-run: resolved configuration written, no training started.")
        compare_with_previous_runs(args.project, record)
        return 0

    # Import late so that --dry-run and --help stay instant and the determinism env vars
    # above are already in place.
    from ultralytics import YOLO

    train_kwargs = dict(config["controlled"])
    train_kwargs.update(config["explicit"])
    train_kwargs.update(
        data=str(args.data.resolve()),
        project=str(args.project.resolve()),
        name=save_dir.name,
        exist_ok=True,            # save_dir was just created by unique_run_dir()
        device=args.device,
        workers=args.workers,
        verbose=True,
    )
    # Ultralytics treats an explicit None the same as an absent key for these, but dropping
    # them keeps the resolved-args dump readable.
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    started = time.time()
    status = "failed"
    try:
        model = YOLO(str(args.model))
        results = model.train(**train_kwargs)
        status = "completed"
    finally:
        record["run"]["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        record["run"]["duration_seconds"] = round(time.time() - started, 1)
        record["run"]["status"] = status
        config_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    trainer = getattr(model, "trainer", None)
    actual_dir = Path(getattr(trainer, "save_dir", save_dir))
    record["run"]["save_dir"] = str(actual_dir.resolve())

    if trainer is not None:
        # vars(trainer.args) is the fully resolved configuration Ultralytics actually used,
        # including every default this script did not name. Storing it means a future
        # version bump that changes an unpinned default is still detectable after the fact.
        try:
            record["ultralytics_resolved_args"] = {
                k: v for k, v in sorted(vars(trainer.args).items())
            }
        except TypeError:
            record["ultralytics_resolved_args"] = str(trainer.args)
        record["results"] = {
            "best_weights": str(getattr(trainer, "best", "")),
            "last_weights": str(getattr(trainer, "last", "")),
            "epochs_completed": getattr(trainer, "epoch", None),
            "best_fitness": float(trainer.best_fitness) if getattr(trainer, "best_fitness", None) is not None else None,
        }
    try:
        record["results"] = dict(record["results"] or {})
        record["results"]["metrics"] = {
            k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))
        }
        record["results"]["map50"] = float(results.box.map50)
        record["results"]["map50_95"] = float(results.box.map)
    except AttributeError:
        pass

    config_path = actual_dir / CONFIG_FILENAME
    config_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)
    print("run dir : %s" % actual_dir)
    print("weights : %s" % record["results"].get("best_weights"))
    print("config  : %s" % config_path)
    if record["results"].get("map50") is not None:
        print("val mAP@0.5      : %.4f" % record["results"]["map50"])
        print("val mAP@0.5:0.95 : %.4f" % record["results"]["map50_95"])
    print("\nThese are validation-split numbers reported by training. Evaluate the held-out")
    print("test split with training/eval_yolo.py before quoting anything.")

    compare_with_previous_runs(args.project, record)

    print("\nNext:")
    print('    python training/eval_yolo.py --weights "%s" \\' % record["results"].get("best_weights"))
    print('        --data "%s" --split test --imgsz %d --batch %d'
          % (args.data.resolve(), args.imgsz, args.batch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
