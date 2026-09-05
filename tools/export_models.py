#!/usr/bin/env python
"""Export trained NEU-DET detectors to ONNX (FP32 and INT8) and verify the exports.

Produces four artefacts by default:

    artifacts/onnx/yolov8n_fp32.onnx     artifacts/onnx/yolov8n_int8.onnx
    artifacts/onnx/yolo26n_fp32.onnx     artifacts/onnx/yolo26n_int8.onnx

Two things this script is careful about:

  1. **Calibration data comes only from the training split.** Ultralytics' INT8 calibration
     dataloader defaults to `data[args.split or "val"]`, so without an explicit split it
     would calibrate on validation images -- data the model is scored against. Every INT8
     export here passes split="train".

  2. **Every export is checked against the PyTorch model before it is trusted.** A silently
     wrong export is worse than a failed one: it benchmarks fast and scores plausibly while
     detecting the wrong things. Each ONNX file runs the same images through the same
     Ultralytics pipeline as the .pt and the detections are matched box-for-box.

Usage
-----
    python tools/export_models.py
    python tools/export_models.py --imgsz 320 --suffix _320
    python tools/export_models.py --weights runs/neu_det/yolo26n/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Repository root on sys.path so the shared helpers resolve when this script is run
# directly from any working directory.
sys.path.insert(0, str(REPO_ROOT))
from common.metrics import iou_matrix  # noqa: E402

# Inference settings held identical across PyTorch and ONNX so parity measures the export,
# not a difference in how the two were invoked.
PARITY_CONF = 0.25
PARITY_IOU = 0.7
PARITY_MAX_DET = 300

# --- verification gates ------------------------------------------------------------------
#
# The two export variants need different acceptance criteria, because they make different
# promises.
#
# FP32 is a *lossless* re-expression of the same arithmetic in another runtime. Detections
# must correspond one-to-one and land on the same pixels; anything else is a real bug in the
# export. This gate is strict on purpose and both models clear it with mean IoU 1.000000.
#
# INT8 is *lossy* by construction -- activations are rounded to 256 levels. The question is
# not "are the detections identical" (they cannot be) but "does the model still do its job".
# So the authoritative INT8 gate is task-metric preservation on the held-out test split, and
# per-detection matching is retained only as a diagnostic with a catastrophic-failure floor.
#
# This distinction matters most for YOLO26, whose head is end-to-end (NMS-free): a learned
# one-to-one assignment picks the surviving box, so a small activation perturbation can
# change *which* candidate wins even when the detection set is just as good. Measured on
# this checkpoint: yolo26n INT8 matches only ~46% of PyTorch detections box-for-box, yet
# loses just 1.07 pp of mAP@0.5. Judging it by detection identity would reject a working
# artefact; judging it by mAP tells the truth. YOLOv8's NMS head breaks ties on raw score
# and is far more stable under the same perturbation (~84% box-for-box).
FP32_GATE = {
    "kind": "strict",
    "min_matched_fraction": 1.0,
    "min_mean_iou": 0.99,
    "max_conf_delta": 0.02,
    "max_extra_detections": 0,
    "borderline_band": 0.02,
    "max_map50_drop": 0.0005,      # FP32 must reproduce mAP to within float noise
    "max_map5095_drop": 0.0005,
}
INT8_GATE = {
    "kind": "lossy",
    # Catastrophic-failure floor only. A broken export (wrong tensor layout, failed
    # calibration, garbage boxes) lands near 0% matched with low IoU; these thresholds
    # catch that while tolerating the assignment churn described above.
    "min_matched_fraction": 0.30,
    "min_mean_iou": 0.85,
    "max_conf_delta": 0.50,
    "max_extra_detections": 25,
    "borderline_band": 0.02,
    # The gate that actually decides whether the artefact ships.
    "max_map50_drop": 0.02,        # 2 pp absolute
    "max_map5095_drop": 0.02,
}


class ExportError(RuntimeError):
    """Raised when an export fails or fails to match the PyTorch model."""


@dataclass
class Artefact:
    name: str
    variant: str          # "fp32" | "int8"
    path: Path
    source_weights: Path
    imgsz: int
    export_seconds: float = 0.0
    parity: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 ** 2


# ---------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------


def export_fp32(weights: Path, imgsz: int, staging: Path) -> Path:
    """Export the FP32 ONNX graph.

    Exported with nms=False (the default) so the ONNX carries the same raw head output the
    .pt does, and Ultralytics applies identical postprocessing to both. Fusing NMS into the
    graph would make the two pipelines structurally different and the comparison meaningless.
    """
    from ultralytics import YOLO

    produced = YOLO(str(weights)).export(
        format="onnx",
        imgsz=imgsz,
        batch=1,
        device="cpu",
        simplify=True,
        dynamic=False,
        verbose=False,
    )
    return Path(produced)


def export_int8(weights: Path, imgsz: int, data: Path, calib_fraction: float, split: str) -> Path:
    """Export the INT8 ONNX graph, calibrated on `split` (train).

    Ultralytics runs ONNX Runtime static quantization over Conv/Gemm/MatMul only, leaving
    the detection head's decode arithmetic in float -- a single INT8 scale cannot span box
    coordinates (0..imgsz) and class probabilities (0..1) without collapsing the latter.
    """
    from ultralytics import YOLO

    produced = YOLO(str(weights)).export(
        format="onnx",
        imgsz=imgsz,
        batch=1,
        device="cpu",
        simplify=True,
        dynamic=False,
        quantize=8,
        data=str(data),
        split=split,          # the whole point: calibrate on training images only
        fraction=calib_fraction,
        verbose=False,
    )
    return Path(produced)


# ---------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------


def _detections(model, image_paths, imgsz: int):
    """Run a model over images and return (boxes, classes, confs) per image."""
    out = []
    for path in image_paths:
        result = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=PARITY_CONF,
            iou=PARITY_IOU,
            max_det=PARITY_MAX_DET,
            device="cpu",
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            out.append((np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)))
            continue
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy().astype(np.float64)
        order = np.argsort(-conf)
        out.append((xyxy[order], cls[order], conf[order]))
    return out


def check_parity(reference_model, candidate_model, image_paths, imgsz: int, gate: dict,
                 label: str, match_iou: float) -> dict:
    """Match candidate detections against reference detections, image by image.

    Greedy by descending reference confidence: each reference detection claims the
    highest-IoU unclaimed candidate detection *of the same class*.
    """
    reference = _detections(reference_model, image_paths, imgsz)
    candidate = _detections(candidate_model, image_paths, imgsz)

    total_ref = total_matched = total_extra = 0
    borderline_missed = borderline_extra = 0
    ious: list[float] = []
    conf_deltas: list[float] = []
    class_mismatches: list[str] = []
    worst: list[dict] = []

    for path, (rb, rc, rconf), (cb, cc, cconf) in zip(image_paths, reference, candidate):
        total_ref += len(rb)
        matrix = iou_matrix(rb, cb)
        claimed = np.zeros(len(cb), dtype=bool)

        for i in range(len(rb)):
            options = [
                j for j in range(len(cb))
                if not claimed[j] and cc[j] == rc[i] and matrix[i, j] >= match_iou
            ]
            if options:
                j = max(options, key=lambda j: matrix[i, j])
                claimed[j] = True
                total_matched += 1
                ious.append(float(matrix[i, j]))
                conf_deltas.append(abs(float(rconf[i]) - float(cconf[j])))
                continue

            # Unmatched reference detection. Distinguish "quantisation nudged it across the
            # confidence threshold" from "the export is wrong".
            if abs(float(rconf[i]) - PARITY_CONF) <= gate["borderline_band"]:
                borderline_missed += 1
                continue
            best_any = int(np.argmax(matrix[i])) if len(cb) else -1
            if best_any >= 0 and matrix[i, best_any] >= match_iou and cc[best_any] != rc[i]:
                class_mismatches.append(
                    f"{Path(path).name}: reference class {int(rc[i])} @ conf {rconf[i]:.3f} "
                    f"matched geometrically by candidate class {int(cc[best_any])}"
                )
            worst.append({
                "image": Path(path).name,
                "issue": "missing_in_candidate",
                "class": int(rc[i]),
                "conf": round(float(rconf[i]), 4),
                "best_iou_any": round(float(matrix[i, best_any]) if best_any >= 0 else 0.0, 4),
            })

        for j in np.flatnonzero(~claimed):
            if abs(float(cconf[j]) - PARITY_CONF) <= gate["borderline_band"]:
                borderline_extra += 1
                continue
            total_extra += 1
            worst.append({
                "image": Path(path).name,
                "issue": "extra_in_candidate",
                "class": int(cc[j]),
                "conf": round(float(cconf[j]), 4),
            })

    matched_fraction = total_matched / total_ref if total_ref else 1.0
    report = {
        "label": label,
        "images": len(image_paths),
        "reference_detections": total_ref,
        "matched": total_matched,
        "matched_fraction": round(matched_fraction, 4),
        "extra_in_candidate": total_extra,
        "borderline_threshold_flips": borderline_missed + borderline_extra,
        "mean_iou_of_matches": round(float(np.mean(ious)), 6) if ious else 0.0,
        "min_iou_of_matches": round(float(np.min(ious)), 6) if ious else 0.0,
        "max_conf_delta": round(float(np.max(conf_deltas)), 6) if conf_deltas else 0.0,
        "class_mismatches": class_mismatches,
        "gate": gate,
        "examples": worst[:10],
    }

    failures = []
    if total_ref == 0:
        failures.append("the PyTorch reference produced no detections at all -- cannot verify parity")
    if matched_fraction < gate["min_matched_fraction"]:
        failures.append(
            f"only {matched_fraction:.1%} of reference detections matched "
            f"(gate: {gate['min_matched_fraction']:.0%})"
        )
    if ious and report["mean_iou_of_matches"] < gate["min_mean_iou"]:
        failures.append(
            f"mean IoU of matched boxes {report['mean_iou_of_matches']:.4f} "
            f"< {gate['min_mean_iou']}"
        )
    if report["max_conf_delta"] > gate["max_conf_delta"]:
        failures.append(
            f"max confidence delta {report['max_conf_delta']:.4f} > {gate['max_conf_delta']}"
        )
    if total_extra > gate["max_extra_detections"]:
        failures.append(
            f"{total_extra} detections present in ONNX but not PyTorch "
            f"(allowed: {gate['max_extra_detections']})"
        )
    if class_mismatches:
        failures.append(f"{len(class_mismatches)} box(es) matched geometrically but with a different class")

    report["passed"] = not failures
    report["failures"] = failures
    return report


def measure_map(model, data: Path, imgsz: int, split: str = "test") -> dict:
    """Score a model on the held-out split with Ultralytics' validator."""
    result = model.val(
        data=str(data), split=split, imgsz=imgsz, batch=1, device="cpu",
        rect=False, plots=False, verbose=False,
    )
    return {"map50": float(result.box.map50), "map50_95": float(result.box.map)}


def check_map_preservation(reference: dict, candidate: dict, gate: dict) -> dict:
    """Compare candidate mAP against the PyTorch reference.

    For INT8 this is the authoritative gate: an export is acceptable when it still does the
    job, not when it reproduces every individual box.
    """
    drop50 = reference["map50"] - candidate["map50"]
    drop5095 = reference["map50_95"] - candidate["map50_95"]
    failures = []
    if drop50 > gate["max_map50_drop"]:
        failures.append(
            f"mAP@0.5 dropped {drop50:.4f} ({drop50 * 100:.2f} pp), "
            f"limit {gate['max_map50_drop'] * 100:.2f} pp"
        )
    if drop5095 > gate["max_map5095_drop"]:
        failures.append(
            f"mAP@0.5:0.95 dropped {drop5095:.4f} ({drop5095 * 100:.2f} pp), "
            f"limit {gate['max_map5095_drop'] * 100:.2f} pp"
        )
    return {
        "reference": reference,
        "candidate": candidate,
        "map50_drop": round(drop50, 6),
        "map50_95_drop": round(drop5095, 6),
        "passed": not failures,
        "failures": failures,
    }


def print_map(report: dict, label: str) -> None:
    ref, cand = report["reference"], report["candidate"]
    status = "PASS" if report["passed"] else "FAIL"
    print(f"\n  test-split mAP {label}: {status}")
    print(f"    mAP@0.5      pytorch {ref['map50']:.4f} -> onnx {cand['map50']:.4f}  "
          f"(delta {-report['map50_drop']:+.4f}, {-report['map50_drop'] * 100:+.2f} pp)")
    print(f"    mAP@0.5:0.95 pytorch {ref['map50_95']:.4f} -> onnx {cand['map50_95']:.4f}  "
          f"(delta {-report['map50_95_drop']:+.4f}, {-report['map50_95_drop'] * 100:+.2f} pp)")
    for line in report["failures"]:
        print(f"    !! {line}")


def print_parity(report: dict) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    role = "gate" if report["gate"]["kind"] == "strict" else "diagnostic"
    print(f"\n  detection parity {report['label']}: {status}  ({role})")
    print(f"    reference detections   : {report['reference_detections']} over {report['images']} images")
    print(f"    matched                : {report['matched']} ({report['matched_fraction']:.1%})")
    print(f"    extra in ONNX          : {report['extra_in_candidate']}")
    print(f"    borderline conf flips  : {report['borderline_threshold_flips']} "
          f"(within {report['gate']['borderline_band']} of conf={PARITY_CONF}; "
          f"not counted as failures)")
    print(f"    IoU of matches         : mean {report['mean_iou_of_matches']:.6f}, "
          f"min {report['min_iou_of_matches']:.6f}")
    print(f"    max confidence delta   : {report['max_conf_delta']:.6f}")
    if report["class_mismatches"]:
        print("    CLASS MISMATCHES:")
        for line in report["class_mismatches"][:5]:
            print(f"      {line}")
    for line in report["failures"]:
        print(f"    !! {line}")
    for example in report["examples"][:5]:
        print(f"       {example}")


# ---------------------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export NEU-DET detectors to ONNX FP32/INT8 and verify parity with PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, nargs="+", default=[
        REPO_ROOT / "runs" / "neu_det" / "yolov8n" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "neu_det" / "yolo26n" / "weights" / "best.pt",
    ], help="trained .pt checkpoints to export")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "neu_det" / "data.yaml")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts" / "onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--suffix", default="", help="appended to artefact names, e.g. _320")
    parser.add_argument("--calib-images", type=int, default=512,
                        help="training images used for INT8 calibration (>300 recommended)")
    parser.add_argument("--calib-split", default="train", choices=("train", "val", "test"),
                        help="split to calibrate from; train is the only correct choice")
    parser.add_argument("--parity-images", type=int, default=20,
                        help="test-split images used for the PyTorch/ONNX comparison")
    parser.add_argument("--match-iou", type=float, default=0.5,
                        help="IoU at which a candidate box is considered the same detection")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--skip-parity", action="store_true",
                        help="export without verifying -- not recommended")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO

    if args.calib_split != "train":
        print(f"WARNING: calibrating on '{args.calib_split}'. Calibration data must come from "
              f"the training split, or the INT8 model has seen the data it is scored on.")

    test_images = sorted((args.data.parent / "test" / "images").glob("*.jpg"))
    if not test_images and not args.skip_parity:
        raise ExportError(f"no test images under {args.data.parent / 'test' / 'images'}")
    parity_images = test_images[: args.parity_images]

    # Calibration fraction is expressed as a count for legibility, then converted.
    train_images = sorted((args.data.parent / "train" / "images").glob("*.jpg"))
    calib_fraction = min(1.0, args.calib_images / max(len(train_images), 1))

    args.out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPORT")
    print("=" * 78)
    print(f"imgsz            : {args.imgsz}")
    print(f"output           : {args.out}")
    print(f"calibration      : {args.calib_split} split, "
          f"{min(args.calib_images, len(train_images))}/{len(train_images)} images "
          f"(fraction {calib_fraction:.4f})")
    print(f"parity images    : {len(parity_images)} from the held-out test split")
    print(f"parity inference : conf={PARITY_CONF} iou={PARITY_IOU} max_det={PARITY_MAX_DET} device=cpu")

    artefacts: list[Artefact] = []
    failures: list[str] = []

    for weights in args.weights:
        if not weights.is_file():
            raise ExportError(f"weights not found: {weights}")
        name = weights.parent.parent.name  # runs/neu_det/<name>/weights/best.pt
        print("\n" + "-" * 78)
        print(f"{name}  <-  {weights}")
        print("-" * 78)

        reference = YOLO(str(weights))
        reference_map = None
        if not args.skip_parity:
            print("  scoring the PyTorch reference on the test split...")
            reference_map = measure_map(reference, args.data, args.imgsz)
            print(f"    mAP@0.5 {reference_map['map50']:.4f}   "
                  f"mAP@0.5:0.95 {reference_map['map50_95']:.4f}")

        # --- FP32 -----------------------------------------------------------------------
        started = time.time()
        produced = export_fp32(weights, args.imgsz, args.out)
        fp32_path = args.out / f"{name}{args.suffix}_fp32.onnx"
        # Copy rather than move: the INT8 export below regenerates and then DELETES the
        # FP32 file next to the checkpoint, so the artefact must already be safely aside.
        shutil.copy2(produced, fp32_path)
        produced.unlink(missing_ok=True)
        fp32 = Artefact(name, "fp32", fp32_path, weights, args.imgsz, time.time() - started)
        artefacts.append(fp32)
        print(f"  FP32 -> {fp32_path.name}  {fp32.size_mb:.2f} MB  ({fp32.export_seconds:.1f}s)")

        if not args.skip_parity:
            fp32.parity = check_parity(reference, YOLO(str(fp32_path)), parity_images,
                                       args.imgsz, FP32_GATE, f"{name} fp32 vs pytorch",
                                       args.match_iou)
            print_parity(fp32.parity)
            fp32.metrics = check_map_preservation(
                reference_map, measure_map(YOLO(str(fp32_path)), args.data, args.imgsz), FP32_GATE)
            print_map(fp32.metrics, f"{name} fp32")
            # FP32 is lossless: both the detection gate and the metric gate must hold.
            if not fp32.parity["passed"]:
                failures.append(f"{name} FP32 ONNX failed detection parity")
            if not fp32.metrics["passed"]:
                failures.append(f"{name} FP32 ONNX failed mAP preservation")

        # --- INT8 -----------------------------------------------------------------------
        if args.skip_int8:
            continue
        started = time.time()
        produced = export_int8(weights, args.imgsz, args.data, calib_fraction, args.calib_split)
        int8_path = args.out / f"{name}{args.suffix}_int8.onnx"
        shutil.copy2(produced, int8_path)
        produced.unlink(missing_ok=True)
        int8 = Artefact(name, "int8", int8_path, weights, args.imgsz, time.time() - started)
        artefacts.append(int8)
        print(f"\n  INT8 -> {int8_path.name}  {int8.size_mb:.2f} MB  ({int8.export_seconds:.1f}s)")

        if not args.skip_parity:
            int8.parity = check_parity(reference, YOLO(str(int8_path)), parity_images,
                                       args.imgsz, INT8_GATE, f"{name} int8 vs pytorch",
                                       args.match_iou)
            print_parity(int8.parity)
            int8.metrics = check_map_preservation(
                reference_map, measure_map(YOLO(str(int8_path)), args.data, args.imgsz), INT8_GATE)
            print_map(int8.metrics, f"{name} int8")
            # For INT8 the metric gate decides; the detection floor only catches a broken export.
            if not int8.metrics["passed"]:
                failures.append(f"{name} INT8 ONNX failed mAP preservation")
            if not int8.parity["passed"]:
                failures.append(f"{name} INT8 ONNX fell below the catastrophic-failure floor")

    # --- artefact sizes ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ARTEFACTS")
    print("=" * 78)
    print(f"{'artefact':<34}{'variant':>9}{'bytes':>14}{'MB':>9}{'vs .pt':>9}")
    print("-" * 75)
    for artefact in artefacts:
        source_mb = artefact.source_weights.stat().st_size / 1024 ** 2
        print(f"{artefact.path.name:<34}{artefact.variant:>9}{artefact.size_bytes:>14,}"
              f"{artefact.size_mb:>9.2f}{artefact.size_mb / source_mb:>8.2f}x")
    print("-" * 75)
    for weights in args.weights:
        print(f"{'(source) ' + weights.parent.parent.name + '/best.pt':<34}{'pt':>9}"
              f"{weights.stat().st_size:>14,}{weights.stat().st_size / 1024 ** 2:>9.2f}"
              f"{1.0:>8.2f}x")

    manifest = args.out / f"export_manifest{args.suffix}.json"
    manifest.write_text(json.dumps({
        "imgsz": args.imgsz,
        "calibration": {
            "split": args.calib_split,
            "requested_images": args.calib_images,
            "available_train_images": len(train_images),
            "fraction": calib_fraction,
        },
        "parity_settings": {
            "images": len(parity_images),
            "conf": PARITY_CONF, "iou": PARITY_IOU, "max_det": PARITY_MAX_DET,
            "match_iou": args.match_iou,
        },
        "artefacts": [{
            "name": a.name, "variant": a.variant, "file": a.path.name,
            "bytes": a.size_bytes, "mb": round(a.size_mb, 4),
            "export_seconds": round(a.export_seconds, 2),
            "parity": a.parity,
            "metrics": a.metrics,
        } for a in artefacts],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest}")

    if failures:
        print("\n" + "!" * 78)
        print("EXPORT VERIFICATION FAILED")
        print("!" * 78)
        for line in failures:
            print(f"  {line}")
        print("\nThese artefacts must not be benchmarked or shipped: an export that does not")
        print("reproduce the PyTorch detections will still produce plausible-looking numbers.")
        print("!" * 78)
        return 1

    if not args.skip_parity:
        print("\nAll exports reproduce the PyTorch detections within their gates.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"\nEXPORT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
