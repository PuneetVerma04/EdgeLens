#!/usr/bin/env python
"""Evaluate a trained YOLO run on a NEU-DET split and produce material for error analysis.

Produces, for any trained run:

  * mAP@0.5 and mAP@0.5:0.95                     -> metrics.json, stdout
  * per-class AP (plus precision/recall)         -> per_class_ap.csv, metrics.json
  * a confusion matrix                           -> confusion_matrix{,_normalized}.csv + .png
  * the 20 worst false positives and 20 worst
    false negatives, as annotated images         -> false_positives/, false_negatives/
  * every FP and FN (not just the worst 20)      -> error_analysis.json
  * a human-readable digest                      -> summary.md

Pipeline
--------
    1. Command line                    parse_args
    2. Ultralytics validator (mAP)     run_validator, extract_metrics
    3. Confusion matrix                write_confusion_matrix
    4. Error analysis at one threshold analyse_errors  (uses iou_matrix, load_ground_truth)
    5. Annotated worst-case renders    render_worst -> render_error
    6. Markdown summary                write_summary
    7. Orchestration                   main

Usage
-----
    python training/eval_yolo.py --weights runs/neu_det/yolov8n/weights/best.pt \
        --data data/neu_det/data.yaml --split test --imgsz 640 --batch 16

The mAP numbers come from Ultralytics' own validator, so they are directly comparable with
the numbers printed during training. The FP/FN analysis is computed independently by this
script at a single, explicit operating point (--error-conf, --match-iou) because "worst"
only means something once you fix a threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import yaml
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("Missing dependencies: pip install -r training/requirements.txt")

# A prediction overlapping no ground truth by more than this is a hallucination rather than
# a sloppy localisation. Used only to label FPs in the report, never in the metrics.
BACKGROUND_IOU = 0.1

# BGR-free RGB palette, one colour per class id, plus roles.
CLASS_COLORS = [
    (232, 106, 51), (60, 160, 220), (120, 200, 100),
    (200, 120, 210), (240, 200, 60), (90, 210, 200),
]
GT_COLOR = (70, 200, 90)
PRED_COLOR = (80, 150, 235)
HIGHLIGHT_FP = (235, 60, 60)
HIGHLIGHT_FN = (250, 170, 30)


# --------------------------------------------------------------------------------------
# STEP 1 - Command line
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO run and generate error-analysis material.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, type=Path, help="path to best.pt / last.pt")
    parser.add_argument("--data", required=True, type=Path, help="data.yaml from prepare_neu_det.py")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640,
                        help="must match the imgsz the model was trained at")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default: <run dir>/eval_<split>)")

    thresholds = parser.add_argument_group("thresholds")
    thresholds.add_argument("--val-conf", type=float, default=0.001,
                            help="confidence floor for the mAP computation; keep this low")
    thresholds.add_argument("--nms-iou", type=float, default=0.7,
                            help="NMS IoU threshold (ignored by NMS-free heads such as YOLO26's)")
    thresholds.add_argument("--max-det", type=int, default=300)
    thresholds.add_argument("--error-conf", type=float, default=0.25,
                            help="operating point for the FP/FN analysis")
    thresholds.add_argument("--match-iou", type=float, default=0.5,
                            help="IoU at which a prediction is deemed to match a ground-truth box")

    parser.add_argument("--topk", type=int, default=20,
                        help="how many worst FPs and FNs to render as annotated images")
    parser.add_argument("--render-size", type=int, default=640,
                        help="target long edge for the annotated renders (NEU-DET is 200px)")
    parser.add_argument("--skip-renders", action="store_true",
                        help="compute the error lists but do not draw the annotated images")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# STEP 2 - Run the Ultralytics validator (mAP)
# --------------------------------------------------------------------------------------

def run_validator(model, args, data_yaml: Path, out_dir: Path):
    """Run Ultralytics' validator and pull the standard detection metrics out of it."""
    results = model.val(
        data=str(data_yaml.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.val_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
        rect=False,        # model.val() defaults this to True; pin it so runs stay comparable
        plots=True,
        save_json=False,
        project=str(out_dir),
        name="ultralytics_val",
        exist_ok=True,
        verbose=True,
    )
    return results

def extract_metrics(results, class_names):
    box = results.box
    per_class = []
    for i, class_id in enumerate(box.ap_class_index):
        precision, recall, ap50, ap = box.class_result(i)
        per_class.append({
            "class_id": int(class_id),
            "class_name": class_names.get(int(class_id), str(class_id)),
            "precision": float(precision),
            "recall": float(recall),
            "ap50": float(ap50),
            "ap50_95": float(ap),
        })
    seen = {entry["class_id"] for entry in per_class}
    for class_id, name in sorted(class_names.items()):
        if class_id not in seen:
            # A class with zero ground-truth boxes in this split gets no AP at all; say so
            # rather than reporting it as 0.0, which would drag the mean down misleadingly.
            per_class.append({
                "class_id": int(class_id), "class_name": name,
                "precision": None, "recall": None, "ap50": None, "ap50_95": None,
                "note": "no ground-truth boxes in this split",
            })
    per_class.sort(key=lambda e: e["class_id"])

    mean_precision, mean_recall, map50, map50_95 = box.mean_results()
    return {
        "map50": float(map50),
        "map50_95": float(map50_95),
        "mean_precision": float(mean_precision),
        "mean_recall": float(mean_recall),
        "fitness": float(results.fitness) if hasattr(results, "fitness") else None,
        "speed_ms_per_image": {k: float(v) for k, v in getattr(results, "speed", {}).items()},
        "per_class": per_class,
    }


# --------------------------------------------------------------------------------------
# STEP 3 - Confusion matrix
# --------------------------------------------------------------------------------------

def write_confusion_matrix(results, class_names, out_dir: Path):
    """Dump Ultralytics' confusion matrix as CSV and a self-contained PNG.

    Ultralytics orients the matrix as matrix[predicted, ground_truth] with an extra final
    row/column for background: the last column is a false positive (predicted something,
    no ground truth) and the last row is a false negative (ground truth, predicted nothing).
    """
    cm = getattr(results, "confusion_matrix", None)
    if cm is None or getattr(cm, "matrix", None) is None:
        print("WARNING: the validator returned no confusion matrix; skipping.")
        return None

    matrix = np.asarray(cm.matrix, dtype=np.float64)
    labels = [class_names[i] for i in sorted(class_names)] + ["background"]
    if matrix.shape != (len(labels), len(labels)):
        labels = [str(i) for i in range(matrix.shape[0] - 1)] + ["background"]

    def dump(path, data, fmt):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["predicted \\ ground_truth"] + labels)
            for name, row in zip(labels, data):
                writer.writerow([name] + [fmt(v) for v in row])

    raw_path = out_dir / "confusion_matrix.csv"
    dump(raw_path, matrix, lambda v: int(round(v)))
    # Column-normalised: each column sums to 1, so entries read as "of all real X, this
    # fraction was predicted as Y" -- the per-class recall breakdown.
    column_sums = matrix.sum(axis=0, keepdims=True)
    normalized = np.divide(matrix, column_sums, out=np.zeros_like(matrix), where=column_sums > 0)
    dump(out_dir / "confusion_matrix_normalized.csv", normalized, lambda v: "%.4f" % v)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(1.15 * len(labels) + 2.5, 1.0 * len(labels) + 2.0))
        image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("ground truth")
        ax.set_ylabel("predicted")
        ax.set_title("Confusion matrix (column-normalised)\ncell text: normalised / raw count")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, "%.2f\n%d" % (normalized[i, j], int(round(matrix[i, j]))),
                        ha="center", va="center", fontsize=7,
                        color="white" if normalized[i, j] > 0.5 else "black")
        fig.colorbar(image, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # plotting must never sink an evaluation
        print("WARNING: could not render confusion_matrix.png (%s)" % exc)

    return {"labels": labels, "matrix": matrix.tolist(), "orientation": "matrix[predicted][ground_truth]"}


# --------------------------------------------------------------------------------------
# STEP 4 - Error analysis at one operating point
# --------------------------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes. Returns shape (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)

def load_ground_truth(label_path: Path, width: int, height: int):
    """Read a YOLO label file into (xyxy pixel boxes, class ids)."""
    boxes, classes = [], []
    if label_path.is_file():
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError("%s line %d: expected 5 fields, got %d"
                                 % (label_path, line_no, len(parts)))
            cls, cx, cy, bw, bh = int(parts[0]), *(float(v) for v in parts[1:])
            boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                          (cx + bw / 2) * width, (cy + bh / 2) * height])
            classes.append(cls)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(classes, dtype=np.int32))

def analyse_errors(model, args, images_dir: Path, labels_dir: Path, class_names):
    """Match predictions against ground truth at one explicit operating point.

    Greedy matching by descending confidence: a prediction claims the highest-IoU unmatched
    ground-truth box of the same class at IoU >= --match-iou. Leftover predictions are false
    positives; leftover ground truth boxes are false negatives. This is the standard
    single-threshold view -- it is not how mAP is computed, and it is not supposed to be.
    """
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in
                         {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    if not image_paths:
        raise SystemExit("no images found in %s" % images_dir)

    print("\nrunning predictions over %d images at conf=%.3f for error analysis..."
          % (len(image_paths), args.error_conf))

    false_positives, false_negatives = [], []
    counters = Counter()
    per_class_fp = Counter()
    per_class_fn = Counter()

    stream = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.error_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        device=args.device,
        stream=True,
        verbose=False,
    )

    for image_path, result in zip(image_paths, stream):
        height, width = result.orig_shape
        gt_boxes, gt_classes = load_ground_truth(labels_dir / (image_path.stem + ".txt"),
                                                 width, height)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_classes = np.zeros((0,), dtype=np.int32)
            pred_conf = np.zeros((0,), dtype=np.float32)
        else:
            pred_boxes = boxes.xyxy.cpu().numpy().astype(np.float32)
            pred_classes = boxes.cls.cpu().numpy().astype(np.int32)
            pred_conf = boxes.conf.cpu().numpy().astype(np.float32)

        order = np.argsort(-pred_conf)
        pred_boxes, pred_classes, pred_conf = pred_boxes[order], pred_classes[order], pred_conf[order]

        ious = iou_matrix(pred_boxes, gt_boxes)          # (n_pred, n_gt)
        gt_taken = np.zeros(len(gt_boxes), dtype=bool)
        pred_is_tp = np.zeros(len(pred_boxes), dtype=bool)
        matched_gt_for_pred = np.full(len(pred_boxes), -1, dtype=np.int32)

        for p in range(len(pred_boxes)):
            candidates = [
                g for g in range(len(gt_boxes))
                if not gt_taken[g] and gt_classes[g] == pred_classes[p] and ious[p, g] >= args.match_iou
            ]
            if candidates:
                best = max(candidates, key=lambda g: ious[p, g])
                gt_taken[best] = True
                pred_is_tp[p] = True
                matched_gt_for_pred[p] = best

        counters["true_positives"] += int(pred_is_tp.sum())

        # --- false positives ---------------------------------------------------------
        for p in np.flatnonzero(~pred_is_tp):
            best_gt = int(np.argmax(ious[p])) if len(gt_boxes) else -1
            best_iou = float(ious[p, best_gt]) if best_gt >= 0 else 0.0
            same_class = [g for g in range(len(gt_boxes)) if gt_classes[g] == pred_classes[p]]
            best_same = max((float(ious[p, g]) for g in same_class), default=0.0)

            if best_iou < BACKGROUND_IOU:
                kind = "background"          # nothing is there at all
            elif best_same >= args.match_iou:
                kind = "duplicate"           # right object, already claimed by a better box
            elif best_gt >= 0 and int(gt_classes[best_gt]) != int(pred_classes[p]) and best_iou >= args.match_iou:
                kind = "wrong_class"         # box is right, label is wrong
            else:
                kind = "poor_localisation"   # roughly the right place, not tight enough
            counters["fp_" + kind] += 1
            per_class_fp[int(pred_classes[p])] += 1

            false_positives.append({
                "image": image_path.name,
                "class_id": int(pred_classes[p]),
                "class_name": class_names.get(int(pred_classes[p]), "?"),
                "confidence": float(pred_conf[p]),
                "box_xyxy": [round(float(v), 2) for v in pred_boxes[p]],
                "kind": kind,
                "best_iou_any_gt": round(best_iou, 4),
                "best_iou_same_class": round(best_same, 4),
                "nearest_gt_class": class_names.get(int(gt_classes[best_gt]), "?") if best_gt >= 0 else None,
            })

        # --- false negatives ---------------------------------------------------------
        for g in np.flatnonzero(~gt_taken) if len(gt_boxes) else []:
            same_class = [p for p in range(len(pred_boxes)) if pred_classes[p] == gt_classes[g]]
            best_same = max((float(ious[p, g]) for p in same_class), default=0.0)
            best_any = float(np.max(ious[:, g])) if len(pred_boxes) else 0.0
            best_any_p = int(np.argmax(ious[:, g])) if len(pred_boxes) else -1
            box = gt_boxes[g]
            area = float((box[2] - box[0]) * (box[3] - box[1]))

            if best_any < BACKGROUND_IOU:
                kind = "missed_entirely"
            elif best_same >= args.match_iou:
                kind = "duplicate_suppressed"   # a same-class box matched a different GT
            elif best_any >= args.match_iou:
                kind = "wrong_class"
            else:
                kind = "poor_localisation"
            counters["fn_" + kind] += 1
            per_class_fn[int(gt_classes[g])] += 1

            false_negatives.append({
                "image": image_path.name,
                "class_id": int(gt_classes[g]),
                "class_name": class_names.get(int(gt_classes[g]), "?"),
                "box_xyxy": [round(float(v), 2) for v in box],
                "area_px": round(area, 1),
                "area_fraction": round(area / float(width * height), 5),
                "kind": kind,
                "best_iou_same_class": round(best_same, 4),
                "best_iou_any_class": round(best_any, 4),
                "nearest_pred_class": class_names.get(int(pred_classes[best_any_p]), "?") if best_any_p >= 0 else None,
                "nearest_pred_conf": round(float(pred_conf[best_any_p]), 4) if best_any_p >= 0 else None,
            })

        counters["ground_truth_boxes"] += len(gt_boxes)
        counters["predictions"] += len(pred_boxes)

    # "Worst" needs a definition, and it differs by error type:
    #   FP  -> the model was most sure about something that is not there.
    #   FN  -> the model came nowhere near it, and it was large enough to be obvious.
    false_positives.sort(key=lambda e: -e["confidence"])
    false_negatives.sort(key=lambda e: (e["best_iou_any_class"], -e["area_px"]))

    summary = {
        "operating_point": {
            "error_conf": args.error_conf,
            "match_iou": args.match_iou,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
        },
        "counts": dict(counters),
        "false_positive_total": len(false_positives),
        "false_negative_total": len(false_negatives),
        "false_positives_by_class": {class_names.get(k, str(k)): v for k, v in sorted(per_class_fp.items())},
        "false_negatives_by_class": {class_names.get(k, str(k)): v for k, v in sorted(per_class_fn.items())},
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }
    tp = counters["true_positives"]
    summary["precision_at_operating_point"] = tp / max(tp + len(false_positives), 1)
    summary["recall_at_operating_point"] = tp / max(counters["ground_truth_boxes"], 1)
    return summary, image_paths


# --------------------------------------------------------------------------------------
# STEP 5 - Annotated worst-case renders
# --------------------------------------------------------------------------------------

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:      # Pillow < 10.1 has no size argument
        return ImageFont.load_default()

def render_error(image_path: Path, labels_dir: Path, entry: dict, model, args,
                 class_names, out_path: Path, is_fp: bool):
    """Draw ground truth + predictions on one image and highlight the offending box."""
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    width, height = original.size
    scale = max(1, int(round(args.render_size / max(width, height))))
    canvas_w, canvas_h = width * scale, height * scale
    banner = 62  # three text rows: headline, diagnosis detail, legend

    image = Image.new("RGB", (canvas_w, canvas_h + banner), (24, 24, 28))
    image.paste(original.resize((canvas_w, canvas_h), Image.NEAREST), (0, banner))
    draw = ImageDraw.Draw(image)
    small, tiny = _font(13), _font(11)

    def rect(box, color, thickness, label=None, pad=0, label_below=False):
        """Draw one box. `pad` nudges the rectangle outwards so a prediction that lands
        exactly on a ground-truth box does not hide it."""
        x1, y1, x2, y2 = [v * scale for v in box]
        draw.rectangle([x1 - pad, y1 + banner - pad, x2 + pad, y2 + banner + pad],
                       outline=color, width=thickness)
        if label:
            y = y2 + banner + pad + 2 if label_below else max(banner, y1 + banner - pad - 12)
            draw.text((x1 - pad + 2, y), label, fill=color, font=tiny)

    # Predictions first, ground truth on top: when the two coincide, truth stays visible.
    result = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.error_conf,
                           iou=args.nms_iou, max_det=args.max_det, device=args.device,
                           verbose=False)[0]
    if result.boxes is not None and len(result.boxes):
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy().astype(int),
                                  result.boxes.conf.cpu().numpy()):
            rect(box, PRED_COLOR, 1, "%s %.2f" % (class_names.get(int(cls), "?"), conf),
                 pad=2, label_below=True)

    gt_boxes, gt_classes = load_ground_truth(labels_dir / (image_path.stem + ".txt"), width, height)
    for box, cls in zip(gt_boxes, gt_classes):
        rect(box, GT_COLOR, 1, class_names.get(int(cls), "?"))

    highlight = HIGHLIGHT_FP if is_fp else HIGHLIGHT_FN
    rect(entry["box_xyxy"], highlight, 3, pad=4)

    if is_fp:
        headline = "FALSE POSITIVE  %s  conf %.3f  [%s]" % (
            entry["class_name"], entry["confidence"], entry["kind"])
        detail = "best IoU with any GT %.3f (nearest GT: %s)   %s" % (
            entry["best_iou_any_gt"], entry["nearest_gt_class"], entry["image"])
    else:
        headline = "FALSE NEGATIVE  %s  [%s]" % (entry["class_name"], entry["kind"])
        detail = "best IoU with any prediction %.3f (%s)   area %.1f%% of image   %s" % (
            entry["best_iou_any_class"], entry["nearest_pred_class"] or "no predictions",
            100 * entry["area_fraction"], entry["image"])

    draw.text((6, 5), headline, fill=highlight, font=small)
    draw.text((6, 24), detail, fill=(200, 200, 205), font=tiny)
    x = 6
    for text, color in (("green = ground truth", GT_COLOR),
                        ("blue = prediction", PRED_COLOR),
                        ("thick outer box = this error", highlight)):
        draw.text((x, 44), text, fill=color, font=tiny)
        x += int(draw.textlength(text, font=tiny)) + 18

    image.save(out_path, quality=95)

def render_worst(analysis, images_dir: Path, labels_dir: Path, model, args, class_names,
                 out_dir: Path):
    rendered = {}
    for key, is_fp in (("false_positives", True), ("false_negatives", False)):
        entries = analysis[key][: args.topk]
        target = out_dir / key
        target.mkdir(parents=True, exist_ok=True)
        for rank, entry in enumerate(entries, 1):
            image_path = images_dir / entry["image"]
            if is_fp:
                stem = "fp_%02d__%s__pred-%s__conf%.3f__%s" % (
                    rank, Path(entry["image"]).stem, entry["class_name"],
                    entry["confidence"], entry["kind"])
            else:
                stem = "fn_%02d__%s__missed-%s__iou%.3f__%s" % (
                    rank, Path(entry["image"]).stem, entry["class_name"],
                    entry["best_iou_any_class"], entry["kind"])
            out_path = target / (stem.replace(" ", "_") + ".jpg")
            render_error(image_path, labels_dir, entry, model, args, class_names, out_path, is_fp)
            entry["render"] = out_path.name
        rendered[key] = len(entries)
        print("    wrote %d annotated %s to %s" % (len(entries), key.replace("_", " "), target))
    return rendered


# --------------------------------------------------------------------------------------
# STEP 6 - Markdown summary
# --------------------------------------------------------------------------------------

def write_summary(path: Path, metrics, analysis, meta, class_names):
    lines = [
        "# Evaluation summary",
        "",
        "| field | value |",
        "| --- | --- |",
        "| weights | `%s` |" % meta["weights"],
        "| data | `%s` |" % meta["data"],
        "| split | %s |" % meta["split"],
        "| images | %d |" % meta["images"],
        "| imgsz | %d |" % meta["imgsz"],
        "| ultralytics | %s |" % meta["ultralytics"],
        "| evaluated (UTC) | %s |" % meta["evaluated_utc"],
        "",
        "## Detection metrics (Ultralytics validator)",
        "",
        "| metric | value |",
        "| --- | --- |",
        "| **mAP@0.5** | **%.4f** |" % metrics["map50"],
        "| **mAP@0.5:0.95** | **%.4f** |" % metrics["map50_95"],
        "| mean precision | %.4f |" % metrics["mean_precision"],
        "| mean recall | %.4f |" % metrics["mean_recall"],
        "",
        "## Per-class AP",
        "",
        "| class | AP@0.5 | AP@0.5:0.95 | precision | recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in metrics["per_class"]:
        if entry["ap50"] is None:
            lines.append("| %s | - | - | - | - |" % entry["class_name"])
        else:
            lines.append("| %s | %.4f | %.4f | %.4f | %.4f |" % (
                entry["class_name"], entry["ap50"], entry["ap50_95"],
                entry["precision"], entry["recall"]))

    op = analysis["operating_point"]
    lines += [
        "",
        "## Error analysis",
        "",
        "Single operating point: conf >= %.3f, a prediction matches ground truth at "
        "IoU >= %.2f with the same class." % (op["error_conf"], op["match_iou"]),
        "",
        "| quantity | value |",
        "| --- | --- |",
        "| ground-truth boxes | %d |" % analysis["counts"].get("ground_truth_boxes", 0),
        "| predictions | %d |" % analysis["counts"].get("predictions", 0),
        "| true positives | %d |" % analysis["counts"].get("true_positives", 0),
        "| false positives | %d |" % analysis["false_positive_total"],
        "| false negatives | %d |" % analysis["false_negative_total"],
        "| precision @ this point | %.4f |" % analysis["precision_at_operating_point"],
        "| recall @ this point | %.4f |" % analysis["recall_at_operating_point"],
        "",
        "### False positives by cause",
        "",
        "| cause | count |",
        "| --- | --- |",
    ]
    for key, value in sorted(analysis["counts"].items()):
        if key.startswith("fp_"):
            lines.append("| %s | %d |" % (key[3:], value))
    lines += ["", "### False negatives by cause", "", "| cause | count |", "| --- | --- |"]
    for key, value in sorted(analysis["counts"].items()):
        if key.startswith("fn_"):
            lines.append("| %s | %d |" % (key[3:], value))
    lines += [
        "",
        "### Per class",
        "",
        "| class | false positives | false negatives |",
        "| --- | --- | --- |",
    ]
    for class_id in sorted(class_names):
        name = class_names[class_id]
        lines.append("| %s | %d | %d |" % (
            name,
            analysis["false_positives_by_class"].get(name, 0),
            analysis["false_negatives_by_class"].get(name, 0)))
    lines += [
        "",
        "Annotated worst cases are in `false_positives/` and `false_negatives/`; the full",
        "lists (not just the top %d) are in `error_analysis.json`." % meta["topk"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# STEP 7 - Entry point
# --------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    if not args.weights.is_file():
        sys.exit("weights not found: %s" % args.weights)
    if not args.data.is_file():
        sys.exit("data.yaml not found: %s" % args.data)

    spec = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(spec.get("path", args.data.parent))
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    split_images = root / spec[args.split]
    # Ultralytics derives label paths by swapping the trailing "images" directory for
    # "labels"; do the same, but structurally, so a root path containing the word "images"
    # cannot corrupt it.
    labels_dir = split_images.parent / "labels"
    if not split_images.is_dir():
        sys.exit("split directory does not exist: %s" % split_images)
    if not labels_dir.is_dir():
        sys.exit("labels directory does not exist: %s" % labels_dir)

    names = spec["names"]
    class_names = {int(k): v for k, v in names.items()} if isinstance(names, dict) \
        else {i: v for i, v in enumerate(names)}

    out_dir = args.out or (args.weights.parent.parent / ("eval_" + args.split))
    out_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    import ultralytics

    print("=" * 78)
    print("EVALUATING %s" % args.weights)
    print("=" * 78)
    print("data   : %s" % args.data.resolve())
    print("split  : %s  (%s)" % (args.split, split_images))
    print("out    : %s" % out_dir.resolve())
    print("ultralytics %s" % ultralytics.__version__)

    model = YOLO(str(args.weights))

    print("\n--- Ultralytics validator (mAP) ---")
    results = run_validator(model, args, args.data, out_dir)
    metrics = extract_metrics(results, class_names)

    print("\nmAP@0.5      : %.4f" % metrics["map50"])
    print("mAP@0.5:0.95 : %.4f" % metrics["map50_95"])
    print("\n%-20s%10s%14s%12s%10s" % ("class", "AP@0.5", "AP@0.5:0.95", "precision", "recall"))
    for entry in metrics["per_class"]:
        if entry["ap50"] is None:
            print("%-20s%10s%14s%12s%10s  (%s)" % (entry["class_name"], "-", "-", "-", "-",
                                                   entry.get("note", "")))
        else:
            print("%-20s%10.4f%14.4f%12.4f%10.4f" % (
                entry["class_name"], entry["ap50"], entry["ap50_95"],
                entry["precision"], entry["recall"]))

    with (out_dir / "per_class_ap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class_name", "ap50",
                                                    "ap50_95", "precision", "recall", "note"])
        writer.writeheader()
        for entry in metrics["per_class"]:
            writer.writerow({k: entry.get(k) for k in writer.fieldnames})

    print("\n--- confusion matrix ---")
    confusion = write_confusion_matrix(results, class_names, out_dir)
    if confusion:
        print("    wrote confusion_matrix.csv, confusion_matrix_normalized.csv, confusion_matrix.png")

    print("\n--- error analysis ---")
    analysis, image_paths = analyse_errors(model, args, split_images, labels_dir, class_names)
    print("    ground truth %d | predictions %d | TP %d | FP %d | FN %d" % (
        analysis["counts"].get("ground_truth_boxes", 0),
        analysis["counts"].get("predictions", 0),
        analysis["counts"].get("true_positives", 0),
        analysis["false_positive_total"], analysis["false_negative_total"]))
    print("    precision @ conf %.2f: %.4f | recall: %.4f" % (
        args.error_conf, analysis["precision_at_operating_point"],
        analysis["recall_at_operating_point"]))
    for prefix, title in (("fp_", "false positive causes"), ("fn_", "false negative causes")):
        breakdown = {k[3:]: v for k, v in sorted(analysis["counts"].items()) if k.startswith(prefix)}
        print("    %-22s %s" % (title + ":", breakdown or "none"))

    if not args.skip_renders:
        render_worst(analysis, split_images, labels_dir, model, args, class_names, out_dir)

    meta = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "images": len(image_paths),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "topk": args.topk,
        # Thresholds are part of the result: the same weights on the same split score
        # differently under different NMS/confidence settings, so an artefact that omits
        # them cannot be compared against another run or reproduced later.
        "thresholds": {
            "val_conf": args.val_conf,
            "nms_iou": args.nms_iou,
            "max_det": args.max_det,
            "error_conf": args.error_conf,
            "match_iou": args.match_iou,
        },
        "ultralytics": ultralytics.__version__,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "metrics": metrics, "confusion_matrix": confusion},
                   indent=2, default=str), encoding="utf-8")
    (out_dir / "error_analysis.json").write_text(
        json.dumps({"meta": meta, "analysis": analysis}, indent=2, default=str), encoding="utf-8")
    write_summary(out_dir / "summary.md", metrics, analysis, meta, class_names)

    print("\n" + "=" * 78)
    print("WROTE %s" % out_dir.resolve())
    print("=" * 78)
    for name in ("metrics.json", "per_class_ap.csv", "confusion_matrix.csv",
                 "confusion_matrix_normalized.csv", "confusion_matrix.png",
                 "error_analysis.json", "summary.md"):
        marker = "  " if (out_dir / name).exists() else "  (missing) "
        print("%s%s" % (marker, name))
    if not args.skip_renders:
        print("  false_positives/  %d annotated images" % min(args.topk, analysis["false_positive_total"]))
        print("  false_negatives/  %d annotated images" % min(args.topk, analysis["false_negative_total"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
