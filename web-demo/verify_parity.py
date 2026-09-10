#!/usr/bin/env python
"""Generate the browser-parity reference for web-demo/, and prove the preprocessing first.

Getting letterbox preprocessing wrong does not raise -- it silently shifts every box by a
few pixels, which looks plausible on screen and is invisible without a reference. So this
script does two things, in order:

  1. Reimplements letterbox + normalisation the way `app.js` does it, runs the ONNX graph
     through raw onnxruntime, and asserts the result matches what Ultralytics' own
     pipeline produces for the same image. If that assert fails, the JS port would have
     been wrong too, and we find out here rather than by eyeballing a canvas.

  2. Dumps `reference.json` -- per-image detections in original-image pixel coordinates --
     which the browser compares itself against at runtime (see the parity panel in the
     demo, and `--check` below).

Usage
-----
    python web-demo/verify_parity.py              # regenerate reference.json
    python web-demo/verify_parity.py --check browser_output.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MODEL = HERE / "model" / "yolo26n_int8.onnx"
IMGSZ = 640
CONF = 0.25

# Must match app.js exactly. Ultralytics' LetterBox defaults for a fixed-size ONNX export:
# auto=False (no stride-rounded minimum rectangle), scale_fill=False, scaleup=True,
# center=True, padding_value=114, interpolation=cv2.INTER_LINEAR.
PAD_VALUE = 114


def letterbox_params(width: int, height: int, imgsz: int = IMGSZ):
    """Reproduce ultralytics.data.augment.LetterBox.get_params for a fixed-size export.

        r          = min(imgsz/h, imgsz/w)          # scaleup=True, so never clamped to 1
        new_unpad  = (round(w*r), round(h*r))
        dw, dh     = (imgsz - new_unpad_w)/2, (imgsz - new_unpad_h)/2      # center=True
        top/left   = round(d - 0.1)
        bottom/right = round(d + 0.1)

    The -0.1/+0.1 is not decoration: it forces an odd pixel of padding onto the
    bottom/right rather than leaving the split ambiguous at a .5 boundary.
    """
    r = min(imgsz / height, imgsz / width)
    new_w, new_h = round(width * r), round(height * r)
    dw, dh = (imgsz - new_w) / 2, (imgsz - new_h) / 2
    return {
        "r": r,
        "new_w": new_w,
        "new_h": new_h,
        "left": round(dw - 0.1),
        "top": round(dh - 0.1),
        "right": round(dw + 0.1),
        "bottom": round(dh + 0.1),
    }


def preprocess(bgr: np.ndarray):
    """BGR uint8 HWC -> (1,3,640,640) float32 RGB in [0,1], plus the letterbox params."""
    h, w = bgr.shape[:2]
    p = letterbox_params(w, h)
    resized = cv2.resize(bgr, (p["new_w"], p["new_h"]), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, p["top"], p["bottom"], p["left"], p["right"],
                                cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    rgb = padded[..., ::-1]                       # BGR -> RGB
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)[None], p


def undo_letterbox(boxes: np.ndarray, p: dict, width: int, height: int) -> np.ndarray:
    """Map xyxy boxes from 640x640 letterboxed space back to original-image pixels."""
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - p["left"]) / p["r"]
    out[:, [1, 3]] = (out[:, [1, 3]] - p["top"]) / p["r"]
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def run_onnx(session, image_path: Path):
    """Raw onnxruntime inference, mirroring what app.js does step for step."""
    bgr = cv2.imread(str(image_path))
    h, w = bgr.shape[:2]
    tensor, p = preprocess(bgr)
    raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]  # (1, 300, 6)

    # YOLO26's end-to-end head emits [x1, y1, x2, y2, confidence, class_id], already
    # NMS-free and sorted by confidence. No NMS, no transpose, no sigmoid.
    det = raw[0]
    keep = det[:, 4] >= CONF
    det = det[keep]
    boxes = undo_letterbox(det[:, :4].astype(np.float64), p, w, h)
    return [
        {"class_id": int(c), "confidence": round(float(s), 6),
         "box": [round(float(v), 3) for v in b]}
        for b, s, c in zip(boxes, det[:, 4], det[:, 5])
    ], p, (w, h)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", type=Path, default=None,
                        help="a browser-exported JSON to compare against reference.json")
    # 6 px is not a guess. Feeding the SAME INT8 model two 640x640 tensors that differ by
    # at most +/-1 in 12% of pixels (cv2's fixed-point resize vs float bilinear) moves
    # matched boxes by up to 5.49 px and changes detection counts -- measured, Python vs
    # Python, no browser involved. The identical experiment on the FP32 export moves boxes
    # 0.86 px and changes no counts. So this tolerance is the INT8 model's own sensitivity
    # floor; tightening it below that would fail runs for reasons the browser cannot fix.
    parser.add_argument("--tolerance", type=float, default=6.0,
                        help="max acceptable per-coordinate difference in pixels")
    args = parser.parse_args(argv)

    reference_path = HERE / "reference.json"

    if args.check:
        return compare(reference_path, args.check, args.tolerance)

    import onnxruntime as ort
    from ultralytics import YOLO

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    ultra = YOLO(str(MODEL), task="detect")
    names = ultra.names

    images = sorted((HERE / "examples").iterdir())
    print(f"model : {MODEL.name}  ({MODEL.stat().st_size:,} bytes)")
    print(f"output: {session.get_outputs()[0].shape}  (end-to-end, NMS-free)")
    print(f"images: {len(images)}\n")

    entries = {}
    worst_overall = 0.0
    for path in images:
        mine, p, (w, h) = run_onnx(session, path)

        # --- step 1: does our preprocessing reproduce Ultralytics exactly? --------------
        ref = ultra.predict(str(path), imgsz=IMGSZ, conf=CONF, device="cpu", verbose=False)[0]
        ref_boxes = ref.boxes.xyxy.cpu().numpy().astype(np.float64)
        ref_conf = ref.boxes.conf.cpu().numpy()
        ref_cls = ref.boxes.cls.cpu().numpy().astype(int)

        assert len(mine) == len(ref_boxes), (
            f"{path.name}: our pipeline found {len(mine)} detections, Ultralytics found "
            f"{len(ref_boxes)} -- preprocessing does not match")
        worst = 0.0
        for got, rb, rc, rk in zip(mine, ref_boxes, ref_conf, ref_cls):
            assert got["class_id"] == int(rk), f"{path.name}: class mismatch"
            worst = max(worst, float(np.max(np.abs(np.array(got["box"]) - rb))))
            assert abs(got["confidence"] - float(rc)) < 1e-4, f"{path.name}: confidence mismatch"
        worst_overall = max(worst_overall, worst)

        entries[path.name] = {
            "width": w, "height": h,
            "letterbox": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()},
            "detections": mine,
        }
        classes = sorted({names[d["class_id"]] for d in mine})
        print(f"  {path.name:<26} {len(mine):>2} det  max coord delta vs ultralytics "
              f"{worst:.6f} px  {classes}")

    print(f"\nworst coordinate difference across all images: {worst_overall:.6f} px")
    print("-> our letterbox + normalisation reproduces Ultralytics; safe to port to JS.\n")

    reference_path.write_text(json.dumps({
        "model": MODEL.name,
        "imgsz": IMGSZ,
        "conf": CONF,
        "class_names": [names[i] for i in range(len(names))],
        "note": ("Boxes are xyxy in ORIGINAL image pixels, produced by onnxruntime on CPU "
                 "via web-demo/verify_parity.py. The browser compares against this."),
        "images": entries,
    }, indent=2), encoding="utf-8")
    print(f"wrote {reference_path}")
    return 0


def _iou(a, b):
    """IoU between two xyxy boxes, used to pair Python and browser detections."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare(reference_path: Path, browser_path: Path, tolerance: float) -> int:
    """Compare a browser export against the Python reference, coordinate by coordinate."""
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    got = json.loads(browser_path.read_text(encoding="utf-8"))
    names = ref["class_names"]

    print(f"{'image':<26}{'py':>4}{'js':>4}  {'max Δxy (px)':>13}  {'max Δconf':>10}  result")
    print("-" * 78)
    worst_xy = worst_conf = 0.0
    failures = []
    for name, entry in ref["images"].items():
        mine = got.get("images", {}).get(name)
        if mine is None:
            failures.append(f"{name}: absent from browser output")
            print(f"{name:<26}{len(entry['detections']):>4}{'-':>4}  {'-':>13}  {'-':>10}  MISSING")
            continue
        a, b = entry["detections"], mine["detections"]

        # Match by geometry, not by index. INT8 quantises the score output to a small
        # discrete set, so several detections share EXACTLY the same confidence and their
        # sort order is arbitrary. Comparing detection[i] to detection[i] under those ties
        # compares unrelated objects and reports nonsense (we measured 460 px that way).
        pairs, unmatched_py, used = [], [], set()
        for da in a:
            best, best_iou = None, 0.0
            for j, db in enumerate(b):
                if j in used or db["class_id"] != da["class_id"]:
                    continue
                iou = _iou(da["box"], db["box"])
                if iou > best_iou:
                    best, best_iou = j, iou
            if best is not None and best_iou >= 0.5:
                used.add(best)
                pairs.append((da, b[best]))
            else:
                unmatched_py.append(da)
        unmatched_js = [db for j, db in enumerate(b) if j not in used]

        dxy = dconf = 0.0
        for da, db in pairs:
            dxy = max(dxy, max(abs(x - y) for x, y in zip(da["box"], db["box"])))
            dconf = max(dconf, abs(da["confidence"] - db["confidence"]))
        worst_xy, worst_conf = max(worst_xy, dxy), max(worst_conf, dconf)

        ok = dxy <= tolerance and not unmatched_py and not unmatched_js
        note = ""
        if unmatched_py or unmatched_js:
            note = f"  (+{len(unmatched_js)} js-only, +{len(unmatched_py)} py-only)"
            failures.append(f"{name}: {len(unmatched_py)} detection(s) only in Python, "
                            f"{len(unmatched_js)} only in the browser")
        if dxy > tolerance:
            failures.append(f"{name}: max coordinate difference {dxy:.3f} px > {tolerance}")
        print(f"{name:<26}{len(a):>4}{len(b):>4}  {dxy:>13.4f}  {dconf:>10.6f}  "
              f"{'OK' if ok else 'DIFF'}{note}")

    print("-" * 78)
    print(f"worst coordinate difference : {worst_xy:.4f} px  (tolerance {tolerance})")
    print(f"worst confidence difference : {worst_conf:.6f}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nPASS: the browser reproduces the Python service within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
