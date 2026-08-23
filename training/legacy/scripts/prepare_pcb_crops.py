"""
Extract per-defect crops from the DsPCBSD COCO dataset.

Output structure (ImageFolder-compatible):
  crops/
    train/
      SH/  SP/  SC/  OP/  MB/  HB/  CS/  CFO/  BMFO/
    val/
      SH/  SP/  SC/  OP/  MB/  HB/  CS/  CFO/  BMFO/

Usage:
  python scripts/prepare_pcb_crops.py
  python scripts/prepare_pcb_crops.py --raw-path D:/Datasets/DsPCBSD --crops-path D:/Datasets/DsPCBSD/crops
"""

import argparse
import json
import os
from pathlib import Path

import cv2


CATEGORY_MAP = {1: "SH", 2: "SP", 3: "SC", 4: "OP", 5: "MB",
                6: "HB", 7: "CS", 8: "CFO", 9: "BMFO"}

SPLITS = {
    "train": ("train2017", "instances_train2017.json"),
    "val":   ("val2017",   "instances_val2017.json"),
}

MIN_CROP_SIZE = 10   # pixels — skip degenerate boxes smaller than this


def extract_crops(raw_path: str, crops_path: str, padding: int = 4):
    coco_root = Path(raw_path) / "Data_COCO"
    out_root  = Path(crops_path)

    for split, (img_dir, ann_file) in SPLITS.items():
        ann_path = coco_root / "annotations" / ann_file
        img_root = coco_root / img_dir

        with open(ann_path) as f:
            coco = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        counters = {name: 0 for name in CATEGORY_MAP.values()}

        for split_class in CATEGORY_MAP.values():
            (out_root / split / split_class).mkdir(parents=True, exist_ok=True)

        print(f"\n[{split}] {len(coco['annotations'])} annotations across {len(coco['images'])} images")

        for ann in coco["annotations"]:
            cat_name = CATEGORY_MAP.get(ann["category_id"])
            if cat_name is None:
                continue

            img_path = img_root / id_to_file[ann["image_id"]]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  Warning: could not read {img_path}")
                continue

            x, y, w, h = [int(v) for v in ann["bbox"]]
            ih, iw = img.shape[:2]

            # Apply padding and clamp to image bounds
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(iw, x + w + padding)
            y2 = min(ih, y + h + padding)

            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                continue

            crop = img[y1:y2, x1:x2]
            out_path = out_root / split / cat_name / f"{ann['id']:07d}.jpg"
            cv2.imwrite(str(out_path), crop)
            counters[cat_name] += 1

        print(f"  Crops saved per class: {counters}")

    print(f"\nDone. Crops written to: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path",   default="D:/Datasets/DsPCBSD")
    parser.add_argument("--crops-path", default="D:/Datasets/DsPCBSD/crops")
    parser.add_argument("--padding",    type=int, default=4,
                        help="Pixels of context added around each bounding box")
    args = parser.parse_args()
    extract_crops(args.raw_path, args.crops_path, args.padding)
