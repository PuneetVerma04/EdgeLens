#!/usr/bin/env python
"""Convert the NEU-DET surface-defect dataset (PASCAL VOC) into a YOLO detection dataset.

The script is deliberately noisy and paranoid: it reports before it writes, checks every
converted coordinate, and asserts the splits are disjoint. A silent conversion bug produces
a model that trains happily and detects nothing, so every stage fails loudly instead.

Pipeline
--------
    1. Command line                       parse_args
    2. Discover the source tree           discover        (read-only; reports what it found)
    3. Convert VOC -> YOLO                convert         (validates every coordinate)
    4. Split 70/15/15                     stratified_split / split_from_manifest
    5. Write dataset + data.yaml          write_dataset, write_data_yaml, write_manifest
    6. Report statistics                  report_statistics
    7. Orchestration                      main

Usage
-----
    python training/prepare_neu_det.py --src "D:/Datasets/NEU-DET" --dst data/neu_det
    python training/prepare_neu_det.py --src "D:/Datasets/NEU-DET" --dry-run

Nothing is written in --dry-run mode, so it is safe to run first.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("Pillow is required: pip install -r training/requirements.txt")

# --------------------------------------------------------------------------------------
# Class order.
#
# THIS ORDER IS FROZEN. It is sorted() over the six NEU-DET defect directory names, i.e.
# plain ASCII ascending order. Every artefact downstream of this file -- the .txt label
# files, data.yaml, trained weights, the confusion matrix, the FastAPI response schema --
# uses these integer ids. Changing the order silently invalidates every trained checkpoint,
# so it is hard-coded here rather than derived at runtime from a directory listing.
# --------------------------------------------------------------------------------------
CLASS_NAMES = (
    "crazing",          # 0
    "inclusion",        # 1
    "patches",          # 2
    "pitted_surface",   # 3
    "rolled-in_scale",  # 4
    "scratches",        # 5
)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")

# Tolerance for the [0, 1] range checks. Dividing exact integers by an exact integer image
# dimension can land a hair outside the interval; anything larger than this is a real bug
# in the annotation, not arithmetic noise.
EPS = 1e-9


# --------------------------------------------------------------------------------------
# Shared types and helpers
# --------------------------------------------------------------------------------------

class DatasetError(RuntimeError):
    """Raised when the source dataset violates an invariant this script depends on."""

def check(condition, message):
    """Assertion that fails loudly regardless of `python -O`.

    Plain `assert` statements are stripped by the -O flag, and these checks are the entire
    point of the script, so they are explicit raises instead.
    """
    if not condition:
        raise DatasetError(message)

@dataclass
class Box:
    class_id: int
    class_name: str
    cx: float
    cy: float
    w: float
    h: float
    difficult: bool

    def to_line(self) -> str:
        return "%d %.6f %.6f %.6f %.6f" % (self.class_id, self.cx, self.cy, self.w, self.h)

@dataclass
class Sample:
    stem: str
    image_path: Path
    xml_path: Path
    source_split: str    # the split folder the vendor filed it under (discarded)
    primary_class: str   # the defect directory the image lives in -- the stratification key
    width: int = 0
    height: int = 0
    boxes: list = field(default_factory=list)

    @property
    def cooccurrence(self):
        """Sorted set of every class present in this image (usually just the primary one)."""
        return tuple(sorted({b.class_name for b in self.boxes}))


# --------------------------------------------------------------------------------------
# STEP 1 - Command line
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert NEU-DET (PASCAL VOC) into a stratified YOLO detection dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", type=Path, default=Path("D:/Datasets/NEU-DET"),
                        help="NEU-DET root containing train/ and validation/")
    parser.add_argument("--dst", type=Path, default=repo_root / "data" / "neu_det",
                        help="output directory for the YOLO dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="split seed (fixed for reproducibility)")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="split ratios; normalised if they do not sum to 1")
    parser.add_argument("--drop-difficult", action="store_true",
                        help="exclude VOC difficult=1 boxes (kept by default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report and validate only; write nothing")
    parser.add_argument("--force", action="store_true",
                        help="delete and rebuild --dst if it already exists")
    parser.add_argument("--from-manifest", type=Path, default=None, metavar="SPLIT_MANIFEST",
                        help="rebuild the exact split recorded in a split_manifest.json "
                             "instead of re-deriving one from --seed; use this to reproduce "
                             "a dataset that existing checkpoints were trained on")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# STEP 2 - Discover the source tree (read-only)
# --------------------------------------------------------------------------------------

def discover(src: Path):
    """Walk the source tree, report exactly what was found, and validate pairing.

    Nothing is written here. NEU-DET ships as
    {train,validation}/images/<class>/<stem>.jpg  +  {train,validation}/annotations/<stem>.xml
    but this function does not assume that -- it globs and reports whatever is actually there.
    """
    check(src.is_dir(), "source directory does not exist: %s" % src)

    print("=" * 78)
    print("DISCOVERED LAYOUT (read-only -- nothing has been written yet)")
    print("=" * 78)
    print("source root: %s" % src)

    top_level = sorted(p.name for p in src.iterdir() if p.is_dir())
    print("top-level directories: %s" % top_level)

    images = defaultdict(list)
    annotations = defaultdict(list)
    image_owner = {}  # stem -> (vendor_split, primary_class)

    for image_path in sorted(src.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        vendor_split = image_path.relative_to(src).parts[0]
        # The directory immediately containing the image names the defect class in NEU-DET.
        primary_class = image_path.parent.name
        images[image_path.stem].append(image_path)
        image_owner[image_path.stem] = (vendor_split, primary_class)

    for xml_path in sorted(src.rglob("*.xml")):
        annotations[xml_path.stem].append(xml_path)

    # --- per-directory census -----------------------------------------------------------
    image_dirs = Counter(p.parent.relative_to(src).as_posix() for v in images.values() for p in v)
    xml_dirs = Counter(p.parent.relative_to(src).as_posix() for v in annotations.values() for p in v)
    extensions = Counter(p.suffix.lower() for v in images.values() for p in v)

    print("\nimage files: %d  (extensions: %s)" % (sum(image_dirs.values()), dict(extensions)))
    for d, n in sorted(image_dirs.items()):
        print("    %-48s %5d" % (d, n))
    print("\nannotation files: %d" % sum(xml_dirs.values()))
    for d, n in sorted(xml_dirs.items()):
        print("    %-48s %5d" % (d, n))

    # --- pairing ------------------------------------------------------------------------
    dup_images = {k: [str(p) for p in v] for k, v in images.items() if len(v) > 1}
    dup_xml = {k: [str(p) for p in v] for k, v in annotations.items() if len(v) > 1}
    check(not dup_images, "duplicate image stems (ambiguous label mapping): %s" % dup_images)
    check(not dup_xml, "duplicate annotation stems: %s" % dup_xml)

    only_images = sorted(set(images) - set(annotations))
    only_xml = sorted(set(annotations) - set(images))
    print("\npaired stems: %d" % len(set(images) & set(annotations)))
    print("images with no annotation anywhere in the tree: %d %s" % (len(only_images), only_images[:10]))
    print("annotations with no image anywhere in the tree: %d %s" % (len(only_xml), only_xml[:10]))
    check(not only_images, "images without annotations: %s" % only_images)
    check(not only_xml, "annotations without images: %s" % only_xml)

    # --- misfiled annotations (a known NEU-DET quirk) -----------------------------------
    # In the stock distribution at least one XML sits in the other split's annotations dir
    # (e.g. crazing_240.xml under validation/ while crazing_240.jpg is under train/). This is
    # harmless here because the vendor split is discarded and re-derived, but it is worth
    # surfacing rather than silently absorbing.
    misfiled = [
        stem
        for stem in sorted(set(images) & set(annotations))
        if annotations[stem][0].relative_to(src).parts[0] != image_owner[stem][0]
    ]
    if misfiled:
        print("\nNOTE: %d annotation(s) are filed under a different vendor split than their "
              "image: %s" % (len(misfiled), misfiled[:10]))
        print("      Harmless -- the vendor train/validation split is discarded and re-derived below.")

    unknown = sorted({image_owner[s][1] for s in images} - set(CLASS_NAMES))
    check(not unknown,
          "image directories name classes that are not in the frozen CLASS_NAMES list: %s" % unknown)

    samples = [
        Sample(
            stem=stem,
            image_path=images[stem][0],
            xml_path=annotations[stem][0],
            source_split=image_owner[stem][0],
            primary_class=image_owner[stem][1],
        )
        for stem in sorted(set(images) & set(annotations))
    ]

    vendor = Counter((s.source_split, s.primary_class) for s in samples)
    print("\nvendor split x class (discarded, shown for reference):")
    for (split_name, cls), n in sorted(vendor.items()):
        print("    %-12s %-18s %5d" % (split_name, cls, n))
    print("\ntotal usable samples: %d" % len(samples))
    return samples


# --------------------------------------------------------------------------------------
# STEP 3 - Convert VOC boxes to YOLO, validating every value
# --------------------------------------------------------------------------------------

def convert(samples, drop_difficult):
    """Parse every XML into normalised YOLO boxes, validating each one.

    Coordinates are normalised against the *actual* pixel dimensions read from the image
    file, not the <size> block in the XML, because the XML is the thing more likely to be
    wrong. Any disagreement between the two is reported.
    """
    print("\n" + "=" * 78)
    print("VOC -> YOLO CONVERSION")
    print("=" * 78)

    size_mismatches = []
    difficult_kept = 0
    difficult_dropped = 0

    for sample in samples:
        with Image.open(sample.image_path) as im:
            sample.width, sample.height = im.size
        check(sample.width > 0 and sample.height > 0,
              "%s: image reports non-positive size %dx%d"
              % (sample.image_path.name, sample.width, sample.height))

        root = ET.parse(sample.xml_path).getroot()
        size_node = root.find("size")
        if size_node is not None:
            declared = (int(float(size_node.findtext("width", "0"))),
                        int(float(size_node.findtext("height", "0"))))
            if declared != (sample.width, sample.height):
                size_mismatches.append(
                    "%s: XML says %dx%d, image is %dx%d (using the image)"
                    % (sample.xml_path.name, declared[0], declared[1], sample.width, sample.height)
                )

        for index, obj in enumerate(root.findall("object")):
            name = (obj.findtext("name") or "").strip()
            check(name in CLASS_TO_ID,
                  "%s object #%d: unknown class %r (expected one of %s)"
                  % (sample.xml_path.name, index, name, list(CLASS_NAMES)))

            difficult = (obj.findtext("difficult") or "0").strip() == "1"
            if difficult and drop_difficult:
                difficult_dropped += 1
                continue
            if difficult:
                difficult_kept += 1

            bnd = obj.find("bndbox")
            check(bnd is not None, "%s object #%d: no <bndbox>" % (sample.xml_path.name, index))
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            raw = "(xmin=%s, ymin=%s, xmax=%s, ymax=%s)" % (xmin, ymin, xmax, ymax)
            where = "%s object #%d" % (sample.xml_path.name, index)

            # --- pixel-space checks, before arithmetic can hide the problem --------------
            check(xmax > xmin, "%s: non-positive width, xmax<=xmin %s" % (where, raw))
            check(ymax > ymin, "%s: non-positive height, ymax<=ymin %s" % (where, raw))
            check(0 <= xmin and 0 <= ymin and xmax <= sample.width and ymax <= sample.height,
                  "%s: box %s falls outside the %dx%d image"
                  % (where, raw, sample.width, sample.height))

            cx = (xmin + xmax) / 2.0 / sample.width
            cy = (ymin + ymax) / 2.0 / sample.height
            bw = (xmax - xmin) / float(sample.width)
            bh = (ymax - ymin) / float(sample.height)

            # --- normalised-space checks -------------------------------------------------
            for label, value in (("cx", cx), ("cy", cy), ("w", bw), ("h", bh)):
                check(-EPS <= value <= 1.0 + EPS,
                      "%s: normalised %s=%r is outside [0, 1] -- source box %s, image %dx%d"
                      % (where, label, value, raw, sample.width, sample.height))
            check(bw > EPS,
                  "%s: normalised width %r is not positive -- source box %s" % (where, bw, raw))
            check(bh > EPS,
                  "%s: normalised height %r is not positive -- source box %s" % (where, bh, raw))
            # The centre/size form must still describe an in-frame rectangle.
            check(-EPS <= cx - bw / 2 and cx + bw / 2 <= 1.0 + EPS,
                  "%s: x extent [%r, %r] escapes [0, 1] -- source box %s"
                  % (where, cx - bw / 2, cx + bw / 2, raw))
            check(-EPS <= cy - bh / 2 and cy + bh / 2 <= 1.0 + EPS,
                  "%s: y extent [%r, %r] escapes [0, 1] -- source box %s"
                  % (where, cy - bh / 2, cy + bh / 2, raw))

            sample.boxes.append(Box(CLASS_TO_ID[name], name, cx, cy, bw, bh, difficult))

    total_boxes = sum(len(s.boxes) for s in samples)
    print("converted %d boxes across %d images -- all range checks passed"
          % (total_boxes, len(samples)))
    if drop_difficult:
        print("difficult-flagged boxes: dropped %d" % difficult_dropped)
    else:
        print("difficult-flagged boxes: kept %d" % difficult_kept)
        if difficult_kept:
            print("    (VOC 'difficult' has no YOLO equivalent. They are kept by default, which "
                  "matches how\n     NEU-DET is normally benchmarked; pass --drop-difficult to "
                  "exclude them.)")

    if size_mismatches:
        print("\nWARNING: %d XML <size> block(s) disagree with the image file:" % len(size_mismatches))
        for line in size_mismatches[:20]:
            print("    %s" % line)
    else:
        print("XML <size> blocks agree with the image files for every sample")

    empty = [s.stem for s in samples if not s.boxes]
    if empty:
        print("\nWARNING: %d image(s) have no boxes left and will be written as background "
              "images with empty label files: %s" % (len(empty), empty[:10]))

    multi = [s for s in samples if len(s.cooccurrence) > 1]
    print("\nimages containing more than one defect class: %d (%.1f%%)"
          % (len(multi), 100.0 * len(multi) / max(len(samples), 1)))
    if multi:
        for combo, n in Counter(s.cooccurrence for s in multi).most_common(10):
            print("    %-48s %5d" % (" + ".join(combo), n))
        print("    Stratification below keys on the image's *primary* class (its source "
              "directory);\n    secondary classes fall where the shuffle puts them, which "
              "measures as balanced as\n    any cleverer scheme -- see stratified_split().")

    return {
        "total_boxes": total_boxes,
        "difficult_kept": difficult_kept,
        "difficult_dropped": difficult_dropped,
        "size_mismatches": len(size_mismatches),
        "multi_class_images": len(multi),
        "empty_images": len(empty),
    }


# --------------------------------------------------------------------------------------
# STEP 4 - Split 70/15/15, stratified and seeded
# --------------------------------------------------------------------------------------

def _check_splits(splits, samples):
    """The assertion that actually matters: no image may appear in two splits.

    Everything else about a split is a preference; leakage silently inflates every metric
    downstream and raises no error on its own.
    """
    totals = [len(splits[s]) for s in SPLITS]
    stems = {name: {s.stem for s in splits[name]} for name in SPLITS}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = stems[a] & stems[b]
            check(not overlap, "LEAKAGE: %d image(s) in both %s and %s: %s"
                  % (len(overlap), a, b, sorted(overlap)[:10]))
    for name in SPLITS:
        check(len(stems[name]) == len(splits[name]), "%s contains duplicate stems" % name)
    check(sum(totals) == len(samples),
          "split total %d != %d input samples" % (sum(totals), len(samples)))
    check(set().union(*stems.values()) == {s.stem for s in samples},
          "some samples were dropped by the split")
    print("\nleakage checks passed: splits are pairwise disjoint and cover every image exactly once")


def stratified_split(samples, ratios, seed):
    """70/15/15 split, stratified by primary defect class, with a fixed seed.

    Shuffle the images of each class and slice. With 300 images per class the slice points
    are exact (210/45/45), so per-class image counts need no apportionment arithmetic.

    An earlier version of this function also grouped images by their set of co-occurring
    classes and dealt the groups out proportionally, on the theory that the 123 multi-class
    images might otherwise clump into one split. Measured over 20 seeds, that scheme was no
    better than this one at balancing per-class *box* counts (mean worst deviation from the
    70% target: 2.17 pp for the clever version, 1.94 pp for this one, with the clever version
    ahead in 10 of 20 seeds). It was ~80 lines of hard-to-verify code buying nothing, so it
    is gone. Use --from-manifest to reproduce a split exactly rather than relying on the
    ordering of RNG calls staying stable.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for sample in samples:
        by_class[sample.primary_class].append(sample)

    splits = {name: [] for name in SPLITS}
    print("\n" + "=" * 78)
    print("STRATIFIED SPLIT  ratios=%s  seed=%d" % (tuple(ratios), seed))
    print("=" * 78)
    print("%-18s%8s%8s%8s%8s" % ("class", "total", "train", "val", "test"))

    for class_name in CLASS_NAMES:
        # Sort before shuffling so the result depends on the seed, not on filesystem order.
        members = sorted(by_class.get(class_name, []), key=lambda s: s.stem)
        rng.shuffle(members)
        n = len(members)
        cut_train = int(n * ratios[0])
        cut_val = int(n * (ratios[0] + ratios[1]))
        chunks = (members[:cut_train], members[cut_train:cut_val], members[cut_val:])
        for split_name, chunk in zip(SPLITS, chunks):
            splits[split_name].extend(chunk)
        print("%-18s%8d%8d%8d%8d" % (class_name, n, *(len(c) for c in chunks)))

    totals = [len(splits[s]) for s in SPLITS]
    print("%-18s%8d%8d%8d%8d" % ("TOTAL", sum(totals), totals[0], totals[1], totals[2]))
    _check_splits(splits, samples)
    return splits


def split_from_manifest(samples, manifest_path: Path):
    """Rebuild an exact split recorded by a previous run's split_manifest.json.

    This is what actually guarantees reproducibility. Re-deriving a split from a seed only
    reproduces it while the splitting code consumes random numbers in the same order -- edit
    the function and the "same" seed silently yields a different split, moving test images
    into train and invalidating every trained checkpoint and every metric without raising
    anything. The manifest is immune to that: it names the images.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = manifest["assignments"]
    print("\n" + "=" * 78)
    print("SPLIT RESTORED FROM MANIFEST  %s" % manifest_path)
    print("=" * 78)
    print("recorded seed=%s ratios=%s" % (manifest.get("seed"), manifest.get("ratios")))

    check(sorted(manifest.get("class_names", CLASS_NAMES)) == sorted(CLASS_NAMES),
          "manifest class list %s does not match CLASS_NAMES %s"
          % (manifest.get("class_names"), list(CLASS_NAMES)))

    by_stem = {s.stem: s for s in samples}
    recorded = {stem for stems in assignments.values() for stem in stems}
    missing = sorted(recorded - set(by_stem))
    extra = sorted(set(by_stem) - recorded)
    check(not missing, "manifest names %d image(s) absent from the source tree: %s"
          % (len(missing), missing[:10]))
    check(not extra, "source tree has %d image(s) the manifest does not place: %s"
          % (len(extra), extra[:10]))

    splits = {name: [by_stem[stem] for stem in sorted(assignments[name])] for name in SPLITS}
    print("%-18s%8s%8s%8s" % ("", "train", "val", "test"))
    print("%-18s%8d%8d%8d" % ("images", *(len(splits[s]) for s in SPLITS)))
    _check_splits(splits, samples)
    return splits


# --------------------------------------------------------------------------------------
# STEP 5 - Write images, labels, data.yaml and manifest
# --------------------------------------------------------------------------------------

def write_dataset(splits, dst: Path, force: bool):
    if dst.exists() and any(dst.iterdir()):
        check(force, "%s already exists and is not empty -- pass --force to replace it" % dst)
        print("\n--force given: removing existing %s" % dst)
        shutil.rmtree(dst)

    print("\n" + "=" * 78)
    print("WRITING DATASET -> %s" % dst)
    print("=" * 78)

    for split_name in SPLITS:
        image_dir = dst / split_name / "images"
        label_dir = dst / split_name / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for sample in sorted(splits[split_name], key=lambda s: s.stem):
            shutil.copy2(sample.image_path, image_dir / sample.image_path.name)
            lines = [b.to_line() for b in sample.boxes]
            text = "\n".join(lines) + ("\n" if lines else "")
            (label_dir / (sample.stem + ".txt")).write_text(text, encoding="utf-8")
        print("    %-6s %5d images, %5d boxes"
              % (split_name, len(splits[split_name]),
                 sum(len(s.boxes) for s in splits[split_name])))

def write_data_yaml(dst: Path, seed, ratios, src: Path) -> Path:
    """Write the Ultralytics data.yaml.

    `path` is absolute so Ultralytics never falls back to resolving the dataset against its
    global settings['datasets_dir'].
    """
    names_block = "\n".join("  %d: %s" % (i, name) for i, name in enumerate(CLASS_NAMES))
    text = (
        "# EdgeLens / NEU-DET surface defect detection\n"
        "# Generated by training/prepare_neu_det.py -- do not hand-edit.\n"
        "#\n"
        "# source : %s\n"
        "# split  : %.2f/%.2f/%.2f train/val/test, stratified by primary defect class, seed=%d\n"
        "#\n"
        "# CLASS ORDER IS FROZEN. It is sorted() over the six NEU-DET defect directory names.\n"
        "# The integer ids below are baked into every label file and every trained checkpoint;\n"
        "# reordering them invalidates all existing weights.\n"
        "\n"
        "path: %s\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "\n"
        "nc: %d\n"
        "names:\n"
        "%s\n"
        % (src, ratios[0], ratios[1], ratios[2], seed,
           dst.resolve().as_posix(), len(CLASS_NAMES), names_block)
    )
    yaml_path = dst / "data.yaml"
    yaml_path.write_text(text, encoding="utf-8")
    print("\nwrote %s" % yaml_path)
    return yaml_path

def write_manifest(splits, dst: Path, seed, ratios, src: Path):
    """Record which image landed in which split, so a split can be audited or reproduced."""
    manifest = {
        "source": str(src),
        "seed": seed,
        "ratios": list(ratios),
        "class_names": list(CLASS_NAMES),
        "assignments": {name: sorted(s.stem for s in splits[name]) for name in SPLITS},
    }
    path = dst / "split_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote %s" % path)


# --------------------------------------------------------------------------------------
# STEP 6 - Report per-class and per-split statistics
# --------------------------------------------------------------------------------------

def report_statistics(splits):
    print("\n" + "=" * 78)
    print("PER-CLASS BOX COUNTS PER SPLIT")
    print("=" * 78)

    box_counts = {
        name: Counter(b.class_name for s in splits[name] for b in s.boxes) for name in SPLITS
    }
    image_counts = {
        name: Counter(c for s in splits[name] for c in {b.class_name for b in s.boxes})
        for name in SPLITS
    }

    header = ("%-18s" % "class") + "".join("%10s" % s for s in SPLITS) + \
             ("%10s%9s%8s%8s" % ("total", "train%", "val%", "test%"))
    print(header)
    print("-" * len(header))
    for class_name in CLASS_NAMES:
        row = [box_counts[s][class_name] for s in SPLITS]
        total = sum(row)
        shares = [100.0 * v / total if total else 0.0 for v in row]
        print(("%-18s" % class_name) + "".join("%10d" % v for v in row)
              + ("%10d%8.1f%%%7.1f%%%7.1f%%" % (total, shares[0], shares[1], shares[2])))
    grand = [sum(box_counts[s].values()) for s in SPLITS]
    print("-" * len(header))
    print(("%-18s" % "ALL BOXES") + "".join("%10d" % v for v in grand) + ("%10d" % sum(grand)))
    print(("%-18s" % "IMAGES") + "".join("%10d" % len(splits[s]) for s in SPLITS)
          + ("%10d" % sum(len(splits[s]) for s in SPLITS)))

    print("\nimages containing at least one box of each class (an image can count twice):")
    print(("%-18s" % "class") + "".join("%10s" % s for s in SPLITS))
    for class_name in CLASS_NAMES:
        print(("%-18s" % class_name) + "".join("%10d" % image_counts[s][class_name] for s in SPLITS))

    print("\n" + "=" * 78)
    print("BOXES-PER-IMAGE DISTRIBUTION")
    print("=" * 78)
    per_split = {name: Counter(len(s.boxes) for s in splits[name]) for name in SPLITS}
    overall = Counter()
    for counter in per_split.values():
        overall.update(counter)

    grand_total = sum(overall.values())
    print(("%10s" % "boxes/img") + "".join("%10s" % s for s in SPLITS)
          + ("%10s%9s" % ("total", "share")) + "  histogram")
    for n in range(0, (max(overall) if overall else 0) + 1):
        if overall[n] == 0:
            continue
        share = 100.0 * overall[n] / grand_total
        print(("%10d" % n) + "".join("%10d" % per_split[s][n] for s in SPLITS)
              + ("%10d%8.1f%%" % (overall[n], share))
              + "  " + "#" * max(1, int(round(share / 2))))

    for split_name in SPLITS:
        images = splits[split_name]
        counts = sorted(len(s.boxes) for s in images)
        boxes = sum(counts)
        median = counts[len(counts) // 2] if counts else 0
        print("\n%-6s mean %.2f boxes/image, median %d, max %d"
              % (split_name, boxes / max(len(images), 1), median, counts[-1] if counts else 0))


# --------------------------------------------------------------------------------------
# STEP 7 - Entry point
# --------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    ratios = tuple(args.ratios)
    check(all(r > 0 for r in ratios), "all split ratios must be positive, got %s" % (ratios,))

    try:
        samples = discover(args.src)
        stats = convert(samples, drop_difficult=args.drop_difficult)
        if args.from_manifest:
            splits = split_from_manifest(samples, args.from_manifest)
        else:
            splits = stratified_split(samples, ratios, args.seed)

        if args.dry_run:
            report_statistics(splits)
            print("\n--dry-run: nothing was written.")
            return 0

        write_dataset(splits, args.dst, force=args.force)
        yaml_path = write_data_yaml(args.dst, args.seed, ratios, args.src)
        write_manifest(splits, args.dst, args.seed, ratios, args.src)
        report_statistics(splits)

        print("\n" + "=" * 78)
        print("DONE")
        print("=" * 78)
        print("dataset  : %s" % args.dst.resolve())
        print("data.yaml: %s" % yaml_path.resolve())
        print("boxes    : %d" % stats["total_boxes"])
        print("\nNext:")
        print('    python training/train_yolo.py --model yolov8n.pt --data "%s" \\'
              % yaml_path.resolve())
        print("        --epochs 100 --batch 16 --imgsz 640 --seed 0 --patience 50")
        return 0
    except DatasetError as exc:
        print("\nFAILED: %s" % exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
