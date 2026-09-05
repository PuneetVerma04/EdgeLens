#!/usr/bin/env python
"""Render a benchmark payload into a markdown report.

Kept separate from run_benchmark.py so a report can be regenerated from stored results
without re-running a ~15 minute benchmark:

    python benchmarks/write_results.py benchmarks/results_cpu1.json

For a multi-run report use aggregate_runs.py instead -- it reuses write_markdown() below
over median values and splices in run-to-run spread, so the analysis lives in one place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The recommendation rule, stated in the document so a reader can disagree with the rule
# rather than reverse-engineer it from the numbers.
ACCURACY_TOLERANCE_PP = 1.0  # "within 1 pp of the best measured mAP@0.5"


def _fmt_bytes(n: int) -> str:
    return f"{n:,}"


def _pp(delta: float) -> str:
    """Percentage points, signed, two decimals -- never rounded to zero if non-zero."""
    if delta == 0:
        return "0.00"
    text = f"{delta * 100:+.2f}"
    if text in ("+0.00", "-0.00"):
        return f"{delta * 100:+.4f}"
    return text


def _speed(ratio: float) -> str:
    return f"{ratio:.2f}x"


def write_markdown(payload: dict, out_path: Path,
                   extra_after_results: list | None = None,
                   extra_at_end: list | None = None,
                   subtitle: str | None = None) -> None:
    """Render a benchmark payload to markdown.

    `extra_after_results` is spliced in directly after the headline results table, so
    aggregate_runs.py can add run-to-run spread and ratio tables without duplicating any
    of the analysis below it.
    """
    host = payload["host"]
    protocol = payload["protocol"]
    results = payload["results"]

    main = [r for r in results if not r.get("sweep")]
    sweep_model = "yolo26n"
    sweep = [r for r in results
             if r["model"] == sweep_model and r["backend"] == "ONNX Runtime"
             and r["precision"] == "FP32"]
    sweep.sort(key=lambda r: r["imgsz"])

    artefacts = [r for r in main if r["backend"] == "ONNX Runtime"]
    baselines = {r["model"]: r for r in main if r["backend"] == "PyTorch"}

    lines: list[str] = []
    add = lines.append

    device = protocol.get("device", "cpu")
    add("# Edge benchmark (%s) - NEU-DET surface defect detection"
        % ("CPU" if device == "cpu" else "GPU"))
    add("")
    if subtitle:
        add(subtitle)
    else:
        add(f"Generated {payload['generated_utc']} by `benchmarks/run_benchmark.py`. "
            f"Raw data in [`results.json`](results.json).")
    add("")

    # --- host ---------------------------------------------------------------------------
    add("## Test machine")
    add("")
    add("| | |")
    add("| --- | --- |")
    add(f"| CPU | {host['cpu']} |")
    add(f"| Cores | {host['physical_cores']} physical / {host['logical_cores']} logical |")
    add(f"| Threads per variant | {host['threads_used']} |")
    add(f"| RAM | {host['ram_gib']} GiB |")
    add(f"| OS | {host['platform']} |")
    add(f"| Python | {host['python']} |")
    add(f"| PyTorch | {host['torch']} |")
    add(f"| ONNX Runtime | {host['onnxruntime']} ({', '.join(host['onnxruntime_providers'])}) |")
    add(f"| Ultralytics | {host['ultralytics']} |")
    add("")
    if device == "cpu":
        add("**All inference on CPU.** No GPU is used anywhere in this benchmark, "
            "including by the PyTorch baselines.")
    else:
        add(f"**All inference on GPU** (`device={device}`, {host.get('gpu')}). CUDA "
            "kernels launch asynchronously, so every timed iteration calls "
            "`torch.cuda.synchronize()` before stopping the clock - without it the "
            "numbers would measure kernel launch overhead, not execution.")
    add("")

    # --- protocol -----------------------------------------------------------------------
    add("## Protocol")
    add("")
    add(f"- **Latency**: {protocol['warmup']} warmup runs, then {protocol['runs']} timed runs "
        f"at batch 1. Percentiles use the **{protocol['percentile_method']}**, so every figure "
        f"is an observation that actually occurred rather than an interpolation.")
    add(f"- Timing covers the **full `predict()` call** - preprocess, inference, and "
        f"postprocess/NMS - because that is what a caller waits for. The per-stage breakdown "
        f"below separates them.")
    add(f"- Images are decoded to memory before timing starts and cycled from a pool of "
        f"{protocol['latency_image_pool']} test images, so JPEG decode and disk I/O are excluded "
        f"but a realistic spread of detection counts is included.")
    add(f"- **Isolation**: {protocol['isolation']}. Peak working set is a per-process "
        f"high-water mark, so measuring several variants in one process would report the "
        f"maximum across all of them for each one.")
    add(f"- **Accuracy**: mAP on the **{protocol['split']} split** (held out from training and "
        f"from INT8 calibration), computed by the Ultralytics validator.")
    add(f"- Inference settings identical across variants: conf={protocol['conf']}, "
        f"iou={protocol['iou']}, max_det={protocol['max_det']}.")
    add("")

    # --- main table ---------------------------------------------------------------------
    add("## Results")
    add("")
    add("The four exported artefacts are the ONNX rows. The two PyTorch rows are the "
        "references they were exported from, included so the ONNX gains are measurable "
        "against something.")
    add("")
    add("| Variant | Backend | Precision | imgsz | p50 (ms) | p95 (ms) | Peak mem (MiB) | "
        "Size (MB) | Size (bytes) | mAP@0.5 | mAP@0.5:0.95 |")
    add("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in main:
        add(f"| `{r['id']}` | {r['backend']} | {r['precision']} | {r['imgsz']} | "
            f"{r['latency']['p50_ms']:.2f} | {r['latency']['p95_ms']:.2f} | "
            f"{r['memory_mib']['peak_process']:.1f} | {r['size_mb']:.2f} | "
            f"{_fmt_bytes(r['size_bytes'])} | {r['accuracy']['map50']:.4f} | "
            f"{r['accuracy']['map50_95']:.4f} |")
    add("")
    if extra_after_results:
        lines.extend(extra_after_results)
        add("")

    # --- latency detail -----------------------------------------------------------------
    add("### Latency detail")
    add("")
    add("| Variant | min | p50 | p90 | p95 | p99 | max | mean | stdev | FPS @ p50 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in main:
        lat = r["latency"]
        add(f"| `{r['id']}` | {lat['min_ms']:.2f} | {lat['p50_ms']:.2f} | {lat['p90_ms']:.2f} | "
            f"{lat['p95_ms']:.2f} | {lat['p99_ms']:.2f} | {lat['max_ms']:.2f} | "
            f"{lat['mean_ms']:.2f} | {lat['stdev_ms']:.2f} | "
            f"{lat['throughput_fps_at_p50']:.1f} |")
    add("")
    add("All times in milliseconds. Where each millisecond goes:")
    add("")
    add("| Variant | preprocess | inference | postprocess |")
    add("| --- | ---: | ---: | ---: |")
    for r in main:
        stage = r["latency"]["stage_mean_ms"]
        add(f"| `{r['id']}` | {stage['preprocess']:.2f} | {stage['inference']:.2f} | "
            f"{stage['postprocess']:.2f} |")
    add("")

    # --- memory -------------------------------------------------------------------------
    add("### Memory detail")
    add("")
    add("| Variant | Baseline RSS | After model load | Model delta | Peak working set |")
    add("| --- | ---: | ---: | ---: | ---: |")
    for r in main:
        mem = r["memory_mib"]
        add(f"| `{r['id']}` | {mem['baseline_rss']:.1f} | {mem['after_model_load']:.1f} | "
            f"{mem['model_load_delta']:+.1f} | {mem['peak_process']:.1f} |")
    add("")
    add("All values MiB. Baseline RSS is the process after imports but before the model "
        "loads; it is not attributable to the artefact. The peak column is what a deployment "
        "must actually provision for.")
    add("")

    # --- vs PyTorch ---------------------------------------------------------------------
    add("### ONNX artefacts vs their PyTorch reference")
    add("")
    add("| Artefact | p50 speedup | p95 speedup | Size vs .pt | mAP@0.5 delta | "
        "mAP@0.5:0.95 delta |")
    add("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in artefacts:
        base = baselines.get(r["model"])
        if not base or base["imgsz"] != r["imgsz"]:
            continue
        add(f"| `{r['id']}` | "
            f"{_speed(base['latency']['p50_ms'] / r['latency']['p50_ms'])} | "
            f"{_speed(base['latency']['p95_ms'] / r['latency']['p95_ms'])} | "
            f"{r['size_mb'] / base['size_mb']:.2f}x | "
            f"{_pp(r['accuracy']['map50'] - base['accuracy']['map50'])} pp | "
            f"{_pp(r['accuracy']['map50_95'] - base['accuracy']['map50_95'])} pp |")
    add("")

    # --- resolution sweep ---------------------------------------------------------------
    if len(sweep) > 1:
        add(f"## Resolution trade-off ({sweep_model}, ONNX FP32)")
        add("")
        add("NEU-DET images are natively **200x200**, so every setting here upscales. The "
            "question is whether paying for that upscale buys accuracy.")
        add("")
        add("| imgsz | Upscale from 200px | p50 (ms) | p95 (ms) | Peak mem (MiB) | "
            "mAP@0.5 | mAP@0.5:0.95 |")
        add("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in sweep:
            add(f"| {r['imgsz']} | {r['imgsz'] / 200:.2f}x | {r['latency']['p50_ms']:.2f} | "
                f"{r['latency']['p95_ms']:.2f} | {r['memory_mib']['peak_process']:.1f} | "
                f"{r['accuracy']['map50']:.4f} | {r['accuracy']['map50_95']:.4f} |")
        add("")
        add("Relative to the smallest setting:")
        add("")
        add("| imgsz | p50 cost | mAP@0.5 gain | mAP@0.5:0.95 gain | mAP@0.5 per extra ms |")
        add("| ---: | ---: | ---: | ---: | ---: |")
        smallest = sweep[0]
        for r in sweep:
            extra_ms = r["latency"]["p50_ms"] - smallest["latency"]["p50_ms"]
            gain50 = r["accuracy"]["map50"] - smallest["accuracy"]["map50"]
            per_ms = f"{gain50 * 100 / extra_ms:+.3f} pp/ms" if extra_ms > 1e-9 else "-"
            add(f"| {r['imgsz']} | {extra_ms:+.2f} ms | "
                f"{_pp(gain50)} pp | "
                f"{_pp(r['accuracy']['map50_95'] - smallest['accuracy']['map50_95'])} pp | "
                f"{per_ms} |")
        add("")

    # --- per-class ----------------------------------------------------------------------
    add("## Per-class AP@0.5")
    add("")
    class_names = sorted({c for r in main for c in r["accuracy"]["per_class_ap50"]})
    if class_names:
        add("| Variant | " + " | ".join(class_names) + " |")
        add("| --- | " + " | ".join("---:" for _ in class_names) + " |")
        for r in main:
            cells = [f"{r['accuracy']['per_class_ap50'].get(c, float('nan')):.4f}"
                     for c in class_names]
            add(f"| `{r['id']}` | " + " | ".join(cells) + " |")
        add("")

    # --- quantization diagnosis ---------------------------------------------------------
    diagnosis = _quantization_diagnosis(results)
    if diagnosis:
        lines.extend(diagnosis)
        add("")

    # --- analysis -----------------------------------------------------------------------
    add("## Which operating point to ship")
    add("")
    lines.extend(_analysis(main, results, artefacts, baselines, sweep))
    add("")
    if extra_at_end:
        lines.extend(extra_at_end)
        add("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quantization_diagnosis(results) -> list[str]:
    """Explain the INT8 latency result by inspecting the graphs ONNX Runtime actually runs.

    Computed here rather than stored in results.json so the report can be regenerated
    without re-running the benchmark. Cheap: it only loads and optimises the graphs.
    """
    int8 = [r for r in results if r["precision"] == "INT8" and r["backend"] == "ONNX Runtime"]
    if not int8:
        return []
    try:
        import collections

        import onnx
        import onnxruntime as ort
    except ImportError:
        return []

    import tempfile

    rows = []
    for r in int8:
        path = Path(r["path"])
        fp32 = next((o for o in results
                     if o["model"] == r["model"] and o["precision"] == "FP32"
                     and o["backend"] == "ONNX Runtime" and o["imgsz"] == r["imgsz"]), None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                options.optimized_model_filepath = str(Path(tmp) / "opt.onnx")
                ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
                counts = collections.Counter(
                    n.op_type for n in onnx.load(options.optimized_model_filepath).graph.node)
        except Exception as exc:
            # Never let the diagnosis sink a report -- but say why it is missing, otherwise
            # a genuine bug here is indistinguishable from "onnxruntime is not installed".
            print(f"WARNING: could not inspect INT8 graph for {r['id']}: "
                  f"{type(exc).__name__}: {exc}")
            continue
        integer_conv = counts.get("QLinearConv", 0) + counts.get("ConvInteger", 0)
        float_conv = counts.get("Conv", 0)
        qdq = counts.get("QuantizeLinear", 0) + counts.get("DequantizeLinear", 0)
        total_conv = integer_conv + float_conv
        rows.append({
            "id": r["id"], "integer_conv": integer_conv, "float_conv": float_conv,
            "total_conv": total_conv, "qdq": qdq,
            "fused_pct": 100.0 * integer_conv / total_conv if total_conv else 0.0,
            "fp32_infer": fp32["latency"]["stage_mean_ms"]["inference"] if fp32 else None,
            "int8_infer": r["latency"]["stage_mean_ms"]["inference"],
        })
    if not rows:
        return []

    out = ["## Why INT8 is slower here, not faster", ""]
    out.append(
        "The INT8 artefacts are ~3.3x smaller on disk but **1.7-2.5x slower** to run. That is "
        "not a measurement artefact - the per-stage breakdown shows the cost is entirely in "
        "the inference stage, with preprocess and postprocess unchanged. Inspecting the graph "
        "that ONNX Runtime actually executes (after `ORT_ENABLE_ALL` optimisation, which is "
        "where QDQ fusion happens) shows why:"
    )
    out.append("")
    out.append("| Artefact | Convs fused to integer kernels | Convs still float | "
               "Quantize/Dequantize nodes | Inference FP32 -> INT8 |")
    out.append("| --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        timing = (f"{row['fp32_infer']:.2f} -> {row['int8_infer']:.2f} ms"
                  if row["fp32_infer"] else f"{row['int8_infer']:.2f} ms")
        out.append(f"| `{row['id']}` | {row['integer_conv']} / {row['total_conv']} "
                   f"({row['fused_pct']:.1f}%) | {row['float_conv']} | {row['qdq']} | {timing} |")
    out.append("")
    out.append(
        "Almost none of the convolutions became integer kernels. The weights really are stored "
        "as INT8 - that is where the file-size win comes from - but at runtime the great "
        "majority of convs dequantize back to float, run as float `Conv`, and requantize. The "
        "model therefore pays the full FP32 compute cost **plus** several hundred conversion "
        "ops it did not have before."
    )
    out.append("")
    out.append(
        "The likely cause is the node exclusion list. Ultralytics quantizes only Conv/Gemm/"
        "MatMul and excludes every other node by name (`nodes_to_exclude`), which is a "
        "deliberate and correct choice for accuracy - one INT8 scale cannot span box "
        "coordinates and class probabilities. But in a YOLO backbone the convolutions are "
        "separated by SiLU (`Sigmoid` + `Mul`), `Concat` and `Add`, so excluding those breaks "
        "the contiguous quantized regions ONNX Runtime needs in order to fuse "
        "`DequantizeLinear -> Conv -> QuantizeLinear` into a single `QLinearConv`. Both models "
        "converge to ~48 ms inference regardless of architecture, which is consistent with the "
        "conversion machinery dominating rather than the model."
    )
    out.append("")
    out.append(
        "**Practical consequence:** on this CPU and runtime, ship INT8 only when the binding "
        "constraint is storage or memory footprint. If the constraint is latency, INT8 as "
        "exported here is the wrong tool - lowering the input resolution buys far more (see "
        "the resolution table above). Worth revisiting with ONNX Runtime's "
        "`quant_pre_process` step, or a per-op exclusion list that keeps quantized regions "
        "contiguous, before concluding INT8 cannot help on this hardware."
    )
    return out


def _analysis(main, all_results, artefacts, baselines, sweep) -> list[str]:
    """Prose driven entirely by the measured numbers."""
    best_acc = max(main, key=lambda r: r["accuracy"]["map50"])
    fastest = min(main, key=lambda r: r["latency"]["p50_ms"])

    # Pareto front on (p50 latency down, mAP@0.5 up), over EVERY measured variant including
    # the reduced-resolution ones -- those are real deployable operating points, not just
    # illustrations of a trade-off.
    front = [
        r for r in all_results
        if not any(o["latency"]["p50_ms"] <= r["latency"]["p50_ms"]
                   and o["accuracy"]["map50"] >= r["accuracy"]["map50"]
                   and o["id"] != r["id"]
                   and (o["latency"]["p50_ms"] < r["latency"]["p50_ms"]
                        or o["accuracy"]["map50"] > r["accuracy"]["map50"])
                   for o in all_results)
    ]
    front.sort(key=lambda r: r["latency"]["p50_ms"])

    threshold = best_acc["accuracy"]["map50"] - ACCURACY_TOLERANCE_PP / 100.0
    eligible = [r for r in main if r["accuracy"]["map50"] >= threshold]
    pick = min(eligible, key=lambda r: r["latency"]["p50_ms"])

    out = []
    out.append(
        f"**Rule used:** among every variant whose mAP@0.5 is within "
        f"{ACCURACY_TOLERANCE_PP:.1f} pp of the best measured, take the lowest p50 latency. "
        f"Best measured mAP@0.5 is **{best_acc['accuracy']['map50']:.4f}** "
        f"(`{best_acc['id']}`), so the eligibility threshold is "
        f"{threshold:.4f} and {len(eligible)} of {len(main)} variants qualify."
    )
    out.append("")
    out.append(
        f"**The pick is `{pick['id']}`** - "
        f"p50 **{pick['latency']['p50_ms']:.2f} ms**, p95 {pick['latency']['p95_ms']:.2f} ms, "
        f"mAP@0.5 **{pick['accuracy']['map50']:.4f}**, mAP@0.5:0.95 "
        f"{pick['accuracy']['map50_95']:.4f}, {pick['size_mb']:.2f} MB on disk, "
        f"{pick['memory_mib']['peak_process']:.1f} MiB peak."
    )
    out.append("")

    if pick["id"] != best_acc["id"]:
        acc_cost = best_acc["accuracy"]["map50"] - pick["accuracy"]["map50"]
        acc_cost_5095 = best_acc["accuracy"]["map50_95"] - pick["accuracy"]["map50_95"]
        speed_gain = best_acc["latency"]["p50_ms"] / pick["latency"]["p50_ms"]
        out.append(
            f"Against the most accurate variant (`{best_acc['id']}`, mAP@0.5 "
            f"{best_acc['accuracy']['map50']:.4f}, p50 {best_acc['latency']['p50_ms']:.2f} ms), "
            f"the pick gives up **{acc_cost * 100:.2f} pp** of mAP@0.5 "
            f"and {acc_cost_5095 * 100:.2f} pp "
            f"of mAP@0.5:0.95 to run **{speed_gain:.2f}x faster** "
            f"({best_acc['latency']['p50_ms']:.2f} ms -> {pick['latency']['p50_ms']:.2f} ms, "
            f"a saving of {best_acc['latency']['p50_ms'] - pick['latency']['p50_ms']:.2f} ms "
            f"per frame)."
        )
    else:
        out.append(
            f"`{pick['id']}` is both the most accurate variant and fast enough to win on the "
            f"rule above - there is no accuracy/latency trade to make here."
        )
    out.append("")

    # The lowest-latency point on the frontier, which is usually a reduced-resolution
    # variant and is the answer whenever the latency budget, not accuracy, is binding.
    budget = front[0] if front else fastest
    if budget["id"] != pick["id"]:
        out.append(
            f"**If latency is the binding constraint instead**, the frontier's fastest point "
            f"is `{budget['id']}` at **{budget['latency']['p50_ms']:.2f} ms** p50 "
            f"({pick['latency']['p50_ms'] / budget['latency']['p50_ms']:.2f}x quicker than the "
            f"pick, {pick['latency']['p50_ms'] - budget['latency']['p50_ms']:.2f} ms saved per "
            f"frame), at mAP@0.5 {budget['accuracy']['map50']:.4f} - a drop of "
            f"{abs(budget['accuracy']['map50'] - pick['accuracy']['map50']) * 100:.2f} pp "
            f"against the pick."
        )
        out.append("")

    fastest = min(all_results, key=lambda r: r["latency"]["p50_ms"])
    if fastest["id"] not in {pick["id"], budget["id"]}:
        out.append(
            f"The outright fastest variant is `{fastest['id']}` at "
            f"{fastest['latency']['p50_ms']:.2f} ms p50 "
            f"({pick['latency']['p50_ms'] / fastest['latency']['p50_ms']:.2f}x quicker than the "
            f"pick), but it scores mAP@0.5 {fastest['accuracy']['map50']:.4f} - "
            f"{_pp(fastest['accuracy']['map50'] - best_acc['accuracy']['map50'])} pp against the "
            f"best. That is outside the tolerance, so it only makes sense if the latency "
            f"budget forces it."
        )
        out.append("")

    out.append("**Pareto front** (nothing is both faster and more accurate):")
    out.append("")
    for r in front:
        out.append(f"- `{r['id']}` - p50 {r['latency']['p50_ms']:.2f} ms, "
                   f"mAP@0.5 {r['accuracy']['map50']:.4f}, "
                   f"mAP@0.5:0.95 {r['accuracy']['map50_95']:.4f}, "
                   f"{r['size_mb']:.2f} MB")
    out.append("")

    # INT8 verdict, per model, with real numbers.
    for model in sorted({r["model"] for r in artefacts}):
        fp32 = next((r for r in artefacts if r["model"] == model and r["precision"] == "FP32"
                     and not r.get("sweep")), None)
        int8 = next((r for r in artefacts if r["model"] == model and r["precision"] == "INT8"), None)
        if not fp32 or not int8:
            continue
        ratio = int8["latency"]["p50_ms"] / fp32["latency"]["p50_ms"]
        direction = f"**{ratio:.2f}x SLOWER**" if ratio > 1 else f"**{1 / ratio:.2f}x faster**"
        out.append(
            f"**INT8 on {model}:** p50 {fp32['latency']['p50_ms']:.2f} -> "
            f"{int8['latency']['p50_ms']:.2f} ms, i.e. {direction}; "
            f"size {fp32['size_mb']:.2f} -> {int8['size_mb']:.2f} MB "
            f"({fp32['size_mb'] / int8['size_mb']:.2f}x smaller); "
            f"mAP@0.5 {_pp(int8['accuracy']['map50'] - fp32['accuracy']['map50'])} pp, "
            f"mAP@0.5:0.95 {_pp(int8['accuracy']['map50_95'] - fp32['accuracy']['map50_95'])} pp."
        )
    out.append("")

    if len(sweep) > 1:
        lo, hi = sweep[0], sweep[-1]
        out.append(
            f"**Resolution:** going from imgsz {lo['imgsz']} to {hi['imgsz']} costs "
            f"{hi['latency']['p50_ms'] - lo['latency']['p50_ms']:+.2f} ms p50 "
            f"({hi['latency']['p50_ms'] / lo['latency']['p50_ms']:.2f}x) and buys "
            f"{_pp(hi['accuracy']['map50'] - lo['accuracy']['map50'])} pp of mAP@0.5 on "
            f"200x200 native images. The intermediate settings are in the table above - the "
            f"gain is not linear in pixels, so the middle of that range is where the "
            f"interesting operating points are."
        )
    return out


def main(argv=None) -> int:
    usage = "\n".join([
        "usage: write_results.py <results_*.json>",
        "  Renders one benchmark run to <results_*.md> alongside it.",
        "  For a multi-run report with medians and spread, use aggregate_runs.py.",
    ])
    if not argv or argv[0] in ("-h", "--help"):
        print(usage)
        return 0 if argv else 2
    target = Path(argv[0])
    if not target.is_file():
        raise SystemExit(f"no such results file: {target}\n{usage}")
    payload = json.loads(target.read_text(encoding="utf-8"))
    out_path = target.with_suffix(".md")
    write_markdown(payload, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
