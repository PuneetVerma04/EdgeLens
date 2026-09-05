# Edge benchmark (CPU) - NEU-DET surface defect detection

Generated 2026-08-30T10:09:21Z by `benchmarks/run_benchmark.py`. Raw data in [`results.json`](results.json).

## Test machine

| | |
| --- | --- |
| CPU | AMD Ryzen 7 7840HS w/ Radeon 780M Graphics |
| Cores | 8 physical / 16 logical |
| Threads per variant | 8 |
| RAM | 15.29 GiB |
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.12.9 |
| PyTorch | 2.13.0+cu130 |
| ONNX Runtime | 1.29.0 (AzureExecutionProvider, CPUExecutionProvider) |
| Ultralytics | 8.4.126 |

**All inference on CPU.** No GPU is used anywhere in this benchmark, including by the PyTorch baselines.

## Protocol

- **Latency**: 20 warmup runs, then 200 timed runs at batch 1. Percentiles use the **nearest-rank (no interpolation)**, so every figure is an observation that actually occurred rather than an interpolation.
- Timing covers the **full `predict()` call** - preprocess, inference, and postprocess/NMS - because that is what a caller waits for. The per-stage breakdown below separates them.
- Images are decoded to memory before timing starts and cycled from a pool of 20 test images, so JPEG decode and disk I/O are excluded but a realistic spread of detection counts is included.
- **Isolation**: one subprocess per variant. Peak working set is a per-process high-water mark, so measuring several variants in one process would report the maximum across all of them for each one.
- **Accuracy**: mAP on the **test split** (held out from training and from INT8 calibration), computed by the Ultralytics validator.
- Inference settings identical across variants: conf=0.25, iou=0.7, max_det=300.

## Results

The four exported artefacts are the ONNX rows. The two PyTorch rows are the references they were exported from, included so the ONNX gains are measurable against something.

| Variant | Backend | Precision | imgsz | p50 (ms) | p95 (ms) | Peak mem (MiB) | Size (MB) | Size (bytes) | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | PyTorch | FP32 | 640 | 33.82 | 36.38 | 714.6 | 5.97 | 6,258,410 | 0.7561 | 0.4233 |
| `yolo26n-pytorch-640` | PyTorch | FP32 | 640 | 35.60 | 39.14 | 724.2 | 5.15 | 5,397,573 | 0.7332 | 0.4184 |
| `yolov8n-onnx-fp32-640` | ONNX Runtime | FP32 | 640 | 27.73 | 28.58 | 824.5 | 11.70 | 12,269,259 | 0.7561 | 0.4233 |
| `yolov8n-onnx-int8-640` | ONNX Runtime | INT8 | 640 | 50.34 | 51.86 | 833.3 | 3.23 | 3,389,567 | 0.7555 | 0.4301 |
| `yolo26n-onnx-fp32-640` | ONNX Runtime | FP32 | 640 | 21.88 | 22.54 | 871.3 | 9.35 | 9,809,346 | 0.7332 | 0.4184 |
| `yolo26n-onnx-int8-640` | ONNX Runtime | INT8 | 640 | 50.67 | 52.17 | 885.5 | 2.80 | 2,938,750 | 0.7225 | 0.4067 |

### Latency detail

| Variant | min | p50 | p90 | p95 | p99 | max | mean | stdev | FPS @ p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 26.94 | 33.82 | 35.83 | 36.38 | 37.88 | 39.21 | 33.69 | 1.90 | 29.6 |
| `yolo26n-pytorch-640` | 29.74 | 35.60 | 38.11 | 39.14 | 40.89 | 43.42 | 35.73 | 2.04 | 28.1 |
| `yolov8n-onnx-fp32-640` | 27.05 | 27.73 | 28.22 | 28.58 | 28.97 | 30.48 | 27.81 | 0.39 | 36.1 |
| `yolov8n-onnx-int8-640` | 48.98 | 50.34 | 51.24 | 51.86 | 53.36 | 73.53 | 50.61 | 1.74 | 19.9 |
| `yolo26n-onnx-fp32-640` | 21.43 | 21.88 | 22.35 | 22.54 | 22.97 | 23.21 | 21.93 | 0.31 | 45.7 |
| `yolo26n-onnx-int8-640` | 48.88 | 50.67 | 51.54 | 52.17 | 52.79 | 52.94 | 50.79 | 0.62 | 19.7 |

All times in milliseconds. Where each millisecond goes:

| Variant | preprocess | inference | postprocess |
| --- | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 2.06 | 30.90 | 0.46 |
| `yolo26n-pytorch-640` | 2.02 | 33.01 | 0.19 |
| `yolov8n-onnx-fp32-640` | 2.65 | 24.29 | 0.63 |
| `yolov8n-onnx-int8-640` | 2.53 | 47.20 | 0.63 |
| `yolo26n-onnx-fp32-640` | 2.63 | 18.57 | 0.28 |
| `yolo26n-onnx-int8-640` | 2.66 | 47.41 | 0.28 |

### Memory detail

| Variant | Baseline RSS | After model load | Model delta | Peak working set |
| --- | ---: | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 502.4 | 533.4 | +30.9 | 714.6 |
| `yolo26n-pytorch-640` | 502.7 | 532.2 | +29.5 | 724.2 |
| `yolov8n-onnx-fp32-640` | 502.7 | 513.8 | +11.1 | 824.5 |
| `yolov8n-onnx-int8-640` | 502.5 | 512.9 | +10.4 | 833.3 |
| `yolo26n-onnx-fp32-640` | 502.8 | 513.3 | +10.5 | 871.3 |
| `yolo26n-onnx-int8-640` | 502.7 | 513.1 | +10.4 | 885.5 |

All values MiB. Baseline RSS is the process after imports but before the model loads; it is not attributable to the artefact. The peak column is what a deployment must actually provision for.

### ONNX artefacts vs their PyTorch reference

| Artefact | p50 speedup | p95 speedup | Size vs .pt | mAP@0.5 delta | mAP@0.5:0.95 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-onnx-fp32-640` | 1.22x | 1.27x | 1.96x | 0.00 pp | 0.00 pp |
| `yolov8n-onnx-int8-640` | 0.67x | 0.70x | 0.54x | -0.06 pp | +0.68 pp |
| `yolo26n-onnx-fp32-640` | 1.63x | 1.74x | 1.82x | +0.0001 pp | +0.0001 pp |
| `yolo26n-onnx-int8-640` | 0.70x | 0.75x | 0.54x | -1.08 pp | -1.17 pp |

## Resolution trade-off (yolo26n, ONNX FP32)

NEU-DET images are natively **200x200**, so every setting here upscales. The question is whether paying for that upscale buys accuracy.

| imgsz | Upscale from 200px | p50 (ms) | p95 (ms) | Peak mem (MiB) | mAP@0.5 | mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1.28x | 5.20 | 5.43 | 728.9 | 0.6115 | 0.3129 |
| 320 | 1.60x | 6.77 | 7.12 | 744.8 | 0.6986 | 0.3888 |
| 640 | 3.20x | 21.88 | 22.54 | 871.3 | 0.7332 | 0.4184 |

Relative to the smallest setting:

| imgsz | p50 cost | mAP@0.5 gain | mAP@0.5:0.95 gain | mAP@0.5 per extra ms |
| ---: | ---: | ---: | ---: | ---: |
| 256 | +0.00 ms | 0.00 pp | 0.00 pp | - |
| 320 | +1.57 ms | +8.71 pp | +7.59 pp | +5.537 pp/ms |
| 640 | +16.69 ms | +12.17 pp | +10.55 pp | +0.730 pp/ms |

## Per-class AP@0.5

| Variant | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 0.4576 | 0.8067 | 0.9172 | 0.7864 | 0.6384 | 0.9303 |
| `yolo26n-pytorch-640` | 0.4625 | 0.8095 | 0.8977 | 0.7389 | 0.5881 | 0.9026 |
| `yolov8n-onnx-fp32-640` | 0.4576 | 0.8067 | 0.9172 | 0.7864 | 0.6384 | 0.9303 |
| `yolov8n-onnx-int8-640` | 0.4262 | 0.8103 | 0.9347 | 0.7893 | 0.6433 | 0.9293 |
| `yolo26n-onnx-fp32-640` | 0.4625 | 0.8095 | 0.8977 | 0.7389 | 0.5881 | 0.9026 |
| `yolo26n-onnx-int8-640` | 0.4493 | 0.7883 | 0.8939 | 0.7285 | 0.5908 | 0.8839 |

## Why INT8 is slower here, not faster

The INT8 artefacts are ~3.3x smaller on disk but **1.7-2.5x slower** to run. That is not a measurement artefact - the per-stage breakdown shows the cost is entirely in the inference stage, with preprocess and postprocess unchanged. Inspecting the graph that ONNX Runtime actually executes (after `ORT_ENABLE_ALL` optimisation, which is where QDQ fusion happens) shows why:

| Artefact | Convs fused to integer kernels | Convs still float | Quantize/Dequantize nodes | Inference FP32 -> INT8 |
| --- | ---: | ---: | ---: | ---: |
| `yolov8n-onnx-int8-640` | 7 / 64 (10.9%) | 57 | 437 | 24.29 -> 47.20 ms |
| `yolo26n-onnx-int8-640` | 12 / 102 (11.8%) | 90 | 684 | 18.57 -> 47.41 ms |

Almost none of the convolutions became integer kernels. The weights really are stored as INT8 - that is where the file-size win comes from - but at runtime the great majority of convs dequantize back to float, run as float `Conv`, and requantize. The model therefore pays the full FP32 compute cost **plus** several hundred conversion ops it did not have before.

The likely cause is the node exclusion list. Ultralytics quantizes only Conv/Gemm/MatMul and excludes every other node by name (`nodes_to_exclude`), which is a deliberate and correct choice for accuracy - one INT8 scale cannot span box coordinates and class probabilities. But in a YOLO backbone the convolutions are separated by SiLU (`Sigmoid` + `Mul`), `Concat` and `Add`, so excluding those breaks the contiguous quantized regions ONNX Runtime needs in order to fuse `DequantizeLinear -> Conv -> QuantizeLinear` into a single `QLinearConv`. Both models converge to ~48 ms inference regardless of architecture, which is consistent with the conversion machinery dominating rather than the model.

**Practical consequence:** on this CPU and runtime, ship INT8 only when the binding constraint is storage or memory footprint. If the constraint is latency, INT8 as exported here is the wrong tool - lowering the input resolution buys far more (see the resolution table above). Worth revisiting with ONNX Runtime's `quant_pre_process` step, or a per-op exclusion list that keeps quantized regions contiguous, before concluding INT8 cannot help on this hardware.

## Which operating point to ship

**Rule used:** among every variant whose mAP@0.5 is within 1.0 pp of the best measured, take the lowest p50 latency. Best measured mAP@0.5 is **0.7561** (`yolov8n-pytorch-640`), so the eligibility threshold is 0.7461 and 3 of 6 variants qualify.

**The pick is `yolov8n-onnx-fp32-640`** - p50 **27.73 ms**, p95 28.58 ms, mAP@0.5 **0.7561**, mAP@0.5:0.95 0.4233, 11.70 MB on disk, 824.5 MiB peak.

Against the most accurate variant (`yolov8n-pytorch-640`, mAP@0.5 0.7561, p50 33.82 ms), the pick gives up **0.00 pp** of mAP@0.5 and 0.00 pp of mAP@0.5:0.95 to run **1.22x faster** (33.82 ms -> 27.73 ms, a saving of 6.09 ms per frame).

**If latency is the binding constraint instead**, the frontier's fastest point is `yolo26n-onnx-fp32-256` at **5.20 ms** p50 (5.34x quicker than the pick, 22.54 ms saved per frame), at mAP@0.5 0.6115 - a drop of 14.46 pp against the pick.

**Pareto front** (nothing is both faster and more accurate):

- `yolo26n-onnx-fp32-256` - p50 5.20 ms, mAP@0.5 0.6115, mAP@0.5:0.95 0.3129, 9.22 MB
- `yolo26n-onnx-fp32-320` - p50 6.77 ms, mAP@0.5 0.6986, mAP@0.5:0.95 0.3888, 9.23 MB
- `yolo26n-onnx-fp32-640` - p50 21.88 ms, mAP@0.5 0.7332, mAP@0.5:0.95 0.4184, 9.35 MB
- `yolov8n-onnx-fp32-640` - p50 27.73 ms, mAP@0.5 0.7561, mAP@0.5:0.95 0.4233, 11.70 MB

**INT8 on yolo26n:** p50 21.88 -> 50.67 ms, i.e. **2.32x SLOWER**; size 9.35 -> 2.80 MB (3.34x smaller); mAP@0.5 -1.08 pp, mAP@0.5:0.95 -1.17 pp.
**INT8 on yolov8n:** p50 27.73 -> 50.34 ms, i.e. **1.81x SLOWER**; size 11.70 -> 3.23 MB (3.62x smaller); mAP@0.5 -0.06 pp, mAP@0.5:0.95 +0.68 pp.

**Resolution:** going from imgsz 256 to 640 costs +16.69 ms p50 (4.21x) and buys +12.17 pp of mAP@0.5 on 200x200 native images. The intermediate settings are in the table above - the gain is not linear in pixels, so the middle of that range is where the interesting operating points are.

