# Edge benchmark (CPU) - NEU-DET surface defect detection

**Aggregated over 3 independent runs** (`results_cpu1.json`, `results_cpu2.json`, `results_cpu3.json`). Every latency and memory figure below is the **median across those runs**; the run-to-run spread is reported directly under the results table. Accuracy is deterministic and identical in every run. Raw data in [`results_aggregate.json`](results_aggregate.json).

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
| `yolov8n-pytorch-640` | PyTorch | FP32 | 640 | 34.46 | 37.04 | 714.6 | 5.97 | 6,258,410 | 0.7561 | 0.4233 |
| `yolo26n-pytorch-640` | PyTorch | FP32 | 640 | 35.60 | 39.14 | 723.9 | 5.15 | 5,397,573 | 0.7332 | 0.4184 |
| `yolov8n-onnx-fp32-640` | ONNX Runtime | FP32 | 640 | 27.83 | 28.66 | 824.5 | 11.70 | 12,269,259 | 0.7561 | 0.4233 |
| `yolov8n-onnx-int8-640` | ONNX Runtime | INT8 | 640 | 50.54 | 51.86 | 833.3 | 3.23 | 3,389,567 | 0.7555 | 0.4301 |
| `yolo26n-onnx-fp32-640` | ONNX Runtime | FP32 | 640 | 21.88 | 22.45 | 871.4 | 9.35 | 9,809,346 | 0.7332 | 0.4184 |
| `yolo26n-onnx-int8-640` | ONNX Runtime | INT8 | 640 | 50.75 | 52.18 | 885.4 | 2.80 | 2,938,750 | 0.7225 | 0.4067 |

Accuracy was **identical across all 3 runs** for every variant, as expected - the same weights over the same images is a deterministic computation. Only latency and memory vary between runs.

### Run-to-run spread

Every latency figure in this report is the median of 3 runs. This table shows how far the individual runs sat apart, which is the honest precision of any number quoted above.

| Variant | p50 median | p50 observed range | spread | p95 median | p95 spread |
| --- | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 34.46 | 33.82-34.48 | 1.9% | 37.04 | 3.9% |
| `yolo26n-pytorch-640` | 35.60 | 35.11-36.75 | 4.6% | 39.14 | 5.8% |
| `yolov8n-onnx-fp32-640` | 27.83 | 27.73-27.90 | 0.6% | 28.66 | 0.7% |
| `yolov8n-onnx-int8-640` | 50.54 | 50.34-50.63 | 0.6% | 51.86 | 1.4% |
| `yolo26n-onnx-fp32-640` | 21.88 | 21.83-21.94 | 0.5% | 22.45 | 0.9% |
| `yolo26n-onnx-int8-640` | 50.75 | 50.67-51.16 | 1.0% | 52.18 | 1.8% |
| `yolo26n-onnx-fp32-256` | 5.20 | 5.20-5.25 | 1.1% | 5.59 | 3.0% |
| `yolo26n-onnx-fp32-320` | 6.70 | 6.67-6.77 | 1.5% | 7.12 | 6.2% |

Worst observed spread: **4.6%** on p50 (`yolo26n-pytorch-640`) and **6.2%** on p95 (`yolo26n-onnx-fp32-320`). Anything quoted to tighter precision than that is reading noise. Note that a benchmark run alongside other heavy work on the same machine is far less reproducible than this - these runs were taken on an idle host.

### Ratios between variants

Each ratio is computed **within** a run and then median-ed across runs, so both measurements share the same thermal and load conditions. Ratios are consequently tighter than either endpoint and are the safest form in which to quote a speed comparison.

| Variant | p50 median | x slower than `yolo26n-onnx-fp32-640` |
| --- | ---: | ---: |
| `yolo26n-onnx-fp32-640` | 21.88 ms | 1.00x (range 1.00-1.00) |
| `yolov8n-onnx-fp32-640` | 27.83 ms | 1.27x (range 1.27-1.28) |
| `yolov8n-pytorch-640` | 34.46 ms | 1.57x (range 1.55-1.58) |
| `yolo26n-pytorch-640` | 35.60 ms | 1.63x (range 1.61-1.67) |
| `yolov8n-onnx-int8-640` | 50.54 ms | 2.31x (range 2.30-2.31) |
| `yolo26n-onnx-int8-640` | 50.75 ms | 2.32x (range 2.32-2.33) |

### Latency detail

| Variant | min | p50 | p90 | p95 | p99 | max | mean | stdev | FPS @ p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 26.94 | 34.46 | 36.51 | 37.04 | 38.33 | 39.21 | 34.51 | 1.90 | 29.0 |
| `yolo26n-pytorch-640` | 29.74 | 35.60 | 38.11 | 39.14 | 40.89 | 43.42 | 35.73 | 2.04 | 28.1 |
| `yolov8n-onnx-fp32-640` | 27.06 | 27.83 | 28.43 | 28.66 | 29.71 | 31.35 | 27.94 | 0.51 | 35.9 |
| `yolov8n-onnx-int8-640` | 49.28 | 50.54 | 51.28 | 51.86 | 53.36 | 54.96 | 50.61 | 0.83 | 19.8 |
| `yolo26n-onnx-fp32-640` | 21.43 | 21.88 | 22.31 | 22.45 | 22.79 | 23.21 | 21.93 | 0.29 | 45.7 |
| `yolo26n-onnx-int8-640` | 49.56 | 50.75 | 51.73 | 52.18 | 53.31 | 55.83 | 50.90 | 0.75 | 19.7 |

All times in milliseconds. Where each millisecond goes:

| Variant | preprocess | inference | postprocess |
| --- | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 2.11 | 31.35 | 0.46 |
| `yolo26n-pytorch-640` | 2.02 | 33.01 | 0.19 |
| `yolov8n-onnx-fp32-640` | 2.65 | 24.38 | 0.64 |
| `yolov8n-onnx-int8-640` | 2.53 | 47.20 | 0.65 |
| `yolo26n-onnx-fp32-640` | 2.67 | 18.56 | 0.28 |
| `yolo26n-onnx-int8-640` | 2.72 | 47.45 | 0.28 |

### Memory detail

| Variant | Baseline RSS | After model load | Model delta | Peak working set |
| --- | ---: | ---: | ---: | ---: |
| `yolov8n-pytorch-640` | 502.6 | 533.3 | +30.7 | 714.6 |
| `yolo26n-pytorch-640` | 502.7 | 532.1 | +29.4 | 723.9 |
| `yolov8n-onnx-fp32-640` | 502.4 | 513.4 | +10.9 | 824.5 |
| `yolov8n-onnx-int8-640` | 502.8 | 512.9 | +10.4 | 833.3 |
| `yolo26n-onnx-fp32-640` | 502.8 | 513.3 | +10.5 | 871.4 |
| `yolo26n-onnx-int8-640` | 502.7 | 513.1 | +10.7 | 885.4 |

All values MiB. Baseline RSS is the process after imports but before the model loads; it is not attributable to the artefact. The peak column is what a deployment must actually provision for.

### ONNX artefacts vs their PyTorch reference

| Artefact | p50 speedup | p95 speedup | Size vs .pt | mAP@0.5 delta | mAP@0.5:0.95 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-onnx-fp32-640` | 1.24x | 1.29x | 1.96x | 0.00 pp | 0.00 pp |
| `yolov8n-onnx-int8-640` | 0.68x | 0.71x | 0.54x | -0.06 pp | +0.68 pp |
| `yolo26n-onnx-fp32-640` | 1.63x | 1.74x | 1.82x | +0.0001 pp | +0.0001 pp |
| `yolo26n-onnx-int8-640` | 0.70x | 0.75x | 0.54x | -1.08 pp | -1.17 pp |

## Resolution trade-off (yolo26n, ONNX FP32)

NEU-DET images are natively **200x200**, so every setting here upscales. The question is whether paying for that upscale buys accuracy.

| imgsz | Upscale from 200px | p50 (ms) | p95 (ms) | Peak mem (MiB) | mAP@0.5 | mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 1.28x | 5.20 | 5.59 | 729.3 | 0.6115 | 0.3129 |
| 320 | 1.60x | 6.70 | 7.12 | 745.3 | 0.6986 | 0.3888 |
| 640 | 3.20x | 21.88 | 22.45 | 871.4 | 0.7332 | 0.4184 |

Relative to the smallest setting:

| imgsz | p50 cost | mAP@0.5 gain | mAP@0.5:0.95 gain | mAP@0.5 per extra ms |
| ---: | ---: | ---: | ---: | ---: |
| 256 | +0.00 ms | 0.00 pp | 0.00 pp | - |
| 320 | +1.50 ms | +8.71 pp | +7.59 pp | +5.822 pp/ms |
| 640 | +16.68 ms | +12.17 pp | +10.55 pp | +0.730 pp/ms |

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
| `yolov8n-onnx-int8-640` | 7 / 64 (10.9%) | 57 | 437 | 24.38 -> 47.20 ms |
| `yolo26n-onnx-int8-640` | 12 / 102 (11.8%) | 90 | 684 | 18.56 -> 47.45 ms |

Almost none of the convolutions became integer kernels. The weights really are stored as INT8 - that is where the file-size win comes from - but at runtime the great majority of convs dequantize back to float, run as float `Conv`, and requantize. The model therefore pays the full FP32 compute cost **plus** several hundred conversion ops it did not have before.

The likely cause is the node exclusion list. Ultralytics quantizes only Conv/Gemm/MatMul and excludes every other node by name (`nodes_to_exclude`), which is a deliberate and correct choice for accuracy - one INT8 scale cannot span box coordinates and class probabilities. But in a YOLO backbone the convolutions are separated by SiLU (`Sigmoid` + `Mul`), `Concat` and `Add`, so excluding those breaks the contiguous quantized regions ONNX Runtime needs in order to fuse `DequantizeLinear -> Conv -> QuantizeLinear` into a single `QLinearConv`. Both models converge to ~48 ms inference regardless of architecture, which is consistent with the conversion machinery dominating rather than the model.

**Practical consequence:** on this CPU and runtime, ship INT8 only when the binding constraint is storage or memory footprint. If the constraint is latency, INT8 as exported here is the wrong tool - lowering the input resolution buys far more (see the resolution table above). Worth revisiting with ONNX Runtime's `quant_pre_process` step, or a per-op exclusion list that keeps quantized regions contiguous, before concluding INT8 cannot help on this hardware.

## Which operating point to ship

**Rule used:** among every variant whose mAP@0.5 is within 1.0 pp of the best measured, take the lowest p50 latency. Best measured mAP@0.5 is **0.7561** (`yolov8n-pytorch-640`), so the eligibility threshold is 0.7461 and 3 of 6 variants qualify.

**The pick is `yolov8n-onnx-fp32-640`** - p50 **27.83 ms**, p95 28.66 ms, mAP@0.5 **0.7561**, mAP@0.5:0.95 0.4233, 11.70 MB on disk, 824.5 MiB peak.

Against the most accurate variant (`yolov8n-pytorch-640`, mAP@0.5 0.7561, p50 34.46 ms), the pick gives up **0.00 pp** of mAP@0.5 and 0.00 pp of mAP@0.5:0.95 to run **1.24x faster** (34.46 ms -> 27.83 ms, a saving of 6.63 ms per frame).

**If latency is the binding constraint instead**, the frontier's fastest point is `yolo26n-onnx-fp32-256` at **5.20 ms** p50 (5.35x quicker than the pick, 22.62 ms saved per frame), at mAP@0.5 0.6115 - a drop of 14.46 pp against the pick.

**Pareto front** (nothing is both faster and more accurate):

- `yolo26n-onnx-fp32-256` - p50 5.20 ms, mAP@0.5 0.6115, mAP@0.5:0.95 0.3129, 9.22 MB
- `yolo26n-onnx-fp32-320` - p50 6.70 ms, mAP@0.5 0.6986, mAP@0.5:0.95 0.3888, 9.23 MB
- `yolo26n-onnx-fp32-640` - p50 21.88 ms, mAP@0.5 0.7332, mAP@0.5:0.95 0.4184, 9.35 MB
- `yolov8n-onnx-fp32-640` - p50 27.83 ms, mAP@0.5 0.7561, mAP@0.5:0.95 0.4233, 11.70 MB

**INT8 on yolo26n:** p50 21.88 -> 50.75 ms, i.e. **2.32x SLOWER**; size 9.35 -> 2.80 MB (3.34x smaller); mAP@0.5 -1.08 pp, mAP@0.5:0.95 -1.17 pp.
**INT8 on yolov8n:** p50 27.83 -> 50.54 ms, i.e. **1.82x SLOWER**; size 11.70 -> 3.23 MB (3.62x smaller); mAP@0.5 -0.06 pp, mAP@0.5:0.95 +0.68 pp.

**Resolution:** going from imgsz 256 to 640 costs +16.68 ms p50 (4.20x) and buys +12.17 pp of mAP@0.5 on 200x200 native images. The intermediate settings are in the table above - the gain is not linear in pixels, so the middle of that range is where the interesting operating points are.

## Appendix: what a GPU changes

Measured 2026-08-30T10:31:08Z on **NVIDIA GeForce RTX 4050 Laptop GPU** (6.0 GiB), PyTorch (both devices), torch 2.13.0+cu130. 20 warmup + 200 timed runs. 3 interleaved rounds per device in one process, medians reported; torch.cuda.synchronize() before stopping the clock. Raw data in [`gpu_context.json`](gpu_context.json).

This benchmark targets CPU because that is the edge deployment story. This appendix answers only two questions: how much a GPU actually buys for a nano detector, and whether the resolution finding above survives on one.

### Head to head at imgsz 640

| Model | CPU p50 | GPU p50 | End-to-end speedup | range | Inference-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n | 33.34 ms (±0.9%) | 8.07 ms (±1.8%) | **4.12x** | 4.12-4.17 | 5.65x |
| yolo26n | 35.03 ms (±4.4%) | 10.29 ms (±8.5%) | **3.40x** | 3.29-3.42 | 4.00x |

**3.40-4.12x, not the 10-20x a GPU usually suggests.** A nano YOLO at 640 is roughly 8 GFLOPs against this card's ~20 TFLOPS, so the arithmetic is sub-millisecond and what remains is kernel launch, host-device transfer and Python overhead. The CPU numbers in the tables above are not embarrassing by comparison because the model is too small to saturate a GPU.

### The resolution trade-off does not survive on GPU

| imgsz | CPU p50 | GPU p50 | GPU inference stage |
| ---: | ---: | ---: | ---: |
| 256 | 13.38 ms | 9.50 ms | 8.71 ms |
| 320 | 15.54 ms | 9.01 ms | 8.22 ms |
| 640 | 34.90 ms | 10.42 ms | 8.10 ms |

Across imgsz 256 to 640 the CPU spans **2.61x** but the GPU spans only **1.10x**. The GPU *inference stage* barely moves at all - 8.10 to 8.71 ms, a 7.6% change for **6.25x more pixels**.

In fact the inference stage is *inversely* ordered: 8.71 > 8.22 > 8.10 ms as the input grows 256 -> 640. More pixels cannot make convolutions finish sooner, so this is measurement noise around a roughly constant ~8.3 ms floor. The number being measured is not compute - it is fixed per-call overhead, and the model is small enough to hide entirely underneath it.

**Consequence:** the headline CPU conclusion - that input resolution is the biggest latency lever - is a CPU conclusion. On a GPU you should run at 640 and take the accuracy, because the smaller inputs cost almost nothing less. Anyone reading these results for a GPU target should start from that inversion.

### Reading this for Pi and Jetson

| Transfers to other hardware | Does not |
| --- | --- |
| Rankings between variants | Absolute milliseconds |
| The mechanisms (nano models are overhead-bound on GPU; ONNX INT8 leaves most convolutions in float) | Tail latency (p95/p99) |
| Every mAP figure - accuracy is hardware-independent | Anything from a TensorRT engine built on a different GPU |

Neither target device was measured here. An x86 Ryzen is not a Pi 5's ARM Cortex-A76, and a laptop RTX 4050 is not a Jetson Orin. Treat the CPU table as Pi-*shaped* and this appendix as Jetson-*indicative*; quote ratios and mechanisms rather than absolute times. Two specifics worth knowing:

- **TensorRT engines are not portable.** An engine built here (sm_89) will not load on a Jetson Orin (sm_87). Jetson figures must be built and measured on the Jetson.
- **The harness runs unmodified on both.** `run_benchmark.py` reads the CPU name from `/proc/cpuinfo` on Linux and falls back to `resource.getrusage` for peak memory where Windows' `peak_wset` does not exist. Running it on the device is the only way to get defensible numbers for it.

Each figure is the median of **3 interleaved rounds** per device, with the observed spread shown alongside. Speedups are computed within a round and then median-ed, so both halves of every ratio share one thermal state.

This appendix reports no mAP (accuracy is device-independent - see the main tables) and measures PyTorch on both devices rather than ONNX, because ONNX on GPU needs `onnxruntime-gpu` for the CUDA execution provider. The ratios are what it is for.

