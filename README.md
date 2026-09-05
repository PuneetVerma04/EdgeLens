# EdgeLens - Industrial Defect Detection System

[![tests](https://github.com/PuneetVerma04/EdgeLens/actions/workflows/tests.yml/badge.svg)](https://github.com/PuneetVerma04/EdgeLens/actions/workflows/tests.yml)

Training now lives in this repo, under [`training/`](training/). The former [DefectDetectionEdgeLens](https://github.com/PuneetVerma04/DefectDetectionEdgeLens) repo is **archived** — its contents were moved to [`training/legacy/`](training/legacy/) as project history and are not the current pipeline.

## What problem this solves

Steel surface-defect inspection: given a photo of a rolled-steel surface, **locate and classify
defects** into six NEU-DET classes (crazing, inclusion, patches, pitted_surface, rolled-in_scale,
scratches). The repo covers the whole path — dataset preparation, controlled training, evaluation,
ONNX export with verification, and a CPU edge benchmark — plus a FastAPI service and Streamlit UI
for serving.

> **Migration in progress.** The detection pipeline is trained, evaluated, exported and
> benchmarked. The **serving layer has not been switched over yet**: `/predict` still runs the
> earlier binary casting classifier (ResNet50, OK/Defective). That rewrite is tracked in
> `NEU_DET_YOLO_Plan.md` and touches `backend/app/core/model.py` and
> `backend/app/utils/postprocess.py`. Until it lands, the mAP figures on this page describe the
> **detector**, not the running API. See [Limitations](#limitations-read-before-relying-on-this).

## What it is technically

**Detection pipeline (current work)**

- **`training/`** — `prepare_neu_det.py` converts PASCAL VOC to YOLO with a seeded, stratified
  70/15/15 split; `train_yolo.py` pins every hyperparameter so two runs differ only by the model
  weights; `eval_yolo.py` produces mAP, per-class AP, a confusion matrix and annotated worst-case
  false positives/negatives.
- **`tools/export_models.py`** — ONNX FP32 and INT8 export. INT8 calibrates on the **training**
  split only, and every artefact is verified against the PyTorch model before it is trusted.
- **`benchmarks/`** — CPU latency, memory, size and mAP per artefact, one subprocess per variant.
  See [the benchmark report](benchmarks/results_aggregate.md).
- **Models**: `yolov8n` and `yolo26n` (Ultralytics 8.4.126), 640×640 input, six classes.

**Serving layer (still the legacy classifier)**

- **Backend** (`backend/`) — FastAPI service. Loads a PyTorch checkpoint once at startup, exposes
  `/predict`, `/history`, and a retraining-simulation endpoint (see Limitations). Inference results
  are logged to MongoDB asynchronously via `BackgroundTasks`, non-blocking.
- **Frontend** (`frontend/`) — a single-file Streamlit app: upload an image, see the prediction,
  confidence, and inference time, plus a history dashboard.
- **Model currently served**: ResNet50 (torchvision) with the final FC layer replaced for 2-class
  output (OK / Defective). 224×224 RGB input, ImageNet normalization, CPU/GPU auto-detection.
  **This is the pre-migration model**, not the NEU-DET detector.

## Results / metrics

> **Detector results, not API results** — see the migration note at the top. `/predict` still
> serves the 2-class casting ResNet50.

Two nano detectors were trained to compare a 2023 architecture against a 2026 edge-optimised one.

**Everything below is the held-out `test` split** (270 images / 627 boxes), never used for training
or model selection. Both models were trained with **identical** hyperparameters — 100 epochs,
batch 16, imgsz 640, seed 0, patience 50, same augmentation — so the only variable is the
architecture. Both were then evaluated at **identical thresholds**: `--nms-iou 0.7`,
`--val-conf 0.001`, `imgsz 640`, `batch 16`, ultralytics 8.4.126. Per-run config is in
`runs/neu_det/<model>/edgelens_train_config.json`.

```bash
python training/eval_yolo.py --weights runs/neu_det/yolov8n/weights/best.pt   --data data/neu_det/data.yaml --split test --imgsz 640 --batch 16 --nms-iou 0.7
```

| Model | mAP@0.5 | mAP@0.5:0.95 | mean P | mean R |
|---|---:|---:|---:|---:|
| **yolov8n** | **0.7560** | **0.4234** | 0.7491 | 0.6928 |
| yolo26n | 0.7329 | 0.4180 | 0.6998 | 0.7031 |

**Read this as a tie, not a win.** YOLOv8n is nominally ahead by 2.3 mAP@0.5 points and 0.5
mAP@0.5:0.95 points. On a 270-image test split with a single seed, a gap that small is at or below
the level where the difference is meaningful — no repeated-seed significance testing was done, and
none of these numbers should be presented as one architecture beating the other.

That near-tie is itself the finding: **three years of YOLO architecture iteration bought
essentially nothing on this task at this scale.** This echoes the pattern in published NEU-DET
results, where YOLOv8l (74.1%) barely edges YOLOv8n (73.6%) despite being far larger.

One genuine, measured difference: **yolo26n's metrics are completely invariant to `--nms-iou`** —
re-evaluating it at 0.7 and 0.9 produces bit-identical mAP, precision and recall, because its head
is NMS-free. YOLOv8n is not: at `--nms-iou 0.9` its mAP@0.5 drops from 0.7560 to 0.6575. For a
project targeting ONNX export and in-browser inference, removing NMS removes both a
post-processing tuning knob and the hardest part of the export graph. That is the more interesting
YOLO26 result here than its mAP.

### Per-class AP

| Class | yolov8n AP@0.5 | yolov8n AP@0.5:0.95 | yolo26n AP@0.5 | yolo26n AP@0.5:0.95 |
|---|---:|---:|---:|---:|
| crazing | 0.4575 | 0.2060 | 0.4624 | 0.1963 |
| inclusion | 0.8066 | 0.4366 | 0.8095 | 0.4421 |
| patches | 0.9172 | 0.5695 | 0.8977 | 0.5720 |
| pitted_surface | 0.7864 | 0.5161 | 0.7376 | 0.5076 |
| rolled-in_scale | 0.6383 | 0.2720 | 0.5878 | 0.2572 |
| scratches | 0.9301 | 0.5404 | 0.9021 | 0.5329 |

`crazing` is by far the hardest class for both models and `rolled-in_scale` second — both are
low-contrast and diffuse, with boundaries even human annotators box loosely. `patches` and
`scratches` are easiest. This ordering matches what is generally reported on NEU-DET. Per-class
differences between the two models are small and inconsistent in direction, which is what a tie
looks like.

The large mAP@0.5 → mAP@0.5:0.95 drop (0.76 → 0.42) is the most informative number here: the
models **find** defects but **localise them loosely**. Strict-IoU scoring punishes that, and on
this dataset the boundaries are genuinely ambiguous.

### Confusion matrix — yolov8n

Rows are predictions, columns ground truth, at the mAP confidence floor of **conf ≥ 0.001**. That
floor is deliberately near-zero so the mAP integral is correct, which is why the `background`
column is large — it counts every marginal low-confidence box, not a deployment false-positive
rate. Use the operating-point table below for that.

| pred \ true | crazing | inclusion | patches | pitted_surface | rolled-in_scale | scratches | background |
|---|---:|---:|---:|---:|---:|---:|---:|
| crazing | 106 | 1 | 4 | 2 | 0 | 1 | 7969 |
| inclusion | 0 | 143 | 4 | 2 | 2 | 3 | 3297 |
| patches | 0 | 0 | 118 | 2 | 0 | 0 | 2662 |
| pitted_surface | 0 | 3 | 1 | 56 | 0 | 0 | 1903 |
| rolled-in_scale | 0 | 4 | 3 | 1 | 89 | 0 | 4734 |
| scratches | 0 | 1 | 0 | 0 | 0 | 79 | 858 |
| background | 0 | 1 | 1 | 0 | 0 | 0 | 0 |

Class-versus-class confusion is almost nil — off-diagonal counts are single digits. The models
rarely mistake one defect type for another; nearly all error is detection error (miss, spurious
box, or loose box), not classification error. yolo26n's matrix is in
`runs/neu_det/yolo26n/eval_test_nms07/confusion_matrix.csv`.

### At a deployment threshold (conf ≥ 0.25, match IoU ≥ 0.5)

| | yolov8n | yolo26n |
|---|---:|---:|
| true positives | 462 | 442 |
| false positives | 234 | 173 |
| false negatives | 165 | 185 |
| precision | 0.6638 | 0.7187 |
| recall | **0.7368** | 0.7049 |
| FP from background | 92 | **53** |
| FP from duplicate boxes | 40 | **36** |

**This is where the two models actually differ, and they trade off in opposite directions.**
YOLOv8n catches more defects (recall 0.737 vs 0.705); yolo26n raises fewer false alarms
(precision 0.719 vs 0.664) and produces markedly fewer background and duplicate boxes — the
duplicate reduction being consistent with its NMS-free head.

For industrial QA these are not equivalent errors: a missed defect ships to a customer, a false
alarm costs an inspection. On that criterion YOLOv8n's higher recall is the more useful operating
point, and yolo26n's precision advantage matters less. Both still miss roughly 30% of defect
boxes, which is the honest headline for anyone considering this for real use.

For both models about half the false negatives are **localisation** failures rather than misses
(yolov8n: 80 missed entirely vs 84 boxed too loosely) — the defect was found but the box was not
tight enough to count at IoU 0.5.

Every figure above comes from `runs/neu_det/<model>/eval_test/` (`metrics.json`,
`per_class_ap.csv`, `confusion_matrix.csv`), with annotated worst-case false positives and false
negatives alongside them. The per-run write-up is in [`training/RESULTS.md`](training/RESULTS.md).

**Reference point:** published YOLOv8n baselines on NEU-DET land around 73–78% mAP@0.5. Both
models sit inside that band (75.6% and 73.3%), so neither result is anomalous.

## Edge benchmark (CPU)

Full report, including protocol, per-stage timing, memory, run-to-run spread and a GPU appendix:
**[`benchmarks/results_aggregate.md`](benchmarks/results_aggregate.md)**. Raw data in
`benchmarks/results_cpu{1,2,3}.json`.

Median of **three independent runs** on an idle AMD Ryzen 7 7840HS (8 physical cores, 8 threads
per variant), batch 1, 20 warmup + 200 timed runs each. Latency is the full `predict()` call —
preprocess, inference and postprocess.

| Variant | p50 (ms) | spread | Size (MB) | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| `yolo26n` ONNX FP32 | **21.88** | 0.5% | 9.35 | 0.7332 | 0.4184 |
| `yolov8n` ONNX FP32 | 27.83 | 0.6% | 11.70 | **0.7561** | **0.4233** |
| `yolov8n` PyTorch | 34.46 | 1.9% | 5.97 | 0.7561 | 0.4233 |
| `yolo26n` PyTorch | 35.60 | 4.6% | 5.15 | 0.7332 | 0.4184 |
| `yolov8n` ONNX INT8 | 50.54 | 0.6% | **3.23** | 0.7555 | 0.4301 |
| `yolo26n` ONNX INT8 | 50.75 | 1.0% | **2.80** | 0.7225 | 0.4067 |

Three findings worth the space:

- **INT8 is 1.8–2.3× *slower*, not faster.** It delivers the expected 3.3–3.6× size reduction, but
  inspecting the graph ONNX Runtime actually executes shows only **7 of 64** (yolov8n) and **12 of
  102** (yolo26n) convolutions became integer kernels. The rest dequantize to float, run as float
  `Conv`, and requantize — full FP32 compute plus several hundred conversion nodes. Ship INT8 only
  when storage or memory is the binding constraint, not latency.
- **Input resolution is the biggest CPU lever, and it is steeply non-linear.** For yolo26n ONNX
  FP32: 256 → 5.20 ms at 0.6115 mAP@0.5, 320 → 6.70 ms at 0.6986, 640 → 21.88 ms at 0.7332. The
  step 256→320 buys **8.71 pp for 1.50 ms**; 320→640 buys only **3.46 pp for 15.18 ms** — about
  **25× worse value per millisecond** (5.81 vs 0.23 pp/ms).
- **That lever disappears on GPU.** On an RTX 4050 the same sweep spans 1.10× instead of 2.61×,
  and the inference stage is *inversely* ordered (8.71 → 8.22 → 8.10 ms as pixels grow 6.25×) —
  it is measuring fixed per-call overhead, not compute. On GPU, run at 640 and take the accuracy.

Reproduce:

```bash
python tools/export_models.py                                   # exports + verifies parity
python benchmarks/run_benchmark.py --device cpu --tag cpu1       # repeat as cpu2, cpu3
python benchmarks/aggregate_runs.py --glob "benchmarks/results_cpu[123].json"
```

> The mAP here (batch 1) differs from the training-side table above (batch 16) in the fourth
> decimal — e.g. yolov8n 0.756088 vs 0.756010. Same weights, same split; the difference is batch
> composition inside the validator, not a discrepancy.

## Limitations (read before relying on this)

- **Model retraining is a simulation, not a real training pipeline.** `POST /api/edgelens/retrain`
  (see Model retraining below) just sleeps for 10 seconds and returns `"status": "processing"` — it
  does not load data, does not train, and writes no checkpoint. There is no real retraining code in
  this repo.
- **No input sanitization beyond size and content-type checks.** The "Security" section below used
  to claim broader sanitization; only file size and content-type are validated.
- **The API does not serve the NEU-DET detector.** `/predict` still runs the binary casting
  ResNet50 and returns `{label, confidence, inference_time}` — no boxes. Everything under
  `training/`, `tools/` and `benchmarks/` operates on the detector directly, not through the API.
  Wiring the two together is outstanding work.
- **Detector accuracy is modest.** Both models miss roughly 30% of defect boxes at the deployment
  threshold. `crazing` in particular sits near 0.46 AP@0.5.
- **Benchmarked on one machine.** All latency figures come from a single x86 laptop. Rankings and
  mechanisms should transfer to a Pi or Jetson; absolute milliseconds will not. The harness runs
  unmodified on Linux/ARM if you need real numbers for a target device.

## Model weights and artefacts

No binary model files are committed. What each thing is, and how to get it:

| Artefact | Size | Needed for | How to obtain |
|---|---:|---|---|
| `backend/app/defect_detection_resnet_casting_data.pth` | ~94 MB | Starting the backend | [Release v0.1.0-weights](https://github.com/PuneetVerma04/EdgeLens/releases/tag/v0.1.0-weights) |
| `runs/neu_det/<model>/weights/best.pt` | ~5–6 MB each | Export, benchmark, eval | Retrain, or fetch from a release |
| `artifacts/onnx/*.onnx` | ~47 MB total | The CPU benchmark | `python tools/export_models.py` |
| `data/neu_det/` | ~25 MB | Training and evaluation | `python training/prepare_neu_det.py` |

Without the `.pth`, `load_model()` raises `FileNotFoundError` at startup. The gitignore keeps the
*metadata* that backs every published number — run configs, `results.csv`, eval outputs, export
manifests — so the figures on this page stay checkable from a clean clone even though the weights
themselves do not.

**Rebuilding the dataset:** use `--from-manifest` to reproduce the exact split the committed
checkpoints were trained on. Re-deriving from `--seed` alone will *not* reproduce it — the split
code changed, and a fresh seed-42 split moves 229 of 270 test images:

```bash
python training/prepare_neu_det.py --from-manifest data/neu_det/split_manifest.json
```

## Quick start

### Docker Compose (recommended)

```bash
cp .env.example .env
docker-compose up --build
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000  (docs at /docs)
```

### Local development

```bash
# Backend
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env   # edit as needed
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (separate terminal)
cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Requires the model weights in place first — see [Model weights](#model-weights-and-artefacts) above.

## Features

### Backend (FastAPI)

- ✅ **Health Check**: `GET /api/edgelens/`
- 🔍 **Defect Detection**: `POST /api/edgelens/predict`
- 📜 **Prediction History**: `GET /api/edgelens/history`
- 🔄 **Model Retraining (simulation stub)**: `POST /api/edgelens/retrain` — does not actually
  retrain; see [Limitations](#limitations-read-before-relying-on-this)

### Frontend (Streamlit)

- 🖼️ Real-time image upload and prediction
- 📊 Prediction history dashboard with statistics
- 🎨 Color-coded results (Green: OK, Red: Defective)
- ⚙️ Live backend health monitoring
- 📈 Confidence scores and inference timing

## Testing

### Automated (pytest)

Runs on every push and pull request via [`.github/workflows/tests.yml`](.github/workflows/tests.yml).
No model weights and no MongoDB are needed — the fixtures stub both out, so the suite works on a
fresh clone.

```bash
pip install -r backend/requirements.txt -r tests/requirements.txt
pytest
```

Covers preprocessing output shape, the predict happy path against a `test_samples/` image,
oversized-file and wrong-content-type rejection, the health endpoint, and `Settings` loading.

### Manual (against a running service)

```bash
# Health check
curl http://localhost:8000/api/edgelens/

# Predict defects using sample images
curl -X POST "http://localhost:8000/api/edgelens/predict" -F "file=@test_samples/cast_ok_0_119.jpeg"
curl -X POST "http://localhost:8000/api/edgelens/predict" -F "file=@test_samples/cast_def_0_127.jpeg"

# Get prediction history (last 10)
curl http://localhost:8000/api/edgelens/history

# Trigger retraining simulation (see Limitations — this does not really retrain)
curl -X POST "http://localhost:8000/api/edgelens/retrain"
```

```bash
# Validate the model loads and runs on the sample images.
# Resolves its own paths, so it can be run from any working directory, and picks CUDA or
# CPU automatically. Requires the weights — see Model weights above.
python scripts/validate_model.py
```

## Architecture

```
Frontend (Streamlit) ←→ Backend API (FastAPI)
                           ↓
                    ┌──────────────────┐
                    │   API Layer      │  (api/) — HTTP only
                    └──────────────────┘
                           ↓
                    ┌──────────────────┐
                    │  Business Logic  │  (core/, utils/)
                    └──────────────────┘
                           ↓
                    ┌──────────────────┐
                    │  Data Layer      │  (database/)
                    └──────────────────┘
                           ↓
                       MongoDB
```

- **API Layer**: HTTP request handling and response formatting
- **Core Layer**: Model loading, configuration, and schemas
- **Utils Layer**: Image preprocessing and output postprocessing
- **Database Layer**: Async MongoDB operations for logging

## Configuration

Copy `.env.example` to `.env` and adjust. See that file for the full list of variables (MongoDB
connection, file size limit, CORS origins, API host/port, model path).

## Project structure

```
EdgeLens/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── defect_detection_resnet_casting_data.pth  # NOT in repo — see Model weights
│   │   ├── api/            # predict.py, history.py, retrain.py
│   │   ├── core/           # model.py, config.py, schemas.py
│   │   ├── utils/          # preprocess.py, postprocess.py
│   │   └── database/       # db.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                # Streamlit app
│   ├── streamlit_app.py
│   ├── Dockerfile
│   └── requirements.txt
├── training/                # ML pipeline (NEU-DET detection) — the current work
│   ├── prepare_neu_det.py   # PASCAL VOC -> YOLO conversion + stratified 70/15/15 split
│   ├── train_yolo.py        # training entrypoint; all hyperparameters explicit
│   ├── eval_yolo.py         # mAP, per-class AP, confusion matrix, FP/FN error analysis
│   ├── RESULTS.md           # per-run metrics write-up
│   ├── requirements.txt
│   └── legacy/              # ARCHIVED casting classifier — known-bad metrics, see its README
├── tools/
│   └── export_models.py     # ONNX FP32/INT8 export + PyTorch parity verification
├── benchmarks/
│   ├── run_benchmark.py     # latency / memory / size / mAP, one subprocess per variant
│   ├── gpu_context.py       # CPU-vs-GPU appendix measurement
│   ├── aggregate_runs.py    # medians + spread across runs -> the report
│   ├── write_results.py     # shared markdown renderer
│   └── results_aggregate.md # THE benchmark report
├── common/
│   └── metrics.py           # iou_matrix + percentile, shared by eval/export/benchmark
├── tests/                   # pytest: preprocessing, predict API, config
├── artifacts/onnx/          # exported graphs — gitignored, rebuild with export_models.py
├── runs/neu_det/            # run configs + eval metadata committed; weights gitignored
├── data/neu_det/            # generated dataset — gitignored, rebuild with prepare_neu_det.py
├── scripts/
│   └── validate_model.py
├── test_samples/
├── docs/
├── .github/workflows/       # CI: pytest on push and PR
├── docker-compose.yml
├── .env.example
└── README.md
```

## Dependencies

**Backend** (`backend/requirements.txt`): FastAPI, PyTorch, Motor (async MongoDB driver), Uvicorn.
**Frontend** (`frontend/requirements.txt`): Streamlit, Requests, Pandas, Pillow.
**Training / export / benchmark** (`training/requirements.txt`): Ultralytics **pinned to 8.4.126**,
NumPy, Pillow, PyYAML, matplotlib. ONNX work additionally needs `onnx`, `onnxruntime`, `onnxslim`.

Ultralytics is pinned exactly because its defaults are not stable across releases and several are
adaptive — an unpinned version silently changes the experiment. Install torch **and torchvision
together** from the CUDA index before this file; installing torchvision from PyPI will downgrade a
working CUDA torch to a CPU build without any error, and training then runs ~20× slower.
`train_yolo.py` refuses to start a CPU run when a GPU is present, so this fails loudly rather than
costing you an afternoon.

## Docker notes

- Backend Dockerfile uses a multi-stage build (builder installs/compiles, runtime stays minimal).
- Ports: Backend 8000, Frontend 8501, MongoDB 27017 (internal only).

```bash
docker-compose up -d --build   # build and start
docker-compose logs -f         # tail logs
docker-compose down            # stop
docker-compose down -v         # stop and wipe the MongoDB volume
```

## API documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## License

MIT — see [LICENSE](LICENSE).
