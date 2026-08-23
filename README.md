# EdgeLens - Industrial Defect Detection System

[![tests](https://github.com/PuneetVerma04/EdgeLens/actions/workflows/tests.yml/badge.svg)](https://github.com/PuneetVerma04/EdgeLens/actions/workflows/tests.yml)

Training now lives in this repo, under [`training/`](training/). The former [DefectDetectionEdgeLens](https://github.com/PuneetVerma04/DefectDetectionEdgeLens) repo is **archived** — its contents were moved to [`training/legacy/`](training/legacy/) as project history and are not the current pipeline.

## What problem this solves

Casting-defect inspection: given a photo of a cast part, classify it as OK or Defective. This repo
is the **serving** side — a FastAPI microservice that loads a trained ResNet50 checkpoint and
exposes it over HTTP, plus a Streamlit UI to exercise it and a MongoDB log of past predictions.

## What it is technically

- **Backend** (`backend/`) — FastAPI service. Loads a PyTorch ResNet50 checkpoint once at startup,
  exposes `/predict`, `/history`, and a retraining-simulation endpoint (see Limitations). Inference
  results are logged to MongoDB asynchronously via `BackgroundTasks`, non-blocking.
- **Frontend** (`frontend/`) — a single-file Streamlit app: upload an image, see the prediction,
  confidence, and inference time, plus a history dashboard.
- **Model**: ResNet50 (torchvision) with the final FC layer replaced for 2-class output (OK /
  Defective). 224×224 RGB input, ImageNet normalization, CPU/GPU auto-detection.

## Results / metrics

> **These are training-side results. They are not what the API currently serves.**
> `backend/` still loads the legacy 2-class casting ResNet50 (`backend/app/core/model.py` builds
> `resnet50` with a 2-output head; `postprocess.py` maps to `OK` / `Defective`). The NEU-DET
> detector below is trained and evaluated but **not yet wired into `/predict`** — that rewrite is
> outstanding. Do not read the mAP figures here as the accuracy of the running service.

The live pipeline is **6-class surface-defect detection on NEU-DET**, not the earlier binary
casting classifier. Two nano detectors were trained to compare a 2023 architecture against a 2026
edge-optimised one.

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

**Not benchmarked yet:** ONNX export, INT8 quantisation, CPU latency, and the imgsz sweep are
planned but unmeasured — no numbers for them are claimed here.

## Limitations (read before relying on this)

- **Model retraining is a simulation, not a real training pipeline.** `POST /api/edgelens/retrain`
  (see Model retraining below) just sleeps for 10 seconds and returns `"status": "processing"` — it
  does not load data, does not train, and writes no checkpoint. There is no real retraining code in
  this repo.
- **No input sanitization beyond size and content-type checks.** The "Security" section below used
  to claim broader sanitization; only file size and content-type are validated.

## Model weights

**The trained weights (`defect_detection_resnet_casting_data.pth`, ~94 MB) are not in this repo.**
They're gitignored (`backend/app/*.pth`) and must be obtained separately before the backend can
start — without them, `load_model()` raises `FileNotFoundError` at startup.

Model weights available at ([Model Weights](https://github.com/PuneetVerma04/EdgeLens/releases/tag/v0.1.0-weights)) and it at `backend/app/defect_detection_resnet_casting_data.pth`.

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

Requires the model weights in place first — see [Model weights](#model-weights) above.

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
├── runs/neu_det/            # training + eval outputs, weights, configs (not a clean-clone dep)
├── data/neu_det/            # generated dataset — gitignored, rebuild with prepare_neu_det.py
├── scripts/
│   └── validate_model.py
├── test_samples/
├── docker-compose.yml
├── .env.example
└── README.md
```

## Dependencies

**Backend**: FastAPI, PyTorch, Motor (async MongoDB driver), Uvicorn.
**Frontend**: Streamlit, Requests, Pandas, Pillow.

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
