# EdgeLens - Industrial Defect Detection System

Training repo: [DefectDetectionEdgeLens](https://github.com/PuneetVerma04/DefectDetectionEdgeLens) (model training/notebooks — this repo serves the resulting model)

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

No accuracy figures live in this repo — training, dataset, and evaluation are in the separate
[DefectDetectionEdgeLens](https://github.com/PuneetVerma04/DefectDetectionEdgeLens) repo. See that
repo's README for the current state of the reported numbers (including known validation-leakage
caveats there). TODO: link the specific run/commit that produced the weights currently used here,
once that's tracked.

## Limitations (read before relying on this)

- **Model retraining is a simulation, not a real training pipeline.** `POST /api/edgelens/retrain`
  (see Model retraining below) just sleeps for 10 seconds and returns `"status": "processing"` — it
  does not load data, does not train, and writes no checkpoint. There is no real retraining code in
  this repo.
- **No input sanitization beyond size and content-type checks.** The "Security" section below used
  to claim broader sanitization; only file size and content-type are validated.
- **`CORS_ORIGINS` bypasses the app's config layer.** Every other setting in `.env` is read through
  `pydantic-settings` (`backend/app/core/config.py`); `CORS_ORIGINS` is instead read directly via
  `os.getenv(...)` in `main.py` and is not a field on `Settings`. Functionally it still works, but
  it won't show up if you inspect `Settings` in code, and won't get validated the way other options
  do.

## Model weights

**The trained weights (`defect_detection_resnet_casting_data.pth`, ~94 MB) are not in this repo.**
They're gitignored (`backend/app/*.pth`) and must be obtained separately before the backend can
start — without them, `load_model()` raises `FileNotFoundError` at startup.

TODO: publish a download URL (e.g. a GitHub Release asset or object storage link) for the trained
weights. Until then, train your own via the notebooks in
[DefectDetectionEdgeLens](https://github.com/PuneetVerma04/DefectDetectionEdgeLens) and place the
resulting `.pth` file at `backend/app/defect_detection_resnet_casting_data.pth`.

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
# Validate the model loads and runs on the sample images (run from repo root)
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
