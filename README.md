# EdgeLens - Industrial Defect Detection System

A ML microservice for industrial defect detection using PyTorch ResNet50, featuring a FastAPI backend, Streamlit frontend, and MongoDB logging.

## 📁 Project Structure

```
EdgeLens/
│
├── backend/                      # FastAPI Backend
│   ├── app/
│   │   ├── main.py              # Application entry point with lifespan events
│   │   ├── defect_detection_resnet_casting_data.pth  # Model weights
│   │   ├── api/                 # API route handlers
│   │   │   ├── predict.py       # Prediction endpoint
│   │   │   ├── history.py       # History endpoint
│   │   │   └── retrain.py       # Retraining endpoint
│   │   ├── core/                # Core business logic
│   │   │   ├── model.py         # Model loading & inference
│   │   │   ├── config.py        # Settings management
│   │   │   └── schemas.py       # Pydantic models
│   │   ├── utils/               # Utility functions
│   │   │   ├── preprocess.py    # Image preprocessing pipeline
│   │   │   └── postprocess.py   # Output postprocessing
│   │   └── database/            # Database layer
│   │       └── db.py            # Async MongoDB operations
│   ├── Dockerfile               # Multi-stage backend container
│   └── requirements.txt         # Backend dependencies
│
├── frontend/                     # Streamlit Frontend
│   ├── streamlit_app.py         # Interactive web UI
│   ├── Dockerfile               # Frontend container
│   └── requirements.txt         # Frontend dependencies
│
├── scripts/                      # Utility scripts
│   └── validate_model.py        # Model validation script
│
├── test_samples/                 # Sample test images
│   ├── cast_def_0_127.jpeg      # Defective sample
│   ├── cast_def_0_240.jpeg      # Defective sample
│   └── cast_ok_0_119.jpeg       # OK sample
│
├── docker-compose.yml           # Multi-container orchestration
├── .env                         # Environment configuration
└── README.md                    # This file
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Start all services (MongoDB, Backend, Frontend)
docker-compose up --build

# Access the application
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Local Development

#### Backend Setup

```bash
cd backend
pip install -r requirements.txt

# Set environment variables (optional)
export MONGODB_URL="mongodb://localhost:27017"
export ENVIRONMENT="development"

# Start the backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend Setup

```bash
cd frontend
pip install -r requirements.txt

# Start the Streamlit app
streamlit run streamlit_app.py
```

## 🎯 Features

### Backend (FastAPI)

- ✅ **Health Check**: `GET /api/edgelens/`
- 🔍 **Defect Detection**: `POST /api/edgelens/predict`
- 📜 **Prediction History**: `GET /api/edgelens/history`
- 🔄 **Model Retraining**: `POST /api/edgelens/trigger-retraining`

### Frontend (Streamlit)

- 🖼️ Real-time image upload and prediction
- 📊 Prediction history dashboard with statistics
- 🎨 Color-coded results (Green: OK, Red: Defective)
- ⚙️ Live backend health monitoring
- 📈 Confidence scores and inference timing

## 🏗️ Architecture

### Layered Architecture Pattern

```
Frontend (Streamlit) ←→ Backend API (FastAPI)
                           ↓
                    ┌──────────────────┐
                    │   API Layer      │
                    │  (routes/api/)   │
                    └──────────────────┘
                           ↓
                    ┌──────────────────┐
                    │  Business Logic  │
                    │   (core/utils)   │
                    └──────────────────┘
                           ↓
                    ┌──────────────────┐
                    │  Data Layer      │
                    │   (database/)    │
                    └──────────────────┘
                           ↓
                       MongoDB
```

### Key Components

- **API Layer**: HTTP request handling and response formatting
- **Core Layer**: Model loading, configuration, and schemas
- **Utils Layer**: Image preprocessing and output postprocessing
- **Database Layer**: Async MongoDB operations for logging

## 📊 Model Details

- **Architecture**: ResNet50 (modified final FC layer)
- **Classes**: 2 (Defective, OK)
- **Input Size**: 224x224 RGB images
- **Framework**: PyTorch (torchvision)
- **Weights**: Pre-trained on casting defect dataset
- **Inference**: CPU/GPU support with automatic device detection
- **Preprocessing**: Resize → Normalize (ImageNet stats) → Tensor

## 🔧 Configuration

Environment variables (set in `.env` file):

```bash
# MongoDB
MONGODB_URL=mongodb://mongodb:27017  # Use 'localhost' for local dev, 'mongodb' for Docker
MONGODB_DB_NAME=edgelens_db
MONGODB_COLLECTION_NAME=inference_logs

# Application
ENVIRONMENT=development
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=5

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:8501,http://localhost:3000

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Model
MODEL_PATH=./app/defect_detection_resnet_casting_data.pth
```

**Note**: When running with Docker Compose, the MongoDB URL should use the service name `mongodb`. For local development, use `localhost`.

## 🧪 Testing

### Test the Backend API

```bash
# Health check
curl http://localhost:8000/api/edgelens/

# Predict defects using sample images
curl -X POST "http://localhost:8000/api/edgelens/predict" \
     -F "file=@test_samples/cast_ok_0_119.jpeg"

curl -X POST "http://localhost:8000/api/edgelens/predict" \
     -F "file=@test_samples/cast_def_0_127.jpeg"

# Get prediction history (last 10)
curl http://localhost:8000/api/edgelens/history

# Trigger retraining simulation
curl -X POST "http://localhost:8000/api/edgelens/trigger-retraining"
```

### Test the Frontend

1. Open http://localhost:8501 in your browser
2. Upload a test image from `test_samples/` directory
3. Click "Analyze for Defects"
4. View real-time results with confidence scores
5. Check prediction history dashboard

### Validate Model Locally

```bash
python scripts/validate_model.py
```

## 📦 Dependencies

### Backend

- FastAPI: Web framework
- PyTorch: ML inference
- Motor: Async MongoDB driver
- Uvicorn: ASGI server

### Frontend

- Streamlit: Web UI framework
- Requests: HTTP client
- Pandas: Data manipulation
- Pillow: Image processing

## 🔐 Security

- File size limits (5MB)
- Content type validation
- Error handling and sanitization
- Environment-based configuration

## 📈 Performance

- Async database logging (non-blocking)
- Request/response timing middleware
- Model loaded once at startup
- GPU acceleration support

## 🛠️ Development

### Adding New Endpoints

1. Create route handler in `backend/app/api/`
2. Register router in `backend/app/main.py` with appropriate prefix/tags
3. Add Pydantic schemas to `backend/app/core/schemas.py`
4. Update tests and documentation

### Code Organization Principles

- **API Layer** (`api/`): Handle HTTP concerns only - validation, response formatting
- **Core Layer** (`core/`): Model loading, configuration, schemas
- **Utils Layer** (`utils/`): Shared preprocessing/postprocessing functions
- **Database Layer** (`database/`): Data persistence and async logging
- **Never mix concerns**: Routes shouldn't contain ML code or direct DB calls

### Local Development Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Run backend
cd backend
uvicorn app.main:app --reload

# Run frontend (in separate terminal)
cd frontend
streamlit run streamlit_app.py
```

## 🐳 Docker Notes

### Multi-Stage Build

The backend Dockerfile uses multi-stage builds:

1. **Builder stage**: Install gcc, compile dependencies
2. **Runtime stage**: Copy compiled wheels, minimal footprint

### Container Orchestration

- **Backend**: Port 8000 (FastAPI + Uvicorn)
- **Frontend**: Port 8501 (Streamlit)
- **MongoDB**: Port 27017 (Internal, not exposed)

## 🚀 Deployment

### Docker Compose Production

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Remove volumes (clears database)
docker-compose down -v
```

## 📝 API Documentation

Interactive API documentation available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License

## 👥 Author

EdgeLens Development Team

---

**Built with ❤️ using FastAPI, PyTorch, and Streamlit**
