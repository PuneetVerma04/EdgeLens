from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from app.api.predict import router as predict_router
from app.api.history import router as history_router
from app.api.retrain import router as retrain_router
from app.core.config import get_settings
from app.core.model import load_model
from app.database.db import connect_db, close_db

# Global variables (created once)
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    # Startup: Load model and connect to DB
    model = load_model()
    await connect_db()

    # Make model available to all routes
    app.state.model = model

    yield
    # Shutdown: Close DB connection
    await close_db()


app = FastAPI(title="EdgeLens Inference Service", lifespan=lifespan)

# Configure CORS - Allow Streamlit Cloud and other frontend origins.
# Read through pydantic-settings like every other setting, so CORS_ORIGINS is a real field
# on Settings and is validated and inspectable alongside the rest of the configuration.
allowed_origins = get_settings().cors_origins_list

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Streamlit Cloud domain will be added via env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict_router, prefix="/api/edgelens", tags=["Prediction"])
app.include_router(history_router, prefix="/api/edgelens", tags=["History"])
app.include_router(retrain_router, prefix="/api/edgelens", tags=["Retraining"])

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
