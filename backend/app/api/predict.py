from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
import torch
import time
from app.utils.preprocess import preprocess_image
from app.utils.postprocess import postprocess_output
from app.database.db import log_inference
from app.core.schemas import PredictionResponse, HealthCheckResponse
from app.core.config import get_settings

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    max_file_size = get_settings().max_file_size_mb * 1024 * 1024

    # Check file size manually since UploadFile does not have spool_max_size
    file_size = len(await file.read())
    await file.seek(0)  # Reset file pointer after reading
    if file_size > max_file_size:
        raise HTTPException(413, f"File too large. Max size is {get_settings().max_file_size_mb} MB.")
    
    model = request.app.state.model
    if not file.content_type or file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image."
        )
    
    # Read file bytes
    images_bytes = await file.read()

    # Preprocess
    input_tensor = preprocess_image(images_bytes)

    # Inference
    start_time = time.perf_counter()
    with torch.no_grad():
        prediction = model(input_tensor)
    if prediction.ndim != 2 or prediction.shape[0] != 1:
        raise RuntimeError("Unexpected model output shape.")
    inference_time = time.perf_counter() - start_time

    # Postprocess
    result = postprocess_output(prediction)
    result["inference_time"] = inference_time

    background_tasks.add_task(log_inference, result)

    return PredictionResponse(
        label=result["label"],
        confidence=result["confidence"],
        inference_time=inference_time
    )

@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service status
    """
    return HealthCheckResponse(
        status="ok",
        message="EdgeLens Inference Service is running."
    )
