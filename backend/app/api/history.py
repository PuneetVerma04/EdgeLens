from fastapi import APIRouter, HTTPException
from app.core.schemas import PredictionLogResponse
from app.database import db

router = APIRouter()

@router.get("/history", response_model=list[PredictionLogResponse])
async def get_prediction_history():
    """
    Get the last 10 prediction logs from MongoDB.
    
    Returns:
        List of prediction logs with timestamp, label, and confidence
    """
    if db.collection is None:
        raise HTTPException(503, "Database not available.")
    
    cursor = db.collection.find().sort("timestamp", -1).limit(10)
    logs = await cursor.to_list(length=10)

    return [
        PredictionLogResponse(
            id=idx,
            timestamp=log["timestamp"],
            label=log["label"],
            confidence=log["confidence"]
        )
        for idx, log in enumerate(logs)
    ]
