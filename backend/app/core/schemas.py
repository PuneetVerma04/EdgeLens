from pydantic import BaseModel
from datetime import datetime

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    label: str
    confidence: float
    inference_time: float

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str
    message: str

class PredictionLogResponse(BaseModel):
    """Response model for prediction history logs"""
    id: int
    timestamp: datetime
    label: str
    confidence: float

    class Config:
        from_attributes = True
