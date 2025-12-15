from fastapi import APIRouter, BackgroundTasks
import time

router = APIRouter()

@router.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks):
    
    def retrain_model():
        time.sleep(10)  # Simulate retraining time
        print("Model retraining completed.")
    
    background_tasks.add_task(retrain_model)
    return {
        "message": "Model retraining has been triggered.",
        "status": "processing"
    }
