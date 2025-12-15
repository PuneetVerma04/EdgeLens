from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # MongoDB Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "edgelens_db"
    mongodb_collection_name: str = "inference_logs"
    mongo_root_username: str = ""
    mongo_root_password: str = ""
    
    # Application Configuration
    environment: str = "development"
    log_level: str = "INFO"
    max_file_size_mb: int = 5
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Model Configuration
    model_path: str = "./app/defect_detection_resnet_casting_data.pth"
    
    class Config:
        env_file = str(Path(__file__).parent.parent.parent.parent / ".env")
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
