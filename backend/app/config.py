"""
Application Configuration
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# Get the backend directory (where .env file should be)
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    DEBUG: bool = False
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = "0"
    USE_GPU: bool = True
    
    # Model Configuration
    MODEL_NAME: str = "resnet50"
    MODEL_WEIGHTS_PATH: str = ""
    BATCH_SIZE: int = 8
    
    # API Configuration
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    API_PREFIX: str = "/api/v1"
    
    # Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_BATCH_SIZE: int = 20  # Maximum images in batch
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # GitHub Integration (for feedback -> issues)
    GITHUB_TOKEN: str = ""
    GITHUB_REPO_OWNER: str = ""
    GITHUB_REPO_NAME: str = ""


def get_settings() -> Settings:
    """Get settings instance (reloads .env on each call during development)."""
    return Settings()


# Create settings instance - will reload when module is reimported
settings = get_settings()
