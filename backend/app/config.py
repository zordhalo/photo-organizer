"""
Application Configuration
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = "0"
    USE_GPU: bool = True
    
    # Model Configuration
    MODEL_NAME: str = "resnet50"
    BATCH_SIZE: int = 8
    
    # API Configuration
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    # Upload Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
