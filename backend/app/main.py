"""
FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("🚀 Starting Construction Photo Analyzer API...")
    print(f"📊 GPU Enabled: {settings.USE_GPU}")
    
    # Initialize vision service on startup
    from app.services.vision_service import vision_service
    if settings.USE_GPU:
        print(f"🎮 CUDA Device: {settings.CUDA_VISIBLE_DEVICES}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Construction Photo Analyzer API...")


app = FastAPI(
    title="Construction Photo Analyzer",
    description="AI-powered construction photo analysis using deep learning",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Construction Photo Analyzer API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import torch
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
