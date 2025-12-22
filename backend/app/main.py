"""
FastAPI Application Entry Point
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)

# Track startup time for stats
startup_time: float = 0.0


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log incoming request
        logger.info(f"➡️  {request.method} {request.url.path}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"⬅️  {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.2f}ms"
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    global startup_time
    
    # Startup
    startup_time = time.time()
    logger.info("🚀 Starting Construction Photo Analyzer API...")
    logger.info(f"📊 GPU Enabled: {settings.USE_GPU}")
    logger.info(f"🔧 Model: {settings.MODEL_NAME}")
    logger.info(f"🌐 CORS Origins: {settings.CORS_ORIGINS}")
    
    # Initialize vision service on startup
    from app.services.vision_service import vision_service
    if settings.USE_GPU:
        logger.info(f"🎮 CUDA Device: {settings.CUDA_VISIBLE_DEVICES}")
    
    logger.info(f"✅ Server ready at http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    
    yield
    
    # Shutdown
    uptime = time.time() - startup_time
    logger.info(f"👋 Shutting down Construction Photo Analyzer API... (uptime: {uptime:.2f}s)")


app = FastAPI(
    title="Construction Photo Analyzer",
    description="AI-powered construction photo analysis using deep learning",
    version="0.1.0",
    lifespan=lifespan,
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

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
app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Construction Photo Analyzer API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns server status, GPU availability, and system information.
    """
    import torch
    
    gpu_info = {}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_info = {
            "device_name": torch.cuda.get_device_name(device),
            "memory_total_gb": round(torch.cuda.get_device_properties(device).total_memory / 1e9, 2),
        }
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": gpu_info,
        "model": settings.MODEL_NAME,
        "uptime_seconds": round(time.time() - startup_time, 2) if startup_time else 0,
    }


@app.get("/stats")
async def get_stats():
    """
    Get API statistics.
    
    Returns request counts, processing times, and system metrics.
    """
    from app.models.schemas import StatsResponse
    
    return StatsResponse(
        total_requests=0,  # Would be tracked in production
        total_images_analyzed=0,
        average_processing_time_ms=0.0,
        uptime_seconds=round(time.time() - startup_time, 2) if startup_time else 0,
        model_name=settings.MODEL_NAME,
        gpu_enabled=settings.USE_GPU,
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
    )
