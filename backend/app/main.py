"""
FastAPI Application Entry Point

Construction Photo Analyzer - AI-powered construction photo analysis using deep learning.

This module provides:
- Application initialization and lifecycle management
- CORS and middleware configuration
- Health check and root endpoints
- Request logging and timing
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
startup_datetime: datetime = None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests with timing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log incoming request
        logger.info(f"➡️  {request.method} {request.url.path}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000
            
            # Log response with timing
            logger.info(
                f"⬅️  {request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.2f}ms"
            )
            
            # Add timing headers
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            response.headers["X-Request-ID"] = str(int(time.time() * 1000000))
            
            return response
            
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"❌ {request.method} {request.url.path} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.2f}ms"
            )
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events for startup and shutdown."""
    global startup_time, startup_datetime
    
    # ============================================================
    # STARTUP
    # ============================================================
    startup_time = time.time()
    startup_datetime = datetime.utcnow()
    
    logger.info("🚀 Starting Construction Photo Analyzer API...")
    logger.info(f"📊 GPU Enabled: {settings.USE_GPU}")
    logger.info(f"🔧 Model: {settings.MODEL_NAME}")
    logger.info(f"🌐 CORS Origins: {settings.CORS_ORIGINS}")
    logger.info(f"📁 Max File Size: {settings.MAX_FILE_SIZE / 1024 / 1024:.0f}MB")
    logger.info(f"📦 Max Batch Size: {settings.MAX_BATCH_SIZE}")
    
    # Initialize vision service on startup (lazy loading, just validates config)
    from app.services.vision_service import vision_service
    
    if settings.USE_GPU:
        logger.info(f"🎮 CUDA Device: {settings.CUDA_VISIBLE_DEVICES}")
    
    logger.info(f"✅ Server ready at http://{settings.HOST}:{settings.PORT}")
    logger.info(f"📚 API Documentation: http://{settings.HOST}:{settings.PORT}/docs")
    logger.info(f"📖 ReDoc: http://{settings.HOST}:{settings.PORT}/redoc")
    
    yield
    
    # ============================================================
    # SHUTDOWN
    # ============================================================
    uptime = time.time() - startup_time
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    seconds = int(uptime % 60)
    
    logger.info(
        f"👋 Shutting down Construction Photo Analyzer API... "
        f"(uptime: {hours}h {minutes}m {seconds}s)"
    )


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Construction Photo Analyzer",
    description="""
## AI-Powered Construction Photo Analysis API

Analyze construction site photos using deep learning with GPU acceleration.

### Features
- **Single Image Analysis**: Classify individual construction photos
- **Batch Processing**: Analyze up to 20 images with GPU optimization
- **Construction Categories**: Automatic categorization into construction phases
- **GPU Acceleration**: CUDA-optimized for fast inference

### Performance Targets
- Single image: <100ms response time
- Batch (10 images): <150ms response time
- Concurrent requests: 10+ simultaneous
- Max file size: 10MB per image

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Configure CORS
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else []
# Add common development origins
origins.extend([
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    "null",  # For file:// protocol (local HTML files)
])
# Remove duplicates while preserving order
origins = list(dict.fromkeys(origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)

# Include API routes with versioned prefix
app.include_router(router, prefix=settings.API_PREFIX)


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information and links.
    """
    return {
        "name": "Construction Photo Analyzer API",
        "version": "0.1.0",
        "description": "AI-powered construction photo analysis using deep learning",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
        },
        "endpoints": {
            "health": "/health",
            "analyze": f"{settings.API_PREFIX}/analyze",
            "batch_analyze": f"{settings.API_PREFIX}/batch-analyze",
            "categories": f"{settings.API_PREFIX}/categories",
            "stats": f"{settings.API_PREFIX}/stats",
        },
    }


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns server status, GPU availability, and system information.
    Use this endpoint for load balancer health checks.
    """
    import torch
    
    gpu_info = None
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            gpu_info = {
                "device_name": torch.cuda.get_device_name(device),
                "memory_total_gb": round(
                    torch.cuda.get_device_properties(device).total_memory / 1e9, 2
                ),
                "memory_allocated_gb": round(
                    torch.cuda.memory_allocated(device) / 1e9, 2
                ),
            }
        except Exception:
            pass
    
    uptime = time.time() - startup_time if startup_time else 0
    
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_info": gpu_info,
        "model": settings.MODEL_NAME,
        "uptime_seconds": round(uptime, 2),
        "started_at": startup_datetime.isoformat() + "Z" if startup_datetime else None,
    }


@app.get("/ready", tags=["health"])
async def readiness_check():
    """
    Readiness check endpoint.
    
    Verifies that the model is loaded and ready to process requests.
    """
    from app.services.vision_service import vision_service
    
    try:
        # Check if vision service is initialized
        if not vision_service._initialized:
            vision_service.initialize()
        
        model_info = vision_service.get_model_info()
        
        return {
            "ready": True,
            "model_loaded": model_info.get("initialized", False),
            "device": model_info.get("device", "unknown"),
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "ready": False,
                "error": str(e),
            }
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred",
        }
    )


# ============================================================================
# Development Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
