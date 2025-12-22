"""
API Routes
"""

import logging
import time
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AnalysisResponse, 
    HealthResponse, 
    BatchAnalysisResponse,
    BatchImageUpload,
)
from app.services.vision_service import vision_service
from app.api.dependencies import validate_image_file, validate_file_size
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
):
    """
    Analyze a construction photo using deep learning.
    
    Returns classification results, detected objects, and confidence scores.
    """
    # Validate file
    validate_image_file(file)
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Analyze image
        result = await vision_service.analyze_image(image_data)
        
        return AnalysisResponse(
            success=True,
            filename=file.filename,
            analysis=result,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )


@router.get("/models")
async def list_models():
    """List available models."""
    return {
        "available_models": ["resnet50", "resnet101", "efficientnet_b0"],
        "current_model": vision_service.model_name,
    }


@router.get("/gpu-status")
async def gpu_status():
    """Get GPU status and memory usage."""
    import torch
    
    if not torch.cuda.is_available():
        return {"gpu_available": False}
    
    device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
    
    return {
        "gpu_available": True,
        "device_name": torch.cuda.get_device_name(device),
        "memory_allocated_gb": round(memory_allocated, 2),
        "memory_reserved_gb": round(memory_reserved, 2),
        "memory_total_gb": round(memory_total, 2),
        "utilization_percent": round((memory_allocated / memory_total) * 100, 2),
    }


@router.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    files: List[UploadFile] = File(..., description="Image files to analyze (max 20)"),
):
    """
    Analyze multiple construction photos in batch.
    
    Returns classification results for each image.
    Limited to 20 images per request.
    """
    # Validate batch size
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size: {settings.MAX_BATCH_SIZE}"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Validate file
            validate_image_file(file)
            
            # Read and validate size
            image_data = await validate_file_size(file)
            
            # Analyze image
            result = await vision_service.analyze_image(image_data)
            
            results.append(AnalysisResponse(
                success=True,
                filename=file.filename,
                analysis=result,
            ))
            successful += 1
            
        except HTTPException as e:
            results.append(AnalysisResponse(
                success=False,
                filename=file.filename,
                error=e.detail,
            ))
            failed += 1
            logger.warning(f"Failed to analyze {file.filename}: {e.detail}")
            
        except Exception as e:
            results.append(AnalysisResponse(
                success=False,
                filename=file.filename,
                error=str(e),
            ))
            failed += 1
            logger.error(f"Error analyzing {file.filename}: {str(e)}")
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchAnalysisResponse(
        success=failed == 0,
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
        processing_time_ms=round(processing_time, 2),
    )
