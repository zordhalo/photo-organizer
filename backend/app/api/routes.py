"""
API Routes
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import AnalysisResponse, HealthResponse
from app.services.vision_service import vision_service
from app.api.dependencies import validate_image_file


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
