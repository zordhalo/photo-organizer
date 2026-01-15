"""
API Routes for Construction Photo Analyzer

This module provides REST API endpoints for:
- Single image analysis with GPU optimization
- Batch image analysis with GPU batch processing
- Available construction categories
- API statistics and GPU monitoring

Performance Targets:
- Single image: <100ms response time
- Batch (10 images): <150ms response time
- Concurrent requests: 10+ simultaneous
- Max file size: 10MB per image
- Batch limit: 20 images per request
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from app.models.schemas import (
    AnalysisResponse,
    AnalysisResult,
    HealthResponse,
    BatchAnalysisResponse,
    CategoriesResponse,
    CategoryInfo,
    StatsResponse,
    GPUStats,
    ModelStats,
    ProcessingStats,
    ErrorResponse,
    CATEGORY_DESCRIPTIONS,
    ConstructionCategoryEnum,
    FeedbackRequest,
    FeedbackResponse,
)
from app.services.vision_service import vision_service
from app.api.dependencies import (
    validate_image_file,
    validate_file_size,
    validate_image_dimensions,
    validate_and_process_upload,
    validate_batch_upload,
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analysis"])


# ============================================================================
# Request Statistics Tracking
# ============================================================================

class RequestStats:
    """Track API request statistics."""
    
    def __init__(self):
        self.total_requests: int = 0
        self.total_images_analyzed: int = 0
        self.batch_requests: int = 0
        self.single_requests: int = 0
        self.failed_requests: int = 0
        self.total_processing_time_ms: float = 0.0
        self.started_at: datetime = datetime.now(timezone.utc)
    
    def record_single_request(self, processing_time_ms: float, success: bool = True):
        """Record a single image analysis request."""
        self.total_requests += 1
        self.single_requests += 1
        if success:
            self.total_images_analyzed += 1
            self.total_processing_time_ms += processing_time_ms
        else:
            self.failed_requests += 1
    
    def record_batch_request(
        self,
        total_images: int,
        successful: int,
        processing_time_ms: float
    ):
        """Record a batch analysis request."""
        self.total_requests += 1
        self.batch_requests += 1
        self.total_images_analyzed += successful
        self.failed_requests += (total_images - successful)
        self.total_processing_time_ms += processing_time_ms
    
    @property
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time per image."""
        if self.total_images_analyzed == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_images_analyzed


# Global stats instance
stats = RequestStats()


# ============================================================================
# Single Image Analysis Endpoint
# ============================================================================

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        200: {"description": "Successful analysis"},
        400: {"description": "Invalid file or request", "model": ErrorResponse},
        413: {"description": "File too large", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Analyze a single image",
    description="""
Analyze a construction photo using deep learning.

Returns classification results with confidence scores and construction category mapping.

**Supported formats:** JPEG, PNG, WebP
**Max file size:** 10MB
**Expected response time:** <100ms on GPU
    """,
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze (JPEG, PNG, WebP)"),
):
    """
    Analyze a construction photo using deep learning.
    
    Returns classification results, detected objects, and confidence scores.
    The image is automatically categorized into construction-specific categories.
    """
    start_time = time.time()
    
    try:
        # Complete validation pipeline
        image_data, width, height = await validate_and_process_upload(file)
        
        # Analyze image using vision service
        result = await vision_service.analyze_image(image_data)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Record statistics
        stats.record_single_request(processing_time, success=True)
        
        logger.info(
            f"✅ Analyzed {file.filename} in {processing_time:.2f}ms - "
            f"Category: {result.metadata.get('construction_category', 'Unknown')}"
        )
        
        return AnalysisResponse(
            success=True,
            filename=file.filename,
            analysis=result,
        )
        
    except HTTPException:
        stats.record_single_request(0, success=False)
        raise
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        stats.record_single_request(processing_time, success=False)
        
        logger.error(f"❌ Error analyzing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )


# ============================================================================
# Batch Analysis Endpoint
# ============================================================================

@router.post(
    "/batch-analyze",
    response_model=BatchAnalysisResponse,
    responses={
        200: {"description": "Successful batch analysis"},
        400: {"description": "Invalid files or request", "model": ErrorResponse},
        413: {"description": "Request too large", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
    },
    summary="Analyze multiple images in batch",
    description="""
Analyze multiple construction photos using GPU-optimized batch processing.

Returns classification results for each image with aggregated statistics.

**Supported formats:** JPEG, PNG, WebP
**Max files:** 20 images per request
**Max file size:** 10MB per image
**Expected response time:** <150ms for 10 images on GPU
    """,
)
async def analyze_batch(
    files: List[UploadFile] = File(
        ...,
        description="Image files to analyze (max 20)",
    ),
):
    """
    Analyze multiple construction photos in batch.
    
    Uses GPU batch processing for optimal performance.
    Returns classification results for each image with success/failure tracking.
    """
    start_time = time.time()
    
    # Validate batch constraints
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Maximum batch size: {settings.MAX_BATCH_SIZE}"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    results = []
    images_data = []
    file_info = []
    validation_errors = []
    
    # Phase 1: Validate all files first
    for file in files:
        try:
            # Validate file type
            validate_image_file(file)
            
            # Read and validate size
            image_data = await validate_file_size(file)
            
            # Validate dimensions
            validate_image_dimensions(image_data)
            
            images_data.append(image_data)
            file_info.append({"filename": file.filename, "size": len(image_data)})
            
        except HTTPException as e:
            validation_errors.append({
                "filename": file.filename,
                "error": e.detail
            })
            
        except Exception as e:
            validation_errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    # Phase 2: Batch process valid images using GPU optimization
    successful = 0
    failed = len(validation_errors)
    batch_optimized = False
    
    if images_data:
        try:
            # Use GPU-optimized batch processing
            batch_results, batch_time = await vision_service.analyze_batch(images_data)
            batch_optimized = len(images_data) > 1
            
            # Build results for valid images
            for i, result in enumerate(batch_results):
                filename = file_info[i]["filename"]
                
                if result.metadata and result.metadata.get("error"):
                    results.append(AnalysisResponse(
                        success=False,
                        filename=filename,
                        error=result.metadata["error"],
                    ))
                    failed += 1
                else:
                    results.append(AnalysisResponse(
                        success=True,
                        filename=filename,
                        analysis=result,
                    ))
                    successful += 1
                    
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            # Fall back to individual processing
            for i, image_data in enumerate(images_data):
                filename = file_info[i]["filename"]
                try:
                    result = await vision_service.analyze_image(image_data)
                    results.append(AnalysisResponse(
                        success=True,
                        filename=filename,
                        analysis=result,
                    ))
                    successful += 1
                except Exception as img_error:
                    results.append(AnalysisResponse(
                        success=False,
                        filename=filename,
                        error=str(img_error),
                    ))
                    failed += 1
    
    # Phase 3: Add validation error results
    for error_info in validation_errors:
        results.append(AnalysisResponse(
            success=False,
            filename=error_info["filename"],
            error=error_info["error"],
        ))
    
    processing_time = (time.time() - start_time) * 1000
    
    # Record statistics
    stats.record_batch_request(
        total_images=len(files),
        successful=successful,
        processing_time_ms=processing_time
    )
    
    logger.info(
        f"📦 Batch analysis complete: {successful}/{len(files)} successful in {processing_time:.2f}ms"
        f" (batch_optimized={batch_optimized})"
    )
    
    return BatchAnalysisResponse(
        success=failed == 0,
        total=len(files),
        successful=successful,
        failed=failed,
        results=results,
        processing_time_ms=round(processing_time, 2),
        batch_optimized=batch_optimized,
    )


# ============================================================================
# Categories Endpoint
# ============================================================================

@router.get(
    "/categories",
    response_model=CategoriesResponse,
    summary="Get available construction categories",
    description="""
Returns the list of available construction categories with descriptions and confidence thresholds.

Categories are used to automatically classify construction photos based on their content.
    """,
)
async def get_categories():
    """
    Get list of available construction categories.
    
    Returns category names, descriptions, keywords, and confidence thresholds.
    """
    categories = []
    
    for category_enum in ConstructionCategoryEnum:
        category_info = CATEGORY_DESCRIPTIONS.get(category_enum, {})
        categories.append(CategoryInfo(
            name=category_enum.value,
            description=category_info.get("description", ""),
            keywords=category_info.get("keywords", []),
            confidence_threshold=category_info.get("confidence_threshold", 0.3),
        ))
    
    return CategoriesResponse(
        categories=categories,
        total=len(categories),
    )


# ============================================================================
# Statistics Endpoint
# ============================================================================

@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get API statistics",
    description="""
Returns comprehensive API statistics including:
- GPU memory usage and availability
- Model information
- Processing statistics
- Uptime and request counters
    """,
)
async def get_stats():
    """
    Get API statistics including GPU usage, model info, and request counts.
    """
    import torch
    
    # Calculate uptime
    uptime_seconds = (datetime.now(timezone.utc) - stats.started_at).total_seconds()
    
    # GPU stats
    gpu_stats = GPUStats(available=False)
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_reserved = torch.cuda.memory_reserved(device)
            memory_total = torch.cuda.get_device_properties(device).total_memory
            
            gpu_stats = GPUStats(
                available=True,
                device_name=torch.cuda.get_device_name(device),
                memory_allocated_mb=round(memory_allocated / 1e6, 2),
                memory_reserved_mb=round(memory_reserved / 1e6, 2),
                memory_total_mb=round(memory_total / 1e6, 2),
                utilization_percent=round((memory_allocated / memory_total) * 100, 2) if memory_total > 0 else 0,
            )
        except Exception as e:
            logger.warning(f"Error getting GPU stats: {e}")
    
    # Model stats
    model_info = vision_service.get_model_info()
    model_stats = ModelStats(
        name=model_info.get("model_name", settings.MODEL_NAME),
        parameters=model_info.get("model_parameters", 0),
        device=model_info.get("device", "cpu"),
        initialized=model_info.get("initialized", False),
    )
    
    # Processing stats
    processing_stats = ProcessingStats(
        total_requests=stats.total_requests,
        total_images_analyzed=stats.total_images_analyzed,
        average_processing_time_ms=round(stats.average_processing_time_ms, 2),
        batch_requests=stats.batch_requests,
        single_requests=stats.single_requests,
        failed_requests=stats.failed_requests,
    )
    
    return StatsResponse(
        uptime_seconds=round(uptime_seconds, 2),
        started_at=stats.started_at.isoformat() + "Z",
        gpu=gpu_stats,
        model=model_stats,
        processing=processing_stats,
        version="0.1.0",
    )


# ============================================================================
# Additional Utility Endpoints
# ============================================================================

@router.get("/models")
async def list_models():
    """
    List available deep learning models.
    
    Returns available model names and the currently active model.
    """
    return {
        "available_models": ["resnet50", "resnet101", "efficientnet_b0"],
        "current_model": vision_service.model_name,
        "model_info": {
            "resnet50": {
                "parameters": "25.6M",
                "description": "Good balance of speed and accuracy",
                "recommended": True,
            },
            "resnet101": {
                "parameters": "44.5M",
                "description": "Higher accuracy, slower inference",
                "recommended": False,
            },
            "efficientnet_b0": {
                "parameters": "5.3M",
                "description": "Best efficiency, smaller memory footprint",
                "recommended": False,
            },
        }
    }


@router.get("/gpu-status")
async def gpu_status():
    """
    Get detailed GPU status and memory usage.
    
    Returns GPU availability, memory allocation, and utilization metrics.
    """
    import torch
    
    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "message": "CUDA is not available. Running on CPU.",
        }
    
    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        memory_total = props.total_memory
        
        return {
            "gpu_available": True,
            "device_id": device,
            "device_name": torch.cuda.get_device_name(device),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
            "memory": {
                "allocated_mb": round(memory_allocated / 1e6, 2),
                "reserved_mb": round(memory_reserved / 1e6, 2),
                "total_mb": round(memory_total / 1e6, 2),
                "free_mb": round((memory_total - memory_allocated) / 1e6, 2),
                "utilization_percent": round((memory_allocated / memory_total) * 100, 2),
            },
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        
    except Exception as e:
        return {
            "gpu_available": True,
            "error": str(e),
        }


@router.post("/clear-cache")
async def clear_gpu_cache():
    """
    Clear GPU memory cache.
    
    Useful for freeing up GPU memory after processing large batches.
    """
    import torch
    
    if not torch.cuda.is_available():
        return {
            "success": False,
            "message": "CUDA is not available",
        }
    
    try:
        # Get memory before clearing
        device = torch.cuda.current_device()
        memory_before = torch.cuda.memory_allocated(device)
        
        # Clear cache
        vision_service.clear_cuda_cache()
        
        # Get memory after clearing
        memory_after = torch.cuda.memory_allocated(device)
        memory_freed = memory_before - memory_after
        
        return {
            "success": True,
            "memory_freed_mb": round(memory_freed / 1e6, 2),
            "current_allocation_mb": round(memory_after / 1e6, 2),
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================================
# Feedback Endpoint - Creates GitHub Issues
# ============================================================================

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        200: {"description": "Feedback submitted successfully"},
        500: {"description": "GitHub API error or configuration issue", "model": ErrorResponse},
    },
    summary="Submit user feedback",
    description="""
Submit user feedback which creates a GitHub issue in the project repository.

Requires GitHub integration to be configured via environment variables:
- GITHUB_TOKEN: Personal access token with 'repo' scope
- GITHUB_REPO_OWNER: GitHub username or organization
- GITHUB_REPO_NAME: Repository name
    """,
)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback and create a GitHub issue.
    
    The feedback is posted to the configured GitHub repository as a new issue
    with the 'feedback' and 'user-submitted' labels.
    """
    import httpx
    
    # Check if GitHub integration is configured
    if not settings.GITHUB_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="GitHub integration not configured. Please set GITHUB_TOKEN environment variable."
        )
    
    if not settings.GITHUB_REPO_OWNER or not settings.GITHUB_REPO_NAME:
        raise HTTPException(
            status_code=500,
            detail="GitHub repository not configured. Please set GITHUB_REPO_OWNER and GITHUB_REPO_NAME."
        )
    
    # Build issue body
    issue_body = f"""## User Feedback

{request.feedback}

---
**Submitted via:** Construction Photo Analyzer v0.1.0
**Timestamp:** {datetime.now(timezone.utc).isoformat()}
"""
    
    if request.system_status:
        issue_body += f"""
**System Status at submission:**
- API Server: {request.system_status.api_status or 'Unknown'}
- Backend: {request.system_status.backend_info or 'Unknown'}
"""

    # Create issue title (truncate feedback for title)
    feedback_preview = request.feedback[:50].replace('\n', ' ')
    if len(request.feedback) > 50:
        feedback_preview += "..."
    issue_title = f"[Feedback] {feedback_preview}"

    # Create GitHub issue
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.github.com/repos/{settings.GITHUB_REPO_OWNER}/{settings.GITHUB_REPO_NAME}/issues",
                headers={
                    "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28"
                },
                json={
                    "title": issue_title,
                    "body": issue_body,
                    "labels": ["feedback", "user-submitted"]
                },
                timeout=30.0
            )
        
        if response.status_code == 201:
            issue_data = response.json()
            logger.info(f"✅ Feedback submitted - GitHub issue #{issue_data['number']} created")
            return FeedbackResponse(
                success=True,
                issue_url=issue_data["html_url"],
                issue_number=issue_data["number"],
                message="Feedback submitted successfully"
            )
        else:
            error_detail = response.text
            logger.error(f"❌ GitHub API error: {response.status_code} - {error_detail}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"GitHub API error: {error_detail}"
            )
            
    except httpx.TimeoutException:
        logger.error("❌ GitHub API timeout")
        raise HTTPException(
            status_code=504,
            detail="GitHub API request timed out. Please try again."
        )
    except httpx.RequestError as e:
        logger.error(f"❌ GitHub API request error: {str(e)}")
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to GitHub API: {str(e)}"
        )
