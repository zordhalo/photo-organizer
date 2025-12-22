"""
API Dependencies - File Validation and Processing Utilities

This module provides comprehensive file validation for image uploads including:
- File type validation (JPEG, PNG, WebP)
- File size limits (max 10MB)
- Image dimension validation
- EXIF orientation handling
- Request size limits
"""

import io
import os
from typing import Tuple, Optional

from fastapi import UploadFile, HTTPException, Request
from PIL import Image, ExifTags

from app.config import settings


# Constants for validation
MIN_IMAGE_DIMENSION = 32  # Minimum width/height in pixels
MAX_IMAGE_DIMENSION = 10000  # Maximum width/height in pixels
ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
}


def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file type and extension.
    
    Args:
        file: FastAPI UploadFile object
        
    Raises:
        HTTPException: If file type is invalid
    """
    # Check for empty filename
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    # Check file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
        )
    
    # Check content type if provided
    if file.content_type:
        # Normalize content type (some clients send image/jpg instead of image/jpeg)
        content_type = file.content_type.lower()
        if content_type not in ALLOWED_MIME_TYPES and not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type '{file.content_type}'. File must be an image."
            )


async def validate_file_size(file: UploadFile) -> bytes:
    """
    Read and validate file size.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Raw image bytes
        
    Raises:
        HTTPException: If file is too large
    """
    content = await file.read()
    
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded"
        )
    
    if len(content) > settings.MAX_FILE_SIZE:
        size_mb = settings.MAX_FILE_SIZE / 1024 / 1024
        actual_mb = len(content) / 1024 / 1024
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({actual_mb:.1f}MB). Maximum size: {size_mb:.0f}MB"
        )
    
    return content


def validate_image_dimensions(image_data: bytes) -> Tuple[int, int]:
    """
    Validate image dimensions and format.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Tuple of (width, height)
        
    Raises:
        HTTPException: If image is invalid or dimensions are out of range
    """
    try:
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: Could not read image data. {str(e)}"
        )
    
    width, height = image.size
    
    # Check minimum dimensions
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({width}x{height}). Minimum: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels"
        )
    
    # Check maximum dimensions
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({width}x{height}). Maximum: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels"
        )
    
    # Check color mode
    if image.mode not in ["RGB", "RGBA", "L", "P"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image mode: {image.mode}. Supported: RGB, RGBA, L, P"
        )
    
    return width, height


async def validate_and_process_upload(file: UploadFile) -> Tuple[bytes, int, int]:
    """
    Complete validation pipeline for a single image upload.
    
    Performs:
    1. File type validation
    2. File size validation
    3. Image dimension validation
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        Tuple of (image_bytes, width, height)
        
    Raises:
        HTTPException: If any validation fails
    """
    # Validate file type
    validate_image_file(file)
    
    # Read and validate file size
    image_data = await validate_file_size(file)
    
    # Validate image dimensions
    width, height = validate_image_dimensions(image_data)
    
    return image_data, width, height


def get_exif_orientation(image_data: bytes) -> Optional[int]:
    """
    Extract EXIF orientation from image data.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        EXIF orientation value (1-8) or None if not found
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        exif = image._getexif()
        if exif is None:
            return None
            
        # Find orientation tag
        for key, value in ExifTags.TAGS.items():
            if value == "Orientation":
                return exif.get(key)
                
    except Exception:
        pass
    
    return None


def estimate_memory_usage(image_data: bytes) -> float:
    """
    Estimate GPU memory usage for processing an image.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Estimated memory usage in MB
    """
    try:
        image = Image.open(io.BytesIO(image_data))
        width, height = image.size
        
        # Estimate based on image dimensions and model input size
        # Model input: 224x224x3 float32 = ~0.6MB per image in batch
        # Plus intermediate activations: ~10-20MB per image
        base_memory = 0.6  # Input tensor
        activation_memory = 15.0  # Intermediate activations
        
        return base_memory + activation_memory
        
    except Exception:
        return 20.0  # Default estimate


async def validate_request_size(request: Request, max_size: int = None) -> None:
    """
    Validate total request body size.
    
    Args:
        request: FastAPI Request object
        max_size: Maximum allowed size in bytes (defaults to settings)
        
    Raises:
        HTTPException: If request is too large
    """
    if max_size is None:
        # For batch requests, allow larger total size
        max_size = settings.MAX_FILE_SIZE * settings.MAX_BATCH_SIZE
    
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            size = int(content_length)
            if size > max_size:
                max_mb = max_size / 1024 / 1024
                actual_mb = size / 1024 / 1024
                raise HTTPException(
                    status_code=413,
                    detail=f"Request too large ({actual_mb:.1f}MB). Maximum: {max_mb:.0f}MB"
                )
        except ValueError:
            pass  # Invalid content-length header, ignore


class BatchValidationResult:
    """Result of batch file validation."""
    
    def __init__(self):
        self.valid_files: list = []
        self.valid_data: list = []
        self.errors: list = []
        self.total_size: int = 0
        
    @property
    def valid_count(self) -> int:
        return len(self.valid_files)
    
    @property
    def error_count(self) -> int:
        return len(self.errors)


async def validate_batch_upload(
    files: list,
    max_batch_size: int = None
) -> BatchValidationResult:
    """
    Validate a batch of image uploads.
    
    Args:
        files: List of FastAPI UploadFile objects
        max_batch_size: Maximum number of files allowed
        
    Returns:
        BatchValidationResult with valid files and errors
        
    Raises:
        HTTPException: If batch constraints are violated
    """
    if max_batch_size is None:
        max_batch_size = settings.MAX_BATCH_SIZE
    
    # Check batch size limits
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Maximum batch size: {max_batch_size}"
        )
    
    result = BatchValidationResult()
    
    for file in files:
        try:
            # Validate file type
            validate_image_file(file)
            
            # Read and validate size
            image_data = await validate_file_size(file)
            
            # Validate dimensions
            validate_image_dimensions(image_data)
            
            result.valid_files.append(file)
            result.valid_data.append(image_data)
            result.total_size += len(image_data)
            
        except HTTPException as e:
            result.errors.append({
                "filename": file.filename,
                "error": e.detail
            })
            
        except Exception as e:
            result.errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return result
