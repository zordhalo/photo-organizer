"""
API Dependencies
"""

import os
from fastapi import UploadFile, HTTPException

from app.config import settings


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    
    # Check file extension
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
    
    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )


async def validate_file_size(file: UploadFile) -> bytes:
    """Read and validate file size."""
    content = await file.read()
    
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    return content
