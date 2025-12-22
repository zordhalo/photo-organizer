"""
Helper Utilities
"""

import os
import base64
from typing import Optional
from PIL import Image
import io


def decode_base64_image(base64_string: str) -> bytes:
    """
    Decode a base64 encoded image string.
    
    Args:
        base64_string: Base64 encoded image data
        
    Returns:
        Raw image bytes
    """
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    return base64.b64decode(base64_string)


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_info(image_data: bytes) -> dict:
    """
    Get information about an image from bytes.
    
    Args:
        image_data: Raw image bytes
        
    Returns:
        Dictionary with image info
    """
    image = Image.open(io.BytesIO(image_data))
    
    return {
        "width": image.width,
        "height": image.height,
        "format": image.format,
        "mode": image.mode,
        "size_bytes": len(image_data),
    }


def resize_image(
    image_data: bytes,
    max_size: int = 1024,
    quality: int = 85
) -> bytes:
    """
    Resize an image while maintaining aspect ratio.
    
    Args:
        image_data: Raw image bytes
        max_size: Maximum dimension (width or height)
        quality: JPEG quality for output
        
    Returns:
        Resized image bytes
    """
    image = Image.open(io.BytesIO(image_data))
    
    # Calculate new size
    ratio = min(max_size / image.width, max_size / image.height)
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Save to bytes
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    return output.getvalue()


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"
