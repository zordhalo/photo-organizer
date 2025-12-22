"""
Helper Utilities Tests

Tests for utility functions in app/utils/helpers.py.
"""

import io
import os
import base64
import tempfile

import pytest
from PIL import Image

from app.utils.helpers import (
    decode_base64_image,
    encode_image_to_base64,
    get_image_info,
    resize_image,
    format_bytes,
)


# ============================================================================
# Test Utilities
# ============================================================================

def create_test_image(width: int = 256, height: int = 256, color: str = "red") -> bytes:
    """Create a test image for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


# ============================================================================
# decode_base64_image Tests
# ============================================================================

class TestDecodeBase64Image:
    """Tests for decode_base64_image function."""
    
    def test_decode_simple_base64(self):
        """Test decoding simple base64 string."""
        original = b"Hello, World!"
        encoded = base64.b64encode(original).decode("utf-8")
        
        result = decode_base64_image(encoded)
        
        assert result == original
    
    def test_decode_with_data_url_prefix(self):
        """Test decoding base64 with data URL prefix."""
        original = b"Test data"
        encoded = base64.b64encode(original).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{encoded}"
        
        result = decode_base64_image(data_url)
        
        assert result == original
    
    def test_decode_image_bytes(self):
        """Test decoding actual image base64."""
        image_data = create_test_image(100, 100)
        encoded = base64.b64encode(image_data).decode("utf-8")
        
        result = decode_base64_image(encoded)
        
        assert result == image_data
        
        # Verify it's a valid image
        img = Image.open(io.BytesIO(result))
        assert img.size == (100, 100)


# ============================================================================
# encode_image_to_base64 Tests
# ============================================================================

class TestEncodeImageToBase64:
    """Tests for encode_image_to_base64 function."""
    
    def test_encode_image_file(self):
        """Test encoding image file to base64."""
        image_data = create_test_image(50, 50)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_data)
            temp_path = f.name
        
        try:
            result = encode_image_to_base64(temp_path)
            
            # Should be valid base64
            assert isinstance(result, str)
            decoded = base64.b64decode(result)
            assert decoded == image_data
        finally:
            os.unlink(temp_path)
    
    def test_encode_produces_valid_base64(self):
        """Test that encoding produces decodable base64."""
        image_data = create_test_image(100, 100)
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_data)
            temp_path = f.name
        
        try:
            encoded = encode_image_to_base64(temp_path)
            
            # Decode and verify
            decoded = decode_base64_image(encoded)
            assert decoded == image_data
        finally:
            os.unlink(temp_path)


# ============================================================================
# get_image_info Tests
# ============================================================================

class TestGetImageInfo:
    """Tests for get_image_info function."""
    
    def test_get_jpeg_info(self):
        """Test getting info for JPEG image."""
        image_data = create_test_image(320, 240)
        
        info = get_image_info(image_data)
        
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["format"] == "JPEG"
        assert info["mode"] == "RGB"
        assert info["size_bytes"] == len(image_data)
    
    def test_get_png_info(self):
        """Test getting info for PNG image."""
        img = Image.new("RGB", (128, 128), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        
        info = get_image_info(image_data)
        
        assert info["width"] == 128
        assert info["height"] == 128
        assert info["format"] == "PNG"
        assert info["mode"] == "RGB"
    
    def test_get_rgba_info(self):
        """Test getting info for RGBA image."""
        img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = buffer.getvalue()
        
        info = get_image_info(image_data)
        
        assert info["mode"] == "RGBA"
    
    def test_size_bytes_accurate(self):
        """Test that size_bytes is accurate."""
        image_data = create_test_image(200, 200)
        
        info = get_image_info(image_data)
        
        assert info["size_bytes"] == len(image_data)


# ============================================================================
# resize_image Tests
# ============================================================================

class TestResizeImage:
    """Tests for resize_image function."""
    
    def test_resize_larger_image(self):
        """Test resizing image larger than max_size."""
        # Create 2000x1000 image
        img = Image.new("RGB", (2000, 1000), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        result = resize_image(image_data, max_size=500)
        
        # Check resized dimensions
        resized = Image.open(io.BytesIO(result))
        assert resized.width <= 500
        assert resized.height <= 500
        # Should maintain aspect ratio
        assert abs(resized.width / resized.height - 2.0) < 0.1
    
    def test_no_resize_smaller_image(self):
        """Test that smaller images are not upscaled."""
        image_data = create_test_image(200, 200)
        original_size = len(image_data)
        
        result = resize_image(image_data, max_size=500)
        
        # Should still be a valid image of similar size
        resized = Image.open(io.BytesIO(result))
        assert resized.size == (200, 200)
    
    def test_resize_maintains_aspect_ratio(self):
        """Test that aspect ratio is maintained."""
        # Create 800x400 image (2:1 aspect)
        img = Image.new("RGB", (800, 400), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        result = resize_image(image_data, max_size=200)
        
        resized = Image.open(io.BytesIO(result))
        aspect = resized.width / resized.height
        assert abs(aspect - 2.0) < 0.1
    
    def test_resize_quality_parameter(self):
        """Test that quality parameter affects output."""
        image_data = create_test_image(500, 500)
        
        high_quality = resize_image(image_data, max_size=200, quality=95)
        low_quality = resize_image(image_data, max_size=200, quality=20)
        
        # Higher quality should produce larger file
        assert len(high_quality) > len(low_quality)


# ============================================================================
# format_bytes Tests
# ============================================================================

class TestFormatBytes:
    """Tests for format_bytes function."""
    
    def test_format_bytes(self):
        """Test formatting bytes."""
        assert "B" in format_bytes(512)
        assert format_bytes(512) == "512.00 B"
    
    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        result = format_bytes(2048)
        assert "KB" in result
        assert "2.00" in result
    
    def test_format_megabytes(self):
        """Test formatting megabytes."""
        result = format_bytes(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.00" in result
    
    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        result = format_bytes(3 * 1024 * 1024 * 1024)
        assert "GB" in result
        assert "3.00" in result
    
    def test_format_terabytes(self):
        """Test formatting terabytes."""
        result = format_bytes(2 * 1024 * 1024 * 1024 * 1024)
        assert "TB" in result
        assert "2.00" in result
    
    def test_format_zero(self):
        """Test formatting zero bytes."""
        result = format_bytes(0)
        assert "0.00 B" == result
