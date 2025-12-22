"""
API Dependencies Tests

Comprehensive tests for file validation and processing utilities including:
- File type validation
- File size validation
- Image dimension validation
- EXIF orientation handling
- Upload processing pipeline
"""

import io
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image

from fastapi import UploadFile, HTTPException

from app.api.dependencies import (
    validate_image_file,
    validate_file_size,
    validate_image_dimensions,
    validate_and_process_upload,
    get_exif_orientation,
    MIN_IMAGE_DIMENSION,
    MAX_IMAGE_DIMENSION,
    ALLOWED_MIME_TYPES,
)
from app.config import settings


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
# File Type Validation Tests
# ============================================================================

class TestValidateImageFile:
    """Tests for validate_image_file function."""
    
    def test_valid_jpeg(self):
        """Test validation of valid JPEG file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.content_type = "image/jpeg"
        
        # Should not raise
        validate_image_file(file)
    
    def test_valid_jpeg_uppercase(self):
        """Test validation of JPEG with uppercase extension."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.JPG"
        file.content_type = "image/jpeg"
        
        validate_image_file(file)
    
    def test_valid_png(self):
        """Test validation of valid PNG file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.png"
        file.content_type = "image/png"
        
        validate_image_file(file)
    
    def test_valid_webp(self):
        """Test validation of valid WebP file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.webp"
        file.content_type = "image/webp"
        
        validate_image_file(file)
    
    def test_invalid_extension(self):
        """Test rejection of invalid file extension."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.gif"
        file.content_type = "image/gif"
        
        with pytest.raises(HTTPException) as exc_info:
            validate_image_file(file)
        
        assert exc_info.value.status_code == 400
        assert "Invalid file type" in exc_info.value.detail
    
    def test_missing_filename(self):
        """Test rejection of missing filename."""
        file = MagicMock(spec=UploadFile)
        file.filename = ""
        file.content_type = "image/jpeg"
        
        with pytest.raises(HTTPException) as exc_info:
            validate_image_file(file)
        
        assert exc_info.value.status_code == 400
        assert "Filename is required" in exc_info.value.detail
    
    def test_none_filename(self):
        """Test rejection of None filename."""
        file = MagicMock(spec=UploadFile)
        file.filename = None
        file.content_type = "image/jpeg"
        
        with pytest.raises(HTTPException) as exc_info:
            validate_image_file(file)
        
        assert exc_info.value.status_code == 400
    
    def test_invalid_content_type_text(self):
        """Test rejection of text content type."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.content_type = "text/plain"
        
        with pytest.raises(HTTPException) as exc_info:
            validate_image_file(file)
        
        assert exc_info.value.status_code == 400
        assert "Invalid content type" in exc_info.value.detail
    
    def test_allows_missing_content_type(self):
        """Test that missing content type is allowed if extension is valid."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.content_type = None
        
        # Should not raise
        validate_image_file(file)


class TestAllowedMimeTypes:
    """Tests for ALLOWED_MIME_TYPES constant."""
    
    def test_jpeg_variations(self):
        """Test JPEG variations are allowed."""
        assert "image/jpeg" in ALLOWED_MIME_TYPES
        assert "image/jpg" in ALLOWED_MIME_TYPES
    
    def test_png_allowed(self):
        """Test PNG is allowed."""
        assert "image/png" in ALLOWED_MIME_TYPES
    
    def test_webp_allowed(self):
        """Test WebP is allowed."""
        assert "image/webp" in ALLOWED_MIME_TYPES
    
    def test_gif_not_allowed(self):
        """Test GIF is not in allowed types."""
        assert "image/gif" not in ALLOWED_MIME_TYPES


# ============================================================================
# File Size Validation Tests
# ============================================================================

class TestValidateFileSize:
    """Tests for validate_file_size function."""
    
    @pytest.mark.asyncio
    async def test_valid_file_size(self):
        """Test validation of valid file size."""
        file = MagicMock(spec=UploadFile)
        file.read = AsyncMock(return_value=b"x" * 1024)  # 1KB
        
        content = await validate_file_size(file)
        
        assert len(content) == 1024
    
    @pytest.mark.asyncio
    async def test_empty_file(self):
        """Test rejection of empty file."""
        file = MagicMock(spec=UploadFile)
        file.read = AsyncMock(return_value=b"")
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_file_size(file)
        
        assert exc_info.value.status_code == 400
        assert "Empty file" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_file_too_large(self):
        """Test rejection of file exceeding size limit."""
        file = MagicMock(spec=UploadFile)
        # Create content larger than MAX_FILE_SIZE
        file.read = AsyncMock(return_value=b"x" * (settings.MAX_FILE_SIZE + 1))
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_file_size(file)
        
        assert exc_info.value.status_code == 413
        assert "too large" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_file_at_max_size(self):
        """Test file exactly at max size is allowed."""
        file = MagicMock(spec=UploadFile)
        file.read = AsyncMock(return_value=b"x" * settings.MAX_FILE_SIZE)
        
        content = await validate_file_size(file)
        
        assert len(content) == settings.MAX_FILE_SIZE


# ============================================================================
# Image Dimension Validation Tests
# ============================================================================

class TestValidateImageDimensions:
    """Tests for validate_image_dimensions function."""
    
    def test_valid_dimensions(self):
        """Test validation of valid dimensions."""
        img = Image.new("RGB", (640, 480))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        width, height = validate_image_dimensions(image_data)
        
        assert width == 640
        assert height == 480
    
    def test_minimum_dimensions(self):
        """Test validation at minimum dimensions."""
        img = Image.new("RGB", (MIN_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        width, height = validate_image_dimensions(image_data)
        
        assert width == MIN_IMAGE_DIMENSION
        assert height == MIN_IMAGE_DIMENSION
    
    def test_below_minimum_width(self):
        """Test rejection of image below minimum width."""
        img = Image.new("RGB", (MIN_IMAGE_DIMENSION - 1, MIN_IMAGE_DIMENSION))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        with pytest.raises(HTTPException) as exc_info:
            validate_image_dimensions(image_data)
        
        assert exc_info.value.status_code == 400
        assert "too small" in exc_info.value.detail.lower()
    
    def test_below_minimum_height(self):
        """Test rejection of image below minimum height."""
        img = Image.new("RGB", (MIN_IMAGE_DIMENSION, MIN_IMAGE_DIMENSION - 1))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        with pytest.raises(HTTPException) as exc_info:
            validate_image_dimensions(image_data)
        
        assert exc_info.value.status_code == 400
    
    def test_above_maximum_width(self):
        """Test rejection of image above maximum width."""
        # Create a small image and pretend it has large dimensions
        # (Actually creating a 10001x100 image would be memory-intensive)
        img = Image.new("RGB", (MAX_IMAGE_DIMENSION + 1, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        with pytest.raises(HTTPException) as exc_info:
            validate_image_dimensions(image_data)
        
        assert exc_info.value.status_code == 400
        assert "too large" in exc_info.value.detail.lower()
    
    def test_invalid_image_data(self):
        """Test rejection of invalid image data."""
        with pytest.raises(HTTPException) as exc_info:
            validate_image_dimensions(b"not an image")
        
        assert exc_info.value.status_code == 400
        assert "Invalid image file" in exc_info.value.detail
    
    def test_rgb_mode_allowed(self):
        """Test RGB mode is allowed."""
        img = Image.new("RGB", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        
        width, height = validate_image_dimensions(buffer.getvalue())
        assert (width, height) == (100, 100)
    
    def test_rgba_mode_allowed(self):
        """Test RGBA mode is allowed."""
        img = Image.new("RGBA", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        width, height = validate_image_dimensions(buffer.getvalue())
        assert (width, height) == (100, 100)
    
    def test_grayscale_mode_allowed(self):
        """Test grayscale (L) mode is allowed."""
        img = Image.new("L", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        
        width, height = validate_image_dimensions(buffer.getvalue())
        assert (width, height) == (100, 100)
    
    def test_palette_mode_allowed(self):
        """Test palette (P) mode is allowed."""
        img = Image.new("P", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        width, height = validate_image_dimensions(buffer.getvalue())
        assert (width, height) == (100, 100)


class TestDimensionConstants:
    """Tests for dimension constants."""
    
    def test_min_dimension_reasonable(self):
        """Test minimum dimension is reasonable."""
        assert MIN_IMAGE_DIMENSION > 0
        assert MIN_IMAGE_DIMENSION < 100
    
    def test_max_dimension_reasonable(self):
        """Test maximum dimension is reasonable."""
        assert MAX_IMAGE_DIMENSION > 1000
        assert MAX_IMAGE_DIMENSION <= 20000


# ============================================================================
# EXIF Orientation Tests
# ============================================================================

class TestGetExifOrientation:
    """Tests for get_exif_orientation function."""
    
    def test_no_exif_data(self):
        """Test image with no EXIF data returns None."""
        img = Image.new("RGB", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        
        orientation = get_exif_orientation(buffer.getvalue())
        
        assert orientation is None
    
    def test_png_no_exif(self):
        """Test PNG (no EXIF support) returns None."""
        img = Image.new("RGB", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        orientation = get_exif_orientation(buffer.getvalue())
        
        assert orientation is None
    
    def test_invalid_data_returns_none(self):
        """Test invalid data returns None instead of raising."""
        orientation = get_exif_orientation(b"not an image")
        
        assert orientation is None


# ============================================================================
# Complete Upload Validation Pipeline Tests
# ============================================================================

class TestValidateAndProcessUpload:
    """Tests for validate_and_process_upload function."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_valid(self):
        """Test complete validation pipeline with valid image."""
        # Create valid image
        img = Image.new("RGB", (640, 480))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        # Create mock upload file
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=image_data)
        
        result = await validate_and_process_upload(file)
        
        assert len(result) == 3
        data, width, height = result
        assert data == image_data
        assert width == 640
        assert height == 480
    
    @pytest.mark.asyncio
    async def test_pipeline_fails_on_invalid_type(self):
        """Test pipeline fails on invalid file type."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.txt"
        file.content_type = "text/plain"
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_and_process_upload(file)
        
        assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_pipeline_fails_on_empty_file(self):
        """Test pipeline fails on empty file."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=b"")
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_and_process_upload(file)
        
        assert exc_info.value.status_code == 400
    
    @pytest.mark.asyncio
    async def test_pipeline_fails_on_small_image(self):
        """Test pipeline fails on image below minimum size."""
        # Create too-small image
        img = Image.new("RGB", (10, 10))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=buffer.getvalue())
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_and_process_upload(file)
        
        assert exc_info.value.status_code == 400
        assert "too small" in exc_info.value.detail.lower()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_multiple_extensions(self):
        """Test file with multiple extensions."""
        file = MagicMock(spec=UploadFile)
        file.filename = "test.tar.jpg"
        file.content_type = "image/jpeg"
        
        # Should work - checks last extension
        validate_image_file(file)
    
    def test_hidden_file(self):
        """Test hidden file (starting with dot)."""
        file = MagicMock(spec=UploadFile)
        file.filename = ".hidden.jpg"
        file.content_type = "image/jpeg"
        
        validate_image_file(file)
    
    def test_unicode_filename(self):
        """Test file with unicode characters in name."""
        file = MagicMock(spec=UploadFile)
        file.filename = "建筑照片.jpg"
        file.content_type = "image/jpeg"
        
        validate_image_file(file)
    
    def test_spaces_in_filename(self):
        """Test file with spaces in name."""
        file = MagicMock(spec=UploadFile)
        file.filename = "my construction photo.jpg"
        file.content_type = "image/jpeg"
        
        validate_image_file(file)
    
    def test_truncated_image(self):
        """Test handling of truncated/corrupted image data."""
        # Create valid image header but truncate it
        img = Image.new("RGB", (100, 100))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        truncated = buffer.getvalue()[:100]  # Only first 100 bytes
        
        # Depending on PIL behavior, this might raise or return partial data
        try:
            width, height = validate_image_dimensions(truncated)
            # If it succeeds, dimensions should be valid
            assert width > 0 and height > 0
        except HTTPException as e:
            # If it fails, should be proper error
            assert e.status_code == 400


# ============================================================================
# estimate_memory_usage Tests
# ============================================================================

class TestEstimateMemoryUsage:
    """Tests for estimate_memory_usage function."""
    
    def test_estimate_valid_image(self):
        """Test memory estimation for valid image."""
        from app.api.dependencies import estimate_memory_usage
        
        image_data = create_test_image(256, 256)
        estimate = estimate_memory_usage(image_data)
        
        # Should return a reasonable positive value
        assert estimate > 0
        assert estimate < 100  # Should be reasonable for a single image
    
    def test_estimate_large_image(self):
        """Test memory estimation for large image."""
        from app.api.dependencies import estimate_memory_usage
        
        image_data = create_test_image(1024, 1024)
        estimate = estimate_memory_usage(image_data)
        
        assert estimate > 0
    
    def test_estimate_invalid_data(self):
        """Test memory estimation for invalid data returns default."""
        from app.api.dependencies import estimate_memory_usage
        
        invalid_data = b"not an image"
        estimate = estimate_memory_usage(invalid_data)
        
        # Should return default estimate
        assert estimate == 20.0


# ============================================================================
# validate_request_size Tests
# ============================================================================

class TestValidateRequestSize:
    """Tests for validate_request_size function."""
    
    @pytest.mark.asyncio
    async def test_valid_request_size(self):
        """Test request with valid size passes."""
        from app.api.dependencies import validate_request_size
        
        # Create a proper mock with headers as a MagicMock
        mock_headers = MagicMock()
        mock_headers.get = MagicMock(return_value="1000")
        
        mock_request = MagicMock()
        mock_request.headers = mock_headers
        
        # Should not raise
        await validate_request_size(mock_request, max_size=10000)
    
    @pytest.mark.asyncio
    async def test_oversized_request(self):
        """Test oversized request is rejected."""
        from app.api.dependencies import validate_request_size
        
        large_size = str(100 * 1024 * 1024)  # 100MB
        mock_headers = MagicMock()
        mock_headers.get = MagicMock(return_value=large_size)
        
        mock_request = MagicMock()
        mock_request.headers = mock_headers
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_request_size(mock_request, max_size=10 * 1024 * 1024)  # 10MB limit
        
        assert exc_info.value.status_code == 413
        assert "too large" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_missing_content_length(self):
        """Test request without content-length header passes."""
        from app.api.dependencies import validate_request_size
        
        mock_headers = MagicMock()
        mock_headers.get = MagicMock(return_value=None)
        
        mock_request = MagicMock()
        mock_request.headers = mock_headers
        
        # Should not raise
        await validate_request_size(mock_request, max_size=10000)
    
    @pytest.mark.asyncio
    async def test_invalid_content_length_header(self):
        """Test invalid content-length header is handled."""
        from app.api.dependencies import validate_request_size
        
        mock_headers = MagicMock()
        mock_headers.get = MagicMock(return_value="not-a-number")
        
        mock_request = MagicMock()
        mock_request.headers = mock_headers
        
        # Should not raise - invalid header is ignored
        await validate_request_size(mock_request, max_size=10000)


# ============================================================================
# BatchValidationResult Tests
# ============================================================================

class TestBatchValidationResult:
    """Tests for BatchValidationResult class."""
    
    def test_create_empty_result(self):
        """Test creating empty batch validation result."""
        from app.api.dependencies import BatchValidationResult
        
        result = BatchValidationResult()
        
        assert result.valid_files == []
        assert result.valid_data == []
        assert result.errors == []
        assert result.total_size == 0
    
    def test_valid_count_property(self):
        """Test valid_count property."""
        from app.api.dependencies import BatchValidationResult
        
        result = BatchValidationResult()
        result.valid_files = ["file1", "file2", "file3"]
        
        assert result.valid_count == 3
    
    def test_error_count_property(self):
        """Test error_count property."""
        from app.api.dependencies import BatchValidationResult
        
        result = BatchValidationResult()
        result.errors = [{"filename": "bad.txt", "error": "Invalid"}]
        
        assert result.error_count == 1


# ============================================================================
# validate_batch_upload Tests
# ============================================================================

class TestValidateBatchUpload:
    """Tests for validate_batch_upload function."""
    
    @pytest.mark.asyncio
    async def test_empty_batch_rejected(self):
        """Test empty file list is rejected."""
        from app.api.dependencies import validate_batch_upload
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_batch_upload([])
        
        assert exc_info.value.status_code == 400
        assert "no files" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_oversized_batch_rejected(self):
        """Test batch exceeding max size is rejected."""
        from app.api.dependencies import validate_batch_upload
        
        files = []
        for i in range(20):
            file = MagicMock(spec=UploadFile)
            file.filename = f"image{i}.jpg"
            file.content_type = "image/jpeg"
            files.append(file)
        
        with pytest.raises(HTTPException) as exc_info:
            await validate_batch_upload(files, max_batch_size=5)
        
        assert exc_info.value.status_code == 400
        assert "too many files" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_valid_batch_processing(self):
        """Test valid batch is processed correctly."""
        from app.api.dependencies import validate_batch_upload
        
        image_data = create_test_image(256, 256)
        
        async def mock_read():
            return image_data
        
        files = []
        for i in range(3):
            file = MagicMock(spec=UploadFile)
            file.filename = f"image{i}.jpg"
            file.content_type = "image/jpeg"
            file.read = mock_read
            files.append(file)
        
        result = await validate_batch_upload(files)
        
        assert result.valid_count == 3
        assert result.error_count == 0
        assert result.total_size == len(image_data) * 3
    
    @pytest.mark.asyncio
    async def test_batch_with_invalid_files(self):
        """Test batch with some invalid files."""
        from app.api.dependencies import validate_batch_upload
        
        # Create mix of valid and invalid files
        valid_image = create_test_image(256, 256)
        
        async def mock_read_valid():
            return valid_image
        
        valid_file = MagicMock(spec=UploadFile)
        valid_file.filename = "valid.jpg"
        valid_file.content_type = "image/jpeg"
        valid_file.read = mock_read_valid
        
        invalid_file = MagicMock(spec=UploadFile)
        invalid_file.filename = "invalid.txt"
        invalid_file.content_type = "text/plain"
        
        result = await validate_batch_upload([valid_file, invalid_file])
        
        assert result.valid_count == 1
        assert result.error_count == 1
        assert result.errors[0]["filename"] == "invalid.txt"
