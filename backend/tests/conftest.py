"""
Shared Test Fixtures and Configuration

This module provides reusable pytest fixtures for all tests including:
- Test client configuration
- Image generation utilities
- Mock objects
- GPU test utilities
- Performance measurement helpers
"""

import io
import os
import sys
import time
from typing import List, Tuple, Generator
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.models.schemas import AnalysisResult, Classification
from app.config import settings


# ============================================================================
# Test Client Fixtures
# ============================================================================

@pytest.fixture
def client() -> TestClient:
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def client_no_gpu() -> TestClient:
    """Create a test client with GPU disabled."""
    with patch.object(settings, "USE_GPU", False):
        yield TestClient(app)


# ============================================================================
# Image Generation Fixtures
# ============================================================================

@pytest.fixture
def valid_jpeg_bytes() -> bytes:
    """Create a valid JPEG image in memory."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def valid_png_bytes() -> bytes:
    """Create a valid PNG image in memory."""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def valid_webp_bytes() -> bytes:
    """Create a valid WebP image in memory."""
    img = Image.new("RGB", (100, 100), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", quality=85)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def small_image_bytes() -> bytes:
    """Create an image that's too small (below minimum dimensions)."""
    img = Image.new("RGB", (10, 10), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def large_image_bytes() -> bytes:
    """Create a larger test image (1024x1024)."""
    img = Image.new("RGB", (1024, 1024), color="purple")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=80)
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def grayscale_image_bytes() -> bytes:
    """Create a grayscale image."""
    img = Image.new("L", (100, 100), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def rgba_image_bytes() -> bytes:
    """Create an RGBA image with transparency."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def test_image() -> Image.Image:
    """Create a test PIL Image."""
    return Image.new("RGB", (256, 256), color="red")


@pytest.fixture
def test_image_bytes(test_image) -> bytes:
    """Create test image bytes from PIL Image."""
    buffer = io.BytesIO()
    test_image.save(buffer, format="JPEG")
    return buffer.getvalue()


def create_test_image(
    width: int = 256,
    height: int = 256,
    color: str = "red",
    mode: str = "RGB",
    format: str = "JPEG"
) -> bytes:
    """
    Helper function to create test images with custom parameters.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: Fill color (name or hex)
        mode: Image mode (RGB, RGBA, L, etc.)
        format: Output format (JPEG, PNG, WEBP)
    
    Returns:
        Image bytes
    """
    if mode == "RGBA" and format == "JPEG":
        # JPEG doesn't support transparency, convert to RGB
        img = Image.new("RGBA", (width, height), color=color)
        rgb_img = Image.new("RGB", (width, height))
        rgb_img.paste(img, (0, 0), img)
        img = rgb_img
    else:
        img = Image.new(mode, (width, height), color=color if mode != "L" else 128)
    
    buffer = io.BytesIO()
    save_kwargs = {}
    if format == "JPEG":
        save_kwargs["quality"] = 85
    elif format == "WEBP":
        save_kwargs["quality"] = 85
    
    img.save(buffer, format=format, **save_kwargs)
    buffer.seek(0)
    return buffer.getvalue()


def create_batch_images(count: int = 5, varied: bool = True) -> List[bytes]:
    """
    Create multiple test images for batch testing.
    
    Args:
        count: Number of images to create
        varied: Whether to vary image properties
    
    Returns:
        List of image bytes
    """
    images = []
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]
    
    for i in range(count):
        if varied:
            color = colors[i % len(colors)]
            size = 100 + (i * 20)  # Vary size from 100 to 100+(count-1)*20
        else:
            color = "red"
            size = 256
        
        images.append(create_test_image(size, size, color))
    
    return images


# ============================================================================
# Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_analysis_result() -> AnalysisResult:
    """Create a mock analysis result."""
    return AnalysisResult(
        classifications=[
            Classification(label="construction_site", confidence=0.85),
            Classification(label="crane", confidence=0.12),
            Classification(label="building", confidence=0.02),
            Classification(label="scaffolding", confidence=0.005),
            Classification(label="truck", confidence=0.003),
        ],
        model_used="resnet50",
        processing_time_ms=50.0,
        image_size=(100, 100),
        metadata={
            "device": "cpu",
            "construction_category": "Safety & Equipment",
            "input_size": [224, 224],
        }
    )


@pytest.fixture
def mock_vision_service():
    """Create a mock vision service."""
    mock = MagicMock()
    mock.analyze_image = AsyncMock()
    mock.analyze_batch = AsyncMock()
    mock.get_model_info.return_value = {
        "model_name": "resnet50",
        "device": "cpu",
        "cuda_available": False,
        "initialized": True,
    }
    mock.model_name = "resnet50"
    return mock


# ============================================================================
# Performance Measurement Fixtures
# ============================================================================

class PerformanceTimer:
    """Context manager for measuring execution time."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0
        self.elapsed_ms = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return False
    
    def __str__(self):
        return f"{self.name}: {self.elapsed_ms:.2f}ms"


@pytest.fixture
def perf_timer():
    """Provide a performance timer factory."""
    def _create_timer(name: str = "Operation"):
        return PerformanceTimer(name)
    return _create_timer


class BenchmarkResult:
    """Store benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []
        self.min_ms = 0.0
        self.max_ms = 0.0
        self.avg_ms = 0.0
        self.std_ms = 0.0
        self.throughput = 0.0  # items per second
    
    def add_sample(self, time_ms: float, count: int = 1):
        """Add a timing sample."""
        self.times.append(time_ms)
        self._update_stats(count)
    
    def _update_stats(self, count: int = 1):
        """Update statistics from samples."""
        if not self.times:
            return
        
        self.min_ms = min(self.times)
        self.max_ms = max(self.times)
        self.avg_ms = sum(self.times) / len(self.times)
        
        if len(self.times) > 1:
            variance = sum((t - self.avg_ms) ** 2 for t in self.times) / len(self.times)
            self.std_ms = variance ** 0.5
        
        if self.avg_ms > 0:
            self.throughput = (count * 1000) / self.avg_ms
    
    def __str__(self):
        return (
            f"{self.name}: avg={self.avg_ms:.2f}ms, "
            f"min={self.min_ms:.2f}ms, max={self.max_ms:.2f}ms, "
            f"std={self.std_ms:.2f}ms, throughput={self.throughput:.1f}/s"
        )


@pytest.fixture
def benchmark():
    """Provide a benchmark result factory."""
    def _create_benchmark(name: str = "Benchmark"):
        return BenchmarkResult(name)
    return _create_benchmark


# ============================================================================
# GPU Test Utilities
# ============================================================================

def requires_gpu(func):
    """Decorator to skip tests that require GPU."""
    import torch
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )(func)


def get_gpu_memory_usage() -> Tuple[float, float, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Tuple of (allocated_mb, reserved_mb, total_mb)
    """
    import torch
    
    if not torch.cuda.is_available():
        return (0.0, 0.0, 0.0)
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1e6
    reserved = torch.cuda.memory_reserved(device) / 1e6
    total = torch.cuda.get_device_properties(device).total_memory / 1e6
    
    return (allocated, reserved, total)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    import torch
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_categories():
    """Get sample construction categories."""
    return [
        "Foundation & Excavation",
        "Framing & Structure",
        "Roofing",
        "Electrical & Plumbing",
        "Interior Finishing",
        "Exterior & Landscaping",
        "Safety & Equipment",
        "Progress Documentation",
        "Uncategorized",
    ]


@pytest.fixture
def sample_imagenet_classes():
    """Get sample ImageNet class mappings for testing."""
    return {
        518: "crane",
        517: "construction site",
        587: "hammer",
        757: "power drill",
        867: "tractor",
        556: "fence",
        627: "greenhouse",
    }


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def reset_stats():
    """Reset API stats before each test."""
    from app.api.routes import stats
    stats.total_requests = 0
    stats.total_images_analyzed = 0
    stats.batch_requests = 0
    stats.single_requests = 0
    stats.failed_requests = 0
    stats.total_processing_time_ms = 0.0
    yield


@pytest.fixture
def clean_gpu():
    """Ensure GPU memory is clean before and after test."""
    import torch
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    yield
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Async Test Utilities
# ============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
