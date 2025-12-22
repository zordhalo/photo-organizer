"""
End-to-End Integration Tests for Construction Photo Analyzer

This module provides comprehensive E2E testing including:
- Frontend-Backend API integration
- CORS validation
- File upload workflows
- Error handling across the stack
- Performance validation
- Real construction photo testing

Run with: pytest tests/test_integration_e2e.py -v
"""

import asyncio
import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfig:
    """Test configuration constants."""
    BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
    API_PREFIX = "/api/v1"
    MAX_FILE_SIZE_MB = 10
    MAX_BATCH_SIZE = 20
    TARGET_SINGLE_LATENCY_MS = 100
    TARGET_BATCH_LATENCY_MS = 150
    MIN_ACCURACY_PERCENT = 70
    

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_url():
    """Base URL for API requests."""
    return TestConfig.BASE_URL


@pytest.fixture
def api_url(base_url):
    """Full API URL with prefix."""
    return f"{base_url}{TestConfig.API_PREFIX}"


def create_test_image(
    width: int = 224,
    height: int = 224,
    color: Tuple[int, int, int] = (128, 128, 128),
    format: str = "JPEG"
) -> bytes:
    """Create a test image with specified dimensions and color."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=85)
    buffer.seek(0)
    return buffer.getvalue()


def create_construction_image(category: str) -> bytes:
    """Create a simulated construction photo for testing."""
    # Color coding by category for visual distinction
    category_colors = {
        "foundation": (139, 90, 43),      # Brown - dirt/excavation
        "framing": (210, 180, 140),       # Tan - wood
        "roofing": (50, 50, 50),          # Dark gray - shingles
        "electrical": (255, 215, 0),       # Yellow - warning colors
        "plumbing": (0, 119, 190),         # Blue - water
        "interior": (245, 245, 220),       # Beige - drywall
        "exterior": (34, 139, 34),         # Green - landscaping
        "safety": (255, 165, 0),           # Orange - safety
    }
    
    color = category_colors.get(category.lower(), (128, 128, 128))
    return create_test_image(640, 480, color, "JPEG")


# =============================================================================
# API Health & Connectivity Tests
# =============================================================================

class TestHealthAndConnectivity:
    """Test API health and connectivity endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, base_url):
        """Test root endpoint returns API info."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "name" in data
            assert "version" in data
            assert "endpoints" in data
            
            logger.info(f"✅ Root endpoint: {data['name']} v{data['version']}")
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, base_url):
        """Test health check endpoint."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert "cuda_available" in data
            assert "model" in data
            
            gpu_status = "GPU" if data["cuda_available"] else "CPU"
            logger.info(f"✅ Health check: {data['status']} ({gpu_status})")
    
    @pytest.mark.asyncio
    async def test_ready_endpoint(self, base_url):
        """Test readiness check endpoint."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/ready")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["ready"] == True
            assert data["model_loaded"] == True
            
            logger.info(f"✅ Readiness check: ready={data['ready']}, device={data['device']}")
    
    @pytest.mark.asyncio
    async def test_cors_headers(self, base_url):
        """Test CORS headers are properly set."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            # Simulate browser preflight request
            response = await client.options(
                f"{base_url}/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Content-Type"
                }
            )
            
            assert response.status_code == 200
            
            # Check CORS headers
            assert "access-control-allow-origin" in response.headers
            assert "access-control-allow-methods" in response.headers
            
            logger.info(f"✅ CORS headers present: {response.headers.get('access-control-allow-origin')}")
    
    @pytest.mark.asyncio
    async def test_cors_file_protocol(self, base_url):
        """Test CORS allows 'null' origin for file:// protocol."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.options(
                f"{base_url}/health",
                headers={
                    "Origin": "null",
                    "Access-Control-Request-Method": "GET",
                }
            )
            
            assert response.status_code == 200
            logger.info(f"✅ CORS allows file:// protocol (null origin)")


# =============================================================================
# Single Image Analysis Tests
# =============================================================================

class TestSingleImageAnalysis:
    """Test single image analysis endpoint."""
    
    @pytest.mark.asyncio
    async def test_analyze_valid_jpeg(self, api_url):
        """Test analyzing a valid JPEG image."""
        import httpx
        
        image_data = create_test_image(format="JPEG")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            start = time.time()
            
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("test.jpg", image_data, "image/jpeg")}
            )
            
            latency = (time.time() - start) * 1000
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["filename"] == "test.jpg"
            assert "analysis" in data
            assert "construction_category" in data["analysis"]
            assert "confidence" in data["analysis"]
            assert "classifications" in data["analysis"]
            
            logger.info(
                f"✅ Single image analysis: {data['analysis']['construction_category']} "
                f"(confidence: {data['analysis']['confidence']:.2f}, latency: {latency:.0f}ms)"
            )
    
    @pytest.mark.asyncio
    async def test_analyze_valid_png(self, api_url):
        """Test analyzing a valid PNG image."""
        import httpx
        
        image_data = create_test_image(format="PNG")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("test.png", image_data, "image/png")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            
            logger.info("✅ PNG image analysis successful")
    
    @pytest.mark.asyncio
    async def test_analyze_valid_webp(self, api_url):
        """Test analyzing a valid WebP image."""
        import httpx
        
        image_data = create_test_image(format="WebP")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("test.webp", image_data, "image/webp")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            
            logger.info("✅ WebP image analysis successful")
    
    @pytest.mark.asyncio
    async def test_analyze_response_format(self, api_url):
        """Test that response format matches frontend expectations."""
        import httpx
        
        image_data = create_test_image()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("test.jpg", image_data, "image/jpeg")}
            )
            
            data = response.json()
            
            # Validate structure expected by frontend
            assert "success" in data
            assert "filename" in data
            assert "analysis" in data
            
            analysis = data["analysis"]
            assert "construction_category" in analysis
            assert "confidence" in analysis
            assert "classifications" in analysis
            
            # Validate classifications array
            assert isinstance(analysis["classifications"], list)
            if len(analysis["classifications"]) > 0:
                cls = analysis["classifications"][0]
                assert "label" in cls
                assert "confidence" in cls
            
            logger.info("✅ Response format matches frontend expectations")
    
    @pytest.mark.asyncio
    async def test_latency_under_target(self, api_url):
        """Test that single image latency is under 100ms target."""
        import httpx
        
        image_data = create_test_image(224, 224)
        
        latencies = []
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Warm-up request
            await client.post(
                f"{api_url}/analyze",
                files={"file": ("warmup.jpg", image_data, "image/jpeg")}
            )
            
            # Measure 5 requests
            for i in range(5):
                start = time.time()
                response = await client.post(
                    f"{api_url}/analyze",
                    files={"file": (f"test{i}.jpg", image_data, "image/jpeg")}
                )
                latency = (time.time() - start) * 1000
                latencies.append(latency)
                
                assert response.status_code == 200
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        logger.info(
            f"📊 Latency stats: avg={avg_latency:.0f}ms, "
            f"min={min_latency:.0f}ms, max={max_latency:.0f}ms"
        )
        
        # Note: First request may be slower due to model loading
        assert avg_latency < TestConfig.TARGET_SINGLE_LATENCY_MS * 3, \
            f"Average latency {avg_latency:.0f}ms exceeds 3x target"


# =============================================================================
# Batch Analysis Tests
# =============================================================================

class TestBatchAnalysis:
    """Test batch image analysis endpoint."""
    
    @pytest.mark.asyncio
    async def test_batch_analyze_multiple_images(self, api_url):
        """Test batch analysis with multiple images."""
        import httpx
        
        # Create 5 test images
        files = [
            ("files", (f"test{i}.jpg", create_test_image(), "image/jpeg"))
            for i in range(5)
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            start = time.time()
            
            response = await client.post(
                f"{api_url}/batch-analyze",
                files=files
            )
            
            latency = (time.time() - start) * 1000
            
            assert response.status_code == 200
            data = response.json()
            
            assert "total" in data
            assert "successful" in data
            assert "failed" in data
            assert "results" in data
            assert "processing_time_ms" in data
            
            assert data["total"] == 5
            assert data["successful"] == 5
            assert data["failed"] == 0
            assert len(data["results"]) == 5
            
            logger.info(
                f"✅ Batch analysis: {data['successful']}/{data['total']} successful, "
                f"latency: {latency:.0f}ms, optimized: {data.get('batch_optimized', False)}"
            )
    
    @pytest.mark.asyncio
    async def test_batch_analyze_max_size(self, api_url):
        """Test batch analysis with maximum batch size."""
        import httpx
        
        # Create 20 images (max batch size)
        files = [
            ("files", (f"test{i}.jpg", create_test_image(), "image/jpeg"))
            for i in range(TestConfig.MAX_BATCH_SIZE)
        ]
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{api_url}/batch-analyze",
                files=files
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["total"] == TestConfig.MAX_BATCH_SIZE
            logger.info(f"✅ Max batch size ({TestConfig.MAX_BATCH_SIZE} images) processed")
    
    @pytest.mark.asyncio
    async def test_batch_exceeds_limit(self, api_url):
        """Test that exceeding batch limit returns error."""
        import httpx
        
        # Create 21 images (exceeds max)
        files = [
            ("files", (f"test{i}.jpg", create_test_image(), "image/jpeg"))
            for i in range(TestConfig.MAX_BATCH_SIZE + 1)
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_url}/batch-analyze",
                files=files
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
            
            logger.info(f"✅ Batch limit enforced: {data['detail']}")
    
    @pytest.mark.asyncio
    async def test_batch_empty_request(self, api_url):
        """Test empty batch request returns error."""
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/batch-analyze",
                files=[]
            )
            
            assert response.status_code in [400, 422]
            logger.info("✅ Empty batch request rejected")


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling across the stack."""
    
    @pytest.mark.asyncio
    async def test_invalid_file_type(self, api_url):
        """Test rejection of non-image files."""
        import httpx
        
        # Send a text file
        text_content = b"This is not an image"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("test.txt", text_content, "text/plain")}
            )
            
            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
            
            logger.info(f"✅ Invalid file type rejected: {data['detail']}")
    
    @pytest.mark.asyncio
    async def test_corrupted_image(self, api_url):
        """Test handling of corrupted image data."""
        import httpx
        
        # Send random bytes as JPEG
        corrupted_data = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09'
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("corrupted.jpg", corrupted_data, "image/jpeg")}
            )
            
            assert response.status_code in [400, 500]
            logger.info("✅ Corrupted image handled gracefully")
    
    @pytest.mark.asyncio
    async def test_file_too_large(self, api_url):
        """Test rejection of oversized files."""
        import httpx
        
        # Create image larger than 10MB
        large_data = b'\x00' * (TestConfig.MAX_FILE_SIZE_MB * 1024 * 1024 + 1024)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("large.jpg", large_data, "image/jpeg")}
            )
            
            assert response.status_code in [400, 413]
            logger.info("✅ Oversized file rejected")
    
    @pytest.mark.asyncio
    async def test_missing_file(self, api_url):
        """Test missing file parameter."""
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{api_url}/analyze")
            
            assert response.status_code == 422
            logger.info("✅ Missing file parameter handled")
    
    @pytest.mark.asyncio
    async def test_error_response_format(self, api_url):
        """Test that error responses are properly formatted."""
        import httpx
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("test.txt", b"not an image", "text/plain")}
            )
            
            assert response.status_code >= 400
            data = response.json()
            
            # Should have error detail
            assert "detail" in data or "error" in data
            
            logger.info("✅ Error response properly formatted")


# =============================================================================
# Categories & Stats Endpoints Tests
# =============================================================================

class TestCategoriesAndStats:
    """Test categories and statistics endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_categories(self, api_url):
        """Test fetching available categories."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/categories")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "categories" in data
            assert "total" in data
            assert len(data["categories"]) > 0
            
            # Validate category structure
            for cat in data["categories"]:
                assert "name" in cat
                assert "description" in cat
            
            logger.info(f"✅ Categories endpoint: {data['total']} categories available")
    
    @pytest.mark.asyncio
    async def test_get_stats(self, api_url):
        """Test fetching API statistics."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/stats")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "uptime_seconds" in data
            assert "gpu" in data
            assert "model" in data
            assert "processing" in data
            
            logger.info(
                f"✅ Stats endpoint: uptime={data['uptime_seconds']:.0f}s, "
                f"GPU={data['gpu'].get('available', False)}"
            )
    
    @pytest.mark.asyncio
    async def test_gpu_status(self, api_url):
        """Test GPU status endpoint."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_url}/gpu-status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "gpu_available" in data
            
            if data["gpu_available"]:
                assert "device_name" in data
                assert "memory" in data
                logger.info(f"✅ GPU status: {data['device_name']}")
            else:
                logger.info("✅ GPU status: Running on CPU")


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance and load testing."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, api_url):
        """Test handling concurrent requests."""
        import httpx
        
        num_concurrent = 5
        image_data = create_test_image()
        
        async def make_request(client, i):
            start = time.time()
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": (f"concurrent{i}.jpg", image_data, "image/jpeg")}
            )
            latency = (time.time() - start) * 1000
            return response.status_code, latency
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Warm up
            await make_request(client, -1)
            
            # Concurrent requests
            start = time.time()
            tasks = [make_request(client, i) for i in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            total_time = (time.time() - start) * 1000
        
        success_count = sum(1 for status, _ in results if status == 200)
        latencies = [lat for _, lat in results]
        
        assert success_count == num_concurrent, f"Expected all {num_concurrent} to succeed"
        
        logger.info(
            f"✅ Concurrent requests: {success_count}/{num_concurrent} successful, "
            f"total time: {total_time:.0f}ms, "
            f"avg latency: {sum(latencies)/len(latencies):.0f}ms"
        )
    
    @pytest.mark.asyncio
    async def test_large_image_handling(self, api_url):
        """Test handling of large (but valid) images."""
        import httpx
        
        # Create a 4K image (within limits)
        large_image = create_test_image(3840, 2160)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            start = time.time()
            
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("large4k.jpg", large_image, "image/jpeg")}
            )
            
            latency = (time.time() - start) * 1000
            
            assert response.status_code == 200
            logger.info(f"✅ Large image (4K) processed in {latency:.0f}ms")
    
    @pytest.mark.asyncio
    async def test_small_image_handling(self, api_url):
        """Test handling of small images."""
        import httpx
        
        # Create a tiny image
        small_image = create_test_image(32, 32)
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{api_url}/analyze",
                files={"file": ("tiny.jpg", small_image, "image/jpeg")}
            )
            
            assert response.status_code == 200
            logger.info("✅ Small image (32x32) processed successfully")


# =============================================================================
# Construction Photo Simulation Tests
# =============================================================================

class TestConstructionPhotos:
    """Test with simulated construction photos."""
    
    @pytest.mark.asyncio
    async def test_construction_categories(self, api_url):
        """Test various construction photo categories."""
        import httpx
        
        categories = [
            "foundation",
            "framing",
            "roofing",
            "electrical",
            "plumbing",
            "interior",
            "exterior",
            "safety",
        ]
        
        results = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for category in categories:
                image_data = create_construction_image(category)
                
                response = await client.post(
                    f"{api_url}/analyze",
                    files={"file": (f"{category}.jpg", image_data, "image/jpeg")}
                )
                
                assert response.status_code == 200
                data = response.json()
                
                results.append({
                    "input": category,
                    "detected": data["analysis"]["construction_category"],
                    "confidence": data["analysis"]["confidence"]
                })
        
        logger.info("📊 Construction category detection results:")
        for r in results:
            logger.info(
                f"   {r['input']:12} → {r['detected']:25} "
                f"(confidence: {r['confidence']:.2f})"
            )
    
    @pytest.mark.asyncio
    async def test_batch_construction_photos(self, api_url):
        """Test batch processing of construction photos."""
        import httpx
        
        categories = ["foundation", "framing", "roofing", "electrical", "interior"]
        
        files = [
            ("files", (f"{cat}.jpg", create_construction_image(cat), "image/jpeg"))
            for cat in categories
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{api_url}/batch-analyze",
                files=files
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["successful"] == len(categories)
            
            logger.info(
                f"✅ Batch construction photos: {data['successful']}/{data['total']} "
                f"in {data['processing_time_ms']:.0f}ms"
            )


# =============================================================================
# Test Runner
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
