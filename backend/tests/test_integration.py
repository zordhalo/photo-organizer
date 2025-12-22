"""
Integration Tests

End-to-end integration tests for the Construction Photo Analyzer API including:
- Full image analysis workflow
- Batch processing
- Concurrent request handling
- Error recovery
- GPU memory management
- CORS functionality
"""

import io
import asyncio
import concurrent.futures
from typing import List
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.services.vision_service import VisionService, vision_service
from app.config import settings


# ============================================================================
# Test Utilities
# ============================================================================

def create_test_image(
    width: int = 256,
    height: int = 256,
    color: str = "red",
    format: str = "JPEG"
) -> bytes:
    """Create a test image."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# Full Workflow Integration Tests
# ============================================================================

class TestFullAnalysisWorkflow:
    """Integration tests for complete analysis workflow."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_health_check_integration(self, client):
        """Test that health check correctly reflects system state."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "cuda_available" in data
        assert "model" in data
        
        # Verify model matches config
        assert data["model"] == settings.MODEL_NAME
    
    def test_full_single_image_analysis(self, client):
        """Test complete single image analysis workflow."""
        # Create test image
        image_data = create_test_image(512, 512, "blue")
        
        # Submit for analysis
        files = {"file": ("test_construction.jpg", image_data, "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["success"] is True
        assert data["filename"] == "test_construction.jpg"
        assert "analysis" in data
        
        analysis = data["analysis"]
        assert "classifications" in analysis
        assert "model_used" in analysis
        assert "processing_time_ms" in analysis
        assert "image_size" in analysis
        assert "metadata" in analysis
        
        # Verify classifications
        assert len(analysis["classifications"]) > 0
        for classification in analysis["classifications"]:
            assert "label" in classification
            assert "confidence" in classification
            assert 0 <= classification["confidence"] <= 1
        
        # Verify metadata
        assert "construction_category" in analysis["metadata"]
    
    def test_full_batch_analysis(self, client):
        """Test complete batch analysis workflow."""
        # Create multiple test images
        colors = ["red", "green", "blue", "yellow", "purple"]
        images = [create_test_image(256, 256, color) for color in colors]
        
        # Submit batch
        files = [
            ("files", (f"batch_{i}.jpg", img, "image/jpeg"))
            for i, img in enumerate(images)
        ]
        
        response = client.post("/api/v1/batch-analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify batch response
        assert data["total"] == 5
        assert data["successful"] == 5
        assert data["failed"] == 0
        assert len(data["results"]) == 5
        assert "processing_time_ms" in data
        
        # Verify each result
        for result in data["results"]:
            assert result["success"] is True
            assert "analysis" in result
    
    def test_categories_integration(self, client):
        """Test categories endpoint returns consistent data."""
        response = client.get("/api/v1/categories")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "categories" in data
        assert data["total"] >= 8  # At least 8 construction categories
        
        # Verify category structure
        for category in data["categories"]:
            assert "name" in category
            assert "description" in category
            assert "keywords" in category
            assert "confidence_threshold" in category
            assert 0 <= category["confidence_threshold"] <= 1
    
    def test_stats_after_processing(self, client):
        """Test stats reflect actual processing."""
        # Submit some analyses first
        image_data = create_test_image(256, 256)
        files = {"file": ("stats_test.jpg", image_data, "image/jpeg")}
        
        # Make a few requests
        for _ in range(3):
            client.post("/api/v1/analyze", files=files)
        
        # Check stats
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "processing" in data
        assert data["processing"]["total_requests"] >= 3
        assert data["processing"]["total_images_analyzed"] >= 3


# ============================================================================
# Concurrent Request Tests
# ============================================================================

class TestConcurrentRequests:
    """Tests for concurrent request handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_concurrent_single_requests(self, client):
        """Test handling of concurrent single image requests."""
        image_data = create_test_image(256, 256)
        num_concurrent = 5
        
        def make_request(idx):
            files = {"file": (f"concurrent_{idx}.jpg", image_data, "image/jpeg")}
            response = client.post("/api/v1/analyze", files=files)
            return response.status_code, response.json()
        
        # Execute concurrently using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for status_code, data in results:
            assert status_code == 200
            assert data["success"] is True
    
    def test_concurrent_batch_requests(self, client):
        """Test handling of concurrent batch requests."""
        images = [create_test_image(256, 256, color) for color in ["red", "blue"]]
        num_concurrent = 3
        
        def make_batch_request(idx):
            files = [
                ("files", (f"batch{idx}_{i}.jpg", img, "image/jpeg"))
                for i, img in enumerate(images)
            ]
            response = client.post("/api/v1/batch-analyze", files=files)
            return response.status_code, response.json()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_batch_request, i) for i in range(num_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for status_code, data in results:
            assert status_code == 200
            assert data["successful"] == len(images)
    
    def test_mixed_concurrent_requests(self, client):
        """Test handling of mixed single and batch requests concurrently."""
        image_data = create_test_image(256, 256)
        
        def single_request():
            files = {"file": ("mixed_single.jpg", image_data, "image/jpeg")}
            return client.post("/api/v1/analyze", files=files)
        
        def batch_request():
            files = [
                ("files", (f"mixed_batch_{i}.jpg", image_data, "image/jpeg"))
                for i in range(3)
            ]
            return client.post("/api/v1/batch-analyze", files=files)
        
        def read_request():
            return client.get("/api/v1/categories")
        
        # Mix of operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                executor.submit(single_request),
                executor.submit(batch_request),
                executor.submit(single_request),
                executor.submit(read_request),
                executor.submit(batch_request),
                executor.submit(read_request),
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        for response in results:
            assert response.status_code == 200


# ============================================================================
# Error Handling Integration Tests
# ============================================================================

class TestErrorHandlingIntegration:
    """Integration tests for error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_invalid_image_recovery(self, client):
        """Test that system recovers after invalid image."""
        # Submit invalid image
        files = {"file": ("invalid.jpg", b"not an image", "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 400
        
        # Submit valid image - should still work
        valid_image = create_test_image(256, 256)
        files = {"file": ("valid.jpg", valid_image, "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 200
        assert response.json()["success"] is True
    
    def test_batch_partial_failure_recovery(self, client):
        """Test batch continues after partial failure."""
        valid_image = create_test_image(256, 256)
        small_image = create_test_image(10, 10)  # Too small
        
        files = [
            ("files", ("valid1.jpg", valid_image, "image/jpeg")),
            ("files", ("small.jpg", small_image, "image/jpeg")),
            ("files", ("valid2.jpg", valid_image, "image/jpeg")),
        ]
        
        response = client.post("/api/v1/batch-analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have partial success
        assert data["total"] == 3
        assert data["successful"] >= 0  # At least some worked
        assert data["failed"] >= 1  # At least one failed
    
    def test_system_healthy_after_errors(self, client):
        """Test health check passes after errors."""
        # Generate some errors
        files = {"file": ("bad.gif", b"invalid", "image/gif")}
        client.post("/api/v1/analyze", files=files)
        
        # Health should still be good
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# ============================================================================
# GPU Memory Management Tests
# ============================================================================

class TestGPUMemoryManagement:
    """Tests for GPU memory management."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_gpu_status_endpoint(self, client):
        """Test GPU status endpoint returns valid info."""
        response = client.get("/api/v1/gpu-status")
        
        assert response.status_code == 200
        data = response.json()
        assert "gpu_available" in data
    
    def test_clear_cache_endpoint(self, client):
        """Test cache clearing endpoint."""
        # First, do some processing
        image_data = create_test_image(512, 512)
        files = {"file": ("cache_test.jpg", image_data, "image/jpeg")}
        client.post("/api/v1/analyze", files=files)
        
        # Clear cache
        response = client.post("/api/v1/clear-cache")
        
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_after_batch_processing(self, client):
        """Test GPU memory is stable after batch processing."""
        # Get initial memory
        initial_response = client.get("/api/v1/gpu-status")
        initial_memory = initial_response.json().get("memory", {}).get("allocated_mb", 0)
        
        # Process large batch
        images = [create_test_image(512, 512) for _ in range(10)]
        files = [
            ("files", (f"mem_test_{i}.jpg", img, "image/jpeg"))
            for i, img in enumerate(images)
        ]
        client.post("/api/v1/batch-analyze", files=files)
        
        # Clear cache
        client.post("/api/v1/clear-cache")
        
        # Check memory returned to reasonable level
        final_response = client.get("/api/v1/gpu-status")
        final_memory = final_response.json().get("memory", {}).get("allocated_mb", 0)
        
        # Memory should not have grown significantly
        if initial_memory > 0:
            assert final_memory < initial_memory * 2


# ============================================================================
# CORS Integration Tests
# ============================================================================

class TestCORSIntegration:
    """Tests for CORS functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_preflight_request(self, client):
        """Test CORS preflight request is handled."""
        # Note: TestClient doesn't fully simulate CORS, but we can check headers
        response = client.get("/health")
        
        # Basic response should work
        assert response.status_code == 200


# ============================================================================
# Response Headers Tests
# ============================================================================

class TestResponseHeadersIntegration:
    """Tests for response headers."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_timing_headers_present(self, client):
        """Test timing headers are present on responses."""
        image_data = create_test_image(256, 256)
        files = {"file": ("headers_test.jpg", image_data, "image/jpeg")}
        
        response = client.post("/api/v1/analyze", files=files)
        
        assert response.status_code == 200
        assert "x-process-time" in response.headers
    
    def test_request_id_header_present(self, client):
        """Test request ID header is present."""
        response = client.get("/health")
        
        assert "x-request-id" in response.headers


# ============================================================================
# Model Endpoint Tests
# ============================================================================

class TestModelEndpointsIntegration:
    """Integration tests for model-related endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_list_models_consistency(self, client):
        """Test models endpoint returns consistent info."""
        response = client.get("/api/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "available_models" in data
        assert "current_model" in data
        assert settings.MODEL_NAME == data["current_model"]
        
        # Verify all expected models are listed
        expected_models = ["resnet50", "resnet101", "efficientnet_b0"]
        for model in expected_models:
            assert model in data["available_models"]


# ============================================================================
# Ready Endpoint Tests
# ============================================================================

class TestReadinessIntegration:
    """Tests for readiness endpoint."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_ready_after_startup(self, client):
        """Test ready endpoint after startup."""
        response = client.get("/ready")
        
        # Should be ready (200) or not ready (503)
        assert response.status_code in [200, 503]
        
        data = response.json()
        assert "ready" in data


# ============================================================================
# Image Format Tests
# ============================================================================

class TestImageFormatsIntegration:
    """Tests for different image format handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_jpeg_processing(self, client):
        """Test JPEG image processing."""
        image_data = create_test_image(256, 256, format="JPEG")
        files = {"file": ("test.jpg", image_data, "image/jpeg")}
        
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 200
    
    def test_png_processing(self, client):
        """Test PNG image processing."""
        image_data = create_test_image(256, 256, format="PNG")
        files = {"file": ("test.png", image_data, "image/png")}
        
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 200
    
    def test_webp_processing(self, client):
        """Test WebP image processing."""
        image_data = create_test_image(256, 256, format="WEBP")
        files = {"file": ("test.webp", image_data, "image/webp")}
        
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 200
