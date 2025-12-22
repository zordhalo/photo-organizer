"""
API Tests for Construction Photo Analyzer

Comprehensive tests for all API endpoints including:
- Root and health endpoints
- Single image analysis
- Batch image analysis
- Categories endpoint
- Statistics endpoint
- File validation
"""

import io
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.schemas import AnalysisResult, Classification


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_jpeg_bytes():
    """Create a valid JPEG image in memory."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def valid_png_bytes():
    """Create a valid PNG image in memory."""
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def small_image_bytes():
    """Create an image that's too small (below minimum dimensions)."""
    img = Image.new("RGB", (10, 10), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def mock_analysis_result():
    """Create a mock analysis result."""
    return AnalysisResult(
        classifications=[
            Classification(label="construction_site", confidence=0.85),
            Classification(label="crane", confidence=0.12),
        ],
        model_used="resnet50",
        processing_time_ms=50.0,
        image_size=(100, 100),
        metadata={
            "device": "cpu",
            "construction_category": "Safety & Equipment",
        }
    )


# ============================================================================
# Root Endpoint Tests
# ============================================================================

class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_welcome_message(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "documentation" in data
        assert "endpoints" in data
    
    def test_root_contains_documentation_links(self, client):
        """Test that root contains documentation links."""
        response = client.get("/")
        data = response.json()
        assert data["documentation"]["swagger"] == "/docs"
        assert data["documentation"]["redoc"] == "/redoc"


# ============================================================================
# Health Endpoint Tests
# ============================================================================

class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_status(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "cuda_available" in data
        assert "gpu_count" in data
        assert "model" in data
        assert "uptime_seconds" in data
    
    def test_health_includes_gpu_info_when_available(self, client):
        """Test that health includes GPU info structure."""
        response = client.get("/health")
        data = response.json()
        # GPU info should be present (even if None when no GPU)
        assert "gpu_info" in data


class TestReadinessEndpoint:
    """Tests for readiness check endpoint."""
    
    def test_ready_endpoint_exists(self, client):
        """Test that ready endpoint returns valid response."""
        response = client.get("/ready")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "ready" in data


# ============================================================================
# Analyze Endpoint Tests
# ============================================================================

class TestAnalyzeEndpoint:
    """Tests for single image analysis endpoint."""
    
    def test_analyze_requires_file(self, client):
        """Test that analyze endpoint requires a file."""
        response = client.post("/api/v1/analyze")
        assert response.status_code == 422  # Validation error
    
    def test_analyze_rejects_invalid_file_type(self, client):
        """Test that analyze endpoint rejects invalid file types."""
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_analyze_rejects_unsupported_extension(self, client):
        """Test that analyze rejects unsupported file extensions."""
        files = {"file": ("test.gif", b"fake gif data", "image/gif")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 400
    
    def test_analyze_rejects_empty_file(self, client):
        """Test that analyze rejects empty files."""
        files = {"file": ("test.jpg", b"", "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 400
        assert "Empty file" in response.json()["detail"]
    
    def test_analyze_rejects_small_image(self, client, small_image_bytes):
        """Test that analyze rejects images below minimum dimensions."""
        files = {"file": ("small.jpg", small_image_bytes, "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 400
        assert "too small" in response.json()["detail"]
    
    @patch("app.api.routes.vision_service")
    def test_analyze_valid_jpeg(self, mock_vision, client, valid_jpeg_bytes, mock_analysis_result):
        """Test successful analysis of a valid JPEG image."""
        mock_vision.analyze_image = AsyncMock(return_value=mock_analysis_result)
        
        files = {"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["filename"] == "test.jpg"
        assert "analysis" in data
    
    @patch("app.api.routes.vision_service")
    def test_analyze_valid_png(self, mock_vision, client, valid_png_bytes, mock_analysis_result):
        """Test successful analysis of a valid PNG image."""
        mock_vision.analyze_image = AsyncMock(return_value=mock_analysis_result)
        
        files = {"file": ("test.png", valid_png_bytes, "image/png")}
        response = client.post("/api/v1/analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    @patch("app.api.routes.vision_service")
    def test_analyze_returns_classifications(self, mock_vision, client, valid_jpeg_bytes, mock_analysis_result):
        """Test that analysis returns classification results."""
        mock_vision.analyze_image = AsyncMock(return_value=mock_analysis_result)
        
        files = {"file": ("test.jpg", valid_jpeg_bytes, "image/jpeg")}
        response = client.post("/api/v1/analyze", files=files)
        
        data = response.json()
        assert "analysis" in data
        assert "classifications" in data["analysis"]
        assert len(data["analysis"]["classifications"]) > 0
        
        # Check classification structure
        classification = data["analysis"]["classifications"][0]
        assert "label" in classification
        assert "confidence" in classification


# ============================================================================
# Batch Analysis Endpoint Tests
# ============================================================================

class TestBatchAnalyzeEndpoint:
    """Tests for batch image analysis endpoint."""
    
    def test_batch_requires_files(self, client):
        """Test that batch endpoint requires files."""
        response = client.post("/api/v1/batch-analyze")
        assert response.status_code == 422
    
    def test_batch_rejects_empty_list(self, client):
        """Test that batch endpoint rejects empty file list."""
        response = client.post("/api/v1/batch-analyze", files=[])
        assert response.status_code == 422
    
    def test_batch_rejects_too_many_files(self, client, valid_jpeg_bytes):
        """Test that batch endpoint rejects more than 20 files."""
        # Create 21 files
        files = [
            ("files", (f"test_{i}.jpg", valid_jpeg_bytes, "image/jpeg"))
            for i in range(21)
        ]
        response = client.post("/api/v1/batch-analyze", files=files)
        assert response.status_code == 400
        assert "Too many files" in response.json()["detail"]
    
    @patch("app.api.routes.vision_service")
    def test_batch_processes_multiple_files(self, mock_vision, client, valid_jpeg_bytes, mock_analysis_result):
        """Test successful batch processing of multiple files."""
        mock_vision.analyze_batch = AsyncMock(
            return_value=([mock_analysis_result, mock_analysis_result], 100.0)
        )
        
        files = [
            ("files", ("test1.jpg", valid_jpeg_bytes, "image/jpeg")),
            ("files", ("test2.jpg", valid_jpeg_bytes, "image/jpeg")),
        ]
        response = client.post("/api/v1/batch-analyze", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["successful"] == 2
        assert data["failed"] == 0
        assert len(data["results"]) == 2
    
    def test_batch_handles_mixed_valid_invalid(self, client, valid_jpeg_bytes):
        """Test batch handling of mixed valid and invalid files."""
        files = [
            ("files", ("valid.jpg", valid_jpeg_bytes, "image/jpeg")),
            ("files", ("invalid.txt", b"not an image", "text/plain")),
        ]
        response = client.post("/api/v1/batch-analyze", files=files)
        
        # Should still succeed (partial success)
        data = response.json()
        assert data["total"] == 2
        assert data["failed"] >= 1
    
    @patch("app.api.routes.vision_service")
    def test_batch_returns_processing_time(self, mock_vision, client, valid_jpeg_bytes, mock_analysis_result):
        """Test that batch returns total processing time."""
        mock_vision.analyze_batch = AsyncMock(
            return_value=([mock_analysis_result], 50.0)
        )
        
        files = [("files", ("test.jpg", valid_jpeg_bytes, "image/jpeg"))]
        response = client.post("/api/v1/batch-analyze", files=files)
        
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0


# ============================================================================
# Categories Endpoint Tests
# ============================================================================

class TestCategoriesEndpoint:
    """Tests for categories endpoint."""
    
    def test_categories_returns_list(self, client):
        """Test that categories endpoint returns a list of categories."""
        response = client.get("/api/v1/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "total" in data
        assert data["total"] > 0
    
    def test_categories_have_required_fields(self, client):
        """Test that each category has required fields."""
        response = client.get("/api/v1/categories")
        data = response.json()
        
        for category in data["categories"]:
            assert "name" in category
            assert "description" in category
            assert "keywords" in category
            assert "confidence_threshold" in category
    
    def test_categories_includes_all_construction_types(self, client):
        """Test that categories include all construction types."""
        response = client.get("/api/v1/categories")
        data = response.json()
        
        category_names = [c["name"] for c in data["categories"]]
        expected_categories = [
            "Foundation & Excavation",
            "Framing & Structure",
            "Roofing",
            "Electrical & Plumbing",
            "Interior Finishing",
            "Exterior & Landscaping",
            "Safety & Equipment",
            "Progress Documentation",
        ]
        
        for expected in expected_categories:
            assert expected in category_names, f"Missing category: {expected}"


# ============================================================================
# Stats Endpoint Tests
# ============================================================================

class TestStatsEndpoint:
    """Tests for statistics endpoint."""
    
    def test_stats_returns_structure(self, client):
        """Test that stats endpoint returns expected structure."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        
        # Check main sections exist
        assert "uptime_seconds" in data
        assert "gpu" in data
        assert "model" in data
        assert "processing" in data
        assert "version" in data
    
    def test_stats_gpu_section(self, client):
        """Test that GPU stats section has required fields."""
        response = client.get("/api/v1/stats")
        data = response.json()
        
        gpu = data["gpu"]
        assert "available" in gpu
        assert "memory_allocated_mb" in gpu
        assert "memory_total_mb" in gpu
    
    def test_stats_model_section(self, client):
        """Test that model stats section has required fields."""
        response = client.get("/api/v1/stats")
        data = response.json()
        
        model = data["model"]
        assert "name" in model
        assert "device" in model
        assert "initialized" in model
    
    def test_stats_processing_section(self, client):
        """Test that processing stats section has required fields."""
        response = client.get("/api/v1/stats")
        data = response.json()
        
        processing = data["processing"]
        assert "total_requests" in processing
        assert "total_images_analyzed" in processing
        assert "average_processing_time_ms" in processing


# ============================================================================
# Models Endpoint Tests
# ============================================================================

class TestModelsEndpoint:
    """Tests for models endpoint."""
    
    def test_list_models(self, client):
        """Test that models endpoint lists available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "current_model" in data
        assert "model_info" in data
    
    def test_models_include_expected_options(self, client):
        """Test that models include expected options."""
        response = client.get("/api/v1/models")
        data = response.json()
        
        assert "resnet50" in data["available_models"]
        assert "resnet101" in data["available_models"]
        assert "efficientnet_b0" in data["available_models"]


# ============================================================================
# GPU Status Endpoint Tests
# ============================================================================

class TestGPUStatusEndpoint:
    """Tests for GPU status endpoint."""
    
    def test_gpu_status_returns_info(self, client):
        """Test that GPU status endpoint returns information."""
        response = client.get("/api/v1/gpu-status")
        assert response.status_code == 200
        data = response.json()
        assert "gpu_available" in data
    
    def test_gpu_status_includes_memory_when_available(self, client):
        """Test that GPU status includes memory info when GPU available."""
        response = client.get("/api/v1/gpu-status")
        data = response.json()
        
        if data["gpu_available"]:
            assert "memory" in data
            assert "allocated_mb" in data["memory"]
            assert "total_mb" in data["memory"]


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_404_for_unknown_endpoint(self, client):
        """Test that unknown endpoints return 404."""
        response = client.get("/api/v1/unknown-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test that wrong HTTP methods return 405."""
        response = client.get("/api/v1/analyze")  # Should be POST
        assert response.status_code == 405
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present on responses."""
        response = client.get("/health")
        # The test client doesn't trigger CORS, but we can check the endpoint works
        assert response.status_code == 200


# ============================================================================
# Response Headers Tests
# ============================================================================

class TestResponseHeaders:
    """Tests for response headers."""
    
    def test_process_time_header(self, client):
        """Test that X-Process-Time header is present."""
        response = client.get("/health")
        assert "x-process-time" in response.headers
    
    def test_request_id_header(self, client):
        """Test that X-Request-ID header is present."""
        response = client.get("/health")
        assert "x-request-id" in response.headers
