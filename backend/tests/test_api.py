"""
API Tests
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_welcome_message(self, client):
        """Test that root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    

class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_status(self, client):
        """Test that health endpoint returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "cuda_available" in data
        assert "gpu_count" in data


class TestAnalyzeEndpoint:
    """Tests for analyze endpoint."""
    
    def test_analyze_requires_file(self, client):
        """Test that analyze endpoint requires a file."""
        response = client.post("/api/v1/analyze")
        assert response.status_code == 422  # Validation error
    
    def test_analyze_rejects_invalid_file_type(self, client):
        """Test that analyze endpoint rejects invalid file types."""
        # Create a fake text file
        files = {"file": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 400


class TestModelsEndpoint:
    """Tests for models endpoint."""
    
    def test_list_models(self, client):
        """Test that models endpoint lists available models."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "current_model" in data


class TestGPUStatusEndpoint:
    """Tests for GPU status endpoint."""
    
    def test_gpu_status_returns_info(self, client):
        """Test that GPU status endpoint returns information."""
        response = client.get("/api/v1/gpu-status")
        assert response.status_code == 200
        data = response.json()
        assert "gpu_available" in data
