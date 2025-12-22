"""
Pydantic Schema Tests

Comprehensive tests for all Pydantic models including:
- Field validation
- Default values
- Serialization/deserialization
- Model constraints
- Enum values
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from app.models.schemas import (
    ConstructionCategoryEnum,
    Classification,
    AnalysisResult,
    AnalysisResponse,
    HealthResponse,
    ImageUpload,
    BatchImageUpload,
    BatchAnalysisResponse,
    CategoryInfo,
    CategoriesResponse,
    GPUStats,
    ModelStats,
    ProcessingStats,
    StatsResponse,
    ErrorResponse,
    BatchProgressResponse,
    CATEGORY_DESCRIPTIONS,
)


class TestConstructionCategoryEnum:
    """Tests for ConstructionCategoryEnum."""
    
    def test_all_categories_defined(self):
        """Test all expected categories are defined."""
        expected = [
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
        category_values = [c.value for c in ConstructionCategoryEnum]
        for exp in expected:
            assert exp in category_values, f"Missing category: {exp}"
    
    def test_category_count(self):
        """Test number of categories."""
        assert len(ConstructionCategoryEnum) == 9
    
    def test_enum_is_string_based(self):
        """Test enum values are strings."""
        for category in ConstructionCategoryEnum:
            assert isinstance(category.value, str)
    
    def test_enum_name_format(self):
        """Test enum names are uppercase with underscores."""
        for category in ConstructionCategoryEnum:
            assert category.name.isupper() or "_" in category.name


class TestCategoryDescriptions:
    """Tests for CATEGORY_DESCRIPTIONS dictionary."""
    
    def test_all_categories_have_descriptions(self):
        """Test all enum categories have descriptions."""
        for category in ConstructionCategoryEnum:
            assert category in CATEGORY_DESCRIPTIONS, f"Missing description for {category}"
    
    def test_description_structure(self):
        """Test each category description has required fields."""
        for category, info in CATEGORY_DESCRIPTIONS.items():
            assert "description" in info, f"Missing description field for {category}"
            assert "keywords" in info, f"Missing keywords field for {category}"
            assert "confidence_threshold" in info, f"Missing confidence_threshold for {category}"
    
    def test_keywords_are_lists(self):
        """Test keywords are lists of strings."""
        for category, info in CATEGORY_DESCRIPTIONS.items():
            assert isinstance(info["keywords"], list)
            for keyword in info["keywords"]:
                assert isinstance(keyword, str)
    
    def test_confidence_thresholds_valid(self):
        """Test confidence thresholds are between 0 and 1."""
        for category, info in CATEGORY_DESCRIPTIONS.items():
            threshold = info["confidence_threshold"]
            assert 0 <= threshold <= 1, f"Invalid threshold for {category}: {threshold}"


class TestClassification:
    """Tests for Classification model."""
    
    def test_create_valid_classification(self):
        """Test creating a valid classification."""
        c = Classification(label="test_label", confidence=0.85)
        assert c.label == "test_label"
        assert c.confidence == 0.85
    
    def test_confidence_minimum(self):
        """Test confidence minimum value is 0."""
        c = Classification(label="test", confidence=0.0)
        assert c.confidence == 0.0
    
    def test_confidence_maximum(self):
        """Test confidence maximum value is 1."""
        c = Classification(label="test", confidence=1.0)
        assert c.confidence == 1.0
    
    def test_confidence_below_minimum_fails(self):
        """Test confidence below 0 raises error."""
        with pytest.raises(ValueError):
            Classification(label="test", confidence=-0.1)
    
    def test_confidence_above_maximum_fails(self):
        """Test confidence above 1 raises error."""
        with pytest.raises(ValueError):
            Classification(label="test", confidence=1.1)
    
    def test_label_required(self):
        """Test label is required."""
        with pytest.raises(ValueError):
            Classification(confidence=0.5)
    
    def test_serialization(self):
        """Test model serialization to dict."""
        c = Classification(label="test", confidence=0.5)
        data = c.model_dump()
        assert data == {"label": "test", "confidence": 0.5}


class TestAnalysisResult:
    """Tests for AnalysisResult model."""
    
    def test_create_valid_result(self):
        """Test creating a valid analysis result."""
        result = AnalysisResult(
            classifications=[
                Classification(label="test", confidence=0.9),
            ],
            model_used="resnet50",
            processing_time_ms=45.5,
            image_size=(640, 480),
        )
        assert result.model_used == "resnet50"
        assert result.processing_time_ms == 45.5
        assert result.image_size == (640, 480)
    
    def test_empty_classifications(self):
        """Test result with empty classifications list."""
        result = AnalysisResult(
            classifications=[],
            model_used="resnet50",
            processing_time_ms=10.0,
            image_size=(100, 100),
        )
        assert result.classifications == []
    
    def test_metadata_optional(self):
        """Test metadata field is optional."""
        result = AnalysisResult(
            classifications=[],
            model_used="resnet50",
            processing_time_ms=10.0,
            image_size=(100, 100),
        )
        assert result.metadata is None
    
    def test_metadata_dict(self):
        """Test metadata can be a dictionary."""
        result = AnalysisResult(
            classifications=[],
            model_used="resnet50",
            processing_time_ms=10.0,
            image_size=(100, 100),
            metadata={"device": "cuda:0", "custom": 123},
        )
        assert result.metadata["device"] == "cuda:0"
        assert result.metadata["custom"] == 123
    
    def test_image_size_tuple(self):
        """Test image_size is a tuple."""
        result = AnalysisResult(
            classifications=[],
            model_used="resnet50",
            processing_time_ms=10.0,
            image_size=(1920, 1080),
        )
        assert isinstance(result.image_size, tuple)
        assert len(result.image_size) == 2


class TestAnalysisResponse:
    """Tests for AnalysisResponse model."""
    
    def test_successful_response(self):
        """Test creating a successful response."""
        analysis = AnalysisResult(
            classifications=[],
            model_used="resnet50",
            processing_time_ms=10.0,
            image_size=(100, 100),
        )
        response = AnalysisResponse(
            success=True,
            filename="test.jpg",
            analysis=analysis,
        )
        assert response.success is True
        assert response.filename == "test.jpg"
        assert response.analysis is not None
        assert response.error is None
    
    def test_failed_response(self):
        """Test creating a failed response."""
        response = AnalysisResponse(
            success=False,
            filename="test.jpg",
            error="Image too small",
        )
        assert response.success is False
        assert response.error == "Image too small"
        assert response.analysis is None
    
    def test_filename_optional(self):
        """Test filename is optional."""
        response = AnalysisResponse(success=True)
        assert response.filename is None


class TestHealthResponse:
    """Tests for HealthResponse model."""
    
    def test_create_health_response(self):
        """Test creating a health response."""
        response = HealthResponse(
            status="healthy",
            cuda_available=True,
            gpu_count=1,
            model="resnet50",
        )
        assert response.status == "healthy"
        assert response.cuda_available is True
        assert response.gpu_count == 1
        assert response.model == "resnet50"
    
    def test_gpu_info_optional(self):
        """Test gpu_info is optional."""
        response = HealthResponse(
            status="healthy",
            cuda_available=False,
            gpu_count=0,
            model="resnet50",
        )
        assert response.gpu_info is None
    
    def test_uptime_default(self):
        """Test uptime_seconds has default value."""
        response = HealthResponse(
            status="healthy",
            cuda_available=False,
            gpu_count=0,
            model="resnet50",
        )
        assert response.uptime_seconds == 0.0


class TestImageUpload:
    """Tests for ImageUpload model."""
    
    def test_create_image_upload(self):
        """Test creating an image upload."""
        upload = ImageUpload(image_data="base64encodeddata")
        assert upload.image_data == "base64encodeddata"
    
    def test_filename_optional(self):
        """Test filename is optional."""
        upload = ImageUpload(image_data="data")
        assert upload.filename is None
    
    def test_image_data_required(self):
        """Test image_data is required."""
        with pytest.raises(ValueError):
            ImageUpload()


class TestBatchImageUpload:
    """Tests for BatchImageUpload model."""
    
    def test_create_batch_upload(self):
        """Test creating a batch upload."""
        batch = BatchImageUpload(
            images=[
                ImageUpload(image_data="data1"),
                ImageUpload(image_data="data2"),
            ]
        )
        assert len(batch.images) == 2
    
    def test_empty_list_fails(self):
        """Test empty images list fails validation."""
        with pytest.raises(ValueError):
            BatchImageUpload(images=[])
    
    def test_max_images(self):
        """Test maximum 20 images allowed."""
        images = [ImageUpload(image_data=f"data{i}") for i in range(20)]
        batch = BatchImageUpload(images=images)
        assert len(batch.images) == 20
    
    def test_too_many_images_fails(self):
        """Test more than 20 images fails validation."""
        images = [ImageUpload(image_data=f"data{i}") for i in range(21)]
        with pytest.raises(ValueError):
            BatchImageUpload(images=images)


class TestBatchAnalysisResponse:
    """Tests for BatchAnalysisResponse model."""
    
    def test_create_batch_response(self):
        """Test creating a batch response."""
        response = BatchAnalysisResponse(
            success=True,
            total=5,
            successful=4,
            failed=1,
            results=[],
            processing_time_ms=500.0,
        )
        assert response.total == 5
        assert response.successful == 4
        assert response.failed == 1
    
    def test_batch_optimized_default(self):
        """Test batch_optimized defaults to False."""
        response = BatchAnalysisResponse(
            success=True,
            total=1,
            successful=1,
            failed=0,
            results=[],
            processing_time_ms=100.0,
        )
        assert response.batch_optimized is False
    
    def test_results_can_be_empty(self):
        """Test results can be empty list."""
        response = BatchAnalysisResponse(
            success=True,
            total=0,
            successful=0,
            failed=0,
            results=[],
            processing_time_ms=0.0,
        )
        assert response.results == []


class TestCategoryInfo:
    """Tests for CategoryInfo model."""
    
    def test_create_category_info(self):
        """Test creating category info."""
        info = CategoryInfo(
            name="Foundation & Excavation",
            description="Foundation work",
            keywords=["excavator", "concrete"],
            confidence_threshold=0.3,
        )
        assert info.name == "Foundation & Excavation"
        assert len(info.keywords) == 2
    
    def test_empty_keywords_allowed(self):
        """Test empty keywords list is allowed."""
        info = CategoryInfo(
            name="Test",
            description="Test description",
            keywords=[],
            confidence_threshold=0.5,
        )
        assert info.keywords == []


class TestCategoriesResponse:
    """Tests for CategoriesResponse model."""
    
    def test_create_categories_response(self):
        """Test creating categories response."""
        categories = [
            CategoryInfo(
                name="Test",
                description="Test",
                keywords=[],
                confidence_threshold=0.5,
            )
        ]
        response = CategoriesResponse(categories=categories, total=1)
        assert response.total == 1
        assert len(response.categories) == 1


class TestGPUStats:
    """Tests for GPUStats model."""
    
    def test_gpu_available(self):
        """Test GPU available stats."""
        stats = GPUStats(
            available=True,
            device_name="NVIDIA RTX 4090",
            memory_allocated_mb=1024.0,
            memory_reserved_mb=2048.0,
            memory_total_mb=24576.0,
            utilization_percent=8.33,
        )
        assert stats.available is True
        assert stats.device_name == "NVIDIA RTX 4090"
    
    def test_gpu_unavailable(self):
        """Test GPU unavailable stats."""
        stats = GPUStats(available=False)
        assert stats.available is False
        assert stats.device_name is None
        assert stats.memory_allocated_mb == 0.0
    
    def test_default_values(self):
        """Test default values for GPU stats."""
        stats = GPUStats(available=False)
        assert stats.memory_allocated_mb == 0.0
        assert stats.memory_reserved_mb == 0.0
        assert stats.memory_total_mb == 0.0
        assert stats.utilization_percent == 0.0


class TestModelStats:
    """Tests for ModelStats model."""
    
    def test_create_model_stats(self):
        """Test creating model stats."""
        stats = ModelStats(
            name="resnet50",
            parameters=25557032,
            device="cuda:0",
            initialized=True,
        )
        assert stats.name == "resnet50"
        assert stats.parameters == 25557032
        assert stats.initialized is True
    
    def test_default_parameters(self):
        """Test default parameters value."""
        stats = ModelStats(
            name="test",
            device="cpu",
            initialized=False,
        )
        assert stats.parameters == 0


class TestProcessingStats:
    """Tests for ProcessingStats model."""
    
    def test_create_processing_stats(self):
        """Test creating processing stats."""
        stats = ProcessingStats(
            total_requests=100,
            total_images_analyzed=150,
            average_processing_time_ms=45.5,
            batch_requests=10,
            single_requests=90,
            failed_requests=5,
        )
        assert stats.total_requests == 100
        assert stats.total_images_analyzed == 150
    
    def test_all_defaults_zero(self):
        """Test all fields default to zero."""
        stats = ProcessingStats()
        assert stats.total_requests == 0
        assert stats.total_images_analyzed == 0
        assert stats.average_processing_time_ms == 0.0
        assert stats.batch_requests == 0
        assert stats.single_requests == 0
        assert stats.failed_requests == 0


class TestStatsResponse:
    """Tests for StatsResponse model."""
    
    def test_create_stats_response(self):
        """Test creating stats response."""
        response = StatsResponse(
            uptime_seconds=3600.0,
            gpu=GPUStats(available=False),
            model=ModelStats(name="test", device="cpu", initialized=True),
            processing=ProcessingStats(),
        )
        assert response.uptime_seconds == 3600.0
        assert response.version == "0.1.0"
    
    def test_started_at_optional(self):
        """Test started_at is optional."""
        response = StatsResponse(
            gpu=GPUStats(available=False),
            model=ModelStats(name="test", device="cpu", initialized=True),
            processing=ProcessingStats(),
        )
        assert response.started_at is None


class TestErrorResponse:
    """Tests for ErrorResponse model."""
    
    def test_create_error_response(self):
        """Test creating error response."""
        error = ErrorResponse(error="Something went wrong")
        assert error.success is False
        assert error.error == "Something went wrong"
    
    def test_optional_fields(self):
        """Test optional fields."""
        error = ErrorResponse(
            error="Error",
            detail="Detailed information",
            code="ERR_001",
        )
        assert error.detail == "Detailed information"
        assert error.code == "ERR_001"


class TestBatchProgressResponse:
    """Tests for BatchProgressResponse model."""
    
    def test_create_progress_response(self):
        """Test creating progress response."""
        response = BatchProgressResponse(
            batch_id="abc123",
            total=10,
            processed=5,
            progress_percent=50.0,
            status="processing",
        )
        assert response.batch_id == "abc123"
        assert response.progress_percent == 50.0
        assert response.status == "processing"
    
    def test_eta_optional(self):
        """Test ETA is optional."""
        response = BatchProgressResponse(
            batch_id="test",
            total=10,
            processed=0,
            progress_percent=0.0,
            status="pending",
        )
        assert response.eta_seconds is None


class TestModelSerialization:
    """Tests for model JSON serialization."""
    
    def test_classification_json(self):
        """Test Classification serializes to JSON."""
        c = Classification(label="test", confidence=0.5)
        json_data = c.model_dump_json()
        assert "test" in json_data
        assert "0.5" in json_data
    
    def test_analysis_result_json(self):
        """Test AnalysisResult serializes to JSON."""
        result = AnalysisResult(
            classifications=[Classification(label="test", confidence=0.9)],
            model_used="resnet50",
            processing_time_ms=45.5,
            image_size=(640, 480),
        )
        json_data = result.model_dump_json()
        assert "resnet50" in json_data
        assert "classifications" in json_data
    
    def test_nested_model_serialization(self):
        """Test nested models serialize correctly."""
        response = StatsResponse(
            uptime_seconds=100.0,
            gpu=GPUStats(available=True, device_name="RTX 4090"),
            model=ModelStats(name="resnet50", device="cuda:0", initialized=True),
            processing=ProcessingStats(total_requests=50),
        )
        data = response.model_dump()
        assert data["gpu"]["available"] is True
        assert data["model"]["name"] == "resnet50"
        assert data["processing"]["total_requests"] == 50
