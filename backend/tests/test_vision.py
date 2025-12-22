"""
Vision Service Tests - GPU-Accelerated Image Classification

Comprehensive test suite for the VisionService including:
- CUDA/GPU availability and device configuration
- Model loading and initialization
- Single and batch image classification
- Construction category mapping
- EXIF orientation handling
- Image validation
- Performance benchmarks
"""

import io
import time
from typing import List

import pytest
import torch
from PIL import Image

from app.services.vision_service import (
    VisionService,
    ConstructionCategory,
    IMAGENET_TO_CONSTRUCTION,
    CATEGORY_KEYWORDS,
    fix_exif_orientation,
    validate_image,
    load_imagenet_labels,
)


class TestDeviceConfiguration:
    """Tests for GPU/CPU device configuration."""
    
    def test_cuda_availability_detection(self):
        """Test CUDA availability is correctly detected."""
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
    
    def test_torch_version(self):
        """Test PyTorch is installed correctly."""
        assert torch.__version__ is not None
        print(f"PyTorch Version: {torch.__version__}")
    
    def test_tensor_gpu_transfer(self):
        """Test tensor can be moved to GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create tensor on CPU
        cpu_tensor = torch.rand(5, 3, 224, 224)
        
        # Move to GPU
        gpu_tensor = cpu_tensor.cuda()
        
        assert gpu_tensor.device.type == "cuda"
        assert gpu_tensor.shape == cpu_tensor.shape
        
    def test_cuda_memory_management(self):
        """Test CUDA memory allocation and clearing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        device = torch.device("cuda:0")
        
        # Allocate some memory
        tensor = torch.rand(1000, 1000, device=device)
        allocated_before = torch.cuda.memory_allocated(device)
        
        # Delete tensor and clear cache
        del tensor
        torch.cuda.empty_cache()
        allocated_after = torch.cuda.memory_allocated(device)
        
        # Memory should be freed
        assert allocated_after < allocated_before


class TestVisionServiceInitialization:
    """Tests for VisionService initialization."""
    
    def test_service_creation(self):
        """Test VisionService can be created."""
        service = VisionService()
        assert service is not None
        assert service._initialized is False
        assert service.model is None
        
    def test_model_loading(self):
        """Test that model can be loaded."""
        service = VisionService()
        service.initialize()
        
        assert service.model is not None
        assert service._initialized is True
        assert service.transform is not None
        
    def test_device_selection(self):
        """Test correct device is selected."""
        service = VisionService()
        service.initialize()
        
        if torch.cuda.is_available():
            assert service.device.type == "cuda"
        else:
            assert service.device.type == "cpu"
            
    def test_model_eval_mode(self):
        """Test model is in evaluation mode after loading."""
        service = VisionService()
        service.initialize()
        
        assert not service.model.training
        
    def test_get_model_info(self):
        """Test model info retrieval."""
        service = VisionService()
        info = service.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert "cuda_available" in info
        assert "initialized" in info


class TestImagePreprocessing:
    """Tests for image preprocessing functions."""
    
    def test_fix_exif_orientation_no_exif(self):
        """Test EXIF handling with no EXIF data."""
        image = Image.new("RGB", (100, 100), color="red")
        result = fix_exif_orientation(image)
        
        assert result.size == (100, 100)
        
    def test_validate_image_valid(self):
        """Test image validation with valid image."""
        image = Image.new("RGB", (256, 256), color="blue")
        is_valid, error = validate_image(image)
        
        assert is_valid is True
        assert error == ""
        
    def test_validate_image_too_small(self):
        """Test image validation with too small image."""
        image = Image.new("RGB", (10, 10), color="blue")
        is_valid, error = validate_image(image)
        
        assert is_valid is False
        assert "too small" in error.lower()
        
    def test_validate_image_too_large(self):
        """Test image validation with too large image."""
        image = Image.new("RGB", (15000, 15000), color="blue")
        is_valid, error = validate_image(image)
        
        assert is_valid is False
        assert "too large" in error.lower()
        
    def test_rgb_conversion(self):
        """Test image is converted to RGB."""
        service = VisionService()
        service.initialize()
        
        # Create grayscale image
        image = Image.new("L", (256, 256), color=128)
        
        # Preprocess should convert to RGB
        tensor = service._preprocess_image(image)
        assert tensor.shape[0] == 3  # RGB channels


class TestImageClassification:
    """Tests for image classification functionality."""
    
    @pytest.fixture
    def vision_service(self):
        """Create initialized vision service."""
        service = VisionService()
        service.initialize()
        return service
        
    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Create a test image."""
        return Image.new("RGB", (256, 256), color="red")
        
    @pytest.fixture
    def test_image_bytes(self, test_image) -> bytes:
        """Create test image bytes."""
        buffer = io.BytesIO()
        test_image.save(buffer, format="JPEG")
        return buffer.getvalue()
        
    def test_classify_image(self, vision_service, test_image):
        """Test single image classification."""
        result = vision_service.classify_image(test_image)
        
        assert "predictions" in result
        assert "category" in result
        assert "top_confidence" in result
        assert len(result["predictions"]) == 5
        assert result["predictions"][0]["confidence"] > 0
        
    def test_classify_image_predictions_format(self, vision_service, test_image):
        """Test prediction format is correct."""
        result = vision_service.classify_image(test_image)
        
        for pred in result["predictions"]:
            assert "class" in pred
            assert "label" in pred
            assert "confidence" in pred
            assert isinstance(pred["class"], int)
            assert isinstance(pred["confidence"], float)
            assert 0 <= pred["confidence"] <= 1
            
    def test_classify_batch_empty(self, vision_service):
        """Test batch classification with empty list."""
        results = vision_service.classify_batch([])
        assert results == []
        
    def test_classify_batch_single(self, vision_service, test_image):
        """Test batch classification with single image."""
        results = vision_service.classify_batch([test_image])
        
        assert len(results) == 1
        assert "predictions" in results[0]
        assert "category" in results[0]
        
    def test_classify_batch_multiple(self, vision_service):
        """Test batch classification with multiple images."""
        images = [
            Image.new("RGB", (256, 256), color="red"),
            Image.new("RGB", (256, 256), color="green"),
            Image.new("RGB", (256, 256), color="blue"),
        ]
        
        results = vision_service.classify_batch(images)
        
        assert len(results) == 3
        for result in results:
            assert "predictions" in result
            assert "category" in result
            
    @pytest.mark.asyncio
    async def test_analyze_image(self, vision_service, test_image_bytes):
        """Test async image analysis API."""
        result = await vision_service.analyze_image(test_image_bytes)
        
        assert result is not None
        assert len(result.classifications) > 0
        assert result.processing_time_ms > 0
        assert result.image_size == (256, 256)
        assert result.model_used == vision_service.model_name
        assert "construction_category" in result.metadata
        
    @pytest.mark.asyncio
    async def test_analyze_batch(self, vision_service, test_image_bytes):
        """Test async batch analysis."""
        images_data = [test_image_bytes, test_image_bytes, test_image_bytes]
        
        results, total_time = await vision_service.analyze_batch(images_data)
        
        assert len(results) == 3
        assert total_time > 0
        for result in results:
            assert len(result.classifications) > 0


class TestConstructionCategoryMapping:
    """Tests for construction category mapping."""
    
    def test_all_categories_defined(self):
        """Test all construction categories are defined."""
        categories = [
            ConstructionCategory.FOUNDATION_EXCAVATION,
            ConstructionCategory.FRAMING_STRUCTURE,
            ConstructionCategory.ROOFING,
            ConstructionCategory.ELECTRICAL_PLUMBING,
            ConstructionCategory.INTERIOR_FINISHING,
            ConstructionCategory.EXTERIOR_LANDSCAPING,
            ConstructionCategory.SAFETY_EQUIPMENT,
            ConstructionCategory.PROGRESS_DOCUMENTATION,
            ConstructionCategory.UNKNOWN,
        ]
        
        assert len(categories) == 9
        
    def test_imagenet_mapping_exists(self):
        """Test ImageNet to construction mapping is populated."""
        assert len(IMAGENET_TO_CONSTRUCTION) > 0
        
        # All values should be valid construction categories
        valid_categories = {
            ConstructionCategory.FOUNDATION_EXCAVATION,
            ConstructionCategory.FRAMING_STRUCTURE,
            ConstructionCategory.ROOFING,
            ConstructionCategory.ELECTRICAL_PLUMBING,
            ConstructionCategory.INTERIOR_FINISHING,
            ConstructionCategory.EXTERIOR_LANDSCAPING,
            ConstructionCategory.SAFETY_EQUIPMENT,
            ConstructionCategory.PROGRESS_DOCUMENTATION,
        }
        
        for category in IMAGENET_TO_CONSTRUCTION.values():
            assert category in valid_categories
            
    def test_category_keywords_exist(self):
        """Test category keywords are defined."""
        assert len(CATEGORY_KEYWORDS) > 0
        
        for category, keywords in CATEGORY_KEYWORDS.items():
            assert len(keywords) > 0
            assert all(isinstance(k, str) for k in keywords)
            
    def test_map_to_construction_category_direct(self):
        """Test direct class index mapping."""
        service = VisionService()
        service.initialize()
        
        # Test a known mapped class
        for class_idx, expected_category in IMAGENET_TO_CONSTRUCTION.items():
            result = service._map_to_construction_category(class_idx, "")
            assert result == expected_category
            break  # Just test one
            
    def test_map_to_construction_category_keyword(self):
        """Test keyword-based mapping."""
        service = VisionService()
        service.initialize()
        
        # Test keyword matching - use unknown class index to force keyword match
        result = service._map_to_construction_category(99999, "excavator digging")
        assert result == ConstructionCategory.FOUNDATION_EXCAVATION
        
        result = service._map_to_construction_category(99999, "safety helmet")
        assert result == ConstructionCategory.SAFETY_EQUIPMENT
        
    def test_get_construction_categories(self):
        """Test getting all construction categories."""
        service = VisionService()
        categories = service.get_construction_categories()
        
        assert len(categories) == 8
        assert ConstructionCategory.FOUNDATION_EXCAVATION in categories
        assert ConstructionCategory.SAFETY_EQUIPMENT in categories


class TestImageNetLabels:
    """Tests for ImageNet labels loading."""
    
    def test_load_imagenet_labels(self):
        """Test ImageNet labels can be loaded."""
        labels = load_imagenet_labels()
        
        # Should have labels loaded from model weights
        assert len(labels) > 0
        
    def test_labels_are_strings(self):
        """Test all labels are strings."""
        labels = load_imagenet_labels()
        
        for idx, label in labels.items():
            assert isinstance(idx, int)
            assert isinstance(label, str)
            assert len(label) > 0


class TestPerformance:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def vision_service(self):
        """Create initialized vision service."""
        service = VisionService()
        service.initialize()
        return service
        
    def test_single_image_inference_time(self, vision_service):
        """Test single image inference performance."""
        image = Image.new("RGB", (256, 256), color="red")
        
        # Warmup
        vision_service.classify_image(image)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            vision_service.classify_image(image)
            times.append((time.time() - start) * 1000)
        
        avg_time = sum(times) / len(times)
        print(f"Average single image inference: {avg_time:.2f}ms")
        
        # Should be under 100ms even on CPU
        assert avg_time < 1000  # Generous limit for CPU
        
    def test_batch_inference_efficiency(self, vision_service):
        """Test batch inference provides reasonable per-image throughput on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available - batch efficiency test requires GPU")
            
        # Use more images for better batch efficiency measurement
        images = [Image.new("RGB", (256, 256), color=f"#{i:02x}{i:02x}{i:02x}") 
                  for i in range(0, 255, 25)]
        
        # Warmup runs (important for GPU kernel compilation)
        for _ in range(3):
            vision_service.classify_image(images[0])
            vision_service.classify_batch(images[:2])
        
        # Clear GPU cache before benchmark
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Time batch inference
        start = time.time()
        torch.cuda.synchronize()
        vision_service.classify_batch(images)
        torch.cuda.synchronize()
        batch_time = (time.time() - start) * 1000
        
        per_image_time = batch_time / len(images)
        
        print(f"Batch inference ({len(images)} images): {batch_time:.2f}ms")
        print(f"Per-image time (batch): {per_image_time:.2f}ms")
        print(f"Throughput: {1000 / per_image_time:.0f} images/second")
        
        # Performance target: less than 50ms per image in batch mode
        # (actual target is ~11ms on GPU, but allowing slack for test environment)
        assert per_image_time < 50, f"Per-image time {per_image_time:.2f}ms exceeds 50ms target"
            
    def test_memory_usage(self, vision_service):
        """Test GPU memory usage is within limits."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Check memory after model load
        memory_mb = torch.cuda.memory_allocated(vision_service.device) / 1e6
        print(f"GPU memory used: {memory_mb:.2f} MB")
        
        # ResNet50 should use less than 2GB
        assert memory_mb < 2000


class TestErrorHandling:
    """Tests for error handling."""
    
    @pytest.fixture
    def vision_service(self):
        """Create initialized vision service."""
        service = VisionService()
        service.initialize()
        return service
        
    def test_invalid_image_bytes(self, vision_service):
        """Test handling of invalid image bytes."""
        with pytest.raises(ValueError, match="Failed to load image"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                vision_service.analyze_image(b"not an image")
            )
            
    def test_batch_with_invalid_images(self, vision_service):
        """Test batch handling with some invalid images."""
        valid_image = Image.new("RGB", (256, 256), color="red")
        
        # Create an "invalid" image (too small)
        invalid_image = Image.new("RGB", (10, 10), color="blue")
        
        results = vision_service.classify_batch([valid_image, invalid_image, valid_image])
        
        assert len(results) == 3
        assert "predictions" in results[0]
        assert "error" in results[1]  # Invalid image should have error
        assert "predictions" in results[2]
        
    def test_clear_cuda_cache(self, vision_service):
        """Test CUDA cache clearing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
            
        # Should not raise any errors
        vision_service.clear_cuda_cache()
