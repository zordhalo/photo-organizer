"""
Vision Service Tests
"""

import pytest
import torch


class TestVisionService:
    """Tests for VisionService."""
    
    def test_cuda_availability(self):
        """Test CUDA is available (if GPU machine)."""
        # This test will pass regardless - just documents status
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
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
    
    def test_model_loading(self):
        """Test that model can be loaded."""
        from app.services.vision_service import VisionService
        
        service = VisionService()
        service.initialize()
        
        assert service.model is not None
        assert service._initialized is True
    
    @pytest.mark.asyncio
    async def test_image_analysis(self):
        """Test image analysis with a synthetic image."""
        from app.services.vision_service import VisionService
        from PIL import Image
        import io
        
        # Create a synthetic test image
        image = Image.new("RGB", (256, 256), color="red")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_data = buffer.getvalue()
        
        # Analyze
        service = VisionService()
        result = await service.analyze_image(image_data)
        
        assert result is not None
        assert len(result.classifications) > 0
        assert result.processing_time_ms > 0
        assert result.image_size == (256, 256)
