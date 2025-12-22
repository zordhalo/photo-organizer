"""
Vision Service - Deep Learning Image Analysis
"""

import io
import time
from typing import Optional

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from app.config import settings
from app.models.schemas import AnalysisResult, Classification


# ImageNet class labels (subset for demo - full list would be loaded from file)
IMAGENET_LABELS = {
    0: "tench",
    1: "goldfish", 
    # ... more labels would be loaded from a file in production
}


class VisionService:
    """Service for image analysis using deep learning models."""
    
    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.device: torch.device = torch.device("cpu")
        self.model_name: str = settings.MODEL_NAME
        self.transform: Optional[transforms.Compose] = None
        self._initialized: bool = False
        
    def initialize(self) -> None:
        """Initialize the model and move to appropriate device."""
        if self._initialized:
            return
            
        print(f"🔧 Initializing Vision Service with model: {self.model_name}")
        
        # Set device
        if settings.USE_GPU and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{settings.CUDA_VISIBLE_DEVICES}")
            print(f"🎮 Using GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU")
        
        # Load model
        self._load_model()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self._initialized = True
        print("✅ Vision Service initialized successfully")
    
    def _load_model(self) -> None:
        """Load the specified model."""
        model_loaders = {
            "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            "resnet101": lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1),
            "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        }
        
        if self.model_name not in model_loaders:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        self.model = model_loaders[self.model_name]()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"📦 Model {self.model_name} loaded successfully")
    
    async def analyze_image(self, image_data: bytes) -> AnalysisResult:
        """
        Analyze an image and return classification results.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            AnalysisResult with classifications and metadata
        """
        # Initialize if needed
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        original_size = image.size
        
        # Transform image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        classifications = []
        for prob, idx in zip(top5_prob.cpu().numpy(), top5_indices.cpu().numpy()):
            # In production, load actual ImageNet labels
            label = f"class_{idx}"  # Placeholder - use actual labels
            classifications.append(Classification(
                label=label,
                confidence=float(prob)
            ))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return AnalysisResult(
            classifications=classifications,
            model_used=self.model_name,
            processing_time_ms=round(processing_time, 2),
            image_size=original_size,
            metadata={
                "device": str(self.device),
                "input_size": [224, 224],
            }
        )
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        if not self._initialized:
            self.initialize()
            
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_mb": (
                torch.cuda.memory_allocated(self.device) / 1e6 
                if self.device.type == "cuda" else 0
            ),
        }


# Singleton instance
vision_service = VisionService()
