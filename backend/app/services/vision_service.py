"""
Vision Service - GPU-Accelerated Deep Learning Image Analysis for Construction Photos

This module provides a comprehensive vision service for analyzing construction site photos
using pre-trained deep learning models with GPU acceleration support.

Features:
- GPU detection and automatic device allocation with CPU fallback
- CUDA memory management for optimal GPU utilization
- EXIF orientation correction for mobile photos
- Single and batch inference with GPU optimization
- Construction-specific category mapping from ImageNet classes
- Confidence score calculation and post-processing
"""

import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ExifTags

from app.config import settings
from app.models.schemas import AnalysisResult, Classification

logger = logging.getLogger(__name__)


# ============================================================================
# Construction Category Definitions
# ============================================================================

class ConstructionCategory:
    """Construction-specific photo categories."""
    FOUNDATION_EXCAVATION = "Foundation & Excavation"
    FRAMING_STRUCTURE = "Framing & Structure"
    ROOFING = "Roofing"
    ELECTRICAL_PLUMBING = "Electrical & Plumbing"
    INTERIOR_FINISHING = "Interior Finishing"
    EXTERIOR_LANDSCAPING = "Exterior & Landscaping"
    SAFETY_EQUIPMENT = "Safety & Equipment"
    PROGRESS_DOCUMENTATION = "Progress Documentation"
    UNKNOWN = "Uncategorized"


# ImageNet class indices mapped to construction categories
# This mapping covers relevant ImageNet classes that may appear in construction photos
IMAGENET_TO_CONSTRUCTION: Dict[int, str] = {
    # Foundation & Excavation - earth moving, concrete
    524: ConstructionCategory.FOUNDATION_EXCAVATION,  # container, cargo container
    603: ConstructionCategory.FOUNDATION_EXCAVATION,  # golfcart (often used on sites)
    829: ConstructionCategory.FOUNDATION_EXCAVATION,  # streetcar
    
    # Framing & Structure - wood, steel, scaffolding
    581: ConstructionCategory.FRAMING_STRUCTURE,  # grille
    654: ConstructionCategory.FRAMING_STRUCTURE,  # minivan (construction vehicle)
    671: ConstructionCategory.FRAMING_STRUCTURE,  # moving van
    717: ConstructionCategory.FRAMING_STRUCTURE,  # pickup
    734: ConstructionCategory.FRAMING_STRUCTURE,  # pole
    757: ConstructionCategory.FRAMING_STRUCTURE,  # power drill
    
    # Roofing - shingles, tiles, equipment
    539: ConstructionCategory.ROOFING,  # dome
    560: ConstructionCategory.ROOFING,  # fire screen
    
    # Electrical & Plumbing - wires, pipes, fixtures
    536: ConstructionCategory.ELECTRICAL_PLUMBING,  # dock
    614: ConstructionCategory.ELECTRICAL_PLUMBING,  # geyser
    709: ConstructionCategory.ELECTRICAL_PLUMBING,  # pipe organ (pipes)
    828: ConstructionCategory.ELECTRICAL_PLUMBING,  # streetcar (wiring)
    
    # Interior Finishing - walls, flooring, fixtures
    508: ConstructionCategory.INTERIOR_FINISHING,  # computer keyboard
    527: ConstructionCategory.INTERIOR_FINISHING,  # desktop computer
    553: ConstructionCategory.INTERIOR_FINISHING,  # file
    557: ConstructionCategory.INTERIOR_FINISHING,  # filing cabinet
    609: ConstructionCategory.INTERIOR_FINISHING,  # gas pump
    621: ConstructionCategory.INTERIOR_FINISHING,  # grand piano
    673: ConstructionCategory.INTERIOR_FINISHING,  # mouse
    681: ConstructionCategory.INTERIOR_FINISHING,  # notebook (computer)
    737: ConstructionCategory.INTERIOR_FINISHING,  # pool table
    742: ConstructionCategory.INTERIOR_FINISHING,  # printer
    744: ConstructionCategory.INTERIOR_FINISHING,  # projector
    752: ConstructionCategory.INTERIOR_FINISHING,  # radiator
    782: ConstructionCategory.INTERIOR_FINISHING,  # screen
    851: ConstructionCategory.INTERIOR_FINISHING,  # television
    
    # Exterior & Landscaping - outdoor, gardens, fencing
    401: ConstructionCategory.EXTERIOR_LANDSCAPING,  # accordion
    517: ConstructionCategory.EXTERIOR_LANDSCAPING,  # construction site
    534: ConstructionCategory.EXTERIOR_LANDSCAPING,  # digital clock
    556: ConstructionCategory.EXTERIOR_LANDSCAPING,  # fence
    598: ConstructionCategory.EXTERIOR_LANDSCAPING,  # fountain
    627: ConstructionCategory.EXTERIOR_LANDSCAPING,  # greenhouse
    648: ConstructionCategory.EXTERIOR_LANDSCAPING,  # mailbox
    669: ConstructionCategory.EXTERIOR_LANDSCAPING,  # mosquito net
    694: ConstructionCategory.EXTERIOR_LANDSCAPING,  # paddle
    728: ConstructionCategory.EXTERIOR_LANDSCAPING,  # plastic bag
    811: ConstructionCategory.EXTERIOR_LANDSCAPING,  # space heater
    920: ConstructionCategory.EXTERIOR_LANDSCAPING,  # traffic light
    
    # Safety & Equipment - hardhats, vests, tools
    518: ConstructionCategory.SAFETY_EQUIPMENT,  # crane
    545: ConstructionCategory.SAFETY_EQUIPMENT,  # electric fan
    587: ConstructionCategory.SAFETY_EQUIPMENT,  # hammer
    596: ConstructionCategory.SAFETY_EQUIPMENT,  # forklift
    607: ConstructionCategory.SAFETY_EQUIPMENT,  # gasmask
    676: ConstructionCategory.SAFETY_EQUIPMENT,  # nail
    688: ConstructionCategory.SAFETY_EQUIPMENT,  # oxygen mask
    708: ConstructionCategory.SAFETY_EQUIPMENT,  # pickup truck
    726: ConstructionCategory.SAFETY_EQUIPMENT,  # plane
    740: ConstructionCategory.SAFETY_EQUIPMENT,  # power saw
    753: ConstructionCategory.SAFETY_EQUIPMENT,  # radiotelescope
    795: ConstructionCategory.SAFETY_EQUIPMENT,  # ski
    800: ConstructionCategory.SAFETY_EQUIPMENT,  # slot
    817: ConstructionCategory.SAFETY_EQUIPMENT,  # stage
    864: ConstructionCategory.SAFETY_EQUIPMENT,  # tow truck
    867: ConstructionCategory.SAFETY_EQUIPMENT,  # tractor
    895: ConstructionCategory.SAFETY_EQUIPMENT,  # warplane
}

# Keywords for fallback category detection from ImageNet labels
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    ConstructionCategory.FOUNDATION_EXCAVATION: [
        "excavator", "bulldozer", "crane", "concrete", "cement",
        "dirt", "soil", "gravel", "foundation", "hole", "trench",
        "loader", "backhoe", "digger"
    ],
    ConstructionCategory.FRAMING_STRUCTURE: [
        "scaffold", "beam", "steel", "frame", "wood", "timber",
        "lumber", "truss", "column", "girder", "structure"
    ],
    ConstructionCategory.ROOFING: [
        "roof", "shingle", "tile", "gutter", "chimney", "vent",
        "skylight", "attic", "rafter", "eave"
    ],
    ConstructionCategory.ELECTRICAL_PLUMBING: [
        "wire", "cable", "pipe", "conduit", "electrical", "plumbing",
        "outlet", "switch", "panel", "duct", "hvac", "valve", "faucet"
    ],
    ConstructionCategory.INTERIOR_FINISHING: [
        "wall", "drywall", "floor", "tile", "paint", "carpet",
        "cabinet", "counter", "door", "window", "trim", "ceiling"
    ],
    ConstructionCategory.EXTERIOR_LANDSCAPING: [
        "fence", "gate", "siding", "brick", "stone", "patio",
        "deck", "garden", "lawn", "driveway", "walkway", "landscap"
    ],
    ConstructionCategory.SAFETY_EQUIPMENT: [
        "helmet", "hardhat", "vest", "safety", "harness", "goggles",
        "gloves", "boots", "cone", "barrier", "sign", "tool", "equipment"
    ],
    ConstructionCategory.PROGRESS_DOCUMENTATION: [
        "building", "construction", "site", "project", "work",
        "progress", "photo", "document"
    ],
}


# ============================================================================
# ImageNet Labels Loader
# ============================================================================

def load_imagenet_labels() -> Dict[int, str]:
    """
    Load ImageNet class labels.
    
    Returns a mapping of class index to human-readable label.
    Falls back to a bundled subset if external file not available.
    """
    # Try to load from external file first
    labels_path = Path(__file__).parent.parent / "data" / "imagenet_labels.json"
    
    if labels_path.exists():
        try:
            with open(labels_path, "r") as f:
                labels = json.load(f)
            logger.info(f"Loaded {len(labels)} ImageNet labels from {labels_path}")
            return {int(k): v for k, v in labels.items()}
        except Exception as e:
            logger.warning(f"Failed to load labels from file: {e}")
    
    # Use torchvision's built-in labels if available (ResNet50 weights include them)
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        categories = weights.meta.get("categories", [])
        if categories:
            labels = {i: label for i, label in enumerate(categories)}
            logger.info(f"Loaded {len(labels)} ImageNet labels from model weights")
            return labels
    except Exception as e:
        logger.warning(f"Failed to load labels from model weights: {e}")
    
    # Fallback to empty dict - will use class indices
    logger.warning("No ImageNet labels available, using class indices")
    return {}


# ============================================================================
# Image Processing Utilities
# ============================================================================

def fix_exif_orientation(image: Image.Image) -> Image.Image:
    """
    Correct image orientation based on EXIF metadata.
    
    Mobile phones often store images in landscape orientation with EXIF rotation tags.
    This function applies the correct rotation to display the image properly.
    
    Args:
        image: PIL Image that may have EXIF orientation data
        
    Returns:
        Properly oriented PIL Image
    """
    try:
        # Get EXIF data
        exif = image._getexif()
        if exif is None:
            return image
        
        # Find orientation tag
        orientation_key = None
        for key, value in ExifTags.TAGS.items():
            if value == "Orientation":
                orientation_key = key
                break
        
        if orientation_key is None or orientation_key not in exif:
            return image
        
        orientation = exif[orientation_key]
        
        # Apply rotation based on orientation value
        rotation_map = {
            3: 180,
            6: 270,
            8: 90,
        }
        
        if orientation in rotation_map:
            image = image.rotate(rotation_map[orientation], expand=True)
            logger.debug(f"Applied EXIF rotation: {rotation_map[orientation]}°")
        
        # Handle mirroring (orientations 2, 4, 5, 7)
        if orientation in [2, 4]:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation in [5, 7]:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
    except Exception as e:
        logger.debug(f"Could not process EXIF orientation: {e}")
    
    return image


def validate_image(image: Image.Image, min_size: int = 32, max_size: int = 10000) -> Tuple[bool, str]:
    """
    Validate image dimensions and format.
    
    Args:
        image: PIL Image to validate
        min_size: Minimum dimension size
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    width, height = image.size
    
    if width < min_size or height < min_size:
        return False, f"Image too small: {width}x{height} (minimum: {min_size}x{min_size})"
    
    if width > max_size or height > max_size:
        return False, f"Image too large: {width}x{height} (maximum: {max_size}x{max_size})"
    
    if image.mode not in ["RGB", "RGBA", "L"]:
        return False, f"Unsupported image mode: {image.mode}"
    
    return True, ""


# ============================================================================
# Vision Service Class
# ============================================================================

class VisionService:
    """
    GPU-accelerated vision service for construction photo classification.
    
    This service uses pre-trained deep learning models (ResNet50/101, EfficientNet)
    to classify images and map them to construction-specific categories.
    
    Features:
    - Automatic GPU/CPU device selection
    - CUDA memory management
    - Single and batch inference
    - EXIF orientation correction
    - Construction category mapping
    
    Performance Targets:
    - Single image inference: ~50ms on GPU
    - Batch inference (10 images): ~11ms per image
    - GPU memory usage: <2GB for ResNet50
    - Throughput: 90+ images/second in batch mode
    """
    
    def __init__(self):
        """Initialize the vision service (lazy loading - model not loaded until first use)."""
        self.model: Optional[torch.nn.Module] = None
        self.device: torch.device = torch.device("cpu")
        self.model_name: str = settings.MODEL_NAME
        self.transform: Optional[transforms.Compose] = None
        self._initialized: bool = False
        self._imagenet_labels: Dict[int, str] = {}
        
        # CUDA memory management settings
        self._cuda_memory_fraction: float = 0.8  # Use up to 80% of GPU memory
        self._enable_cudnn_benchmark: bool = True
        
    def initialize(self) -> None:
        """
        Initialize the model and configure device.
        
        This method handles:
        - GPU detection and device allocation
        - CUDA memory management configuration
        - Model loading with appropriate weights
        - Image transformation pipeline setup
        - ImageNet labels loading
        """
        if self._initialized:
            return
            
        logger.info(f"🔧 Initializing Vision Service with model: {self.model_name}")
        
        # Configure device and CUDA settings
        self._configure_device()
        
        # Load model
        self._load_model()
        
        # Setup image preprocessing pipeline
        self._setup_transforms()
        
        # Load ImageNet labels for human-readable output
        self._imagenet_labels = load_imagenet_labels()
        
        self._initialized = True
        logger.info("✅ Vision Service initialized successfully")
        
    def _configure_device(self) -> None:
        """
        Configure compute device (GPU/CPU) with appropriate settings.
        
        Handles:
        - GPU detection and availability check
        - CUDA memory management
        - cuDNN optimization settings
        - Fallback to CPU if GPU unavailable
        """
        if settings.USE_GPU and torch.cuda.is_available():
            try:
                device_id = int(settings.CUDA_VISIBLE_DEVICES.split(",")[0])
                self.device = torch.device(f"cuda:{device_id}")
                
                # Get GPU info
                gpu_name = torch.cuda.get_device_name(self.device)
                gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1e9
                
                logger.info(f"🎮 Using GPU: {gpu_name}")
                logger.info(f"📊 GPU Memory: {gpu_memory:.2f} GB")
                
                # Configure CUDA memory management
                self._configure_cuda_memory()
                
                # Enable cuDNN auto-tuning for optimal performance
                if self._enable_cudnn_benchmark:
                    torch.backends.cudnn.benchmark = True
                    logger.info("⚡ cuDNN benchmark mode enabled")
                    
            except Exception as e:
                logger.warning(f"Failed to configure GPU: {e}. Falling back to CPU.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            if settings.USE_GPU:
                logger.warning("💻 GPU requested but CUDA not available, using CPU")
            else:
                logger.info("💻 Using CPU (GPU disabled in settings)")
                
    def _configure_cuda_memory(self) -> None:
        """
        Configure CUDA memory management for optimal performance.
        
        Settings applied:
        - Memory fraction limit to prevent OOM errors
        - Enable memory caching for faster allocation
        """
        try:
            # Set memory fraction (prevents using all GPU memory)
            torch.cuda.set_per_process_memory_fraction(
                self._cuda_memory_fraction, 
                self.device
            )
            
            # Clear any cached memory
            torch.cuda.empty_cache()
            
            logger.info(f"📊 CUDA memory fraction set to {self._cuda_memory_fraction:.0%}")
            
        except Exception as e:
            logger.warning(f"Could not configure CUDA memory: {e}")
            
    def _load_model(self) -> None:
        """
        Load the specified pre-trained model.
        
        Supported models:
        - resnet50: Good balance of speed and accuracy
        - resnet101: Higher accuracy, slower inference
        - efficientnet_b0: Best efficiency, smaller memory footprint
        """
        model_loaders = {
            "resnet50": lambda: models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2
            ),
            "resnet101": lambda: models.resnet101(
                weights=models.ResNet101_Weights.IMAGENET1K_V2
            ),
            "efficientnet_b0": lambda: models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            ),
        }
        
        if self.model_name not in model_loaders:
            available = ", ".join(model_loaders.keys())
            raise ValueError(f"Unknown model: {self.model_name}. Available: {available}")
        
        logger.info(f"📦 Loading model: {self.model_name}")
        
        # Load model
        self.model = model_loaders[self.model_name]()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Set to evaluation mode (disables dropout, batch norm uses running stats)
        self.model.eval()
        
        # Log model info
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"📦 Model {self.model_name} loaded ({param_count:.1f}M parameters)")
        
        # Log GPU memory usage after loading
        if self.device.type == "cuda":
            memory_allocated = torch.cuda.memory_allocated(self.device) / 1e9
            logger.info(f"📊 GPU memory after model load: {memory_allocated:.2f} GB")
            
    def _setup_transforms(self) -> None:
        """
        Setup image preprocessing pipeline.
        
        Standard ImageNet preprocessing:
        - Resize to 256x256
        - Center crop to 224x224
        - Convert to tensor
        - Normalize with ImageNet mean/std
        """
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        logger.debug("Image transform pipeline configured")
        
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single image for model inference.
        
        Args:
            image: PIL Image in any mode
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Fix EXIF orientation
        image = fix_exif_orientation(image)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Apply transforms
        return self.transform(image)
        
    def _get_imagenet_label(self, class_idx: int) -> str:
        """
        Get human-readable label for ImageNet class index.
        
        Args:
            class_idx: ImageNet class index (0-999)
            
        Returns:
            Human-readable class label
        """
        if class_idx in self._imagenet_labels:
            return self._imagenet_labels[class_idx]
        return f"class_{class_idx}"
        
    def _map_to_construction_category(self, class_idx: int, label: str = "") -> str:
        """
        Map ImageNet class to construction-specific category.
        
        Uses two-tier matching:
        1. Direct class index mapping for known relevant classes
        2. Keyword-based matching on class labels as fallback
        
        Args:
            class_idx: ImageNet class index
            label: Human-readable class label
            
        Returns:
            Construction category string
        """
        # First check direct mapping
        if class_idx in IMAGENET_TO_CONSTRUCTION:
            return IMAGENET_TO_CONSTRUCTION[class_idx]
        
        # Fallback to keyword matching on label
        label_lower = label.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in label_lower:
                    return category
        
        return ConstructionCategory.PROGRESS_DOCUMENTATION
        
    def _calculate_confidence_scores(
        self, 
        probabilities: torch.Tensor,
        top_k: int = 5
    ) -> Tuple[List[float], List[int]]:
        """
        Calculate top-k confidence scores from model output.
        
        Args:
            probabilities: Softmax probabilities tensor
            top_k: Number of top predictions to return
            
        Returns:
            Tuple of (confidence_scores, class_indices)
        """
        top_prob, top_idx = torch.topk(probabilities, min(top_k, len(probabilities)))
        return top_prob.cpu().tolist(), top_idx.cpu().tolist()
        
    @torch.no_grad()
    def classify_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Classify a single image.
        
        Args:
            image: PIL Image to classify
            
        Returns:
            Dictionary containing:
            - predictions: List of {class, label, confidence} dicts
            - category: Construction category string
            - top_confidence: Highest confidence score
        """
        # Initialize if needed
        if not self._initialized:
            self.initialize()
            
        # Validate image
        is_valid, error = validate_image(image)
        if not is_valid:
            raise ValueError(error)
        
        # Preprocess
        input_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        
        # Run inference
        output = self.model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        
        # Get top predictions
        confidences, indices = self._calculate_confidence_scores(probabilities)
        
        # Build predictions list
        predictions = []
        for conf, idx in zip(confidences, indices):
            label = self._get_imagenet_label(idx)
            predictions.append({
                "class": idx,
                "label": label,
                "confidence": round(conf, 4)
            })
        
        # Determine construction category from top prediction
        top_label = predictions[0]["label"] if predictions else ""
        category = self._map_to_construction_category(
            indices[0] if indices else 0,
            top_label
        )
        
        return {
            "predictions": predictions,
            "category": category,
            "top_confidence": predictions[0]["confidence"] if predictions else 0.0
        }
        
    @torch.no_grad()
    def classify_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Classify multiple images in batch for GPU optimization.
        
        Batch processing provides significant speedup on GPU by:
        - Amortizing data transfer overhead
        - Maximizing GPU utilization
        - Enabling parallel computation
        
        Args:
            images: List of PIL Images to classify
            
        Returns:
            List of classification results (same format as classify_image)
        """
        if not images:
            return []
            
        # Initialize if needed
        if not self._initialized:
            self.initialize()
            
        # Validate and preprocess all images
        tensors = []
        valid_indices = []
        results = [None] * len(images)
        
        for i, image in enumerate(images):
            try:
                is_valid, error = validate_image(image)
                if not is_valid:
                    results[i] = {
                        "predictions": [],
                        "category": ConstructionCategory.UNKNOWN,
                        "error": error
                    }
                    continue
                    
                tensors.append(self._preprocess_image(image))
                valid_indices.append(i)
                
            except Exception as e:
                results[i] = {
                    "predictions": [],
                    "category": ConstructionCategory.UNKNOWN,
                    "error": str(e)
                }
        
        if not tensors:
            return results
            
        # Stack into batch tensor and move to device
        batch_tensor = torch.stack(tensors).to(self.device)
        
        # Run batch inference
        outputs = self.model(batch_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Process each result
        for batch_idx, original_idx in enumerate(valid_indices):
            probs = probabilities[batch_idx]
            confidences, indices = self._calculate_confidence_scores(probs)
            
            predictions = []
            for conf, idx in zip(confidences, indices):
                label = self._get_imagenet_label(idx)
                predictions.append({
                    "class": idx,
                    "label": label,
                    "confidence": round(conf, 4)
                })
            
            top_label = predictions[0]["label"] if predictions else ""
            category = self._map_to_construction_category(
                indices[0] if indices else 0,
                top_label
            )
            
            results[original_idx] = {
                "predictions": predictions,
                "category": category,
                "top_confidence": predictions[0]["confidence"] if predictions else 0.0
            }
        
        return results
        
    async def analyze_image(self, image_data: bytes) -> AnalysisResult:
        """
        Analyze an image and return classification results.
        
        This is the main API method that integrates with FastAPI routes.
        
        Args:
            image_data: Raw image bytes (JPEG, PNG, WebP)
            
        Returns:
            AnalysisResult with classifications, category, and metadata
        """
        # Initialize if needed
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Load image from bytes
        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
            
        original_size = image.size
        
        # Fix EXIF orientation
        image = fix_exif_orientation(image)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Validate image
        is_valid, error = validate_image(image)
        if not is_valid:
            raise ValueError(error)
        
        # Preprocess and run inference
        input_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs[0], dim=0)
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        # Build classifications list
        classifications = []
        for prob, idx in zip(top5_prob.cpu().numpy(), top5_indices.cpu().numpy()):
            label = self._get_imagenet_label(int(idx))
            classifications.append(Classification(
                label=label,
                confidence=float(prob)
            ))
        
        # Determine construction category
        top_class = int(top5_indices[0].cpu().item())
        top_label = classifications[0].label if classifications else ""
        construction_category = self._map_to_construction_category(top_class, top_label)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return AnalysisResult(
            classifications=classifications,
            model_used=self.model_name,
            processing_time_ms=round(processing_time, 2),
            image_size=original_size,
            metadata={
                "device": str(self.device),
                "input_size": [224, 224],
                "construction_category": construction_category,
                "cuda_memory_mb": round(
                    torch.cuda.memory_allocated(self.device) / 1e6, 2
                ) if self.device.type == "cuda" else 0,
            }
        )
        
    async def analyze_batch(
        self, 
        images_data: List[bytes]
    ) -> Tuple[List[AnalysisResult], float]:
        """
        Analyze multiple images using GPU-optimized batch processing.
        
        Args:
            images_data: List of raw image bytes
            
        Returns:
            Tuple of (list of AnalysisResults, total processing time in ms)
        """
        if not images_data:
            return [], 0.0
            
        # Initialize if needed
        if not self._initialized:
            self.initialize()
            
        start_time = time.time()
        
        # Load all images
        images = []
        original_sizes = []
        load_errors = []
        
        for i, data in enumerate(images_data):
            try:
                image = Image.open(io.BytesIO(data))
                image = fix_exif_orientation(image)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)
                original_sizes.append(image.size)
                load_errors.append(None)
            except Exception as e:
                images.append(None)
                original_sizes.append((0, 0))
                load_errors.append(str(e))
        
        # Run batch classification
        valid_images = [img for img in images if img is not None]
        batch_results = self.classify_batch(valid_images) if valid_images else []
        
        # Map results back to original indices
        results = []
        valid_idx = 0
        
        for i, error in enumerate(load_errors):
            if error is not None:
                # Image failed to load
                results.append(AnalysisResult(
                    classifications=[],
                    model_used=self.model_name,
                    processing_time_ms=0.0,
                    image_size=(0, 0),
                    metadata={"error": error}
                ))
            else:
                # Get result from batch
                result = batch_results[valid_idx]
                valid_idx += 1
                
                if "error" in result:
                    results.append(AnalysisResult(
                        classifications=[],
                        model_used=self.model_name,
                        processing_time_ms=0.0,
                        image_size=original_sizes[i],
                        metadata={"error": result["error"]}
                    ))
                else:
                    classifications = [
                        Classification(
                            label=p["label"],
                            confidence=p["confidence"]
                        )
                        for p in result["predictions"]
                    ]
                    results.append(AnalysisResult(
                        classifications=classifications,
                        model_used=self.model_name,
                        processing_time_ms=0.0,  # Will be filled below
                        image_size=original_sizes[i],
                        metadata={
                            "device": str(self.device),
                            "input_size": [224, 224],
                            "construction_category": result["category"],
                        }
                    ))
        
        total_time = (time.time() - start_time) * 1000
        
        # Update processing times (distributed evenly for batch)
        per_image_time = total_time / len(results) if results else 0.0
        for result in results:
            result.processing_time_ms = round(per_image_time, 2)
        
        return results, round(total_time, 2)
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model and device.
        
        Returns:
            Dictionary with model configuration and resource usage
        """
        if not self._initialized:
            self.initialize()
            
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "initialized": self._initialized,
        }
        
        if self.device.type == "cuda":
            info.update({
                "gpu_name": torch.cuda.get_device_name(self.device),
                "gpu_memory_allocated_mb": round(
                    torch.cuda.memory_allocated(self.device) / 1e6, 2
                ),
                "gpu_memory_reserved_mb": round(
                    torch.cuda.memory_reserved(self.device) / 1e6, 2
                ),
                "gpu_memory_total_mb": round(
                    torch.cuda.get_device_properties(self.device).total_memory / 1e6, 2
                ),
            })
        
        if self.model is not None:
            info["model_parameters"] = sum(
                p.numel() for p in self.model.parameters()
            )
            
        return info
        
    def clear_cuda_cache(self) -> None:
        """Clear CUDA memory cache to free up GPU memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
            
    def get_construction_categories(self) -> List[str]:
        """Get list of all construction categories."""
        return [
            ConstructionCategory.FOUNDATION_EXCAVATION,
            ConstructionCategory.FRAMING_STRUCTURE,
            ConstructionCategory.ROOFING,
            ConstructionCategory.ELECTRICAL_PLUMBING,
            ConstructionCategory.INTERIOR_FINISHING,
            ConstructionCategory.EXTERIOR_LANDSCAPING,
            ConstructionCategory.SAFETY_EQUIPMENT,
            ConstructionCategory.PROGRESS_DOCUMENTATION,
        ]


# ============================================================================
# Singleton Instance
# ============================================================================

# Global singleton instance for use across the application
vision_service = VisionService()
