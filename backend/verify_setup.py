#!/usr/bin/env python
"""
Environment Verification Script

Run this script to verify that all dependencies are correctly installed
and the GPU is properly configured.
"""

import sys


def check_python_version():
    """Check Python version."""
    print(f"Python Version: {sys.version}")
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 10:
        print("  ✅ Python version OK (3.10+)")
        return True
    else:
        print("  ❌ Python 3.10+ required")
        return False


def check_pytorch():
    """Check PyTorch installation and CUDA."""
    try:
        import torch
        import torchvision
        
        print(f"\nPyTorch Version: {torch.__version__}")
        print(f"Torchvision Version: {torchvision.__version__}")
        
        # CUDA check
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1e9
                print(f"\n  GPU {i}: {props.name}")
                print(f"    Memory: {memory_gb:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # Test GPU tensor operations
            print("\n  Testing GPU tensor operations...")
            x = torch.rand(5, 3, 224, 224).cuda()
            y = x * 2
            print(f"    ✅ Tensor operations on GPU working")
            print(f"    Test tensor device: {x.device}")
            
            # Clear memory
            del x, y
            torch.cuda.empty_cache()
            
        else:
            print("  ⚠️  GPU not available, will use CPU")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ PyTorch not installed: {e}")
        return False


def check_fastapi():
    """Check FastAPI installation."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        
        print(f"\nFastAPI Version: {fastapi.__version__}")
        print(f"Uvicorn Version: {uvicorn.__version__}")
        print(f"Pydantic Version: {pydantic.__version__}")
        print("  ✅ Web framework packages OK")
        return True
        
    except ImportError as e:
        print(f"  ❌ FastAPI packages not installed: {e}")
        return False


def check_image_processing():
    """Check image processing libraries."""
    try:
        from PIL import Image
        import numpy as np
        
        print(f"\nPillow Version: {Image.__version__}")
        print(f"NumPy Version: {np.__version__}")
        
        # Test image creation
        img = Image.new("RGB", (224, 224), color="red")
        arr = np.array(img)
        print(f"  ✅ Image processing OK (test image shape: {arr.shape})")
        return True
        
    except ImportError as e:
        print(f"  ❌ Image processing libraries not installed: {e}")
        return False


def check_testing_tools():
    """Check testing tools."""
    try:
        import pytest
        import httpx
        
        print(f"\nPytest Version: {pytest.__version__}")
        print(f"HTTPX Version: {httpx.__version__}")
        print("  ✅ Testing tools OK")
        return True
        
    except ImportError as e:
        print(f"  ❌ Testing tools not installed: {e}")
        return False


def check_dev_tools():
    """Check development tools."""
    try:
        import black
        import flake8
        
        print(f"\nBlack Version: {black.__version__}")
        print("Flake8: installed")
        print("  ✅ Development tools OK")
        return True
        
    except ImportError as e:
        print(f"  ⚠️  Some dev tools not installed: {e}")
        return True  # Not critical


def run_model_test():
    """Test loading a model."""
    try:
        import torch
        from torchvision import models
        
        print("\n" + "=" * 50)
        print("Model Loading Test")
        print("=" * 50)
        
        print("Loading ResNet50 model...")
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        if torch.cuda.is_available():
            model = model.cuda()
            device = "cuda"
        else:
            device = "cpu"
        
        model.eval()
        
        # Test inference
        print(f"Running test inference on {device}...")
        dummy_input = torch.rand(1, 3, 224, 224)
        if device == "cuda":
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  Output shape: {output.shape}")
        print(f"  ✅ Model inference successful on {device.upper()}")
        
        # Memory info
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / 1e6
            print(f"  GPU Memory used: {memory_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 50)
    print("🔍 Construction Photo Analyzer")
    print("   Environment Verification")
    print("=" * 50)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("PyTorch & CUDA", check_pytorch()))
    results.append(("FastAPI", check_fastapi()))
    results.append(("Image Processing", check_image_processing()))
    results.append(("Testing Tools", check_testing_tools()))
    results.append(("Dev Tools", check_dev_tools()))
    results.append(("Model Loading", run_model_test()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Verification Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All checks passed! Environment is ready.")
    else:
        print("⚠️  Some checks failed. Please review the output above.")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
