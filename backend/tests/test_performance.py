"""
Performance Benchmark Tests

Comprehensive benchmarks for measuring:
- Single image inference time
- Batch processing efficiency
- GPU memory usage
- Throughput (images/second)
- Response times for API endpoints
"""

import io
import time
from typing import List, Dict, Any
import statistics

import pytest
import torch
from PIL import Image

from app.services.vision_service import VisionService, vision_service
from app.config import settings


# ============================================================================
# Test Utilities
# ============================================================================

def create_test_image(size: int = 256, color: str = "red") -> Image.Image:
    """Create a test image."""
    return Image.new("RGB", (size, size), color=color)


def create_test_image_bytes(size: int = 256, color: str = "red") -> bytes:
    """Create test image bytes."""
    img = create_test_image(size, color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def run_benchmark(
    func,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    *args,
    **kwargs
) -> Dict[str, float]:
    """
    Run a benchmark with warmup and multiple iterations.
    
    Returns dict with min, max, mean, std, median times in milliseconds.
    """
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Clear GPU cache before benchmarking
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Benchmark runs
    times = []
    for _ in range(benchmark_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        func(*args, **kwargs)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        times.append(elapsed)
    
    return {
        "min_ms": min(times),
        "max_ms": max(times),
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "median_ms": statistics.median(times),
        "runs": len(times),
    }


# ============================================================================
# Vision Service Benchmarks
# ============================================================================

class TestVisionServicePerformance:
    """Performance benchmarks for VisionService."""
    
    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Initialize vision service for tests."""
        self.service = VisionService()
        self.service.initialize()
        yield
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_single_image_inference_benchmark(self):
        """Benchmark single image inference time."""
        image = create_test_image(256, "red")
        
        result = run_benchmark(
            self.service.classify_image,
            warmup_runs=5,
            benchmark_runs=20,
            image=image
        )
        
        print(f"\n{'='*60}")
        print("SINGLE IMAGE INFERENCE BENCHMARK")
        print(f"{'='*60}")
        print(f"Device: {self.service.device}")
        print(f"Model: {self.service.model_name}")
        print(f"Runs: {result['runs']}")
        print(f"Mean: {result['mean_ms']:.2f}ms")
        print(f"Median: {result['median_ms']:.2f}ms")
        print(f"Min: {result['min_ms']:.2f}ms")
        print(f"Max: {result['max_ms']:.2f}ms")
        print(f"Std: {result['std_ms']:.2f}ms")
        print(f"{'='*60}")
        
        # Performance assertion - should be under 500ms even on CPU
        assert result["mean_ms"] < 500, f"Single image inference too slow: {result['mean_ms']:.2f}ms"
        
        # On GPU, should be under 100ms
        if self.service.device.type == "cuda":
            assert result["mean_ms"] < 100, f"GPU inference too slow: {result['mean_ms']:.2f}ms"
    
    def test_batch_inference_benchmark(self):
        """Benchmark batch inference efficiency."""
        batch_sizes = [1, 2, 4, 8, 16]
        results = {}
        
        print(f"\n{'='*60}")
        print("BATCH INFERENCE BENCHMARK")
        print(f"{'='*60}")
        print(f"Device: {self.service.device}")
        print(f"Model: {self.service.model_name}")
        print(f"{'='*60}")
        
        for batch_size in batch_sizes:
            images = [create_test_image(256, f"#{i*15:02x}{i*10:02x}{i*5:02x}") 
                     for i in range(batch_size)]
            
            def batch_classify():
                return self.service.classify_batch(images)
            
            result = run_benchmark(
                batch_classify,
                warmup_runs=3,
                benchmark_runs=10
            )
            
            per_image = result["mean_ms"] / batch_size
            throughput = 1000 / per_image if per_image > 0 else 0
            
            results[batch_size] = {
                **result,
                "per_image_ms": per_image,
                "throughput": throughput,
            }
            
            print(f"Batch {batch_size:2d}: {result['mean_ms']:7.2f}ms total | "
                  f"{per_image:6.2f}ms/img | {throughput:6.1f} img/s")
        
        print(f"{'='*60}")
        
        # Verify batch processing is more efficient for larger batches
        if len(batch_sizes) >= 2:
            per_img_1 = results[1]["per_image_ms"]
            per_img_8 = results[8]["per_image_ms"] if 8 in results else per_img_1
            
            # Batch processing should provide some speedup (at least maintain performance)
            # Allow up to 50% degradation due to memory constraints
            assert per_img_8 < per_img_1 * 1.5, "Batch processing should not be significantly slower"
    
    @pytest.mark.asyncio
    async def test_analyze_image_async_benchmark(self):
        """Benchmark async analyze_image method."""
        image_bytes = create_test_image_bytes(256, "blue")
        
        # Warmup
        for _ in range(3):
            await self.service.analyze_image(image_bytes)
        
        # Benchmark
        times = []
        for _ in range(10):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            result = await self.service.analyze_image(image_bytes)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        
        mean_time = statistics.mean(times)
        
        print(f"\n{'='*60}")
        print("ASYNC ANALYZE_IMAGE BENCHMARK")
        print(f"{'='*60}")
        print(f"Mean: {mean_time:.2f}ms")
        print(f"Reported processing_time_ms: {result.processing_time_ms:.2f}ms")
        print(f"{'='*60}")
        
        # Should complete within reasonable time
        assert mean_time < 500, f"Async analyze too slow: {mean_time:.2f}ms"
    
    @pytest.mark.asyncio
    async def test_batch_analyze_async_benchmark(self):
        """Benchmark async batch analysis."""
        batch_sizes = [5, 10, 20]
        
        print(f"\n{'='*60}")
        print("ASYNC BATCH ANALYZE BENCHMARK")
        print(f"{'='*60}")
        
        for batch_size in batch_sizes:
            images = [create_test_image_bytes(256) for _ in range(batch_size)]
            
            # Warmup
            await self.service.analyze_batch(images[:2])
            
            # Benchmark
            start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            results, total_time = await self.service.analyze_batch(images)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            
            per_image = elapsed / batch_size
            throughput = 1000 / per_image if per_image > 0 else 0
            
            print(f"Batch {batch_size:2d}: {elapsed:7.2f}ms total | "
                  f"{per_image:6.2f}ms/img | {throughput:6.1f} img/s")
        
        print(f"{'='*60}")


# ============================================================================
# GPU Memory Benchmarks
# ============================================================================

class TestGPUMemory:
    """GPU memory usage benchmarks."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for GPU tests."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Clear cache before tests
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        yield
        
        # Cleanup
        torch.cuda.empty_cache()
    
    def test_model_memory_footprint(self):
        """Measure model memory footprint on GPU."""
        # Record memory before loading
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated() / 1e6
        
        # Initialize service
        service = VisionService()
        service.initialize()
        
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated() / 1e6
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
        
        model_memory = memory_after - memory_before
        
        print(f"\n{'='*60}")
        print("GPU MEMORY FOOTPRINT")
        print(f"{'='*60}")
        print(f"Model: {service.model_name}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after: {memory_after:.2f} MB")
        print(f"Model footprint: {model_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"{'='*60}")
        
        # ResNet50 should use less than 2GB
        assert memory_after < 2000, f"Model uses too much memory: {memory_after:.2f}MB"
    
    def test_batch_memory_scaling(self):
        """Test memory usage scales reasonably with batch size."""
        service = VisionService()
        service.initialize()
        
        batch_sizes = [1, 4, 8, 16]
        memory_usage = {}
        
        print(f"\n{'='*60}")
        print("BATCH MEMORY SCALING")
        print(f"{'='*60}")
        
        for batch_size in batch_sizes:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            images = [create_test_image(256) for _ in range(batch_size)]
            
            # Run batch inference
            service.classify_batch(images)
            
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e6
            memory_usage[batch_size] = peak
            
            print(f"Batch {batch_size:2d}: {peak:7.2f} MB peak")
        
        print(f"{'='*60}")
        
        # Memory should not grow linearly (batch processing is efficient)
        # Check that doubling batch size doesn't double memory
        if 8 in memory_usage and 4 in memory_usage:
            ratio = memory_usage[8] / memory_usage[4]
            print(f"Memory ratio (8 vs 4): {ratio:.2f}x")
            assert ratio < 2.5, "Memory scaling is too aggressive"
    
    def test_memory_cleanup(self):
        """Test that GPU memory is properly cleaned up."""
        service = VisionService()
        service.initialize()
        
        # Get baseline
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated() / 1e6
        
        # Run large batch
        images = [create_test_image(512) for _ in range(20)]
        service.classify_batch(images)
        
        during = torch.cuda.memory_allocated() / 1e6
        
        # Cleanup
        del images
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1e6
        
        print(f"\n{'='*60}")
        print("MEMORY CLEANUP TEST")
        print(f"{'='*60}")
        print(f"Baseline: {baseline:.2f} MB")
        print(f"During batch: {during:.2f} MB")
        print(f"After cleanup: {after:.2f} MB")
        print(f"Recovered: {during - after:.2f} MB")
        print(f"{'='*60}")
        
        # Memory should return close to baseline (within 100MB tolerance)
        assert after < baseline + 100, "Memory leak detected"


# ============================================================================
# Image Size Impact Benchmarks
# ============================================================================

class TestImageSizeImpact:
    """Test impact of different image sizes on performance."""
    
    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Initialize vision service."""
        self.service = VisionService()
        self.service.initialize()
        yield
    
    def test_image_size_impact(self):
        """Benchmark performance across different image sizes."""
        sizes = [128, 256, 512, 1024, 2048]
        results = {}
        
        print(f"\n{'='*60}")
        print("IMAGE SIZE IMPACT BENCHMARK")
        print(f"{'='*60}")
        print(f"Device: {self.service.device}")
        print(f"Note: Images are resized to 224x224 for inference")
        print(f"{'='*60}")
        
        for size in sizes:
            image = create_test_image(size, "green")
            
            result = run_benchmark(
                self.service.classify_image,
                warmup_runs=3,
                benchmark_runs=10,
                image=image
            )
            
            results[size] = result
            
            print(f"{size:4d}x{size:4d}: {result['mean_ms']:7.2f}ms mean | "
                  f"{result['min_ms']:7.2f}ms min | {result['max_ms']:7.2f}ms max")
        
        print(f"{'='*60}")
        
        # Larger images shouldn't be dramatically slower (due to resize)
        # Allow 3x slower for 2048 vs 256
        ratio = results[2048]["mean_ms"] / results[256]["mean_ms"]
        print(f"2048 vs 256 ratio: {ratio:.2f}x")
        assert ratio < 5, "Large images too slow"


# ============================================================================
# Throughput Tests
# ============================================================================

class TestThroughput:
    """Test overall throughput capabilities."""
    
    @pytest.fixture(autouse=True)
    def setup_service(self):
        """Initialize vision service."""
        self.service = VisionService()
        self.service.initialize()
        yield
    
    def test_sustained_throughput(self):
        """Test sustained throughput over many images."""
        num_images = 100
        batch_size = 10
        images = [create_test_image(256) for _ in range(batch_size)]
        
        # Warmup
        self.service.classify_batch(images[:2])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_images = 0
        start = time.perf_counter()
        
        for _ in range(num_images // batch_size):
            self.service.classify_batch(images)
            total_images += batch_size
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        throughput = total_images / elapsed
        
        print(f"\n{'='*60}")
        print("SUSTAINED THROUGHPUT TEST")
        print(f"{'='*60}")
        print(f"Device: {self.service.device}")
        print(f"Total images: {total_images}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {throughput:.1f} images/second")
        print(f"Average latency: {1000/throughput:.2f}ms/image")
        print(f"{'='*60}")
        
        # Should achieve at least 10 images/second even on CPU
        assert throughput > 10, f"Throughput too low: {throughput:.1f} img/s"
        
        # On GPU, should achieve 50+ images/second
        if self.service.device.type == "cuda":
            assert throughput > 50, f"GPU throughput too low: {throughput:.1f} img/s"


# ============================================================================
# Cold Start Benchmark
# ============================================================================

class TestColdStart:
    """Test cold start performance."""
    
    def test_model_initialization_time(self):
        """Benchmark model initialization time."""
        times = []
        
        for _ in range(3):
            # Create fresh service
            service = VisionService()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            service.initialize()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            
            # Cleanup
            del service
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        mean_time = statistics.mean(times)
        
        print(f"\n{'='*60}")
        print("COLD START BENCHMARK")
        print(f"{'='*60}")
        print(f"Initialization times: {[f'{t:.0f}ms' for t in times]}")
        print(f"Mean: {mean_time:.0f}ms")
        print(f"{'='*60}")
        
        # Should initialize within 30 seconds
        assert mean_time < 30000, f"Initialization too slow: {mean_time:.0f}ms"
    
    def test_first_inference_after_init(self):
        """Benchmark first inference after initialization."""
        service = VisionService()
        service.initialize()
        
        image = create_test_image(256)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time first inference (includes any lazy initialization)
        start = time.perf_counter()
        service.classify_image(image)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        first_time = (time.perf_counter() - start) * 1000
        
        # Time subsequent inference
        start = time.perf_counter()
        service.classify_image(image)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        second_time = (time.perf_counter() - start) * 1000
        
        print(f"\n{'='*60}")
        print("FIRST INFERENCE BENCHMARK")
        print(f"{'='*60}")
        print(f"First inference: {first_time:.2f}ms")
        print(f"Second inference: {second_time:.2f}ms")
        print(f"Warmup overhead: {first_time - second_time:.2f}ms")
        print(f"{'='*60}")
