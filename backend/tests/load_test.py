"""
Load Testing with Locust

This module provides load testing scenarios using Locust for:
- Concurrent user simulation
- Stress testing
- Performance profiling under load
- Identifying bottlenecks

Usage:
    locust -f tests/load_test.py --host http://localhost:8000
    
Or headless:
    locust -f tests/load_test.py --host http://localhost:8000 --headless -u 10 -r 2 -t 60s
"""

import io
import random
import time
from typing import List

from locust import HttpUser, task, between, events
from PIL import Image


# ============================================================================
# Test Image Generation
# ============================================================================

def create_test_image(
    width: int = 256,
    height: int = 256,
    color: str = "red"
) -> bytes:
    """Create a test image for uploading."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return buffer.getvalue()


# Pre-generate test images for efficiency
TEST_IMAGES = {
    "small": create_test_image(128, 128, "red"),
    "medium": create_test_image(256, 256, "green"),
    "large": create_test_image(512, 512, "blue"),
    "hd": create_test_image(1024, 1024, "purple"),
}

BATCH_IMAGES = [
    create_test_image(256, 256, f"#{i*20:02x}{i*15:02x}{i*10:02x}")
    for i in range(10)
]


# ============================================================================
# Load Test User Classes
# ============================================================================

class PhotoAnalyzerUser(HttpUser):
    """
    Standard user simulation for photo analyzer API.
    
    Simulates typical usage patterns with mixed single and batch requests.
    """
    
    wait_time = between(0.5, 2.0)  # Wait between 0.5-2 seconds between tasks
    
    def on_start(self):
        """Called when a user starts."""
        # Verify API is healthy
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Warning: Health check failed: {response.status_code}")
    
    @task(5)
    def analyze_single_image(self):
        """Analyze a single image (most common operation)."""
        # Randomly select image size
        image_key = random.choice(["small", "medium", "large"])
        image_data = TEST_IMAGES[image_key]
        
        files = {
            "file": (f"test_{image_key}.jpg", image_data, "image/jpeg")
        }
        
        with self.client.post(
            "/api/v1/analyze",
            files=files,
            catch_response=True,
            name="/api/v1/analyze"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Analysis failed: {data.get('error')}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def analyze_batch_small(self):
        """Analyze a small batch of images (3 images)."""
        files = [
            ("files", (f"batch_{i}.jpg", BATCH_IMAGES[i], "image/jpeg"))
            for i in range(3)
        ]
        
        with self.client.post(
            "/api/v1/batch-analyze",
            files=files,
            catch_response=True,
            name="/api/v1/batch-analyze (3 images)"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("successful") == 3:
                    response.success()
                else:
                    response.failure(f"Batch incomplete: {data.get('successful')}/3")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def analyze_batch_large(self):
        """Analyze a larger batch of images (10 images)."""
        files = [
            ("files", (f"batch_{i}.jpg", BATCH_IMAGES[i], "image/jpeg"))
            for i in range(10)
        ]
        
        with self.client.post(
            "/api/v1/batch-analyze",
            files=files,
            catch_response=True,
            name="/api/v1/batch-analyze (10 images)"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("successful") == 10:
                    response.success()
                else:
                    response.failure(f"Batch incomplete: {data.get('successful')}/10")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_categories(self):
        """Get available categories."""
        self.client.get("/api/v1/categories", name="/api/v1/categories")
    
    @task(1)
    def get_stats(self):
        """Get API statistics."""
        self.client.get("/api/v1/stats", name="/api/v1/stats")
    
    @task(1)
    def health_check(self):
        """Perform health check."""
        self.client.get("/health", name="/health")


class HeavyUser(HttpUser):
    """
    Heavy user simulation with more batch operations.
    
    Simulates power users who frequently use batch processing.
    """
    
    wait_time = between(0.1, 0.5)  # Faster operations
    
    @task(3)
    def analyze_batch_max(self):
        """Analyze maximum batch size (20 images)."""
        # Generate 20 images on the fly
        files = [
            ("files", (f"heavy_{i}.jpg", TEST_IMAGES["medium"], "image/jpeg"))
            for i in range(20)
        ]
        
        with self.client.post(
            "/api/v1/batch-analyze",
            files=files,
            catch_response=True,
            name="/api/v1/batch-analyze (20 images)"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("successful") == 20:
                    response.success()
                else:
                    response.failure(f"Batch incomplete: {data.get('successful')}/20")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def analyze_large_images(self):
        """Analyze large HD images."""
        files = {
            "file": ("hd_image.jpg", TEST_IMAGES["hd"], "image/jpeg")
        }
        
        self.client.post(
            "/api/v1/analyze",
            files=files,
            name="/api/v1/analyze (HD)"
        )
    
    @task(1)
    def rapid_single_requests(self):
        """Send rapid single image requests."""
        for _ in range(5):
            files = {
                "file": ("rapid.jpg", TEST_IMAGES["small"], "image/jpeg")
            }
            self.client.post(
                "/api/v1/analyze",
                files=files,
                name="/api/v1/analyze (rapid)"
            )


class ReadOnlyUser(HttpUser):
    """
    Read-only user simulation.
    
    Simulates users who primarily check status and metadata.
    """
    
    wait_time = between(1.0, 3.0)
    
    @task(3)
    def browse_categories(self):
        """Browse available categories."""
        self.client.get("/api/v1/categories")
    
    @task(2)
    def check_stats(self):
        """Check API statistics."""
        self.client.get("/api/v1/stats")
    
    @task(2)
    def check_health(self):
        """Check health status."""
        self.client.get("/health")
    
    @task(1)
    def check_gpu_status(self):
        """Check GPU status."""
        self.client.get("/api/v1/gpu-status")
    
    @task(1)
    def list_models(self):
        """List available models."""
        self.client.get("/api/v1/models")


class StressTestUser(HttpUser):
    """
    Stress test user for finding breaking points.
    
    Minimal wait times and maximum load operations.
    """
    
    wait_time = between(0.01, 0.1)  # Minimal wait
    
    @task(1)
    def stress_single(self):
        """Rapid single image analysis."""
        files = {
            "file": ("stress.jpg", TEST_IMAGES["medium"], "image/jpeg")
        }
        self.client.post("/api/v1/analyze", files=files)
    
    @task(1)
    def stress_batch(self):
        """Rapid batch analysis."""
        files = [
            ("files", (f"stress_{i}.jpg", TEST_IMAGES["small"], "image/jpeg"))
            for i in range(5)
        ]
        self.client.post("/api/v1/batch-analyze", files=files)


# ============================================================================
# Event Hooks for Metrics
# ============================================================================

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, **kwargs):
    """Log request metrics."""
    # Can add custom logging or metrics collection here
    pass


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("="*60)
    print("LOAD TEST STARTED")
    print("="*60)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.parsed_options.num_users if hasattr(environment.parsed_options, 'num_users') else 'N/A'}")
    print("="*60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("="*60)
    print("LOAD TEST COMPLETED")
    print("="*60)
    
    # Print summary statistics
    stats = environment.stats
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Failures: {stats.total.num_failures}")
    print(f"Avg response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Median response time: {stats.total.median_response_time}ms")
    print(f"95th percentile: {stats.total.get_response_time_percentile(0.95)}ms")
    print(f"99th percentile: {stats.total.get_response_time_percentile(0.99)}ms")
    print(f"Requests/s: {stats.total.current_rps:.2f}")
    print("="*60)


# ============================================================================
# Custom Test Scenarios (for programmatic use)
# ============================================================================

def run_quick_load_test(host: str = "http://localhost:8000", duration: int = 30):
    """
    Run a quick load test programmatically.
    
    Args:
        host: API host URL
        duration: Test duration in seconds
    """
    import subprocess
    
    cmd = [
        "locust",
        "-f", __file__,
        "--host", host,
        "--headless",
        "-u", "10",  # 10 users
        "-r", "2",   # Spawn 2 users/second
        "-t", f"{duration}s",
        "--csv", "load_test_results",
    ]
    
    subprocess.run(cmd)


def run_stress_test(host: str = "http://localhost:8000", max_users: int = 50):
    """
    Run a stress test to find breaking point.
    
    Args:
        host: API host URL
        max_users: Maximum number of concurrent users
    """
    import subprocess
    
    cmd = [
        "locust",
        "-f", __file__,
        "--host", host,
        "--headless",
        "-u", str(max_users),
        "-r", "5",   # Spawn 5 users/second
        "-t", "120s",
        "--csv", "stress_test_results",
        "--only-summary",
    ]
    
    subprocess.run(cmd)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        run_quick_load_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "stress":
        run_stress_test()
    else:
        print("Usage:")
        print("  locust -f tests/load_test.py --host http://localhost:8000")
        print("  python tests/load_test.py quick  # Quick 30s test")
        print("  python tests/load_test.py stress # Stress test")
