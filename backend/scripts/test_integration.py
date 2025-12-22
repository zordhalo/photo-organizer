"""
Interactive Integration Testing Script

This script provides manual testing capabilities for the Construction Photo Analyzer
including performance benchmarking, error scenario testing, and report generation.

Usage:
    python scripts/test_integration.py --all           # Run all tests
    python scripts/test_integration.py --quick         # Quick smoke test
    python scripts/test_integration.py --performance   # Performance benchmarks
    python scripts/test_integration.py --errors        # Error scenario testing
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    from PIL import Image
except ImportError:
    print("Missing dependencies. Install with: pip install httpx pillow")
    sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TestConfig:
    base_url: str = "http://localhost:8000"
    api_prefix: str = "/api/v1"
    timeout: float = 30.0
    
    @property
    def api_url(self) -> str:
        return f"{self.base_url}{self.api_prefix}"


@dataclass
class TestResult:
    name: str
    passed: bool
    duration_ms: float
    message: str = ""
    details: Dict = field(default_factory=dict)


@dataclass
class TestReport:
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    results: List[TestResult] = field(default_factory=list)
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def total_duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_image(
    width: int = 224,
    height: int = 224,
    color: Tuple[int, int, int] = (128, 128, 128),
    format: str = "JPEG"
) -> bytes:
    """Create a test image."""
    img = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format=format, quality=85)
    buffer.seek(0)
    return buffer.getvalue()


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def print_result(result: TestResult):
    """Print a single test result."""
    icon = "✅" if result.passed else "❌"
    print(f"  {icon} {result.name}: {result.message} ({result.duration_ms:.0f}ms)")


def print_report(report: TestReport):
    """Print the final test report."""
    report.ended_at = datetime.now()
    duration = (report.ended_at - report.started_at).total_seconds()
    
    print_header("TEST REPORT")
    print(f"  Started:  {report.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Finished: {report.ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {duration:.2f}s")
    print()
    print(f"  Total:    {report.total}")
    print(f"  Passed:   {report.passed}")
    print(f"  Failed:   {report.failed}")
    print()
    
    if report.failed > 0:
        print("  Failed tests:")
        for r in report.results:
            if not r.passed:
                print(f"    ❌ {r.name}: {r.message}")
    
    print()
    status = "✅ ALL TESTS PASSED" if report.failed == 0 else "❌ SOME TESTS FAILED"
    print(f"  {status}")
    print()


# =============================================================================
# Test Cases
# =============================================================================

async def test_health(config: TestConfig) -> TestResult:
    """Test health endpoint."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(f"{config.base_url}/health")
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                gpu = "GPU" if data.get("cuda_available") else "CPU"
                return TestResult(
                    name="Health Check",
                    passed=True,
                    duration_ms=duration,
                    message=f"Healthy ({gpu})",
                    details=data
                )
            else:
                return TestResult(
                    name="Health Check",
                    passed=False,
                    duration_ms=duration,
                    message=f"Status {response.status_code}"
                )
    except Exception as e:
        return TestResult(
            name="Health Check",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_ready(config: TestConfig) -> TestResult:
    """Test readiness endpoint."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(f"{config.base_url}/ready")
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                device = data.get("device", "unknown")
                return TestResult(
                    name="Readiness Check",
                    passed=data.get("ready", False),
                    duration_ms=duration,
                    message=f"Ready, device={device}",
                    details=data
                )
            else:
                return TestResult(
                    name="Readiness Check",
                    passed=False,
                    duration_ms=duration,
                    message=f"Status {response.status_code}"
                )
    except Exception as e:
        return TestResult(
            name="Readiness Check",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_single_image_analysis(config: TestConfig) -> TestResult:
    """Test single image analysis."""
    start = time.time()
    try:
        image_data = create_test_image()
        
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{config.api_url}/analyze",
                files={"file": ("test.jpg", image_data, "image/jpeg")}
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                category = data.get("analysis", {}).get("construction_category", "Unknown")
                confidence = data.get("analysis", {}).get("confidence", 0)
                return TestResult(
                    name="Single Image Analysis",
                    passed=data.get("success", False),
                    duration_ms=duration,
                    message=f"{category} (confidence: {confidence:.2f})",
                    details=data
                )
            else:
                return TestResult(
                    name="Single Image Analysis",
                    passed=False,
                    duration_ms=duration,
                    message=f"Status {response.status_code}"
                )
    except Exception as e:
        return TestResult(
            name="Single Image Analysis",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_batch_analysis(config: TestConfig, count: int = 5) -> TestResult:
    """Test batch image analysis."""
    start = time.time()
    try:
        files = [
            ("files", (f"test{i}.jpg", create_test_image(), "image/jpeg"))
            for i in range(count)
        ]
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{config.api_url}/batch-analyze",
                files=files
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                successful = data.get("successful", 0)
                total = data.get("total", 0)
                return TestResult(
                    name=f"Batch Analysis ({count} images)",
                    passed=successful == total,
                    duration_ms=duration,
                    message=f"{successful}/{total} successful",
                    details=data
                )
            else:
                return TestResult(
                    name=f"Batch Analysis ({count} images)",
                    passed=False,
                    duration_ms=duration,
                    message=f"Status {response.status_code}"
                )
    except Exception as e:
        return TestResult(
            name=f"Batch Analysis ({count} images)",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_categories(config: TestConfig) -> TestResult:
    """Test categories endpoint."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(f"{config.api_url}/categories")
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                count = data.get("total", 0)
                return TestResult(
                    name="Get Categories",
                    passed=count > 0,
                    duration_ms=duration,
                    message=f"{count} categories available",
                    details=data
                )
            else:
                return TestResult(
                    name="Get Categories",
                    passed=False,
                    duration_ms=duration,
                    message=f"Status {response.status_code}"
                )
    except Exception as e:
        return TestResult(
            name="Get Categories",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_stats(config: TestConfig) -> TestResult:
    """Test stats endpoint."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(f"{config.api_url}/stats")
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                uptime = data.get("uptime_seconds", 0)
                gpu = data.get("gpu", {}).get("available", False)
                return TestResult(
                    name="Get Stats",
                    passed=True,
                    duration_ms=duration,
                    message=f"Uptime: {uptime:.0f}s, GPU: {gpu}",
                    details=data
                )
            else:
                return TestResult(
                    name="Get Stats",
                    passed=False,
                    duration_ms=duration,
                    message=f"Status {response.status_code}"
                )
    except Exception as e:
        return TestResult(
            name="Get Stats",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_cors(config: TestConfig) -> TestResult:
    """Test CORS configuration."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.options(
                f"{config.base_url}/health",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "GET"
                }
            )
            duration = (time.time() - start) * 1000
            
            cors_origin = response.headers.get("access-control-allow-origin", "")
            has_cors = bool(cors_origin)
            
            return TestResult(
                name="CORS Configuration",
                passed=has_cors,
                duration_ms=duration,
                message=f"Allow-Origin: {cors_origin}" if has_cors else "CORS not configured"
            )
    except Exception as e:
        return TestResult(
            name="CORS Configuration",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_file_protocol_cors(config: TestConfig) -> TestResult:
    """Test CORS for file:// protocol (null origin)."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.options(
                f"{config.base_url}/health",
                headers={
                    "Origin": "null",
                    "Access-Control-Request-Method": "GET"
                }
            )
            duration = (time.time() - start) * 1000
            
            cors_origin = response.headers.get("access-control-allow-origin", "")
            
            return TestResult(
                name="File Protocol CORS",
                passed=cors_origin == "null",
                duration_ms=duration,
                message="Allows file:// origin" if cors_origin == "null" else f"Origin: {cors_origin}"
            )
    except Exception as e:
        return TestResult(
            name="File Protocol CORS",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_invalid_file(config: TestConfig) -> TestResult:
    """Test rejection of invalid file types."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(
                f"{config.api_url}/analyze",
                files={"file": ("test.txt", b"not an image", "text/plain")}
            )
            duration = (time.time() - start) * 1000
            
            # Should reject with 400
            passed = response.status_code == 400
            
            return TestResult(
                name="Invalid File Rejection",
                passed=passed,
                duration_ms=duration,
                message=f"Status {response.status_code}" + (" (correct)" if passed else " (should be 400)")
            )
    except Exception as e:
        return TestResult(
            name="Invalid File Rejection",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_large_image(config: TestConfig) -> TestResult:
    """Test handling of large images."""
    start = time.time()
    try:
        # Create a 4K image
        image_data = create_test_image(3840, 2160)
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{config.api_url}/analyze",
                files={"file": ("large.jpg", image_data, "image/jpeg")}
            )
            duration = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                return TestResult(
                    name="Large Image (4K)",
                    passed=data.get("success", False),
                    duration_ms=duration,
                    message=f"Processed successfully",
                    details=data
                )
            else:
                return TestResult(
                    name="Large Image (4K)",
                    passed=False,
                    duration_ms=duration,
                    message=f"Status {response.status_code}"
                )
    except Exception as e:
        return TestResult(
            name="Large Image (4K)",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_concurrent_requests(config: TestConfig, count: int = 5) -> TestResult:
    """Test concurrent request handling."""
    start = time.time()
    try:
        image_data = create_test_image()
        
        async def make_request(client, i):
            response = await client.post(
                f"{config.api_url}/analyze",
                files={"file": (f"concurrent{i}.jpg", image_data, "image/jpeg")}
            )
            return response.status_code == 200
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = [make_request(client, i) for i in range(count)]
            results = await asyncio.gather(*tasks)
        
        duration = (time.time() - start) * 1000
        successful = sum(results)
        
        return TestResult(
            name=f"Concurrent Requests ({count})",
            passed=successful == count,
            duration_ms=duration,
            message=f"{successful}/{count} successful"
        )
    except Exception as e:
        return TestResult(
            name=f"Concurrent Requests ({count})",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


async def test_latency_benchmark(config: TestConfig, iterations: int = 10) -> TestResult:
    """Benchmark single image latency."""
    start = time.time()
    try:
        image_data = create_test_image()
        latencies = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Warm-up
            await client.post(
                f"{config.api_url}/analyze",
                files={"file": ("warmup.jpg", image_data, "image/jpeg")}
            )
            
            # Benchmark
            for i in range(iterations):
                req_start = time.time()
                response = await client.post(
                    f"{config.api_url}/analyze",
                    files={"file": (f"bench{i}.jpg", image_data, "image/jpeg")}
                )
                if response.status_code == 200:
                    latencies.append((time.time() - req_start) * 1000)
        
        duration = (time.time() - start) * 1000
        
        if latencies:
            avg = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            
            return TestResult(
                name=f"Latency Benchmark ({iterations} iterations)",
                passed=True,
                duration_ms=duration,
                message=f"Avg: {avg:.0f}ms, Min: {min_lat:.0f}ms, Max: {max_lat:.0f}ms",
                details={"latencies": latencies, "avg": avg, "min": min_lat, "max": max_lat}
            )
        else:
            return TestResult(
                name=f"Latency Benchmark ({iterations} iterations)",
                passed=False,
                duration_ms=duration,
                message="No successful requests"
            )
    except Exception as e:
        return TestResult(
            name=f"Latency Benchmark ({iterations} iterations)",
            passed=False,
            duration_ms=(time.time() - start) * 1000,
            message=str(e)
        )


# =============================================================================
# Test Runners
# =============================================================================

async def run_quick_tests(config: TestConfig) -> TestReport:
    """Run quick smoke tests."""
    print_header("QUICK SMOKE TESTS")
    
    report = TestReport()
    
    tests = [
        test_health(config),
        test_ready(config),
        test_single_image_analysis(config),
        test_categories(config),
        test_cors(config),
    ]
    
    for test_coro in tests:
        result = await test_coro
        report.results.append(result)
        print_result(result)
    
    return report


async def run_all_tests(config: TestConfig) -> TestReport:
    """Run all tests."""
    print_header("FULL TEST SUITE")
    
    report = TestReport()
    
    tests = [
        # Connectivity
        test_health(config),
        test_ready(config),
        test_cors(config),
        test_file_protocol_cors(config),
        
        # Core functionality
        test_single_image_analysis(config),
        test_batch_analysis(config, 5),
        test_categories(config),
        test_stats(config),
        
        # Error handling
        test_invalid_file(config),
        
        # Performance
        test_large_image(config),
        test_concurrent_requests(config, 5),
    ]
    
    for test_coro in tests:
        result = await test_coro
        report.results.append(result)
        print_result(result)
    
    return report


async def run_performance_tests(config: TestConfig) -> TestReport:
    """Run performance benchmarks."""
    print_header("PERFORMANCE BENCHMARKS")
    
    report = TestReport()
    
    tests = [
        test_latency_benchmark(config, 10),
        test_concurrent_requests(config, 10),
        test_batch_analysis(config, 10),
        test_large_image(config),
    ]
    
    for test_coro in tests:
        result = await test_coro
        report.results.append(result)
        print_result(result)
    
    return report


async def run_error_tests(config: TestConfig) -> TestReport:
    """Run error scenario tests."""
    print_header("ERROR SCENARIO TESTS")
    
    report = TestReport()
    
    tests = [
        test_invalid_file(config),
    ]
    
    for test_coro in tests:
        result = await test_coro
        report.results.append(result)
        print_result(result)
    
    return report


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Integration Testing Script for Construction Photo Analyzer"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL for API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick smoke tests"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Run error scenario tests"
    )
    
    args = parser.parse_args()
    
    # Default to quick tests if no option specified
    if not any([args.all, args.quick, args.performance, args.errors]):
        args.quick = True
    
    config = TestConfig(base_url=args.url)
    
    print_header("CONSTRUCTION PHOTO ANALYZER - INTEGRATION TESTS")
    print(f"  Target: {config.base_url}")
    print(f"  API:    {config.api_url}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run selected tests
    reports = []
    
    if args.all:
        report = asyncio.run(run_all_tests(config))
        reports.append(report)
    else:
        if args.quick:
            report = asyncio.run(run_quick_tests(config))
            reports.append(report)
        
        if args.performance:
            report = asyncio.run(run_performance_tests(config))
            reports.append(report)
        
        if args.errors:
            report = asyncio.run(run_error_tests(config))
            reports.append(report)
    
    # Combine reports if multiple
    if len(reports) > 1:
        combined = TestReport()
        for r in reports:
            combined.results.extend(r.results)
        print_report(combined)
        return 0 if combined.failed == 0 else 1
    elif reports:
        print_report(reports[0])
        return 0 if reports[0].failed == 0 else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
