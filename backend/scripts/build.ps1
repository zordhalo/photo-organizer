# ============================================
# Photo Organizer - Docker Build Script (Windows)
# ============================================

param(
    [string]$Tag = "latest",
    [string]$Target = "production"
)

$ErrorActionPreference = "Stop"

# Configuration
$ImageName = "photo-organizer"
$Dockerfile = "Dockerfile"

Write-Host "============================================" -ForegroundColor Green
Write-Host "  Photo Organizer - Docker Build" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Check Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker is not running" -ForegroundColor Red
    exit 1
}

# Check for NVIDIA Docker support
Write-Host "Checking NVIDIA Container Toolkit..." -ForegroundColor Yellow
try {
    $nvidiaSmi = docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ NVIDIA Container Toolkit is available" -ForegroundColor Green
    } else {
        throw "NVIDIA not available"
    }
} catch {
    Write-Host "⚠ Warning: NVIDIA Container Toolkit not detected" -ForegroundColor Yellow
    Write-Host "  GPU acceleration may not be available" -ForegroundColor Yellow
    Write-Host ""
}

# Build the image
Write-Host ""
Write-Host "Building Docker image..." -ForegroundColor Yellow
Write-Host "  Image: ${ImageName}:${Tag}"
Write-Host "  Target: ${Target}"
Write-Host ""

docker build `
    --target $Target `
    --tag "${ImageName}:${Tag}" `
    --build-arg BUILDKIT_INLINE_CACHE=1 `
    --progress=plain `
    -f $Dockerfile `
    .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Build failed" -ForegroundColor Red
    exit 1
}

# Get image size
$ImageInfo = docker images "${ImageName}:${Tag}" --format "{{.Size}}"

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Image: ${ImageName}:${Tag}"
Write-Host "  Size: ${ImageInfo}"
Write-Host ""
Write-Host "To run the container:"
Write-Host "  docker-compose up" -ForegroundColor Yellow
Write-Host "  docker-compose -f docker-compose.prod.yml up -d" -ForegroundColor Yellow
Write-Host ""
