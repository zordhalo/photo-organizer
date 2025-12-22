# ============================================
# Photo Organizer - Docker Run Script (Windows)
# ============================================

param(
    [string]$Tag = "latest",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# Configuration
$ImageName = "photo-organizer"
$ContainerName = "photo-organizer-run"

Write-Host "============================================" -ForegroundColor Green
Write-Host "  Photo Organizer - Docker Run" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Check Docker is running
try {
    docker info | Out-Null
} catch {
    Write-Host "Error: Docker is not running" -ForegroundColor Red
    exit 1
}

# Stop existing container if running
$existing = docker ps -q -f "name=${ContainerName}"
if ($existing) {
    Write-Host "Stopping existing container..." -ForegroundColor Yellow
    docker stop $ContainerName | Out-Null
}

# Remove existing container
$existingAll = docker ps -aq -f "name=${ContainerName}"
if ($existingAll) {
    Write-Host "Removing existing container..." -ForegroundColor Yellow
    docker rm $ContainerName | Out-Null
}

# Run the container
Write-Host ""
Write-Host "Starting container..." -ForegroundColor Yellow
Write-Host "  Image: ${ImageName}:${Tag}"
Write-Host "  Port: ${Port}"
Write-Host ""

docker run -d `
    --name $ContainerName `
    --gpus all `
    -p "${Port}:8000" `
    -e USE_GPU=true `
    -e CUDA_VISIBLE_DEVICES=0 `
    -v "photo-organizer-cache:/home/appuser/.cache/torch" `
    --restart unless-stopped `
    "${ImageName}:${Tag}"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start container" -ForegroundColor Red
    exit 1
}

# Wait for container to start
Write-Host "Waiting for container to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check health
$MaxRetries = 30
$RetryCount = 0

while ($RetryCount -lt $MaxRetries) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:${Port}/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host ""
            Write-Host "============================================" -ForegroundColor Green
            Write-Host "  Container is running!" -ForegroundColor Green
            Write-Host "============================================" -ForegroundColor Green
            Write-Host "  API URL: http://localhost:${Port}"
            Write-Host "  Health: http://localhost:${Port}/health"
            Write-Host "  Docs: http://localhost:${Port}/docs"
            Write-Host ""
            Write-Host "To view logs:"
            Write-Host "  docker logs -f ${ContainerName}" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "To stop:"
            Write-Host "  docker stop ${ContainerName}" -ForegroundColor Yellow
            exit 0
        }
    } catch {
        $RetryCount++
        Write-Host "  Waiting for health check... (${RetryCount}/${MaxRetries})"
        Start-Sleep -Seconds 2
    }
}

Write-Host "Error: Container failed to start" -ForegroundColor Red
Write-Host "Check logs with: docker logs ${ContainerName}"
exit 1
