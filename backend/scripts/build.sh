#!/bin/bash
# ============================================
# Photo Organizer - Docker Build Script
# ============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="photo-organizer"
TAG="${1:-latest}"
DOCKERFILE="Dockerfile"
BUILD_TARGET="${2:-production}"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Photo Organizer - Docker Build${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check for NVIDIA Docker support
echo -e "${YELLOW}Checking NVIDIA Container Toolkit...${NC}"
if docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✓ NVIDIA Container Toolkit is available${NC}"
else
    echo -e "${YELLOW}⚠ Warning: NVIDIA Container Toolkit not detected${NC}"
    echo -e "${YELLOW}  GPU acceleration may not be available${NC}"
    echo ""
fi

# Build the image
echo ""
echo -e "${YELLOW}Building Docker image...${NC}"
echo -e "  Image: ${IMAGE_NAME}:${TAG}"
echo -e "  Target: ${BUILD_TARGET}"
echo ""

docker build \
    --target ${BUILD_TARGET} \
    --tag ${IMAGE_NAME}:${TAG} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --progress=plain \
    -f ${DOCKERFILE} \
    .

# Get image size
IMAGE_SIZE=$(docker images ${IMAGE_NAME}:${TAG} --format "{{.Size}}")

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Build Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "  Image: ${IMAGE_NAME}:${TAG}"
echo -e "  Size: ${IMAGE_SIZE}"
echo ""
echo -e "To run the container:"
echo -e "  ${YELLOW}docker-compose up${NC}"
echo -e "  ${YELLOW}docker-compose -f docker-compose.prod.yml up -d${NC}"
echo ""
