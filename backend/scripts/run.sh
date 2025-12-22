#!/bin/bash
# ============================================
# Photo Organizer - Docker Run Script
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
CONTAINER_NAME="photo-organizer-run"
PORT="${2:-8000}"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Photo Organizer - Docker Run${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Stop existing container if running
if docker ps -q -f name=${CONTAINER_NAME} | grep -q .; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME}
fi

# Remove existing container
if docker ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
    echo -e "${YELLOW}Removing existing container...${NC}"
    docker rm ${CONTAINER_NAME}
fi

# Run the container
echo ""
echo -e "${YELLOW}Starting container...${NC}"
echo -e "  Image: ${IMAGE_NAME}:${TAG}"
echo -e "  Port: ${PORT}"
echo ""

docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -p ${PORT}:8000 \
    -e USE_GPU=true \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v photo-organizer-cache:/home/appuser/.cache/torch \
    --restart unless-stopped \
    ${IMAGE_NAME}:${TAG}

# Wait for container to start
echo -e "${YELLOW}Waiting for container to start...${NC}"
sleep 5

# Check health
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo ""
        echo -e "${GREEN}============================================${NC}"
        echo -e "${GREEN}  Container is running!${NC}"
        echo -e "${GREEN}============================================${NC}"
        echo -e "  API URL: http://localhost:${PORT}"
        echo -e "  Health: http://localhost:${PORT}/health"
        echo -e "  Docs: http://localhost:${PORT}/docs"
        echo ""
        echo -e "To view logs:"
        echo -e "  ${YELLOW}docker logs -f ${CONTAINER_NAME}${NC}"
        echo ""
        echo -e "To stop:"
        echo -e "  ${YELLOW}docker stop ${CONTAINER_NAME}${NC}"
        exit 0
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -e "  Waiting for health check... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 2
done

echo -e "${RED}Error: Container failed to start${NC}"
echo -e "Check logs with: docker logs ${CONTAINER_NAME}"
exit 1
