#!/bin/bash
# ============================================
# Photo Organizer - Docker Entrypoint Script
# ============================================

set -e

echo "============================================"
echo "  Photo Organizer API - Starting..."
echo "============================================"

# Check GPU availability
if [ "$USE_GPU" = "true" ]; then
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        echo "GPU acceleration enabled"
    else
        echo "Warning: nvidia-smi not found, GPU may not be available"
    fi
else
    echo "GPU acceleration disabled (USE_GPU=$USE_GPU)"
fi

# Set default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}

echo ""
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  Log Level: $LOG_LEVEL"
echo "  GPU Enabled: $USE_GPU"
echo "  CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Execute the main command
exec uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --access-log
