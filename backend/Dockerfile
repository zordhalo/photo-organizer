# ============================================
# Photo Organizer - Production Dockerfile
# Multi-stage build with NVIDIA GPU support
# ============================================

# ----------------------------------------
# Stage 1: Builder
# ----------------------------------------
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04 AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt /tmp/requirements.txt

# Install PyTorch with CUDA support first (for better caching)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
RUN pip install -r /tmp/requirements.txt

# ----------------------------------------
# Stage 2: Production Runtime
# ----------------------------------------
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04 AS production

# Labels for container metadata
LABEL maintainer="Photo Organizer Team" \
    version="1.0.0" \
    description="Photo Organizer API with GPU-accelerated image analysis"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Application settings
    APP_HOME=/app \
    APP_USER=appuser \
    APP_UID=1000 \
    APP_GID=1000 \
    # CUDA settings
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # API settings
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    # GPU settings
    USE_GPU=true \
    CUDA_VISIBLE_DEVICES=0

# Install Python runtime and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create non-root user for security
RUN groupadd --gid ${APP_GID} ${APP_USER} \
    && useradd --uid ${APP_UID} --gid ${APP_GID} --shell /bin/bash --create-home ${APP_USER}

# Set working directory
WORKDIR ${APP_HOME}

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy entrypoint script
COPY --chown=${APP_USER}:${APP_USER} docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} app/ ./app/

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/cache /app/uploads \
    && chown -R ${APP_USER}:${APP_USER} /app

# Switch to non-root user
USER ${APP_USER}

# Expose API port
EXPOSE 8000

# Health check configuration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - production server with Uvicorn
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info", \
    "--access-log"]

# ----------------------------------------
# Stage 3: Development (optional)
# ----------------------------------------
FROM production AS development

# Switch back to root for dev dependencies
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install dev Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    ipython

# Switch back to non-root user
USER ${APP_USER}

# Override CMD for development with hot reload
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--reload", \
    "--log-level", "debug"]
