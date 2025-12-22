# Photo Organizer - Docker Deployment Guide

This guide covers deploying the Photo Organizer API using Docker with NVIDIA GPU support.

## 📋 Prerequisites

### Hardware Requirements
- NVIDIA GPU (RTX 20 series or newer recommended)
- 8GB+ GPU memory for optimal performance
- 16GB+ system RAM

### Software Requirements
- Docker 20.10+
- Docker Compose v2.0+
- NVIDIA Container Toolkit
- NVIDIA Driver 525+

## 🛠️ NVIDIA Container Toolkit Setup

### Ubuntu/Debian
```bash
# Add NVIDIA repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Windows (WSL2)
1. Install Docker Desktop with WSL2 backend
2. Enable GPU support in Docker Desktop settings
3. Ensure NVIDIA drivers are installed on Windows host

### Verify Installation
```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## 🚀 Quick Start

### Development Mode
```bash
cd backend

# Build and start with hot reload
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f
```

### Production Mode
```bash
cd backend

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Build and deploy
docker-compose -f docker-compose.prod.yml up -d --build

# Check status
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs -f
```

## 📁 File Structure

```
backend/
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Development configuration
├── docker-compose.prod.yml    # Production configuration
├── docker-entrypoint.sh       # Container entrypoint script
├── .dockerignore              # Build context exclusions
├── .env.example               # Environment template
└── scripts/
    ├── build.sh               # Linux build script
    ├── build.ps1              # Windows build script
    ├── run.sh                 # Linux run script
    └── run.ps1                # Windows run script
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU` | `true` | Enable GPU acceleration |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index(es) |
| `HOST_PORT` | `8000` | External port mapping |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `WORKERS` | `1` | Uvicorn worker count |
| `MODEL_NAME` | `resnet152` | Model architecture |
| `CONFIDENCE_THRESHOLD` | `0.1` | Min prediction confidence |
| `TOP_K_PREDICTIONS` | `5` | Max predictions returned |

### Resource Limits

```yaml
# In docker-compose.prod.yml
deploy:
  resources:
    limits:
      cpus: "4"
      memory: "8G"
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 🔨 Build Options

### Build with Scripts

**Linux/macOS:**
```bash
# Default production build
./scripts/build.sh

# Specific tag
./scripts/build.sh v1.0.0

# Development target
./scripts/build.sh latest development
```

**Windows (PowerShell):**
```powershell
# Default production build
.\scripts\build.ps1

# Specific tag
.\scripts\build.ps1 -Tag v1.0.0

# Development target
.\scripts\build.ps1 -Tag latest -Target development
```

### Manual Build
```bash
# Production build
docker build --target production -t photo-organizer:latest .

# Development build
docker build --target development -t photo-organizer:dev .
```

## ▶️ Running Containers

### Using Docker Compose (Recommended)
```bash
# Development with hot reload
docker-compose up

# Production
docker-compose -f docker-compose.prod.yml up -d
```

### Using Scripts

**Linux/macOS:**
```bash
./scripts/run.sh           # Default: latest tag, port 8000
./scripts/run.sh v1.0.0    # Specific tag
./scripts/run.sh latest 8080   # Custom port
```

**Windows (PowerShell):**
```powershell
.\scripts\run.ps1                  # Default
.\scripts\run.ps1 -Tag v1.0.0      # Specific tag
.\scripts\run.ps1 -Port 8080       # Custom port
```

### Manual Run
```bash
docker run -d \
  --name photo-organizer \
  --gpus all \
  -p 8000:8000 \
  -e USE_GPU=true \
  -v photo-organizer-cache:/home/appuser/.cache/torch \
  photo-organizer:latest
```

## ✅ Verification

### Health Check
```bash
# API health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "gpu_available": true}
```

### GPU Verification
```bash
# Check GPU inside container
docker exec photo-organizer-dev nvidia-smi

# Python GPU check
docker exec photo-organizer-dev python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### API Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🔍 Troubleshooting

### GPU Not Detected

1. **Verify NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```

2. **Check Docker GPU support:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Verify NVIDIA Container Toolkit:**
   ```bash
   nvidia-ctk --version
   ```

### Container Fails to Start

1. **Check logs:**
   ```bash
   docker-compose logs backend
   ```

2. **Verify image built successfully:**
   ```bash
   docker images | grep photo-organizer
   ```

3. **Check port availability:**
   ```bash
   # Linux
   lsof -i :8000
   
   # Windows
   netstat -ano | findstr :8000
   ```

### Out of Memory

1. **Reduce model size:**
   ```bash
   MODEL_NAME=resnet50  # Instead of resnet152
   ```

2. **Limit GPU memory (if supported):**
   ```yaml
   environment:
     - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

### Slow First Request

This is normal - the model loads on first inference. Use health checks with adequate start periods:
```yaml
healthcheck:
  start_period: 120s  # Allow 2 minutes for model loading
```

## 🚢 Production Deployment

### Single Server
```bash
# Deploy with production config
docker-compose -f docker-compose.prod.yml up -d

# Update deployment
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d --build
```

### With Nginx Reverse Proxy
```bash
# Create nginx config directory
mkdir -p nginx

# Add your nginx.conf and SSL certs
# Then deploy with nginx profile
docker-compose -f docker-compose.prod.yml --profile with-nginx up -d
```

### Kubernetes (Example)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: photo-organizer
spec:
  replicas: 1
  selector:
    matchLabels:
      app: photo-organizer
  template:
    metadata:
      labels:
        app: photo-organizer
    spec:
      containers:
      - name: api
        image: photo-organizer:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: USE_GPU
          value: "true"
```

## 📊 Monitoring

### Container Stats
```bash
# Real-time resource usage
docker stats photo-organizer-prod

# GPU monitoring
watch -n 1 nvidia-smi
```

### Logs
```bash
# Follow logs
docker-compose -f docker-compose.prod.yml logs -f backend

# Last 100 lines
docker-compose -f docker-compose.prod.yml logs --tail=100 backend
```

## 🧹 Cleanup

```bash
# Stop containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove images
docker rmi photo-organizer:latest

# Full cleanup
docker system prune -a
```

## 📝 CI/CD Example

### GitHub Actions
```yaml
name: Build and Push

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Registry
      uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: ./backend
        push: true
        tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}
        target: production
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Docker Compose GPU Support](https://docs.docker.com/compose/gpu-support/)
- [PyTorch CUDA](https://pytorch.org/get-started/locally/)
