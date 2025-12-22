# Construction Photo Analyzer - Backend

AI-powered construction photo analysis using deep learning with GPU acceleration.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (Python 3.13 recommended)
- NVIDIA GPU with CUDA support (optional, but recommended)
- CUDA Toolkit 12.x (for GPU acceleration)

### Installation

1. **Clone and navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   # For CUDA 12.4
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   
   # For CPU only
   pip install torch torchvision torchaudio
   ```

4. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Verify installation:**
   ```bash
   python verify_setup.py
   ```

### Running the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python -m app.main
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # API endpoints
│   │   └── dependencies.py  # Dependency injection
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── vision_service.py  # Deep learning inference
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # API tests
│   └── test_vision.py       # Vision service tests
├── requirements.txt
├── .env.example
├── .gitignore
├── verify_setup.py
└── README.md
```

## 🔧 Configuration

Configuration is handled via environment variables. See `.env.example` for all options:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `USE_GPU` | `true` | Enable GPU acceleration |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |
| `MODEL_NAME` | `resnet50` | Classification model |
| `BATCH_SIZE` | `8` | Inference batch size |
| `CORS_ORIGINS` | `localhost:3000,...` | Allowed CORS origins |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

## 📊 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root - API info |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/analyze` | Analyze image |
| `GET` | `/api/v1/models` | List available models |
| `GET` | `/api/v1/gpu-status` | GPU memory/utilization |

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Analyze image
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"

# GPU status
curl http://localhost:8000/api/v1/gpu-status
```

## 🎮 GPU Support

This project is optimized for NVIDIA GPUs with CUDA support. Verify your setup:

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA in Python
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Supported GPU Models
- NVIDIA RTX 30/40/50 series
- NVIDIA RTX 3060+ recommended (8GB+ VRAM)

## 🔍 Troubleshooting

### CUDA not detected
1. Verify NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with correct CUDA version
3. Check `CUDA_VISIBLE_DEVICES` environment variable

### Out of Memory (OOM)
1. Reduce `BATCH_SIZE` in `.env`
2. Use a smaller model (`efficientnet_b0`)
3. Reduce input image resolution

### Import errors
1. Ensure virtual environment is activated
2. Reinstall dependencies: `pip install -r requirements.txt`

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/)

## 📄 License

MIT License - see LICENSE file for details.
