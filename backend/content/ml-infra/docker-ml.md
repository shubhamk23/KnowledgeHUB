---
title: "Docker for Machine Learning"
slug: docker-ml
summary: "Containerizing ML workloads with Docker — reproducible environments, GPU support, multi-stage builds, and production deployment patterns."
tags: ["docker", "containers", "ML-infrastructure", "GPU", "reproducibility", "deployment"]
visibility: public
---

# Docker for Machine Learning

## Why Docker for ML?

ML projects suffer from the classic "it works on my machine" problem — complex dependencies (CUDA versions, Python packages, system libraries) make reproducibility hard.

**Docker solves:**
- **Reproducibility:** Same environment everywhere (dev → staging → prod)
- **Isolation:** No dependency conflicts between projects
- **Portability:** Run on any machine with Docker installed
- **Scalability:** Kubernetes can orchestrate Docker containers

---

## Core Concepts

### Image vs Container

- **Image:** Immutable snapshot of filesystem + config (like a class)
- **Container:** Running instance of an image (like an object)

```bash
docker pull python:3.11-slim     # Pull image
docker run python:3.11-slim      # Start container
docker ps                        # List running containers
docker stop <container_id>       # Stop container
```

### Dockerfile Anatomy

```dockerfile
# Base image — CUDA version must match your GPU driver
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## GPU Support with Docker

### NVIDIA Container Toolkit

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Run with GPU Access

```bash
# Single GPU
docker run --gpus device=0 my-ml-image

# All GPUs
docker run --gpus all my-ml-image

# Specific count
docker run --gpus '"device=0,1"' my-ml-image

# Verify GPU inside container
docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### GPU Base Images

```dockerfile
# For inference (smaller)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# For training (larger, has dev tools)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# PyTorch official (GPU)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# HuggingFace ecosystem
FROM huggingface/transformers-pytorch-gpu:4.40.0
```

---

## Multi-Stage Builds

Reduce final image size by separating build and runtime stages:

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/build/deps -r requirements.txt

# Stage 2: Runtime (smaller final image)
FROM python:3.11-slim as runtime

WORKDIR /app
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY . .

CMD ["python", "inference.py"]
```

**Size comparison:**
- Single-stage with build tools: ~4GB
- Multi-stage optimized: ~800MB

---

## Docker Compose for ML Projects

Orchestrate multiple services (API + database + model server):

```yaml
# docker-compose.yml
version: "3.9"

services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/models/llm.gguf
    volumes:
      - ./models:/models
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  model-server:
    image: vllm/vllm-openai:latest
    command: --model meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.9
    ports:
      - "8080:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface

volumes:
  model-cache:
```

---

## Optimizing Docker for ML

### Layer Caching

Order layers from least-frequently to most-frequently changed:

```dockerfile
# GOOD: requirements.txt changes rarely → cached until it does
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .                          # code changes frequently

# BAD: cache invalidated every time src changes
COPY . .
RUN pip install -r requirements.txt
```

### .dockerignore

Exclude unnecessary files from build context:

```
# .dockerignore
__pycache__/
*.pyc
*.pyo
.git/
.env
*.gguf          # Large model files — mount as volumes instead
data/           # Large datasets
venv/
.pytest_cache/
```

### Model Management

**Don't bake models into images** — they're too large:

```dockerfile
# BAD
COPY llama-3.1-8b.gguf /app/models/  # 4.5GB in image

# GOOD: Mount at runtime
CMD ["python", "serve.py", "--model", "${MODEL_PATH}"]
```

```bash
docker run -v /local/models:/app/models \
           -e MODEL_PATH=/app/models/llama-3.1-8b.gguf \
           my-ml-api
```

---

## Production Patterns

### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Non-Root User

```dockerfile
RUN useradd -m -u 1000 mluser
USER mluser
```

### Secrets Management

```bash
# Use environment variables for secrets, not baked into image
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY my-app

# Or Docker secrets (in Swarm/Kubernetes)
docker secret create openai_key ./key.txt
```

---

## Serving ML Models with Docker

### TorchServe

```bash
# Create model archive
torch-model-archiver --model-name bert-classifier \
    --version 1.0 \
    --serialized-file model.pt \
    --handler custom_handler.py

# Start TorchServe
torchserve --start --model-store model_store/ --models bert-classifier.mar
```

### vLLM (LLM Serving)

```bash
docker run --gpus all \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2
```

### Triton Inference Server (NVIDIA)

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /models:/models \
    nvcr.io/nvidia/tritonserver:23.12-py3 \
    tritonserver --model-repository=/models
```

---

## Key Takeaways

1. **Docker solves reproducibility** — same environment from dev to production
2. **NVIDIA Container Toolkit** enables GPU access inside containers — essential for training
3. **Multi-stage builds** dramatically reduce image size by separating build and runtime
4. **Don't bake models into images** — mount as volumes for flexibility and small images
5. **Docker Compose** orchestrates multi-service ML stacks (API + model server + cache)
6. **Health checks + non-root users** are production essentials

## References

- Docker Documentation — https://docs.docker.com
- NVIDIA Container Toolkit — https://docs.nvidia.com/datacenter/cloud-native/
- vLLM Documentation — https://docs.vllm.ai
