# GLM-OCR API Server

OpenAI-compatible OCR API powered by the GLM-OCR model, served via vLLM.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client     │────▶│   API Gateway    │────▶│    vLLM Server   │
│  (curl/SDK)  │◀────│   FastAPI :3000  │◀────│   GLM-OCR :8080  │
│              │     │                  │     │                  │
└──────────────┘     └──────────────────┘     └─────────────────┘
                         │                         │
                    ┌────┴────┐            ┌───────┴───────┐
                    │uploads/ │            │  models cache  │
                    │output/  │            │ (huggingface)  │
                    └─────────┘            └───────────────┘
```

## Quick Start

```bash
# 1. Clone and setup
cd ocr
cp .env.example .env
# Edit .env if needed (GPU settings, API key, etc.)

# 2. Launch
docker compose up -d

# 3. Wait for vLLM to download the model (first time ~5 min)
docker compose logs -f glm-ocr

# 4. Test
curl http://localhost:3000/health
```

## Endpoints

### Health Check
```bash
curl http://localhost:3000/health
```

### List Models
```bash
curl http://localhost:3000/v1/models
```

### Chat Completions (OpenAI-compatible)
```bash
# Text only
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-ocr",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# With image (base64)
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-ocr",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Extract all text from this document"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }]
  }'

# Streaming
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-ocr",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### PDF OCR
```bash
# Upload PDF for OCR
curl -X POST http://localhost:3000/v1/ocr \
  -F "file=@document.pdf" \
  -F "prompt=Extract all text and tables as markdown"

# Get result by job ID
curl http://localhost:3000/v1/ocr/{job_id}

# List all jobs
curl http://localhost:3000/v1/ocr
```

### Python SDK (openai)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="your-api-key",  # if set in .env
)

# Use like any OpenAI model
response = client.chat.completions.create(
    model="glm-ocr",
    messages=[{"role": "user", "content": "Describe this image"}],
)
print(response.choices[0].message.content)
```

### Python - PDF OCR
```python
import requests

response = requests.post(
    "http://localhost:3000/v1/ocr",
    files={"file": open("document.pdf", "rb")},
    data={"prompt": "Extract text and tables as markdown"},
)
result = response.json()
print(result["full_markdown"])
```

## Persistent Volumes

All data is stored in `volumes/`:

| Directory | Purpose |
|-----------|---------|
| `volumes/models/` | HuggingFace model cache (GLM-OCR weights) |
| `volumes/uploads/` | Temporary PDF uploads |
| `volumes/output/` | OCR results (JSON + Markdown) |

These survive container restarts. The model is downloaded once on first startup.

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `zai-org/GLM-OCR` | HuggingFace model ID |
| `VLLM_PORT` | `8080` | vLLM service port |
| `API_PORT` | `3000` | API gateway port |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction |
| `MAX_MODEL_LEN` | `4096` | Max context length |
| `PDF_DPI` | `200` | PDF image conversion DPI |
| `API_KEY` | (empty) | Bearer token for auth |
| `HUGGING_FACE_HUB_TOKEN` | (empty) | HF token if needed |

## GPU Requirements

- **Minimum**: 1x GPU with 8GB VRAM (RTX 3060+)
- **Recommended**: 1x GPU with 12GB+ VRAM (RTX 3090/4090)
- Model size: ~0.9B parameters in BF16 (~1.8GB VRAM)

## Docker Images

- `vllm/vllm-openai:latest` - Official vLLM image with NVIDIA support
- Custom `api/` image built from `python:3.12-slim` with poppler-utils

## Model

GLM-OCR by Zhipu AI (THUDM):
- 0.9B parameters multimodal model
- SOTA on OmniDocBench V1.5 (94.62)
- Supports layout analysis + parallel region recognition
- Apache 2.0 license

GitHub: https://github.com/zai-org/GLM-OCR
HuggingFace: https://huggingface.co/zai-org/GLM-OCR
