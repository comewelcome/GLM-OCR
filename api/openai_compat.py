"""OpenAI-compatible endpoints for the model server.

Implements:
    GET  /v1/models           - List models from vLLM
    POST /v1/chat/completions - Chat with image support
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://glm-ocr:8000")
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])


# ─── Pydantic models ───────────────────────────────────────────────

class ImageURLContent(BaseModel):
    url: str  # data:image/...;base64,... or http(s) URL
    detail: Optional[str] = "auto"


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageURL(BaseModel):
    type: str = "image_url"
    image_url: ImageURLContent


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    images: Optional[list[str]] = None  # Convenience: list of base64/image URLs


class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-vl-2b"
    messages: list[dict[str, Any]]  # Accept raw dicts for flexibility
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[list[str]] = None
    # OpenAI compat fields (ignored but accepted)
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: dict[str, str] = Field(default_factory=lambda: {"role": "assistant", "content": ""})
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    model: str = "qwen3-vl-2b"
    choices: list[ChatChoice]
    usage: Optional[UsageInfo] = None


# ─── Endpoints ─────────────────────────────────────────────────────

@router.get("/models")
async def list_models():
    """List available models (proxied from vLLM)."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="vLLM unavailable")
        return resp.json()


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports images in messages:
      - {"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}
      - {"role":"user","content":"Describe this image","images":["data:image/png;base64,..."]}
    """
    # Build vLLM-compatible messages
    vllm_messages = _build_vllm_messages(req.messages)

    payload = {
        "model": req.model,
        "messages": vllm_messages,
    }
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.stop is not None:
        payload["stop"] = req.stop

    if req.stream:
        return _stream_response(req.model, payload)

    # Non-streaming
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code != 200:
            logger.error("vLLM error: %s", resp.text)
            raise HTTPException(status_code=502, detail=resp.text)

        data = resp.json()

        return ChatCompletionResponse(
            model=req.model,
            choices=[
                ChatChoice(
                    message=c.get("message", {"role": "assistant", "content": ""}),
                    finish_reason=c.get("finish_reason"),
                )
                for c in data.get("choices", [])
            ],
            usage=UsageInfo(**data["usage"]) if "usage" in data else None,
        )


async def _stream_response(model: str, payload: dict) -> AsyncGenerator[str, None]:
    """Stream SSE response from vLLM as text lines.

    Yields strings formatted as SSE `data: ...` lines.
    """
    import json as j

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status_code != 200:
                error = f"vLLM error: {resp.status_code}"
                yield f"data: {j.dumps({'error': error})}\n\n"
                yield "data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    yield line + "\n\n"
                else:
                    yield f"data: {line}\n\n"

    yield "data: [DONE]\n\n"


# ─── Helpers ───────────────────────────────────────────────────────

def _build_vllm_messages(messages: list[dict]) -> list[dict]:
    """Convert user messages to vLLM-compatible multimodal format."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        images = msg.get("images", [])

        if isinstance(content, str):
            # Check if it's a text prompt with separate images
            if images:
                content_parts = [{"type": "text", "text": content}]
                for img in images:
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                result.append({"role": role, "content": content_parts})
            else:
                result.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Already multimodal format
            result.append({"role": role, "content": content})
        else:
            result.append({"role": role, "content": str(content)})

    return result
