"""
PDF OCR endpoint.

Uploads PDF, converts to images, sends to vLLM GLM-OCR, returns structured result.
"""

import json
import logging
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://glm-ocr:8000")
PDF_DPI = int(os.getenv("PDF_DPI", "200"))
PDF_IMAGE_FORMAT = os.getenv("PDF_IMAGE_FORMAT", "jpeg").lower()
PDF_MAX_IMAGE_BYTES = int(os.getenv("PDF_MAX_IMAGE_BYTES", "140000"))
PDF_MAX_IMAGE_DIM = int(os.getenv("PDF_MAX_IMAGE_DIM", "1200"))
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/output")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OCR"])


class OCRPageResult(BaseModel):
    page: int
    markdown: str
    json_data: Optional[dict] = None


class OCRResponse(BaseModel):
    id: str
    object: str = "ocr.result"
    model: str = "glm-ocr"
    created: int
    pages: list[OCRPageResult]
    full_markdown: str


@router.post("/ocr")
async def ocr_pdf(
    file: UploadFile = File(..., description="PDF file to OCR"),
    prompt: str = Form(
        default="Extract all text, tables, and formulas from this document. "
                "Preserve the structure and formatting as markdown.",
        description="OCR prompt instruction",
    ),
    model: str = Form(default="glm-ocr", description="Model to use"),
):
    """
    Upload a PDF and get OCR results.

    Returns structured JSON with per-page markdown output.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Check file size (max 100MB)
    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 100MB)")

    job_id = uuid.uuid4().hex[:12]
    work_dir = UPLOAD_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded PDF
    pdf_path = work_dir / file.filename
    pdf_path.write_bytes(contents)
    logger.info("Saved PDF: %s (%d bytes)", pdf_path, len(contents))

    try:
        # Convert PDF pages to images
        images = _pdf_to_images(pdf_path, work_dir)
        if not images:
            raise HTTPException(status_code=400, detail="No pages extracted from PDF")

        logger.info("Converted PDF to %d images at %d DPI", len(images), PDF_DPI)

        # Process each page through vLLM GLM-OCR
        pages = []
        full_markdown = ""

        for page_num, img_path in enumerate(images, 1):
            logger.info("Processing page %d/%d", page_num, len(images))

            prepared_img = _prepare_image_for_vllm(img_path)
            if prepared_img != img_path:
                logger.info(
                    "Using compressed image %s (%d bytes) for page %d",
                    prepared_img.name,
                    prepared_img.stat().st_size,
                    page_num,
                )

            # Read image as base64
            img_b64 = _image_to_base64(prepared_img)
            image_type = "jpeg" if prepared_img.suffix.lower() in {".jpg", ".jpeg"} else prepared_img.suffix.lower().lstrip('.')

            # Build vLLM request
            vllm_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{image_type};base64,{img_b64}"},
                        },
                    ],
                }
            ]

            payload = {
                "model": model,
                "messages": vllm_messages,
            }

            # Call vLLM
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{VLLM_BASE_URL}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )

                if resp.status_code != 200:
                    logger.error("vLLM page %d error: %s", page_num, resp.text)
                    pages.append(OCRPageResult(
                        page=page_num,
                        markdown=f"[Error processing page {page_num}]",
                    ))
                    continue

                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Try to extract JSON if present
                json_data = None
                if "```json" in content:
                    try:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        json_data = json.loads(json_str)
                    except (json.JSONDecodeError, IndexError):
                        pass

                pages.append(OCRPageResult(
                    page=page_num,
                    markdown=content,
                    json_data=json_data,
                ))
                full_markdown += content + "\n\n---\n\n"

        # Save output
        output_dir = OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        result = OCRResponse(
            id=f"ocr-{job_id}",
            created=int(datetime.now(timezone.utc).timestamp()),
            pages=pages,
            full_markdown=full_markdown,
        )

        # Save result to disk
        result_path = output_dir / "result.json"
        result_path.write_text(result.model_dump_json(indent=2))

        md_path = output_dir / "result.md"
        md_path.write_text(full_markdown)

        logger.info("OCR complete: %s -> %s", job_id, result_path)
        return result

    finally:
        # Cleanup upload temp files
        shutil.rmtree(work_dir, ignore_errors=True)


@router.get("/ocr/{job_id}")
async def get_ocr_result(job_id: str):
    """Retrieve a previously processed OCR result."""
    result_path = OUTPUT_DIR / job_id / "result.json"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    data = json.loads(result_path.read_text())
    return data


@router.get("/ocr")
async def list_ocr_jobs():
    """List all processed OCR jobs."""
    if not OUTPUT_DIR.exists():
        return {"jobs": []}

    jobs = []
    for d in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and (d / "result.json").exists():
            result = json.loads((d / "result.json").read_text())
            jobs.append({
                "id": result.get("id"),
                "created": result.get("created"),
                "pages": len(result.get("pages", [])),
            })

    return {"jobs": jobs, "total": len(jobs)}


# ─── Helpers ───────────────────────────────────────────────────────

def _pdf_to_images(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Convert PDF to images using pdftoppm."""

    ext = PDF_IMAGE_FORMAT
    prefix = output_dir / "page"

    if ext in ("jpg", "jpeg"):
        format_flag = "-jpeg"
    else:
        format_flag = f"-{ext}"

    cmd = [
        "pdftoppm",
        "-r", str(PDF_DPI),
        "-f", "1",
        "-l", "9999",
        format_flag,
        "-progress",
        str(pdf_path),
        str(prefix),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes
        )
        if result.returncode != 0:
            logger.error("pdftoppm error: %s", result.stderr)
            return []
    except subprocess.TimeoutExpired:
        logger.error("pdftoppm timed out")
        return []

    # Find all generated page images
    # pdftoppm outputs files like prefix-1.png, prefix-2.png, or prefix-1.jpg.
    images = []
    search_ext = "jpg" if ext in ("jpg", "jpeg") else ext
    for f in sorted(output_dir.glob(f"page-*.{search_ext}")):
        images.append(f)

    return images


def _prepare_image_for_vllm(img_path: Path) -> Path:
    """Resize and compress page images to fit model context limits."""
    if img_path.suffix.lower() in {".jpg", ".jpeg"} and img_path.stat().st_size <= PDF_MAX_IMAGE_BYTES:
        return img_path

    with Image.open(img_path) as image:
        image = image.convert("RGB")
        max_size = max(image.width, image.height)
        if max_size > PDF_MAX_IMAGE_DIM:
            ratio = PDF_MAX_IMAGE_DIM / max_size
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                Image.LANCZOS,
            )

        quality = 75
        buffer = BytesIO()
        while True:
            buffer.seek(0)
            buffer.truncate(0)
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            size = buffer.tell()
            if size <= PDF_MAX_IMAGE_BYTES or quality <= 30:
                break
            quality -= 10
            if quality <= 30 and size > PDF_MAX_IMAGE_BYTES:
                image = image.resize(
                    (max(1, int(image.width * 0.85)), max(1, int(image.height * 0.85))),
                    Image.LANCZOS,
                )
                quality = 50

    output_path = img_path.with_suffix(".jpg")
    output_path.write_bytes(buffer.getvalue())
    logger.info(
        "Prepared image %s for vLLM (%d bytes, quality=%d)",
        img_path.name,
        output_path.stat().st_size,
        quality,
    )
    return output_path


def _image_to_base64(img_path: Path) -> str:
    """Read image file and return base64 string."""
    import base64

    return base64.b64encode(img_path.read_bytes()).decode("ascii")
