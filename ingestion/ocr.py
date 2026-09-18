"""
OCR / vision-model fallback for scanned pages and image-only slides.

Two backends:
  1. `pytesseract` -- fast, free, local, decent on clean scans.
  2. OpenAI (`gpt-4o-mini`) as the vision-model fallback -- cheap,
     and far more robust than OCR alone on messy scans, rotated
     pages, or slides that are mostly a diagram/chart where OCR
     loses the meaning even if it gets some text right.


Default: try pytesseract first; if the result looks too short/garbled
to be real content, escalate to OpenAI vision. This keeps cost usage
down since most scans are handled by free local OCR, and only the
hard cases spend an OpenAI call.
"""
from __future__ import annotations

import io
import os

OPENAI_VISION_MODEL = "gpt-4o-mini"


def ocr_page_image(page, source: str, page_num: int) -> str:
    """OCR a pdfplumber Page object (renders it to an image first)."""
    try:
        im = page.to_image(resolution=200).original  # PIL Image
    except Exception:
        return ""
    return ocr_pil_image(im, source, page_num)


def ocr_image_bytes(image_bytes: bytes, source: str, page_num: int) -> str:
    from PIL import Image

    im = Image.open(io.BytesIO(image_bytes))
    return ocr_pil_image(im, source, page_num)


def ocr_pil_image(im, source: str, page_num: int) -> str:
    text = try_tesseract(im)
    if looks_sufficient(text):
        return text.strip()

    # Escalate to OpenAI vision for low-confidence / low-yield pages.
    vision_text = try_openai_vision(im, source, page_num)
    return vision_text or text or ""


def try_tesseract(im) -> str:
    try:
        import pytesseract

        # print("\n ===== OCR using pytesseract... ===== \n")
        return pytesseract.image_to_string(im)
    except Exception:
        return ""


def looks_sufficient(text: str, min_chars: int = 40) -> bool:
    """Very rough heuristic: a real page of content is rarely under
    ~40 characters of OCR output. Tune against your actual corpus."""
    return len(text.strip()) >= min_chars


def try_openai_vision(im, source: str, page_num: int) -> str:
    """
    Send the page image to an OpenAI vision-capable chat model and ask
    it to transcribe the content, including describing non-text
    visuals (charts/diagrams) in words so they're still searchable.
    Requires OPENAI_API_KEY. Uses gpt-4o-mini for a good cost/quality
    balance.
    """
    try:
        import base64
        from io import BytesIO

        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ""

        # Vision models take an image as a base64 data URL, not a PIL
        # object directly, so re-encode the page image as PNG bytes.
        buf = BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        # print("\n ===== OCR using OpenAI Vision model... ===== \n")

        client = OpenAI(api_key=api_key)
        prompt = (
            "Transcribe all readable text from this document page exactly. "
            "If it contains a chart, diagram, or figure with no transcribable "
            "text, describe what it shows in 1-3 sentences instead. "
            "Do not add commentary."
        )
        response = client.chat.completions.create(
            model=OPENAI_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception:
        # Don't let a single failed OCR/vision call crash the whole ingestion run -- log and move on with whatever text exists.
        print("\n ===== Failed to OCR page with OpenAI Vision model. ===== \n")
        return ""