# SPDX-License-Identifier: Apache-2.0
"""Google Gemini image generation provider (API key).

Uses the ``google-genai`` SDK with the Gemini image-generation model. Supports
both text-to-image generation (no refs) and image editing (with refs).

- No refs: text-to-image generation.
- With refs: image editing. The first ref is the source image; additional refs
  are passed as additional reference images for the edit.

Requires the ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` environment variable.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

from .base import GenRequest, GenTimeoutError, ProviderRun, verify_png


DEFAULT_MODEL = "gemini-2.0-flash-exp-image-generation"


def _resolve_api_key() -> str:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise SystemExit(
        "gemini-gen: no API key found; set GEMINI_API_KEY or GOOGLE_API_KEY "
        "environment variable"
    )


def _load_ref_images(refs: list[Path]) -> list[Any]:
    try:
        from PIL import Image as PILImage
    except ImportError as exc:
        raise SystemExit(
            "gemini-gen: pillow is required to read reference images"
        ) from exc
    images: list[Any] = []
    for ref in refs:
        ref_path = Path(ref).expanduser().resolve()
        if not ref_path.is_file():
            raise SystemExit(f"gemini-gen: reference image not found: {ref_path}")
        try:
            images.append(PILImage.open(ref_path).convert("RGB"))
        except Exception as exc:
            raise SystemExit(
                f"gemini-gen: cannot read reference image {ref_path}: {exc}"
            ) from exc
    return images


class GeminiProvider:
    """Generate one image through the Google Gemini image-generation API."""

    name = "gemini"

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise SystemExit(
                "gemini-gen: google-genai is not installed; "
                "install it with `pip install google-genai`"
            ) from exc

        api_key = _resolve_api_key()
        client = genai.Client(api_key=api_key)
        model = request.model or DEFAULT_MODEL

        request.raw.parent.mkdir(parents=True, exist_ok=True)

        # Build contents: reference images first (for edit), then the prompt.
        contents: list[Any] = []
        if request.refs:
            contents.extend(_load_ref_images(request.refs))
        contents.append(request.prompt)

        config = types.GenerateContentConfig(
            response_modalities=["Text", "Image"],
            temperature=1.0,
            max_output_tokens=8192,
        )

        started = time.monotonic()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise SystemExit(f"gemini-gen: generation failed: {exc}") from exc
        elapsed = time.monotonic() - started

        if not response.candidates:
            raise SystemExit("gemini-gen: no candidates in response")
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            raise SystemExit("gemini-gen: empty response from Gemini")

        image_bytes: bytes | None = None
        response_texts: list[str] = []
        for part in candidate.content.parts:
            if part.inline_data is not None and part.inline_data.data:
                image_bytes = part.inline_data.data
            if part.text:
                response_texts.append(part.text)

        if image_bytes is None:
            raise SystemExit(
                "gemini-gen: no image data in response "
                "(Gemini returned text without an image part)"
            )

        request.raw.write_bytes(image_bytes)
        verify_png(request.raw)

        return ProviderRun(
            provider=self.name,
            elapsed_seconds=elapsed,
            model=model,
            session_id=None,
            extra={"response_text": response_texts},
        )
