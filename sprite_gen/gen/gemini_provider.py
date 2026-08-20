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


DEFAULT_MODEL = "gemini-3.1-flash-lite-image"


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
    """Generate or edit images using Google GenAI SDK."""

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
        started = time.monotonic()

        # Branch 1: Imagen Models (Text-to-Image only)
        if model.startswith("imagen-"):
            if request.refs:
                raise SystemExit(
                    f"gemini-gen: model '{model}' does not support reference images via generate_images. "
                    "Use a Gemini multimodal model (e.g., gemini-2.5-flash) for image-to-image."
                )
            try:
                result = client.models.generate_images(
                    model=model,
                    prompt=request.prompt,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                        output_mime_type="image/png",
                    ),
                )
                image_bytes = result.generated_images[0].image.image_bytes
                request.raw.write_bytes(image_bytes)
                verify_png(request.raw)

                return ProviderRun(
                    provider=self.name,
                    elapsed_seconds=time.monotonic() - started,
                    model=model,
                    session_id=None,
                    extra={},
                )
            except Exception as exc:
                raise SystemExit(f"gemini-gen: imagen generation failed: {exc}") from exc

        # Branch 2: Gemini Multimodal Models (Image Editing / Text + Image Inputs)
        contents: list[Any] = []
        if request.refs:
            contents.extend(_load_ref_images(request.refs))
        
        # Explicit instruction encourages the model to emit an image modality
        prompt_text = f"Generate an image matching this request: {request.prompt}"
        contents.append(prompt_text)

        config = types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            temperature=0.4,  # Lower temperature reduces conversational text drift
        )

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
        image_mime: str | None = None
        response_texts: list[str] = []

        for part in candidate.content.parts:
            if part.inline_data is not None and part.inline_data.data:
                image_bytes = part.inline_data.data
                image_mime = part.inline_data.mime_type
            if part.text:
                response_texts.append(part.text)

        if image_bytes is None:
            printed_text = "\n".join(response_texts) if response_texts else "No text returned."
            raise SystemExit(
                f"gemini-gen: no image data in response from model '{model}'.\n"
                f"Gemini output standard text instead:\n--- FEEDBACK ---\n{printed_text}"
            )

        # Gemini may return JPEG/WebP instead of PNG. Convert to PNG via PIL so
        # the downstream verify_png magic-byte check always passes.
        if image_mime and image_mime.lower() != "image/png":
            try:
                from PIL import Image as PILImage
                import io as _io
                img = PILImage.open(_io.BytesIO(image_bytes))
                img.save(request.raw, format="PNG")
            except Exception as exc:
                raise SystemExit(
                    f"gemini-gen: failed to convert {image_mime} to PNG: {exc}"
                ) from exc
        else:
            request.raw.write_bytes(image_bytes)
        verify_png(request.raw)

        return ProviderRun(
            provider=self.name,
            elapsed_seconds=elapsed,
            model=model,
            session_id=None,
            extra={"response_text": response_texts},
        )