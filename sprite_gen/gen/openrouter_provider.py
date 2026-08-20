# SPDX-License-Identifier: Apache-2.0
"""OpenRouter image generation provider.

OpenRouter exposes an OpenAI-compatible ``/images`` API. Any image-gen model
OpenRouter supports can be selected with ``--model`` (default ``openai/dall-e-3``).

- No refs: calls ``images/generations`` for text-to-image.
- With refs: calls ``images/edits`` using the first ref as the image to edit.
  A second ref is treated as the edit mask. Additional refs are ignored by the
  endpoint (OpenAI image edit accepts only one image and one mask).

Requires the ``OPENROUTER_API_KEY`` environment variable.
"""

from __future__ import annotations

import base64
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from .base import GenRequest, GenTimeoutError, ProviderRun, verify_png


DEFAULT_MODEL = "openai/dall-e-3"
BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_api_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not value:
        raise SystemExit(
            "openrouter-gen: no API key found; set OPENROUTER_API_KEY "
            "environment variable"
        )
    return value


def _download_image(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            return response.read()
    except Exception as exc:
        raise SystemExit(f"openrouter-gen: failed to download image from {url}: {exc}") from exc


def _write_image(data: bytes, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    verify_png(dest)


class OpenRouterProvider:
    """Generate one image through the OpenRouter images API."""

    name = "openrouter"

    def generate(self, request: GenRequest, workdir: Path) -> ProviderRun:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "openrouter-gen: openai is not installed; "
                "install it with `pip install openai`"
            ) from exc

        client = OpenAI(
            base_url=BASE_URL,
            api_key=_resolve_api_key(),
            timeout=120,
        )
        model = request.model or DEFAULT_MODEL

        started = time.monotonic()
        if request.refs:
            image_path = Path(request.refs[0]).expanduser().resolve()
            if not image_path.is_file():
                raise SystemExit(f"openrouter-gen: reference image not found: {image_path}")
            mask = None
            if len(request.refs) > 1:
                mask_path = Path(request.refs[1]).expanduser().resolve()
                if not mask_path.is_file():
                    raise SystemExit(f"openrouter-gen: mask image not found: {mask_path}")
                mask = mask_path
                if len(request.refs) > 2:
                    extra = ", ".join(str(ref) for ref in request.refs[2:])
                    print(
                        f"[openrouter] using only first two refs; ignoring {extra}",
                        file=os.sys.stderr,
                    )
            response = client.images.edit(
                image=image_path,
                prompt=request.prompt,
                mask=mask,
                model=model,
                n=1,
                size="1024x1024",
            )
        else:
            response = client.images.generate(
                model=model,
                prompt=request.prompt,
                n=1,
                size="1024x1024",
            )
        elapsed = time.monotonic() - started

        if not response.data:
            raise SystemExit("openrouter-gen: API returned no image data")

        image = response.data[0]
        if image.b64_json:
            png = base64.b64decode(image.b64_json)
        elif image.url:
            png = _download_image(image.url)
        else:
            raise SystemExit("openrouter-gen: API response has neither b64_json nor url")

        _write_image(png, request.raw)

        return ProviderRun(
            provider=self.name,
            elapsed_seconds=elapsed,
            model=model,
            session_id=None,
            extra={"model": model, "revised_prompt": image.revised_prompt},
        )
