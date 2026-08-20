#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Quick health check for the gemini image provider.

By default this validates the environment and imports without making any API
call. Pass --call to make one real (billable) image generation.

Usage:
    python3 scripts/test_gemini_provider.py
    python3 scripts/test_gemini_provider.py --call
    python3 scripts/test_gemini_provider.py --call --model gemini-2.0-flash-exp-image-generation
    python3 scripts/test_gemini_provider.py --call --with-refs
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _import_or_fail() -> tuple[type, type]:
    try:
        from sprite_gen.gen.base import GenRequest
        from sprite_gen.gen.gemini_provider import GeminiProvider
        return GenRequest, GeminiProvider
    except ImportError as exc:
        raise SystemExit(f"gemini-test: cannot import provider: {exc}")


def _check_env() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _make_temp_image() -> Path:
    from PIL import Image
    path = Path(tempfile.gettempdir()) / "gemini_test_ref.png"
    Image.new("RGBA", (256, 256), (0, 255, 255, 255)).save(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--call", action="store_true", help="make a real API call (costs money)")
    parser.add_argument("--model", default="gemini-2.0-flash-exp-image-generation")
    parser.add_argument("--out", default=None, type=Path, help="output path for --call")
    parser.add_argument("--with-refs", action="store_true",
                        help="test the image-edit path with a dummy reference image")
    args = parser.parse_args(argv)

    print("[gemini] checking imports...", flush=True)
    GenRequest, GeminiProvider = _import_or_fail()
    print("[gemini] GeminiProvider loaded", flush=True)

    key = _check_env()
    if key is None:
        print("[gemini] FAIL: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set", file=sys.stderr)
        return 1
    print("[gemini] API key is set", flush=True)

    if not args.call:
        print("[gemini] setup OK -- pass --call to make a real (billable) API test")
        return 0

    out = args.out
    if out is None:
        out = Path(tempfile.gettempdir()) / "gemini_test_out.png"
    out = out.expanduser().resolve()

    provider = GeminiProvider()
    prompt = "A 256x256 flat cyan background, no objects, solid #00FFFF color only"
    refs = []
    if args.with_refs:
        refs = [_make_temp_image()]
        prompt = "Edit this image: keep the size, fill it with solid #00FFFF cyan."

    workdir = Path(tempfile.mkdtemp(prefix="gemini-test-"))
    raw = workdir / "raw.png"
    request = GenRequest(prompt=prompt, raw=raw, refs=refs, model=args.model)

    try:
        result = provider.generate(request, workdir)
        print(f"[gemini] OK: generated {raw}")
        print(f"[gemini]   provider={result.provider} model={result.model} elapsed={result.elapsed_seconds:.2f}s")
        with open(raw, "rb") as src, open(out, "wb") as dst:
            dst.write(src.read())
        print(f"[gemini]   copied result to {out}")
        return 0
    except SystemExit as exc:
        print(f"[gemini] FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
