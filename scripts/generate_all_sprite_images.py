#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batch-generate every sprite row for a prepared run.

This wraps scripts/generate_sprite_image.py once per state in sprite-request.json,
using the prepared prompts and layout guides. References always include the
base-source and the per-state layout guide; additional refs can be appended.

Usage:
    python3 scripts/generate_all_sprite_images.py --run-dir ./test --provider codex
    python3 scripts/generate_all_sprite_images.py --run-dir ./test --provider grok --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sprite_gen.spec.layout import guide_rel, prompt_rel, raw_rel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--provider", default="codex")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ref", action="append", default=[], type=Path,
                        help="additional reference image (repeatable)")
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--chroma-key", default="magenta")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands instead of running them")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    request_path = run_dir / "sprite-request.json"
    if not request_path.is_file():
        raise SystemExit(f"missing request file: {request_path}")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    states = request.get("states", {})
    if not states:
        raise SystemExit(f"no states in {request_path}")

    base_ref = run_dir / "base-source.png"
    if not base_ref.is_file():
        base_ref = next((p for p in run_dir.glob("base-source.*") if p.is_file()), None)
        if base_ref is None:
            raise SystemExit(f"missing base source image in {run_dir}")

    script = Path(__file__).resolve().parent / "generate_sprite_image.py"
    if not script.is_file():
        raise SystemExit(f"missing per-state generator: {script}")

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str]]] = []
    for state in sorted(states):
        prompt_file = run_dir / prompt_rel(request, state)
        if not prompt_file.is_file():
            print(f"[warn] missing prompt for {state}: {prompt_file}; skipping", file=sys.stderr)
            continue
        guide = run_dir / guide_rel(request, state)
        if not guide.is_file():
            print(f"[warn] missing layout guide for {state}: {guide}; skipping", file=sys.stderr)
            continue

        out = run_dir / raw_rel(request, state)
        if out.is_file() and not args.force:
            print(f"[skip] {state} already exists at {out}; pass --force to regenerate")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3", str(script),
            "--provider", args.provider,
            "--prompt-file", str(prompt_file),
            "--out", str(out),
            "--ref", str(base_ref),
            "--ref", str(guide),
        ]
        for ref in args.ref:
            cmd.extend(["--ref", str(ref)])
        if args.transparent:
            cmd.append("--transparent")
            cmd.extend(["--chroma-key", args.chroma_key])
        commands.append((state, cmd))

    if not commands:
        raise SystemExit("no states to generate")

    ok = 0
    failed = 0
    for state, cmd in commands:
        print(f"[gen] {state}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        try:
            result = subprocess.run(cmd, capture_output=False, text=True, encoding="utf-8")
            if result.returncode == 0:
                ok += 1
            else:
                failed += 1
                print(f"[gen] {state}: failed (exit {result.returncode})", file=sys.stderr)
        except (OSError, subprocess.SubprocessError) as exc:
            failed += 1
            print(f"[gen] {state}: {exc}", file=sys.stderr)

    print(f"\n[gen] done: {ok} ok, {failed} failed, {len(commands)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
