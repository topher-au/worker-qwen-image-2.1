#!/usr/bin/env python3
"""Build-time assertions for the Qwen-Image 2.1 worker image.

Runs after ComfyUI's own --quick-test-for-ci smoke test. Everything here is a
property this image exists for; if one of them is false the image is not worth
shipping, so the build fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

failures: list[str] = []
notes: list[str] = []


def check(cond: bool, msg: str) -> None:
    (notes if cond else failures).append(("ok: " if cond else "FAIL: ") + msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/models.json")
    ap.add_argument("--models-root", default="/comfyui/models")
    args = ap.parse_args()

    import torch

    check(torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 13,
          f"torch {torch.__version__} built for CUDA {torch.version.cuda} (Comfy Kitchen needs cu13+)")

    import comfy_kitchen
    from comfy_kitchen import sage_attention

    check(callable(comfy_kitchen.int8_attention_is_available), "comfy_kitchen exposes int8_attention_is_available")
    check(hasattr(sage_attention, "int8_attention"), "comfy_kitchen ships the INT8 attention kernel wrapper")
    backends = comfy_kitchen.list_backends()
    notes.append("info: comfy_kitchen backends " + json.dumps({k: v.get("available") for k, v in backends.items()}))

    out = subprocess.run(
        [sys.executable, "-c", "from comfy.cli_args import args; print(hasattr(args, 'use_ck_attention'))"],
        cwd="/comfyui", capture_output=True, text=True,
    )
    check(out.stdout.strip() == "True", "ComfyUI parses --use-ck-attention (from /comfyui)")

    for name in ("ComfyUI-Easy-Use", "rgthree-comfy", "ComfyUI-KJNodes", "ComfyUI-GGUF"):
        check(os.path.isdir(f"/comfyui/custom_nodes/{name}"), f"custom node present: {name}")

    try:
        import gguf  # noqa: F401  (ComfyUI-GGUF dependency)
        check(True, "gguf python package importable for ComfyUI-GGUF")
    except Exception as exc:  # pragma: no cover
        check(False, f"gguf python package importable for ComfyUI-GGUF ({exc})")

    check(os.path.exists("/comfyui/extra_model_paths.yaml"), "network-volume model paths installed")
    check(os.access("/custom-start.sh", os.X_OK), "/custom-start.sh is executable")

    # All three checkpoints are converted from the original Qwen repository at
    # first start, so the converter has to be installed next to the fetcher.
    sys.path.insert(0, "/")
    try:
        import safetensors_convert

        check(all(hasattr(safetensors_convert, n) for n in ("convert", "output_matches", "read_header")),
              "safetensors_convert is installed for the first-start conversion")
    except Exception as exc:  # pragma: no cover
        check(False, f"safetensors_convert importable ({exc})")

    # The entrypoint is vendored from the base repo (see src/start.sh): whatever
    # else changes in it, the CK attention wiring has to survive.
    with open("/start.sh") as fh:
        start_sh = fh.read()
    check('CK_ATTENTION_ARG="--use-ck-attention"' in start_sh,
          "/start.sh starts ComfyUI with --use-ck-attention by default")
    check(start_sh.count("${CK_ATTENTION_ARG} ${COMFY_EXTRA_ARGS:-}") == 2,
          "/start.sh passes the CK attention + extra args to both ComfyUI launch lines")

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    baked_expected = 0
    for entry in manifest["models"]:
        path = os.path.join(args.models_root, entry["dest"], entry["name"])
        exists = os.path.exists(path)
        if entry.get("bake"):
            baked_expected += 1
            check(exists, f"baked model present: {path}")
        else:
            notes.append(f"info: {path} {'present' if exists else 'will be fetched at first start'}")

    with open("/comfyui/comfyui_version.py") as fh:
        version_line = [l for l in fh if "__version__" in l]
    notes.append("info: " + (version_line[0].strip() if version_line else "ComfyUI version unknown"))

    print("\n".join(notes))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"verify_image: all {len(notes)} checks passed ({baked_expected} baked model(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
