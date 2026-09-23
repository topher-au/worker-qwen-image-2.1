# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Qwen-Image 2.1 serverless worker for RunPod
#
#   * latest ComfyUI (pinned, currently 0.37.1) with Comfy Kitchen attention
#   * ComfyUI-Easy-Use, rgthree-comfy, ComfyUI-KJNodes, ComfyUI-GGUF baked in
#   * Qwen-Image 2.1 (original Qwen/Qwen-Image-2.1 weights) + Qwen3-VL 8B text encoder + VAE
#
# See README.md for the model-storage decisions (what is baked, what is fetched
# at first start, why) and the CUDA 13 / --use-ck-attention requirement.
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=runpod/worker-comfyui:5.10.0-base
FROM ${BASE_IMAGE}

# --- versions / build knobs -------------------------------------------------
# BASE_IMAGE is declared before FROM (for the FROM line itself) and re-declared
# here so it is visible to RUN/ENV steps - the start.sh drift check prints it.
ARG BASE_IMAGE
ARG COMFYUI_VERSION=0.37.1
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ARG NODE_PACKAGES="comfyui-easy-use rgthree-comfy comfyui-kjnodes comfyui-gguf"
ARG EXTRA_NODE_PACKAGES=""
ARG BAKE_MODELS=false
ARG REMOVE_WORKSPACE_VENV=false

ENV VENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV COMFYUI_VERSION=${COMFYUI_VERSION}
# Comfy Kitchen INT8 attention: src/start.sh (vendored from the base repo and
# installed as /start.sh) launches ComfyUI with --use-ck-attention unless
# CK_ATTENTION=false. It requires a Comfy Kitchen build with CUDA support
# (torch cu13+), otherwise ComfyUI exits with "Comfy Kitchen attention is
# unavailable" at startup. COMFY_EXTRA_ARGS appends further flags.
ENV CK_ATTENTION=true
ENV COMFY_EXTRA_ARGS=""

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ---------------------------------------------------------------------------
# 1. ComfyUI at the requested version
#    comfy-cli installs ComfyUI as a git clone, so we can fast-forward it
#    in-place. The check right after guarantees the pinned version really
#    landed and that it knows the --use-ck-attention flag.
# ---------------------------------------------------------------------------
RUN cd /comfyui \
 && if [ -d .git ]; then \
      git fetch --depth 1 origin "+refs/tags/v${COMFYUI_VERSION}:refs/tags/v${COMFYUI_VERSION}" \
      && git -c advice.detachedHead=false checkout -f "v${COMFYUI_VERSION}" \
      && git clean -fdq; \
    else \
      curl -fsSL "https://github.com/comfyanonymous/ComfyUI/archive/refs/tags/v${COMFYUI_VERSION}.tar.gz" \
        | tar -xz -C /comfyui --strip-components=1; \
    fi \
 && grep -q -- "--use-ck-attention" /comfyui/comfy/cli_args.py \
 && grep -q "quick-test-for-ci" /comfyui/comfy/cli_args.py \
 && grep -q "\"${COMFYUI_VERSION}\"" /comfyui/comfyui_version.py

# ---------------------------------------------------------------------------
# 2. PyTorch on CUDA 13, then the rest of ComfyUI's dependencies.
#    Comfy Kitchen's CUDA kernels come from cuBLAS 13, and comfy/quant_ops.py
#    disables the CUDA backend outright when torch.version.cuda < 13 - which
#    would make --use-ck-attention a hard startup failure (attention.py calls
#    exit(-1)) and turn the INT8 ConvRot text encoder into a slow pure-Python
#    fallback. torch goes in first so requirements.txt (which declares a bare
#    `torch`) never has a reason to download a second, cu12 build.
#    cu12 wheels left over from the base image are removed afterwards: several
#    GB of dead weight once torch is cu13.
# ---------------------------------------------------------------------------
RUN uv pip install --python "${VENV}/bin/python" --force-reinstall \
      --index-url https://pypi.org/simple \
      --extra-index-url "${TORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}+cu130" \
      "torchvision==${TORCHVISION_VERSION}+cu130" \
      "torchaudio==${TORCHAUDIO_VERSION}+cu130" \
 && uv pip install --python "${VENV}/bin/python" -r /comfyui/requirements.txt \
 && uv pip install --python "${VENV}/bin/python" "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
 && uv pip list --python "${VENV}/bin/python" --format freeze \
      | grep -Eo '^nvidia-[a-z0-9_-]+-cu12' | sort -u > /tmp/cu12-packages.txt || true \
 && if [ -s /tmp/cu12-packages.txt ]; then \
      uv pip uninstall --python "${VENV}/bin/python" $(cat /tmp/cu12-packages.txt); \
    fi \
 && rm -f /tmp/cu12-packages.txt \
 && "${VENV}/bin/python" -c "import torch, comfy_kitchen; assert torch.version.cuda and int(torch.version.cuda.split('.')[0]) >= 13, 'need a cu13 torch, got ' + str(torch.version.cuda); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'comfy_kitchen ok')"

# ---------------------------------------------------------------------------
# 3. Custom node packs
#    comfy-node-install fails the build when a node cannot be installed, which
#    is why it is preferred over `comfy node install` directly.
#    comfy-cli puts node python deps into its own workspace venv, while the
#    worker runs ComfyUI from /opt/venv - so mirror every node requirements.txt
#    into /opt/venv or the nodes import-fail at startup.
# ---------------------------------------------------------------------------
RUN comfy-node-install ${NODE_PACKAGES} \
 && if [ -n "${EXTRA_NODE_PACKAGES}" ]; then comfy-node-install ${EXTRA_NODE_PACKAGES}; fi \
 && for r in /comfyui/custom_nodes/*/requirements.txt; do \
      [ -f "$r" ] && uv pip install --python "${VENV}/bin/python" -r "$r" || true; \
    done \
 && uv pip install --python "${VENV}/bin/python" "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
 && for d in ComfyUI-Easy-Use rgthree-comfy ComfyUI-KJNodes ComfyUI-GGUF; do \
      [ -d "/comfyui/custom_nodes/$d" ] || { echo "missing custom node: $d" >&2; exit 1; }; \
    done

# ---------------------------------------------------------------------------
# 4. Models
#    The worker builds all three checkpoints itself from the original
#    Qwen/Qwen-Image-2.1 repository (diffusers layout) with
#    src/safetensors_convert.py: shard merge, key renames, row-order fusion of
#    the SwiGLU FFN, and the unit axes the WanVAE decoder expects. Nothing is
#    baked by default (33 GB of output); at first start the fetcher reads the
#    source shards from RunPod's cached-model store and converts them onto the
#    network volume. Set BAKE_MODELS=true to do it at build time instead.
# ---------------------------------------------------------------------------
COPY models.json /models.json
COPY src/ensure_models.py /ensure_models.py
COPY src/safetensors_convert.py /safetensors_convert.py
RUN if [ "${BAKE_MODELS}" = "true" ]; then \
      python3 -u /ensure_models.py --manifest /models.json --dest-root /comfyui/models --all; \
    else \
      python3 -u /ensure_models.py --manifest /models.json --dest-root /comfyui/models --only-baked; \
    fi

# ---------------------------------------------------------------------------
# 5. Runtime wiring
# ---------------------------------------------------------------------------
# Search paths for a network volume. Superset of the upstream file, using the
# modern folder keys (text_encoders / diffusion_models) as well as the legacy
# aliases ComfyUI maps them to.
COPY src/extra_model_paths.yaml /comfyui/extra_model_paths.yaml

# Entrypoint wrapper: make sure the models exist, then run the upstream
# entrypoint (GPU pre-flight check, SSH, ComfyUI, RunPod handler).
COPY src/custom-start.sh /custom-start.sh
RUN chmod +x /custom-start.sh

# The worker entrypoint is vendored from the base repository into src/start.sh.
# Two local changes, both visible in that file: a CK_ATTENTION block that starts
# ComfyUI with --use-ck-attention, and ${COMFY_EXTRA_ARGS:-} on the launch lines
# for extra flags. Re-import after a BASE_IMAGE bump:  make sync-start-sh REF=<tag>
#
# The vendored file is installed as /start.sh. The checks below fail the build if
# the vendored file lost that wiring, and print a reviewable diff (without
# failing) when the base image's own /start.sh has drifted in some other way.
COPY src/start.sh /start.sh.vendored
RUN set -euo pipefail; \
    bash -n /start.sh.vendored; \
    [ "$(grep -c -F 'CK_ATTENTION_ARG="--use-ck-attention"' /start.sh.vendored)" -eq 1 ]; \
    [ "$(grep -c -F '${CK_ATTENTION_ARG}' /start.sh.vendored)" -eq 2 ]; \
    [ "$(grep -c -F '${COMFY_EXTRA_ARGS:-}' /start.sh.vendored)" -eq 2 ]; \
    if ! grep -q -F 'python -u /comfyui/main.py' /start.sh; then \
      echo "ERROR: ${BASE_IMAGE} /start.sh does not look like the worker entrypoint" >&2; \
      echo "       this repo vendors; re-import with:" >&2; \
      echo "         make sync-start-sh REF=<tag of ${BASE_IMAGE}>" >&2; \
      exit 1; \
    fi; \
    sed -e '/^# Enable Comfy Kitchen INT8 attention/,/^esac$/d' \
        -e 's|python -u /comfyui/main.py ${CK_ATTENTION_ARG} ${COMFY_EXTRA_ARGS:-} --disable-auto-launch|python -u /comfyui/main.py --disable-auto-launch|g' \
        /start.sh.vendored > /tmp/start.sh.unpatched; \
    if ! diff -q /tmp/start.sh.unpatched /start.sh > /dev/null; then \
      echo "WARNING: ${BASE_IMAGE} /start.sh differs from src/start.sh minus the local" >&2; \
      echo "         changes (base image may have moved since the vendored copy):" >&2; \
      diff -u /tmp/start.sh.unpatched /start.sh | head -40 >&2 || true; \
      echo "         the vendored src/start.sh is what gets installed; refresh with" >&2; \
      echo "         make sync-start-sh REF=<tag of ${BASE_IMAGE}>" >&2; \
    fi; \
    install -m 0755 /start.sh.vendored /start.sh; \
    rm -f /start.sh.vendored /tmp/start.sh.unpatched

# Optional: drop comfy-cli's own workspace venv (a second copy of torch, ~6 GB).
# Only do this if you never run comfy-cli commands inside a derived image.
RUN if [ "${REMOVE_WORKSPACE_VENV}" = "true" ]; then rm -rf /comfyui/.venv; fi

# ---------------------------------------------------------------------------
# 6. Build-time verification: import the full node graph on CPU, then check the
#    invariants this image exists for (cu13 torch, Comfy Kitchen, models).
# ---------------------------------------------------------------------------
COPY src/verify_image.py /verify_image.py
RUN cd /comfyui && timeout 900 "${VENV}/bin/python" main.py --quick-test-for-ci --cpu \
 && python3 -u /verify_image.py --manifest /models.json --models-root /comfyui/models

CMD ["/custom-start.sh"]
