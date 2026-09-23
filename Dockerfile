# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Qwen-Image 2.1 serverless worker for RunPod
#
#   * latest ComfyUI (pinned, currently 0.37.1) with Comfy Kitchen attention
#   * ComfyUI-Easy-Use, rgthree-comfy, ComfyUI-KJNodes, ComfyUI-GGUF baked in
#   * Qwen-Image 2.1 GGUF diffusion model + Qwen3-VL 8B text encoder + VAE
#
# See README.md for the model-storage decisions (what is baked, what is fetched
# at first start, why) and the CUDA 13 / --use-ck-attention requirement.
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=runpod/worker-comfyui:5.10.0-base
FROM ${BASE_IMAGE}

# --- versions / build knobs -------------------------------------------------
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
# Comfy Kitchen INT8 attention becomes the global attention path. It requires a
# Comfy Kitchen build with CUDA support (torch cu13+), otherwise ComfyUI exits
# with "Comfy Kitchen attention is unavailable" at startup.
ENV COMFY_EXTRA_ARGS="--use-ck-attention"

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
# 2. Python deps, then PyTorch on CUDA 13.
#    Comfy Kitchen's CUDA kernels come from cuBLAS 13, and comfy/quant_ops.py
#    disables the CUDA backend outright when torch.version.cuda < 13 - which
#    would make --use-ck-attention a hard startup failure and turn the INT8
#    ConvRot text encoder into a slow pure-Python fallback.
#    cu12 wheels left over from the base image are removed afterwards: they are
#    several GB of dead weight once torch is cu13.
# ---------------------------------------------------------------------------
RUN uv pip install --python "${VENV}/bin/python" -r /comfyui/requirements.txt \
 && uv pip install --python "${VENV}/bin/python" --force-reinstall \
      --index-url https://pypi.org/simple \
      --extra-index-url "${TORCH_INDEX_URL}" \
      "torch==${TORCH_VERSION}+cu130" \
      "torchvision==${TORCHVISION_VERSION}+cu130" \
      "torchaudio==${TORCHAUDIO_VERSION}+cu130" \
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
#    Small and never-changing files are baked now; the multi-GB ones are left
#    to the runtime fetcher (src/ensure_models.py) unless BAKE_MODELS=true.
# ---------------------------------------------------------------------------
COPY models.json /models.json
COPY src/ensure_models.py /ensure_models.py
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

# The upstream entrypoint launches ComfyUI with a fixed argument list. Splice in
# an env-driven hook so COMFY_EXTRA_ARGS (default --use-ck-attention) reaches
# main.py, without forking the whole script. The assertion below fails the build
# if upstream ever changes those lines.
RUN sed -i 's|python -u /comfyui/main.py --disable-auto-launch|python -u /comfyui/main.py ${COMFY_EXTRA_ARGS:-} --disable-auto-launch|g' /start.sh \
 && [ "$(grep -c 'COMFY_EXTRA_ARGS' /start.sh)" -eq 2 ]

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
