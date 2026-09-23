# worker-qwen-image-2.1

A RunPod **serverless** worker for **Qwen-Image 2.1** on ComfyUI, built on
[`runpod-workers/worker-comfyui`](https://github.com/runpod-workers/worker-comfyui)
(the `-base` image + its customization guide).

What is inside:

* **ComfyUI 0.37.1** (latest release tag), launched with **`--use-ck-attention`**
* **ComfyUI-Easy-Use**, **rgthree-comfy**, **ComfyUI-KJNodes**, **ComfyUI-GGUF**
* **Qwen-Image 2.1 GGUF** diffusion model (city96's GGUF loaders)
* **Qwen3-VL 8B INT8 ConvRot** text encoder + **Qwen-Image 2.1 VAE**

---

## 1. Model storage: what is baked, what is fetched, and why

The numbers decide this. Measured, not guessed:

| Item | Size |
| --- | --- |
| `runpod/worker-comfyui:5.10.0-base` (ComfyUI + torch, compressed image) | 14.67 GB |
| `qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |
| `qwen_image_2.1_Q6_K.gguf` | 5.84 GB |
| `qwen3vl_8b_int8_convrot.safetensors` | 9.35 GB |
| **total if everything is baked** | **30.5 GB** |

A RunPod serverless template's container disk defaults to **20 GB**, and that disk holds the
image *plus* the writable layer. Baking all three models in does not fit, and baking nothing
means every cold start re-downloads ~15 GB.

So:

| Model | Storage | Why |
| --- | --- | --- |
| VAE (0.68 GB) | **baked into the image** | tiny, immutable, needed by every job |
| GGUF diffusion model (5.84 GB) | **fetched at first start** | most likely to be swapped (Q5/Q6/Q8), and too big for the image |
| Text encoder (9.35 GB) | **fetched at first start** | same reason, biggest single file |

The fetch happens in this worker's entrypoint (`src/custom-start.sh` →
`src/ensure_models.py`) **before** ComfyUI starts, and defaults to writing into an attached
**network volume** (`/runpod-volume/models/...`). That means:

* the first worker of the endpoint pays the download, every later worker (and every
  subsequent cold start, and every other endpoint sharing the volume) starts instantly;
* the image stays ~15.5 GB, comfortably inside the 20 GB container disk;
* on a host without a volume the files go to the container's writable layer instead — that
  works, but it repeats on every cold start and needs `containerDiskInGb: 40`.

Prefer to bake everything anyway (`docker build --build-arg BAKE_MODELS=true`, then
`containerDiskInGb: 40`)? Also supported, same manifest drives both paths.

Downloads are staged as `<file>.part` and `os.replace()`d into place, so a worker that dies
mid-download never leaves a truncated file that ComfyUI would load and then crash on.
Workers sharing a volume serialise on an `flock`ed lock file; where the filesystem has no
`flock` the atomic rename is still what keeps things consistent. Expected sizes come from
Hugging Face (HEAD request), so a re-uploaded file warns instead of breaking a cold start.

---

## 2. CUDA 13 is required (the `--use-ck-attention` trap)

`--use-ck-attention` does not merely enable a speed-up — it is checked at ComfyUI import
time, and **ComfyUI calls `exit(-1)` if Comfy Kitchen's INT8 attention is unavailable**:

```text
Comfy Kitchen attention is unavailable. Install a Comfy Kitchen build with attention support to use --use-ck-attention.
```

Comfy Kitchen's CUDA kernels are built against **cuBLAS 13**, and
`comfy/quant_ops.py` disables its CUDA backend outright when `torch.version.cuda < 13`
("You need pytorch with cu130 or higher to use optimized CUDA operations").
The base image ships **torch cu128**, so this image replaces it with **torch 2.11.0+cu130**
and removes the now-dead `nvidia-*-cu12` wheels. Consequences worth knowing:

* hosts must run a **CUDA 13 driver (≥ 580)** — see `.runpod/hub.json`,
  `allowedCudaVersions: ["13.0", "13.1", "13.2"]`;
* if your pool has no CUDA 13 hosts, build with
  `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128` **and** remove
  `--use-ck-attention` from `COMFY_EXTRA_ARGS` (a flag that cannot be satisfied kills the
  worker at startup). The INT8 ConvRot text encoder still works without Comfy Kitchen's
  CUDA backend — via its pure-Python eager fallback — just slower;
* Comfy Kitchen is pulled in by ComfyUI itself (`comfy-kitchen==0.2.35` in
  `requirements.txt`), it is not installed by this repo.

`--use-ck-attention` is spliced into the upstream `/start.sh` launch line through
`COMFY_EXTRA_ARGS`, so you can add other flags without rebuilding
(e.g. `COMFY_EXTRA_ARGS="--use-ck-attention --fast"`).

---

## 3. Build

```bash
docker build --platform linux/amd64 -t <registry>/worker-qwen-image-2.1:0.37.1 .
docker push <registry>/worker-qwen-image-2.1:0.37.1
# or: make build push REGISTRY=registry.example.com
```

Useful build args:

| Arg | Default | Notes |
| --- | --- | --- |
| `BASE_IMAGE` | `runpod/worker-comfyui:5.10.0-base` | any `worker-comfyui:<ver>-base` tag |
| `COMFYUI_VERSION` | `0.37.1` | any tag with `--use-ck-attention` (≥ 0.34.0); the build verifies it |
| `TORCH_INDEX_URL` | `…/whl/cu130` | `…/whl/cu128` only with `COMFY_EXTRA_ARGS` adjusted |
| `NODE_PACKAGES` | the four packs above | space separated registry ids |
| `EXTRA_NODE_PACKAGES` | empty | additional `comfy-node-install` ids |
| `BAKE_MODELS` | `false` | `true` bakes the multi-GB models (needs a 40 GB container disk) |
| `REMOVE_WORKSPACE_VENV` | `false` | `true` deletes comfy-cli's own venv (a second torch, ~6 GB) |

The build ends with two safety nets: ComfyUI's own `--quick-test-for-ci --cpu` smoke test
(imports the whole node graph, catches a node that cannot import) and
`src/verify_image.py`, which asserts cu13 torch, a usable `comfy_kitchen`, the
`--use-ck-attention` flag, all four node packs, and the baked models.

---

## 4. Deploy on RunPod

1. Create a **network volume** in the datacenter you will run in, and attach it to the
   endpoint (Serverless → Advanced → *Select Network Volume*). The worker populates it
   itself on first start; nothing to pre-stage.
   Model layout it creates:

   ```text
   /runpod-volume/models/diffusion_models/qwen_image_2.1_Q6_K.gguf
   /runpod-volume/models/text_encoders/qwen3vl_8b_int8_convrot.safetensors
   /runpod-volume/models/vae/qwen_image_2.1_vae_bf16.safetensors   (only if the VAE is missing from the image)
   ```

2. Create the template from `.runpod/hub.json` or by hand: your image, container disk
   **20 GB** (40 GB if you built with `BAKE_MODELS=true`), GPU with ≥ 24 GB VRAM,
   allowed CUDA versions **13.x**.

3. Environment variables are all optional — see `.runpod/hub.json` and the table below.

4. Send a job:

   ```bash
   curl -s https://api.runpod.ai/v2/<endpoint-id>/runsync \
     -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d @test_input.json
   ```

   `test_input.json` contains the text-to-image workflow; the response carries the PNG as a
   base64 string (or an S3 URL when the `BUCKET_*` variables are set).

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | – | gated repos / higher Hugging Face rate limits (these models are public) |
| `HF_ENDPOINT` | `https://huggingface.co` | mirror |
| `MODEL_STORAGE` | `auto` | `auto` \| `volume` \| `container` |
| `SKIP_MODEL_DOWNLOAD` | `false` | skip all model checks (volume pre-populated) |
| `MODEL_DOWNLOAD_STRICT` | `true` | exit instead of serving with missing models |
| `MODEL_DOWNLOAD_RETRIES` | `5` | download attempts per file |
| `COMFY_EXTRA_ARGS` | `--use-ck-attention` | extra flags for `main.py` |
| `COMFY_LOG_LEVEL` | `DEBUG` (upstream) | ComfyUI verbosity |
| `REFRESH_WORKER`, `BUCKET_*`, `COMFY_ORG_API_KEY`, `SERVE_API_LOCALLY` | upstream defaults | see worker-comfyui's configuration docs |

---

## 5. Models

| File | Source | Destination in ComfyUI |
| --- | --- | --- |
| `qwen_image_2.1_Q6_K.gguf` | [`AlperKTS/Qwen-Image-2.1-GGUF`](https://huggingface.co/AlperKTS/Qwen-Image-2.1-GGUF) | `models/diffusion_models/` |
| `qwen3vl_8b_int8_convrot.safetensors` | [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | `models/text_encoders/` |
| `qwen_image_2.1_vae_bf16.safetensors` | [`Comfy-Org/Qwen-Image-2.1`](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) | `models/vae/` |

Alternatives from the same repos (edit `models.json`, the manifest drives both build and
runtime): `qwen_image_2.1_Q8_0.gguf` (7.62 GB) / `qwen_image_2.1_Q5_K_M.gguf` (4.89 GB),
and `qwen3vl_8b_w4a8.safetensors` (6.31 GB) or `qwen3vl_8b_bf16.safetensors` (17.53 GB).
`qwen3vl_8b_fp8_scaled.safetensors` does **not** exist in `Comfy-Org/Qwen-Image-2.1` — the
options there are bf16, int8_convrot and w4a8.

Node packs come from the registry by id: `comfyui-easy-use`, `rgthree-comfy`,
`comfyui-kjnodes`, `comfyui-gguf`, installed with `comfy-node-install` (which fails the
build instead of silently skipping a broken node). Their `requirements.txt` files are
mirrored into `/opt/venv`, because comfy-cli installs them into its own workspace venv while
ComfyUI runs from `/opt/venv`.

---

## 6. Workflows

ComfyUI's official UI templates for this model (`image_qwen_image_2_1_t2i.json`,
`image_qwen_image_2_1_image_edit.json`) ship inside the image via the
`comfyui-workflow-templates` package; load one in the frontend and use
**Workflow → Export (API)** to get a submittable workflow.

API-format workflows are included here, already switched to the GGUF loaders:

* `workflows/qwen_image_2_1_gguf_t2i_api.json` — text-to-image, 1024×1024, 25 steps,
  cfg 1, euler/simple (the sampler settings the official template uses)
* `workflows/qwen_image_2_1_gguf_edit_api.json` — image edit; the reference image is
  passed as `input.images[0]` (`input_image_1.png`) and wired into
  `TextEncodeQwenImage21`'s `images.image_1` slot, whose `latent` output feeds the sampler

Both reference the same three model files the manifest installs, so they run unmodified.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Worker dies with "Comfy Kitchen attention is unavailable" | the worker landed on a CUDA 12.x host, or the image was rebuilt with a cu12 torch. Restrict `allowedCudaVersions` to 13.x |
| `no kernel image is available` / CUDA init failure | driver older than 580 on the host; same fix |
| Worker exits during startup with "model setup failed" | download failed (network/disk). `MODEL_DOWNLOAD_STRICT=false` starts anyway; check free space with the container-disk note above |
| `ENOSPC` while downloading | container disk too small for the container-local layout; attach a network volume |
| `Model in folder 'text_encoders' ... not found` | the volume lacks the file — let the entrypoint fetch it (`SKIP_MODEL_DOWNLOAD=false`), or check the layout in §4 |
| Models on the volume invisible | set `NETWORK_VOLUME_DEBUG=true` (upstream diagnostic) and compare with §4; only `/runpod-volume/models/...` is searched |

---

## 8. Repository layout

```text
Dockerfile                     image: ComfyUI version, node packs, torch cu130, baked models
models.json                    single source of truth for every model (bake vs runtime)
src/ensure_models.py           idempotent, resumable, atomic model fetcher (stdlib only)
src/custom-start.sh            entrypoint: prepare models, then run the upstream /start.sh
src/extra_model_paths.yaml     network-volume search paths (modern + legacy folder keys)
src/verify_image.py            build-time assertions
workflows/*_api.json           API-format workflows for the API
test_input.json                ready-to-post request body
.runpod/hub.json               RunPod template definition (disks, GPUs, CUDA, env)
```
