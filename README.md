# worker-qwen-image-2.1

A RunPod **serverless** worker for **Qwen-Image 2.1** on ComfyUI, built on
[`runpod-workers/worker-comfyui`](https://github.com/runpod-workers/worker-comfyui)
(the `-base` image + its customization guide).

What is inside:

* **ComfyUI 0.37.1** (latest release tag), launched with **`--use-ck-attention`**
* **ComfyUI-Easy-Use**, **rgthree-comfy**, **ComfyUI-KJNodes**, **ComfyUI-GGUF**
* **Qwen-Image 2.1** 20B transformer (bf16) built from the original
  [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1) weights
* **Qwen3-VL 8B** text encoder (bf16) and the **Qwen-Image 2.1 VAE** (fp32) from
  the same repository, converted to ComfyUI checkpoints on the worker

The original repository ships **diffusers format** (sharded safetensors plus
`index.json`): it cannot be handed to ComfyUI as-is, which is why this worker
carries `src/safetensors_convert.py`. See §5.

---

## 1. Model storage: what is baked, what is fetched, and why

The numbers decide this. Measured against the upstream repositories, not guessed:

| Item | Size |
| --- | --- |
| `runpod/worker-comfyui:5.10.0-base` (ComfyUI + torch, compressed image) | 14.67 GB |
| Qwen-Image 2.1 transformer, bf16 → `qwen_image_2.1_bf16.safetensors` | 14.23 GB |
| Qwen3-VL 8B text encoder, bf16 → `qwen3vl_8b_bf16.safetensors` | 17.53 GB |
| Qwen-Image 2.1 VAE, fp32 → `qwen_image_2.1_vae_fp32.safetensors` | 1.35 GB |
| **models on disk (the three checkpoints the workflows load)** | **33.12 GB** |

A RunPod serverless template's container disk defaults to **20 GB**, and that disk holds the
image *plus* the writable layer — 33 GB of models never fits there. So nothing is baked by
default and the models are **converted at first start** into
`/runpod-volume/models/…` (network volume) or, without a volume, into the container's
writable layer.

The fetch/build happens in this worker's entrypoint (`src/custom-start.sh` →
`src/ensure_models.py` → `src/safetensors_convert.py`) **before** ComfyUI starts:

* the first worker of the endpoint pays for the download and the conversion, every later
  worker (and every subsequent cold start, and every other endpoint sharing the volume)
  starts instantly;
* the image stays ~15.5 GB, comfortably inside the 20 GB container disk;
* without a volume everything is redone on every cold start, and that needs
  `containerDiskInGb: 40`.

Prefer to build them into the image anyway (`docker build --build-arg BAKE_MODELS=true`,
plus `containerDiskInGb: 40`)? Also supported — the same manifest drives both paths.

Downloads are staged as `<file>.part` and `os.replace()`d into place, and conversions are
written the same way, so a worker that dies mid-way never leaves a truncated file that
ComfyUI would load and then crash on. Workers sharing a volume serialise on an `flock`ed
lock file; where the filesystem has no `flock` the atomic rename is still what keeps things
consistent. Expected sizes come from Hugging Face (HEAD request), so a re-uploaded file
warns instead of breaking a cold start. A free-space check runs before every download or
conversion and names the fix (volume / 40 GB container disk / `BAKE_MODELS=true`) instead
of dying on `ENOSPC`. Downloaded source shards are deleted after a successful conversion
(`MODEL_SOURCES_KEEP=true` keeps them), so the peak footprint is 33 GB, not 66 GB.

---

## 1b. Cold-boot cost: exactly what gets downloaded and built

| Cold boot | Work done | Bytes over the network |
| --- | --- | --- |
| First start, `Qwen/Qwen-Image-2.1` set as the endpoint's cached model, volume attached | read the 33.12 GB of shards from the host's local HF cache, convert → volume | **0** |
| Any later worker / cold boot on that endpoint (volume attached) | verify the three outputs (header check, milliseconds) | **0** |
| Volume attached, nothing cached | download 33.12 GB of shards, convert → volume, delete the shards | 33.12 GB, once |
| Same host, no network volume | everything in the container's writable layer, redone every cold boot | 33.12 GB **every** cold boot |
| `BAKE_MODELS=true` image | nothing at runtime | **0, ever** |

Nothing else is downloaded: ComfyUI's node packs, the frontend, `comfy-kitchen` and the
source shards above are the whole list. Run
`python3 /ensure_models.py --manifest /models.json --dest-root /comfyui/models --check`
inside a running worker for the current state (`ok` / `DAMAGED` / `MISSING`, with the
RunPod-cache hit when there is one).

The conversion is a single sequential pass — read 33 GB, write 33 GB — so it is disk-bound,
not GPU-bound: expect roughly a minute or two on a volume at 200–400 MB/s, plus the read of
the source shards. No GPU is needed for it.

### Three ways to make that first start cheap or free

1. **RunPod "cached models"** (Serverless → the endpoint's *Model* field). RunPod
   pre-downloads one Hugging Face repo onto the host's local disk **before** the worker
   starts, does **not bill** for it, and prefers scheduling workers onto hosts that already
   hold it — the fastest cold start available. Set it to **`Qwen/Qwen-Image-2.1`** (33.1 GB
   of shards + the small config/processor JSONs). The repo is unpacked into the standard HF
   cache layout:

   ```text
   /runpod-volume/huggingface-cache/hub/models--Qwen--Qwen-Image-2.1/snapshots/<rev>/transformer/…
   ```

   That is *not* where ComfyUI looks, so `src/ensure_models.py` searches those roots and
   reads the shards **in place** (no second copy of 33 GB), logging the full path of every
   source it uses. `MODEL_CACHE_MODE=off` disables the lookup; `copy` materialises real
   files instead of symlinks (only relevant to single-file entries).

   Limits to keep in mind: **one cached model per endpoint**, and RunPod downloads *the
   whole repo*, which is why this manifest is built around exactly that repo.

2. **Network volume (for the 33 GB of output).** The converted checkpoints have to live
   somewhere that survives a cold start, and `$0.07/GB/month` standard-tier storage
   (50 GB ≈ $3.50/mo) is what makes the second boot free. NVMe-backed at 200–400 MB/s
   typical, so each *new worker* spends ~1–3 minutes reading weights off the volume before
   its first inference. Trade-off: the endpoint is pinned to the volume's datacenter.
   Several workers writing the same volume is what the `flock` in `src/ensure_models.py`
   covers.

3. **Bake everything** (`--build-arg BAKE_MODELS=true` + `containerDiskInGb: 40`). Zero
   runtime work and reads come off local NVMe, at the cost of a ~48 GB image (slower host
   image pulls, no swapping models without a rebuild).

Also worth setting: RunPod marks a worker **unhealthy if the cold start exceeds 7 minutes**
(`RUNPOD_INIT_TIMEOUT=800` extends it, and the template already sets it) — relevant for the
no-cache/no-volume case, where the first start downloads 33 GB. And `active workers ≥ 1`
removes cold starts entirely, at the cost of an always-on worker.

**Recommended configuration here:** cached model `Qwen/Qwen-Image-2.1` **plus** a 50 GB
network volume, `RUNPOD_INIT_TIMEOUT=800`. Then exactly one worker ever converts (no network
I/O), and every cold start afterwards reads 33 GB off the volume.

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
  `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128` **and** set
  `CK_ATTENTION=false` (a flag that cannot be satisfied kills the worker at startup).
  The INT8 ConvRot text encoder still works without Comfy Kitchen's CUDA backend —
  via its pure-Python eager fallback — just slower;
* Comfy Kitchen is pulled in by ComfyUI itself (`comfy-kitchen==0.2.35` in
  `requirements.txt`), it is not installed by this repo.

### How the flag reaches ComfyUI

`src/start.sh` is **imported from the base repository** (`runpod-workers/worker-comfyui`,
`src/start.sh`) and `COPY`ied into the image as `/start.sh`, wrapped by
`src/custom-start.sh`. It carries two local changes, both greppable in that one file:

```bash
# Enable Comfy Kitchen INT8 attention. ...
case "${CK_ATTENTION:-true}" in
    true) CK_ATTENTION_ARG="--use-ck-attention" ;;
    false) CK_ATTENTION_ARG="" ;;
    *) echo "worker-comfyui: CK_ATTENTION must be true or false, got '${CK_ATTENTION}' -" \
            "launching without --use-ck-attention" >&2
       CK_ATTENTION_ARG="" ;;
esac
...
python -u /comfyui/main.py ${CK_ATTENTION_ARG} ${COMFY_EXTRA_ARGS:-} --disable-auto-launch ...
```

So ComfyUI is started with `--use-ck-attention` by default, `CK_ATTENTION=false` turns it
off, and `COMFY_EXTRA_ARGS="--fast"` appends anything else — no rebuild needed. The file is
vendored rather than patched at build time so what runs is reviewable here and in the git
history; when `BASE_IMAGE` moves, re-import it:

```bash
make sync-start-sh WORKER_REF=5.11.0        # or: scripts/sync-start-sh.sh 5.11.0
```

That script refuses to write anything if the patch stops applying and verifies that
reverse-applying its patch reproduces the upstream file byte for byte. The build then checks
the vendored file again (both launch lines wired, `bash -n` clean) and prints a diff — without
failing — if the base image's own `/start.sh` has drifted from the vendored copy.

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
| `TORCH_INDEX_URL` | `…/whl/cu130` | `…/whl/cu128` only with `CK_ATTENTION=false` |
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
| `MODEL_CACHE_MODE` | `link` | use RunPod cached models: `link` \| `copy` \| `off` |
| `MODEL_HF_CACHE_ROOT`, `HF_CACHE_ROOT`, `HUGGINGFACE_HUB_CACHE`, `HF_HOME` | `/runpod-volume/huggingface-cache/hub`, also `/root/.cache/huggingface/hub` | roots searched for cached models / source shards |
| `MODEL_SOURCES_KEEP` | `false` | keep downloaded source shards after a conversion (costs 33 GB more) |
| `MODEL_SOURCES_DIR` | `model-sources` | where those shards are staged (next to the models root) |
| `MODEL_DOWNLOAD_RETRIES` | `5` | download attempts per file |
| `RUNPOD_INIT_TIMEOUT` | – | seconds before an unfinished cold start is marked unhealthy (`800` is set by the template) |
| `CK_ATTENTION` | `true` | start ComfyUI with `--use-ck-attention` (set `false` only on a cu12 torch build) |
| `COMFY_EXTRA_ARGS` | empty | extra flags appended to `main.py` (e.g. `--fast`) |
| `COMFY_LOG_LEVEL` | `DEBUG` (upstream) | ComfyUI verbosity |
| `REFRESH_WORKER`, `BUCKET_*`, `COMFY_ORG_API_KEY`, `SERVE_API_LOCALLY` | upstream defaults | see worker-comfyui's configuration docs |

---

## 5. Models: converted from the original Qwen repository

Every model comes from [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1).
That repository is **diffusers format** — sharded safetensors plus an `index.json` per
component — and ComfyUI needs single-file checkpoints in its own key naming. So the worker
converts them, byte for byte, on the first start (`src/safetensors_convert.py`):

| Output file | Destination | Sources in `Qwen/Qwen-Image-2.1` | Conversion |
| --- | --- | --- | --- |
| `qwen_image_2.1_bf16.safetensors` (14.23 GB, 265 tensors) | `models/diffusion_models/` | `transformer/diffusion_pytorch_model-0000{1,2}-of-00002.safetensors` | merge 2 shards; fuse the SwiGLU FFN (`img_mlp.gate_layer` + `img_mlp.proj` → `img_mlp.gate_up`, row order) |
| `qwen3vl_8b_bf16.safetensors` (17.53 GB, 750 tensors) | `models/text_encoders/` | `text_encoder/model-0000{1..4}-of-00004.safetensors` | merge 4 shards; rename `model.language_model.*` → `model.*` |
| `qwen_image_2.1_vae_fp32.safetensors` (1.35 GB, 238 tensors) | `models/vae/` | `vae/diffusion_pytorch_model.safetensors` | rename diffusers → ComfyUI's `WanVAE` naming; insert the unit axis the 3D kernels expect (`[C, C, 3, 3]` → `[C, C, 1, 3, 3]`) |

Why those transformations and no others:

* **Fusion order is not a guess.** ComfyUI's Qwen-Image 2.1 FFN is a fused SwiGLU
  (`img_mlp.gate_up`), the original ships the two halves separately; concatenating
  `gate_layer` then `proj` along dim 0 was compared tensor-by-tensor against Comfy-Org's
  repackage and is byte-identical (checked on blocks 0, 5 and 31, i.e. across both shards).
* **The text encoder rename reproduces Comfy-Org's file exactly** — 750 keys, same shapes,
  same key order, and ComfyUI's own loader applies the same prefix map at load time, so a
  renamed file loads identically.
* **The VAE mapping was verified by value**, not by name: for 14 key families spanning
  middle/down/upsample blocks, the head and the quant convs, every fp32 element of the
  original is within one ULP of Comfy-Org's bf16 checkpoint. It is kept in fp32 (upstream
  precision, 1.35 GB rather than 0.68 GB).
* **Nothing is recomputed or re-quantised.** Every tensor is copied from its source offsets;
  the only non-copy operations are concatenating two tensors, renaming keys and inserting a
  unit axis.

### Verifying the manifest without downloading 33 GB

```bash
make check-specs        # or: python3 scripts/check_specs.py
```

`scripts/check_specs.py` fetches only the safetensors **headers** and `index.json` from
Hugging Face (a few hundred KB of HTTP range requests), writes sparse placeholders, and runs
the same converter the worker uses. It fails if any of it stops matching:

```text
=== qwen_image_2.1_bf16.safetensors  (Qwen/Qwen-Image-2.1) ===
  transformer/…-00001-of-00002.safetensors: 211 tensors, 9,968,332,504 bytes (sparse)
  transformer/…-00002-of-00002.safetensors: 86 tensors, 4,261,951,904 bytes (sparse)
  index.json: 297 keys, total_size=14230249472 (matches)
  layout: 265 tensors, sha256 e8585f9dac243ab0…, expect 265 / e8585f9dac243ab0…  -> OK
```

The manifest also pins the **output layout** (`tensors`, `dtype`, `data_bytes`,
`layout_sha256`) for each model, and the converter refuses to publish a file that does not
match. That is what makes a first-start conversion safe: a wrong mapping fails loudly instead
of producing a plausible-looking checkpoint. On later starts the output's header is re-checked
against the same block (no source files needed), so a truncated or stale file is rebuilt
instead of being served.

Alternatives, if you would rather not carry the conversion: `Comfy-Org/Qwen-Image-2.1` ships
ready-made `diffusion_models/qwen_image_2.1_bf16.safetensors` (14.23 GB) and
`qwen_image_2.1_int8_convrot.safetensors` (7.26 GB), `text_encoders/qwen3vl_8b_bf16.safetensors`
(17.53 GB) and `qwen3vl_8b_int8_convrot.safetensors` (9.35 GB), plus
`qwen_image_2.1_vae_bf16.safetensors` (0.68 GB) — add them to `models.json` as plain entries
(no `convert` block, with `path`) and they are fetched or adopted from the RunPod cache
instead. (`qwen3vl_8b_fp8_scaled.safetensors` does **not** exist there; the options are bf16,
int8_convrot and w4a8.)

Node packs come from the registry by id: `comfyui-easy-use`, `rgthree-comfy`,
`comfyui-kjnodes`, `comfyui-gguf`, installed with `comfy-node-install` (which fails the
build instead of silently skipping a broken node). Their `requirements.txt` files are
mirrored into `/opt/venv`, because comfy-cli installs them into its own workspace venv while
ComfyUI runs from `/opt/venv`. ComfyUI-GGUF is kept so GGUF quants remain an option.

---

## 6. Workflows

ComfyUI's official UI templates for this model (`image_qwen_image_2_1_t2i.json`,
`image_qwen_image_2_1_image_edit.json`) ship inside the image via the
`comfyui-workflow-templates` package; load one in the frontend and use
**Workflow → Export (API)** to get a submittable workflow.

API-format workflows are included here, using the converted checkpoints:

* `workflows/qwen_image_2_1_t2i_api.json` — text-to-image, 1024×1024, 25 steps,
  cfg 1, euler/simple (the sampler settings the official template uses)
* `workflows/qwen_image_2_1_edit_api.json` — image edit; the reference image is
  passed as `input.images[0]` (`input_image_1.png`) and wired into
  `TextEncodeQwenImage21`'s `images.image_1` slot, whose `latent` output feeds the sampler

Both load `qwen_image_2.1_bf16.safetensors` with **UNETLoader** (not the GGUF loader
ComfyUI-GGUF provides), `qwen3vl_8b_bf16.safetensors` with **CLIPLoader** `type: qwen_image`
— which is the Qwen3-VL 8B path ComfyUI 0.37 uses for this model — and
`qwen_image_2.1_vae_fp32.safetensors` with **VAELoader**. They reference the same three files
the manifest builds, so they run unmodified.

---

## 7. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Worker dies with "Comfy Kitchen attention is unavailable" | the worker landed on a CUDA 12.x host, or the image was rebuilt with a cu12 torch. Restrict `allowedCudaVersions` to 13.x |
| `no kernel image is available` / CUDA init failure | driver older than 580 on the host; same fix |
| Worker exits during startup with "model setup failed" | download or conversion failed. `MODEL_DOWNLOAD_STRICT=false` starts anyway; check free space with the container-disk note above |
| `not enough space in /runpod-volume/models` | the volume/container disk cannot hold the 33 GB of output (plus 33 GB of shards while converting, unless they come from the RunPod cache) — raise the volume, set `containerDiskInGb: 40`, or build with `BAKE_MODELS=true` |
| `converted layout does not match the manifest` | the upstream files changed (`make check-specs` shows what), or a source shard was damaged mid-download. Re-run; if it persists, re-check the spec against upstream |
| `index.json says total_size …, models.json says …` | upstream re-uploaded the shards: run `make check-specs`, then update `models.json` |
| `rename '…' matched nothing - upstream keys changed?` | same cause at key level; the renamed key set no longer exists upstream |
| Cached model configured but the worker still downloads | check the repo is on the endpoint's *Model* field (one per endpoint) and that the log says `source cache /…/models--Qwen--Qwen-Image-2.1/snapshots/…`; `MODEL_CACHE_MODE=off` disables the lookup |
| Cold start killed as unhealthy | first start without a cache downloads 33 GB and converts it — set `RUNPOD_INIT_TIMEOUT=800` (the template does) or use the cached model + volume |
| Build warns that the base image's `/start.sh` differs from the vendored one | `BASE_IMAGE` moved ahead of `src/start.sh`; re-import with `make sync-start-sh WORKER_REF=<tag>` |
| Log says `CK_ATTENTION must be true or false` | a value other than lowercase `true`/`false` was set; the worker launches without `--use-ck-attention` rather than failing |
| `Model in folder 'diffusion_models' ... not found` | the volume lacks the converted checkpoint — let the entrypoint build it (`SKIP_MODEL_DOWNLOAD=false`), or check the layout in §4 |
| Models on the volume invisible | set `NETWORK_VOLUME_DEBUG=true` (upstream diagnostic) and compare with §4; only `/runpod-volume/models/...` is searched |

---

## 8. Repository layout

```text
Dockerfile                     image: ComfyUI version, node packs, torch cu130, model wiring
models.json                    single source of truth: sources, conversion rules, expected layout
src/safetensors_convert.py     diffusers -> ComfyUI conversion (merge, rename, fuse, reshape)
src/ensure_models.py           idempotent, resumable, atomic fetcher (stdlib only)
src/start.sh                   entrypoint vendored from the base repo; enables CK attention
src/custom-start.sh            wrapper: prepare models, then run the vendored /start.sh
src/extra_model_paths.yaml     network-volume search paths (modern + legacy folder keys)
src/verify_image.py            build-time assertions (incl. the /start.sh CK wiring)
scripts/sync-start-sh.sh       re-import src/start.sh from the base repo + apply the patch
scripts/check_specs.py         verify models.json against upstream headers (no 33 GB download)
tests/test_convert.py          conversion + first-start build driven by a fake RunPod cache
tests/test_ensure_models.py    local harness: download, resume, cache adoption, storage guard
tests/test_start_sh.py         local harness: the real launch line yields the right argv
workflows/*_api.json           API-format workflows for the API
test_input.json                ready-to-post request body
.runpod/hub.json               RunPod template definition (disks, GPUs, CUDA, env)
```

`make check` runs everything that does not need a GPU or a Docker build;
`make check-specs` needs network access and validates the conversion rules against
the live Hugging Face repositories.
