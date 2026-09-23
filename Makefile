IMAGE  ?= worker-qwen-image-2.1
TAG    ?= 0.37.1
BASE_IMAGE ?= runpod/worker-comfyui:5.10.0-base
COMFYUI_VERSION ?= 0.37.1
BAKE_MODELS ?= false
PLATFORM ?= linux/amd64
# Tag of runpod-workers/worker-comfyui that src/start.sh was imported from.
WORKER_REF ?= 5.10.0

.PHONY: build push release check check-specs shell sync-start-sh

build:
	docker build --platform $(PLATFORM) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg COMFYUI_VERSION=$(COMFYUI_VERSION) \
		--build-arg BAKE_MODELS=$(BAKE_MODELS) \
		-t $(REGISTRY)/$(IMAGE):$(TAG) .

push:
	@test -n "$(REGISTRY)" || (echo "set REGISTRY, e.g. make push REGISTRY=ghcr.io/topher" >&2; exit 1)
	docker push $(REGISTRY)/$(IMAGE):$(TAG)

release: build push

# Re-import src/start.sh (the entrypoint COPYied into the image) from the base
# repository and re-apply this repo's local changes. Run it when WORKER_REF moves.
sync-start-sh:
	scripts/sync-start-sh.sh $(WORKER_REF)

# Static sanity checks that do not need a GPU or a build.
check:
	python3 -c "import json;[json.load(open(f)) for f in ('models.json','test_input.json','.runpod/hub.json','workflows/qwen_image_2_1_t2i_api.json','workflows/qwen_image_2_1_edit_api.json')];print('json ok')"
	bash -n src/custom-start.sh src/start.sh scripts/sync-start-sh.sh && echo "shell ok"
	python3 -m py_compile src/ensure_models.py src/safetensors_convert.py src/verify_image.py \
		scripts/check_specs.py && echo "python ok"
	python3 tests/test_ensure_models.py
	python3 tests/test_start_sh.py
	python3 tests/test_convert.py

# Validate models.json against the upstream repositories. Fetches only the
# safetensors headers and index.json (a few hundred KB) and runs the real
# converter, so it catches upstream key/shape/size changes before a deploy.
check-specs:
	python3 scripts/check_specs.py --manifest models.json

shell:
	docker run --rm -it --entrypoint /bin/bash $(REGISTRY)/$(IMAGE):$(TAG)