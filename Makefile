IMAGE  ?= worker-qwen-image-2.1
TAG    ?= 0.37.1
COMFYUI_VERSION ?= 0.37.1
BAKE_MODELS ?= false
PLATFORM ?= linux/amd64

.PHONY: build push release check shell

build:
	docker build --platform $(PLATFORM) \
		--build-arg COMFYUI_VERSION=$(COMFYUI_VERSION) \
		--build-arg BAKE_MODELS=$(BAKE_MODELS) \
		-t $(REGISTRY)/$(IMAGE):$(TAG) .

push:
	@test -n "$(REGISTRY)" || (echo "set REGISTRY, e.g. make push REGISTRY=ghcr.io/topher" >&2; exit 1)
	docker push $(REGISTRY)/$(IMAGE):$(TAG)

release: build push

# Static sanity checks that do not need a GPU or a build.
check:
	python3 -c "import json;[json.load(open(f)) for f in ('models.json','test_input.json','.runpod/hub.json','workflows/qwen_image_2_1_gguf_t2i_api.json','workflows/qwen_image_2_1_gguf_edit_api.json')];print('json ok')"
	bash -n src/custom-start.sh && echo "shell ok"
	python3 -m py_compile src/ensure_models.py src/verify_image.py && echo "python ok"
	python3 tests/test_ensure_models.py

shell:
	docker run --rm -it --entrypoint /bin/bash $(REGISTRY)/$(IMAGE):$(TAG)
