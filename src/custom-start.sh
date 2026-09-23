#!/usr/bin/env bash
# Entrypoint: guarantee the Qwen-Image 2.1 models exist, then hand over to the
# upstream worker entrypoint (/start.sh: GPU pre-flight check, optional SSH,
# ComfyUI, RunPod handler).
#
# Where the models live is decided here, not in the image:
#   MODEL_STORAGE=auto      -> a RunPod network volume if one is mounted, else the container
#   MODEL_STORAGE=volume    -> force /runpod-volume (fail loudly if it is not there)
#   MODEL_STORAGE=container -> force the container's own models directory
set -uo pipefail

: "${MODELS_MANIFEST:=/models.json}"
: "${MODEL_STORAGE:=auto}"
: "${CONTAINER_MODELS_DIR:=/comfyui/models}"
: "${VOLUME_PATH:=/runpod-volume}"
: "${SKIP_MODEL_DOWNLOAD:=false}"
: "${MODEL_DOWNLOAD_STRICT:=true}"

log() { printf 'worker-qwen-image: %s\n' "$*"; }

if [ "${SKIP_MODEL_DOWNLOAD}" = "true" ]; then
    log "SKIP_MODEL_DOWNLOAD=true - leaving model storage alone"
    exec /start.sh
fi

dest=""
case "${MODEL_STORAGE}" in
    volume)
        dest="${VOLUME_PATH}/models"
        if [ ! -d "${VOLUME_PATH}" ]; then
            log "ERROR: MODEL_STORAGE=volume but ${VOLUME_PATH} is not mounted"
            exit 1
        fi
        ;;
    container)
        dest="${CONTAINER_MODELS_DIR}"
        ;;
    auto)
        if [ -d "${VOLUME_PATH}" ] && [ -w "${VOLUME_PATH}" ]; then
            dest="${VOLUME_PATH}/models"
        fi
        ;;
    *)
        log "ERROR: unknown MODEL_STORAGE='${MODEL_STORAGE}' (auto|volume|container)"
        exit 1
        ;;
esac

if [ -z "${dest}" ]; then
    dest="${CONTAINER_MODELS_DIR}"
    log "no network volume at ${VOLUME_PATH}: downloading into ${dest}"
    log "note: that is the container's writable layer, so this repeats on every cold"
    log "note: start and needs a container disk of 40 GB. Attach a network volume instead."
else
    log "model storage: ${dest}"
fi

mkdir -p "${dest}"

python3 -u /ensure_models.py \
    --manifest "${MODELS_MANIFEST}" \
    --dest-root "${dest}" \
    --search-root "${CONTAINER_MODELS_DIR}"
rc=$?

if [ "${rc}" -ne 0 ]; then
    if [ "${MODEL_DOWNLOAD_STRICT}" = "true" ]; then
        log "ERROR: model setup failed (exit ${rc}) - refusing to start a worker that cannot serve jobs"
        log "note: set MODEL_DOWNLOAD_STRICT=false to start anyway"
        exit "${rc}"
    fi
    log "WARNING: model setup failed (exit ${rc}); continuing because MODEL_DOWNLOAD_STRICT=false"
fi

exec /start.sh
