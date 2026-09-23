#!/usr/bin/env bash
# Import the upstream worker entrypoint into this repository.
#
#   scripts/sync-start-sh.sh [REF]
#
# REF is a tag, branch or commit of runpod-workers/worker-comfyui (default:
# 5.10.0, the tag the default BASE_IMAGE is built from). The upstream
# src/start.sh is downloaded and this repository's local changes are applied:
#
#   1. a CK_ATTENTION block that enables Comfy Kitchen INT8 attention by
#      default, so ComfyUI is always started with --use-ck-attention
#   2. ${COMFY_EXTRA_ARGS:-} on the two ComfyUI launch lines, for extra flags
#
# The result is written to src/start.sh and COPYied into the image, so what runs
# as /start.sh is reviewable here and in the git history instead of being
# patched by sed during a build.
#
# The script is idempotent and self-checking: it refuses to write anything when
# the patch no longer applies cleanly, and it verifies that reverse-applying the
# patch reproduces the upstream file byte for byte.
set -euo pipefail

REF="${1:-${REF:-5.10.0}}"
REPO="${REPO:-runpod-workers/worker-comfyui}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/src/start.sh"
URL="https://raw.githubusercontent.com/${REPO}/${REF}/src/start.sh"

ANCHOR='echo "worker-comfyui: Starting ComfyUI"'
TARGET='python -u /comfyui/main.py --disable-auto-launch'
HOOK='python -u /comfyui/main.py ${CK_ATTENTION_ARG} ${COMFY_EXTRA_ARGS:-} --disable-auto-launch'
BLOCK_FIRST='# Enable Comfy Kitchen INT8 attention'
BLOCK_LAST='esac'

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

echo "fetching ${URL}"
curl -fsSL "${URL}" -o "${tmp}/upstream.sh"
bash -n "${tmp}/upstream.sh"

grep -q -F "${ANCHOR}" "${tmp}/upstream.sh" || {
    echo "ERROR: upstream ${REPO}@${REF} no longer contains the anchor line:" >&2
    echo "       ${ANCHOR}" >&2
    exit 1
}

cat > "${tmp}/block.sh" <<'EOF'
# Enable Comfy Kitchen INT8 attention. This is what puts --use-ck-attention on
# ComfyUI's command line; it needs a Comfy Kitchen build with CUDA support
# (torch cu13+, see README "CUDA 13 is required"). Set CK_ATTENTION=false to
# launch without it, for example on a cu12 torch build.
case "${CK_ATTENTION:-true}" in
    true) CK_ATTENTION_ARG="--use-ck-attention" ;;
    false) CK_ATTENTION_ARG="" ;;
    *) echo "worker-comfyui: CK_ATTENTION must be true or false, got '${CK_ATTENTION}' -" \
            "launching without --use-ck-attention" >&2
       CK_ATTENTION_ARG="" ;;
esac
EOF

# The block is read from the file rather than passed with -v so its bytes survive
# (awk -v would eat the line continuation above).
awk -v anchor="${ANCHOR}" -v blockfile="${tmp}/block.sh" '
    { print }
    !inserted && $0 == anchor {
        while ((getline line < blockfile) > 0) print line
        close(blockfile)
        inserted = 1
    }
    END { if (!inserted) exit 1 }
' "${tmp}/upstream.sh" > "${tmp}/patched.sh"

sed "s|${TARGET}|${HOOK}|g" "${tmp}/patched.sh" > "${tmp}/final.sh"

# awk's print adds a trailing newline that upstream's file may not have; mirror
# upstream's exact byte ending so the round-trip check below stays byte-exact.
python3 - "${tmp}/upstream.sh" "${tmp}/final.sh" <<'PY'
import sys
up, fin = sys.argv[1], sys.argv[2]
want_newline = open(up, "rb").read().endswith(b"\n")
data = open(fin, "rb").read().rstrip(b"\n")
open(fin, "wb").write(data + (b"\n" if want_newline else b""))
PY

check_count() {  # needle expected
    local count
    count="$(grep -c -F "$1" "${tmp}/final.sh" || true)"
    if [ "${count}" -ne "$2" ]; then
        echo "ERROR: expected ${2}x '$1' after patching, found ${count}." >&2
        echo "       ${REPO}@${REF} changed shape - review src/start.sh manually." >&2
        exit 1
    fi
}
check_count "${BLOCK_FIRST}" 1
check_count 'CK_ATTENTION_ARG="--use-ck-attention"' 1
check_count 'CK_ATTENTION_ARG=""' 2
check_count '${CK_ATTENTION_ARG}' 2
check_count '${COMFY_EXTRA_ARGS:-}' 2
bash -n "${tmp}/final.sh"

# Self-check: undoing the patch has to reproduce the upstream file exactly.
sed -e "/^${BLOCK_FIRST}/,/^${BLOCK_LAST}\$/d" \
    -e "s|${HOOK}|${TARGET}|g" "${tmp}/final.sh" > "${tmp}/roundtrip.sh"
if ! cmp -s "${tmp}/roundtrip.sh" "${tmp}/upstream.sh"; then
    echo "ERROR: reverse-applying the patch does not reproduce upstream." >&2
    diff -u "${tmp}/upstream.sh" "${tmp}/roundtrip.sh" >&2 || true
    exit 1
fi
echo "patch round-trips cleanly against ${REPO}@${REF}"

if [ -f "${DEST}" ] && cmp -s "${tmp}/final.sh" "${DEST}"; then
    echo "src/start.sh already matches ${REPO}@${REF} plus the local changes"
    exit 0
fi

cp "${tmp}/final.sh" "${DEST}"
chmod 0755 "${DEST}"
echo "wrote src/start.sh from ${REPO}@${REF}"
diff -u "${tmp}/upstream.sh" "${DEST}" || true
