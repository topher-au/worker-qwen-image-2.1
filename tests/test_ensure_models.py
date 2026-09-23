"""Local harness for src/ensure_models.py: download paths, cache paths, storage guards.

Serves a fake "Hugging Face" over localhost and a fake RunPod cached-model tree,
so none of these checks need network access or a GPU.
"""
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import threading

ROOT = "/home/user/.hermes/cache/scratch/modeltest"
SCRIPT = "/opt/projects/worker-qwen-image-2.1/src/ensure_models.py"

shutil.rmtree(ROOT, ignore_errors=True)
for d in ("serve/some-repo/resolve/main/text_encoders", "cache", "cache2", "volume/models",
          "volume/huggingface-cache/hub/models--cached-repo--thing/snapshots/abc123"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)

BLOB = os.urandom(3 * 1024 * 1024 + 123)
BLOB_CACHED = os.urandom(2 * 1024 * 1024 + 7)
open(f"{ROOT}/serve/some-repo/resolve/main/text_encoders/fake.safetensors", "wb").write(BLOB)
open(f"{ROOT}/volume/huggingface-cache/hub/models--cached-repo--thing/snapshots/abc123/cached.gguf", "wb").write(BLOB_CACHED)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=f"{ROOT}/serve", **kw)

    def translate_path(self, path):
        return os.path.join(f"{ROOT}/serve", path.split("?", 1)[0].lstrip("/"))

    def log_message(self, *a):
        pass


srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

manifest = {"models": [
    {"name": "fake.safetensors", "repo": "some-repo", "path": "text_encoders/fake.safetensors",
     "dest": "text_encoders", "size": len(BLOB), "bake": False},
    {"name": "baked.safetensors", "repo": "some-repo", "path": "text_encoders/fake.safetensors",
     "dest": "vae", "size": len(BLOB), "bake": True},
]}
json.dump(manifest, open(f"{ROOT}/models.json", "w"))

cached_manifest = {"models": [
    {"name": "cached.gguf", "repo": "cached-repo/thing", "path": "cached.gguf",
     "dest": "diffusion_models", "size": len(BLOB_CACHED), "bake": False},
]}
json.dump(cached_manifest, open(f"{ROOT}/cached.json", "w"))

base_env = dict(os.environ, HF_ENDPOINT=f"http://127.0.0.1:{port}", MODEL_DOWNLOAD_RETRIES="2",
                HF_CACHE_ROOT=f"{ROOT}/volume/huggingface-cache/hub")

ok = True


def check(cond, msg):
    global ok
    print(("PASS " if cond else "FAIL ") + msg)
    ok = ok and cond


def run(manifest_path, dest_root, *args, env=None, search=()):
    cmd = [sys.executable, SCRIPT, "--manifest", manifest_path, "--dest-root", dest_root, *args]
    for s in search:
        cmd += ["--search-root", s]
    return subprocess.run(cmd, capture_output=True, text=True, env=env or base_env)


# --- download paths ---------------------------------------------------------
p = run(f"{ROOT}/models.json", f"{ROOT}/cache", "--only-baked")
check(p.returncode == 0 and os.path.exists(f"{ROOT}/cache/vae/baked.safetensors"),
      "build-time --only-baked fetches only baked entries")
check(not os.path.exists(f"{ROOT}/cache/text_encoders/fake.safetensors"),
      "build-time leaves runtime entries alone")

p = run(f"{ROOT}/models.json", f"{ROOT}/cache")
target = f"{ROOT}/cache/text_encoders/fake.safetensors"
check(p.returncode == 0 and open(target, "rb").read() == BLOB, "runtime fetch is byte-exact")
check(not any(f.endswith(".part") for f in os.listdir(f"{ROOT}/cache/text_encoders")),
      "no .part files left behind")

check("present:" in run(f"{ROOT}/models.json", f"{ROOT}/cache").stdout, "second run is a no-op")

open(target + ".part", "wb").write(BLOB[:1_000_000])
run(f"{ROOT}/models.json", f"{ROOT}/cache")
check(open(target, "rb").read() == BLOB, "resumes a truncated .part")

check("cached" not in run(f"{ROOT}/models.json", f"{ROOT}/cache", "--check").stdout,
      "--check reports ok for downloaded files")

# --- RunPod cached-models path ---------------------------------------------
p = run(f"{ROOT}/cached.json", f"{ROOT}/cache2")
link = f"{ROOT}/cache2/diffusion_models/cached.gguf"
check(p.returncode == 0 and os.path.islink(link), "HF-cache hit is adopted as a symlink")
check(open(link, "rb").read() == BLOB_CACHED, "adopted file reads back byte-exact")
check("cached.gguf" in p.stdout, "cache adoption is logged")

os.remove(link)  # drop the symlink so the copy path is exercised
p = run(f"{ROOT}/cached.json", f"{ROOT}/cache2", env=dict(base_env, MODEL_CACHE_MODE="copy"))
check(not os.path.islink(link) and open(link, "rb").read() == BLOB_CACHED,
      "MODEL_CACHE_MODE=copy materialises a real file")
shutil.rmtree(f"{ROOT}/cache2/diffusion_models")
p = run(f"{ROOT}/cached.json", f"{ROOT}/cache2", env=dict(base_env, MODEL_CACHE_MODE="off"))
check(p.returncode != 0 and not os.path.exists(link),
      "MODEL_CACHE_MODE=off ignores the cache (download fails without a route)")

p = run(f"{ROOT}/cached.json", f"{ROOT}/cache2", "--check")
check(p.returncode == 0 and "cached" in p.stdout, "--check recognises a cached-only file")

# --- storage guard ----------------------------------------------------------
# HEAD fails for the missing path, so the declared (absurd) size is what the
# space guard sees - exactly the case of a manifest that no longer matches what
# the volume/container disk can hold.
big = {"models": [{"name": "huge.safetensors", "repo": "some-repo",
                   "path": "text_encoders/missing.safetensors", "dest": "vae",
                   "size": 10**15, "bake": False}]}
json.dump(big, open(f"{ROOT}/big.json", "w"))
p = run(f"{ROOT}/big.json", f"{ROOT}/cache")
check(p.returncode != 0 and "not enough space" in p.stderr,
      "insufficient disk space fails with an actionable error")

p = run(f"{ROOT}/rootless.json", f"{ROOT}/cache")
check(p.returncode != 0 and "No such file" in p.stderr, "missing manifest fails loudly")

# --- search-root / strict behaviour ----------------------------------------
p = run(f"{ROOT}/models.json", f"{ROOT}/cache", search=[f"{ROOT}/cache"])
check(p.returncode == 0 and "present:" in p.stdout, "search-root short-circuits existing files")

bad = {"models": [{"name": "nope.safetensors", "repo": "does-not-exist", "path": "x/nope.safetensors",
                   "dest": "vae", "bake": False}]}
json.dump(bad, open(f"{ROOT}/bad.json", "w"))
p = run(f"{ROOT}/bad.json", f"{ROOT}/cache2")
check(p.returncode != 0, "unreachable model exits non-zero for strict mode")

srv.shutdown()
print("\nALL PASSED" if ok else "\nFAILURES PRESENT")
sys.exit(0 if ok else 1)
