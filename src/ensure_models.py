#!/usr/bin/env python3
"""Make sure every model the worker needs is on disk before ComfyUI starts.

Used twice, from the same manifest (models.json):

  * at image build time  -> ``--only-baked`` fetches the small, immutable files
  * on the worker's first start -> everything else is fetched, ideally onto a
    RunPod network volume so only the first worker of an endpoint pays for it

Design notes
------------
* A file already staged by RunPod's *cached models* feature is reused instead of
  downloaded: those land in the Hugging Face cache layout under
  /runpod-volume/huggingface-cache/hub, and ``adopt()`` links (or copies) them
  into ComfyUI's model directories, so the host-local copy is what gets read.
* ``.part`` staging + ``os.replace()``: the rename is atomic on one filesystem,
  so a worker that dies mid-download never leaves a truncated file that ComfyUI
  would happily mmap and then crash on.
* A ``flock``ed lock file serialises workers that share a network volume. Some
  NFS mounts do not implement flock; that is logged and ignored, because the
  atomic rename is what actually keeps a shared volume consistent.
* The expected size comes from Hugging Face (HEAD request), not from the
  manifest: if upstream re-uploads a file the manifest only warns instead of
  breaking a cold start. Size verification is the only integrity check that is
  cheap enough to run on every start.
* Only the Python standard library is used - this runs before the venv is set up
  for anything else and must never break the worker because of a missing dep.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

CHUNK = 8 * 1024 * 1024
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
LOCK_TIMEOUT_S = float(os.environ.get("MODEL_LOCK_TIMEOUT_S", "1800"))
RETRIES = int(os.environ.get("MODEL_DOWNLOAD_RETRIES", "5"))
LOCK_WAIT_S = float(os.environ.get("MODEL_LOCK_WAIT_S", "3600"))
# RunPod's "cached models" feature stages whole Hugging Face repos here, on the
# host's local disk, before the worker starts. Using them avoids the download
# entirely - see README "Cold-boot cost".
CACHE_MODE = os.environ.get("MODEL_CACHE_MODE", "link")  # link | copy | off
HF_CACHE_ROOT = os.environ.get("HF_CACHE_ROOT", "/runpod-volume/huggingface-cache/hub")


def log(msg: str) -> None:
    print(f"ensure-models: {msg}", flush=True)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


def headers(token: str | None) -> dict[str, str]:
    h = {"User-Agent": "runpod-worker-qwen-image-2.1"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def remote_size(url: str, token: str | None) -> int | None:
    """Content-Length of the (redirected) file, or None when unavailable."""
    req = urllib.request.Request(url, method="HEAD", headers=headers(token))
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length else None
    except Exception as exc:  # gated repo, offline, HEAD unsupported, ...
        log(f"warning: could not read remote size for {url}: {exc}")
        return None


def download(url: str, dest: str, token: str | None, expected: int | None) -> None:
    """Stream `url` into `dest` atomically, resuming a previous .part file."""
    tmp = dest + ".part"
    for attempt in range(1, RETRIES + 1):
        done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if expected is not None and done >= expected:
            break
        req_headers = headers(token)
        mode = "wb"
        if done:
            req_headers["Range"] = f"bytes={done}-"
            mode = "ab"
            log(f"resuming {os.path.basename(dest)} at {human(done)} (attempt {attempt}/{RETRIES})")
        else:
            log(f"downloading {os.path.basename(dest)} (attempt {attempt}/{RETRIES})")
        req = urllib.request.Request(url, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if done and r.status != 206:  # server ignored the Range header
                    log("server does not support resume, restarting the download")
                    mode, done = "wb", 0
                total = expected
                if total is None or done:
                    cl = r.headers.get("Content-Length")
                    if cl:
                        total = int(cl) + done
                start = time.monotonic()
                last_report = 0.0
                with open(tmp, mode) as fh:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 15:
                            last_report = now
                            rate = done / max(now - start, 1e-6)
                            pct = f" ({100 * done / total:.1f}%)" if total else ""
                            log(f"  {human(done)}{pct} @ {human(rate)}/s")
                    fh.flush()
                    os.fsync(fh.fileno())
            if expected is None or os.path.getsize(tmp) == expected:
                os.replace(tmp, dest)
                log(f"ready: {dest} ({human(os.path.getsize(dest))})")
                return
            log(f"warning: {os.path.basename(dest)} is {os.path.getsize(tmp)} bytes, expected {expected}")
            if os.path.exists(tmp):
                os.remove(tmp)  # a stale/partial file makes no sense to resume
        except Exception as exc:
            log(f"attempt {attempt}/{RETRIES} failed: {exc}")
        if attempt < RETRIES:
            time.sleep(min(2**attempt * 3, 60))
    raise RuntimeError(f"could not download {url} -> {dest}")


def cached_candidates(entry: dict) -> list[str]:
    """Files already staged on this host by RunPod's cached-models feature.

    RunPod unpacks a cached Hugging Face repo into the standard HF cache layout
    under /runpod-volume/huggingface-cache/hub, with slashes in the repo name
    replaced by "--" and one directory per revision:

        models--<org>--<name>/snapshots/<hash>/<path inside the repo>

    Returns the matching paths (newest snapshot last), or [] when the repo is
    not cached on this host.
    """
    if CACHE_MODE == "off":
        return []
    snapshots = os.path.join(HF_CACHE_ROOT, "models--" + entry["repo"].replace("/", "--"), "snapshots")
    if not os.path.isdir(snapshots):
        return []
    hits = []
    for snap in sorted(os.listdir(snapshots)):
        path = os.path.join(snapshots, snap, entry["path"])
        if os.path.isfile(path):
            hits.append(path)
    return hits


def ensure_space(root: str, need: int) -> None:
    """Fail early, with an actionable message, instead of dying on ENOSPC."""
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        return
    if free < need * 1.05:
        raise RuntimeError(
            f"not enough space in {root}: {human(free)} free, {human(need)} needed. "
            "Attach a network volume (models land on /runpod-volume/models), raise the "
            "container disk to 40 GB, or bake the models into the image (BAKE_MODELS=true)."
        )


def adopt(src: str, dest: str) -> None:
    """Make a cached file usable from ComfyUI's model directories."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.islink(dest) or os.path.exists(dest):
        os.remove(dest)
    size = os.path.getsize(src)
    if CACHE_MODE == "copy":
        ensure_space(os.path.dirname(dest), size)
        shutil.copy2(src, dest)
        log(f"copied cached model: {src} -> {dest} ({human(size)})")
        return
    try:
        os.symlink(src, dest)
        log(f"linked cached model: {dest} -> {src} ({human(size)})")
    except OSError as exc:  # filesystem without symlinks: fall back to a copy
        log(f"note: symlink failed ({exc}), copying instead")
        ensure_space(os.path.dirname(dest), size)
        shutil.copy2(src, dest)
        log(f"copied cached model: {src} -> {dest} ({human(size)})")


def resolve(token: str | None, entry: dict, roots: list[str], dest_root: str) -> None:
    name, subdir = entry["name"], entry["dest"]
    present = [os.path.join(root, subdir, name) for root in roots]
    for path in present:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            log(f"present: {path} ({human(os.path.getsize(path))})")
            if path != os.path.join(dest_root, subdir, name):
                log(f"  (using the copy that is already on disk instead of {dest_root})")
            return
    dest = os.path.join(dest_root, subdir, name)
    cached = cached_candidates(entry)
    if cached:
        adopt(cached[-1], dest)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{HF_ENDPOINT}/{entry['repo']}/resolve/main/{entry['path']}?download=true"
    size = remote_size(url, token)
    declared = entry.get("size")
    if size and declared and size != declared:
        log(f"warning: {name} is {size} bytes upstream, manifest says {declared} - update models.json")
    need = size or declared or 0
    if need:
        ensure_space(os.path.dirname(dest), need)
    download(url, dest, token, size or declared)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dest-root", required=True,
                    help="models root downloads go into (models/<dest>/<file>)")
    ap.add_argument("--search-root", action="append", default=[],
                    help="additional models roots that already satisfy an entry")
    ap.add_argument("--only-baked", action="store_true",
                    help="process only entries with \"bake\": true (build time)")
    ap.add_argument("--all", action="store_true",
                    help="process every entry regardless of \"bake\"")
    ap.add_argument("--check", action="store_true",
                    help="report status without downloading anything")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    entries = manifest["models"]

    if args.check:
        missing = 0
        for e in entries:
            path = os.path.join(args.dest_root, e["dest"], e["name"])
            cached = cached_candidates(e)
            if os.path.exists(path):
                log(f"ok      {path}")
            elif cached:
                log(f"cached  {path} (usable from {cached[-1]})")
            else:
                missing += 1
                log(f"MISSING {path}")
        return 1 if missing else 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    roots = [args.dest_root] + [r for r in args.search_root if r != args.dest_root]

    lock_path = os.path.join(args.dest_root, ".ensure-models.lock")
    os.makedirs(args.dest_root, exist_ok=True)
    lock_fh = open(lock_path, "w")
    locked = False
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        log("acquired the download lock")
    except OSError as exc:
        log(f"note: could not take the download lock ({exc}); "
            "waiting up to %ds, another worker may be fetching the same files" % LOCK_WAIT_S)
        deadline = time.monotonic() + LOCK_WAIT_S
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                time.sleep(10)
        if not locked:
            log("note: proceeding without the lock (filesystems without flock); "
                "atomic renames still keep the shared volume consistent")

    try:
        for entry in entries:
            if args.only_baked and not entry.get("bake", False):
                log(f"skipping {entry['name']} (not baked at build time)")
                continue
            resolve(token, entry, roots, args.dest_root)
    finally:
        if locked:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()
    log("all models available")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ensure-models: ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
