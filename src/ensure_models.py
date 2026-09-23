#!/usr/bin/env python3
"""Make sure every model the worker needs is on disk before ComfyUI starts.

Used twice, from the same manifest (models.json):

  * at image build time  -> ``--only-baked`` fetches the small, immutable files
  * on the worker's first start -> everything else is prepared, ideally onto a
    RunPod network volume so only the first worker of an endpoint pays for it

Two kinds of entry

  * plain      one upstream file, used as-is (linked from the RunPod model cache
               or downloaded). Still supported; no entry in models.json uses it.
  * converted  N upstream shards turned into one ComfyUI checkpoint by
               src/safetensors_convert.py (renames, row-order fusion, unit axes).
               The original Qwen repository is diffusers-format, so all three of
               this worker's models take this path.

Design notes
------------
* RunPod's *cached models* feature stages whole Hugging Face repos on the host's
  local disk before the worker starts, in the standard HF cache layout under
  /runpod-volume/huggingface-cache/hub. Sources are read straight from there when
  present - no download, and no second copy of 30+ GB.
* ``.part`` staging + ``os.replace()``: the rename is atomic on one filesystem,
  so a worker that dies mid-download (or mid-conversion) never leaves a
  truncated file that ComfyUI would happily mmap and then crash on.
* A ``flock``ed lock file serialises workers that share a network volume. Some
  NFS mounts do not implement flock; that is logged and ignored, because the
  atomic rename is what actually keeps a shared volume consistent.
* Expected sizes come from Hugging Face (HEAD request), not from the manifest, so
  an upstream re-upload warns instead of breaking a cold start. For converted
  models the manifest instead pins the *output* layout (tensor count, dtype, data
  bytes, key/shape hash) and the converter refuses to publish anything else.
* Only the Python standard library is used - this runs before anything else is
  set up and must never break the worker because of a missing dependency.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from safetensors_convert import (  # noqa: E402
    ConvertError,
    convert,
    human,
    output_matches,
    read_header,
)

CHUNK = 8 * 1024 * 1024
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
LOCK_TIMEOUT_S = float(os.environ.get("MODEL_LOCK_TIMEOUT_S", "1800"))
RETRIES = int(os.environ.get("MODEL_DOWNLOAD_RETRIES", "5"))
LOCK_WAIT_S = float(os.environ.get("MODEL_LOCK_WAIT_S", "3600"))
CACHE_MODE = os.environ.get("MODEL_CACHE_MODE", "link")  # link | copy | off
# RunPod's "cached models" feature stages whole Hugging Face repos on the host's
# local disk before the worker starts, so the shards can be read straight from
# there. Honour HF's own variables when the host sets them, and fall back to the
# layouts that appear in practice; the first root that has the repo wins, and
# every hit is logged with its full path so a mismatch is easy to spot.
def _cache_roots() -> list[str]:
    roots: list[str] = []
    for key in ("MODEL_HF_CACHE_ROOT", "HF_CACHE_ROOT", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(key):
            roots.append(os.environ[key])
    if os.environ.get("HF_HOME"):
        roots.append(os.path.join(os.environ["HF_HOME"], "hub"))
    roots += [
        "/runpod-volume/huggingface-cache/hub",
        "/runpod-volume/huggingface-cache",
        "/root/.cache/huggingface/hub",
    ]
    seen, unique = set(), []
    for root in roots:
        root = root.rstrip("/")
        if root and root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


HF_CACHE_ROOTS = _cache_roots()
# Where source shards go when they are not in the RunPod cache. Sits next to the
# models root (i.e. on the same volume) so a conversion never has to be repeated.
SOURCES_DIRNAME = os.environ.get("MODEL_SOURCES_DIR", "model-sources")
# Keep the downloaded shards after a successful conversion? Deleting them halves
# the space a first start needs (33 GB instead of 66 GB) at the cost of having to
# fetch them again if the output is ever rebuilt. Shards taken from the RunPod
# cache are never touched.
KEEP_SOURCES = os.environ.get("MODEL_SOURCES_KEEP", "false").lower() in ("1", "true", "yes")


def log(msg: str) -> None:
    print(f"ensure-models: {msg}", flush=True)


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


def cache_candidates(repo: str, path: str) -> list[str]:
    """Paths for `path` inside a repo staged by RunPod's cached-models feature.

    RunPod unpacks a cached Hugging Face repo into the standard HF cache layout,
    with slashes in the repo name replaced by "--" and one directory per revision:

        <cache root>/models--<org>--<name>/snapshots/<hash>/<path inside the repo>

    Returns the matching paths (newest snapshot last), or [] when that repo is
    not cached on this host.
    """
    if CACHE_MODE == "off":
        return []
    hits = []
    for root in HF_CACHE_ROOTS:
        snapshots = os.path.join(root, "models--" + repo.replace("/", "--"), "snapshots")
        if not os.path.isdir(snapshots):
            continue
        for snap in sorted(os.listdir(snapshots)):
            candidate = os.path.join(snapshots, snap, path)
            if os.path.isfile(candidate):
                hits.append(candidate)
    return hits


def is_cached_path(path: str) -> bool:
    return any(path.startswith(root + os.sep) for root in HF_CACHE_ROOTS)


def sources_path(dest_root: str, repo: str, path: str) -> str:
    return os.path.normpath(os.path.join(dest_root, "..", SOURCES_DIRNAME,
                                         repo.replace("/", "--"), path))


def ensure_space(root: str, need: int, what: str = "the models") -> None:
    """Fail early, with an actionable message, instead of dying on ENOSPC."""
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        return
    if free < need * 1.05:
        raise RuntimeError(
            f"not enough space in {root} for {what}: {human(free)} free, {human(need)} needed. "
            "Attach a network volume (models land on /runpod-volume/models), or raise the "
            "container disk to at least 40 GB."
        )


def adopt(src: str, dest: str) -> None:
    """Make a cached file usable from ComfyUI's model directories."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.islink(dest) or os.path.exists(dest):
        os.remove(dest)
    size = os.path.getsize(src)
    if CACHE_MODE == "copy":
        ensure_space(os.path.dirname(dest), size, what=os.path.basename(dest))
        shutil.copy2(src, dest)
        log(f"copied cached model: {src} -> {dest} ({human(size)})")
        return
    try:
        os.symlink(src, dest)
        log(f"linked cached model: {dest} -> {src} ({human(size)})")
    except OSError as exc:  # filesystem without symlinks: fall back to a copy
        log(f"note: symlink failed ({exc}), copying instead")
        ensure_space(os.path.dirname(dest), size, what=os.path.basename(dest))
        shutil.copy2(src, dest)
        log(f"copied cached model: {src} -> {dest} ({human(size)})")


def present_path(entry: dict, roots: list[str]) -> str | None:
    """The path that already satisfies this entry, if any."""
    name, subdir = entry["name"], entry["dest"]
    expect = (entry.get("convert") or {}).get("expect")
    for root in roots:
        path = os.path.join(root, subdir, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        if expect and not output_matches(path, expect):
            log(f"rebuilding {path}: it does not match the layout declared in models.json")
            continue
        return path
    return None


def local_source(entry: dict, path: str, dest_root: str) -> str | None:
    """Where this upstream file already is on this host, or None."""
    cached = cache_candidates(entry["repo"], path)
    if cached:
        return cached[-1]
    staged = sources_path(dest_root, entry["repo"], path)
    return staged if os.path.isfile(staged) else None


def fetch_source(entry: dict, path: str, token: str | None, dest_root: str) -> tuple[str, bool]:
    """Make one upstream file available locally -> (path, came from the RunPod cache)."""
    local = local_source(entry, path, dest_root)
    if local:
        return local, is_cached_path(local)
    dest = sources_path(dest_root, entry["repo"], path)
    url = f"{HF_ENDPOINT}/{entry['repo']}/resolve/main/{path}"
    size = remote_size(url, token)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not path.endswith(".json"):
        url = f"{url}?download=true"
    download(url, dest, token, size)
    return dest, False


def source_report(entry: dict, dest_root: str) -> tuple[int, int]:
    """How many source files are available locally, out of how many are needed."""
    wanted = list(entry["sources"]) + ([entry["index"]["path"]] if entry.get("index") else [])
    have = sum(1 for p in wanted if local_source(entry, p, dest_root))
    return have, len(wanted)


def verify_index(entry: dict, token: str | None, dest_root: str, sources: list[str]) -> None:
    """Cross-check the shards we have against the repository's own index.json."""
    local, _ = fetch_source(entry, entry["index"]["path"], token, dest_root)
    with open(local) as fh:
        index = json.load(fh)
    declared = entry["index"].get("total_size")
    actual = (index.get("metadata") or {}).get("total_size")
    if declared and actual and actual != declared:
        raise ConvertError(f"{entry['name']}: index.json says total_size {actual}, "
                           f"models.json says {declared}")
    weights = set(index["weight_map"])
    present: set[str] = set()
    for path in sources:
        header, _ = read_header(path)
        present.update(k for k in header if k != "__metadata__")
    missing, extra = weights - present, present - weights
    if missing or extra:
        raise ConvertError(
            f"{entry['name']}: the shards do not match index.json "
            f"({len(missing)} key(s) missing, {len(extra)} unexpected)")
    log(f"  index.json: {len(weights)} keys, all accounted for in the shard(s)")


def resolve(token: str | None, entry: dict, roots: list[str], dest_root: str) -> None:
    name, subdir = entry["name"], entry["dest"]
    found = present_path(entry, roots)
    if found:
        log(f"present: {found} ({human(os.path.getsize(found))})")
        if found != os.path.join(dest_root, subdir, name):
            log(f"  (using the copy that is already on disk instead of {dest_root})")
        return

    dest = os.path.join(dest_root, subdir, name)

    if entry.get("convert"):
        log(f"building {name} from {entry['repo']} "
            f"({len(entry['sources'])} source file(s) -> {human(entry.get('output_bytes') or 0)})")
        # Space: the output plus whatever still has to be downloaded. Sources read
        # from the RunPod cache are on the host's own disk and cost nothing here.
        need = entry.get("output_bytes") or 0
        for path in entry["sources"]:
            if local_source(entry, path, dest_root):
                continue
            url = f"{HF_ENDPOINT}/{entry['repo']}/resolve/main/{path}"
            need += remote_size(url, token) or 0
        ensure_space(dest_root, need, what=name)
        sources, from_cache = [], 0
        for path in entry["sources"]:
            local, cached = fetch_source(entry, path, token, dest_root)
            sources.append(local)
            from_cache += int(cached)
            log(f"  source {'cache' if cached else 'disk '} {local}")
        if entry.get("index"):
            verify_index(entry, token, dest_root, sources)
        convert(sources, dest, entry["convert"], log=log)
        staged = [s for s in sources if not is_cached_path(s)]
        if staged and KEEP_SOURCES:
            log(f"  kept {len(staged)} downloaded source shard(s) ({SOURCES_DIRNAME}, MODEL_SOURCES_KEEP=true)")
        elif staged:
            for path in staged:
                try:
                    os.remove(path)
                except OSError as exc:
                    log(f"  note: could not remove {path}: {exc}")
            log(f"  removed {len(staged)} downloaded source shard(s); set MODEL_SOURCES_KEEP=true "
                "to keep them next to the models root instead")
        return

    cached = cache_candidates(entry["repo"], entry["path"])
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
        ensure_space(os.path.dirname(dest), need, what=name)
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
                    help="report status without downloading or converting anything")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)
    entries = manifest["models"]

    if args.check:
        bad = 0
        for e in entries:
            path = os.path.join(args.dest_root, e["dest"], e["name"])
            expect = (e.get("convert") or {}).get("expect")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                if not expect or output_matches(path, expect):
                    log(f"ok      {path}")
                    continue
                bad += 1
                log(f"DAMAGED {path} (does not match models.json; the next start rebuilds it)")
                continue
            bad += 1
            if e.get("convert"):
                have, want = source_report(e, args.dest_root)
                log(f"MISSING {path} ({have}/{want} source file(s) available for the first-start build)")
            else:
                cached = cache_candidates(e["repo"], e["path"])
                log(f"MISSING {path}" + (f" (usable from {cached[-1]})" if cached else ""))
        return 1 if bad else 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    roots = [args.dest_root] + [r for r in args.search_root if r != args.dest_root]

    lock_path = os.path.join(args.dest_root, ".ensure-models.lock")
    os.makedirs(args.dest_root, exist_ok=True)
    lock_fh = open(lock_path, "w")
    locked = False
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        log("acquired the model lock")
    except OSError as exc:
        log(f"note: could not take the model lock ({exc}); waiting up to "
            f"{LOCK_WAIT_S:.0f}s, another worker may be preparing the same files")
        deadline = time.monotonic() + LOCK_WAIT_S
        while time.monotonic() < deadline:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                time.sleep(10)
        if locked:
            log("acquired the model lock after waiting")
        else:
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
