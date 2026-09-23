#!/usr/bin/env python3
"""Check models.json against the upstream repositories - without the weights.

Every conversion rule is applied to the *real* key lists of the Qwen sources:
the safetensors headers and index.json are fetched with HTTP range requests
(a few hundred KB), written into sparse placeholder files, and the same
converter the worker runs is asked to plan the conversion. The manifest's
'expect' block then has to match, which is what proves the key renames, the
row-order fusion and the reshape rules still line up with upstream.

Usage:
    python3 scripts/check_specs.py [--manifest models.json]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from safetensors_convert import (  # noqa: E402
    ConvertError,
    Converter,
    layout_sha256,
    read_header,
)

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def fetch(url: str, rng: tuple[int, int] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "runpod-worker-qwen-image-2.1"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def remote_header(repo: str, path: str) -> tuple[bytes, dict, int]:
    """(raw header bytes, parsed header, bytes before the tensor data)."""
    url = f"{HF_ENDPOINT}/{repo}/resolve/main/{path}"
    header_len = struct.unpack("<Q", fetch(url, (0, 7)))[0]
    payload = fetch(url, (8, 8 + header_len - 1))
    return payload, json.loads(payload.decode("utf-8")), 8 + header_len


def placeholder(directory: str, repo: str, path: str) -> tuple[str, dict]:
    """A sparse file carrying the real header, sized like the real shard."""
    payload, header, data_start = remote_header(repo, path)
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    highest = max((v["data_offsets"][1] for v in tensors.values()), default=0)
    local = os.path.join(directory, os.path.basename(path))
    with open(local, "wb") as fh:
        fh.write(struct.pack("<Q", len(payload)))
        fh.write(payload)
        fh.truncate(data_start + highest)
    return local, tensors


def check_entry(entry: dict, workdir: str) -> None:
    name = entry["name"]
    print(f"\n=== {name}  ({entry['repo']}) ===")
    sources, keys = [], set()
    for path in entry["sources"]:
        local, tensors = placeholder(workdir, entry["repo"], path)
        sources.append(local)
        keys.update(tensors)
        print(f"  {path}: {len(tensors)} tensors, header {os.path.getsize(local):,} bytes (sparse)")

    if entry.get("index"):
        raw = fetch(f"{HF_ENDPOINT}/{entry['repo']}/resolve/main/{entry['index']['path']}")
        index = json.loads(raw.decode())
        weights = set(index["weight_map"])
        declared = entry["index"].get("total_size")
        claimed = (index.get("metadata") or {}).get("total_size")
        ok = weights == keys
        print(f"  index.json: {len(weights)} keys, total_size={claimed} "
              f"({'matches' if ok else 'MISMATCH'}) {'' if declared == claimed else 'size differs!'}")
        if not ok:
            raise ConvertError(f"{name}: index.json and the shards disagree "
                               f"({len(weights - keys)} missing, {len(keys - weights)} extra)")

    converter = Converter(sources, entry["convert"], log=lambda m: print("  " + m.strip()))
    converter.plan()
    shapes = {k: t.shape for k, t in converter.tensors.items()}
    expect = entry["convert"]["expect"]
    print(f"  layout: {len(shapes)} tensors, sha256 {layout_sha256(shapes)[:16]}…, "
          f"expect {expect['tensors']} / {expect['layout_sha256'][:16]}…  -> OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=os.path.join(ROOT, "models.json"))
    ap.add_argument("--only", help="single model name to check")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        models = json.load(fh)["models"]
    if args.only:
        models = [m for m in models if m["name"] == args.only]

    failures = 0
    with tempfile.TemporaryDirectory(prefix="check-specs-") as workdir:
        for entry in models:
            try:
                check_entry(entry, workdir)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAILED: {exc}", file=sys.stderr)
    print("\ncheck_specs: everything matches upstream" if not failures
          else f"\ncheck_specs: {failures} model(s) do not match models.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
