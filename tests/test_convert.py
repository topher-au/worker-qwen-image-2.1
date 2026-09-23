#!/usr/bin/env python3
"""Checks for the demo-to-ComfyUI conversion path.

Synthetic shards stand in for the 33 GB of Qwen-Image 2.1 weights: the manifest
specs in models.json are exercised rule by rule (rename, row-order fusion, unit
axes), the output is verified the same way the worker verifies it (tensor count,
dtype, data bytes, key/shape hash), and the bytes are compared back to the
sources. The end-to-end part then runs ensure_models.py against a fake RunPod
model cache, which is exactly the path a cold start takes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from safetensors_convert import (  # noqa: E402
    ConvertError,
    Converter,
    convert,
    layout_sha256,
    output_matches,
    read_header,
)

failures = 0
DTYPE_SIZES = {"BF16": 2, "F32": 4, "I8": 1}


def check(cond: bool, msg: str) -> None:
    global failures
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures += 1


def raises(fn, needle: str, msg: str) -> None:
    try:
        fn()
    except ConvertError as exc:
        check(needle in str(exc), f"{msg} ({exc})")
        return
    except Exception as exc:  # noqa: BLE001
        check(False, f"{msg} - wrong exception {type(exc).__name__}: {exc}")
        return
    check(False, f"{msg} - no error raised")


def write_safetensors(path: str, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, data, offset = {}, bytearray(), 0
    for key, (dtype, shape, payload) in tensors.items():
        assert len(payload) == DTYPE_SIZES[dtype] * _count(shape), key
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(payload)]}
        data += payload
        offset += len(payload)
    payload = json.dumps(header, separators=(",", ":")).encode()
    payload += b" " * ((-len(payload)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(payload)))
        fh.write(payload)
        fh.write(data)


def _count(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n


def read_tensors(path: str) -> dict[str, bytes]:
    header, data_start = read_header(path)
    with open(path, "rb") as fh:
        out = {}
        for key, info in header.items():
            if key == "__metadata__":
                continue
            start, end = info["data_offsets"]
            fh.seek(data_start + start)
            out[key] = fh.read(end - start)
        return out


def blob(tag: str, dtype: str, shape: list[int]) -> bytes:
    """Deterministic pseudo-weights, distinct per key."""
    salt = hashlib.sha256(tag.encode()).digest()
    return (salt * (_count(shape) * DTYPE_SIZES[dtype] // len(salt) + 1))[: _count(shape) * DTYPE_SIZES[dtype]]


# ---------------------------------------------------------------------------
tmp = tempfile.mkdtemp(prefix="convert-test-")
try:
    # --- 1. two shards + row-order fusion (the transformer's shape of problem)
    shards = {
        "t-00001.safetensors": {
            "transformer_blocks.0.img_mlp.gate_layer.weight": ("BF16", [4, 3], blob("g0", "BF16", [4, 3])),
            "transformer_blocks.0.img_mlp.proj.weight": ("BF16", [4, 3], blob("p0", "BF16", [4, 3])),
            "img_in.weight": ("BF16", [2, 2], blob("img", "BF16", [2, 2])),
        },
        "t-00002.safetensors": {
            "transformer_blocks.1.img_mlp.gate_layer.weight": ("BF16", [4, 3], blob("g1", "BF16", [4, 3])),
            "transformer_blocks.1.img_mlp.proj.weight": ("BF16", [4, 3], blob("p1", "BF16", [4, 3])),
            "transformer_blocks.1.img_mlp.out.weight": ("BF16", [3, 4], blob("o1", "BF16", [3, 4])),
        },
    }
    paths = []
    for name, tensors in shards.items():
        p = os.path.join(tmp, name)
        write_safetensors(p, tensors)
        paths.append(p)

    out = os.path.join(tmp, "fused.safetensors")
    spec = {
        "fuse": [{"pattern": r"^(?P<base>.+)\.img_mlp\.gate_layer\.weight$",
                  "with": "${base}.img_mlp.proj.weight",
                  "into": "${base}.img_mlp.gate_up.weight", "axis": 0}],
        "expect": {"tensors": 4, "dtype": "BF16", "data_bytes": 128},
    }
    stats = convert(paths, out, spec, log=lambda *_: None)
    got = read_tensors(out)
    check(set(got) == {"transformer_blocks.0.img_mlp.gate_up.weight",
                       "transformer_blocks.1.img_mlp.gate_up.weight",
                       "transformer_blocks.1.img_mlp.out.weight", "img_in.weight"},
          "fusion renamed the pair and kept every other key")
    check(got["transformer_blocks.0.img_mlp.gate_up.weight"] ==
          blob("g0", "BF16", [4, 3]) + blob("p0", "BF16", [4, 3]),
          "gate_up is gate_layer followed by proj, byte for byte")
    check(read_header(out)[0]["transformer_blocks.0.img_mlp.gate_up.weight"]["shape"] == [8, 3],
          "fused shape is the row sum [8, 3]")
    check(stats["bytes"] == os.path.getsize(out), "reported size matches the file")
    expect = {**spec["expect"],
              "layout_sha256": layout_sha256({
                  "transformer_blocks.0.img_mlp.gate_up.weight": [8, 3],
                  "transformer_blocks.1.img_mlp.gate_up.weight": [8, 3],
                  "transformer_blocks.1.img_mlp.out.weight": [3, 4],
                  "img_in.weight": [2, 2]})}
    check(output_matches(out, expect), "output_matches accepts the file it produced")

    # --- 2. rename + unit-axis reshape (text encoder and VAE's shape of problem)
    src = os.path.join(tmp, "vae.safetensors")
    write_safetensors(src, {
        "encoder.conv_in.weight": ("F32", [2, 3, 3, 3], blob("ci", "F32", [2, 3, 3, 3])),
        "encoder.mid_block.attentions.0.to_qkv.weight": ("F32", [6, 2, 1, 1], blob("qkv", "F32", [6, 2, 1, 1])),
        "decoder.up_blocks.0.resnets.0.norm1.gamma": ("F32", [2, 1, 1, 1], blob("n1", "F32", [2, 1, 1, 1])),
        "decoder.up_blocks.0.resnets.0.conv1.weight": ("F32", [2, 2, 3, 3], blob("c1", "F32", [2, 2, 3, 3])),
    })
    out2 = os.path.join(tmp, "vae-converted.safetensors")
    spec2 = {
        "rename": [
            [r"^encoder\.conv_in\.", "encoder.conv1."],
            [r"^encoder\.mid_block\.attentions\.0\.", "encoder.middle.1."],
            [r"^decoder\.up_blocks\.(?P<n>\d+)\.resnets\.(?P<m>\d+)\.", "decoder.upsamples.${n}.upsamples.${m}."],
            [r"^(?P<blk>decoder\.upsamples\.\d+\.upsamples\.\d+|encoder\.middle\.\d+)\.norm1\.", "${blk}.residual.0."],
            [r"^(?P<blk>decoder\.upsamples\.\d+\.upsamples\.\d+|encoder\.middle\.\d+)\.conv1\.", "${blk}.residual.2."],
        ],
        "reshape": [{"pattern": r"\.weight$", "ndim": 4,
                     "exclude": [r"middle\.\d+\.(?:to_qkv|proj)\.weight$", r"\.resample\.1\.weight$"],
                     "insert_axis": 2}],
        "expect": {"tensors": 4, "dtype": "F32", "data_bytes": 416},
    }
    convert([src], out2, spec2, log=lambda *_: None)
    h2, _ = read_header(out2)
    check("encoder.conv1.weight" in h2,
          "a sub-key rule anchored to resnet blocks leaves the renamed conv_in alone")
    check(h2["encoder.conv1.weight"]["shape"] == [2, 3, 1, 3, 3], "4D conv kernel became a 5D WanVAE kernel")
    check(h2["encoder.middle.1.to_qkv.weight"]["shape"] == [6, 2, 1, 1],
          "attention weights are left at 4D")
    check(h2["decoder.upsamples.0.upsamples.0.residual.0.gamma"]["shape"] == [2, 1, 1, 1],
          "gamma norms are untouched")
    check(read_tensors(out2)["encoder.conv1.weight"] == blob("ci", "F32", [2, 3, 3, 3]),
          "reshaping did not reorder any bytes")

    # --- 3. the failure modes that protect a cold start
    raises(lambda: convert([src], os.path.join(tmp, "x1"), {"rename": [[r"^nope\.", "yes."]]},
                           log=lambda *_: None),
           "matched nothing", "a rename rule that matches nothing is rejected")
    raises(lambda: convert([paths[0][:0] or paths[0]], os.path.join(tmp, "x2"),
                           {"fuse": [{"pattern": r"^(?P<base>.+)\.img_mlp\.gate_layer\.weight$",
                                      "with": "${base}.img_mlp.missing.weight",
                                      "into": "${base}.img_mlp.gate_up.weight"}]},
                           log=lambda *_: None),
           "no partner", "a fusion without its partner is rejected")
    raises(lambda: convert(paths, os.path.join(tmp, "x3"),
                           {"fuse": spec["fuse"], "expect": {"tensors": 99}},
                           log=lambda *_: None),
           "does not match the manifest", "an unexpected tensor count is rejected")
    with open(out, "r+b") as fh:  # truncate the converted file
        fh.truncate(os.path.getsize(out) - 3)
    check(not output_matches(out, expect), "a truncated converter output no longer matches")

    # --- 4. end to end through ensure_models.py, driven by a fake RunPod cache
    cache = os.path.join(tmp, "huggingface-cache", "hub", "models--Some--Repo", "snapshots", "abc123")
    os.makedirs(os.path.join(cache, "part"))
    shutil.copy(paths[0], os.path.join(cache, "part", "shard-00001.safetensors"))
    shutil.copy(paths[1], os.path.join(cache, "part", "shard-00002.safetensors"))
    with open(os.path.join(cache, "part", "index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": 128},
                   "weight_map": {k: "part/shard-00001.safetensors" for k in shards["t-00001.safetensors"]}
                   | {k: "part/shard-00002.safetensors" for k in shards["t-00002.safetensors"]}}, fh)

    manifest = {
        "models": [{
            "name": "fused.safetensors", "dest": "diffusion_models", "repo": "Some/Repo",
            "sources": ["part/shard-00001.safetensors", "part/shard-00002.safetensors"],
            "index": {"path": "part/index.json", "total_size": 128},
            "output_bytes": 128,
            "convert": {**spec, "expect": expect},
        }]
    }
    manifest_path = os.path.join(tmp, "manifest.json")
    models_root = os.path.join(tmp, "models")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh)

    env = {**os.environ, "HF_CACHE_ROOT": os.path.join(tmp, "huggingface-cache", "hub"),
           "HF_ENDPOINT": "http://127.0.0.1:1"}  # any download attempt must fail loudly

    def run(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, os.path.join(ROOT, "src", "ensure_models.py"),
                               "--manifest", manifest_path, "--dest-root", models_root, *extra],
                              capture_output=True, text=True, env=env)

    p = run()
    produced = os.path.join(models_root, "diffusion_models", "fused.safetensors")
    check(p.returncode == 0 and os.path.exists(produced),
          f"ensure_models built the model from the cached shards ({p.stderr.strip()[:80]})")
    check("source cache" in p.stdout, "the sources were read from the model cache, not downloaded")
    check(len(read_tensors(produced)) == 4, "the built file has the expected tensors")

    p = run()
    check(p.returncode == 0 and "present:" in p.stdout, "a second run is a no-op")
    p = run("--check")
    check(p.returncode == 0 and "ok " in p.stdout, "--check reports the model as ok")

    with open(produced, "r+b") as fh:
        fh.truncate(os.path.getsize(produced) - 3)
    p = run("--check")
    check(p.returncode == 1 and "DAMAGED" in p.stdout, "--check catches a damaged file")
    p = run()
    check(p.returncode == 0 and "rebuilding" in p.stdout and output_matches(produced, expect),
          "the next start rebuilds a damaged file")

    # a source that is nowhere and no network: the run must fail, not invent a model
    manifest["models"][0]["repo"] = "Missing/Repo"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh)
    os.remove(produced)
    p = run()
    check(p.returncode == 1 and "ERROR" in p.stderr, "a run without sources fails loudly")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"test_convert: {'all checks passed' if not failures else f'{failures} failed'}")
sys.exit(1 if failures else 0)
