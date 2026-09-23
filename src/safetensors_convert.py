#!/usr/bin/env python3
"""Turn the original Qwen-Image 2.1 weights into ComfyUI-loadable checkpoints.

The upstream repo (`Qwen/Qwen-Image-2.1`) ships diffusers-format files: sharded
safetensors and key names that ComfyUI does not use. What is needed here is
larger than a rename, so the conversion lives in this repository - and it is
byte level: every tensor is *copied*, never recomputed, so the result carries
exactly the upstream numbers.

Supported operations (declared per model in models.json)

  rename    regex rename of keys, e.g. model.language_model.* -> model.*
  fuse      concatenate two tensors along axis 0 into one key. ComfyUI's
            qwen_image21 FFN is a fused SwiGLU (img_mlp.gate_up + img_mlp.out)
            while the original ships gate_layer + proj; the fused tensor is
            exactly the two halves in row order (verified byte-identical
            against Comfy-Org's repackage).
  reshape   insert a unit axis, e.g. 4D conv weights -> 5D WanVAE kernels
            (temporal kernel 1). No data change, header only.
  drop      drop keys (used to strip things ComfyUI has no place for).

There is no offsets-and-weights bookkeeping in the repo: the manifest declares
what the *output* must look like - tensor count, dtype, data bytes and a hash of
the sorted "key|shape" listing - and this module refuses to publish a file that
does not match. A wrong mapping therefore fails the cold start loudly instead of
handing ComfyUI plausible-looking garbage.

Only the standard library is used (see ensure_models.py for why).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import time

CHUNK = 8 * 1024 * 1024
DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "U16": 2, "U32": 4,
    "U64": 8, "BOOL": 1,
}


class ConvertError(RuntimeError):
    """Anything that makes the conversion unsafe to publish."""


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"


def read_header(path: str) -> tuple[dict, int]:
    """Return (header, bytes before the tensor data) for a safetensors file."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ConvertError(f"{path}: too short to be a safetensors file")
            (header_len,) = struct.unpack("<Q", raw)
            if not 0 < header_len < 512 * 1024 * 1024:
                raise ConvertError(f"{path}: implausible safetensors header length {header_len}")
            payload = fh.read(header_len)
            if len(payload) != header_len:
                raise ConvertError(f"{path}: truncated header (want {header_len} bytes)")
    except OSError as exc:
        raise ConvertError(f"{path}: {exc}") from exc
    try:
        header = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConvertError(f"{path}: unreadable safetensors header ({exc})") from exc
    if not isinstance(header, dict):
        raise ConvertError(f"{path}: safetensors header is not an object")
    return header, 8 + header_len


def layout_sha256(shapes: dict[str, list[int]]) -> str:
    """Hash of the sorted "key|shape" listing - the output layout fingerprint."""
    lines = "".join(f"{k}|{list(v)}\n" for k, v in sorted(shapes.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


class _Tensor:
    __slots__ = ("key", "dtype", "shape", "parts", "source")

    def __init__(self, key: str, dtype: str, shape: list[int], parts: list[tuple[int, int, int]],
                 source: str | None = None):
        self.key = key
        self.dtype = dtype
        self.shape = list(shape)
        self.parts = parts  # [(source index, data offset, length), ...] concatenated in order
        self.source = source

    @property
    def data_size(self) -> int:
        return sum(p[2] for p in self.parts)

    @property
    def nbytes_expected(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count * DTYPE_SIZES[self.dtype]


class Converter:
    """One conversion: N source files -> one ComfyUI checkpoint."""

    def __init__(self, sources: list[str], spec: dict, log=print):
        self.sources = sources
        self.spec = spec
        self.log = log
        self.headers: list[dict] = []
        self.data_starts: list[int] = []
        self.tensors: dict[str, _Tensor] = {}

    # -- planning ----------------------------------------------------------
    def load_sources(self) -> None:
        for idx, path in enumerate(self.sources):
            header, data_start = read_header(path)
            self.headers.append(header)
            self.data_starts.append(data_start)
            size = os.path.getsize(path)
            data_region = size - data_start
            highest = 0
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                start, end = info["data_offsets"]
                if start > end or end > data_region:
                    raise ConvertError(f"{path}: {key} data_offsets {start}..{end} outside the file")
                highest = max(highest, end)
                if key in self.tensors:
                    other = self.tensors[key].source
                    raise ConvertError(f"{key} appears in both {other} and {path}")
                self.tensors[key] = _Tensor(key, info["dtype"], info["shape"],
                                            [(idx, start, end - start)], path)
            if highest != data_region:
                raise ConvertError(
                    f"{path}: tensor data ends at {highest} but the file has {data_region} "
                    "bytes of data - the file is truncated or has trailing garbage")
        self.log(f"  read {len(self.sources)} source file(s), {len(self.tensors)} tensors")

    def _rename(self) -> None:
        for pattern, replacement in self.spec.get("rename") or []:
            rx = re.compile(pattern)
            # ${name} is the template syntax the fuse rules use; Python's own is
            # \g<name>. Accept either so the manifest stays uniform.
            repl = re.sub(r"\$\{(\w+)\}", r"\\g<\1>", replacement)
            renamed: dict[str, _Tensor] = {}
            hits = 0
            for key, tensor in self.tensors.items():
                new = rx.sub(repl, key)
                if new != key:
                    hits += 1
                if new in renamed:
                    raise ConvertError(f"rename {pattern!r} collides on {new}")
                tensor.key = new
                renamed[new] = tensor
            if not hits:
                raise ConvertError(f"rename {pattern!r} matched nothing - upstream keys changed?")
            self.tensors = renamed
            self.log(f"  renamed {hits} key(s) with {pattern!r}")

    def _fuse(self) -> None:
        for rule in self.spec.get("fuse") or []:
            pattern = re.compile(rule["pattern"])
            with_template, into_template = rule["with"], rule["into"]
            axis = rule.get("axis", 0)
            fused: dict[str, _Tensor] = {}
            consumed: set[str] = set()
            for key in sorted(self.tensors):
                match = pattern.match(key)
                if not match:
                    continue
                groups = match.groupdict()
                partner = self._expand(with_template, groups)
                target = self._expand(into_template, groups)
                first, second = self.tensors.get(key), self.tensors.get(partner)
                if second is None:
                    raise ConvertError(f"fuse: {key} has no partner {partner}")
                if first.dtype != second.dtype:
                    raise ConvertError(f"fuse: {key} and {partner} have different dtypes")
                if len(first.shape) != len(second.shape):
                    raise ConvertError(f"fuse: {key} and {partner} have different rank")
                if first.shape[1:] != second.shape[1:]:
                    raise ConvertError(f"fuse: {key} {first.shape} and {partner} {second.shape} "
                                       "do not line up along axis 0")
                if axis != 0:
                    raise ConvertError("fuse: only axis 0 is supported")
                shape = [first.shape[0] + second.shape[0]] + list(first.shape[1:])
                fused[target] = _Tensor(target, first.dtype, shape,
                                        list(first.parts) + list(second.parts),
                                        f"{first.source} + {second.source}")
                consumed.update({key, partner})
                self.log(f"  fused {key} + {partner} -> {target} {shape}")
            if not fused:
                raise ConvertError(f"fuse rule {rule['pattern']!r} matched nothing")
            for key in consumed:
                self.tensors.pop(key, None)
            for key, tensor in fused.items():
                if key in self.tensors:
                    raise ConvertError(f"fuse: target {key} already exists")
                self.tensors[key] = tensor

    @staticmethod
    def _expand(template: str, groups: dict[str, str | None]) -> str:
        out = template
        for name, value in groups.items():
            out = out.replace("${" + name + "}", value or "")
        return out

    def _reshape(self) -> None:
        for rule in self.spec.get("reshape") or []:
            pattern = re.compile(rule["pattern"])
            exclude = [re.compile(e) for e in rule.get("exclude") or []]
            hits = 0
            for key, tensor in self.tensors.items():
                if not pattern.search(key) or any(e.search(key) for e in exclude):
                    continue
                if rule.get("ndim") is not None and len(tensor.shape) != rule["ndim"]:
                    continue
                axis = rule["insert_axis"]
                if not 0 <= axis <= len(tensor.shape):
                    raise ConvertError(f"reshape: axis {axis} out of range for {key}")
                tensor.shape = tensor.shape[:axis] + [1] + tensor.shape[axis:]
                hits += 1
            if not hits:
                raise ConvertError(f"reshape rule {rule!r} matched nothing")
            self.log(f"  reshaped {hits} key(s) with {rule['pattern']!r}")

    def _drop(self) -> None:
        for pattern in self.spec.get("drop") or []:
            rx = re.compile(pattern)
            victims = [k for k in self.tensors if rx.search(k)]
            if not victims:
                raise ConvertError(f"drop {pattern!r} matched nothing")
            for key in victims:
                self.tensors.pop(key)
            self.log(f"  dropped {len(victims)} key(s) matching {pattern!r}")

    def plan(self) -> dict[str, _Tensor]:
        self.load_sources()
        self._drop()
        self._rename()
        self._fuse()
        self._reshape()
        self.verify_plan()
        return self.tensors

    def verify_plan(self) -> None:
        sizes = {t.dtype for t in self.tensors.values()}
        if len(sizes) > 1:
            raise ConvertError(f"mixed dtypes in the output: {sorted(sizes)}")
        for key, tensor in self.tensors.items():
            if tensor.data_size != tensor.nbytes_expected:
                raise ConvertError(
                    f"{key}: {tensor.data_size} bytes of tensor data but shape {tensor.shape} "
                    f"and dtype {tensor.dtype} imply {tensor.nbytes_expected}")
        expect = self.spec.get("expect")
        if not expect:
            return
        got = {
            "tensors": len(self.tensors),
            "dtype": sorted(sizes)[0] if sizes else None,
            "data_bytes": sum(t.data_size for t in self.tensors.values()),
            "layout_sha256": layout_sha256({k: t.shape for k, t in self.tensors.items()}),
        }
        mismatch = {k: (got[k], v) for k, v in expect.items() if k in got and got[k] != v}
        if mismatch:
            detail = ", ".join(f"{k}: got {a!r}, expected {b!r}" for k, (a, b) in mismatch.items())
            raise ConvertError(f"converted layout does not match the manifest ({detail})")
        self.log(f"  verified layout: {got['tensors']} tensors, {got['dtype']}, "
                 f"{human(got['data_bytes'])}, sha {got['layout_sha256'][:12]}")

    # -- writing -----------------------------------------------------------
    def _output_header(self) -> tuple[dict, int]:
        data = {"__metadata__": {"format": "pt"}}
        offset = 0
        for key in sorted(self.tensors):
            tensor = self.tensors[key]
            data[key] = {"dtype": tensor.dtype, "shape": tensor.shape,
                         "data_offsets": [offset, offset + tensor.data_size]}
            offset += tensor.data_size
        return data, offset

    def write(self, dest: str, sample_verify: bool = True) -> dict:
        header, data_bytes = self._output_header()
        payload = json.dumps(header, separators=(",", ":")).encode("utf-8")
        pad = (-len(payload)) % 8
        payload += b" " * pad
        tmp = dest + ".part"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        started = time.monotonic()
        written = 0
        digest = hashlib.sha256()
        fhs = [open(path, "rb") for path in self.sources]
        try:
            with open(tmp, "wb") as out:
                out.write(struct.pack("<Q", len(payload)))
                out.write(payload)
                for key in sorted(self.tensors):
                    for src_idx, offset, length in self.tensors[key].parts:
                        fh = fhs[src_idx]
                        fh.seek(self.data_starts[src_idx] + offset)
                        remaining = length
                        while remaining:
                            chunk = fh.read(min(CHUNK, remaining))
                            if not chunk:
                                raise ConvertError(f"{self.sources[src_idx]}: unexpected EOF in {key}")
                            out.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                            remaining -= len(chunk)
                out.flush()
                os.fsync(out.fileno())
        finally:
            for fh in fhs:
                fh.close()
        if written != data_bytes:
            raise ConvertError(f"wrote {written} bytes of tensor data, expected {data_bytes}")
        if sample_verify:
            self._verify_sample(tmp, payload)
        os.replace(tmp, dest)
        elapsed = max(time.monotonic() - started, 1e-6)
        stats = {"path": dest, "bytes": 8 + len(payload) + data_bytes,
                 "tensors": len(self.tensors), "seconds": elapsed,
                 "data_sha256": digest.hexdigest(),
                 "throughput": (written + len(payload)) / elapsed}
        self.log(f"  wrote {dest} ({human(stats['bytes'])}, {stats['tensors']} tensors, "
                 f"{elapsed:.0f}s, {human(stats['throughput'])}/s)")
        return stats

    def _verify_sample(self, path: str, payload: bytes) -> None:
        """Re-read a few tensors from the file we just wrote and compare bytes.

        Cheap insurance against an off-by-header or off-by-offset bug: a wrong
        data start would produce a file with the right shape, count and size and
        completely wrong weights, and nothing else in the pipeline would notice.
        """
        keys = sorted(self.tensors)
        step = max(1, len(keys) // 8)
        picks = sorted({0, len(keys) // 2, len(keys) - 1, *range(0, len(keys), step)})
        base = 8 + len(payload)
        checked = 0
        with open(path, "rb") as out:
            for idx in picks:
                tensor = self.tensors[keys[idx]]
                if tensor.data_size > 64 * 1024 * 1024:  # keep the check cheap
                    continue
                offset = base + sum(self.tensors[k].data_size for k in keys[:idx])
                out.seek(offset)
                got = out.read(tensor.data_size)
                want = b"".join(self._read_source(p) for p in tensor.parts)
                if got != want:
                    raise ConvertError(f"{tensor.key}: written bytes differ from the source")
                checked += 1
        self.log(f"  re-read {checked} tensor(s) back: identical to the source")

    def _read_source(self, part: tuple[int, int, int]) -> bytes:
        src_idx, offset, length = part
        with open(self.sources[src_idx], "rb") as fh:
            fh.seek(self.data_starts[src_idx] + offset)
            return fh.read(length)


def output_matches(path: str, expect: dict) -> bool:
    """Cheap re-check of an already converted file (header only, no source files)."""
    if not expect or not os.path.exists(path):
        return False
    try:
        header, data_start = read_header(path)
    except ConvertError:
        return False
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if expect.get("tensors") is not None and len(tensors) != expect["tensors"]:
        return False
    sizes = {v["dtype"] for v in tensors.values()}
    if expect.get("dtype") and sizes != {expect["dtype"]}:
        return False
    data_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in tensors.values())
    if expect.get("data_bytes") is not None and data_bytes != expect["data_bytes"]:
        return False
    if os.path.getsize(path) != data_start + data_bytes:
        return False
    if expect.get("layout_sha256"):
        return layout_sha256({k: v["shape"] for k, v in tensors.items()}) == expect["layout_sha256"]
    return True


def convert(sources: list[str], dest: str, spec: dict, log=print,
            sample_verify: bool = True) -> dict:
    converter = Converter(sources, spec, log=log)
    converter.plan()
    return converter.write(dest, sample_verify=sample_verify)
