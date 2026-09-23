#!/usr/bin/env python3
"""Checks on the vendored entrypoint (src/start.sh, installed as /start.sh).

The interesting property is behavioural, not textual: whatever ComfyUI is
launched with has to include --use-ck-attention by default, keep every upstream
argument, and stay clean of empty argv entries. So the real launch line from the
file is executed here with `python` stubbed out to print the argv it would have
received.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START_SH = os.path.join(ROOT, "src", "start.sh")

failures = 0


def check(cond: bool, msg: str) -> None:
    global failures
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        failures += 1


def main() -> int:
    with open(START_SH) as fh:
        text = fh.read()

    check(os.access(START_SH, os.X_OK), "src/start.sh is executable")
    check(subprocess.run(["bash", "-n", START_SH]).returncode == 0, "src/start.sh parses cleanly")

    block = re.search(r"^# Enable Comfy Kitchen INT8 attention.*?^esac$", text, re.S | re.M)
    check(block is not None, "src/start.sh carries the CK attention block")
    launches = re.findall(r"^.*python -u /comfyui/main\.py .*$", text, re.M)
    check(len(launches) == 2, f"src/start.sh has both ComfyUI launch lines (found {len(launches)})")
    if block is None or len(launches) != 2:
        return 1

    def argv(**env: str) -> list[str]:
        """Run the real launch line with `python` stubbed, return the argv."""
        script = "\n".join([
            "set -euo pipefail",
            'python() { printf "%s\\n" "$@"; }',
            'export COMFY_LOG_LEVEL="${COMFY_LOG_LEVEL:-DEBUG}"',
            block.group(0),
            launches[0],
            "wait",
        ])
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True,
                             env={**os.environ, "PATH": os.environ["PATH"], **env})
        if out.returncode != 0:
            raise AssertionError(out.stderr or f"launch line exited {out.returncode}")
        return out.stdout.splitlines()

    default = argv()
    check("--use-ck-attention" in default, "CK attention is on when CK_ATTENTION is unset")
    check(default[:3] == ["-u", "/comfyui/main.py", "--use-ck-attention"],
          f"the flag follows main.py: {default[:3]}")
    for keep in ("--disable-auto-launch", "--disable-metadata", "--log-stdout"):
        check(keep in default, f"upstream argument preserved: {keep}")
    check("" not in default, "no empty argv entries reach main.py")

    off = argv(CK_ATTENTION="false")
    check("--use-ck-attention" not in off, "CK_ATTENTION=false drops the flag")
    check("--disable-auto-launch" in off, "CK_ATTENTION=false still launches ComfyUI normally")

    extra = argv(CK_ATTENTION="true", COMFY_EXTRA_ARGS="--fast --cache-none")
    check("--fast" in extra and "--cache-none" in extra, "COMFY_EXTRA_ARGS reaches main.py")
    check("--use-ck-attention" in extra, "COMFY_EXTRA_ARGS does not disable CK attention")
    check("" not in extra, "unset extras do not create empty argv entries")

    bogus = argv(CK_ATTENTION="True")
    check("--use-ck-attention" not in bogus, "an unrecognised CK_ATTENTION value does not enable the flag")
    check("--disable-auto-launch" in bogus, "...and still launches ComfyUI normally")

    print(f"test_start_sh: {'all checks passed' if not failures else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
