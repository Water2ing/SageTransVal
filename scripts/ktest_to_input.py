"""Convert KLEE .ktest output files into the SageTransVal harness input format.

KLEE writes one .ktest file per discovered path. Each contains the symbolic
variable values that drive that path. This script:

  1. Walks a KLEE output directory (``klee-last/``)
  2. Parses each .ktest using the bundled ``ktest-tool`` (or our own minimal
     parser if the tool is absent)
  3. Reconstructs typed Python values per parameter using the subject's
     declared parameter_types
  4. Emits a JSONL file of inputs the harness can consume directly

Usage:
    python ktest_to_input.py --klee-out runs/klee/<subject>/klee-last \
        --subject data/crust_bench/<subject>/metadata.json \
        --out runs/klee/<subject>/inputs.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys


# Minimal .ktest parser. Each ktest is little-endian:
#   header: "KTEST" + uint32 version
#   uint32 num_args, [arg]*
#   uint32 sym_argvs, [argv]*
#   uint32 num_objects, [object]*
# object: uint32 name_len, name_bytes, uint32 data_len, data_bytes

def parse_ktest(path: pathlib.Path) -> list[tuple[str, bytes]]:
    data = path.read_bytes()
    pos = 0
    def u32() -> int:
        nonlocal pos
        v = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        return v
    def take(n: int) -> bytes:
        nonlocal pos
        v = data[pos:pos+n]
        pos += n
        return v
    header = take(5)
    if header != b"KTEST":
        raise ValueError(f"not a .ktest file: {path}")
    _version = u32()
    # Skip args
    n_args = u32()
    for _ in range(n_args):
        n = u32()
        take(n)
    _sym_argvs = u32()
    _sym_argv_len = u32()
    n_objs = u32()
    out = []
    for _ in range(n_objs):
        nlen = u32()
        name = take(nlen).decode("utf-8", "replace")
        dlen = u32()
        dat = take(dlen)
        out.append((name, dat))
    return out


def decode(name: str, data: bytes, parameter_types: dict[str, str]):
    t = parameter_types.get(name)
    if t in ("i32",):
        return struct.unpack("<i", data[:4])[0]
    if t in ("i64",):
        return struct.unpack("<q", data[:8])[0]
    if t in ("u32",):
        return struct.unpack("<I", data[:4])[0]
    if t in ("u64",):
        return struct.unpack("<Q", data[:8])[0]
    if t in ("f64",):
        return struct.unpack("<d", data[:8])[0]
    if t == "bool":
        return bool(struct.unpack("<i", data[:4])[0])
    if t == "string":
        return data.rstrip(b"\x00").decode("utf-8", "replace")
    if t and t.startswith("array<"):
        # Best-effort: split bytes into 8 i64 elements (matches preprocessor's stub layout)
        if len(data) >= 64:
            return list(struct.unpack("<8q", data[:64]))
        return []
    return data.hex()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klee-out", type=pathlib.Path, required=True)
    ap.add_argument("--subject", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    meta = json.loads(args.subject.read_text())
    param_types = meta.get("parameter_types", {})
    inputs = []
    for ktest in sorted(args.klee_out.glob("*.ktest")):
        try:
            objs = parse_ktest(ktest)
        except (ValueError, struct.error) as e:
            print(f"  warn: {ktest.name}: {e}", file=sys.stderr)
            continue
        kwargs = {}
        for name, blob in objs:
            if name in param_types:
                kwargs[name] = decode(name, blob, param_types)
        if kwargs:
            inputs.append({"args": kwargs, "source": ktest.name})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for inp in inputs:
            f.write(json.dumps(inp) + "\n")
    print(f"converted {len(inputs)} klee paths -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
