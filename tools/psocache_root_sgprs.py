"""Where a graphics shader's root parameters sit in its user data, measured on the PS\\0\\x09 PSO cache.

Reads a carved cache (tools/psocache_extract.py) and, for every graphics pipeline record, the shader slots at +0x38:
seven `{i32 shader index, i32 meta entry}` pairs, -1 when empty; slot 0 holds the pixel, 1 the vertex and 2 the mesh
shader. Each shader's meta entry (docs/pso_cache.md) is a header (0x10 bytes for kind 2), 32-byte records tagged 0x41
or 0x42, then 0x43 and the container's 16-byte HASH digest. In a 0x41 record:

    +0x04 byte  binding class: 0 SRV, 1 sampler, 2 CBV, 3 UAV
    +0x14 byte  size in dwords
    +0x15 byte  the user-data index of the binding's root argument (a table's address, a root view's two dwords,
                the root constants); 0 for a static sampler
    +0x16 byte  the root parameter index (a static sampler's own index for a static sampler)

These offsets come from five-star-recomp's notes (docs/pso_cache.md credits them); this tool checks them rather than
assuming them. Per stage it counts, over the distinct shaders:

- the user-data index of root parameter 0;
- for consecutive root parameters, the (parameter step, index step) pairs: a table or constant parameter takes one
  index and a root view two, so a step of one parameter is one or two indices;
- for every pipeline, whether its vertex or mesh shader and its pixel shader give a shared parameter the same index.

With --dxbc and --index it also joins each cached shader to the carved container with the same HASH digest and
compares the lowest user SGPR the translated program reads at or above s8 (spirv/index.json `user_sgpr_reads`) with
the lowest index the cache gives: the difference is how far above its user-data index the stage's hardware puts a
user SGPR (0 for vertex and pixel shaders, 8 for mesh shaders on this build).

It prints counts only, never a hash, a name or a byte of the cache.

Usage:
    python -I tools/psocache_root_sgprs.py <cache> [--dxbc <work>/shaders/dxbc --index <work>/shaders/spirv/index.json]
                                           [--json] [--expect]

--expect exits 1 unless the graphics layout is the one the runtime uses: root parameter 0 at index 8 in every stage
that binds it, and every vertex+pixel and mesh+pixel pipeline agreeing on its shared parameters (and, with the
join, the most common SGPR shift 0 for vertex and pixel shaders and 8 for mesh shaders).
Exit code: 0 on success, 1 when --expect fails, 2 when the cache cannot be read.
"""

import argparse
import collections
import json
import os
import struct
import sys

MAGIC = b"PS\x00\x09"
HEADER_SIZE = 0x80
GRAPHICS_STRIDE = 0x70
COMPUTE_STRIDE = 0x18
SLOTS_OFF = 0x38
SLOT_COUNT = 7
STAGE_SLOTS = {0: "ps", 1: "vs", 2: "ms"}
RECORD = 32
TAG_BINDING, TAG_TABLE, TAG_END = 0x41, 0x42, 0x43
SAMPLER = 1
EXPECTED_BASE = 8
EXPECTED_SHIFT = {"ps": 0, "vs": 0, "ms": 8}


class CacheError(Exception):
    pass


class Cache:
    def __init__(self, data):
        if len(data) < HEADER_SIZE or data[:4] != MAGIC:
            raise CacheError("not a PS\\0\\x09 cache")
        self.m = data
        self.graphics, self.compute, self.metas, self.shaders = struct.unpack_from("<4I", data, 0x50)
        self.meta_bytes = struct.unpack_from("<I", data, 0x60)[0]
        self.meta_index = HEADER_SIZE + self.graphics * GRAPHICS_STRIDE + self.compute * COMPUTE_STRIDE
        self.meta_area = self.meta_index + 8 * self.metas + 8 * self.shaders
        if self.meta_area + self.meta_bytes > len(data):
            raise CacheError("cache is shorter than its header says")

    def slots(self, record):
        """{stage: meta entry} of graphics pipeline `record`."""
        base = HEADER_SIZE + record * GRAPHICS_STRIDE + SLOTS_OFF
        out = {}
        for k in range(SLOT_COUNT):
            shader, meta = struct.unpack_from("<ii", self.m, base + 8 * k)
            if k in STAGE_SLOTS and shader >= 0 and 0 <= meta < self.metas:
                out[STAGE_SLOTS[k]] = meta
        return out

    def bindings(self, meta):
        """([(root parameter, user-data index)], HASH digest or None) for one shader's meta entry."""
        off, size = struct.unpack_from("<II", self.m, self.meta_index + 8 * meta)
        start = self.meta_area + off
        end = min(start + size, self.meta_area + self.meta_bytes)
        kind = struct.unpack_from("<I", self.m, start)[0]
        p = start + (0x10 if kind == 2 else 0x18)
        out, digest = [], None
        while p + 4 <= end:
            tag = struct.unpack_from("<I", self.m, p)[0]
            if tag == TAG_END:
                digest = bytes(self.m[p + 4:p + 20])
                break
            if tag not in (TAG_BINDING, TAG_TABLE) or p + RECORD > end:
                raise CacheError(f"meta entry {meta}: unexpected record")
            if tag == TAG_BINDING:
                cls, index, param = self.m[p + 4], self.m[p + 0x15], self.m[p + 0x16]
                if not (cls == SAMPLER and index == 0):   # a static sampler has no user data
                    out.append((param, index))
            p += RECORD
        return out, digest


def parameter_index(bindings):
    """{root parameter: user-data index}; None when one parameter names two indices."""
    seen = {}
    for param, index in bindings:
        if seen.setdefault(param, index) != index:
            return None
    return seen


def container_digests(dxbc_dir):
    """{HASH digest: [container name]} over the carved containers (the HASH part's 16-byte digest)."""
    out = collections.defaultdict(list)
    for name in sorted(os.listdir(dxbc_dir)):
        if not name.endswith(".dxbc"):
            continue
        with open(os.path.join(dxbc_dir, name), "rb") as f:
            d = f.read()
        if len(d) < 32 or d[:4] != b"DXBC":
            continue
        count = struct.unpack_from("<I", d, 28)[0]
        for k in range(min(count, (len(d) - 32) // 4)):
            o = struct.unpack_from("<I", d, 32 + 4 * k)[0]
            if d[o:o + 4] == b"HASH" and o + 28 <= len(d):
                out[d[o + 12:o + 28]].append(name[:-5])
                break
    return out


def measure(cache, digests=None, index=None):
    stages = {s: {"shaders": 0, "no_root_params": 0, "conflicts": 0, "param0_index": collections.Counter(),
                  "steps": collections.Counter(), "shift": collections.Counter()} for s in STAGE_SLOTS.values()}
    pairs = collections.Counter()
    done = set()
    for record in range(cache.graphics):
        slots = cache.slots(record)
        layout = {}
        for stage, meta in slots.items():
            binds, digest = cache.bindings(meta)
            params = parameter_index(binds)
            layout[stage] = params
            if meta in done:
                continue
            done.add(meta)
            s = stages[stage]
            s["shaders"] += 1
            if params is None:
                s["conflicts"] += 1
                continue
            if not params:
                s["no_root_params"] += 1
                continue
            if 0 in params:
                s["param0_index"][params[0]] += 1
            order = sorted(params)
            for a, b in zip(order, order[1:]):
                s["steps"][(b - a, params[b] - params[a])] += 1
            if digests is not None:
                names = [n for n in digests.get(digest, ()) if n in index]
                reads = [r["sgpr"] for r in index[names[0]].get("user_sgpr_reads", ())] if names else []
                above = [r for r in reads if r >= EXPECTED_BASE]
                s["shift"]["no program" if not names else "no read at s8 or above" if not above
                           else min(above) - min(params.values())] += 1
        if "ps" in layout:
            for other in ("vs", "ms"):
                if other not in layout:
                    continue
                a, b = layout["ps"], layout[other]
                if a is None or b is None:
                    pairs[f"{other}+ps: a parameter with two indices"] += 1
                    continue
                shared = set(a) & set(b)
                key = "no shared parameter" if not shared else \
                    "agree" if all(a[p] == b[p] for p in shared) else "disagree"
                pairs[f"{other}+ps: {key}"] += 1
    return {"graphics_pipelines": cache.graphics, "stages": stages, "pairs": pairs}


def check(report, joined):
    problems = []
    for stage, s in report["stages"].items():
        bases = set(s["param0_index"])
        if bases and bases != {EXPECTED_BASE}:
            problems.append(f"{stage}: root parameter 0 at {sorted(bases)}, not only {EXPECTED_BASE}")
        if s["conflicts"]:
            problems.append(f"{stage}: {s['conflicts']} shaders give one parameter two indices")
        for (dp, di), n in s["steps"].items():
            if not dp <= di <= 2 * dp:
                problems.append(f"{stage}: {n} parameter steps of {dp} move {di} indices")
        if joined:
            numeric = {k: n for k, n in s["shift"].items() if isinstance(k, int)}
            if numeric and max(numeric, key=numeric.get) != EXPECTED_SHIFT[stage]:
                problems.append(f"{stage}: most common SGPR shift {max(numeric, key=numeric.get)}, "
                                f"not {EXPECTED_SHIFT[stage]}")
    for key, n in report["pairs"].items():
        if not key.endswith((": agree", ": no shared parameter")):
            problems.append(f"{n} pipelines: {key}")
    return problems


def printable(report):
    out = {"graphics_pipelines": report["graphics_pipelines"], "stages": {}, "pairs": dict(report["pairs"])}
    for stage, s in report["stages"].items():
        out["stages"][stage] = {
            "shaders": s["shaders"], "no_root_params": s["no_root_params"], "conflicts": s["conflicts"],
            "param0_index": {str(k): v for k, v in sorted(s["param0_index"].items())},
            "steps": {f"{dp} param -> {di} index": n for (dp, di), n in sorted(s["steps"].items())},
            "sgpr_shift": {str(k): v for k, v in sorted(s["shift"].items(), key=lambda kv: str(kv[0]))}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cache", help="carved PS\\0\\x09 cache (read-only)")
    ap.add_argument("--dxbc", help="the work folder's carved containers (shaders/dxbc), for the SGPR join")
    ap.add_argument("--index", help="the work folder's shaders/spirv/index.json, for the SGPR join")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--expect", action="store_true", help="exit 1 unless the runtime's graphics layout holds")
    a = ap.parse_args(argv)
    if bool(a.dxbc) != bool(a.index):
        ap.error("--dxbc and --index go together")
    try:
        with open(a.cache, "rb") as f:
            cache = Cache(f.read())
        digests = index = None
        if a.dxbc:
            digests = container_digests(a.dxbc)
            with open(a.index, encoding="utf-8-sig") as f:
                index = json.load(f)
        report = measure(cache, digests, index)
    except (OSError, ValueError, struct.error, CacheError) as e:
        print(f"psocache_root_sgprs: {e}", file=sys.stderr)
        return 2
    out = printable(report)
    if a.json:
        print(json.dumps(out, indent=1, sort_keys=True))
    else:
        print(f"graphics pipelines: {out['graphics_pipelines']}")
        for stage, s in out["stages"].items():
            print(f"{stage}: {s['shaders']} shaders, {s['no_root_params']} without root parameters, "
                  f"{s['conflicts']} with a parameter at two indices")
            print(f"  root parameter 0 at index: {s['param0_index']}")
            print(f"  consecutive parameters: {s['steps']}")
            if a.dxbc:
                print(f"  lowest read at s8+ minus lowest index: {s['sgpr_shift']}")
        for key, n in sorted(out["pairs"].items()):
            print(f"{key}: {n}")
    if a.expect:
        problems = check(report, bool(a.dxbc))
        for p in problems:
            print(f"expect: {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
