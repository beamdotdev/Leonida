"""tools/psocache_root_sgprs.py on a synthetic PS\\0\\x09 cache: no game data.

The cache is built here: a header, graphics records whose slots at +0x38 name meta entries, the meta index and the
meta area, each entry a kind-2 header, 0x41 binding records (class, size, user-data index, root parameter) and the
0x43 digest. The join reads containers and an index written to a temporary folder.
"""
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import psocache_root_sgprs as prs  # noqa: E402


def binding(cls, index, param, size=8):
    r = bytearray(32)
    struct.pack_into("<I", r, 0, 0x41)
    r[4] = cls
    r[0x14], r[0x15], r[0x16] = size, index, param
    return bytes(r)


def meta_entry(bindings, digest):
    return (struct.pack("<4I", 2, 0, 0, 0x40) + b"".join(bindings) + struct.pack("<I", 0x43) + digest
            + bytes(12))


def cache(pipelines, metas):
    """pipelines: [{slot: meta entry}]; metas: [bytes]."""
    head = bytearray(prs.HEADER_SIZE)
    head[:4] = prs.MAGIC
    area = b"".join(metas)
    struct.pack_into("<4I", head, 0x50, len(pipelines), 0, len(metas), 0)
    struct.pack_into("<I", head, 0x60, len(area))
    recs = b""
    for slots in pipelines:
        r = bytearray(prs.GRAPHICS_STRIDE)
        for k in range(prs.SLOT_COUNT):
            struct.pack_into("<ii", r, prs.SLOTS_OFF + 8 * k, *((k, slots[k]) if k in slots else (-1, -1)))
        recs += bytes(r)
    index, off = b"", 0
    for m in metas:
        index += struct.pack("<II", off, len(m))
        off += len(m)
    return bytes(head) + recs + index + area


def dxbc_with_hash(digest):
    part = b"HASH" + struct.pack("<I", 20) + bytes(4) + digest
    return b"DXBC" + bytes(16) + struct.pack("<4I", 1, 36 + len(part), 1, 36) + part


TABLE, ROOT_VIEW = 0, 2   # binding classes: SRV in a table, CBV as a root view


def console_metas():
    """A mesh+pixel and a vertex+pixel pipeline in the console layout: [0] table at 8, [1] root view at 9..10,
    [2] table at 11; a static sampler (class 1, index 0) in the pixel shader."""
    ps = meta_entry([binding(TABLE, 8, 0), binding(ROOT_VIEW, 9, 1, size=2), binding(1, 0, 0, size=4)],
                    b"P" * 16)
    ms = meta_entry([binding(TABLE, 8, 0), binding(TABLE, 11, 2)], b"M" * 16)
    vs = meta_entry([binding(ROOT_VIEW, 9, 1, size=2)], b"V" * 16)
    return [ps, ms, vs]


class RootSgprTests(unittest.TestCase):
    def report(self, pipelines, metas, **join):
        return prs.measure(prs.Cache(cache(pipelines, metas)), **join)

    def test_console_layout_holds(self):
        r = self.report([{0: 0, 2: 1}, {0: 0, 1: 2}], console_metas())
        self.assertEqual(dict(r["stages"]["ps"]["param0_index"]), {8: 1})
        self.assertEqual(dict(r["stages"]["ms"]["param0_index"]), {8: 1})
        self.assertEqual(dict(r["stages"]["ms"]["steps"]), {(2, 3): 1})       # a root view between two tables
        self.assertEqual(dict(r["stages"]["ps"]["steps"]), {(1, 1): 1})
        self.assertEqual(dict(r["pairs"]), {"ms+ps: agree": 1, "vs+ps: agree": 1})
        self.assertEqual(r["stages"]["ps"]["shaders"], 1)                     # counted once, used twice
        self.assertEqual(prs.check(r, joined=False), [])

    def test_pack_layout_fails_the_check(self):
        # the old graphics model: from index 0, two indices per table
        ps = meta_entry([binding(TABLE, 0, 0), binding(TABLE, 2, 1)], b"P" * 16)
        ms = meta_entry([binding(TABLE, 8, 0)], b"M" * 16)
        r = self.report([{0: 0, 2: 1}], [ps, ms])
        problems = prs.check(r, joined=False)
        self.assertTrue(any("ps: root parameter 0 at [0]" in p for p in problems), problems)
        self.assertTrue(any("ms+ps: disagree" in p for p in problems), problems)

    def test_one_parameter_at_two_indices_is_a_conflict(self):
        ps = meta_entry([binding(TABLE, 8, 0), binding(TABLE, 9, 0)], b"P" * 16)
        r = self.report([{0: 0}], [ps])
        self.assertEqual(r["stages"]["ps"]["conflicts"], 1)

    def test_join_measures_the_mesh_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, digest in (("ps_a", b"P" * 16), ("ms_a", b"M" * 16), ("vs_a", b"V" * 16)):
                with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
                    f.write(dxbc_with_hash(digest))
            index = {"ps_a": {"user_sgpr_reads": [{"kind": "ptr32_lo", "sgpr": 8}, {"kind": "const32", "sgpr": 1}]},
                     "ms_a": {"user_sgpr_reads": [{"kind": "ptr32_lo", "sgpr": 16}, {"kind": "ptr32_lo", "sgpr": 19}]},
                     "vs_a": {"user_sgpr_reads": [{"kind": "ptr64_lo", "sgpr": 9}]}}
            digests = prs.container_digests(tmp)
            r = self.report([{0: 0, 2: 1}, {0: 0, 1: 2}], console_metas(), digests=digests, index=index)
        self.assertEqual(dict(r["stages"]["ms"]["shift"]), {8: 1})
        self.assertEqual(dict(r["stages"]["ps"]["shift"]), {0: 1})
        self.assertEqual(dict(r["stages"]["vs"]["shift"]), {0: 1})
        self.assertEqual(prs.check(r, joined=True), [])

    def test_main_expect_and_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.bin")
            with open(good, "wb") as f:
                f.write(cache([{0: 0, 2: 1}], console_metas()))
            bad = os.path.join(tmp, "bad.bin")
            with open(bad, "wb") as f:
                f.write(b"not a cache")
            with open(os.devnull, "w") as null:
                saved = sys.stdout
                sys.stdout = null
                try:
                    self.assertEqual(prs.main([good, "--expect"]), 0)
                    self.assertEqual(prs.main([good, "--json"]), 0)
                finally:
                    sys.stdout = saved
                saved = sys.stderr
                sys.stderr = null
                try:
                    self.assertEqual(prs.main([bad]), 2)
                finally:
                    sys.stderr = saved


if __name__ == "__main__":
    unittest.main()
