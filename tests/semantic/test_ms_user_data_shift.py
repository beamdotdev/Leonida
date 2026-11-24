"""A mesh shader loads its user SGPR i + 8 from user-data dword i; the other stages load SGPR i from dword i.

The game's PSO cache gives every shader's root parameters as user-data indices (docs/pso_cache.md, "Root parameters
in user SGPRs"): parameter 0 is index 8 in the vertex, pixel and mesh shaders alike, and the shaders of a pipeline
agree on every index. The mesh programs read those arguments 8 SGPRs higher (NGG keeps eight system SGPRs below the
user data), so one push-constant block serves a mesh+pixel pipeline only if the mesh shader's copy is shifted.

The programs and the PSV0 part are written here; no game data.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)
from test_ms_lds_need import mesh_dxbc  # noqa: E402

rdna2spirv = support.rdna2spirv

PROGRAM = ["s_mov_b64 exec, -1", "v_mov_b32_e32 v1, s16", "s_endpgm"]


def user_data_copy(spvasm):
    """{push-constant dword: SGPR} as the entry block's user-data copy stores them."""
    consts = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(%\w+) = OpConstant %uint (\d+)$", spvasm, re.M)}
    dword = {m.group(1): consts[m.group(2)]
             for m in re.finditer(r"^%ud(\d+)p = OpAccessChain %_ptr_PC_uint %user_data %c0 (%\w+)$", spvasm, re.M)}
    sgpr = {m.group(1): consts[m.group(2)]
            for m in re.finditer(r"^%ud(\d+)s = OpAccessChain %_ptr_Function_uint %sgpr (%\w+)$", spvasm, re.M)}
    return {dword[k]: sgpr[k] for k in dword}


class MeshUserDataShiftTests(unittest.TestCase):
    def test_mesh_sgpr_is_dword_plus_eight(self):
        copy = user_data_copy(support.translate_lines("ms_user_shift", PROGRAM, dxbc=mesh_dxbc(0)))
        self.assertEqual(rdna2spirv.MS_USER_SHIFT, 8)
        # dwords 0..23 fill s8..s31; dword 8, a graphics pipeline's first root argument, is the s16 the census sees
        self.assertEqual(copy, {d: d + 8 for d in range(24)})

    def test_pixel_sgpr_is_dword(self):
        copy = user_data_copy(support.translate_lines("ps_user_shift", ["v_mov_b32_e32 v1, s8", "s_endpgm"]))
        self.assertEqual(copy, {d: d for d in range(rdna2spirv.USER_SGPRS)})


if __name__ == "__main__":
    unittest.main()
