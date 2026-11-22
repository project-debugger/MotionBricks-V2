#!/usr/bin/env python3
"""Extract NVIDIA's SOMA body mesh (upstream USD asset) into a compact skinning file for the renderer.

Source: third_party/motionbricks/motionbricks/assets/human/soma_base_skel_minimal_no_root.usd
(fetched + hash-verified by tools/fetch_upstream.py). The output goes to site/data/ (NVIDIA SOMA mesh, CC BY 4.0).

Output:
  site/data/soma_mesh.bin   float32 positions [V,3] (m) | uint32 triangles [T,3] | uint16 skin index [V,4]
                        | float32 skin weight [V,4] | float32 inverse-bind matrices [77,16] (column-major)
  site/data/soma_mesh.json  sizes / byte offsets + joint names
The 10 USD influences per vertex are reduced to the 4 largest and renormalised (three.js limit).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
USD = ROOT / "third_party/motionbricks/motionbricks/assets/human/soma_base_skel_minimal_no_root.usd"
OUT = ROOT / "site/data"


def main() -> int:
    from pxr import Usd, UsdGeom, UsdSkel
    if not USD.exists() or USD.stat().st_size < 10_000:
        print("SOMA mesh missing: run tools/fetch_upstream.py (LFS asset)", file=sys.stderr)
        return 1
    stage = Usd.Stage.Open(str(USD))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    skel = UsdSkel.Skeleton(stage.GetPrimAtPath("/OUTPUT/c_skeleton_grp/Hips"))
    joints = [str(j).split("/")[-1] for j in skel.GetJointsAttr().Get()]
    bind = np.array([np.array(m) for m in skel.GetBindTransformsAttr().Get()], dtype=np.float64)  # row-vector
    prim = stage.GetPrimAtPath("/OUTPUT/c_geometry_grp/MainMesh/Meshes/c_skin_mid")
    mesh = UsdGeom.Mesh(prim)
    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64) * mpu
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    binding = UsdSkel.BindingAPI(prim)
    geom_bind = np.array(binding.GetGeomBindTransformAttr().Get())
    if not np.allclose(geom_bind, np.eye(4)):
        pts = (np.c_[pts / mpu, np.ones(len(pts))] @ geom_bind)[:, :3] * mpu
    mesh_joints = binding.GetJointsAttr().Get()
    remap = np.arange(len(joints))
    if mesh_joints:
        names = [str(j).split("/")[-1] for j in mesh_joints]
        remap = np.array([joints.index(n) for n in names])
    ji = binding.GetJointIndicesPrimvar()
    jw = binding.GetJointWeightsPrimvar()
    es = ji.GetElementSize()
    sidx = remap[np.array(ji.Get()).reshape(-1, es)]
    swt = np.array(jw.Get(), dtype=np.float64).reshape(-1, es)
    order = np.argsort(-swt, axis=1)[:, :4]
    sidx4 = np.take_along_axis(sidx, order, 1).astype(np.uint16)
    swt4 = np.take_along_axis(swt, order, 1)
    swt4 /= np.maximum(swt4.sum(1, keepdims=True), 1e-9)
    # triangulate (fan) - faces are tris/quads
    tris = []
    o = 0
    for c in counts:
        f = idx[o:o + c]
        for k in range(1, c - 1):
            tris.append((f[0], f[k], f[k + 1]))
        o += c
    tris = np.array(tris, dtype=np.uint32)
    # USD row-vector bind (translation in row 3, cm) -> column-major inverse bind in metres for three.js
    inv = []
    for B in bind:
        M = B.T.copy()            # column-vector convention
        M[:3, 3] *= mpu
        inv.append(np.linalg.inv(M).T.reshape(-1))  # .T then row-major flatten == column-major of inv
    inv = np.array(inv, dtype=np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [pts.astype(np.float32), tris, sidx4, swt4.astype(np.float32), inv]
    meta, off = {"joints": joints, "vertices": len(pts), "triangles": len(tris)}, 0
    with open(OUT / "soma_mesh.bin", "wb") as f:
        for name, arr in zip(("positions", "indices", "skinIndex", "skinWeight", "inverseBind"), parts):
            b = np.ascontiguousarray(arr).tobytes()
            pad = (-len(b)) % 4
            meta[name] = {"offset": off, "length": int(arr.size), "dtype": str(arr.dtype)}
            f.write(b + b"\0" * pad)
            off += len(b) + pad
    (OUT / "soma_mesh.json").write_text(json.dumps(meta))
    print(f"SOMA mesh: {len(pts)} vertices, {len(tris)} triangles, {len(joints)} joints -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
