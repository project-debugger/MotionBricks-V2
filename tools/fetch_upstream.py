#!/usr/bin/env python3
"""Fetch the upstream MotionBricks software (Apache-2.0) at a pinned commit.

The clone lands in `third_party/motionbricks/` (git-ignored, never committed).
Git LFS objects are not served by the anonymous git lane, so the SOMA-related LFS
files we need are fetched from GitHub's public media endpoint and verified against
the OID + size recorded in each LFS pointer file. G1 assets are never fetched.

Fetched LFS groups:
  rig      tools/onnx/assets/soma_model_rig.npz   (needed by the exporter graph)
  motions  assets/motions/soma/**/*.bvh           (evaluation clips; NVIDIA Sample
                                                   Data Evaluation License - local only)
  onnx     motionbricks/assets/onnx/soma/*.onnx   (official reference exports, optional)

Usage: python3 tools/fetch_upstream.py [--groups rig,motions] [--with-official-onnx]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "third_party" / "motionbricks"
URL = "https://github.com/NVlabs/motionbricks"
COMMIT = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"
MEDIA = f"https://media.githubusercontent.com/media/NVlabs/motionbricks/{COMMIT}/"

GROUPS = {
    "rig": ["tools/onnx/assets/soma_model_rig.npz"],
    "motions": "assets/motions/soma",
    "onnx": ["motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx",
             "motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx"],
}


def run(*cmd: str, **kw) -> None:
    subprocess.run(cmd, check=True, **kw)


def clone() -> None:
    if (DEST / ".git").exists():
        head = subprocess.run(["git", "-C", str(DEST), "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
        if head == COMMIT:
            print(f"upstream already at {COMMIT[:10]}")
            return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if not DEST.exists():
        run("git", "init", "-q", str(DEST))
        run("git", "-C", str(DEST), "remote", "add", "origin", URL)
    run("git", "-C", str(DEST), "fetch", "-q", "--depth", "1", "origin", COMMIT, env=env)
    run("git", "-C", str(DEST), "checkout", "-q", "FETCH_HEAD", env=env)
    print(f"cloned {URL} @ {COMMIT[:10]}")


def parse_pointer(path: Path) -> tuple[str, int] | None:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    if not text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    fields = dict(line.split(" ", 1) for line in text.strip().splitlines() if " " in line)
    return fields["oid"].removeprefix("sha256:"), int(fields["size"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_lfs(rel: str) -> str:
    path = DEST / rel
    ptr = parse_pointer(path)
    pointer_file = path.with_name(path.name + ".lfs-pointer")
    if ptr is None and pointer_file.exists():
        ptr = parse_pointer(pointer_file)
        if ptr and path.exists() and path.stat().st_size == ptr[1] and sha256(path) == ptr[0]:
            return f"ok (cached) {rel}"
    if ptr is None:
        return f"skip (not an LFS pointer) {rel}"
    oid, size = ptr
    pointer_file.write_text(path.read_text()) if not pointer_file.exists() else None
    tmp = path.with_name(path.name + ".download")
    with requests.get(MEDIA + rel, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    if tmp.stat().st_size != size or sha256(tmp) != oid:
        tmp.unlink()
        raise RuntimeError(f"LFS verification failed for {rel}")
    os.replace(tmp, path)
    return f"ok (fetched {size} B) {rel}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="rig,motions")
    ap.add_argument("--with-official-onnx", action="store_true")
    args = ap.parse_args()
    clone()
    groups = [g for g in args.groups.split(",") if g]
    if args.with_official_onnx and "onnx" not in groups:
        groups.append("onnx")
    files: list[str] = []
    for g in groups:
        spec = GROUPS[g]
        if isinstance(spec, str):
            files += sorted(str(p.relative_to(DEST)) for p in (DEST / spec).rglob("*.bvh"))
        else:
            files += spec
    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(fetch_lfs, files):
            print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
