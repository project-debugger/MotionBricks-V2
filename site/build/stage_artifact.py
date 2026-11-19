#!/usr/bin/env python3
"""Stage the claude.ai artifact copy of the playground (Light model only) into a directory.

claude.ai artifacts take standard web media types only and at most 15 MB per binary file, so every
.bin file (model parts, motion targets, mesh) is copied byte-for-byte under a .wasm name; the page's
<meta name="soma-bin-ext"> tells main.js, and models.json lists the renamed parts.

    python3 site/build/make_index.py && python3 site/build/stage_artifact.py OUT_DIR [--models light]
"""
import argparse
import json
import shutil
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
MAX_BINARY = 15_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--models", nargs="+", default=["light"])
    a = ap.parse_args()
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "js").mkdir(parents=True)
    shutil.copy(SITE / "build/artifact/index.html", out / "index.html")
    for f in (SITE / "js").glob("*.js"):
        shutil.copy(f, out / "js" / f.name)
    for sub, files in {"vendor/ort": ["ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"],
                       "vendor/three": ["three.module.min.js"]}.items():
        (out / sub).mkdir(parents=True)
        for f in files:
            shutil.copy(SITE / sub / f, out / sub / f)
        shutil.copy(SITE / sub / "LICENSE", out / sub / "LICENSE.txt")
    (out / "data").mkdir()
    for f in (SITE / "data").iterdir():
        shutil.copy(f, out / "data" / (f.stem + ".wasm" if f.suffix == ".bin" else f.name))
    manifest = json.loads((SITE / "models/models.json").read_text())
    manifest["models"] = {k: manifest["models"][k] for k in a.models}
    for k, m in manifest["models"].items():
        (out / "models" / k).mkdir(parents=True)
        for p in m["parts"]:
            name = p["file"].replace(".bin", ".wasm")
            shutil.copy(SITE / "models" / k / p["file"], out / "models" / k / name)
            p["file"] = name
    (out / "models/models.json").write_text(json.dumps(manifest, indent=1))
    big = [f for f in out.rglob("*") if f.is_file() and f.stat().st_size > MAX_BINARY]
    assert not big, f"over the 15 MB per-file limit: {big}"
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"staged {sum(1 for f in out.rglob('*') if f.is_file())} files, {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
