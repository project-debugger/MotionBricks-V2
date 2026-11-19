#!/usr/bin/env python3
"""Download ONLY the `soma/` folder of nvidia/MotionBricks from Hugging Face.

Features
- Lists the folder through the HF tree API (recursive) - never touches `g1/`.
- Parallel, resumable downloads (HTTP Range on `<file>.part`).
- Verifies every file against BOTH the repository-level `SHA256SUMS`
  and the folder-level `soma/SHA256SUMS` before it is moved into place.
- Idempotent: files already present with a matching hash are skipped.
- Writes a verification report to models/soma/checksums/.

Usage:
    python3 tools/download_soma.py                # download + verify
    python3 tools/download_soma.py --verify-only  # re-hash what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO = "nvidia/MotionBricks"
REVISION = "main"
SUBFOLDER = "soma"
API_TREE = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}/{SUBFOLDER}?recursive=true"
RESOLVE = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "soma" / "original"
CHECKSUM_DIR = ROOT / "models" / "soma" / "checksums"
CHUNK = 8 * 1024 * 1024


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_sums(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        out[name.lstrip("*").lstrip("./")] = digest.lower()
    return out


def fetch_text(session: requests.Session, rel: str) -> str:
    r = session.get(RESOLVE + rel, timeout=60)
    r.raise_for_status()
    return r.text


def list_files(session: requests.Session) -> list[dict]:
    r = session.get(API_TREE, timeout=60)
    r.raise_for_status()
    entries = [e for e in r.json() if e["type"] == "file"]
    for e in entries:
        if not e["path"].startswith(SUBFOLDER + "/"):
            raise RuntimeError(f"Refusing non-{SUBFOLDER} path: {e['path']}")
    return entries


def download_one(entry: dict, expected: str, retries: int = 5) -> tuple[str, str]:
    rel = entry["path"][len(SUBFOLDER) + 1:]
    target = DEST / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    size = entry.get("size", 0)

    if target.exists() and target.stat().st_size == size and sha256_file(target) == expected:
        return rel, "skipped (verified)"

    part = target.with_name(target.name + ".part")
    session = requests.Session()
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            if size and have > size:
                part.unlink()
                have = 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            if size and have == size:
                break
            with session.get(RESOLVE + entry["path"], headers=headers, stream=True,
                             timeout=(30, 120), allow_redirects=True) as r:
                if have and r.status_code == 200:  # server ignored Range -> restart
                    have = 0
                    mode = "wb"
                elif r.status_code in (200, 206):
                    mode = "ab" if have else "wb"
                else:
                    r.raise_for_status()
                    mode = "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            if not size or part.stat().st_size == size:
                break
        except (requests.RequestException, OSError) as exc:
            wait = 2 ** attempt
            log(f"  retry {attempt}/{retries} for {rel}: {exc!r} (sleep {wait}s)")
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to download {rel}")

    got = sha256_file(part)
    if got != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"SHA256 mismatch for {rel}: got {got}, expected {expected}")
    os.replace(part, target)
    return rel, f"downloaded ({size} B)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    CHECKSUM_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()

    root_sums = {k[len(SUBFOLDER) + 1:]: v for k, v in parse_sums(fetch_text(s, "SHA256SUMS")).items()
                 if k.startswith(SUBFOLDER + "/")}
    folder_sums_text = fetch_text(s, f"{SUBFOLDER}/SHA256SUMS")
    folder_sums = parse_sums(folder_sums_text)
    (CHECKSUM_DIR / "SHA256SUMS.repo-root.soma").write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(root_sums.items())))
    (CHECKSUM_DIR / "SHA256SUMS.soma-folder").write_text(folder_sums_text)

    # Cross-check: both manifests must agree wherever they overlap.
    for k in set(root_sums) & set(folder_sums):
        if root_sums[k] != folder_sums[k]:
            raise RuntimeError(f"Upstream checksum files disagree on {k}")

    entries = list_files(s)
    log(f"{len(entries)} files, {sum(e.get('size', 0) for e in entries)/1e6:.1f} MB in {SUBFOLDER}/")

    results: dict[str, dict] = {}
    if args.verify_only:
        for e in entries:
            rel = e["path"][len(SUBFOLDER) + 1:]
            p = DEST / rel
            exp = root_sums.get(rel) or folder_sums.get(rel)
            ok = p.exists() and exp is not None and sha256_file(p) == exp
            results[rel] = {"expected": exp, "ok": ok}
    else:
        jobs = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            # biggest first so the long tail is short
            for e in sorted(entries, key=lambda e: -e.get("size", 0)):
                rel = e["path"][len(SUBFOLDER) + 1:]
                exp = root_sums.get(rel) or folder_sums.get(rel)
                if exp is None:
                    raise RuntimeError(f"No checksum published for {rel}")
                jobs[ex.submit(download_one, e, exp)] = (rel, exp)
            for fut in as_completed(jobs):
                rel, exp = jobs[fut]
                try:
                    _, status = fut.result()
                    results[rel] = {"expected": exp, "ok": True, "status": status}
                    log(f"OK   {rel}: {status}")
                except Exception as exc:  # noqa: BLE001
                    results[rel] = {"expected": exp, "ok": False, "status": str(exc)}
                    log(f"FAIL {rel}: {exc}")

    # Files present on disk but not upstream (e.g. stale partials) are reported, never deleted.
    upstream = {e["path"][len(SUBFOLDER) + 1:] for e in entries}
    extra = [str(p.relative_to(DEST)) for p in DEST.rglob("*") if p.is_file()
             and str(p.relative_to(DEST)) not in upstream]

    report = {
        "repo": REPO, "revision": REVISION, "subfolder": SUBFOLDER,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_ok": all(r["ok"] for r in results.values()) and len(results) == len(entries),
        "files": dict(sorted(results.items())), "unexpected_local_files": extra,
    }
    (CHECKSUM_DIR / "verification_report.json").write_text(json.dumps(report, indent=2))
    (CHECKSUM_DIR / "SHA256SUMS.local").write_text(
        "".join(f"{r['expected']}  {k}\n" for k, r in sorted(results.items()) if r["ok"]))
    log(f"ALL OK = {report['all_ok']}; unexpected local files: {extra}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
