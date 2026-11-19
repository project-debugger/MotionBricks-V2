#!/usr/bin/env python3
"""Generate every showcase motion with the REAL SOMA runtime (no keyframes, no clip playback).

For each video tile: Config (balanced preset + tile overrides) -> SomaEngine (ONNX Runtime, the
tile's model file) -> CharacterController driven by the scripted timeline at 30 Hz. Synchronous
generation is used so the result is frame-exact and identical on every run; the controller logic
(trajectory, receding-horizon planning, velocity matching, foot locking) is the production code.

Writes showcase/build/cache/motions/<video>/tileN.bin (+ <video>.json) and a provenance manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(HERE))
from scenarios import VIDEOS  # noqa: E402
from somart import codec  # noqa: E402
from somart import contract as C  # noqa: E402
from somart.backends import OrtBackend  # noqa: E402
from somart.clips import ClipLibrary  # noqa: E402
from somart.config import Config  # noqa: E402
from somart.controller import CharacterController, Command  # noqa: E402
from somart.engine import SomaEngine  # noqa: E402

FPS = 30
OUT = HERE / "cache" / "motions"
CAMERA = {  # layout -> distance, height, look-at height, vertical fov (deg), azimuth offset from facing (deg)
    "1": (3.7, 1.25, 0.90, 36.0, 62.0), "2": (4.0, 1.25, 0.90, 38.0, 62.0), "3": (4.3, 1.25, 0.90, 38.0, 62.0),
    "4": (3.3, 1.25, 0.90, 38.0, 62.0), "feet": (4.6, 3.1, 0.30, 40.0, 72.0),
}


def command_at(timeline, t, fired):
    row = timeline[0]
    for r in timeline:
        if t >= r[0]:
            row = r
    start, direction, mag, gait, action, caption = row
    act = None
    if action and start not in fired:
        fired.add(start)
        act = action
    move = (0.0, 0.0) if direction is None else (math.sin(math.radians(direction)) * mag,
                                                 math.cos(math.radians(direction)) * mag)
    return Command(move=move, gait=gait, action=act), caption


def gaussian_smooth(x, sigma_s):
    """Centered (non-causal) Gaussian smoothing along time: no lag, offline only."""
    sig = sigma_s * FPS
    r = int(3 * sig)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)], 0)
    return np.stack([np.convolve(pad[:, c], k, mode="valid") for c in range(x.shape[1])], -1)


def camera_path(world, layout, sk):
    """Smooth orbit camera at a front-three-quarter angle relative to the (smoothed) facing."""
    d, h, look_h, fov, off = CAMERA[layout]
    b = codec.SkeletonBinding.bind(sk)
    hips = world[:, 0, :3] * np.array([1.0, 0.0, 1.0])
    tgt = gaussian_smooth(hips, 0.35)
    face = codec.heading_from_hips(world[:, b.right_leg, :3], world[:, b.left_leg, :3])
    vec = gaussian_smooth(np.stack([np.sin(face), np.cos(face)], -1), 1.1)
    yaw = np.arctan2(vec[:, 0], vec[:, 1])
    t = np.arange(len(tgt)) / FPS
    az = yaw + np.radians(off + 12.0 * np.sin(2 * np.pi * t / 14.0))
    pos = tgt + np.stack([np.sin(az) * d, np.full_like(az, h), np.cos(az) * d], -1)
    look = tgt + np.array([0.0, look_h, 0.0])
    return np.concatenate([pos, look], -1), fov


def detect_props(world, sk, timeline, fps=FPS):
    props = []
    hips_y = world[:, 0, 1]
    feet = [sk.index(n) for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase")]
    b = codec.SkeletonBinding.bind(sk)
    for start, _, _, _, action, _ in timeline:
        if action not in ("jump", "sit"):
            continue
        a, z = int(start * fps), min(len(world), int((start + 3.0) * fps))
        if z - a < 5:
            continue
        seg = hips_y[a:z]
        k = a + int(np.argmax(seg) if action == "jump" else np.argmin(seg))
        p = world[k, 0, :3]
        head = float(codec.heading_from_hips(world[k, b.right_leg, :3], world[k, b.left_leg, :3]))
        if action == "jump":
            lo, hi = max(0, k - 2), min(len(world), k + 3)
            clearance = float(world[lo:hi][:, feet, 1].min())
            props.append({"type": "hurdle", "x": float(p[0]), "z": float(p[2]), "yaw": head,
                          "height": float(np.clip(clearance - 0.05, 0.12, 0.5)),
                          "note": "visual prop placed at the generated jump apex (the model does not see it)"})
        else:
            back = np.array([math.sin(head), 0, math.cos(head)])
            c = p - back * 0.08
            props.append({"type": "chair", "x": float(c[0]), "z": float(c[2]), "yaw": head,
                          "seat": float(max(0.3, hips_y[k] - 0.12)),
                          "note": "visual prop placed under the generated seated pose"})
    return props


def run_tile(spec, layout, lib, engines, out_bin, feet_cam):
    over = json.loads(json.dumps(spec["overrides"]))
    model = spec["model"]
    over.setdefault("inference", {}).update(
        {"model": model, "models_by_device": {"cpu": model, "gpu": model}, "threads": 4, "async": False})
    cfg = Config.load("balanced", overrides=over)
    if model not in engines:
        engines[model] = SomaEngine(OrtBackend(ROOT / model, intra_threads=4), lib.skeleton)
    eng = engines[model]
    ctrl = CharacterController(eng, lib, cfg, synchronous=True)
    n = int(spec["end"] * FPS)
    fired: set = set()
    local, contacts, goals, captions, net_ms = [], [], [], [], []
    for i in range(n):
        cmd, cap = command_at(spec["timeline"], i / FPS, fired)
        before = ctrl.stats.generations
        local.append(ctrl.update(cmd, 1 / FPS))
        if ctrl.stats.generations > before:
            net_ms.append(eng.last_timings["inference"])
        contacts.append(ctrl._sample_contacts(ctrl.time))
        g = ctrl._goal_velocity(cmd)
        goals.append([g[0], g[2]])
        captions.append(cap)
    local = np.array(local)
    world = lib.skeleton.fk(local)
    arr = np.concatenate([world.reshape(n, -1), np.array(contacts)], axis=1).astype(np.float32)
    out_bin.write_bytes(arr.tobytes())
    hips_ground = world[:, 0, :3] * np.array([1.0, 0.0, 1.0])
    cam, fov = camera_path(world, "feet" if feet_cam else layout, lib.skeleton)
    speed = np.zeros(n)
    speed[1:] = np.linalg.norm(np.diff(hips_ground, axis=0), axis=-1) * FPS
    speed = np.convolve(speed, np.ones(9) / 9, mode="same")
    return {
        "label": spec["label"], "sub": spec["sub"], "bin": out_bin.name, "frames": n,
        "camera": np.round(cam, 4).tolist(), "fov": fov, "captions": captions,
        "goal": np.round(goals, 3).tolist(), "speed": np.round(speed, 2).tolist(),
        "props": detect_props(world, lib.skeleton, spec["timeline"]) if spec.get("props") else [],
        "provenance": {"model": model, "model_sha256_16": sha16(ROOT / model),
                       "generations": ctrl.stats.generations, "late": ctrl.stats.late_results,
                       "network_ms_mean": float(np.mean(net_ms)) if net_ms else None,
                       "config": cfg.diff()},
    }


_SHA: dict = {}


def sha16(p: Path) -> str:
    if p not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        _SHA[p] = h.hexdigest()[:16]
    return _SHA[p]


def main() -> int:
    only = set(sys.argv[1:])
    lib = ClipLibrary(ROOT / "third_party/motionbricks/assets/motions/soma").load()
    cfg0 = Config.load("balanced")
    lib.index_windows([cfg0.get(f"actions.{a}.source") for a in ("walk", "strafe", "crouch", "idle", "crouch_idle")])
    engines: dict = {}
    for name, vid in VIDEOS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        tiles = [run_tile(s, vid["layout"], lib, engines, d / f"tile{i}.bin", vid.get("feet_cam", False))
                 for i, s in enumerate(vid["tiles"])]
        spec = {"name": name, "title": vid["title"], "layout": vid["layout"], "fps": FPS,
                "frames": max(t["frames"] for t in tiles), "joints": lib.skeleton.names,
                "parents": lib.skeleton.parents.tolist(), "arrows": vid.get("arrows", False),
                "footprints": vid.get("feet_cam", False), "tiles": tiles}
        (d / "video.json").write_text(json.dumps(spec))
        gens = sum(t["provenance"]["generations"] for t in tiles)
        print(f"[motion] {name}: {len(tiles)} tile(s), {spec['frames']} frames, {gens} SOMA generations, "
              f"{time.time() - t0:.1f}s", flush=True)
    for e in engines.values():
        e.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
