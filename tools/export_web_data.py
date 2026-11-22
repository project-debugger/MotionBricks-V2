#!/usr/bin/env python3
"""Export the motion data the browser playground needs into site/data/, from freely licensed motion capture only.

Sources (both retargeted onto the SOMA rig; fetch and retarget them first):
  100STYLE (CC BY 4.0)   walking forward/backward/sideways, slow jogs, idle, starts and stops, walk styles
                         python3 -I tools/mocap/style100.py fetch && python3 -I tools/mocap/style100.py retarget
  CMU mocap (free use)   running (2-4.5 m/s) with turns and stops, crouch walking, jump, sit, duck, side step
                         python3 -I tools/mocap/cmu.py fetch && python3 -I tools/mocap/cmu.py retarget

Writes:
  windows.bin/.json    canonical 4-frame target windows for locomotion matching + action key poses
                       (root position float32, quaternions int16-normalised: 2.5 KB per window)
  clips.bin/.json      continuous 30 fps locomotion/idle clips (phase-aware targets)
  styles/*.bin/.json   optional 100STYLE walk styles, loaded by the page when chosen
  joint_limits.json    anatomical limits fitted on the same clips (tools/fit_joint_limits.py)

Unchanged inputs kept in site/data: skeleton.json (the SOMA rig) and soma_mesh.bin/.json (NVIDIA SOMA body mesh,
CC BY 4.0, from showcase-era tools/extract_mesh.py). The SOMA network weights are not touched here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
sys.path.insert(0, str(ROOT / "tools/mocap"))
from somart import contract as C  # noqa: E402
from somart import math3d as m3  # noqa: E402
from somart.anatomy import JointLimits  # noqa: E402
from somart.clips import ClipLibrary, canonicalize  # noqa: E402
from somart.skeleton import Clip  # noqa: E402
import cmu  # noqa: E402
import style100  # noqa: E402

OUT = ROOT / "site/data"
FPS = 30.0
MAX_FRAMES = 1800        # at most 60 s per 100STYLE clip in clips.bin (keeps the first download small)
CROUCH_H = 0.92          # crouch clips: keep the longest stretch with the pelvis below this (they start and end standing)

# locomotion categories indexed as target windows / phase-aware clips
LOCO = {"100style_walk", "100style_jog", "100style_idle", "100style_transitions",
        "cmu_run", "cmu_run_turn", "cmu_run_stop", "cmu_jog", "cmu_crouch"}

RUN_CATS = {"100style_jog", "cmu_run", "cmu_run_turn", "cmu_run_stop", "cmu_jog"}

# actions: clip, key frame (fraction of the clip, a named key, or {"frame": n}), kind, anchor, and for clip-anchored
# actions the frame their root motion is measured from. "static": the key frame held still (a pose, not a motion).
ACTIONS = {
    "idle":        {"clip": "100STYLE/Neutral_ID.bvh", "key": {"frame": 30}, "kind": "loop"},
    "crouch_idle": {"clip": "CMU/136_10", "key": "crouch_still", "kind": "loop", "static": True},
    "jump":        {"clip": "CMU/16_01", "key": "max_height", "kind": "oneshot"},
    "duck":        {"clip": "CMU/77_09", "key": {"frame": 76}, "kind": "oneshot"},  # first duck: squat, head down, arms up
    "dodge":       {"clip": "CMU/127_13", "key": "wide_step", "kind": "oneshot"},
    "sit":         {"clip": "CMU/143_18", "key": "min_height", "kind": "oneshot"},
    "sit_idle":    {"clip": "CMU/143_18", "key": "min_height", "kind": "oneshot", "anchor": "clip", "from": "min_height"},
    "stand_up":    {"clip": "CMU/143_18", "key": "stood_up", "kind": "oneshot", "anchor": "clip", "from": "min_height"},
}


def load_free(sk) -> tuple[ClipLibrary, list[dict]]:
    """All retargeted clips in one library (name -> Clip with category / crouch tags) + the raw 100STYLE list."""
    lib = ClipLibrary(ROOT, skeleton=sk, fps=FPS)
    real = style100.load_retargeted()
    if not real:
        raise SystemExit("100STYLE not built: python3 -I tools/mocap/style100.py fetch && ... retarget")
    trials = cmu.load_retargeted()
    if not trials:
        raise SystemExit("CMU not built: python3 -I tools/mocap/cmu.py fetch && ... retarget")
    for c in real:
        if c["style"] == "Neutral":
            lib.add_clip(c["name"], Clip(sk, c["local"], FPS, c["name"]), {"category": c["category"], "source": "100STYLE"})
    for c in trials:
        local, crouch = c["local"], c["category"] == "cmu_crouch"
        if crouch:  # longest run of crouched frames
            low = local[:, 0, 1] < CROUCH_H
            best, start = (0, 0), None
            for i, v in enumerate(np.append(low, False)):
                if v and start is None:
                    start = i
                elif not v and start is not None:
                    best = max(best, (i - start, start)); start = None
            local = local[best[1]:best[1] + best[0]]
        lib.add_clip(c["name"], Clip(sk, local, FPS, c["name"]), {"category": c["category"], "source": "CMU", "crouch": crouch})
    return lib, real


def key_frame(sk, clip, key) -> int:
    local = clip.local
    last = clip.num_frames - C.POSES_PER_TOKEN
    if isinstance(key, dict):
        return int(np.clip(key["frame"], 0, last))
    if isinstance(key, (int, float)):
        return int(np.clip(float(key) * last, 0, last))
    h = local[:last + 1, 0, 1]
    if key == "min_height":
        return int(np.argmin(h))
    if key == "max_height":
        return int(np.argmax(h))
    if key == "end":
        return last
    w = sk.fk(local)
    if key == "wide_step":  # widest sideways spread of the feet relative to the body heading (a side step)
        from somart.clips import heading_of
        yaw = heading_of(w, sk)
        d = w[:, sk.index("LeftFoot"), [0, 2]] - w[:, sk.index("RightFoot"), [0, 2]]
        lat = np.abs(d[:, 0] * np.cos(yaw) - d[:, 1] * np.sin(yaw))
        return int(np.argmax(lat[:last + 1]))
    if key == "stood_up":  # first frame after the lowest pelvis where the body is upright again
        k0, stand = int(np.argmin(h)), np.percentile(h, 90)
        up = np.flatnonzero(h[k0:] >= 0.97 * stand)
        return k0 + int(up[0]) if len(up) else last
    if key == "crouch_still":  # slowest crouched frame with the feet close together (both planted)
        lf, rf = sk.index("LeftFoot"), sk.index("RightFoot")
        sp = np.linalg.norm(np.gradient(w[:, 0, [0, 2]], axis=0), axis=1) * FPS
        sep = np.linalg.norm(w[:, lf, [0, 2]] - w[:, rf, [0, 2]], axis=1)
        return int(np.argmin((sp + 2 * sep)[:last + 1]))
    return last // 2


def ground_yaw(sk, local):
    from somart.controller import CharacterController
    return CharacterController._ground_and_yaw(type("S", (), {"sk": sk})(), local)


def build_windows(lib, sk, lim):
    # sparse: the phase-aware clips (clips.bin) are the main target source; windows are the fallback and the actions.
    # 100STYLE clips are long (one stride every ~35 frames), the CMU ones short and varied.
    wins = []
    for src, stride in (("100STYLE", 40), ("CMU", 12)):
        lib.index_windows([n for n, c in lib.clips.items() if c.tags["category"] in LOCO and c.tags["source"] == src], stride=stride)
        wins += lib.windows
    meta = [{"v": [round(float(w.velocity[0]), 4), round(float(w.velocity[1]), 4)], "yr": round(float(w.yaw_rate), 4),
             "h": round(float(w.height), 4), "cat": w.tags.get("category", ""), "kind": "loco",
             "crouch": bool(w.tags.get("crouch", False))} for w in wins]
    actions = {}
    for name, a in ACTIONS.items():
        clip = lib.clips[a["clip"]]
        k = key_frame(sk, clip, a["key"])
        w = lib.window_at(a["clip"], k)
        if a.get("static"):
            w.local = np.repeat(w.local[:1], C.POSES_PER_TOKEN, axis=0)
        f0 = key_frame(sk, clip, a.get("from", {"frame": 0}))
        g0, gk = ground_yaw(sk, clip.local[f0]), ground_yaw(sk, clip.local[k])
        actions[name] = {"window": len(wins), "anchor": a.get("anchor", ""), "kind": a["kind"], "source": a["clip"], "frame": k,
                         "root0": [float(g0[0][0]), float(g0[0][2]), g0[1]], "rootk": [float(gk[0][0]), float(gk[0][2]), gk[1]]}
        wins.append(w)
        meta.append({"v": [0, 0], "yr": 0, "h": round(float(w.height), 4), "cat": name, "kind": "action",
                     "crouch": name == "crouch_idle"})
    local = np.stack([lim.apply(w.local, relax_unmapped=True) for w in wins])     # [W, 4, J, 7]
    root = local[:, :, 0, :3].astype(np.float32)
    q = local[..., 3:]
    q = np.where(q[..., 3:4] < 0, -q, q)
    qi = np.round(q * 32767).astype(np.int16)
    with open(OUT / "windows.bin", "wb") as f:
        f.write(root.tobytes())
        f.write(qi.tobytes())
    (OUT / "windows.json").write_text(json.dumps({
        "count": len(wins), "frames": 4, "joints": sk.num_joints, "layout": "float32 root[W,4,3] then int16 quat[W,4,J,4] (/32767)",
        "windows": meta, "actions": actions, "license": f"{style100.LICENSE}; {cmu.LICENSE}"}))
    print(f"{len(wins)} windows ({(OUT / 'windows.bin').stat().st_size / 1e6:.2f} MB), {len(actions)} actions")


def blob_of(lim, joints, local, offset, entry):
    loc = lim.apply(local, relax_unmapped=True)
    q = loc[:, joints, 3:]
    q = np.where(q[..., 3:4] < 0, -q, q)
    blob = loc[:, 0, :3].astype(np.float32).tobytes() + np.round(q * 32767).astype(np.int16).tobytes()
    return blob, dict(entry, frames=int(len(loc)), offset=offset, mirrored=False)


def export_clips(lib, sk, lim, real) -> None:
    """clips.bin: per clip, float32 root position [F, 3] then int16 quaternions [F, Jm, 4] (/32767) for the Jm joints
    the network predicts; every other joint takes the relaxed pose from joint_limits.json."""
    joints = [sk.index(n) for n in C.JOINT_NAMES]
    entries, blobs, off = [], [], 0
    for name, clip in lib.clips.items():
        t = clip.tags
        if t["category"] not in LOCO:
            continue
        local = clip.local[:MAX_FRAMES] if t["source"] == "100STYLE" else clip.local
        run = t["category"] in RUN_CATS  # jogs and runs: never a walking target, whatever their stride period
        blob, e = blob_of(lim, joints, local, off, {"name": name, "category": t["category"], "source": t["source"],
                                                     **({"crouch": True} if t.get("crouch") else {}), **({"gait": "run"} if run else {})})
        entries.append(e); blobs.append(blob); off += len(blob)
    # optional walk styles: one file each, fetched by the page only when the style is chosen.
    # pace = the style's typical forward-walk speed (75th percentile), used as its walking speed.
    styles = []
    (OUT / "styles").mkdir(exist_ok=True)
    for label, st in style100.WALK_STYLES.items():
        cs = [dict(c, local=c["local"][:3600]) for c in real if c["style"] == st]  # at most 2 minutes per kind
        if not cs:
            continue
        s_entries, s_blobs, s_off = [], [], 0
        for c in cs:
            blob, e = blob_of(lim, joints, c["local"], s_off, {"name": c["name"], "category": c["category"], "source": "100STYLE"})
            s_entries.append(e); s_blobs.append(blob); s_off += len(blob)
        (OUT / "styles" / f"{st}.bin").write_bytes(b"".join(s_blobs))
        (OUT / "styles" / f"{st}.json").write_text(json.dumps({
            "fps": FPS, "joints": C.JOINT_NAMES, "clips": s_entries, "style": st,
            "layout": "per clip: float32 root[F,3] then int16 quat[F,Jm,4] (/32767); other joints: relaxed pose",
            "license": style100.LICENSE}))
        fw = next(c["local"] for c in cs if c["kind"] == "FW")
        sp = np.linalg.norm(np.gradient(fw[:, 0, [0, 2]], axis=0), axis=1) * FPS
        pace = round(float(np.percentile(sp[sp > 0.3], 75)) * 20) / 20
        styles.append({"key": st, "label": label, "file": f"data/styles/{st}", "pace": pace,
                       "frames": sum(e["frames"] for e in s_entries), "bytes": s_off})
        print(f"style {label} ({st}): {styles[-1]['frames']} frames, {s_off / 1e6:.2f} MB, pace {pace} m/s")
    with open(OUT / "clips.bin", "wb") as f:
        for b in blobs:
            f.write(b)
    (OUT / "clips.json").write_text(json.dumps({
        "fps": FPS, "joints": C.JOINT_NAMES, "clips": entries,
        "layout": "per clip: float32 root[F,3] then int16 quat[F,Jm,4] (/32767); other joints: relaxed pose",
        "license": f"clips named 100STYLE/*: {style100.LICENSE}; clips named CMU/*: {cmu.LICENSE}", "styles": styles}))
    print(f"{len(entries)} clips, {sum(e['frames'] for e in entries)} frames ({off / 1e6:.2f} MB) -> clips.bin")


HINGES = {f"{s}{j}" for s in ("Left", "Right") for j in ("Shin", "ForeArm", "ToeBase")}  # knees, elbows, toes
# normal adult range of motion (clinical reference values): the flexion side of a hinge reaches at least this far,
# so deep crouches, landings and sitting that the free clips do not contain are not clipped
ROM_DEG = {"Shin": 145.0, "ForeArm": 150.0}


# Relaxed hand: extra flexion (deg) per finger joint 1..4 on top of the SOMA bind pose, whose fingers are nearly
# straight (retargeted clips carry no finger motion). Authored to look like a loose, resting hand; the page's finger
# layer (FingerAnimator) adds speed-dependent curl and sway on top.
CURL = {"Thumb": [0, 14, 10], "Index": [0, 14, 12, 14], "Middle": [0, 16, 14, 14], "Ring": [0, 18, 14, 14], "Pinky": [0, 18, 14, 12]}


def relaxed_hands(sk, relaxed: dict) -> dict:
    """Curl the fingers of the relaxed pose toward the palm (same axis construction as FingerAnimator in controller.js)."""
    loc = sk.reference.copy()
    for n, q in relaxed.items():
        loc[sk.index(n), 3:] = q
    w = sk.fk(loc[None])[0]
    out = dict(relaxed)
    for side in ("Left", "Right"):
        h, m1, mt = (sk.index(f"{side}{n}") for n in ("Hand", "HandMiddle1", "HandMiddleEnd"))
        base = w[m1, :3] - w[h, :3]; base /= np.linalg.norm(base)
        tip = w[mt, :3] - w[m1, :3]
        palm = tip - base * (tip @ base); palm /= np.linalg.norm(palm)
        for f, angles in CURL.items():
            for k, deg in enumerate(angles, 1):
                c = sk.index(f"{side}Hand{f}{k}")
                kid = int(np.flatnonzero(sk.parents == c)[0]) if c >= 0 and (sk.parents == c).any() else -1
                if c < 0 or kid < 0 or not deg:
                    continue
                bdir = w[kid, :3] - w[c, :3]; bdir /= np.linalg.norm(bdir)
                n = palm + 0.6 * base if f == "Thumb" else palm
                ax = np.cross(bdir, n / np.linalg.norm(n)); ax /= np.linalg.norm(ax)
                ax_local = m3.quat_rotate(m3.quat_conj(w[c, 3:]), ax)
                q = m3.quat_mul(np.asarray(out.get(sk.names[c], loc[c, 3:])), m3.quat_from_axis_angle(ax_local, np.radians(deg)))
                out[sk.names[c]] = m3.normalize(q).tolist()
    return out


def fit_limits(lib, sk) -> JointLimits:
    """Anatomical limits from every free clip (locomotion and actions).

    Knees, elbows and toes are hinges about the SOMA network's own axes (configs/skeletons/soma_hinges.json, measured
    by tools/fit_hinges.py on its raw output); a free fit on retargeted rigs would keep their twist (bowed shins,
    twisted forearms) and a fit of one axis across rigs lands between them.
    Model joints the retargeters do not drive (jaw, eyes, finger ends) stay at the SOMA rig's reference rotation;
    Neck2, also undriven, stays free. Everything else uses JointLimits.fit (constrained where the data is low-rank).
    """
    from somart.anatomy import JointLimit
    frames = np.concatenate([c.local for c in lib.clips.values()])
    lim = JointLimits.fit(frames, sk)
    hinges = json.loads(style100.HINGES.read_text())["joints"]
    driven = set(style100.MAP) | set(cmu.MAP)
    out = []
    for name in C.JOINT_NAMES[1:]:
        j = sk.index(name)
        if name in HINGES:  # the network's own axis (tools/fit_hinges.py); range covers the clips and the network
            h = hinges[name]
            axis = np.asarray(h["axis"], float)[None]
            r = m3.quat_to_scaled_axis(m3.quat_abs(frames[:, j, 3:]))
            coords = r @ axis.T
            m = np.radians(3.0)
            lo = np.minimum(np.percentile(coords, 0.2, axis=0), h["lo"]) - m
            hi = np.maximum(np.percentile(coords, 99.8, axis=0), h["hi"]) + m
            rom = ROM_DEG.get(name.removeprefix("Left").removeprefix("Right"))
            if rom:  # the flexion side is the one the data reaches furthest on
                if -lo[0] > hi[0]:
                    lo[0] = min(lo[0], -np.radians(rom))
                else:
                    hi[0] = max(hi[0], np.radians(rom))
            out.append(JointLimit(name, 1, axis, lo, hi, m3.quat_from_scaled_axis((coords.mean(0) * axis).sum(0))))
        elif name not in driven and name != "Neck2":
            out.append(JointLimit(name, 0, np.zeros((0, 3)), np.zeros(0), np.zeros(0), sk.reference[j, 3:].copy()))
        else:
            out += [x for x in lim.limits if x.joint == name]
    lim = JointLimits(out, relaxed_hands(sk, lim.relaxed))
    lim.save(ROOT / "configs/skeletons/soma_joint_limits.json")
    (OUT / "joint_limits.json").write_text((ROOT / "configs/skeletons/soma_joint_limits.json").read_text())
    print(f"joint limits: {len(lim.limits)} constrained, {len(lim.relaxed)} relaxed joints from {len(frames)} frames")
    return lim


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sk, _ = style100.soma_skeleton()
    lib, real = load_free(sk)
    lim = fit_limits(lib, sk).bind(sk)
    build_windows(lib, sk, lim)
    export_clips(lib, sk, lim, real)
    return 0


if __name__ == "__main__":
    sys.exit(main())
