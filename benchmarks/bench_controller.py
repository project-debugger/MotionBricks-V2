#!/usr/bin/env python3
"""Game-loop benchmark: a scripted 20 s gameplay session at 60 Hz, real-time paced.

Measures what a player feels, for a given preset and options:
  frame time / FPS     main-thread cost of CharacterController.update (avg, p99, min FPS, stalls)
  responsiveness       delay between an input change and the rendered body reaching 50 % / 90 % of
                       the commanded velocity change (and the same for the gameplay trajectory alone)
  tracking             distance between the rendered hips and the simulated gameplay trajectory
  foot sliding         horizontal foot speed while planted (height-based contacts, 60 Hz)
  smoothness           mean joint jerk, and worst acceleration spike relative to median ("pops")
  generation           count, latency, late results hidden by inertialization

Usage:
  .venv/bin/python benchmarks/bench_controller.py --preset balanced
  .venv/bin/python benchmarks/bench_controller.py --preset balanced --sync --no-foot-lock
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from somart import metrics as M  # noqa: E402
from somart.bvh_writer import write_bvh  # noqa: E402
from somart.clips import ClipLibrary  # noqa: E402
from somart.config import Config  # noqa: E402
from somart.controller import CharacterController, Command  # noqa: E402
from somart.engine import SomaEngine  # noqa: E402
from somart.hardware import resolve_inference  # noqa: E402
from somart.backends import make_backend  # noqa: E402

# (start_s, move_x, move_z, gait, action)
SCRIPT = [
    (0.0, 0.0, 0.0, "run", None),
    (2.0, 0.0, 1.0, "run", None),        # start running forward
    (5.0, 1.0, 0.0, "run", None),        # 90 degree turn
    (7.0, -1.0, 0.0, "run", None),       # 180 degree reversal
    (9.0, -1.0, 0.0, "sprint", None),    # sprint
    (11.0, 0.0, 0.0, "run", None),       # stop
    (12.0, 0.7, 0.7, "walk", None),      # diagonal walk
    (14.0, 0.0, 1.0, "crouch", None),    # crouch move
    (16.0, 0.0, 1.0, "run", "jump"),     # jump while running
    (18.0, 0.0, 0.0, "run", None),       # stop, idle
]
DURATION = 20.0


def command_at(t: float, fired: set) -> Command:
    cur = SCRIPT[0]
    for row in SCRIPT:
        if t >= row[0]:
            cur = row
    action = None
    if cur[4] and cur[0] not in fired:
        fired.add(cur[0])
        action = cur[4]
    return Command(move=(cur[1], cur[2]), gait=cur[3], action=action)


def response_times(t: np.ndarray, vel: np.ndarray, events: list[tuple[float, np.ndarray, np.ndarray]]) -> list[dict]:
    out = []
    for te, v_before, v_goal in events:
        dv = v_goal - v_before
        mag = np.linalg.norm(dv)
        if mag < 0.5:
            continue
        sel = t >= te
        prog = ((vel[sel] - v_before) @ dv) / (mag * mag)
        ts = t[sel] - te
        r = {"event_t": te, "dv": float(mag)}
        for q in (0.5, 0.9):
            idx = np.flatnonzero(prog >= q)
            r[f"t{int(q*100)}"] = float(ts[idx[0]]) if idx.size else None
        out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="balanced")
    ap.add_argument("--sync", action="store_true", help="generate on the main thread (for comparison)")
    ap.add_argument("--no-foot-lock", action="store_true")
    ap.add_argument("--hz", type=float, default=60.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--bvh", default="")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. locomotion.run_speed=4.0")
    args = ap.parse_args()

    cfg = Config.load(args.preset)
    if args.no_foot_lock:
        cfg.set("animation.foot_locking.enabled", False)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg.set(k, json.loads(v))
    inf = resolve_inference(cfg.section("inference"))
    lib = ClipLibrary(ROOT / cfg.get("clips.root")).load()
    loco_sources = [cfg.get(f"actions.{a}.source") for a in ("walk", "strafe", "crouch")] + [
        cfg.get("actions.idle.source"), cfg.get("actions.crouch_idle.source")]
    lib.index_windows(loco_sources)
    engine = SomaEngine(make_backend(inf), lib.skeleton)
    engine.generate(lib.windows[0].local, lib.windows[1].local)  # warm-up
    ctrl = CharacterController(engine, lib, cfg, synchronous=args.sync)

    dt = 1.0 / args.hz
    n = int(DURATION * args.hz)
    fired: set = set()
    poses, frame_ms, sim_pos, sim_vel, goals = [], [], [], [], []
    t_next = time.perf_counter()
    for i in range(n):
        t = i * dt
        cmd = command_at(t, fired)
        s = time.perf_counter()
        pose = ctrl.update(cmd, dt)
        frame_ms.append((time.perf_counter() - s) * 1e3)
        poses.append(pose)
        sim_pos.append(ctrl.sim_pos.copy())
        sim_vel.append(ctrl.sim_vel.copy())
        goals.append(ctrl._goal_velocity(cmd))
        t_next += dt
        sleep = t_next - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
        else:
            t_next = time.perf_counter()  # fell behind: do not try to catch up
    ctrl.close()
    poses = np.array(poses)
    sk = lib.skeleton
    world = sk.fk(poses)
    tt = np.arange(n) * dt
    frame_ms = np.array(frame_ms)

    # --- responsiveness: input-change events
    goals = np.array(goals)
    sim_vel = np.array(sim_vel)
    hips = world[:, 0, :3]
    body_vel = np.zeros_like(hips)
    body_vel[1:-1] = (hips[2:] - hips[:-2]) / (2 * dt)
    # smooth body velocity over ~ one step cycle so gait bounce does not count as response
    k = int(0.25 * args.hz)
    kern = np.ones(k) / k
    body_vel_s = np.stack([np.convolve(body_vel[:, c], kern, mode="same") for c in range(3)], -1)
    events = []
    for i in range(1, n):
        if np.linalg.norm(goals[i] - goals[i - 1]) > 0.5:
            events.append((tt[i], body_vel_s[i - 1].copy(), goals[i]))
    # same filter on the gameplay velocity, so both carry the identical filter delay
    sim_vel_s = np.stack([np.convolve(sim_vel[:, c], kern, mode="same") for c in range(3)], -1)
    resp_body = response_times(tt, body_vel_s, events)
    resp_sim = response_times(tt, sim_vel_s, [(e[0], sim_vel_s[int(e[0] * args.hz) - 1], e[2]) for e in events])

    foot_idx = [sk.index(nm) for nm in M.FOOT_JOINTS]
    contacts = M.foot_contacts_from_height(world, foot_idx, fps=args.hz)
    accel = M.accel_profile(world, args.hz)
    track = np.linalg.norm((hips - np.array(sim_pos))[:, [0, 2]], axis=-1)
    budget = 1000.0 / args.hz
    lat = np.array(ctrl.stats.latency_ms) if ctrl.stats.latency_ms else np.zeros(1)
    res = {
        "preset": args.preset, "mode": "sync" if args.sync else "async",
        "foot_locking": not args.no_foot_lock, "model": inf["model"], "threads": inf["threads"], "hz": args.hz,
        "frame_ms": M.summarize(frame_ms.tolist()),
        "fps": {"avg_capacity": float(1000.0 / frame_ms.mean()), "min_capacity_p99": float(1000.0 / np.percentile(frame_ms, 99)),
                "worst_frame_fps": float(1000.0 / frame_ms.max()),
                "frames_over_budget": int((frame_ms > budget).sum()), "budget_ms": budget,
                "frame_time_std_ms": float(frame_ms.std())},
        "responsiveness": {"body": resp_body, "gameplay": resp_sim,
                           "body_t90_mean_s": float(np.mean([r["t90"] for r in resp_body if r["t90"] is not None] or [np.nan])),
                           "gameplay_t90_mean_s": float(np.mean([r["t90"] for r in resp_sim if r["t90"] is not None] or [np.nan]))},
        "tracking_error_m": M.summarize(track.tolist()),
        "foot_slide_cm_s": M.foot_slide(world, contacts, foot_idx, fps=args.hz) * 100,
        "jerk": M.jerk(world, args.hz),
        "pop_ratio_max": float(accel.max() / (np.median(accel) + 1e-6)),
        "pop_ratio_p99": float(np.percentile(accel, 99) / (np.median(accel) + 1e-6)),
        "generation": {"count": ctrl.stats.generations, "late": ctrl.stats.late_results,
                       "input_replans": ctrl.stats.replans_from_input,
                       "latency_ms": M.summarize(lat.tolist()),
                       "per_second": ctrl.stats.generations / DURATION},
    }
    if args.bvh:
        # 60 Hz -> write every 2nd frame at 30 Hz
        write_bvh(args.bvh, sk, poses[:: max(1, int(round(args.hz / 30)))], fps=30.0)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
    f = res["fps"]
    print(f"[controller] {args.preset} {res['mode']} footlock={res['foot_locking']}: frame p50 "
          f"{res['frame_ms']['p50']:.2f} ms p99 {res['frame_ms']['p99']:.2f} max {res['frame_ms']['max']:.1f} | "
          f"over budget {f['frames_over_budget']} | body t90 {res['responsiveness']['body_t90_mean_s']:.2f}s "
          f"(gameplay {res['responsiveness']['gameplay_t90_mean_s']:.2f}s) | track {res['tracking_error_m']['mean']:.2f} m | "
          f"slide {res['foot_slide_cm_s']:.1f} cm/s | gens {ctrl.stats.generations} late {ctrl.stats.late_results}",
          flush=True)
    engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
