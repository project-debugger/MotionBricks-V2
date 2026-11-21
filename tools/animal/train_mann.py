#!/usr/bin/env python3
"""Train a quadruped motion network on dog motion capture (Mode-Adaptive Neural Network style).

The SOMA model released by NVIDIA is human-only and ships without training code, so the playground's
animals get their own network: a gating network blends K expert MLPs per frame (Zhang et al. 2018,
"Mode-Adaptive Neural Networks for Quadruped Motion Control"), trained here from scratch on the
authors' dog motion capture (CC BY-NC 4.0). It is autoregressive: given the current pose and a 2 s
root trajectory (1 s past from history, 1 s future from the player's input) it predicts the next frame,
the root motion, the future trajectory and paw contacts. Small enough to run per frame in plain JS.

    .venv/bin/python tools/animal/train_mann.py --data <folder with the dog .bvh files> --out site/data/animal
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dogdata import load_all  # noqa: E402

FPS = 60
OFFS = list(range(-60, 61, 10))           # 13 trajectory samples, +-1 s
FUT = [o for o in OFFS if o > 0]          # predicted future samples
LABELS = ["lie", "jump", "rear"]
PAWS = ["LeftHandEnd", "RightHandEnd", "LeftFootEnd", "RightFootEnd"]


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(pad, k, mode="valid")


def rot_inv(yaw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world (x, z) vectors into the root frame whose facing is (sin yaw, cos yaw) -> +Z."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, z = v[..., 0], v[..., 1]
    return np.stack([c * x - s * z, s * x + c * z], -1)


def clip_features(pos: np.ndarray, ix: dict, fwd_sign: float):
    """Per-frame root (position, yaw), labels and contacts for one clip of global positions [F, P, 3]."""
    F = len(pos)
    across = (pos[:, ix["LeftUpLeg"]] - pos[:, ix["RightUpLeg"]]) + (pos[:, ix["LeftShoulder"]] - pos[:, ix["RightShoulder"]])
    across[:, 1] = 0
    fwd = fwd_sign * np.cross(np.array([0.0, 1.0, 0.0]), across)
    fwd = gaussian(fwd[:, [0, 2]], 4)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    yaw = np.arctan2(fwd[:, 0], fwd[:, 1])
    center = 0.5 * (pos[:, ix["Hips"]] + pos[:, ix["Spine1"]])
    root = gaussian(center[:, [0, 2]], 2)
    # labels: lying (low posture) and jumping (short ballistic flights), softened over time
    hip, chest = pos[:, ix["Hips"], 1] / 0.469, pos[:, ix["Spine1"], 1] / 0.437
    lie = ((hip < 0.5) & (chest < 0.75)).astype(np.float64)
    paws = pos[:, [ix[p] for p in PAWS], 1]
    air = paws.min(1) > 0.06
    jump = np.zeros(F)
    i = 0
    while i < F:
        if air[i]:
            j = i
            while j < F and air[j]:
                j += 1
            hv = np.linalg.norm(np.diff(pos[i:j + 1, ix["Hips"], [0, 2]], axis=0), axis=1).mean() * FPS if j > i else 0
            if 4 <= j - i <= 60 and hv > 1.2:  # forward leaps only
                jump[max(0, i - 12):min(F, j + 8)] = 1
            i = j
        else:
            i += 1
    # rearing: standing on the hind legs (spine pitched up, paws on the ground) - 4.4 min of the data
    pitch = np.degrees(np.arctan2(pos[:, ix["Spine1"], 1] - pos[:, ix["Hips"], 1], np.linalg.norm((pos[:, ix["Spine1"]] - pos[:, ix["Hips"]])[:, [0, 2]], axis=1)))
    rear = ((pitch > 35) & ~air).astype(np.float64)
    labels = np.clip(gaussian(np.stack([lie, jump, rear], 1), 8), 0, 1)
    pv = np.linalg.norm(np.diff(pos[:, [ix[p] for p in PAWS]], axis=0, prepend=pos[:1, [ix[p] for p in PAWS]]), axis=-1) * FPS
    contacts = ((paws < 0.05) & (pv < 0.45)).astype(np.float64)
    elevated = np.zeros(F, bool)  # long stretches with every paw off the ground (dog on a raised surface)
    i = 0
    while i < F:
        if air[i]:
            j = i
            while j < F and air[j]:
                j += 1
            if j - i > 60:
                elevated[i:j] = True
            i = j
        else:
            i += 1
    return root, yaw, labels, contacts, elevated


def to_root(yaw: np.ndarray, v3: np.ndarray) -> np.ndarray:
    """World 3-vectors [..., 3] -> root frame (x, y, z) for per-row yaw (broadcast over the last-but-one axis)."""
    xz = rot_inv(yaw, v3[..., [0, 2]])
    return np.stack([xz[..., 0], v3[..., 1], xz[..., 1]], -1)


TS = 6 + len(LABELS)  # floats per trajectory sample: pos2, dir2, vel2, labels


def build(clips, ix, swap, mirror: bool):
    X, Y, G, keep = [], [], [], []
    names = clips[0][1].names
    paw_ix = [names.index(p) for p in PAWS]
    # facing sign: forward must point from the hips toward the chest
    p0 = np.concatenate([p for _, _, p in clips])
    a = (p0[:, ix["LeftUpLeg"]] - p0[:, ix["RightUpLeg"]]); a[:, 1] = 0
    sgn = 1.0 if np.mean(np.sum(np.cross([0.0, 1.0, 0.0], a) * (p0[:, ix["Spine1"]] - p0[:, ix["Hips"]]), axis=1)) > 0 else -1.0
    for cname, _, pos in clips:
        for m in ([False, True] if mirror else [False]):
            P = pos.copy()
            if m:
                P = P[:, swap] * np.array([-1.0, 1.0, 1.0])
            root, yaw, labels, contacts, elevated = clip_features(P, ix, sgn)
            F = len(P)
            t = np.arange(1, F - 1)
            t = t[~(elevated[t] | elevated[t + 1])]
            if len(t) == 0:
                continue
            rvel = np.gradient(root, axis=0) * FPS
            fwd = np.stack([np.sin(yaw), np.cos(yaw)], 1)
            y0 = yaw[t][:, None]
            def traj(tt, yy, offs, with_labels=True):
                cols = []
                for o in offs:
                    k = np.clip(tt + o, 0, F - 1)
                    cols += [rot_inv(yy, (root[k] - root[tt])[:, None, :])[:, 0], rot_inv(yy, fwd[k][:, None, :])[:, 0], rot_inv(yy, rvel[k][:, None, :])[:, 0]]
                    if with_labels:
                        cols.append(labels[k])
                return np.concatenate(cols, 1)
            r3 = lambda tt: np.stack([root[tt][:, 0], np.zeros(len(tt)), root[tt][:, 1]], 1)[:, None, :]
            tr = traj(t, y0, OFFS)
            lp = to_root(y0, P[t] - r3(t)); lv = to_root(y0, (P[t] - P[t - 1]) * FPS)
            x = np.concatenate([tr, lp.reshape(len(t), -1), lv.reshape(len(t), -1)], 1)
            t1 = t + 1; y1 = yaw[t1][:, None]
            lp1 = to_root(y1, P[t1] - r3(t1)); lv1 = to_root(y1, (P[t1] - P[t]) * FPS)
            dr = rot_inv(y0, (root[t1] - root[t])[:, None, :])[:, 0] * FPS
            dyaw = np.arctan2(np.sin(yaw[t1] - yaw[t]), np.cos(yaw[t1] - yaw[t]))[:, None] * FPS
            fut = traj(t1, y1, FUT, with_labels=False)
            y = np.concatenate([lp1.reshape(len(t), -1), lv1.reshape(len(t), -1), dr, dyaw, fut, contacts[t1]], 1)
            i30, i60 = OFFS.index(30) * TS, OFFS.index(60) * TS
            g = np.concatenate([lv[:, paw_ix].reshape(len(t), -1), rot_inv(y0, rvel[t][:, None, :])[:, 0], tr[:, i30 + 4:i30 + 6], tr[:, i60 + 4:i60 + 6],
                                labels[t], labels[np.clip(t + 30, 0, F - 1)]], 1)
            X.append(x); Y.append(y); G.append(g); keep += [cname] * len(t)
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32), np.concatenate(G).astype(np.float32), np.array(keep)


class MANN(nn.Module):
    def __init__(self, gin, xin, yout, K=4, H=256, GH=64, drop=0.2):
        super().__init__()
        self.K, self.drop = K, drop
        self.gate = nn.Sequential(nn.Linear(gin, GH), nn.ELU(), nn.Linear(GH, GH), nn.ELU(), nn.Linear(GH, K))
        def w(i, o):
            return nn.Parameter(torch.randn(K, i, o) * math.sqrt(2.0 / (i + o)))
        self.W = nn.ParameterList([w(xin, H), w(H, H), w(H, yout)])
        self.B = nn.ParameterList([nn.Parameter(torch.zeros(K, H)), nn.Parameter(torch.zeros(K, H)), nn.Parameter(torch.zeros(K, yout))])

    def forward(self, g, x):
        a = torch.softmax(self.gate(g), -1)                                  # [B, K] expert blend
        h = nn.functional.dropout(x, self.drop, self.training)
        for li in range(3):
            # blending the weights == blending each expert's (linear) output
            out = torch.einsum("bi,kio->bko", h, self.W[li]) + self.B[li][None]
            h = torch.einsum("bk,bko->bo", a, out)
            if li < 2:
                h = nn.functional.dropout(nn.functional.elu(h), self.drop, self.training)
        return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=40.0, help="wall-clock training budget")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0); np.random.seed(0)
    t0 = time.time()
    clips = load_all(a.data)
    names = clips[0][1].names; ix = {n: i for i, n in enumerate(names)}
    swap = [names.index(n.replace("Left", "#").replace("Right", "Left").replace("#", "Right")) for n in names]
    val_names = {c[0] for c in clips[::12]}                     # every 12th clip held out
    train = [c for c in clips if c[0] not in val_names]; val = [c for c in clips if c[0] in val_names]
    X, Y, G, _ = build(train, ix, swap, mirror=True)
    Xv, Yv, Gv, _ = build(val, ix, swap, mirror=False)
    print(f"data: {len(X)} train / {len(Xv)} val samples, x {X.shape[1]}, y {Y.shape[1]}, g {G.shape[1]} ({time.time() - t0:.0f} s)", flush=True)
    stats = {}
    for k, A in (("x", X), ("y", Y), ("g", G)):
        mu, sd = A.mean(0), A.std(0)
        sd = np.maximum(sd, 1e-2 * max(1e-6, sd.max()) if k != "g" else 1e-3); sd = np.maximum(sd, 1e-4)
        stats[k] = (mu, sd)
    nz = lambda A, k: (A - stats[k][0]) / stats[k][1]
    Xn, Yn, Gn = map(torch.from_numpy, (nz(X, "x"), nz(Y, "y"), nz(G, "g")))
    Xvn, Yvn, Gvn = map(torch.from_numpy, (nz(Xv, "x"), nz(Yv, "y"), nz(Gv, "g")))
    model = MANN(G.shape[1], X.shape[1], Y.shape[1], K=a.experts, H=a.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xn); steps_per_epoch = math.ceil(n / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=a.epochs * steps_per_epoch, pct_start=0.08)
    hist = []
    budget = a.minutes * 60
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(n); tl = 0.0; te = time.time()
        for s in range(steps_per_epoch):
            b = perm[s * a.batch:(s + 1) * a.batch]
            loss = nn.functional.mse_loss(model(Gn[b], Xn[b]), Yn[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([nn.functional.mse_loss(model(Gvn[i:i + 4096], Xvn[i:i + 4096]), Yvn[i:i + 4096]).item() for i in range(0, len(Xvn), 4096)]))
        hist.append({"epoch": ep + 1, "train": tl / n, "val": vl, "seconds": time.time() - te})
        print(f"epoch {ep + 1}/{a.epochs} train {tl / n:.4f} val {vl:.4f} ({time.time() - te:.0f} s, total {(time.time() - t0) / 60:.1f} min)", flush=True)
        if time.time() - t0 > budget:
            print("time budget reached", flush=True); break
    export(model, stats, names, clips[0][1], a, hist, len(X), len(Xv), sorted(val_names), standing_pose(X, ix))
    return 0


def standing_pose(X: np.ndarray, ix: dict) -> list:
    """Most typical still, standing frame (root-frame joint positions) to start the character from."""
    off = len(OFFS) * TS
    P = X[:, off:off + 27 * 3].reshape(-1, 27, 3); V = X[:, off + 81:].reshape(-1, 27, 3)
    sp = np.linalg.norm(X[:, OFFS.index(0) * TS + 4:OFFS.index(0) * TS + 6], axis=1)
    cand = np.where((np.abs(P[:, ix["Hips"], 1] - 0.469) < 0.02) & (sp < 0.03) & (np.abs(V).max(axis=(1, 2)) < 0.15))[0]
    if not len(cand):
        return P.mean(0).ravel().tolist()
    med = np.median(P[cand], axis=0)
    best = cand[np.argmin(np.linalg.norm((P[cand] - med).reshape(len(cand), -1), axis=1))]
    return [round(float(v), 5) for v in P[best].ravel()]


def export(model, stats, names, bvh, a, hist, ntrain, nval, val_names, start_pose=None):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    parts, meta_t, off = [], [], 0
    def add(name, arr):
        nonlocal off
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        parts.append(arr.astype(np.float16).tobytes()); meta_t.append({"name": name, "shape": list(arr.shape), "offset": off}); off += arr.size * 2
    sd = model.state_dict()
    for k, v in sd.items():
        add(k, v.numpy())
    for k in ("x", "y", "g"):
        add(f"{k}_mean", stats[k][0]); add(f"{k}_std", stats[k][1])
    (out / "dognet.bin").write_bytes(b"".join(parts))
    rest = (np.linalg.norm(bvh.offsets, axis=1) * 0.01).tolist()
    meta = {"tensors": meta_t, "dtype": "float16", "experts": a.experts, "hidden": a.hidden, "fps": FPS, "offsets": OFFS, "future": FUT,
            "labels": LABELS, "points": names, "parents": bvh.parents, "rest_lengths": rest, "paws": PAWS,
            "layout": {"x": "13 traj samples x (pos2, dir2, vel2, labels), then positions P x3, velocities P x3 (root frame)",
                       "y": "positions P x3, velocities P x3, root velocity x,z (m/s), yaw rate (rad/s), 6 future samples x (pos2, dir2, vel2), 4 contacts",
                       "g": "paw velocities 4x3, root velocity 2, trajectory velocity at +0.5 s and +1 s (2+2), labels now and +0.5 s"},
            "start_pose": start_pose,
            "training": {"samples": ntrain, "val_samples": nval, "val_clips": val_names, "history": hist, "mirrored": True},
            "data": "MANN dog motion capture (Zhang, Starke, Komura, Saito, SIGGRAPH 2018), CC BY-NC 4.0",
            "license": "Weights derived from CC BY-NC 4.0 data: non-commercial use only, attribute the dataset authors"}
    (out / "dognet.json").write_text(json.dumps(meta))
    print(f"exported {off / 1e6:.2f} MB -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
