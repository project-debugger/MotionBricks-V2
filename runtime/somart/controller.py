"""Real-time character controller on top of the SOMA engine.

Design (what makes it feel responsive rather than "laggy AI animation"):

* **Simulated trajectory** - desired velocity and facing follow critically-damped springs with
  configurable half-lives (responsiveness, inertia), acceleration/deceleration caps and
  momentum. This is the gameplay-side motion the animation must follow.
* **Receding-horizon generation** - every `replan_interval` frames (or immediately when the input
  changes by more than `input_replan_threshold`) a new segment is generated from the motion
  that will be on screen when the result arrives (latency-compensated) towards target poses
  placed where the simulated trajectory will be `prediction_horizon` seconds later.
* **Asynchronous** - generation runs on a worker thread; the game loop only samples poses.
* **Seamless splicing** - generated segments start exactly at the context pose (endpoint
  correction); any residual mismatch (late result, user override) is hidden by inertialization.
* **Foot locking** - the model's own foot-contact prediction drives two-bone IK pinning.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from . import contract as C
from . import math3d as m3
from .clips import ClipLibrary, place
from .config import Config
from .engine import GenerationRequest, SomaEngine
from .postprocess import FootLocker, Inertializer, Smoother, scale_intensity

FPS = C.SAMPLE_RATE
DT = 1.0 / FPS


def spring_velocity(v: np.ndarray, a: np.ndarray, v_goal: np.ndarray, halflife: float, dt: float):
    """Critically damped velocity spring (Holden 'Spring-It-On' simple_spring_damper_exact)."""
    y = (4.0 * math.log(2.0) / max(halflife, 1e-5)) / 2.0
    j0 = v - v_goal
    j1 = a + j0 * y
    e = math.exp(-y * dt)
    return e * (j0 + j1 * dt) + v_goal, e * (a - j1 * y * dt)


@dataclass
class Command:
    """What gameplay asks for this frame."""

    move: tuple[float, float] = (0.0, 0.0)  # desired world-space direction (x, z), magnitude 0..1
    gait: str = "run"                        # walk | run | sprint | crouch
    facing: tuple[float, float] | None = None  # optional explicit facing direction (strafing)
    action: str | None = None                # one-shot action name from config `actions`


@dataclass
class ControllerStats:
    generations: int = 0
    late_results: int = 0
    replans_from_input: int = 0
    failures: int = 0
    last_error: str = ""
    latency_ms: list = field(default_factory=list)


class CharacterController:
    def __init__(self, engine: SomaEngine, library: ClipLibrary, config: Config, *, name: str = "character",
                 start_position=(0.0, 0.0, 0.0), start_yaw: float = 0.0, synchronous: bool | None = None) -> None:
        self.engine = engine
        self.lib = library
        self.base_cfg = config
        self.cfg = config.for_character(name) if name in (config.get("characters") or {}) else config
        self.name = name
        self.sk = library.skeleton
        self.sync = (not self.cfg.get("inference.async")) if synchronous is None else synchronous
        self.rng = np.random.default_rng(int(self.cfg.get("animation.seed")))
        self._sec_cache: dict = {}
        self.stats = ControllerStats()
        self._latency_est = 0.05

        # simulated gameplay trajectory
        self.sim_pos = np.array(start_position, dtype=np.float64)
        self.sim_vel = np.zeros(3)
        self.sim_acc = np.zeros(3)
        self.sim_yaw = float(start_yaw)
        self.sim_yaw_vel = 0.0

        # motion buffer: 30 Hz local poses with world root, plus contacts
        idle = self._idle_window()
        start = place(idle.local, self.sim_pos, self.sim_yaw)
        self.buffer = np.concatenate([start, np.repeat(start[-1:], 2 * C.POSES_PER_TOKEN, axis=0)], axis=0)
        self.contacts = np.ones((len(self.buffer), 4))
        self.time = (C.POSES_PER_TOKEN - 1) * DT  # playhead in seconds into the buffer
        self.pending: Future | None = None
        self.pending_info: dict | None = None
        self.frames_since_plan = 10**6
        self.last_goal_vel = np.zeros(3)
        self.action_queue: list[str] = []
        self.current_action: str | None = None
        self.commit_end = -1.0

        n = self.sk.num_joints
        self.inertializer = Inertializer(n, self._inert_halflife())
        sk_cfg = self.cfg.skeleton_settings()
        foot_names = sk_cfg.get("foot_joints", list(C.FOOT_CONTACT_JOINTS))
        chains = sk_cfg.get("leg_chains", [["LeftLeg", "LeftShin", "LeftFoot"], ["RightLeg", "RightShin", "RightFoot"]])
        contact_idx = [foot_names.index(ch[2]) if ch[2] in foot_names else i * 2 for i, ch in enumerate(chains)]
        fl = self.cfg.section("animation").get("foot_locking", {})
        self.feet = FootLocker(self.sk, chains, contact_idx, contact_threshold=fl.get("contact_threshold", 0.55),
                               release_threshold=fl.get("release_threshold", 0.35),
                               max_correction=fl.get("max_correction", 0.12), blend_in=fl.get("blend_in", 0.06),
                               blend_out=fl.get("blend_out", 0.1), ground=self.cfg.get("animation.ground_height"))
        self.smoother = Smoother(float(self.cfg.get("animation.smoothing")))
        self.last_display: np.ndarray | None = None
        self.prev_display: np.ndarray | None = None
        config.on_change(self._on_config_change)

    # ------------------------------------------------------------------ config
    def _sec(self, name: str) -> dict:
        """Cached config section (re-read only after a runtime change)."""
        key = (name, self.cfg.version)
        if key not in self._sec_cache:
            self._sec_cache = {k: v for k, v in self._sec_cache.items() if k[1] == self.cfg.version}
            self._sec_cache[key] = self.cfg.section(name)
        return self._sec_cache[key]

    def _inert_halflife(self) -> float:
        return max(float(self.cfg.get("animation.transition_duration")) / 2.5, 1e-3)

    def _on_config_change(self, path: str, value) -> None:
        # live parameter changes: refresh the derived per-character config
        self.cfg = self.base_cfg.for_character(self.name) if self.name in (self.base_cfg.get("characters") or {}) \
            else self.base_cfg
        self._sec_cache = {}
        self.inertializer.halflife = self._inert_halflife()
        self.smoother.amount = float(self.cfg.get("animation.smoothing"))
        fl = self.cfg.section("animation").get("foot_locking", {})
        self.feet.on, self.feet.off = fl.get("contact_threshold", 0.55), fl.get("release_threshold", 0.35)
        self.feet.max_corr = fl.get("max_correction", 0.12)

    def _idle_window(self):
        src = self.cfg.get("actions.idle.source")
        names = self.lib.find(src) if src else []
        return self.lib.window_at(names[0] if names else next(iter(self.lib.clips)), 30)

    # ------------------------------------------------------------- simulation
    def _goal_velocity(self, cmd: Command) -> np.ndarray:
        loco = self._sec("locomotion")
        speed = {"walk": loco["walk_speed"], "run": loco["run_speed"], "sprint": loco["sprint_speed"],
                 "crouch": loco["crouch_speed"]}.get(cmd.gait, loco["run_speed"])
        mv = np.array([cmd.move[0], 0.0, cmd.move[1]], dtype=np.float64)
        mag = min(1.0, float(np.linalg.norm(mv)))
        return (mv / max(np.linalg.norm(mv), 1e-9)) * speed * mag if mag > 1e-3 else np.zeros(3)

    def _simulate(self, goal: np.ndarray, cmd: Command, dt: float) -> None:
        loco = self._sec("locomotion")
        hl = float(loco["responsiveness"]) * float(loco["inertia"])
        # momentum: when reversing, keep a fraction of the old velocity so turns carry weight
        if np.dot(goal, self.sim_vel) < 0:
            goal = goal + self.sim_vel * float(loco["momentum"])
        v_new, a_new = spring_velocity(self.sim_vel, self.sim_acc, goal, hl, dt)
        # acceleration / deceleration caps
        dv = v_new - self.sim_vel
        speeding_up = np.linalg.norm(v_new) > np.linalg.norm(self.sim_vel)
        cap = (loco["acceleration"] if speeding_up else loco["deceleration"]) * dt
        if np.linalg.norm(dv) > cap:
            dv = dv / np.linalg.norm(dv) * cap
            a_new = dv / max(dt, 1e-6)
        self.sim_vel = self.sim_vel + dv
        self.sim_acc = a_new
        self.sim_pos = self.sim_pos + self.sim_vel * dt
        # facing follows movement (or the explicit facing for strafing)
        face = cmd.facing if (cmd.facing is not None or loco.get("strafe")) else (
            (self.sim_vel[0], self.sim_vel[2]) if np.linalg.norm(self.sim_vel) > 0.3 else None)
        if face is not None and np.hypot(*face) > 1e-6:
            target_yaw = math.atan2(face[0], face[1])
            err = float(m3.wrap_angle(target_yaw - self.sim_yaw))
            hlf = float(loco["facing_responsiveness"]) * float(loco["inertia"])
            y = (4.0 * math.log(2.0) / hlf) / 2.0
            step = err * (1 - math.exp(-y * dt))
            lim = math.radians(float(loco["turn_rate"])) * dt
            self.sim_yaw = float(m3.wrap_angle(self.sim_yaw + float(np.clip(step, -lim, lim))))

    def predict(self, horizon: float, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Future (position, velocity, yaw) of the simulated trajectory after `horizon` seconds."""
        p, v, a, yaw = self.sim_pos.copy(), self.sim_vel.copy(), self.sim_acc.copy(), self.sim_yaw
        loco = self._sec("locomotion")
        hl = float(loco["responsiveness"]) * float(loco["inertia"])
        steps = max(1, int(round(horizon * FPS)))
        for _ in range(steps):
            v, a = spring_velocity(v, a, goal, hl, DT)
            p = p + v * DT
        if np.linalg.norm(goal) > 0.3:
            yaw = math.atan2(goal[0], goal[2])
        return p, v, yaw

    # ------------------------------------------------------------- generation
    def _frame_index(self) -> float:
        return self.time * FPS

    def _plan(self, cmd: Command, goal: np.ndarray) -> None:
        gen = self._sec("generation")
        lead = int(math.ceil(self._latency_est * FPS)) + int(gen["replan_lead"])
        k = int(math.floor(self._frame_index())) + max(1, lead)       # switch frame in the buffer
        k = min(max(k, C.POSES_PER_TOKEN - 1), len(self.buffer) - 1)
        context = self.buffer[k - 3:k + 1]
        if context.shape[0] < 4:
            return
        horizon = float(gen["prediction_horizon"])
        allowed = tuple(self.cfg.allowed_tokens)
        if gen.get("duration_mode", "horizon") == "horizon":
            # Pin the segment length to the horizon and place the target exactly where the
            # gameplay trajectory will be when the segment ends: the body cannot lag behind
            # by stretching the transition (the model alone may choose up to 1.7 s).
            T = int(np.clip(round((horizon * FPS - (k - self._frame_index()) + 7) / 4), allowed[0], allowed[-1]))
            allowed = (T,)
            horizon = ((k - self._frame_index()) + C.valid_frame_count(T) - 1) / FPS
        action = self.action_queue.pop(0) if self.action_queue else None
        acfg = self.cfg.get(f"actions.{action}") if action else None
        if action and acfg:
            names = self.lib.find(acfg["source"])
            if not names:
                raise KeyError(f"action '{action}': no clip matches '{acfg['source']}'")
            win = self.lib.window_at(names[0], self._key_frame(names[0], acfg.get("key", "middle")))
            fp = self.sim_pos + np.array([math.sin(self.sim_yaw), 0, math.cos(self.sim_yaw)]) * float(
                np.linalg.norm(self.sim_vel)) * horizon
            target = place(win.local, fp, self.sim_yaw)
            self.current_action = action
            allowed = tuple(self.cfg.allowed_tokens)  # actions keep the model-chosen duration
        else:
            fp, fv, fyaw = self.predict(horizon, goal)
            # velocity in the future facing frame (x lateral, z forward)
            fwd = np.array([math.sin(fyaw), math.cos(fyaw)])
            right = np.array([fwd[1], -fwd[0]])
            v_local = np.array([fv[[0, 2]] @ right, fv[[0, 2]] @ fwd])
            crouch = cmd.gait == "crouch"
            win = self.lib.match(v_local, crouch=crouch, variation=float(self.cfg.get("animation.variation")),
                                 rng=self.rng)
            target = place(win.local, fp, fyaw)
            self.current_action = None
        req = GenerationRequest(context, target, allowed_tokens=allowed,
                                correct_endpoints=bool(gen["correct_endpoints"]),
                                match_velocity=bool(gen.get("velocity_matching", False)),
                                velocity_tau=float(gen.get("velocity_matching_tau", 4.0)))
        info = {"switch": k, "submitted": time.perf_counter(), "action": action}
        if self.sync:
            res = self.engine.generate_batch([req])[0]
            self._splice(res, info)
        else:
            self.pending = self.engine.submit([req])
            self.pending_info = info
        self.frames_since_plan = 0
        self.last_goal_vel = goal.copy()

    def _key_frame(self, name: str, key) -> int:
        """Frame of an action clip used as its target: min_height | max_height | middle | end | 0..1."""
        clip = self.lib.clips[name]
        h = clip.local[:, 0, 1]
        last = clip.num_frames - C.POSES_PER_TOKEN
        if isinstance(key, (int, float)) and not isinstance(key, bool):
            return int(np.clip(float(key) * last, 0, last))
        if key == "min_height":
            return int(np.clip(np.argmin(h), 0, last))
        if key == "max_height":
            return int(np.clip(np.argmax(h), 0, last))
        if key == "end":
            return last
        return last // 2

    def _splice(self, res, info) -> None:
        lat = time.perf_counter() - info["submitted"]
        a = float(self.cfg.get("generation.latency_ema"))
        self._latency_est = (1 - a) * self._latency_est + a * lat
        self.stats.latency_ms.append(lat * 1e3)
        self.stats.generations += 1
        k = info["switch"]
        seg = scale_intensity(res.local, float(self.cfg.get("animation.intensity")))
        now = self._frame_index()
        before_display = self.last_display
        # generated[0] == buffer[k]; keep history up to k, then the new motion
        self.buffer = np.concatenate([self.buffer[:k], seg], axis=0)
        self.contacts = np.concatenate([self.contacts[:k], res.contacts], axis=0)
        # trim old history to keep the buffer small (keep >= 8 frames before playhead)
        drop = max(0, int(now) - 8)
        if drop > 0:
            self.buffer = self.buffer[drop:]
            self.contacts = self.contacts[drop:]
            self.time -= drop * DT
            now -= drop
            self.commit_end -= drop  # keep an existing action commitment in buffer coordinates
        if info.get("action"):
            # commit to the action: no periodic/input replans until the segment is nearly played
            margin = self._latency_est * FPS + float(self.cfg.get("generation.replan_lead")) + 2
            self.commit_end = (k - drop) + len(seg) - 1 - margin
        if now > k - drop and before_display is not None and self.cfg.get("animation.inertialization"):
            # result arrived after the switch frame: hide the jump with inertialization
            self.stats.late_results += 1
            dst = self._sample_raw(self.time)
            dst_prev = self._sample_raw(self.time - DT)
            prev = self.prev_display if self.prev_display is not None else before_display
            self.inertializer.transition(before_display, prev, dst, dst_prev, DT)

    def _sample_raw(self, t: float) -> np.ndarray:
        f = np.clip(t * FPS, 0, len(self.buffer) - 1)
        i0 = int(math.floor(f))
        i1 = min(i0 + 1, len(self.buffer) - 1)
        return m3.tf_blend(self.buffer[i0], self.buffer[i1], f - i0)

    def _sample_contacts(self, t: float) -> np.ndarray:
        f = np.clip(t * FPS, 0, len(self.contacts) - 1)
        i0 = int(math.floor(f))
        i1 = min(i0 + 1, len(self.contacts) - 1)
        return self.contacts[i0] * (1 - (f - i0)) + self.contacts[i1] * (f - i0)

    # ------------------------------------------------------------------- tick
    def trigger(self, action: str) -> None:
        if self.cfg.get(f"actions.{action}") is None:
            raise KeyError(f"unknown action '{action}'")
        self.action_queue.append(action)
        self.frames_since_plan = 10**6  # replan now

    def update(self, cmd: Command, dt: float) -> np.ndarray:
        """Advance by `dt` seconds of game time and return the pose to render (local [J, 7])."""
        gen = self._sec("generation")
        anim = self._sec("animation")
        if cmd.action:
            self.trigger(cmd.action)
        goal = self._goal_velocity(cmd)
        self._simulate(goal, cmd, dt)

        if self.pending is not None and self.pending.done():
            info, fut = self.pending_info, self.pending
            self.pending, self.pending_info = None, None
            try:
                self._splice(fut.result()[0], info)
            except Exception as exc:  # noqa: BLE001 - never let a failed generation kill the game loop
                self.stats.failures += 1
                self.stats.last_error = repr(exc)

        remaining = len(self.buffer) - 1 - self._frame_index()
        input_changed = np.linalg.norm(goal - self.last_goal_vel) > float(gen["input_replan_threshold"])
        committed = self._frame_index() < self.commit_end
        need = (remaining < self._latency_est * FPS + gen["replan_lead"] + 2
                or (not committed and (self.frames_since_plan >= gen["replan_interval"]
                                       or (input_changed and self.frames_since_plan >= 2)
                                       or self.action_queue)))
        if need and self.pending is None:
            if input_changed:
                self.stats.replans_from_input += 1
            self._plan(cmd, goal)

        rate = float(anim["playback_rate"])
        self.time = min(self.time + dt * rate, (len(self.buffer) - 1) * DT)
        self.frames_since_plan += 1
        pose = self._sample_raw(self.time)
        if self.inertializer.active:
            pose = self.inertializer.apply(pose, dt)
        if anim["foot_locking"]["enabled"]:
            pose = self.feet.apply(pose, self._sample_contacts(self.time), dt)
        pose = self.smoother.apply(pose, dt)
        self.prev_display, self.last_display = self.last_display, pose
        return pose

    def close(self) -> None:
        if self.pending is not None:
            self.pending.result()
