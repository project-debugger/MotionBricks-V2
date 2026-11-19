"""Layered, validated, live-editable YAML configuration.

    cfg = Config.load("balanced")                    # preset name, or a path to any YAML
    cfg.set("locomotion.run_speed", 4.0)             # runtime change, validated
    hero = cfg.for_character("heavy_guard")          # base + characters.heavy_guard
    jump = cfg.for_action("jump")                    # base + actions.jump overrides
    cfg.save("configs/custom/my_profile.yaml")       # only the keys that differ from defaults

Layering order: configs/default.yaml <- preset/profile (``extends`` chains allowed) <- overrides.
Validation: a key must exist in default.yaml (typos are rejected), keep its type, and respect
the numeric ranges in ``RANGES``.
"""
from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_PATH = CONFIG_DIR / "default.yaml"

RANGES: dict[str, tuple[float, float]] = {
    "inference.threads": (0, 256), "inference.inter_threads": (1, 64), "inference.max_batch": (1, 4096),
    "inference.memory_limit_mb": (0, 1 << 20), "inference.reserve_cores": (0, 64),
    "generation.min_tokens": (6, 16), "generation.max_tokens": (6, 16),
    "generation.prediction_horizon": (0.3, 2.0), "generation.replan_interval": (1, 60),
    "generation.replan_lead": (0, 30), "generation.input_replan_threshold": (0.0, 10.0),
    "generation.latency_ema": (0.0, 1.0), "generation.velocity_matching_tau": (0.5, 30.0),
    "locomotion.walk_speed": (0.0, 20.0), "locomotion.run_speed": (0.0, 20.0),
    "locomotion.sprint_speed": (0.0, 20.0), "locomotion.crouch_speed": (0.0, 20.0),
    "locomotion.acceleration": (0.1, 100.0), "locomotion.deceleration": (0.1, 100.0),
    "locomotion.responsiveness": (0.01, 2.0), "locomotion.inertia": (0.1, 5.0),
    "locomotion.momentum": (0.0, 1.0), "locomotion.turn_rate": (1.0, 3600.0),
    "locomotion.facing_responsiveness": (0.01, 2.0),
    "animation.playback_rate": (0.1, 4.0), "animation.transition_duration": (0.0, 2.0),
    "animation.smoothing": (0.0, 0.95), "animation.intensity": (0.0, 2.0), "animation.variation": (0.0, 1.0),
    "animation.foot_locking.contact_threshold": (0.0, 1.0), "animation.foot_locking.release_threshold": (0.0, 1.0),
    "animation.foot_locking.max_correction": (0.0, 1.0), "animation.foot_locking.blend_in": (0.0, 1.0),
    "animation.foot_locking.blend_out": (0.0, 1.0), "skeleton.scale": (0.01, 100.0),
    "animation.fingers.curl_idle": (-30.0, 90.0), "animation.fingers.curl_walk": (-30.0, 90.0),
    "animation.fingers.curl_run": (-30.0, 90.0), "animation.fingers.follow_through": (0.0, 60.0),
    "animation.fingers.variation": (0.0, 20.0),
}
FREE_FORM = ("actions", "characters", "skeletons", "inference.models_by_device")
CHOICES = {
    "generation.duration_mode": ("horizon", "model"),
    "inference.backend": ("onnxruntime", "torch"),
    "inference.precision": ("fp32", "fp16", "int8", "int4", "bf16"),
    "inference.graph_optimization": ("disable", "basic", "extended", "all"),
}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path, seen: set | None = None) -> dict:
    seen = seen or set()
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"circular 'extends' at {path}")
    seen.add(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent = data.pop("extends", None)
    if parent:
        ppath = (CONFIG_DIR / parent) if not Path(parent).is_absolute() else Path(parent)
        if not ppath.exists():
            ppath = path.parent / parent
        data = deep_merge(_read_yaml(ppath, seen), data)
    return data


def resolve_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)
    candidates = [p, CONFIG_DIR / p, CONFIG_DIR / "presets" / f"{p}.yaml", CONFIG_DIR / "custom" / f"{p}.yaml",
                  CONFIG_DIR / f"{p}.yaml"]
    for c in candidates:
        if c.is_file():
            return c
    raise ConfigError(f"config '{name_or_path}' not found (looked in configs/, presets/, custom/)")


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and not key.startswith(FREE_FORM):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


class Config:
    """Thread-safe nested config with dotted-path access and change listeners."""

    def __init__(self, data: dict, *, defaults: dict | None = None, source: str = "") -> None:
        self._defaults = defaults if defaults is not None else yaml.safe_load(DEFAULT_PATH.read_text())
        self._data = deep_merge(self._defaults, data)
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, Any], None]] = []
        self.source = source
        self.version = 0
        self.validate()

    # ------------------------------------------------------------ construction
    @classmethod
    def load(cls, name_or_path: str | Path | None = None, overrides: dict | None = None) -> "Config":
        data = {} if name_or_path in (None, "default") else _read_yaml(resolve_path(name_or_path))
        return cls(deep_merge(data, overrides or {}), source=str(name_or_path or "default"))

    @classmethod
    def presets(cls) -> list[str]:
        return sorted(p.stem for p in (CONFIG_DIR / "presets").glob("*.yaml"))

    # ------------------------------------------------------------------ access
    def get(self, path: str, default: Any = None) -> Any:
        with self._lock:
            node: Any = self._data
            for part in path.split("."):
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return copy.deepcopy(node)

    def __getitem__(self, path: str) -> Any:
        v = self.get(path, _MISSING)
        if v is _MISSING:
            raise KeyError(path)
        return v

    def section(self, name: str) -> dict:
        return self.get(name, {})

    def as_dict(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)

    def set(self, path: str, value: Any) -> None:
        """Change one value at runtime (validated); listeners are notified."""
        with self._lock:
            self._check(path, value)
            parts = path.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = copy.deepcopy(value)
            self.version += 1
        for cb in list(self._listeners):
            cb(path, value)

    def update(self, overrides: dict) -> None:
        for k, v in _flatten(overrides).items():
            self.set(k, v)

    def on_change(self, callback: Callable[[str, Any], None]) -> None:
        self._listeners.append(callback)

    # ---------------------------------------------------------------- layering
    def _scoped(self, scope: str, name: str) -> "Config":
        over = self.get(f"{scope}.{name}")
        if over is None:
            raise ConfigError(f"unknown {scope[:-1]} '{name}'")
        known = {k: v for k, v in over.items() if isinstance(v, dict) and k in self._defaults}
        return Config(deep_merge(self.as_dict(), known), defaults=self._defaults, source=f"{self.source}:{name}")

    def for_character(self, name: str) -> "Config":
        return self._scoped("characters", name)

    def for_action(self, name: str) -> "Config":
        return self._scoped("actions", name)

    def skeleton_settings(self, name: str | None = None) -> dict:
        return self.get(f"skeletons.{name or self.get('skeleton.name')}", {})

    # -------------------------------------------------------------- validation
    def _check(self, path: str, value: Any) -> None:
        if path.startswith(FREE_FORM):
            return
        default = _flatten(self._defaults).get(path, _MISSING)
        if default is _MISSING:
            raise ConfigError(f"unknown setting '{path}'")
        if isinstance(default, bool):
            ok = isinstance(value, bool)
        elif isinstance(default, (int, float)):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif isinstance(default, list):
            ok = isinstance(value, list)
        else:
            ok = isinstance(value, type(default))
        if not ok:
            raise ConfigError(f"'{path}' expects {type(default).__name__}, got {type(value).__name__}")
        if path in RANGES:
            lo, hi = RANGES[path]
            if not lo <= value <= hi:
                raise ConfigError(f"'{path}'={value} outside [{lo}, {hi}]")
        if path in CHOICES and value not in CHOICES[path]:
            raise ConfigError(f"'{path}' must be one of {CHOICES[path]}")

    def validate(self) -> None:
        for k, v in _flatten(self._data).items():
            if k.startswith(FREE_FORM) or k == "profile":
                continue
            self._check(k, v)
        if self.get("generation.min_tokens") > self.get("generation.max_tokens"):
            raise ConfigError("generation.min_tokens must be <= generation.max_tokens")

    # ------------------------------------------------------------------ saving
    def diff(self) -> dict:
        """Only the keys that differ from configs/default.yaml (what a profile file needs)."""
        base, cur = _flatten(self._defaults), _flatten(self.as_dict())
        out: dict = {}
        for k, v in cur.items():
            if base.get(k, _MISSING) != v:
                node = out
                parts = k.split(".")
                for p in parts[:-1]:
                    node = node.setdefault(p, {})
                node[parts[-1]] = v
        return out

    def save(self, path: str | Path, full: bool = False) -> Path:
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.as_dict() if full else self.diff(), sort_keys=False))
        return path

    @property
    def allowed_tokens(self) -> tuple[int, ...]:
        return tuple(range(int(self.get("generation.min_tokens")), int(self.get("generation.max_tokens")) + 1))


_MISSING = object()
