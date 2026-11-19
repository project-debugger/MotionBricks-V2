"""Multi-skeleton architecture: skeleton profiles, model-family registry, retargeting.

IMPORTANT: the released MotionBricks SOMA weights are a *human* model for the 30-joint SOMA
rig. Nothing here makes that network animate animals. This module provides the
skeleton-agnostic plumbing a future animal model needs:

* ``SkeletonProfile``  - joints + semantic roles (pelvis, spine, legs, head, tail, wings),
                         contact end-effectors and heading joints, loadable from YAML.
* ``FeatureLayout``    - the MotionBricks motion representation generalised to any joint count
                         (root, joint positions, 6D rotations, velocities, contacts). For SOMA it
                         reproduces the released tensor sizes exactly (267 input / 366 output).
* ``MODEL_FAMILIES``   - registry mapping skeleton families to trained model packages; families
                         without weights are registered as ``requires_training`` and refuse to run.
* ``Retargeter``       - name/role-mapped rotation retargeting between rigs of the same
                         morphology (e.g. another biped rig <-> SOMA), with rest-pose correction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import contract as C
from . import math3d as m3
from .skeleton import Skeleton

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "configs" / "skeletons"


@dataclass
class SkeletonProfile:
    name: str
    family: str                             # e.g. biped_human, quadruped, avian
    joints: list[str]
    parents: list[int]
    offsets: list[list[float]] | None = None  # rest offsets (m); None until a rig is bound
    roles: dict = field(default_factory=dict)  # semantic role -> joint name(s)
    contacts: list[str] = field(default_factory=list)
    heading_joints: tuple[str, str] = ("", "")  # (right, left) joints defining the lateral axis
    up_axis: str = "y"
    notes: str = ""

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    def to_skeleton(self) -> Skeleton:
        off = np.zeros((self.num_joints, 3)) if self.offsets is None else np.asarray(self.offsets, dtype=np.float64)
        ref = np.concatenate([off, m3.quat_identity((self.num_joints,))], axis=-1)
        return Skeleton(list(self.joints), np.asarray(self.parents), ref, name=self.name)

    @classmethod
    def load(cls, name_or_path: str | Path) -> "SkeletonProfile":
        p = Path(name_or_path)
        if not p.is_file():
            p = PROFILE_DIR / f"{name_or_path}.yaml"
        d = yaml.safe_load(p.read_text())
        joints, parents = [], []
        for item in d["hierarchy"]:  # list of "Joint: Parent" pairs (root parent = null)
            (j, par), = item.items()
            joints.append(j)
            parents.append(-1 if par is None else joints.index(par))
        return cls(d["name"], d["family"], joints, parents, d.get("offsets"), d.get("roles", {}),
                   d.get("contacts", []), tuple(d.get("heading_joints", ("", ""))), d.get("up_axis", "y"),
                   d.get("notes", ""))

    def validate(self) -> list[str]:
        errs = []
        if self.parents[0] != -1:
            errs.append("first joint must be the root")
        for j, p in enumerate(self.parents[1:], 1):
            if not 0 <= p < j:
                errs.append(f"joint {self.joints[j]} parent must precede it")
        for c in list(self.contacts) + [h for h in self.heading_joints if h]:
            if c not in self.joints:
                errs.append(f"unknown joint '{c}' in contacts/heading")
        for role, names in self.roles.items():
            for n in ([names] if isinstance(names, str) else names):
                if n not in self.joints:
                    errs.append(f"role {role}: unknown joint '{n}'")
        return errs


@dataclass(frozen=True)
class FeatureLayout:
    """MotionBricks dual representation sized for a skeleton with J joints and K contacts."""

    num_joints: int
    num_contacts: int

    @property
    def local_pose_dim(self) -> int:         # input: child joint positions + 6D rotations
        return 3 * (self.num_joints - 1) + 6 * self.num_joints

    @property
    def global_motion_dim(self) -> int:      # output: root(5) + positions + 6D + velocities + contacts
        return 5 + 3 * (self.num_joints - 1) + 6 * self.num_joints + 3 * self.num_joints + self.num_contacts

    global_root_dim: int = 5
    local_root_dim: int = 4

    @classmethod
    def for_profile(cls, prof: SkeletonProfile) -> "FeatureLayout":
        return cls(prof.num_joints, len(prof.contacts))


SOMA_LAYOUT = FeatureLayout(C.NUM_JOINTS, 4)
assert SOMA_LAYOUT.local_pose_dim == C.INPUT_SHAPES["local_poses"][1] == 267
assert SOMA_LAYOUT.global_motion_dim == 366


@dataclass
class ModelFamily:
    name: str
    skeleton_family: str
    status: str                 # available | requires_training
    package: str = ""
    description: str = ""


MODEL_FAMILIES: dict[str, ModelFamily] = {
    "soma_human": ModelFamily("soma_human", "biped_human", "available", "models/soma",
                              "NVIDIA MotionBricks SOMA (30-joint human, trained on Bones RP1 mocap)"),
    "quadruped": ModelFamily("quadruped", "quadruped", "requires_training", "",
                             "Dogs, cats, wolves, horses, bears: needs licensed quadruped mocap + training"),
    "avian": ModelFamily("avian", "avian", "requires_training", "",
                         "Birds: needs flight/locomotion capture + training; wings as extra limb chains"),
}


def model_for(profile: SkeletonProfile) -> ModelFamily:
    for fam in MODEL_FAMILIES.values():
        if fam.skeleton_family == profile.family:
            if fam.status != "available":
                raise NotImplementedError(
                    f"No trained motion model for skeleton family '{profile.family}' ({profile.name}). "
                    f"{fam.description}. See docs/ANIMALS.md for the training path.")
            return fam
    raise KeyError(f"no model family registered for '{profile.family}'")


class Retargeter:
    """Copy rotations between two rigs of the same morphology using a joint-name map.

    For each mapped joint the *world* rotation change relative to the source rest pose is
    applied to the target rest pose, so rigs with different bone orientations retarget
    correctly. Root translation is scaled by the ratio of hip heights.
    """

    def __init__(self, src: Skeleton, dst: Skeleton, mapping: dict[str, str]) -> None:
        self.src, self.dst = src, dst
        self.map = {dst.index(d): src.index(s) for s, d in mapping.items() if src.index(s) >= 0 and dst.index(d) >= 0}
        sw, dw = src.fk(src.reference), dst.fk(dst.reference)
        self.src_rest_world, self.dst_rest_world = sw[:, 3:], dw[:, 3:]
        hs, hd = float(sw[0, 1]), float(dw[0, 1])  # rest root heights
        self.scale = hd / hs if hs > 1e-3 and hd > 1e-3 else 1.0

    def __call__(self, src_local: np.ndarray) -> np.ndarray:
        F = src_local.shape[0]
        sw = self.src.fk(src_local)
        out_world = np.broadcast_to(self.dst_rest_world, (F,) + self.dst_rest_world.shape).copy()
        for dj, sj in self.map.items():
            delta = m3.quat_mul(sw[:, sj, 3:], m3.quat_conj(self.src_rest_world[sj]))
            out_world[:, dj] = m3.quat_mul(delta, self.dst_rest_world[dj])
        out = np.broadcast_to(self.dst.reference, (F,) + self.dst.reference.shape).copy()
        for j in range(self.dst.num_joints):
            p = self.dst.parents[j]
            parent_rot = out_world[:, p] if p >= 0 else m3.quat_identity((F,))
            out[:, j, 3:] = m3.normalize(m3.quat_mul(m3.quat_conj(parent_rot), out_world[:, j]))
        out[:, 0, :3] = src_local[:, 0, :3] * self.scale
        return out
