"""SOMA model tensor contract (mirrors upstream ``motionbricks.model.contract``)."""
from __future__ import annotations

SAMPLE_RATE = 30.0
POSES_PER_TOKEN = 4
CONTEXT_AND_TARGET_POSES = 2 * POSES_PER_TOKEN
MIN_TOKENS = 6
MAX_TOKENS = 16
BOUNDARY_FRAMES = POSES_PER_TOKEN - 1
MAX_RAW_FRAMES = MAX_TOKENS * POSES_PER_TOKEN  # 64
DEFAULT_TOKEN_COUNTS = tuple(range(8, 15))


def valid_frame_count(tokens: int) -> int:
    if not MIN_TOKENS <= tokens <= MAX_TOKENS:
        raise ValueError(f"token count must be {MIN_TOKENS}..{MAX_TOKENS}, got {tokens}")
    return tokens * POSES_PER_TOKEN - 2 * BOUNDARY_FRAMES


MAX_VALID_FRAMES = valid_frame_count(MAX_TOKENS)  # 58

JOINT_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)
NUM_JOINTS = len(JOINT_NAMES)
FOOT_CONTACT_JOINTS = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
HEADING_JOINTS = ("RightLeg", "LeftLeg")

INPUT_NAMES = (
    "global_root_values", "has_global_root_values", "local_root_values", "has_local_root_values",
    "local_poses", "has_local_poses", "allowed_pred_num_tokens",
)
OUTPUT_NAMES = ("pred_global_motions", "pred_num_tokens")
INPUT_SHAPES = {
    "global_root_values": (8, 5), "has_global_root_values": (8,), "local_root_values": (8, 4),
    "has_local_root_values": (8,), "local_poses": (8, 267), "has_local_poses": (8,),
    "allowed_pred_num_tokens": (11,),
}
# Output feature layout per frame (366): root(5) | child joint pos(87) | 6D rot(180) | vel(90) | contacts(4)
OUT_POS = slice(5, 5 + 3 * (NUM_JOINTS - 1))
OUT_ROT_OFFSET = 5 + 3 * (NUM_JOINTS - 1)  # 92
OUT_CONTACTS = slice(362, 366)


def allowed_token_mask(counts) -> list[int]:
    counts = set(int(c) for c in counts)
    if not counts or any(not MIN_TOKENS <= c <= MAX_TOKENS for c in counts):
        raise ValueError(f"allowed token counts must be within {MIN_TOKENS}..{MAX_TOKENS}")
    return [1 if MIN_TOKENS + i in counts else 0 for i in range(MAX_TOKENS - MIN_TOKENS + 1)]
