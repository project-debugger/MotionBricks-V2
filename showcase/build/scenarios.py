"""Scripted gameplay inputs and the video/tile definitions of the showcase.

A timeline row is (start_s, direction_deg | None, magnitude, gait, action, caption):
direction 0 = +Z (away from the start camera), 90 = +X. None = no movement input.
"""
from __future__ import annotations

ONNX = "models/soma/optimized/onnx/"

LOCO = [(0.0, None, 0, "walk", None, "Idle"), (1.5, 0, 1, "walk", None, "Walk"),
        (4.5, 0, 1, "run", None, "Run"), (7.5, 0, 1, "sprint", None, "Sprint"),
        (10.5, None, 0, "run", None, "Stop - deceleration"), (12.5, None, 0, "walk", None, "Idle")]
LOCO_END = 15.0

TURNS = [(0.0, None, 0, "run", None, "Idle"), (1.0, 0, 1, "run", None, "Run"),
         (3.5, 90, 1, "run", None, "90 deg turn"), (6.0, 180, 1, "run", None, "90 deg turn"),
         (8.5, 0, 1, "run", None, "180 deg reversal"), (10.5, 45, 1, "run", None, "Zig-zag"),
         (11.3, -45, 1, "run", None, "Zig-zag"), (12.1, 45, 1, "run", None, "Zig-zag"),
         (12.9, -45, 1, "run", None, "Zig-zag"), (13.7, None, 0, "run", None, "Stop")]
TURNS_END = 16.0

CROUCH = [(0.0, None, 0, "walk", None, "Idle"), (1.0, 0, 1, "walk", None, "Walk"),
          (3.5, 0, 1, "crouch", None, "Crouch walk"), (6.5, 90, 1, "crouch", None, "Crouch turn"),
          (9.0, None, 0, "crouch", None, "Crouch idle"), (11.5, 90, 1, "walk", None, "Stand up + walk"),
          (14.0, None, 0, "walk", None, "Stop")]
CROUCH_END = 15.5

JUMP = [(0.0, None, 0, "run", None, "Idle"), (1.0, 0, 1, "run", None, "Run"),
        (4.0, 0, 1, "run", "jump", "Jump"), (6.5, 0, 1, "run", None, "Run"),
        (8.5, 0, 1, "run", "jump", "Jump"), (11.0, 0, 1, "run", None, "Run"),
        (12.5, None, 0, "run", None, "Stop")]
JUMP_END = 15.0

SIT = [(0.0, None, 0, "walk", None, "Idle"), (1.0, 0, 1, "walk", None, "Walk"),
       (3.5, None, 0, "walk", "sit", "Sit down"), (5.5, None, 0, "walk", "sit_idle", "Seated"),
       (7.0, None, 0, "walk", "sit_idle", "Seated"), (8.5, None, 0, "walk", "sit_idle", "Seated"),
       (10.0, None, 0, "walk", "stand_up", "Stand up"), (12.0, 0, 1, "walk", None, "Walk away"),
       (14.5, None, 0, "walk", None, "Stop")]
SIT_END = 16.0

MEDLEY = [(0.0, None, 0, "walk", None, "Idle"), (1.0, 0, 1, "walk", None, "Walk"),
          (3.0, 0, 1, "run", None, "Run"), (5.0, 90, 1, "run", None, "Turn"),
          (7.0, 90, 1, "sprint", None, "Sprint"), (9.0, None, 0, "run", None, "Stop"),
          (10.0, 90, 1, "crouch", None, "Crouch walk"), (12.5, 90, 1, "run", None, "Run"),
          (14.0, 90, 1, "run", "jump", "Jump"), (16.5, 0, 1, "walk", None, "Turn + walk"),
          (18.5, None, 0, "walk", None, "Stop")]
MEDLEY_END = 20.0

CMP = [(0.0, None, 0, "walk", None, "Idle"), (1.0, 0, 1, "walk", None, "Walk"),
       (3.0, 0, 1, "run", None, "Run"), (5.0, 90, 1, "run", None, "90 deg turn"),
       (7.0, -90, 1, "run", None, "180 deg reversal"), (9.0, -90, 1, "sprint", None, "Sprint"),
       (11.0, -90, 1, "crouch", None, "Crouch"), (13.5, None, 0, "walk", None, "Stop")]
CMP_END = 16.0

RESP = [(0.0, None, 0, "run", None, "Idle"), (1.0, 0, 1, "run", None, "Run"),
        (3.0, 90, 1, "run", None, "Sharp turn"), (4.5, 0, 1, "run", None, "Sharp turn"),
        (6.0, 180, 1, "run", None, "Reversal"), (8.0, 90, 1, "run", None, "Sharp turn"),
        (9.5, None, 0, "run", None, "Stop"), (10.5, 0, 1, "run", None, "Start"),
        (12.5, None, 0, "run", None, "Stop")]
RESP_END = 15.0

FOOT = [(0.0, None, 0, "walk", None, "Idle"), (1.0, 0, 1, "walk", None, "Walk"),
        (3.0, 90, 1, "walk", None, "Turn"), (5.0, 180, 1, "walk", None, "Turn"),
        (6.5, None, 0, "walk", None, "Stop"), (8.0, 270, 1, "walk", None, "Walk"),
        (10.0, 0, 1, "walk", None, "Turn"), (11.5, 90, 0.5, "walk", None, "Slow walk"),
        (13.5, None, 0, "walk", None, "Stop")]
FOOT_END = 16.0

BALANCED = {"model": ONNX + "soma_int8_pose.onnx"}
HIGH_RESP = {"locomotion": {"responsiveness": 0.05, "facing_responsiveness": 0.05, "acceleration": 22.0,
                            "deceleration": 28.0, "inertia": 0.8},
             "generation": {"prediction_horizon": 0.6, "replan_interval": 5, "input_replan_threshold": 0.2}}
MAX_SMOOTH = {"locomotion": {"responsiveness": 0.30, "facing_responsiveness": 0.30, "acceleration": 5.0,
                             "deceleration": 6.0, "inertia": 1.4},
              "generation": {"prediction_horizon": 1.2, "replan_interval": 14},
              "animation": {"smoothing": 0.6, "transition_duration": 0.40}}
FAIR = {"animation": {"variation": 0.0}}  # comparisons: identical target choice rules


def tile(label, sub, timeline, end, model=ONNX + "soma_int8_pose.onnx", overrides=None, props=True):
    return {"label": label, "sub": sub, "timeline": timeline, "end": end, "model": model,
            "overrides": overrides or {}, "props": props}


VIDEOS = {
    "01_walk_run_sprint": {"title": "Walk - Run - Sprint - Stop", "layout": "1",
                           "tiles": [tile("SOMA - INT8 (balanced preset)", "idle > walk > run > sprint > stop",
                                          LOCO, LOCO_END)]},
    "02_turns_and_reversals": {"title": "Turns, reversals and zig-zag", "layout": "1",
                               "tiles": [tile("SOMA - INT8 (balanced preset)", "90 deg turns, 180 deg reversal, zig-zag",
                                              TURNS, TURNS_END)]},
    "03_crouch": {"title": "Crouching", "layout": "1",
                  "tiles": [tile("SOMA - INT8 (balanced preset)", "walk > crouch walk > crouch turn > crouch idle > stand",
                                 CROUCH, CROUCH_END)]},
    "04_jump": {"title": "Running jumps", "layout": "1",
                "tiles": [tile("SOMA - INT8 (balanced preset)", "run > jump > run > jump > stop", JUMP, JUMP_END)]},
    "05_sit_and_stand": {"title": "Sit down and stand up", "layout": "1",
                         "tiles": [tile("SOMA - INT8 (balanced preset)", "walk > sit > seated > stand up > walk",
                                        SIT, SIT_END)]},
    "06_transitions_medley": {"title": "Transitions medley", "layout": "1",
                              "tiles": [tile("SOMA - INT8 (balanced preset)",
                                             "walk > run > turn > sprint > stop > crouch > run > jump > walk",
                                             MEDLEY, MEDLEY_END)]},
    "07_animation_speed": {"title": "Animation speed (playback rate)", "layout": "3",
                           "tiles": [tile("0.6x speed", "animation.playback_rate = 0.6", TURNS, TURNS_END,
                                          overrides={**FAIR, "animation": {"variation": 0.0, "playback_rate": 0.6}}),
                                     tile("1.0x speed (default)", "animation.playback_rate = 1.0", TURNS, TURNS_END,
                                          overrides=FAIR),
                                     tile("1.5x speed", "animation.playback_rate = 1.5", TURNS, TURNS_END,
                                          overrides={**FAIR, "animation": {"variation": 0.0, "playback_rate": 1.5}})]},
    "cmp_fp32_vs_int8": {"title": "Original FP32 vs optimized INT8", "layout": "2",
                         "tiles": [tile("FP32 - original precision", "soma_fp32.onnx (758 MB)", CMP, CMP_END,
                                        model=ONNX + "soma_fp32.onnx", overrides=FAIR),
                                   tile("INT8 - optimized (balanced)", "soma_int8_pose.onnx (350 MB, ~1.5x faster)",
                                        CMP, CMP_END, overrides=FAIR)]},
    "cmp_responsiveness": {"title": "Normal vs high responsiveness", "layout": "2", "arrows": True,
                           "tiles": [tile("Normal (default)", "responsiveness 0.12 s, horizon 0.8 s", RESP, RESP_END,
                                          overrides=FAIR),
                                     tile("High responsiveness", "responsiveness 0.05 s, horizon 0.6 s, accel 22",
                                          RESP, RESP_END, overrides={**HIGH_RESP, **{"animation": {"variation": 0.0}}})]},
    "cmp_smoothness": {"title": "Default vs maximum smoothness", "layout": "2", "arrows": True,
                       "tiles": [tile("Default", "balanced preset", RESP, RESP_END, overrides=FAIR),
                                 tile("Maximum smoothness", "smoothing 0.6, horizon 1.2 s, inertia 1.4",
                                      RESP, RESP_END,
                                      overrides={**MAX_SMOOTH, "animation": {**MAX_SMOOTH["animation"], "variation": 0.0}})]},
    "cmp_model_variants_quality": {"title": "Model variants with FP32-level quality", "layout": "4",
                                   "tiles": [tile("FP32", "758 MB - reference", CMP, CMP_END,
                                                  model=ONNX + "soma_fp32.onnx", overrides=FAIR),
                                             tile("FP16 (mixed)", "380 MB - GPU choice", CMP, CMP_END,
                                                  model=ONNX + "soma_fp16.onnx", overrides=FAIR),
                                             tile("INT8-pose (balanced)", "350 MB - ~1.5x faster", CMP, CMP_END,
                                                  overrides=FAIR),
                                             tile("INT4-pose (low memory)", "305 MB - smallest good-quality", CMP,
                                                  CMP_END, model=ONNX + "soma_int4_pose.onnx", overrides=FAIR)]},
    "cmp_model_variants_aggressive": {"title": "Aggressive quantization vs FP32", "layout": "4",
                                      "tiles": [tile("FP32", "reference", CMP, CMP_END,
                                                     model=ONNX + "soma_fp32.onnx", overrides=FAIR),
                                                tile("INT8-pose (balanced)", "recommended", CMP, CMP_END,
                                                     overrides=FAIR),
                                                tile("INT8 all layers (extreme)", "fastest, less accurate", CMP,
                                                     CMP_END, model=ONNX + "soma_int8.onnx", overrides=FAIR),
                                                tile("INT4 all layers", "not recommended - about 2x the error in benchmarks", CMP,
                                                     CMP_END, model=ONNX + "soma_int4.onnx", overrides=FAIR)]},
    "cmp_foot_locking": {"title": "Foot locking off vs on", "layout": "2", "feet_cam": True,
                         "tiles": [tile("Foot locking OFF", "raw model + endpoint correction", FOOT, FOOT_END,
                                        overrides={**FAIR, "animation": {"variation": 0.0,
                                                                         "foot_locking": {"enabled": False}}}),
                                   tile("Foot locking ON (default)", "model contacts -> two-bone IK", FOOT, FOOT_END,
                                        overrides=FAIR)]},
}
