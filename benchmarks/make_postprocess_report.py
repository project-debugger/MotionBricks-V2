#!/usr/bin/env python3
"""Turn benchmarks/results/latest_ablation.json into benchmarks/REPORT_postprocess.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "benchmarks/results/latest_ablation.json").read_text())
desc = {"A_raw": "raw network output", "B_upstream": "+ upstream endpoint correction (NVIDIA engine behaviour)",
        "C_velocity": "+ velocity-matched joins (somart)", "D_footlock": "upstream + foot locking (somart)",
        "E_both": "upstream + velocity matching + foot locking (**somart default**)",
        "F_contact": "contact-aware root ramp instead of linear (somart, experimental)",
        "G_full": "contact-aware + velocity + foot locking", "GT": "held-out mocap (reference level)"}
L = ["# Animation clean-up ablation\n",
     f"`benchmarks/ablation_postprocess.py --model {r['model']}` on all {r['requests']} held-out eval windows "
     "(duration pinned to the ground-truth gap, so every frame is compared with mocap).\n",
     "| variant | what | MPJPE cm | foot slide cm/s (height contacts) | foot slide cm/s (model contacts) | jerk m/s^3 | "
     "join pop start / end | end-pose error cm |", "|---|---|---|---|---|---|---|---|"]
for k, v in r["results"].items():
    sp = v.get("slide_pred")
    sp = f"{sp*100:.2f}" if sp is not None else "-"
    L.append(f"| {k} | {desc[k]} | {v['mpjpe']*100:.2f} | {v['slide_h']*100:.2f} | {sp} | {v['jerk']:.1f} | "
             f"{v['join_start']:.2f} / {v['join_end']:.2f} | {v['end_err']*100:.2f} |")
b, e, f, gt = (r["results"][k] for k in ("B_upstream", "E_both", "F_contact", "GT"))
pct = lambda x, y: (x / y - 1) * 100  # noqa: E731
L += ["\n## Result\n",
      f"- Velocity matching + foot locking vs upstream correction: MPJPE {b['mpjpe']*100:.2f} -> {e['mpjpe']*100:.2f} cm "
      f"({pct(e['mpjpe'], b['mpjpe']):+.1f} %), foot sliding {b['slide_h']*100:.2f} -> {e['slide_h']*100:.2f} cm/s "
      f"({pct(e['slide_h'], b['slide_h']):+.1f} %), join pops {b['join_start']:.2f}/{b['join_end']:.2f} -> "
      f"{e['join_start']:.2f}/{e['join_end']:.2f} ({pct(e['join_start'], b['join_start']):+.0f} % / "
      f"{pct(e['join_end'], b['join_end']):+.0f} %).",
      "- Upstream endpoint correction removes the raw endpoint error and the large raw join pops, but *increases* "
      "foot sliding (it spreads leg-joint corrections over the segment); foot locking removes that again.",
      f"- Contact-aware root ramp (F): foot sliding {f['slide_h']*100:.2f} vs {b['slide_h']*100:.2f} cm/s -> no "
      "measurable benefit; kept as an option, not enabled (negative result).",
      f"- Remaining gap to mocap: foot sliding {e['slide_h']/gt['slide_h']:.1f}x and join pops "
      f"{e['join_start']/gt['join_start']:.1f}x / {e['join_end']/gt['join_end']:.1f}x the ground-truth level."]
(ROOT / "benchmarks/REPORT_postprocess.md").write_text("\n".join(L) + "\n")
print("wrote benchmarks/REPORT_postprocess.md")
