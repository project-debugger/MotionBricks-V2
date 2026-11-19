#!/usr/bin/env python3
"""Build the showcase gallery: posters, MANIFEST.json, index.html, README.md and the ZIP.

Reads the rendered MP4s (ffprobe) and the provenance recorded by generate_motions.py
(model file, its SHA-256 prefix and how many SOMA generations produced each tile).
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from scenarios import VIDEOS  # noqa: E402

SHOW = ROOT / "showcase"
VID = SHOW / "videos"
POST = VID / "posters"
MOT = HERE / "cache" / "motions"
ZIP_NAME = "MotionBricks_SOMA_showcase.zip"

SECTIONS = [
    ("Movements", "One character, balanced preset (INT8 model)",
     ["01_walk_run_sprint", "02_turns_and_reversals", "03_crouch", "04_jump", "05_sit_and_stand",
      "06_transitions_medley"]),
    ("Animation speed", "Same input, three playback rates", ["07_animation_speed"]),
    ("Comparisons", "Same input sequence in every panel", [
        "cmp_fp32_vs_int8", "cmp_responsiveness", "cmp_smoothness", "cmp_model_variants_quality",
        "cmp_model_variants_aggressive", "cmp_foot_locking"]),
]

TITLES = {
    "01_walk_run_sprint": "Walk, run, sprint, stop",
    "02_turns_and_reversals": "Turns and reversals",
    "03_crouch": "Crouching",
    "04_jump": "Running jumps",
    "05_sit_and_stand": "Sit down, stand up",
    "06_transitions_medley": "Transitions medley",
    "07_animation_speed": "0.6x, 1.0x and 1.5x playback",
    "cmp_fp32_vs_int8": "FP32 vs INT8",
    "cmp_responsiveness": "Normal vs high responsiveness",
    "cmp_smoothness": "Default vs maximum smoothness",
    "cmp_model_variants_quality": "Four variants with FP32-level quality",
    "cmp_model_variants_aggressive": "Aggressive quantization",
    "cmp_foot_locking": "Foot locking off vs on",
}

NOTES = {
    "01_walk_run_sprint": "Idle into a walk, a run and a 5.5 m/s sprint, then a hard stop. The lean and arm swing grow with speed.",
    "02_turns_and_reversals": "Two 90 degree turns, a 180 degree reversal and a fast zig-zag, all at running speed.",
    "03_crouch": "Walk into a crouch walk, a crouched turn and a crouch idle, then stand up and walk off.",
    "04_jump": "Two running jumps. The hurdles are props placed at each generated jump apex afterwards; the model never sees them.",
    "05_sit_and_stand": "Walk, sit, stay seated, stand up and walk away. The chair is a prop placed under the generated seated pose.",
    "06_transitions_medley": "Twenty seconds of chained moves: walk, run, turn, sprint, stop, crouch, run, jump, walk.",
    "07_animation_speed": "The same turns at 0.6x, 1.0x and 1.5x animation playback rate.",
    "cmp_fp32_vs_int8": "The original FP32 model next to the INT8 model of the balanced preset, with the same inputs and settings.",
    "cmp_responsiveness": "The orange arrow is the player's input. Both follow a change of input in about 0.3 s (0.31 s default, 0.28 s high); the high setting mostly makes harsher poses, like the bent-over reversal.",
    "cmp_smoothness": "Maximum smoothness takes about 1.0 s to follow a change of input, against 0.3 s for the default, so turns, starts and stops are softer.",
    "cmp_model_variants_quality": "FP32, mixed FP16, INT8-pose and INT4-pose. These four keep FP32-level accuracy in the benchmarks.",
    "cmp_model_variants_aggressive": "FP32 and INT8-pose next to full INT8 and full INT4, the two variants that lose accuracy in the benchmarks.",
    "cmp_foot_locking": "A walk with turns and stops. A dot is stamped every frame the model reports a planted foot (teal left, orange right): a sliding foot draws a streak, a locked foot stacks its dots on one spot.",
}


def probe(p: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=width,height,codec_name,r_frame_rate,nb_frames",
                          "-of", "json", str(p)], capture_output=True, text=True, check=True)
    j = json.loads(out.stdout)
    s = j["streams"][0]
    return {"duration": float(j["format"]["duration"]), "bytes": int(j["format"]["size"]), "width": s["width"],
            "height": s["height"], "codec": s["codec_name"], "fps": s["r_frame_rate"], "frames": int(s.get("nb_frames", 0))}


def poster(p: Path, t: float, out: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(p), "-frames:v", "1",
                    "-vf", "scale=960:-2", "-q:v", "4", str(out)], check=True)


CSS = """
/* Layout: a single reel column on phones, a two-up contact sheet from tablet width; video first, words second */
:root {
  --bg: #edf1f5; --surface: #ffffff; --ink: #18212c; --muted: #56616f; --line: #d3dbe4;
  --accent: #c4501d; --chip: #e3e9f0;
  --display: "Space Grotesk", "Segoe UI", system-ui, sans-serif;
  --body: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1319; --surface: #161e28; --ink: #e6ebf0; --muted: #9ba7b3; --line: #27323e;
  --accent: #f08048; --chip: #1f2934; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1319; --surface: #161e28; --ink: #e6ebf0; --muted: #9ba7b3; --line: #27323e;
  --accent: #f08048; --chip: #1f2934; color-scheme: dark; }
body { background: var(--bg); color: var(--ink); font: 15px/1.55 var(--body); margin: 0; }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 28px 56px;
        padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header { display: grid; gap: 10px; margin-bottom: 26px; }
h1 { font: 700 clamp(28px, 6vw, 44px)/1.08 var(--display); letter-spacing: -0.02em; margin: 0; text-wrap: balance; }
.lede { color: var(--muted); max-width: 62ch; margin: 0; }
.proof { font: 500 12.5px/1.5 var(--mono); color: var(--ink); background: var(--chip); border-radius: 8px;
         padding: 8px 12px; justify-self: start; max-width: 100%; }
.proof b { color: var(--accent); font-weight: 600; }
section { display: grid; gap: 14px; margin-top: 34px; }
.sec-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 14px; border-top: 1px solid var(--line); padding-top: 14px; }
h2 { font: 600 22px/1.2 var(--display); margin: 0; }
.sec-head span { font: 500 12px/1.4 var(--mono); color: var(--muted); letter-spacing: 0.04em; text-transform: uppercase; }
.grid { display: grid; grid-template-columns: 1fr; gap: 26px 22px; }
@media (min-width: 760px) { .grid { grid-template-columns: 1fr 1fr; } .grid.one { grid-template-columns: 1fr; } }
figure { margin: 0; display: grid; gap: 8px; min-width: 0; }
video { width: 100%; max-width: 100%; aspect-ratio: 16 / 9; display: block; background: #0c1116; border-radius: 10px; }
figcaption { display: grid; gap: 4px; min-width: 0; }
h3 { font: 600 17px/1.3 var(--display); margin: 0; }
figcaption p { margin: 0; color: var(--muted); }
.meta { display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center; font: 500 12px/1.5 var(--mono); color: var(--muted); }
.meta a { color: var(--accent); font-weight: 600; text-decoration: none; border-bottom: 1px solid currentColor; }
.meta a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.how { margin-top: 40px; border-top: 1px solid var(--line); padding-top: 16px; display: grid; gap: 10px; max-width: 78ch; }
.how h2 { font-size: 18px; }
.how p, .how li { color: var(--muted); margin: 0; }
.how ul { margin: 0; padding-left: 18px; display: grid; gap: 4px; }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; font: 12.5px/1.45 var(--mono); min-width: 520px; }
th, td { text-align: left; padding: 5px 12px 5px 0; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--ink); font-weight: 600; } td { color: var(--muted); font-variant-numeric: tabular-nums; }
"""


def page(entries: dict, totals: dict, full_document: bool) -> str:
    head = ('<title>SOMA Motion Showcase</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600'
            '&family=Inter:wght@400;600&family=Space+Grotesk:wght@600;700&display=swap">\n'
            f"<style>{CSS}</style>\n")
    parts = [f"""<div class="wrap">
<header>
  <h1>SOMA Motion Showcase</h1>
  <p class="lede">{len(entries)} short clips of NVIDIA MotionBricks SOMA running in this repository's optimized runtime.
  Tap a video to play. Each one is 1080p H.264 and about {round(totals['avg_seconds'])} seconds long.</p>
  <div class="proof">Every frame is real model output: <b>{totals['generations']:,} SOMA generations</b> across
  {totals['models']} model files. Nothing is keyframed or played back from mocap.</div>
</header>"""]
    for title, sub, names in SECTIONS:
        names = [n for n in names if n in entries]
        if not names:
            continue
        parts.append(f'<section><div class="sec-head"><h2>{html.escape(title)}</h2><span>{html.escape(sub)}</span></div>'
                     f'<div class="grid{" one" if len(names) == 1 else ""}">')
        for n in names:
            e = entries[n]
            mb = e["bytes"] / 1e6
            models = ", ".join(sorted({m.replace("soma_", "").replace(".onnx", "") for m in e["models"]}))
            parts.append(f"""<figure>
  <video controls playsinline preload="metadata" poster="videos/posters/{n}.jpg" src="videos/{n}.mp4"></video>
  <figcaption>
    <h3>{html.escape(TITLES[n])}</h3>
    <p>{html.escape(NOTES[n])}</p>
    <div class="meta"><span>{e['duration']:.0f} s</span><span>{e['generations']} model calls</span>
      <span>{html.escape(models)}</span><span>{mb:.1f} MB</span><a href="videos/{n}.mp4">Open MP4</a></div>
  </figcaption>
</figure>""")
        parts.append("</div></section>")
    rows = "".join(f"<tr><td>{html.escape(m)}</td><td>{sha}</td><td>{c:,}</td></tr>"
                   for m, (sha, c) in sorted(totals["per_model"].items()))
    parts.append(f"""<div class="how">
  <h2>How these were made</h2>
  <ul>
    <li>Gameplay input is scripted (direction, gait, actions). The runtime's character controller turns it into
    target poses, and the SOMA network generates every transition from the 4 poses on screen.</li>
    <li>The body is NVIDIA's SOMA mesh from the MotionBricks software, skinned to the generated joints and rendered
    offline with three.js in headless Chromium, then encoded with x264.</li>
    <li>Comparison panels run the same input sequence with the same random seed; only the labelled setting differs.</li>
    <li>Green rings under the feet show the model's own foot-contact prediction. Hurdles and the chair are props
    placed after generation.</li>
  </ul>
  <div class="tbl"><table><thead><tr><th>Model file</th><th>SHA-256 (first 16)</th><th>Generations</th></tr></thead>
  <tbody>{rows}</tbody></table></div>
  <p>Generated on a 4-core CPU without a GPU. Source: showcase/build/ in the repository.</p>
</div></div>""")
    body = "\n".join(parts)
    if not full_document:
        return head + body + "\n"
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")


def main() -> int:
    POST.mkdir(parents=True, exist_ok=True)
    entries, per_model = {}, {}
    for name, vid in VIDEOS.items():
        mp4 = VID / f"{name}.mp4"
        spec_p = MOT / name / "video.json"
        if not mp4.exists() or not spec_p.exists():
            print(f"skip {name}: not rendered")
            continue
        info = probe(mp4)
        poster(mp4, min(info["duration"] * 0.42, info["duration"] - 0.5), POST / f"{name}.jpg")
        spec = json.loads(spec_p.read_text())
        tiles = []
        for t in spec["tiles"]:
            pv = t["provenance"]
            m = pv["model"].split("/")[-1]
            sha, c = per_model.get(m, (pv["model_sha256_16"], 0))
            per_model[m] = (sha, c + pv["generations"])
            tiles.append({"label": t["label"], "model": m, "model_sha256_16": pv["model_sha256_16"],
                          "generations": pv["generations"], "late_results": pv["late"],
                          "network_ms_mean": pv["network_ms_mean"], "config_changes": pv["config"]})
        entries[name] = dict(info, title=TITLES[name], description=NOTES[name], tiles=tiles,
                             generations=sum(t["generations"] for t in tiles),
                             models=[t["model"] for t in tiles], real_soma_output=True)
    totals = {"generations": sum(e["generations"] for e in entries.values()), "models": len(per_model),
              "avg_seconds": sum(e["duration"] for e in entries.values()) / max(1, len(entries)),
              "per_model": per_model}
    manifest = {"videos": entries, "totals": {k: v for k, v in totals.items() if k != "per_model"},
                "models": {m: {"sha256_16": s, "generations": c} for m, (s, c) in per_model.items()},
                "note": "All motion generated by NVIDIA MotionBricks SOMA through runtime/somart "
                        "(showcase/build/generate_motions.py); no keyframes or mocap playback."}
    (VID / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (SHOW / "index.html").write_text(page(entries, totals, True))
    (HERE / "cache" / "artifact_index.html").write_text(page(entries, totals, False))
    with zipfile.ZipFile(SHOW / ZIP_NAME, "w", compression=zipfile.ZIP_STORED) as z:
        for p in [SHOW / "index.html", SHOW / "README.md", VID / "MANIFEST.json"] + sorted(VID.glob("*.mp4")) + sorted(POST.glob("*.jpg")):
            if p.exists():
                z.write(p, Path("MotionBricks_SOMA_showcase") / p.relative_to(SHOW))
    total_mb = sum(e["bytes"] for e in entries.values()) / 1e6
    print(f"{len(entries)} videos, {total_mb:.1f} MB, {totals['generations']} SOMA generations; "
          f"zip {(SHOW / ZIP_NAME).stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
