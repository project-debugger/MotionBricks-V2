# datasets/

| File | Purpose |
|---|---|
| `build_eval_set.py` | Builds the held-out inbetweening eval set (187 windows) from the 38 upstream SOMA sample clips → `soma_eval/` (git-ignored) |
| `eval_manifest.json` | Which clip / frame / gap each eval window uses (no motion data) — makes results reproducible |
| `prepare_motion_dataset.py` | Converts **your own licensed** BVH clips on the SOMA skeleton into 30 Hz arrays (+ optional encoded windows) with a mandatory licence/provenance field — the data-prep step for fine-tuning (see `docs/TRAINING.md`) |

Licensing rules followed here:

- The upstream sample clips are under the *NVIDIA Sample Data Evaluation License* (internal
  evaluation and testing of NVIDIA technologies). They are used only for evaluation, stay in
  `third_party/`, and nothing derived from them is committed.
- No third-party dataset was downloaded. Candidate human/animal datasets and their licence status
  are listed in `docs/TRAINING.md` and `docs/ANIMALS.md`; verify terms before use.
- Put prepared datasets in their own folder and keep their `manifest.json` (it records the licence).
