# Animal support: architecture and path forward

> **The released MotionBricks SOMA model animates humans only.** It was trained on human motion
> capture for one 30-joint human skeleton. It cannot animate dogs, cats, horses, wolves, bears or
> birds, and nothing in this repository claims otherwise. What exists here is the skeleton-agnostic
> plumbing a future animal model needs, with human functionality untouched.

> The web playground's animals (`site/`) do not use SOMA: they run a separate, small mode-adaptive
> network trained in this repository on 41 minutes of dog motion capture (`tools/animal/train_mann.py`,
> CC BY-NC 4.0 data, non-commercial). Only the dog is real data; the other species are the dog's
> motion re-proportioned. See `site/README.md`, "Animals (trained network)".

## What is implemented

`runtime/somart/skeletons.py`:

| Component | Purpose | Status |
|---|---|---|
| `SkeletonProfile` | Any hierarchy + semantic roles (pelvis, spine, neck, head, legs, wings, tail), contact end-effectors, heading joints; YAML in `configs/skeletons/` | Working, validated by tests |
| `FeatureLayout` | The MotionBricks motion representation generalised to J joints / K contacts (root 5, positions 3(J−1), 6D rotations 6J, velocities 3J, contacts K). Reproduces SOMA's released sizes exactly (267 in / 366 out) | Working |
| `MODEL_FAMILIES` / `model_for()` | Registry mapping skeleton families to trained packages. `soma_human` = available; `quadruped`, `avian` = `requires_training` and **raise `NotImplementedError`** if used | Working |
| `Retargeter` | Rest-pose-corrected rotation retargeting between rigs of the *same morphology* (e.g. another human rig ↔ SOMA) | Working (human↔human); human→quadruped is not meaningful and not offered |
| Per-skeleton config | `skeletons.<name>` in the config: foot joints, IK leg chains (2 for bipeds, 4 for quadrupeds) used by foot locking | Working (foot locker accepts any number of chains) |

Templates (`configs/skeletons/*.yaml`, **topology + roles only, no rest offsets, no motion**):

| Template | Joints | Feature dims (in / out) | Notes |
|---|---|---|---|
| dog | 29 | 258 / 354 | digitigrade, 3 tail joints |
| wolf | 30 | 267 / 366 | dog topology, 4 tail joints |
| cat | 31 | 276 / 378 | flexible spine, 5 tail joints |
| horse | 25 | 222 / 306 | unguligrade (hoof = foot, no toe), long neck |
| bear | 27 | 240 / 330 | plantigrade hind feet |
| bird | 25 | 222 / 304 | wings as limb chains, 2 contacts |

## What a working animal model needs

1. **Licensed animal motion data** on a consistent skeleton per species family. Candidates
   (licences must be verified by you before use — none were downloaded here):
   - *AI4Animation / MANN dog motion capture* (Zhang, Starke et al., SIGGRAPH 2018) — reported as
     CC BY-NC 4.0 (non-commercial) by downstream sources; check the original ReadMe.
   - *Truebones Zoo* (Truebones Motions Animation Studios) — ~70 animal skeletons used in recent
     topology-agnostic research (e.g. AnyTop); commercial asset pack, licence terms must be checked
     with the publisher.
   - Video/RGB-D datasets (e.g. DogMo) need motion reconstruction before use; synthetic sets
     (e.g. DigiDogs) carry their own terms.
2. **A per-family representation** — `FeatureLayout.for_profile(profile)` gives the tensor sizes;
   heading joints become the hind-leg roots for quadrupeds; contacts become 4 paws.
3. **Training** the three MotionBricks networks (VQ-VAE, root model, pose model) for that layout —
   the released human weights cannot be reused directly because every input/output dimension
   differs. Fine-tuning from the human weights is only plausible for the transformer cores with
   new input/output projections, and is unproven.
4. **Registration** — add a `ModelFamily(..., status="available", package=...)` entry, export with
   the same tooling, and benchmark with the same harness (foot sliding matters even more for
   quadrupeds: 4 contacts).

## Multi-skeleton support today (humans)

Characters with a different *human* rig can use SOMA today by retargeting their motion to the SOMA
skeleton for generation and back for display (`Retargeter`), as long as the rig has the 30 model
joints or a name map to them. The engine itself binds any skeleton containing the 30 model joint
names (extra joints such as fingers keep their reference transforms), exactly like upstream.
