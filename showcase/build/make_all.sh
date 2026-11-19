#!/usr/bin/env bash
# Rebuild the whole showcase: SOMA mesh -> real model motion -> rendered MP4s -> gallery + zip.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
import sys; sys.path.insert(0, "tools")
import fetch_upstream as fu
fu.clone()
print(fu.fetch_lfs("motionbricks/assets/human/soma_base_skel_minimal_no_root.usd"))
PY
(cd showcase/build/renderer && npm install --no-audit --no-fund)
.venv/bin/python showcase/build/extract_mesh.py
.venv/bin/python showcase/build/generate_motions.py
bash showcase/build/render_parallel.sh
.venv/bin/python showcase/build/make_gallery.py
