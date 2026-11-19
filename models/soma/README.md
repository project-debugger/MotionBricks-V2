# models/soma

| Folder | Contents | In git? |
|---|---|---|
| `original/` | The complete, **untouched** `soma/` folder of [nvidia/MotionBricks](https://huggingface.co/nvidia/MotionBricks/tree/main/soma) (25 files, 801 MB, FP32 SafeTensors + configs + stats + skeleton). Never modified by any tool. | no (size) — `python3 tools/download_soma.py` |
| `optimized/onnx/` | Exports and optimized variants (FP32, FP32-ORT-optimized, mixed FP16, INT8, INT8-pose, INT4, INT4-pose). | no (size) — `bash tools/export_onnx.sh && .venv/bin/python tools/optimize_models.py all` |
| `optimized/manifest.json` | How each variant was produced, its size and SHA-256. | yes |
| `checksums/` | Upstream SHA256SUMS (repo-level and folder-level), the local verification report and the SHA-256 of every optimized file. | yes |

Verification: `tools/download_soma.py` checks every file against **both** upstream checksum files
before moving it into place (`checksums/verification_report.json`), and refuses any path outside
`soma/` (the G1 package is never downloaded). `python3 tools/download_soma.py --verify-only`
re-hashes what is on disk.

Licence: the weights are under the [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/)
(commercial use and redistribution of modified models permitted, subject to its terms).
Optimized variants are modified models under the same agreement.
