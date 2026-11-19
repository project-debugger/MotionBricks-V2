# Training and fine-tuning: what is (and is not) possible

**No training or fine-tuning has been performed in this repository.** Every model variant here is
derived from NVIDIA's released checkpoint by *export, precision conversion or quantization* —
none of which change what the model has learned. This page records what the package contains,
what fine-tuning would need, and the pieces already in place.

## Terminology used in this repository

| Operation | Changes learned behaviour? | Done here? |
|---|---|---|
| **Export** (SafeTensors → ONNX graph) | No — same weights, same maths | Yes (`tools/export_onnx.sh`, verified equal to NVIDIA's official ONNX) |
| **Precision conversion** (FP32 → mixed FP16, BF16 autocast) | No — rounding only | Yes (`soma_fp16.onnx`, `torch_bf16` backend) |
| **Quantization** (INT8 / INT4 weights) | No — approximates the same weights | Yes (`tools/optimize_models.py`) |
| **Runtime optimization** (graph fusion, threading, batching, async, post-processing) | No | Yes |
| **Fine-tuning / training** (gradient updates on data) | **Yes** | **No** — see below |

## What the released package contains (inspected)

Inspected with `safetensors` on `models/soma/original/`:

| Network | Tensors | Parameters | Notes |
|---|---|---|---|
| Pose VQ-VAE | 94 | 23.4 M | `pose_net.encoder` (40), `pose_net.decoder` (50), `pose_net.quantizer` (4) — **encoder and codebook are included** |
| Pose model (masked transformer) | 210 | 136.5 M | 16 layers × width 1024, token-embedding + logit heads |
| Root model | 174 | 40.4 M | shared transformer (6 layers), root-token transformer (2), conv output head, token-count logits |

Each checkpoint JSON lists the state that was **omitted**: `loops`, `callbacks`,
`optimizer_states`, `lr_schedulers` (PyTorch Lightning 2.4.0, global step 1,500,000, epoch 19,480).
The `config.yaml` files reference NVIDIA-internal training packages (`rnms.*`, `mfm.*`) that are
not public. The model card states the training corpus (Bones RigPlay 1, 313,526 clips) is not
distributed, and no training pipeline is included.

## Is fine-tuning feasible?

Technically **yes, with significant work**; practically it is **blocked on data**.

What makes it feasible:
- The network definitions are available (Apache-2.0) in the upstream exporter
  (`third_party/motionbricks/tools/onnx/models/`), and `somart.backends.TorchBackend` already
  builds the full PyTorch graph from the checkpoints (gradients flow; it is an `nn.Module`).
- The VQ-VAE **encoder + codebook** are present, so new motion can be converted to the discrete
  pose tokens the pose transformer predicts — the targets a masked-token fine-tune needs.
- `somart.codec` reproduces the exact input feature pipeline (verified bit-close to upstream), so
  training batches can be built from any SOMA-skeleton motion.
- `config.yaml` records the original hyper-parameters (min/max tokens, down-sampling, loss
  coefficients such as `commit_loss_coeff`, key-frame dropout probabilities) to start from.

What blocks it:
1. **Data.** A useful fine-tune needs a large, licensed motion dataset on the SOMA skeleton (or
   retargeted to it). The only clips available here are the upstream samples under the *NVIDIA
   Sample Data Evaluation License* (evaluation/testing only) — 38 clips, far too few and not
   licensed for producing a trained model. Public human mocap such as AMASS has its own
   non-commercial research licence and needs retargeting to SOMA.
2. **Training objective re-implementation.** The losses, masking schedule, key-frame sampling and
   root/pose conditioning used in training are only partially described by `config.yaml`; they
   must be re-implemented and validated (e.g. by checking that the released weights achieve low
   loss on held-out data before any update).
3. **Compute.** The released model trained for 1.5 M steps on a multi-GPU node ("4n8g" in the
   checkpoint names suggests 4 nodes × 8 GPUs). A fine-tune of the 136 M-parameter pose
   transformer is realistic on one modern GPU; on this CPU-only machine it is not.

## Recommended fine-tuning path (when data and a GPU are available)

1. Collect licensed motion on (or retargeted to) the 30-joint SOMA rig; resample to 30 Hz
   (`somart.skeleton.load_bvh(...).resample(30)`; `somart.skeletons.Retargeter` for same-morphology rigs).
2. Tokenize with the frozen VQ-VAE encoder; verify reconstruction error with the frozen decoder.
3. Fine-tune the **pose transformer only** (largest network, and the one that tolerates
   quantization — see `docs/OPTIMIZATION.md`) with masked-token cross-entropy, low learning rate,
   root model and VQ-VAE frozen. Keep `models/soma/original/` untouched; write to
   `models/soma/finetuned/<name>/` with its own checksums.
4. Evaluate against the baseline with the *same* harness: `benchmarks/run_benchmarks.py`
   (accuracy vs held-out mocap, duration agreement, foot sliding, join smoothness), on a held-out
   split of the new data plus the existing eval set to detect regressions.
5. Re-run `tools/export_onnx.sh` / `tools/optimize_models.py` on the fine-tuned weights.

Until steps 1–4 are done on real licensed data, no "trained" or "improved" model is claimed.
