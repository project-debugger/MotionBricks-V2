"""Write a test vector (inputs + PyTorch outputs from the exported float16 weights) for the JS DogNet."""
import json, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mann import MANN

out = Path(sys.argv[1]); meta = json.loads((out / "dognet.json").read_text()); raw = (out / "dognet.bin").read_bytes()
T = {t["name"]: np.frombuffer(raw, np.float16, count=int(np.prod(t["shape"])), offset=t["offset"]).astype(np.float32).reshape(t["shape"]) for t in meta["tensors"]}
gin, xin, yout = T["g_mean"].shape[0], T["x_mean"].shape[0], T["y_mean"].shape[0]
m = MANN(gin, xin, yout, K=meta["experts"], H=meta["hidden"]); sd = m.state_dict()
m.load_state_dict({k: torch.from_numpy(T[k]) for k in sd}); m.eval()
rng = np.random.default_rng(0)
g = (T["g_mean"] + rng.standard_normal((3, gin)).astype(np.float32) * T["g_std"]).astype(np.float32)
x = (T["x_mean"] + rng.standard_normal((3, xin)).astype(np.float32) * T["x_std"] * 0.5).astype(np.float32)
with torch.no_grad():
    yn = m(torch.from_numpy((g - T["g_mean"]) / T["g_std"]), torch.from_numpy((x - T["x_mean"]) / T["x_std"])).numpy()
y = yn * T["y_std"] + T["y_mean"]
Path(sys.argv[2]).write_text(json.dumps({"g": g.tolist(), "x": x.tolist(), "y": y.tolist()}))
print("wrote", sys.argv[2])
