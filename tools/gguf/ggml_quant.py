"""GGML block quantization (Q4_0, Q5_0, Q8_0 layouts as in ggml) applied to the int8 weights of the web SOMA model.

The weights are already int8 (dynamic INT8 quantization, MatMulInteger). Here the int8 integers are re-coded with
ggml's block formats for storage only; at load they are decoded and rounded back onto the int8 grid, so the runtime
graph, speed and memory are unchanged and only the download shrinks.
"""
import numpy as np

QK = 32


def _blocks(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).ravel()
    pad = (-len(x)) % QK
    return np.concatenate([x, np.zeros(pad, np.float32)]).reshape(-1, QK)


def q4_0_encode(x: np.ndarray) -> bytes:
    b = _blocks(x)
    idx = np.abs(b).argmax(1); mx = b[np.arange(len(b)), idx]
    d = mx / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0)
    q = np.clip(np.floor(b * inv[:, None] + 8.5), 0, 15).astype(np.uint8)
    qs = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    out = np.zeros(len(b), dtype=[('d', '<f2'), ('qs', 'u1', 16)])
    out['d'] = d.astype(np.float16); out['qs'] = qs
    return out.tobytes()


def q4_0_decode(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=[('d', '<f2'), ('qs', 'u1', 16)])
    lo = (a['qs'] & 0x0F).astype(np.int16) - 8; hi = (a['qs'] >> 4).astype(np.int16) - 8
    v = np.concatenate([lo, hi], 1).astype(np.float32) * a['d'].astype(np.float32)[:, None]
    return v.ravel()[:n]


def q5_0_encode(x: np.ndarray) -> bytes:
    b = _blocks(x)
    idx = np.abs(b).argmax(1); mx = b[np.arange(len(b)), idx]
    d = mx / -16.0
    inv = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0)
    q = np.clip(np.floor(b * inv[:, None] + 16.5), 0, 31).astype(np.uint32)
    qs = ((q[:, :16] & 0x0F) | ((q[:, 16:] & 0x0F) << 4)).astype(np.uint8)
    qh = np.zeros(len(b), np.uint32)
    for j in range(16):
        qh |= ((q[:, j] >> 4) & 1) << j
        qh |= ((q[:, j + 16] >> 4) & 1) << (j + 16)
    out = np.zeros(len(b), dtype=[('d', '<f2'), ('qh', '<u4'), ('qs', 'u1', 16)])
    out['d'] = d.astype(np.float16); out['qh'] = qh; out['qs'] = qs
    return out.tobytes()


def q5_0_decode(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=[('d', '<f2'), ('qh', '<u4'), ('qs', 'u1', 16)])
    lo = (a['qs'] & 0x0F).astype(np.int32); hi = (a['qs'] >> 4).astype(np.int32)
    j = np.arange(16)
    lo |= (((a['qh'][:, None] >> j) & 1) << 4).astype(np.int32)
    hi |= (((a['qh'][:, None] >> (j + 16)) & 1) << 4).astype(np.int32)
    v = (np.concatenate([lo, hi], 1) - 16).astype(np.float32) * a['d'].astype(np.float32)[:, None]
    return v.ravel()[:n]


CODECS = {'q4_0': (q4_0_encode, q4_0_decode, 2, 18), 'q5_0': (q5_0_encode, q5_0_decode, 6, 22)}  # name -> enc, dec, ggml type, bytes/block


def to_int8(v: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(v + 0.5), -128, 127).astype(np.int8)  # half up, as Math.round in js/gguf.js
