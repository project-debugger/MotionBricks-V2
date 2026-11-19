"""Left/right mirroring of SOMA motion (data augmentation for the target-window library).

The reflection is done in world space across the x = 0 plane (the SOMA mesh bind pose faces +Z and
is left/right symmetric about it), left/right joints are swapped, and a fixed per-joint frame
correction derived from the symmetric bind pose maps each mirrored world rotation into the local
axis convention of the counterpart joint. The rig's zero pose cannot be used for this: it is not
symmetric (left and right bones extend along opposite local axes). The result uses the rig's
reference bone offsets like any other clip.
"""
from __future__ import annotations

import numpy as np

from .skeleton import Skeleton


def _qmul(a, b):
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def _qconj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def _rot(q, v):
    qv = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1)
    return _qmul(_qmul(q, qv), _qconj(q))[..., :3]


class Mirror:
    REFLECT_P = np.array([-1.0, 1.0, 1.0])          # x -> -x
    REFLECT_Q = np.array([1.0, -1.0, -1.0, 1.0])     # M R M for M = diag(-1, 1, 1)

    def __init__(self, sk: Skeleton, bind_world_quats: np.ndarray) -> None:
        """bind_world_quats: [J, 4] world rotations of the joints in a left/right symmetric pose."""
        self.sk = sk
        n = sk.names
        self.swap = np.array([n.index(x.replace("Left", "#").replace("Right", "Left").replace("#", "Right")) for x in n])
        b = np.asarray(bind_world_quats, dtype=np.float64)
        mq = b[self.swap] * self.REFLECT_Q                             # mirrored bind rotations of the counterparts
        self.corr = _qmul(_qconj(mq), b)                               # per-joint frame correction

    @staticmethod
    def bind_from_inverse_bind(inv_bind_colmajor: np.ndarray) -> np.ndarray:
        """World rotations [J, 4] from three.js-style (column-major) inverse bind matrices [J, 16]."""
        out = []
        for m in inv_bind_colmajor:
            R = np.linalg.inv(np.asarray(m, dtype=np.float64).reshape(4, 4).T)[:3, :3]
            R /= np.cbrt(np.linalg.det(R))
            w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
            if w > 1e-3:
                q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
            else:  # robust branch (Shepperd)
                i = int(np.argmax(np.diag(R)))
                j, k = (i + 1) % 3, (i + 2) % 3
                r = np.sqrt(max(1e-12, 1 + R[i, i] - R[j, j] - R[k, k]))
                q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / (2 * r)]
                q[i] = r / 2; q[j] = (R[j, i] + R[i, j]) / (2 * r); q[k] = (R[k, i] + R[i, k]) / (2 * r)
            q = np.array(q)
            out.append(q / np.linalg.norm(q))
        return np.array(out)

    def __call__(self, local: np.ndarray) -> np.ndarray:
        sk = self.sk
        w = sk.fk(local)
        wq = _qmul(w[:, self.swap, 3:] * self.REFLECT_Q, self.corr[None])
        wp = w[:, self.swap, :3] * self.REFLECT_P
        out = local.copy()
        out[:, 0, :3] = wp[:, 0]
        out[:, 0, 3:] = wq[:, 0]
        for j in range(1, len(sk.names)):
            p = sk.parents[j]
            out[:, j, 3:] = _qmul(_qconj(wq[:, p]), wq[:, j])
        out[..., 3:] /= np.linalg.norm(out[..., 3:], axis=-1, keepdims=True)
        return out
