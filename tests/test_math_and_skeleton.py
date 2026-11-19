import numpy as np

from somart import math3d as m3
from somart.bvh_writer import write_bvh
from somart.skeleton import Skeleton, load_bvh

RNG = np.random.default_rng(0)


def rand_quat(n):
    return m3.normalize(RNG.normal(size=(n, 4)))


def test_quat_matrix_roundtrip():
    q = rand_quat(500)
    q2 = m3.quat_from_matrix(m3.quat_to_matrix(q))
    assert np.allclose(m3.quat_angle_between(q, q2), 0, atol=1e-6)


def test_rot6d_roundtrip():
    q = rand_quat(500)
    assert np.allclose(m3.quat_angle_between(q, m3.quat_from_rot6d(m3.rot6d_from_quat(q))), 0, atol=1e-6)


def test_quat_fraction_matches_slerp():
    q = rand_quat(200)
    w = RNG.uniform(0, 1, size=200)
    a = m3.quat_fraction(q, w)
    b = m3.quat_slerp(m3.quat_identity((200,)), q, w)
    assert np.allclose(m3.quat_angle_between(a, b), 0, atol=1e-6)


def test_transform_inverse_and_fk_ik():
    parents = np.array([-1, 0, 1, 1, 3])
    local = np.concatenate([RNG.normal(size=(7, 5, 3)), rand_quat(35).reshape(7, 5, 4)], -1)
    world = m3.forward_kinematics(local, parents)
    back = m3.inverse_kinematics_local(world, parents)
    assert np.allclose(back[..., :3], local[..., :3], atol=1e-9)
    assert np.allclose(m3.quat_angle_between(back[..., 3:], local[..., 3:]), 0, atol=1e-6)
    t = m3.tf_mul(m3.tf_inv(world[:, 2]), world[:, 2])
    assert np.allclose(t[:, :3], 0, atol=1e-9)


def test_depth_level_fk_equals_sequential():
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 1])
    local = np.concatenate([RNG.normal(size=(3, 8, 3)), rand_quat(24).reshape(3, 8, 4)], -1)
    seq = np.empty_like(local)
    seq[:, 0] = local[:, 0]
    for j in range(1, 8):
        seq[:, j] = m3.tf_mul(seq[:, parents[j]], local[:, j])
    assert np.allclose(m3.forward_kinematics(local, parents), seq)


def test_bvh_write_read_roundtrip(tmp_path):
    names = ["Hips", "Spine", "Head", "LeftLeg", "LeftFoot"]
    parents = np.array([-1, 0, 1, 0, 3])
    ref = np.concatenate([np.array([[0, 1, 0], [0, .1, 0], [0, .3, 0], [.1, 0, 0], [0, -.4, 0]]),
                          m3.quat_identity((5,))], -1)
    sk = Skeleton(names, parents, ref)
    F = 12
    local = np.broadcast_to(ref, (F, 5, 7)).copy()
    local[:, :, 3:] = rand_quat(F * 5).reshape(F, 5, 4)
    local[:, 0, :3] = RNG.normal(size=(F, 3))
    p = write_bvh(tmp_path / "x.bvh", sk, local, fps=30)
    clip = load_bvh(p, drop_static_root=None)
    assert clip.fps == 30 and clip.skeleton.names == names
    assert np.allclose(clip.local[:, 0, :3], local[:, 0, :3], atol=1e-4)
    assert np.allclose(m3.quat_angle_between(clip.local[..., 3:], local[..., 3:]), 0, atol=1e-3)


def test_skeleton_subset_preserves_world_positions():
    names = ["A", "B", "C", "D"]
    ref = np.concatenate([RNG.normal(size=(4, 3)), rand_quat(4)], -1)
    sk = Skeleton(names, np.array([-1, 0, 1, 2]), ref)
    sub = sk.subset(["A", "C", "D"])
    assert list(sub.parents) == [-1, 0, 1]
    assert np.allclose(sk.fk(ref)[[0, 2, 3], :3], sub.fk(sub.reference)[:, :3])
