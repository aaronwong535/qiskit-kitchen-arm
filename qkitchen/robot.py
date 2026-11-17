"""UR5e model: kinematics, capsule collision geometry and rigid-body dynamics.

Kinematic and mass/COM parameters are the UR5e values published by Universal
Robots ("DH Parameters for calculations of kinematics and dynamics"). UR lists the
link inertia tensors as negligible, so link inertia here comes from the point masses
plus a reflected rotor ("armature") inertia per joint. The armature, friction,
gripper and capsule radii are modelling assumptions, flagged below.

Everything is batched: q has shape (B, 6) so whole sets of rollouts run in one
numpy call.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- parameters
DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])

LINK_MASS = np.array([3.761, 8.058, 2.846, 1.37, 1.3, 0.365])
LINK_COM = np.array([
    [0.0, -0.02561, 0.00193],
    [0.2125, 0.0, 0.11336],
    [0.15, 0.0, 0.0265],
    [0.0, -0.0018, 0.01634],
    [0.0, 0.0018, 0.01634],
    [0.0, 0.0, -0.001159],
])

# --- assumptions (not from the UR datasheet) ---
TOOL_LEN = 0.15            # TCP offset of a 2-finger gripper along z6 [m]
TOOL_MASS = 0.9            # gripper mass [kg] (Robotiq 2F-85 class)
TOOL_COM_Z = 0.06
DISH_OFFSET_Z = 0.04       # held dish centre below the TCP along z6 [m]
DISH_RADIUS = 0.08         # held bowl modelled as a sphere [m]
ARMATURE = np.array([0.35, 0.35, 0.35, 0.06, 0.06, 0.06])   # reflected rotor inertia
VISCOUS = np.array([2.0, 2.0, 1.5, 0.3, 0.3, 0.3])
COULOMB = np.array([3.0, 3.0, 2.0, 0.5, 0.5, 0.5])

# --- UR5e datasheet limits ---
QD_MAX = np.full(6, np.pi)                      # 180 deg/s on every joint
TAU_MAX = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
Q_MIN = np.array([-2 * np.pi, -2 * np.pi, -0.95 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi])
Q_MAX = -Q_MIN

GRAVITY = 9.81


# ---------------------------------------------------------------- kinematics
def _dh_batch(theta: np.ndarray, i: int) -> np.ndarray:
    """Standard DH transform for joint i, theta shape (B,) -> (B, 4, 4)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
    B = theta.shape[0]
    T = np.zeros((B, 4, 4))
    T[:, 0, 0] = ct
    T[:, 0, 1] = -st * ca
    T[:, 0, 2] = st * sa
    T[:, 0, 3] = DH_A[i] * ct
    T[:, 1, 0] = st
    T[:, 1, 1] = ct * ca
    T[:, 1, 2] = -ct * sa
    T[:, 1, 3] = DH_A[i] * st
    T[:, 2, 1] = sa
    T[:, 2, 2] = ca
    T[:, 2, 3] = DH_D[i]
    T[:, 3, 3] = 1.0
    return T


def fk_frames(q: np.ndarray) -> np.ndarray:
    """Frames T0..T6 in the robot base frame. q (B,6) -> (B, 7, 4, 4)."""
    q = np.atleast_2d(q)
    B = q.shape[0]
    F = np.zeros((B, 7, 4, 4))
    F[:, 0] = np.eye(4)
    for i in range(6):
        F[:, i + 1] = F[:, i] @ _dh_batch(q[:, i], i)
    return F


def tcp_pose(q: np.ndarray):
    """TCP position (B,3) and tool z-axis (B,3)."""
    F = fk_frames(q)
    z6 = F[:, 6, :3, 2]
    p = F[:, 6, :3, 3] + TOOL_LEN * z6
    return p, z6


def jacobian(q: np.ndarray, point: str = "tcp", F: np.ndarray | None = None):
    """Geometric Jacobian of the TCP. Returns Jv (B,3,6), Jw (B,3,6)."""
    if F is None:
        F = fk_frames(q)
    Z = F[:, :6, :3, 2]            # joint axes z_0..z_5
    O = F[:, :6, :3, 3]            # joint origins
    p = F[:, 6, :3, 3] + TOOL_LEN * F[:, 6, :3, 2]
    Jv = np.cross(Z, p[:, None, :] - O).transpose(0, 2, 1)
    Jw = Z.transpose(0, 2, 1)
    return Jv, Jw


# ---------------------------------------------------------------- collision geometry
# Capsules approximate the UR5e housings. Physical shoulder/elbow offsets along the
# joint-2..4 axes (0.138 m, then -0.131 m) are collapsed in the DH model, so they are
# added back here. Radii are approximate (assumption).
SHOULDER_OFF = 0.138
FOREARM_OFF = 0.007
CAPSULE_NAMES = ["base", "shoulder", "upper_arm", "elbow", "forearm",
                 "wrist1", "wrist2", "wrist3_gripper"]
CAPSULE_RADII = np.array([0.075, 0.075, 0.060, 0.060, 0.050, 0.045, 0.045, 0.045])


def capsules(q: np.ndarray, F: np.ndarray | None = None):
    """Capsule endpoints for each link. Returns A (B,8,3), Bp (B,8,3), radii (8,)."""
    if F is None:
        F = fk_frames(q)
    o = F[:, :, :3, 3]
    z = F[:, :, :3, 2]
    tcp = o[:, 6] + TOOL_LEN * z[:, 6]
    Bn = F.shape[0]
    A = np.zeros((Bn, 8, 3))
    E = np.zeros((Bn, 8, 3))
    A[:, 0] = o[:, 0] + np.array([0, 0, 0.08]); E[:, 0] = o[:, 1]
    A[:, 1] = o[:, 1]; E[:, 1] = o[:, 1] + SHOULDER_OFF * z[:, 1]
    A[:, 2] = o[:, 1] + SHOULDER_OFF * z[:, 1]; E[:, 2] = o[:, 2] + SHOULDER_OFF * z[:, 1]
    A[:, 3] = o[:, 2] + SHOULDER_OFF * z[:, 1]; E[:, 3] = o[:, 2] + FOREARM_OFF * z[:, 2]
    A[:, 4] = o[:, 2] + FOREARM_OFF * z[:, 2]; E[:, 4] = o[:, 3] + FOREARM_OFF * z[:, 3]
    A[:, 5] = o[:, 3] + FOREARM_OFF * z[:, 3]; E[:, 5] = o[:, 4]
    A[:, 6] = o[:, 4]; E[:, 6] = o[:, 5]
    A[:, 7] = o[:, 5]; E[:, 7] = tcp - 0.02 * z[:, 6]
    return A, E, CAPSULE_RADII


def held_dish_center(q: np.ndarray, F: np.ndarray | None = None):
    if F is None:
        F = fk_frames(q)
    z6 = F[:, 6, :3, 2]
    return F[:, 6, :3, 3] + (TOOL_LEN + DISH_OFFSET_Z) * z6


# Self-collision pairs (capsule index pairs) that are physically possible.
SELF_PAIRS = [(7, 0), (7, 1), (7, 2), (6, 0), (6, 2), (5, 0)]
# held dish (index 8) vs these capsules
DISH_SELF = [0, 1, 2, 3]


# ---------------------------------------------------------------- dynamics
def _point_mass_jacobians(F: np.ndarray, payload: np.ndarray):
    """Linear Jacobians of every point mass. Returns Jv (B, nm, 6, 3), masses (B, nm)."""
    B = F.shape[0]
    Z = F[:, :6, :3, 2]
    O = F[:, :6, :3, 3]
    # link COMs (frame i+1 for link i)
    R = F[:, 1:, :3, :3]
    P = F[:, 1:, :3, 3]
    C = P + np.einsum("blij,lj->bli", R, LINK_COM)                 # (B,6,3)
    z6 = F[:, 6, :3, 2]
    tool_c = F[:, 6, :3, 3] + TOOL_COM_Z * z6
    dish_c = F[:, 6, :3, 3] + (TOOL_LEN + DISH_OFFSET_Z) * z6
    C = np.concatenate([C, tool_c[:, None], dish_c[:, None]], axis=1)  # (B,8,3)
    parent = np.array([0, 1, 2, 3, 4, 5, 5, 5])                     # last joint index moving it
    Jv = np.cross(Z[:, None, :, :], C[:, :, None, :] - O[:, None, :, :])   # (B,8,6,3)
    mask = (np.arange(6)[None, :] <= parent[:, None]).astype(float)        # (8,6)
    Jv = Jv * mask[None, :, :, None]
    m = np.concatenate([np.broadcast_to(LINK_MASS, (B, 6)),
                        np.full((B, 1), TOOL_MASS), payload[:, None]], axis=1)
    return Jv, m


def mass_matrix(q: np.ndarray, payload: np.ndarray):
    F = fk_frames(q)
    Jv, m = _point_mass_jacobians(F, payload)
    M = np.einsum("bljk,blmk,bl->bjm", Jv, Jv, m)
    M = M + np.diag(ARMATURE)[None]
    return M


def gravity_torque(q: np.ndarray, payload: np.ndarray):
    """Torque needed to hold the arm against gravity (B,6)."""
    F = fk_frames(q)
    Jv, m = _point_mass_jacobians(F, payload)
    return GRAVITY * np.einsum("blj,bl->bj", Jv[..., 2], m)


def coriolis(q: np.ndarray, qd: np.ndarray, payload: np.ndarray, eps: float = 1e-5):
    """C(q,qd) qd = dM/dt qd - 1/2 d/dq (qd^T M qd), by central differences."""
    B = q.shape[0]
    qs = [q + eps * qd, q - eps * qd]
    for k in range(6):
        e = np.zeros(6); e[k] = eps
        qs += [q + e, q - e]
    Qs = np.concatenate(qs, axis=0)
    Ms = mass_matrix(Qs, np.tile(payload, len(qs))).reshape(len(qs), B, 6, 6)
    Mdot = (Ms[0] - Ms[1]) / (2 * eps)
    term1 = np.einsum("bij,bj->bi", Mdot, qd)
    quad = np.einsum("nbij,bi,bj->nb", Ms[2:], qd, qd)              # (12,B)
    dquad = (quad[0::2] - quad[1::2]).T / (2 * eps)                  # (B,6)
    return term1 - 0.5 * dquad


def friction(qd: np.ndarray):
    return VISCOUS * qd + COULOMB * np.tanh(qd / 0.02)


# ---------------------------------------------------------------- numerical IK helper
def ik_numeric(p_target, a_target, q0, iters=200, tol=1e-5, damping=1e-2, w_rot=0.5):
    """Damped-least-squares IK for TCP position + tool-axis direction (yaw free)."""
    q = np.array(q0, float).reshape(1, 6)
    for _ in range(iters):
        F = fk_frames(q)
        z6 = F[:, 6, :3, 2]
        p = F[:, 6, :3, 3] + TOOL_LEN * z6
        Jv, Jw = jacobian(q, F=F)
        e_p = p_target - p[0]
        e_a = a_target - z6[0]
        A_a = -_skew(z6[0]) @ Jw[0]
        J = np.vstack([Jv[0], w_rot * A_a])
        e = np.concatenate([e_p, w_rot * e_a])
        if np.linalg.norm(e) < tol:
            break
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        q = q + dq[None]
    return q[0], float(np.linalg.norm(e_p)), float(np.linalg.norm(e_a))


def wrap(q, ref=None):
    """Wrap joint angles to the representation closest to ref (default 0)."""
    q = np.asarray(q, float)
    ref = np.zeros_like(q) if ref is None else np.asarray(ref, float)
    return ref + (q - ref + np.pi) % (2 * np.pi) - np.pi


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
