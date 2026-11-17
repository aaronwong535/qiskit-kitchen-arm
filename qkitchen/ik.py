"""Module 2 - per-step (0.01 s) inverse kinematics as a QUBO.

At every control step the next configuration is q_next = c + Delta, where c is the
constant-velocity prediction (warm start). Delta for joints 1-5 is encoded with 2
qubits per joint, Delta_i = delta * (2*x_msb + x_lsb - 1.5) in {-1.5,-0.5,0.5,1.5}*delta,
and refined over `rounds` rounds with delta shrinking 4x (16 levels per joint after two
rounds). Joint 6 is frozen: with a down-pointing, axisymmetric gripper it changes
neither the task (tool yaw is free) nor any clearance, so it would waste 2 qubits.

The objective, linearised around c, is a sum of squares, hence exactly a QUBO:
    tracking   |Jv D - e_p|^2 / s_p^2  +  |A_a D - e_a|^2 / s_a^2   (tolerance-weighted:
               loose while transiting, tight at grasp/place -> the arm may leave the
               nominal path when that buys clearance or saves motion)
    smoothness |D + c - q - v|^2 / s_acc^2      (deviation from constant velocity)
    energy     |W (c + D - q)|^2 / s_vel^2       (minimum joint turning)
    comfort    |(c + D - q_comf) / half|^2 / s_c^2
    clearance  sum_active (g_j . D - (d_tgt - d_j))^2 / s_col^2  (linearised distances)
Sampled bitstrings are re-ranked with the exact (non-linear) FK and collision model,
so linearisation error can never select a colliding configuration that a better
sample avoids.

Solvers: 'qaoa' (sampled QAOA with transferred angles), 'grid' (brute force over the
same QUBO - the exact discrete optimum), 'lsq' (continuous bounded least squares on the
same objective - strong classical baseline) and 'dls' (tracking-only damped least
squares - the naive baseline with no clearance terms).
"""
from __future__ import annotations

import warnings

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import lsq_linear

from . import env as E
from . import robot as R
from .qaoa import DiagonalProblem, all_bitstrings, qaoa_state, sample, walsh_coefficients


@dataclass
class IKConfig:
    joints: tuple = (0, 1, 2, 3, 4)
    bits: int = 2
    rounds: int = 2
    shrink: float = 4.0
    dt: float = 0.01
    sigma_p_precise: float = 0.001
    sigma_p_transit: float = 0.004
    sigma_a_precise: float = 0.01
    sigma_a_carry: float = 0.05
    sigma_a_free: float = 0.10
    sigma_acc: float = 0.02
    sigma_vel: float = 0.10
    w_energy: tuple = (1.0, 1.0, 0.7, 0.3, 0.3, 0.2)
    sigma_comf: float = 2.0
    d_target: float = 0.025
    d_hard: float = 0.008
    sigma_col: float = 0.004
    top_k: int = 6
    shots: int = 256
    delta_min: float = 1e-4
    delta_max: float = 0.012
    fd_eps: float = 1e-4


@dataclass
class StepInfo:
    cost: float = 0.0
    p_opt: float = np.nan          # QAOA probability of the grid optimum
    opt_sampled: bool = True
    gap: float = 0.0               # QUBO energy of chosen - grid optimum (round 1)
    delta: float = 0.0
    qubo: list = field(default_factory=list)


class StepIK:
    def __init__(self, kit: E.Kitchen, q_comf, half, cfg: IKConfig = IKConfig(),
                 method: str = "qaoa", angles=None, seed: int = 0, keep_qubos: bool = False):
        self.kit, self.cfg, self.method = kit, cfg, method
        self.q_comf, self.half = np.asarray(q_comf), np.asarray(half)
        self.act = np.array(cfg.joints)
        self.angles = angles
        self.rng = np.random.default_rng(seed)
        self.keep_qubos = keep_qubos
        nq = len(self.act) * cfg.bits
        X = all_bitstrings(nq).astype(float)
        G = np.zeros((len(self.act), nq))
        for i in range(len(self.act)):
            G[i, cfg.bits * i] = 1.0          # lsb
            G[i, cfg.bits * i + 1] = 2.0      # msb
        self.levels = X @ G.T - 1.5           # (2^nq, n_act) in {-1.5..1.5}
        self.wE = np.asarray(cfg.w_energy)

    # ------------------------------------------------------------ model
    def _sigmas(self, carrying, precise):
        c = self.cfg
        sp = c.sigma_p_precise if precise else c.sigma_p_transit
        sa = c.sigma_a_precise if precise else (c.sigma_a_carry if carrying else c.sigma_a_free)
        return sp, sa

    def linear_model(self, q, v, c, p_star, a_star, t, carrying, precise):
        """Stacked least-squares model ||A D - b||^2 for D on the active joints."""
        cfg, act = self.cfg, self.act
        Fc = R.fk_frames(c[None])
        z6 = Fc[0, 6, :3, 2]
        p = Fc[0, 6, :3, 3] + R.TOOL_LEN * z6
        Jv, Jw = R.jacobian(c[None], F=Fc)
        Jv, Jw = Jv[0][:, act], Jw[0][:, act]
        Aa = -R._skew(z6) @ Jw
        sp, sa = self._sigmas(carrying, precise)
        rows, rhs = [Jv / sp, Aa / sa], [(p_star - p) / sp, (a_star - z6) / sa]
        I = np.eye(len(act))
        rows.append(I / cfg.sigma_acc); rhs.append(-(c - q - v)[act] / cfg.sigma_acc)
        W = np.diag(self.wE[act]) / cfg.sigma_vel
        rows.append(W); rhs.append(-W @ (c - q)[act])
        Hc = np.diag(1.0 / (self.half[act] * cfg.sigma_comf))
        rows.append(Hc); rhs.append(-Hc @ (c - self.q_comf)[act])
        # clearance: finite-difference gradients of every pair distance
        Qs = np.repeat(c[None], len(act) + 1, axis=0)
        for k, j in enumerate(act):
            Qs[k + 1, j] += cfg.fd_eps
        D = E.pair_distances(self.kit, Qs, t, carrying)
        d0 = D[0]
        with np.errstate(invalid="ignore"):
            g = (D[1:] - d0) / cfg.fd_eps                     # (n_act, P)
        active = np.nonzero(d0 < cfg.d_target)[0]
        if len(active):
            rows.append(g[:, active].T / cfg.sigma_col)
            rhs.append((cfg.d_target - d0[active]) / cfg.sigma_col)
        return np.vstack(rows), np.concatenate(rhs), float(d0.min())

    def exact_cost(self, Q, q, v, p_star, a_star, t, carrying, precise):
        """True (non-linearised) objective for candidate configurations Q (B,6)."""
        cfg, act = self.cfg, self.act
        F = R.fk_frames(Q)
        z6 = F[:, 6, :3, 2]
        p = F[:, 6, :3, 3] + R.TOOL_LEN * z6
        sp, sa = self._sigmas(carrying, precise)
        J = np.sum(((p - p_star) / sp) ** 2, 1) + np.sum(((z6 - a_star) / sa) ** 2, 1)
        J += np.sum(((Q - q - v)[:, act] / cfg.sigma_acc) ** 2, 1)
        J += np.sum((self.wE[act] * (Q - q)[:, act] / cfg.sigma_vel) ** 2, 1)
        J += np.sum(((Q - self.q_comf)[:, act] / (self.half[act] * cfg.sigma_comf)) ** 2, 1)
        D = E.pair_distances(self.kit, Q, t, carrying, F=F)
        J += np.sum((np.maximum(cfg.d_target - D, 0.0) / cfg.sigma_col) ** 2, 1)
        dmin = D.min(1)
        J += 1e6 * (dmin < cfg.d_hard)
        J += 1e6 * np.any(np.abs(Q - q) > R.QD_MAX * cfg.dt + 1e-12, axis=1)
        J += 1e6 * np.any((Q < R.Q_MIN) | (Q > R.Q_MAX), axis=1)
        return J, dmin

    # ------------------------------------------------------------ step
    def step(self, q, v, p_star, a_star, t, carrying, precise):
        cfg, act = self.cfg, self.act
        info = StepInfo()
        c = q + v
        if self.method == "dls":
            return self._dls(q, v, p_star, a_star, carrying, precise), info
        delta = None
        for r in range(cfg.rounds):
            A, b, _ = self.linear_model(q, v, c, p_star, a_star, t, carrying, precise)
            if self.method == "lsq":
                vmax = R.QD_MAX[act] * cfg.dt
                lo = (q - c)[act] - vmax
                hi = (q - c)[act] + vmax
                d = lsq_linear(A, b, bounds=(np.minimum(lo, hi - 1e-9), hi)).x
                c = c.copy(); c[act] += d
                continue
            if delta is None:
                d_ls = np.linalg.lstsq(A, b, rcond=None)[0]
                delta = float(np.clip(np.abs(d_ls).max() / 1.2, cfg.delta_min, cfg.delta_max))
                info.delta = delta
            else:
                delta /= cfg.shrink
            Dgrid = delta * self.levels                          # (2^n, n_act)
            Eq = np.sum((Dgrid @ A.T - b) ** 2, axis=1)          # exact QUBO energies
            i_opt = int(np.argmin(Eq))
            if self.method == "grid":
                cand = np.argsort(Eq)[: cfg.top_k]
            else:   # qaoa
                coeffs = walsh_coefficients(Eq)
                scale = np.abs(coeffs[1:]).max()
                En = (Eq - coeffs[0]) / scale
                g, bta = self.angles
                pr = np.abs(qaoa_state(En, g, bta)) ** 2
                s = np.unique(sample(pr, cfg.shots, self.rng))
                cand = s[np.argsort(Eq[s])][: cfg.top_k]
                if r == 0:
                    info.p_opt = float(pr[i_opt])
                    info.opt_sampled = bool(i_opt in s)
            if self.keep_qubos and r == 0:
                info.qubo.append(Eq)
            Q = np.repeat(c[None], len(cand), axis=0)
            Q[:, act] += Dgrid[cand]
            J, _ = self.exact_cost(Q, q, v, p_star, a_star, t, carrying, precise)
            k = int(np.argmin(J))
            if r == 0:
                info.gap = float(Eq[cand[k]] - Eq[i_opt])
            c = Q[k]
        # safety projection onto the joint-speed box (a real drive's velocity limiter)
        vmax = R.QD_MAX * cfg.dt
        c = q + np.clip(c - q, -vmax, vmax)
        J, _ = self.exact_cost(c[None], q, v, p_star, a_star, t, carrying, precise)
        info.cost = float(J[0])
        return c, info

    def _dls(self, q, v, p_star, a_star, carrying, precise, lam=0.01):
        act = self.act
        c = q.copy()
        F = R.fk_frames(c[None])
        z6 = F[0, 6, :3, 2]
        p = F[0, 6, :3, 3] + R.TOOL_LEN * z6
        Jv, Jw = R.jacobian(c[None], F=F)
        J = np.vstack([Jv[0][:, act], 0.3 * (-R._skew(z6) @ Jw[0][:, act])])
        e = np.concatenate([p_star - p, 0.3 * (a_star - z6)])
        d = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        vmax = R.QD_MAX[act] * self.cfg.dt
        c[act] += np.clip(d, -vmax, vmax)
        return c


def run_ik(kit, solver: StepIK, q0, t, P, carrying, precise, a_star=np.array([0, 0, -1.0]),
           progress=False):
    """Run per-step IK along a sampled TCP reference. Returns q (N,6) and step infos."""
    N = len(t)
    Q = np.zeros((N, 6))
    Q[0] = q0
    v = np.zeros(6)
    infos = [StepInfo()]
    for k in range(N - 1):
        qn, info = solver.step(Q[k], v, P[k + 1], a_star, t[k + 1], bool(carrying[k + 1]),
                               bool(precise[k + 1]))
        v = qn - Q[k]
        Q[k + 1] = qn
        infos.append(info)
        if progress and k % 500 == 0:
            print(f"  step {k}/{N}", flush=True)
    return Q, infos


def evaluate_motion(kit, Q, t, P, carrying, a_star=np.array([0, 0, -1.0]), dt=0.01):
    """Metrics of a joint trajectory against its TCP reference."""
    F = R.fk_frames(Q)
    z6 = F[:, 6, :3, 2]
    p = F[:, 6, :3, 3] + R.TOOL_LEN * z6
    pos_err = np.linalg.norm(p - P, axis=1)
    tilt = np.degrees(np.arccos(np.clip(z6 @ a_star, -1, 1)))
    dmin = np.full(len(t), np.inf)
    for c in (False, True):
        m = carrying == c
        if m.any():
            dmin[m] = E.min_clearance(kit, Q[m], t[m], c, F=F[m])
    dq = np.diff(Q, axis=0)
    w = np.array([1.0, 1.0, 0.7, 0.3, 0.3, 0.2])
    return dict(
        pos_err_rms_mm=1e3 * float(np.sqrt(np.mean(pos_err ** 2))),
        pos_err_max_mm=1e3 * float(pos_err.max()),
        tilt_max_deg=float(tilt.max()),
        min_clearance_mm=1e3 * float(dmin.min()),
        steps_below_8mm=int(np.sum(dmin < 0.008)),
        steps_in_collision=int(np.sum(dmin < 0.0)),
        joint_travel_rad=float(np.abs(dq).sum()),
        weighted_travel=float(np.sum(np.abs(dq) * w)),
        max_joint_speed_dps=float(np.degrees(np.abs(dq).max() / dt)),
        max_joint_acc=float(np.abs(np.diff(dq, axis=0)).max() / dt ** 2),
    ), pos_err, tilt, dmin
