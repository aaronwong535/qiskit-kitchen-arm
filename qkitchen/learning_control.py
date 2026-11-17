"""Module 4 - Slotine-Li adaptive control + learned residual dynamics (quantum-kernel GP).

Plant (truth): UR5e rigid-body dynamics with the true payload, true rotor inertia,
viscous + Coulomb friction, and two effects the controller's model does not contain:
cable drag (a configuration-dependent spring on joints 1-3) and gear ripple (a
position-periodic torque on every joint). These are modelling assumptions chosen to be
typical of real arms; they are what a regressor-based adaptive law cannot represent.

Controller (Slotine & Li 1987), with s = qd - qd_r, qd_r = qd_d - Lambda e:
    tau = M^(q) qdd_r + C^(q, qd) qd_r + g^(q) + Y(q, qd, qd_r, qdd_r) theta^ - K_D s + r^(x)
    theta^' = -Gamma Y^T s
M^, C^, g^ use the nominal model (no payload, rotor inertia 20 % low, no friction).
theta^ holds the payload mass (baseline) and optionally 12 friction coefficients.
r^(x) is the learned residual torque: a GP per joint with either an
RBF kernel or a quantum fidelity kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import robot as R
from .control import reference_derivatives, smooth_reference

GVEC = np.array([0.0, 0.0, R.GRAVITY])


@dataclass
class Unmodelled:
    """Effects present in the plant but absent from the controller's model."""
    cable_k: np.ndarray = field(default_factory=lambda: np.array([4.0, 5.0, 2.5, 0.0, 0.0, 0.0]))
    cable_q0: np.ndarray = field(default_factory=lambda: np.zeros(6))
    ripple_a: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.5, 0.15, 0.15, 0.1]))
    ripple_f: float = 8.0
    ripple_phase: np.ndarray = field(default_factory=lambda: np.array([0.3, 1.1, 2.0, 0.7, 2.6, 1.4]))
    armature_nominal_scale: float = 0.8

    def torque(self, q):
        """Load torque the motors must overcome (added to the plant's equation)."""
        return self.cable_k * (q - self.cable_q0) + self.ripple_a * np.sin(self.ripple_f * q + self.ripple_phase)


# ---------------------------------------------------------------- model terms
def model_terms(q, qd, v, a, payload, armature, eps=1e-6):
    """M(q) a + C(q, qd) v + g(q) for point-mass links (+ constant armature), batched.

    C is the Christoffel-consistent matrix sum_l m_l J_l^T Jdot_l (M_dot - 2C skew).
    Also returns the payload regressor Y_p = J_d^T (J_d a + Jdot_d v + g) and J_d.
    """
    B = q.shape[0]
    Q3 = np.concatenate([q, q + eps * qd, q - eps * qd], 0)
    F = R.fk_frames(Q3)
    Jv, m = R._point_mass_jacobians(F, np.concatenate([payload] * 3))
    J, Jp, Jm = Jv[:B], Jv[B:2 * B], Jv[2 * B:]                       # (B, 8, 6, 3)
    m = m[:B]
    Jdot = (Jp - Jm) / (2 * eps)
    lin_a = np.einsum("bljk,bj->blk", J, a)                             # J a  (B,8,3)
    lin_v = np.einsum("bljk,bj->blk", Jdot, v)                          # Jdot v
    tau = np.einsum("bljk,blk,bl->bj", J, lin_a + lin_v + GVEC, m) + armature * a
    Jd = J[:, 7]                                                        # dish point
    Yp = np.einsum("bjk,bk->bj", Jd, lin_a[:, 7] + lin_v[:, 7] + GVEC)
    return tau, Yp


# ---------------------------------------------------------------- features for the GP
QD_SCALE, QDD_SCALE = 3.0, 15.0


def joint_features(q, qd, qdd):
    """Per-joint GP inputs in [0,1]^4: sin q, cos q, velocity, acceleration. (B,6,4)"""
    return np.clip(np.stack([(np.sin(q) + 1) / 2, (np.cos(q) + 1) / 2,
                             qd / (2 * QD_SCALE) + 0.5, qdd / (2 * QDD_SCALE) + 0.5], -1), 0, 1)


# ---------------------------------------------------------------- simulation
@dataclass
class SLConfig:
    dt_phys: float = 0.001
    ctrl_every: int = 2
    ref_dt: float = 0.01
    delay_steps: int = 1
    enc_noise: float = 2e-5
    acc_noise: float = 0.3            # rad/s^2, std of the logged acceleration estimate
    vel_filter_hz: float = 60.0
    lam: float = 15.0
    kd_scale: float = 30.0            # K_D = kd_scale * nominal joint inertia
    gamma_m: float = 10.0             # payload adaptation gain
    gamma_f: float = 0.5              # friction adaptation gain


@dataclass
class Controller:
    name: str
    adapt_payload: bool = True
    adapt_friction: bool = False
    oracle: bool = False
    residual: object = None           # callable (B,6,4) features -> (B,6) torque, or None


def simulate_sl(q_ref, carrying, payload_true, controllers, unm: Unmodelled, cfg=SLConfig(),
                seed=0, log_residuals=False, q_init=None):
    """Closed-loop run of several controllers in one batch over a 100 Hz reference."""
    B, N = len(controllers), len(q_ref)
    rng = np.random.default_rng(seed)
    qd_ref, qdd_ref = reference_derivatives(q_ref)
    arm_true = R.ARMATURE
    arm_nom = unm.armature_nominal_scale * R.ARMATURE
    from .control import _dyn_terms
    M0, _ = _dyn_terms(np.atleast_2d(q_ref[0]), np.zeros(1))
    KD = cfg.kd_scale * (np.diag(M0[0]) - R.ARMATURE + arm_nom)
    Lam = np.full(6, cfg.lam)
    ad_p = np.array([c.adapt_payload for c in controllers])
    ad_f = np.array([c.adapt_friction for c in controllers])
    orc = np.array([c.oracle for c in controllers])

    q = np.repeat((q_ref[:1] if q_init is None else q_init[None]), B, 0).astype(float)
    qd = np.zeros((B, 6)); qd_f = np.zeros((B, 6)); q_prev_meas = q.copy()
    m_hat = np.zeros(B); f_hat = np.zeros((B, 12))
    res_ff = np.zeros((B, 6))
    tau_apply = np.zeros((B, 6)); pipe = [None] * cfg.delay_steps
    qdd_true = np.zeros((B, 6))
    sub = int(round(cfg.ref_dt / cfg.dt_phys))
    dtc = cfg.dt_phys * cfg.ctrl_every
    alpha = 1 - np.exp(-2 * np.pi * cfg.vel_filter_hz * dtc)
    Q_tr = np.zeros((B, N, 6)); Q_tr[:, 0] = q
    TAU_tr = np.zeros((B, N, 6)); MH = np.zeros((B, N))
    J_chat = np.zeros(B); tau_prev = np.zeros((B, 6)); n_sat = np.zeros(B)
    logs = dict(X=[], Y=[]) if log_residuals else None
    was_carrying = bool(carrying[0])

    for k in range(N - 1):
        pl = np.full(B, payload_true[k])
        if was_carrying and not carrying[k]:
            m_hat[:] = 0.0                      # release is a known event: reset estimate
        was_carrying = bool(carrying[k])
        for s in range(sub):
            if s % cfg.ctrl_every == 0:
                fr = s / sub
                qdes = q_ref[k] + fr * (q_ref[k + 1] - q_ref[k])
                qd_des = qd_ref[k] + fr * (qd_ref[k + 1] - qd_ref[k])
                qdd_des = qdd_ref[k] + fr * (qdd_ref[k + 1] - qdd_ref[k])
                q_m = q + cfg.enc_noise * rng.standard_normal(q.shape)
                vel = (q_m - q_prev_meas) / dtc
                q_prev_meas = q_m
                qd_f = qd_f + alpha * (vel - qd_f)
                e = q_m - qdes
                ed = qd_f - qd_des
                qd_r = qd_des - Lam * e
                qdd_r = qdd_des - Lam * ed
                sv = qd_f - qd_r
                # nominal model + payload regressor (payload term uses theta^)
                tau_nom, Yp = model_terms(q_m, qd_f, qd_r, qdd_r, np.zeros(B), arm_nom)
                Yf = np.concatenate([qd_f, np.tanh(qd_f / 0.02)], 1)             # diag regressor
                tau = tau_nom + Yp * m_hat[:, None] + Yf[:, :6] * f_hat[:, :6] \
                    + Yf[:, 6:] * f_hat[:, 6:] - KD * sv + res_ff
                if orc.any():
                    tau_o, _ = model_terms(q_m[orc], qd_f[orc], qd_r[orc], qdd_r[orc],
                                           pl[orc], arm_true)
                    tau[orc] = tau_o + R.friction(qd_f[orc]) + unm.torque(q_m[orc]) - KD * sv[orc]
                # Slotine-Li adaptation (Euler), projected to physical values
                m_hat = np.where(ad_p & (carrying[k] > 0),
                                 np.maximum(m_hat - cfg.gamma_m * dtc * np.sum(Yp * sv, 1), 0), m_hat)
                f_upd = f_hat - cfg.gamma_f * dtc * Yf * np.concatenate([sv, sv], 1)
                f_hat = np.where(ad_f[:, None], np.maximum(f_upd, 0), f_hat)
                tau_new = np.clip(tau, -R.TAU_MAX, R.TAU_MAX)
                n_sat += np.any(np.abs(tau) > R.TAU_MAX, 1)
                J_chat += dtc * np.sum(((tau_new - tau_prev) / R.TAU_MAX) ** 2, 1)
                tau_prev = tau_new
                pipe.append(tau_new)
                old = pipe.pop(0)
                tau_apply = tau_new if old is None else old
                if s == 0:
                    c_term = R.coriolis(q, qd, pl)
            M, g = _true_terms(q, pl, arm_true)
            rhs = tau_apply - c_term - g - R.friction(qd) - unm.torque(q)
            qdd_true = np.linalg.solve(M, rhs[..., None])[..., 0]
            qd = qd + qdd_true * cfg.dt_phys
            q = q + qd * cfg.dt_phys
        # 100 Hz: residual feedforward update + data logging
        qr_next = min(k + 1, N - 1)
        if any(c.residual is not None for c in controllers):
            Xf = joint_features(q, qd_f, np.repeat(qdd_ref[qr_next][None], B, 0))
            for b, c in enumerate(controllers):
                if c.residual is not None:
                    res_ff[b] = c.residual(Xf[b:b + 1])[0]
        if log_residuals:
            qdd_meas = qdd_true + cfg.acc_noise * rng.standard_normal(qdd_true.shape)
            tau_mod, Ypm = model_terms(q_m, qd_f, qd_f, qdd_meas, np.zeros(B), arm_nom)
            r = tau_apply - tau_mod - Ypm * m_hat[:, None]
            logs["X"].append(joint_features(q_m, qd_f, qdd_meas)[0])
            logs["Y"].append(r[0])
        Q_tr[:, k + 1] = q
        TAU_tr[:, k + 1] = tau_apply
        MH[:, k + 1] = m_hat
    out = dict(Q=Q_tr, TAU=TAU_tr, m_hat=MH, chatter=J_chat / ((N - 1) * cfg.ref_dt), sat=n_sat)
    if log_residuals:
        out["X"] = np.array(logs["X"]); out["Y"] = np.array(logs["Y"])
    return out


def _true_terms(q, payload, armature):
    F = R.fk_frames(q)
    Jv, m = R._point_mass_jacobians(F, payload)
    M = np.einsum("bljk,blmk,bl->bjm", Jv, Jv, m) + np.diag(armature)[None]
    g = R.GRAVITY * np.einsum("blj,bl->bj", Jv[..., 2], m)
    return M, g


# ---------------------------------------------------------------- residual GPs
class JointGP:
    """One GP per joint over 4 features; kernel 'qk' (fidelity) or 'rbf' (ARD)."""

    def __init__(self, kind="qk", n_support=250, seed=0):
        self.kind, self.n_support, self.seed = kind, n_support, seed
        self.models = []

    def fit(self, X, Y):
        """X (T, 6, 4), Y (T, 6)."""
        from .qml import ZZKernel
        rng = np.random.default_rng(self.seed)
        idx = np.sort(rng.choice(len(X), min(self.n_support, len(X)), replace=False))
        self.models = []
        for j in range(6):
            Xj, yj = X[idx, j], Y[idx, j]
            if self.kind == "rbf":
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
                k = ConstantKernel(1.0) * RBF(np.full(4, 0.3), (1e-2, 1e2)) + WhiteKernel(1e-2)
                gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2,
                                              random_state=0).fit(Xj, yj)
                self.models.append(("rbf", gp))
            else:   # feature-map depth chosen per joint by marginal likelihood
                fits = [fit_qk_ard(Xj, yj, reps=r) for r in (1, 2, 3)]
                self.models.append(("qk", min(fits, key=lambda m: m["nlml"])))
        return self

    def predict(self, Xb):
        """Xb (B, 6, 4) -> (B, 6) residual torque."""
        out = np.zeros((Xb.shape[0], 6))
        for j, (kind, m) in enumerate(self.models):
            if kind == "rbf":
                out[:, j] = m.predict(Xb[:, j])
            else:
                Ks = m["amp"] * np.abs(m["zk"].states(Xb[:, j]).conj() @ m["S"].T) ** 2
                out[:, j] = m["mu"] + m["sd"] * (Ks @ m["alpha"])
        return out

    def hyper(self):
        return [dict(bandwidth=[round(float(x), 3) for x in m["bw"]], noise=round(float(m["noise"]), 4),
                     amplitude=round(float(m["amp"]), 3), reps=m["zk"].reps)
                if k == "qk" else dict(kernel=str(m.kernel_))
                for k, m in self.models]


def fit_qk_ard(Xj, yj, reps=2):
    """Quantum-kernel GP with one bandwidth per feature (the analogue of ARD length
    scales), plus amplitude and noise, all by maximising the log marginal likelihood."""
    from scipy.optimize import minimize
    from .qml import ZZKernel
    mu, sd = yj.mean(), yj.std() + 1e-9
    yn = (yj - mu) / sd
    nf = Xj.shape[1]

    def nlml(theta):
        bw, amp, nz = np.exp(theta[:nf]), np.exp(theta[nf]), np.exp(theta[nf + 1])
        if np.any(bw > 3.0) or nz < 1e-4:
            return 1e10
        zk = ZZKernel(nf, reps=reps, bandwidth=bw)
        S = zk.states(Xj)
        Ky = amp * np.abs(S.conj() @ S.T) ** 2 + nz * np.eye(len(Xj))
        try:
            L = np.linalg.cholesky(Ky)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        return 0.5 * yn @ a + np.sum(np.log(np.diag(L)))

    best = None
    for bw0 in (0.1, 0.2, 0.4, 0.8):
        for nz0 in (0.03, 0.1):
            th0 = np.r_[np.log(np.full(nf, bw0)), 0.0, np.log(nz0)]
            v = nlml(th0)
            if best is None or v < best[0]:
                best = (v, th0)
    r = minimize(nlml, best[1], method="Nelder-Mead",
                 options=dict(maxiter=600, xatol=1e-3, fatol=1e-4))
    th = r.x
    bw, amp, nz = np.exp(th[:nf]), np.exp(th[nf]), np.exp(th[nf + 1])
    zk = ZZKernel(nf, reps=reps, bandwidth=bw)
    S = zk.states(Xj)
    Ky = amp * np.abs(S.conj() @ S.T) ** 2 + nz * np.eye(len(Xj))
    L = np.linalg.cholesky(Ky)
    a = np.linalg.solve(L.T, np.linalg.solve(L, yn))
    return dict(zk=zk, S=S, alpha=a, mu=mu, sd=sd, bw=bw, noise=nz, amp=amp, nlml=float(r.fun))
