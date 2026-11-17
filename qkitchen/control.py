"""Module 3 - joint PID with gain scheduling by a quantum-kernel Gaussian process.

Plant: UR5e rigid-body dynamics (robot.py) with armature, viscous + Coulomb friction,
torque limits, encoder noise and an unmodelled payload (the bowl). Physics at 1 kHz,
PID at 500 Hz, reference and gain updates at 100 Hz (the 0.01 s IK step).

Controller per joint:  tau = g_nom(q) + kp*Kp e + kp*Ki int(e) + kd*Kd de/dt
g_nom is the gravity model *without* payload; (kp, kd) are the multipliers the
scheduler outputs every 10 ms from features of the current motion:
  f1 reach of the wrist centre, f2 joint speed, f3 joint acceleration, f4 payload estimate.
Labels come from short simulated rollouts over a grid of multipliers (cost = tracking
error + effort + chatter + saturation); the GP regresses log-multipliers on features.
Kernels: quantum fidelity kernel |<phi(x)|phi(x')>|^2 with a ZZ feature map and a tuned
bandwidth (Shaydulin & Wild 2022) vs a classical ARD-RBF kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import robot as R

KP_GRID = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
KD_GRID = np.array([0.4, 0.6, 0.8, 1.0])


@dataclass
class SimConfig:
    dt_phys: float = 0.001
    ctrl_every: int = 2           # PID at 500 Hz
    ref_dt: float = 0.01          # IK / gain-update step
    enc_noise: float = 2e-5       # rad
    vel_filter_hz: float = 60.0
    delay_steps: int = 1          # one PID period of actuation/communication delay
    err_scale: float = 1e-3       # 1 mrad RMS tracking error -> cost 1 per joint
    w_eff: float = 3.0
    w_chat: float = 30000.0       # torque chatter (noise amplification / vibration)
    w_sat: float = 50.0


def nominal_gains(q0, wn_prox=2 * np.pi * 10, wn_wrist=2 * np.pi * 12, zeta=0.9):
    M = R.mass_matrix(np.atleast_2d(q0), np.zeros(1))[0]
    I = np.diag(M)
    wn = np.array([wn_prox] * 3 + [wn_wrist] * 3)
    Kp = I * wn ** 2
    Kd = 2 * zeta * I * wn
    Ki = Kp * wn / 8
    return Kp, Ki, Kd


def _dyn_terms(q, payload):
    """Mass matrix and gravity torque from one FK pass (batched)."""
    F = R.fk_frames(q)
    Jv, m = R._point_mass_jacobians(F, payload)
    M = np.einsum("bljk,blmk,bl->bjm", Jv, Jv, m) + np.diag(R.ARMATURE)[None]
    g = R.GRAVITY * np.einsum("blj,bl->bj", Jv[..., 2], m)
    return M, g


def simulate(q_ref, qd_ref, payload, kp_mult, kd_mult, gains, cfg=SimConfig(),
             q_init=None, qd_init=None, integ_init=None, seed=0, record=False, gain_fn=None):
    """Batched closed-loop simulation.

    q_ref, qd_ref : (N, 6) reference at ref_dt
    payload       : (N,) or (B, N) true payload mass [kg] over time
    kp_mult, kd_mult : (B, N) multipliers per reference step
    gain_fn       : optional closed-loop policy, called at every reference step k as
                    gain_fn(k, obs) -> (kp (B,), kd (B,)); obs holds the measured joint
                    error and error rate. Overrides kp_mult/kd_mult.
    Returns dict of per-batch costs, per-step running cost, and traces if record.
    """
    Kp, Ki, Kd = gains
    B, N = kp_mult.shape
    kp_mult = np.array(kp_mult, float); kd_mult = np.array(kd_mult, float)
    step_cost = np.zeros((B, N))
    e = np.zeros((B, 6)); ed = np.zeros((B, 6))
    sub = int(round(cfg.ref_dt / cfg.dt_phys))
    rng = np.random.default_rng(seed)
    payload = np.broadcast_to(np.atleast_2d(payload), (B, N))
    q = np.repeat(q_ref[:1], B, 0) if q_init is None else np.array(q_init, float).copy()
    qd = np.repeat(qd_ref[:1], B, 0) if qd_init is None else np.array(qd_init, float).copy()
    integ = np.zeros((B, 6)) if integ_init is None else np.array(integ_init, float).copy()
    q_meas_prev = q.copy()
    qd_f = qd.copy()
    alpha = 1 - np.exp(-2 * np.pi * cfg.vel_filter_hz * cfg.dt_phys * cfg.ctrl_every)
    tau = np.zeros((B, 6))
    tau_prev = np.zeros((B, 6))
    pipe = [None] * cfg.delay_steps
    J_err = np.zeros(B); J_eff = np.zeros(B); J_chat = np.zeros(B); n_sat = np.zeros(B)
    zeros = np.zeros(B)
    if record:
        Q_tr = np.zeros((B, N, 6)); TAU_tr = np.zeros((B, N, 6))
        QD_tr = np.zeros((B, N, 6)); INT_tr = np.zeros((B, N, 6))
    alive = np.ones(B, bool)
    for k in range(N - 1):
        pl = payload[:, k]
        if gain_fn is not None:
            kp_mult[:, k], kd_mult[:, k] = gain_fn(k, dict(e=e, ed=ed))
        snap = (J_err.copy(), J_eff.copy(), J_chat.copy(), n_sat.copy())
        for s in range(sub):
            if s % cfg.ctrl_every == 0:
                frac = s / sub
                qr = q_ref[k] + frac * (q_ref[k + 1] - q_ref[k])
                qdr = qd_ref[k] + frac * (qd_ref[k + 1] - qd_ref[k])
                q_meas = q + cfg.enc_noise * rng.standard_normal(q.shape)
                vel = (q_meas - q_meas_prev) / (cfg.dt_phys * cfg.ctrl_every)
                q_meas_prev = q_meas
                qd_f = qd_f + alpha * (vel - qd_f)
                e = qr - q_meas
                ed = qdr - qd_f
                _, g_nom = _dyn_terms(q_meas, zeros)
                integ = np.clip(integ + e * cfg.dt_phys * cfg.ctrl_every, -0.05, 0.05)
                kp = kp_mult[:, k:k + 1]
                kd = kd_mult[:, k:k + 1]
                fb = kp * (Kp * e + Ki * integ) + kd * Kd * ed
                tau_cmd = g_nom + fb
                tau_new = np.clip(tau_cmd, -R.TAU_MAX, R.TAU_MAX)
                sat = np.any(np.abs(tau_cmd) > R.TAU_MAX, axis=1)
                w = cfg.dt_phys * cfg.ctrl_every
                e_true = qr - q
                J_err += w * np.sum((e_true / cfg.err_scale) ** 2, 1)
                J_eff += w * np.sum((fb / R.TAU_MAX) ** 2, 1)
                J_chat += w * np.sum(((tau_new - tau_prev) / R.TAU_MAX) ** 2, 1)
                n_sat += sat
                tau_prev = tau_new
                if cfg.delay_steps:
                    pipe.append(tau_new)
                    old = pipe.pop(0)
                    tau = tau_new if old is None else old
                else:
                    tau = tau_new
                if s == 0:
                    c_term = R.coriolis(q, qd, pl)
            M, g = _dyn_terms(q, pl)
            qdd = np.linalg.solve(M, (tau - c_term - g - R.friction(qd))[..., None])[..., 0]
            qd = qd + qdd * cfg.dt_phys
            q = q + qd * cfg.dt_phys
        step_cost[:, k] = ((J_err - snap[0]) + cfg.w_eff * (J_eff - snap[1])
                           + cfg.w_chat * (J_chat - snap[2])) / cfg.ref_dt \
            + cfg.w_sat * (n_sat - snap[3]) / (sub / cfg.ctrl_every)
        # divergence guard: an unstable gain set is frozen on the reference and flagged
        bad = ~np.all(np.isfinite(q), 1) | (np.abs(q - q_ref[k + 1]).max(1) > 0.3)
        if bad.any():
            step_cost[bad, k] += 1e3
            alive &= ~bad
            q[bad], qd[bad] = q_ref[k + 1], qd_ref[k + 1]
            integ[bad] = 0.0; qd_f[bad] = qd_ref[k + 1]; q_meas_prev[bad] = q_ref[k + 1]
            tau_prev[bad] = 0.0
        if record:
            Q_tr[:, k + 1] = q
            TAU_tr[:, k + 1] = tau
            QD_tr[:, k + 1] = qd
            INT_tr[:, k + 1] = integ
    T = (N - 1) * cfg.ref_dt
    cost = J_err / T + cfg.w_eff * J_eff / T \
        + cfg.w_chat * J_chat / T + cfg.w_sat * n_sat / (N * sub / cfg.ctrl_every)
    cost = np.where(alive, cost, 1e6)
    out = dict(cost=cost, err=J_err / T, eff=J_eff / T, chat=J_chat / T, sat=n_sat,
               q=q, qd=qd, integ=integ, alive=alive, step_cost=step_cost,
               kp=kp_mult, kd=kd_mult)
    if record:
        Q_tr[:, 0] = q_ref[0] if q_init is None else q_init
        QD_tr[:, 0] = qd_ref[0] if qd_init is None else qd_init
        out.update(Q=Q_tr, TAU=TAU_tr, QD=QD_tr, INT=INT_tr)
    return out


# ---------------------------------------------------------------- features
def features(q_ref, qd_ref, qdd_ref, payload_est):
    """f in [0,1]^4: wrist reach, joint speed, joint acceleration, payload estimate."""
    F = R.fk_frames(q_ref)
    p4 = F[:, 4, :3, 3]
    reach = np.linalg.norm(p4[:, :2], axis=1) / 0.8
    speed = np.linalg.norm(qd_ref[:, :3], axis=1) / 2.0
    acc = np.linalg.norm(qdd_ref[:, :3], axis=1) / 8.0
    pay = np.asarray(payload_est, float) / 1.0
    return np.clip(np.stack([reach, speed, acc, pay], axis=1), 0.0, 1.0)


def smooth_reference(Q, window=9, order=3):
    """Savitzky-Golay smoothing of the 100 Hz IK output (removes grid quantisation)."""
    from scipy.signal import savgol_filter
    return savgol_filter(Q, window, order, axis=0, mode="interp")


def reference_derivatives(Q, dt=0.01):
    qd = np.gradient(Q, dt, axis=0)
    qdd = np.gradient(qd, dt, axis=0)
    return qd, qdd


# ---------------------------------------------------------------- labels
def label_windows(Q, carrying, gains, windows, payload_true, cfg=SimConfig(), seed=0,
                  temp=0.05):
    """Grid-search the best (kp, kd) multipliers for each window -> soft log-labels.

    On-policy: one nominal-gain rollout of the whole task supplies the realistic state
    (tracking error, velocity, integrator) at each window start; every gain pair is then
    rolled out from that state.
    """
    qd_ref, qdd_ref = reference_derivatives(Q)
    KP, KD = np.meshgrid(KP_GRID, KD_GRID, indexing="ij")
    KP, KD = KP.ravel(), KD.ravel()
    B = len(KP)
    N = len(Q)
    nom = simulate(Q, qd_ref, payload_true, np.ones((1, N)), np.ones((1, N)), gains, cfg,
                   seed=seed, record=True)
    X, Y, Jall = [], [], []
    for w, (k0, k1) in enumerate(windows):
        qr, qdr = Q[k0:k1], qd_ref[k0:k1]
        pl = payload_true[k0:k1]
        n = k1 - k0
        res = simulate(qr, qdr, pl, np.repeat(KP[:, None], n, 1), np.repeat(KD[:, None], n, 1),
                       gains, cfg, q_init=np.repeat(nom["Q"][0, k0][None], B, 0),
                       qd_init=np.repeat(nom["QD"][0, k0][None], B, 0),
                       integ_init=np.repeat(nom["INT"][0, k0][None], B, 0) / KP[:, None],
                       seed=seed + w)
        J = res["cost"]
        wts = np.exp(-(J - J.min()) / (temp * J.min() + 1e-9))
        wts /= wts.sum()
        y = np.array([wts @ np.log(KP), wts @ np.log(KD)])
        mid = (k0 + k1) // 2
        f = features(Q[k0:k1], qdr, qdd_ref[k0:k1], 0.6 * carrying[k0:k1]).mean(0)
        X.append(f); Y.append(y); Jall.append(J)
    return np.array(X), np.array(Y), np.array(Jall), (KP, KD)


# ---------------------------------------------------------------- kernels + GP
class QuantumKernel:
    """Fidelity kernel with Qiskit's ZZ feature map; statevectors cached for speed."""

    def __init__(self, n_features=4, reps=2, bandwidth=0.5, entanglement="linear"):
        from qiskit.circuit.library import zz_feature_map
        self.fmap = zz_feature_map(n_features, reps=reps, entanglement=entanglement)
        self.params = list(self.fmap.parameters)
        self.bandwidth = bandwidth

    def encode(self, f):
        return self.bandwidth * np.pi * np.asarray(f)

    def statevectors(self, Xf):
        from qiskit.quantum_info import Statevector
        return np.array([Statevector(self.fmap.assign_parameters(
            dict(zip(self.params, self.encode(x))))).data for x in np.atleast_2d(Xf)])

    def gram(self, A, B=None, SA=None, SB=None):
        SA = self.statevectors(A) if SA is None else SA
        SB = SA if B is None and SB is None else (self.statevectors(B) if SB is None else SB)
        return np.abs(SA.conj() @ SB.T) ** 2

    def compute_uncompute(self, x, y):
        """Circuit whose all-zeros probability equals k(x, y) (for samplers/hardware)."""
        a = self.fmap.assign_parameters(dict(zip(self.params, self.encode(x))))
        b = self.fmap.assign_parameters(dict(zip(self.params, self.encode(y))))
        qc = a.compose(b.inverse())
        qc.measure_all()
        return qc


class GP:
    """Exact GP regression with a precomputed kernel; multi-output with shared kernel."""

    def __init__(self, noise=1e-2, signal=1.0):
        self.noise, self.signal = noise, signal

    def fit(self, K, Y):
        self.m0 = Y.mean(0)
        Ky = self.signal * K + self.noise * np.eye(len(K))
        self.L = np.linalg.cholesky(Ky)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y - self.m0))
        self.Y = Y
        return self

    def log_marginal(self):
        n = len(self.Y)
        a = self.alpha
        r = self.Y - self.m0
        return float(-0.5 * np.sum(r * a) - self.Y.shape[1] * np.sum(np.log(np.diag(self.L)))
                     - 0.5 * n * self.Y.shape[1] * np.log(2 * np.pi))

    def predict(self, Ks, kss=None):
        mu = self.m0 + self.signal * Ks @ self.alpha
        if kss is None:
            return mu
        v = np.linalg.solve(self.L, self.signal * Ks.T)
        var = self.signal * kss - np.sum(v * v, 0)
        return mu, np.maximum(var, 1e-12)


def fit_qkgp(Xtr, Ytr, bandwidths=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0),
             noises=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0), reps=2):
    best = None
    sig = float(np.mean(np.var(Ytr, 0))) + 1e-9
    for bw in bandwidths:
        qk = QuantumKernel(Xtr.shape[1], reps=reps, bandwidth=bw)
        S = qk.statevectors(Xtr)
        K = np.abs(S.conj() @ S.T) ** 2
        for nz in noises:
            gp = GP(noise=nz * sig, signal=sig).fit(K, Ytr)
            lm = gp.log_marginal()
            if best is None or lm > best[0]:
                best = (lm, bw, nz, qk, S, gp)
    lm, bw, nz, qk, S, gp = best
    return dict(kernel=qk, S_train=S, gp=gp, bandwidth=bw, noise=nz, log_ml=lm,
                offdiag_mean=float(np.mean((np.abs(S.conj() @ S.T) ** 2)[~np.eye(len(S), dtype=bool)])))


def fit_rbfgp(Xtr, Ytr):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    k = ConstantKernel(1.0) * RBF(length_scale=np.full(Xtr.shape[1], 0.3),
                                  length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(1e-2)
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=5,
                                    random_state=0).fit(Xtr, Ytr)


def soft_labels(J, KP, KD, mask=None, temp=0.05):
    """Soft-argmin log-gain labels per window, restricted to the safe gain set."""
    mask = np.ones(J.shape[1], bool) if mask is None else mask
    Jm = np.where(mask[None], J, np.inf)
    jmin = Jm.min(1, keepdims=True)
    w = np.exp(-(Jm - jmin) / (temp * jmin + 1e-9))
    w /= w.sum(1, keepdims=True)
    return np.stack([w @ np.log(KP), w @ np.log(KD)], axis=1)


def safe_gain_set(J_train, rel_limit=5.0):
    """Gain pairs that never diverged and never cost > rel_limit x the window optimum."""
    rel = J_train / J_train.min(1, keepdims=True)
    return np.all((J_train < 1e5) & (rel < rel_limit), axis=0)


def schedule_from_model(predict_fn, F, rate_limit=0.08, lo=np.log(0.15), hi=np.log(2.5)):
    """Turn per-step feature predictions into rate-limited multipliers (kp, kd)."""
    Ylog = np.clip(predict_fn(F), lo, hi)
    out = np.zeros_like(Ylog)
    out[0] = Ylog[0]
    for k in range(1, len(Ylog)):
        out[k] = out[k - 1] + np.clip(Ylog[k] - out[k - 1], -rate_limit, rate_limit)
    return np.exp(out[:, 0]), np.exp(out[:, 1])
