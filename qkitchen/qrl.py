"""Module 5 - policy-gradient RL with a variational-quantum-circuit actor.

The agent replaces Module 3's label-based gain scheduling: every 50 ms it observes the
motion (reach, speed, acceleration, payload flag) and the measured tracking error and
error rate, and outputs PID gain multipliers (kp, kd). It is trained with REINFORCE
with a time-indexed baseline on the true multi-step closed-loop cost (tracking error +
effort + torque chatter + saturation), so unlike Module 3 its targets are not myopic.

Actors compared at (almost) equal size: a 6-qubit data re-uploading circuit
(54 parameters, gradients by the parameter-shift rule) and a 6-6-2 tanh MLP
(56 parameters, backprop). Both feed the mean of a Gaussian policy in log-gain space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import control as C

LOG_CENTER = np.array([0.0, np.log(np.sqrt(0.3 * 1.5))])
LOG_HALF = np.array([np.log(2.0), np.log(np.sqrt(1.5 / 0.3))])   # kp in [0.5, 2], kd in [0.3, 1.5]
DECISION_EVERY = 5       # reference steps (50 ms)


class Adam:
    def __init__(self, n, lr=0.05, b1=0.9, b2=0.999):
        self.m = np.zeros(n); self.v = np.zeros(n); self.t = 0
        self.lr, self.b1, self.b2 = lr, b1, b2

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t); vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + 1e-8)


def observation(ref_feats_k, e, ed):
    """6-D observation in roughly [-1, 1]."""
    e_mrad = 1e3 * np.sqrt(np.mean(e ** 2, 1)) + 1e-3
    ed_mrad = 1e3 * np.sqrt(np.mean(ed ** 2, 1)) + 1e-3
    o5 = np.clip(np.log10(e_mrad), -1, 1)
    o6 = np.clip((np.log10(ed_mrad) - 1.3) / 0.7, -1, 1)
    F = np.broadcast_to(2 * ref_feats_k - 1, (e.shape[0], 4))
    return np.column_stack([F, o5, o6])


def center_outputs(model, rng=None, n=512):
    """Shift the actor's output so the initial policy is centred on the same gains for
    every actor type (a fixed offset, part of the policy class for both VQC and MLP)."""
    rng = rng or np.random.default_rng(123)
    O = rng.uniform(-1, 1, (n, 6))
    model.offset = model.forward(O).mean(0)
    return model


def policy_mean(model, O):
    return LOG_CENTER + LOG_HALF * (model.forward(O) - getattr(model, "offset", 0.0))


GAIN_TAU = 0.1    # s, first-order smoothing of commanded gain changes (bumpless switching)


class PolicyRunner:
    """gain_fn for control.simulate: one actor per batch element (or a fixed gain)."""

    def __init__(self, actors, ref_feats, sigma=0.0, rng=None, ref_dt=0.01):
        self.actors = actors            # list: model with forward(), or (kp, kd) tuple
        self.F = ref_feats
        self.sigma = sigma
        self.rng = rng or np.random.default_rng(0)
        self.O, self.A, self.MU, self.K = [], [], [], []
        self.cur = None
        self.applied = None
        self.beta = ref_dt / GAIN_TAU

    def __call__(self, k, obs):
        B = obs["e"].shape[0]
        if k % DECISION_EVERY == 0 or self.cur is None:
            O = observation(self.F[k], obs["e"], obs["ed"])
            a = np.zeros((B, 2)); mu = np.zeros((B, 2))
            for b, act in enumerate(self.actors):
                if isinstance(act, tuple):
                    mu[b] = np.log(act); a[b] = mu[b]
                else:
                    mu[b] = policy_mean(act, O[b:b + 1])[0]
                    a[b] = mu[b] + self.sigma * self.rng.standard_normal(2)
            self.O.append(O); self.A.append(a); self.MU.append(mu); self.K.append(k)
            lo, hi = LOG_CENTER - LOG_HALF, LOG_CENTER + LOG_HALF
            self.cur = np.clip(a, lo, hi)
        if self.applied is None:
            self.applied = self.cur.copy()
        self.applied = self.applied + self.beta * (self.cur - self.applied)   # in log space
        g = np.exp(self.applied)
        return g[:, 0], g[:, 1]


@dataclass
class RLConfig:
    iters: int = 100
    batch: int = 24
    n_ref: int = 4               # nominal-gain rollouts per iteration (for normalised curves)
    window: int = 150
    sigma0: float = 0.30
    sigma1: float = 0.08
    lr: float = 0.05
    lr_vqc: float | None = None    # optional separate step size for the circuit angles


def train(model, Q, carry, feats, nominal_state, gains, cfg=RLConfig(), seed=0, log=print,
          name="agent"):
    """REINFORCE with a per-decision batch-mean baseline on 1.5 s windows."""
    rng = np.random.default_rng(seed)
    qd_ref, _ = C.reference_derivatives(Q)
    is_vqc = hasattr(model, "qiskit_circuit")
    lr = cfg.lr_vqc if (is_vqc and cfg.lr_vqc) else cfg.lr
    center_outputs(model)
    opt = Adam(len(model.get()), lr=lr)
    curve = []
    N = len(Q)
    for it in range(cfg.iters):
        sigma = cfg.sigma0 + (cfg.sigma1 - cfg.sigma0) * it / max(cfg.iters - 1, 1)
        k0 = int(rng.integers(0, N - cfg.window - 1))
        k1 = k0 + cfg.window
        B = cfg.batch + cfg.n_ref
        pay = np.where(carry[k0:k1], rng.uniform(0.3, 1.0), 0.0)
        actors = [model] * cfg.batch + [(1.0, 1.0)] * cfg.n_ref
        runner = PolicyRunner(actors, feats[k0:k1], sigma=sigma,
                              rng=np.random.default_rng(seed * 1000 + it))
        sim = C.simulate(Q[k0:k1], qd_ref[k0:k1], pay, np.ones((B, cfg.window)),
                         np.ones((B, cfg.window)), gains,
                         q_init=np.repeat(nominal_state["Q"][k0][None], B, 0),
                         qd_init=np.repeat(nominal_state["QD"][k0][None], B, 0),
                         integ_init=np.repeat(nominal_state["INT"][k0][None], B, 0),
                         seed=int(rng.integers(1 << 30)), gain_fn=runner)
        sc = sim["step_cost"][:, :-1] * C.SimConfig().ref_dt          # (B, window-1)
        T = len(runner.K)
        r = np.stack([sc[:, k:k + DECISION_EVERY].sum(1) for k in runner.K], 1)   # (B, T)
        G = -np.cumsum(r[:, ::-1], 1)[:, ::-1]                         # returns-to-go
        Gp = G[:cfg.batch]
        A = (Gp - Gp.mean(0)) / (Gp.std(0) + 1e-8)
        O = np.stack(runner.O, 1)[:cfg.batch]                          # (b, T, 6)
        a = np.stack(runner.A, 1)[:cfg.batch]
        mu = np.stack(runner.MU, 1)[:cfg.batch]
        Gout = (A[..., None] * (a - mu) / sigma ** 2 * LOG_HALF).reshape(-1, 2)
        grad = model.grad(O.reshape(-1, 6), Gout) / cfg.batch
        model.set(model.get() + opt.step(grad))
        c_pol = sim["cost"][:cfg.batch].mean()
        c_nom = sim["cost"][cfg.batch:].mean()
        curve.append(dict(it=it, k0=k0, cost=float(c_pol), nominal=float(c_nom),
                          rel=float(c_pol / c_nom), sigma=float(sigma)))
        if it % 10 == 0 or it == cfg.iters - 1:
            log(f"    [{name}] iter {it:3d}  cost/nominal = {c_pol / c_nom:.3f}  (sigma {sigma:.2f})")
    return curve
