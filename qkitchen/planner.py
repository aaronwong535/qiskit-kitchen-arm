"""Module 1 - end-effector path planning as a binary-encoded (HUBO) problem for QAOA.

Each transfer leg (sink -> rack while carrying, rack -> sink while empty) is split into
L layers. Layer l offers K = 2^bits candidate via points around a nominal lifted
path. Choosing one candidate per layer is encoded in `bits` qubits per layer
(binary, not one-hot), so every bitstring is a valid path and no penalty terms are
needed. Costs:
  unary  - arm-body clearance at the via point (IK + capsule model, moving obstacles
           evaluated at the time the arm is predicted to get there), comfort, reachability
  pair   - Cartesian length, joint-space travel ("minimum turning"), and swept-volume
           clearance between consecutive via points.
Because the cost is a chain of pairwise layer interactions, the exact optimum is
available by dynamic programming (the classical baseline) and by brute force.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from . import env as E
from . import robot as R
from .qaoa import DiagonalProblem, all_bitstrings

DOWN = np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------- timing helpers
def minjerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def minjerk_inverse(s):
    """tau such that minjerk(tau) = s (bisection, vectorised)."""
    s = np.asarray(s, float)
    lo, hi = np.zeros_like(s), np.ones_like(s)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        f = minjerk(mid) < s
        lo, hi = np.where(f, mid, lo), np.where(f, hi, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- IK helpers
def ik_track(points, q_seed, a=DOWN, iters=60):
    """Continuation IK along a sequence of TCP points (each seeded by the previous)."""
    qs, res = [], []
    q = np.array(q_seed, float)
    for p in points:
        q, ep, ea = R.ik_numeric(p, a, q, iters=iters)
        qs.append(q.copy())
        res.append(ep + 0.1 * ea)
    return np.array(qs), np.array(res)


def comfort_cost(q, q_comf, half):
    z = (np.atleast_2d(q) - q_comf) / half
    return np.sum(np.maximum(np.abs(z) - 0.6, 0.0) ** 2, axis=-1)


def penalty(d, d_safe=0.06):
    d = np.asarray(d, float)
    return (np.maximum(d_safe - d, 0.0) / d_safe) ** 2 + 2.0 * (d < 0.005)


# ---------------------------------------------------------------- leg problem
@dataclass
class LegProblem:
    A: np.ndarray
    B: np.ndarray
    t0: float
    T: float
    carrying: bool
    cand: np.ndarray        # (L, K, 3)
    q_cand: np.ndarray      # (L, K, 6)
    q_A: np.ndarray
    q_B: np.ndarray
    unary: np.ndarray       # (L, K)
    first: np.ndarray       # (K,)   A -> layer 0
    pair: np.ndarray        # (L-1, K, K)
    last: np.ndarray        # (K,)   layer L-1 -> B
    bits: int

    @property
    def L(self):
        return self.cand.shape[0]

    @property
    def n_qubits(self):
        return self.L * self.bits

    def decode(self, idx: int) -> np.ndarray:
        x = (idx >> np.arange(self.n_qubits)) & 1
        k = np.zeros(self.L, int)
        for b in range(self.bits):
            k += x[b::self.bits] << b
        return k

    def path_cost(self, ks) -> float:
        c = self.first[ks[0]] + self.last[ks[-1]] + self.unary[np.arange(self.L), ks].sum()
        for l in range(self.L - 1):
            c += self.pair[l, ks[l], ks[l + 1]]
        return float(c)

    def energies(self) -> np.ndarray:
        X = all_bitstrings(self.n_qubits).astype(int)
        K = np.zeros((X.shape[0], self.L), int)
        for b in range(self.bits):
            K += X[:, b::self.bits] << b
        Eg = self.first[K[:, 0]] + self.last[K[:, -1]]
        Eg = Eg + self.unary[np.arange(self.L)[None, :], K].sum(1)
        for l in range(self.L - 1):
            Eg = Eg + self.pair[l][K[:, l], K[:, l + 1]]
        return Eg

    def to_problem(self, **meta) -> DiagonalProblem:
        return DiagonalProblem.from_energies(self.energies(), **meta)

    def dp_solve(self):
        """Exact optimum by dynamic programming (Viterbi) - the classical baseline."""
        Kn = self.unary.shape[1]
        cost = self.first + self.unary[0]
        back = []
        for l in range(1, self.L):
            tot = cost[:, None] + self.pair[l - 1] + self.unary[l][None, :]
            back.append(np.argmin(tot, axis=0))
            cost = tot.min(axis=0)
        cost = cost + self.last
        ks = [int(np.argmin(cost))]
        for bk in reversed(back):
            ks.append(int(bk[ks[-1]]))
        ks = ks[::-1]
        return np.array(ks), float(cost.min())

    def encode(self, ks) -> int:
        idx = 0
        for l, k in enumerate(ks):
            for b in range(self.bits):
                idx |= ((int(k) >> b) & 1) << (l * self.bits + b)
        return idx

    def via_points(self, ks):
        return np.vstack([self.A, self.cand[np.arange(self.L), ks], self.B])


def build_leg(kit: E.Kitchen, A, B, t0, T, carrying, q_A, q_comf, half,
              L=5, bits=2, lift=0.10, spread=0.14, up=0.14,
              w_len=1.0, w_q=0.25, w_clear=4.0, w_comf=0.5) -> LegProblem:
    K = 2 ** bits
    A, B = np.asarray(A, float), np.asarray(B, float)
    s = (np.arange(L) + 1) / (L + 1)
    d = B - A
    dh = np.array([d[0], d[1], 0.0])
    u = np.cross([0, 0, 1.0], dh)
    u /= np.linalg.norm(u) + 1e-12
    w = np.array([0, 0, 1.0])
    nominal = A + s[:, None] * d + (lift * np.sin(np.pi * s))[:, None] * w
    base_off = [np.zeros(3), up * w, spread * u + 0.06 * w, -spread * u + 0.06 * w]
    if K > 4:   # 8 candidates: add higher/wider variants
        base_off += [2 * up * w, spread * u + up * w, -spread * u + up * w, 1.6 * up * w + 0.0 * u]
    off = np.array(base_off[:K])
    cand = nominal[:, None, :] + off[None, :, :]
    tau = t0 + T * minjerk_inverse(s)           # time the arm should reach each layer

    # IK: continuation along the nominal path, then each candidate seeded from there
    q_nom, _ = ik_track(nominal, q_A)
    q_cand = np.zeros((L, K, 6))
    res = np.zeros((L, K))
    for l in range(L):
        for k in range(K):
            q, ep, ea = R.ik_numeric(cand[l, k], DOWN, q_nom[l], iters=60)
            q_cand[l, k], res[l, k] = q, ep + 0.1 * ea
    q_B, _ = ik_track([B], q_nom[-1])
    q_B = q_B[0]

    # unary: arm clearance at the via point (incl. held dish), reachability, comfort
    flatq = q_cand.reshape(-1, 6)
    dmin = E.min_clearance(kit, flatq, np.repeat(tau, K), carrying).reshape(L, K)
    unary = (w_clear * penalty(dmin) + 10.0 * (res > 2e-3)
             + w_comf * comfort_cost(flatq, q_comf, half).reshape(L, K))

    wj = np.array([1.0, 1.0, 0.8, 0.4, 0.4, 0.2])

    def edge(qa, qb, pa, pb, ta, tb):
        """Cost of moving between (batched) configurations."""
        length = np.linalg.norm(pb - pa, axis=-1)
        travel = np.linalg.norm((qb - qa) * wj, axis=-1)
        ss = np.linspace(0.2, 0.8, 4)
        qs = qa[..., None, :] + ss[:, None] * (qb - qa)[..., None, :]
        ts = ta[..., None] + ss * (tb - ta)[..., None]
        dseg = E.min_clearance(kit, qs.reshape(-1, 6), ts.reshape(-1), carrying)
        dseg = dseg.reshape(qs.shape[:-1]).min(-1)
        return w_len * length + w_q * travel + w_clear * penalty(dseg)

    first = edge(np.broadcast_to(q_A, (K, 6)), q_cand[0], np.broadcast_to(A, (K, 3)),
                 cand[0], np.full(K, t0), np.full(K, tau[0]))
    last = edge(q_cand[-1], np.broadcast_to(q_B, (K, 6)), cand[-1],
                np.broadcast_to(B, (K, 3)), np.full(K, tau[-1]), np.full(K, t0 + T))
    pair = np.zeros((L - 1, K, K))
    for l in range(L - 1):
        qa = np.broadcast_to(q_cand[l][:, None, :], (K, K, 6))
        qb = np.broadcast_to(q_cand[l + 1][None, :, :], (K, K, 6))
        pa = np.broadcast_to(cand[l][:, None, :], (K, K, 3))
        pb = np.broadcast_to(cand[l + 1][None, :, :], (K, K, 3))
        pair[l] = edge(qa, qb, pa, pb, np.full((K, K), tau[l]), np.full((K, K), tau[l + 1]))
    return LegProblem(A, B, t0, T, carrying, cand, q_cand, np.asarray(q_A), q_B,
                      unary, first, pair, last, bits)


# ---------------------------------------------------------------- reference trajectory
@dataclass
class Segment:
    kind: str
    t0: float
    T: float
    points: np.ndarray          # via points (>= 2)
    carrying: bool
    precision: bool
    dish: int


def segment_sampler(seg: Segment):
    """Return f(t) -> TCP point along the segment with min-jerk timing on arc length."""
    P = np.asarray(seg.points, float)
    if len(P) == 2 or np.allclose(P[0], P[-1]):
        return lambda t: P[0] + minjerk((t - seg.t0) / seg.T)[..., None] * (P[-1] - P[0])
    chord = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    cs = CubicSpline(chord, P, bc_type="clamped")
    uu = np.linspace(0, chord[-1], 400)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cs(uu), axis=0), axis=1))]

    def f(t):
        s = minjerk((np.asarray(t) - seg.t0) / seg.T) * arc[-1]
        return cs(np.interp(s, arc, uu))
    return f


def sample_reference(segments, dt=0.01):
    """Stack the segments into a 0.01 s reference: times, TCP targets, phase flags."""
    t_end = segments[-1].t0 + segments[-1].T
    t = np.arange(0.0, t_end + 1e-9, dt)
    P = np.zeros((len(t), 3))
    carrying = np.zeros(len(t), bool)
    precision = np.zeros(len(t), bool)
    seg_id = np.zeros(len(t), int)
    for i, seg in enumerate(segments):
        m = (t >= seg.t0 - 1e-9) & (t <= seg.t0 + seg.T + 1e-9)
        f = segment_sampler(seg)
        P[m] = f(t[m])
        carrying[m] = seg.carrying
        precision[m] = seg.precision
        seg_id[m] = i
    return t, P, carrying, precision, seg_id
