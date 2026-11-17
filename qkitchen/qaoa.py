"""QAOA engine for diagonal (QUBO / higher-order) cost functions.

* A problem is given by its exact cost for every bitstring (2^n vector). Its Pauli-Z
  expansion is recovered exactly with a fast Walsh-Hadamard transform, so the same
  object drives (a) a fast exact statevector simulator used inside the 100 Hz loop and
  (b) Qiskit circuits for Aer / fake-backend noise / IBM hardware.
* Bit ordering follows Qiskit: index b = sum_i x_i 2^i, x_i is qubit i, and
  z_i = 1 - 2 x_i is the Z eigenvalue.
* Circuits put every term angle in its own Parameter (theta = 2*gamma*c_S), so one
  transpiled ISA circuit serves thousands of instances through a single SamplerV2
  parameter sweep.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------- problem
def all_bitstrings(n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def walsh_coefficients(E: np.ndarray) -> np.ndarray:
    """c_S for E(x) = sum_S c_S prod_{i in S} z_i (S encoded as an int mask)."""
    c = np.array(E, float).copy()
    n = int(np.log2(len(c)))
    h = 1
    for _ in range(n):
        c = c.reshape(-1, 2, h)
        a, b = c[:, 0, :].copy(), c[:, 1, :].copy()
        c[:, 0, :], c[:, 1, :] = a + b, a - b
        c = c.reshape(-1)
        h *= 2
    return c / len(c)


@dataclass
class DiagonalProblem:
    n: int
    energies: np.ndarray                     # exact cost for each bitstring index
    terms: list = field(default_factory=list)   # [(support tuple, coeff)] without identity
    const: float = 0.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_energies(cls, E: np.ndarray, rel_tol: float = 1e-9, max_order: int | None = None,
                      **meta):
        n = int(np.log2(len(E)))
        c = walsh_coefficients(E)
        scale = np.abs(c[1:]).max() if len(c) > 1 else 1.0
        terms = []
        for mask in np.nonzero(np.abs(c) > rel_tol * max(scale, 1e-300))[0]:
            if mask == 0:
                continue
            sup = tuple(i for i in range(n) if (mask >> i) & 1)
            if max_order is not None and len(sup) > max_order:
                continue
            terms.append((sup, float(c[mask])))
        return cls(n=n, energies=np.asarray(E, float), terms=terms, const=float(c[0]),
                   meta=meta)

    @property
    def scale(self) -> float:
        return max(abs(c) for _, c in self.terms) if self.terms else 1.0

    def normalized_energies(self) -> np.ndarray:
        return (self.energies - self.const) / self.scale

    def pruned_energies(self) -> np.ndarray:
        """Energies implied by the (possibly pruned) term list, unnormalised."""
        z = 1 - 2 * all_bitstrings(self.n).astype(float)
        E = np.full(2 ** self.n, self.const)
        for sup, c in self.terms:
            E += c * np.prod(z[:, list(sup)], axis=1)
        return E

    @property
    def ground_index(self) -> int:
        return int(np.argmin(self.energies))

    def approx_ratio(self, probs: np.ndarray) -> float:
        E = self.energies
        lo, hi = E.min(), E.max()
        return float((hi - probs @ E) / (hi - lo)) if hi > lo else 1.0


# ---------------------------------------------------------------- fast simulator
def qaoa_state(E_norm: np.ndarray, gammas, betas) -> np.ndarray:
    """Exact QAOA state for normalized diagonal cost (|+>^n, e^{-i g H}, e^{-i b sum X})."""
    N = len(E_norm)
    n = int(np.log2(N))
    psi = np.full(N, 1 / np.sqrt(N), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E_norm)
        cb, sb = np.cos(b), -1j * np.sin(b)
        for i in range(n):
            v = psi.reshape(-1, 2, 2 ** i)
            a0, a1 = v[:, 0, :].copy(), v[:, 1, :].copy()
            v[:, 0, :] = cb * a0 + sb * a1
            v[:, 1, :] = sb * a0 + cb * a1
            psi = v.reshape(-1)
    return psi


def qaoa_probs(problem: DiagonalProblem, gammas, betas) -> np.ndarray:
    return np.abs(qaoa_state(problem.normalized_energies(), gammas, betas)) ** 2


def _objective(params, Es, p, mode, alpha):
    g, b = params[:p], params[p:]
    vals = []
    for E in Es:
        pr = np.abs(qaoa_state(E, g, b)) ** 2
        lo, hi = E.min(), E.max()
        if mode == "mean":
            vals.append((pr @ E - lo) / (hi - lo + 1e-12))
        else:  # CVaR_alpha (Barkoutsos et al. 2020)
            order = np.argsort(E)
            cp = np.cumsum(pr[order])
            k = np.searchsorted(cp, alpha) + 1
            w = pr[order][:k].copy()
            w[-1] -= max(cp[k - 1] - alpha, 0)
            vals.append((w @ E[order][:k] / alpha - lo) / (hi - lo + 1e-12))
    return float(np.mean(vals))


def optimize_angles(problems, p: int = 2, restarts: int = 8, mode: str = "mean",
                    alpha: float = 0.2, seed: int = 0, init=None, maxiter: int = 400):
    """Optimise one shared (gamma, beta) vector for a list of problems.

    With a single problem this is ordinary per-instance QAOA training; with many it is
    the parameter-transfer recipe (normalised costs share good angles).
    """
    Es = [pr.normalized_energies() for pr in problems]
    rng = np.random.default_rng(seed)
    best = None
    starts = [] if init is None else [np.asarray(init, float)]
    # linear-ramp start (annealing-like), then random starts
    starts.append(np.concatenate([np.linspace(0.1, 0.6, p), np.linspace(0.6, 0.1, p)]))
    while len(starts) < restarts:
        starts.append(np.concatenate([rng.uniform(0, 1.2, p), rng.uniform(0, np.pi / 2, p)]))
    for x0 in starts:
        r = minimize(_objective, x0, args=(Es, p, mode, alpha), method="COBYLA",
                     options={"maxiter": maxiter, "rhobeg": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:p], best.x[p:], float(best.fun)


def interp_init(gammas, betas):
    """INTERP heuristic (Zhou et al., PRX 10, 021067): depth-p angles -> depth p+1 start."""
    def up(v):
        v = np.asarray(v, float)
        p = len(v)
        out = np.zeros(p + 1)
        for i in range(p + 1):
            lo = v[i - 1] if i - 1 >= 0 else 0.0
            hi = v[i] if i < p else 0.0
            out[i] = (i / p) * lo + ((p - i) / p) * hi
        return out
    return np.concatenate([up(gammas), up(betas)])


def sample(probs: np.ndarray, shots: int, rng) -> np.ndarray:
    return rng.choice(len(probs), size=shots, p=probs / probs.sum())


# ---------------------------------------------------------------- Qiskit circuits
def term_union(problems) -> list:
    sups = set()
    for pr in problems:
        sups.update(s for s, _ in pr.terms)
    return sorted(sups, key=lambda s: (len(s), s))


def build_qaoa_circuit(n: int, supports: list, p: int, measure: bool = True):
    """Parameterised QAOA circuit; parameters: theta[l*len(supports)+j], beta[l]."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    theta = ParameterVector("theta", p * len(supports))
    beta = ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for l in range(p):
        for j, sup in enumerate(supports):
            th = theta[l * len(supports) + j]
            if len(sup) == 1:
                qc.rz(th, sup[0])
            elif len(sup) == 2:
                qc.rzz(th, sup[0], sup[1])
            else:
                for a, b in zip(sup[:-1], sup[1:]):
                    qc.cx(a, b)
                qc.rz(th, sup[-1])
                for a, b in reversed(list(zip(sup[:-1], sup[1:]))):
                    qc.cx(a, b)
        for q in range(n):
            qc.rx(2 * beta[l], q)
    if measure:
        qc.measure_all()
    return qc, theta, beta


def parameter_values(problem: DiagonalProblem, supports: list, gammas, betas,
                     circuit=None) -> np.ndarray:
    """Values for build_qaoa_circuit parameters (ordered as circuit.parameters if given)."""
    coeff = dict(problem.terms)
    s = problem.scale
    p = len(gammas)
    th = np.zeros(p * len(supports))
    for l in range(p):
        for j, sup in enumerate(supports):
            th[l * len(supports) + j] = 2 * gammas[l] * coeff.get(sup, 0.0) / s
    vals = {f"theta[{i}]": v for i, v in enumerate(th)}
    vals.update({f"beta[{l}]": b for l, b in enumerate(betas)})
    if circuit is None:
        return np.concatenate([th, betas])
    return np.array([vals[par.name] for par in circuit.parameters])


def counts_to_index_array(counts: dict) -> tuple[np.ndarray, np.ndarray]:
    idx = np.array([int(k, 2) for k in counts])
    cnt = np.array(list(counts.values()))
    return idx, cnt
