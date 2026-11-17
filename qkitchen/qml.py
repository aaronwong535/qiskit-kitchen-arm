"""Fast exact simulators for the two quantum-learning circuits used in Modules 4 and 5.

* ZZ feature map (Qiskit's ``zz_feature_map`` layout, linear entanglement) -> fidelity
  kernel k(x, x') = |<phi(x)|phi(x')>|^2 for Gaussian-process regression.
* Data re-uploading variational circuit (RY(w*x+b), RZ(c), CZ ring) -> <Z_0>, <Z_1>
  used as the mean of a Gaussian RL policy, with exact parameter-shift gradients.

Both are batched NumPy statevector simulations (qubit i = bit i of the basis index,
as in Qiskit) and are checked against Qiskit ``Statevector`` in tests/. The same
circuits are emitted as Qiskit ``QuantumCircuit`` objects for Aer / fake backends /
IBM hardware (see hardware.py).
"""
from __future__ import annotations

import numpy as np


def _bits(n):
    idx = np.arange(2 ** n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)   # (N, n)


def _hadamard_all(psi, n):
    """Apply H on every qubit to a batch of states psi (B, 2^n)."""
    B = psi.shape[0]
    out = psi.reshape(B, -1)
    h = 1
    for _ in range(n):
        v = out.reshape(B, -1, 2, h)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        out = np.stack([a + b, a - b], axis=2).reshape(B, -1)
        h *= 2
    return out / np.sqrt(2 ** n)


# ---------------------------------------------------------------- ZZ feature map
class ZZKernel:
    """Fidelity kernel of Qiskit's zz_feature_map (reps, linear entanglement).

    Inputs are features in [0, 1]; they are scaled by bandwidth * pi before encoding
    (bandwidth tuning as in Shaydulin & Wild 2022).
    """

    def __init__(self, n_features, reps=2, bandwidth=0.5):
        self.n, self.reps, self.bandwidth = n_features, reps, bandwidth
        self.b = _bits(n_features)
        self.pairs = [(i, i + 1) for i in range(n_features - 1)]
        self.xor = np.stack([np.logical_xor(self.b[:, i], self.b[:, j]).astype(float)
                             for i, j in self.pairs], axis=1) if self.pairs else np.zeros((2 ** n_features, 0))

    def encode(self, F):
        return self.bandwidth * np.pi * np.atleast_2d(F)

    def states(self, F):
        X = self.encode(F)                                        # (B, n)
        phase = 2 * X @ self.b.T                                  # (B, N)
        if self.pairs:
            pp = np.stack([(np.pi - X[:, i]) * (np.pi - X[:, j]) for i, j in self.pairs], 1)
            phase = phase + 2 * pp @ self.xor.T
        D = np.exp(1j * phase)
        psi = np.zeros((X.shape[0], 2 ** self.n), complex)
        psi[:, 0] = 1.0
        for _ in range(self.reps):
            psi = _hadamard_all(psi, self.n) * D
        return psi

    def gram(self, FA, FB=None, SA=None, SB=None):
        SA = self.states(FA) if SA is None else SA
        if FB is None and SB is None:
            SB = SA
        elif SB is None:
            SB = self.states(FB)
        return np.abs(SA.conj() @ SB.T) ** 2

    def qiskit_circuit(self):
        from qiskit.circuit.library import zz_feature_map
        return zz_feature_map(self.n, reps=self.reps, entanglement="linear")


# ---------------------------------------------------------------- VQC policy
class VQCPolicy:
    """Data re-uploading circuit: L layers of [RY(w x + b), RZ(c)] per qubit + CZ ring.

    params = (W, Bias, C), each (L, n). Output: expectation values <Z_k>, k < n_out.
    """

    def __init__(self, n_qubits=6, layers=3, n_out=2, seed=0):
        self.n, self.L, self.n_out = n_qubits, layers, n_out
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.4, (layers, n_qubits))      # same input scale as the MLP init
        self.Bb = rng.normal(0.0, 0.1, (layers, n_qubits))
        self.C = rng.normal(0.0, 0.1, (layers, n_qubits))
        b = _bits(n_qubits)
        ring = [(i, (i + 1) % n_qubits) for i in range(n_qubits)] if n_qubits > 2 else [(0, 1)]
        self.cz_sign = np.prod([1 - 2 * b[:, i] * b[:, j] for i, j in ring], axis=0)
        self.zsign = 1 - 2 * b[:, :n_out]                          # (N, n_out)
        self.ring = ring

    @property
    def n_params(self):
        return 3 * self.L * self.n

    def get(self):
        return np.concatenate([self.W.ravel(), self.Bb.ravel(), self.C.ravel()])

    def set(self, v):
        k = self.L * self.n
        self.W, self.Bb, self.C = (v[:k].reshape(self.L, self.n), v[k:2 * k].reshape(self.L, self.n),
                                   v[2 * k:].reshape(self.L, self.n))

    def _run(self, X, TH, PH):
        """X (B, n) inputs; TH, PH (B, L, n) RY / RZ angles -> <Z_k> (B, n_out)."""
        B = X.shape[0]
        N = 2 ** self.n
        psi = np.zeros((B, N), complex)
        psi[:, 0] = 1.0
        for l in range(self.L):
            for i in range(self.n):
                v = psi.reshape(B, -1, 2, 2 ** i)
                c, s = np.cos(TH[:, l, i] / 2)[:, None, None], np.sin(TH[:, l, i] / 2)[:, None, None]
                a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
                n0, n1 = c * a0 - s * a1, s * a0 + c * a1
                e = np.exp(-0.5j * PH[:, l, i])[:, None, None]
                v = np.stack([n0 * e, n1 * np.conj(e)], axis=2)
                psi = v.reshape(B, N)
            psi = psi * self.cz_sign[None, :]
        p = np.abs(psi) ** 2
        return p @ self.zsign

    def angles(self, X):
        X = np.atleast_2d(X)
        TH = self.W[None] * X[:, None, :] + self.Bb[None]
        PH = np.broadcast_to(self.C[None], (X.shape[0], self.L, self.n)).copy()
        return TH, PH

    def forward(self, X):
        TH, PH = self.angles(X)
        return self._run(np.atleast_2d(X), TH, PH)

    def grad(self, X, G):
        """Parameter-shift gradient of sum_b G[b] . <Z>(x_b) w.r.t. (W, Bias, C).

        G (B, n_out) is the upstream gradient (e.g. from the policy log-likelihood).
        Exact for these rotation gates: d<Z>/dtheta = [f(theta+pi/2) - f(theta-pi/2)]/2.
        """
        X = np.atleast_2d(X)
        TH, PH = self.angles(X)
        gW = np.zeros_like(self.W); gB = np.zeros_like(self.Bb); gC = np.zeros_like(self.C)
        for l in range(self.L):
            for i in range(self.n):
                for kind in ("th", "ph"):
                    outs = []
                    for sgn in (+1, -1):
                        T2, P2 = TH.copy(), PH.copy()
                        if kind == "th":
                            T2[:, l, i] += sgn * np.pi / 2
                        else:
                            P2[:, l, i] += sgn * np.pi / 2
                        outs.append(self._run(X, T2, P2))
                    d = 0.5 * (outs[0] - outs[1])                  # (B, n_out)
                    gd = np.sum(d * G, axis=1)                     # (B,)
                    if kind == "th":
                        gW[l, i] = np.sum(gd * X[:, i]); gB[l, i] = np.sum(gd)
                    else:
                        gC[l, i] = np.sum(gd)
        return np.concatenate([gW.ravel(), gB.ravel(), gC.ravel()])

    def qiskit_circuit(self, measure=True):
        """Circuit with input Parameters x[i] and trained weights baked in."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        x = ParameterVector("x", self.n)
        qc = QuantumCircuit(self.n)
        for l in range(self.L):
            for i in range(self.n):
                qc.ry(float(self.W[l, i]) * x[i] + float(self.Bb[l, i]), i)
                qc.rz(float(self.C[l, i]), i)
            for i, j in self.ring:
                qc.cz(i, j)
        if measure:
            qc.measure_all()
        return qc


def z_expectations_from_counts(counts, n_out):
    """<Z_k> for k < n_out from a Qiskit counts dict (bitstring 'c_{n-1}...c_0')."""
    tot = sum(counts.values())
    out = np.zeros(n_out)
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]
        for k in range(n_out):
            out[k] += c * (1 - 2 * int(bits[k]))
    return out / tot


# ---------------------------------------------------------------- classical comparison
class MLPPolicy:
    """Tiny tanh MLP with the same output squashing (n_in -> h -> n_out)."""

    def __init__(self, n_in=6, hidden=6, n_out=2, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (hidden, n_in)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (n_out, hidden)); self.b2 = np.zeros(n_out)

    @property
    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set(self, v):
        a = self.W1.size; b = a + self.b1.size; c = b + self.W2.size
        self.W1 = v[:a].reshape(self.W1.shape); self.b1 = v[a:b]
        self.W2 = v[b:c].reshape(self.W2.shape); self.b2 = v[c:]

    def forward(self, X):
        self._X = np.atleast_2d(X)
        self._H = np.tanh(self._X @ self.W1.T + self.b1)
        return np.tanh(self._H @ self.W2.T + self.b2)

    def grad(self, X, G):
        Y = self.forward(X)
        dZ2 = G * (1 - Y ** 2)
        gW2 = dZ2.T @ self._H; gb2 = dZ2.sum(0)
        dH = dZ2 @ self.W2 * (1 - self._H ** 2)
        gW1 = dH.T @ self._X; gb1 = dH.sum(0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2])
