"""Analysis of measured samples (counts) from a simulator or an IBM QPU.

Building, transpiling and running the Qiskit circuits is done in main.py at the repository
root; these helpers turn the returned counts into the numbers reported in the README.
"""
from __future__ import annotations

import numpy as np


def analyse_qaoa_counts(problems, counts, top_k=6):
    """Samples -> P(opt), mean normalised energy, opt-in-top-k rate, gap of the best sample."""
    out = []
    for pr, cnt in zip(problems, counts):
        idx = np.array([int(k, 2) for k in cnt])
        c = np.array(list(cnt.values()), float)
        p = c / c.sum()
        E = pr.energies
        lo, hi = E.min(), E.max()
        ar = float((hi - p @ E[idx]) / (hi - lo)) if hi > lo else 1.0
        ranked = idx[np.argsort(E[idx])][:top_k]
        out.append(dict(p_opt=float(p[idx == pr.ground_index].sum()), approx_ratio=ar,
                        opt_in_topk=bool(pr.ground_index in ranked),
                        best_gap=float(E[ranked[0]] - lo)))
    return out


def kernel_from_counts(counts, n_qubits):
    """Kernel value = probability of the all-zeros outcome of the compute-uncompute circuit."""
    z = "0" * n_qubits
    return np.array([c.get(z, 0) / sum(c.values()) for c in counts])
