"""Draw every quantum circuit of the project as PNG (and one PDF) in figures/circuits/.

Runs entirely on this computer: no IBM account, no token and no QPU time are used. The
circuits are built by the same functions that main.py sends to the QPU.

    pip install pylatexenc            # needed once by Qiskit's Matplotlib drawer
    python scripts/draw_circuits.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as M                                                  # noqa: E402

OUT = ROOT / "figures" / "circuits"

# what each circuit is, in one line (shown above the drawing and in the README)
NOTES = {
    "m1_p1": "Module 1, path planning. QAOA with 1 layer on the path HUBO: Hadamards, cost layer "
             "(RZ, RZZ and CX-ladder terms up to 4 qubits), RX mixer, measurement.",
    "m1_p3": "Module 1, path planning. The same QAOA circuit with 3 layers.",
    "m2_p1": "Module 2, inverse kinematics. QAOA with 1 layer on the per-step QUBO: RZ on every "
             "qubit, RZZ on every pair, RX mixer.",
    "m2_p3": "Module 2, inverse kinematics. The same QAOA circuit with 3 layers.",
    "m3_kernel": "Module 3, quantum kernel for gain scheduling. ZZ feature map U(x) followed by "
                 "U(y)†; P(0000) is the kernel value k(x, y).",
    "m4_kernel": "Module 4, quantum kernel for residual dynamics. Same construction with a deeper "
                 "feature map (depth chosen by marginal likelihood).",
    "m5_policy": "Module 5, gain policy. Data re-uploading circuit: RY(w·x + b), RZ and a ring of "
                 "CZ gates per layer; <Z0> and <Z1> set the two gain multipliers. Trained weights are "
                 "shown to two decimals.",
}


def rounded_policy_circuit(a):
    """The Module 5 circuit with its trained weights rounded for display only."""
    import pickle

    import numpy as np
    from qkitchen.qml import VQCPolicy
    vq = VQCPolicy(6, 3, 2, seed=a.policy_seed)
    vq.set(np.asarray(pickle.load(open(ROOT / "results" / "m5_agents.pkl", "rb"))[f"vqc_{a.policy_seed}"]))
    return M.policy_circuit(np.round(vq.W, 2), np.round(vq.Bb, 2), np.round(vq.C, 2))


def main():
    try:
        import pylatexenc                                        # noqa: F401
    except ImportError:
        sys.exit("Qiskit's Matplotlib drawer needs pylatexenc:  pip install pylatexenc")
    a = SimpleNamespace(**M.DEFAULTS)
    W = M.build_workloads(a)
    OUT.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT / "all_circuits.pdf") as pdf:
        for key, w in W.items():
            qc = w.circuit
            if key == "m5_policy":                    # same circuit, trained weights shown to 2 decimals
                qc = rounded_policy_circuit(a)
            fig = qc.draw("mpl", fold=-1, scale=0.8, idle_wires=False)          # one row per circuit
            ops = qc.count_ops()
            twoq = sum(v for k, v in ops.items() if k in ("cx", "cz", "rzz"))
            fig.axes[0].set_title(f"{w.title}\n{qc.num_qubits} qubits, {qc.num_parameters} parameters, "
                                  f"{twoq} two-qubit gates before transpiling, depth {qc.depth()}",
                                  fontsize=16, loc="left", pad=14)
            # keep very long circuits under ~30,000 px wide (PNG viewers struggle beyond that)
            w_in = fig.get_size_inches()[0]
            dpi = min(110, int(30000 / max(w_in, 1)))
            fig.savefig(OUT / f"{key}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  {key:10s} -> figures/circuits/{key}.png")
    print("  all circuits in one file -> figures/circuits/all_circuits.pdf")


if __name__ == "__main__":
    main()
