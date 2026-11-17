#!/usr/bin/env python
"""Run every quantum circuit of the kitchen-arm project on IBM Quantum.

This is the single entry point for the hardware run. It

  1. connects to your IBM Quantum account (https://quantum.cloud.ibm.com/docs/en/ibm-quantum-compute/hello-world),
  2. builds the Qiskit circuit of every quantum function in the project (Modules 1-5),
  3. checks each circuit against Qiskit's exact Statevector and the project's fast simulator,
  4. transpiles them for the chosen QPU,
  5. submits them as one batch through the Qiskit Runtime Sampler,
  6. collects the results, compares them with the exact answers and writes results/.

Typical use
-----------
    python main.py --save-account     # once: store your IBM Quantum API key on this computer
    python main.py --dry-run          # free: build + transpile for the QPU, print the time estimate
    python main.py                    # submit, wait, analyse, save (asks before spending QPU time)

Free local rehearsals (no account, no QPU time):
    python main.py --backend aer            # ideal simulator
    python main.py --backend fake:FakeMarrakesh   # simulator with the noise model of ibm_marrakesh

QPU choice: ibm_marrakesh first, ibm_kingston second, ibm_fez third (PREFERRED_QPUS below, or
--prefer). Each one's live status is checked right before submitting; a QPU that is paused, in
maintenance or offline is skipped. If the chosen QPU goes down while your jobs are still
queued, `python main.py --move` cancels the jobs that have not started (free) and sends them to
the next active QPU. `--backend ibm:<name>` forces one QPU; `--list-backends` shows them all.

The job IDs are written to results/ibm_jobs.json as soon as the jobs are submitted. If the
script is interrupted or the queue is long, run `python main.py` again: it picks the same jobs
up and does NOT submit (or spend) anything a second time. If a job failed or was cancelled,
`python main.py --retry` resubmits only that one. Use --new to force a completely fresh run.

What runs where
---------------
Only the quantum circuits run on the QPU. The robot simulation (kinematics, dynamics, PID) is
classical and runs on your computer: scripts/run_all.py and scripts/run_learning.py write the
problem instances into results/ and this script reads them. A QPU cannot sit inside the 100 Hz
control loop (each cloud job takes seconds to minutes), so the hardware run replays recorded
instances of every module and reports how close the device gets to the exact answer.

Credentials
-----------
The API key is never written into this repository. It is read from your saved Qiskit account
(--save-account) or from the QISKIT_IBM_TOKEN environment variable.

Tested with qiskit 2.6, qiskit-aer 0.18 and qiskit-ibm-runtime 0.52.
"""
from __future__ import annotations

import argparse
import getpass
import gzip
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import QAOAAnsatz, zz_feature_map
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import generate_preset_pass_manager

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from qkitchen import control as C                                    # noqa: E402
from qkitchen import hardware as H                                   # noqa: E402  (count analysis)
from qkitchen import qrl                                             # noqa: E402
from qkitchen.qaoa import DiagonalProblem, optimize_angles, qaoa_probs, term_union  # noqa: E402
from qkitchen.qml import VQCPolicy, ZZKernel, z_expectations_from_counts  # noqa: E402

OUT = ROOT / "results"
MANIFEST = OUT / "ibm_jobs.json"
CHANNEL = "ibm_quantum_platform"
# QPUs to use, in order of preference. The first one whose live status is "active" is taken;
# if none is, the least-busy active QPU on the account is used (you are asked to confirm).
PREFERRED_QPUS = ("ibm_marrakesh", "ibm_kingston", "ibm_fez")
WORKLOADS_URL = "https://quantum.cloud.ibm.com/workloads"
FINAL_STATES = ("DONE", "ERROR", "CANCELLED")


# =============================================================================================
# 1. IBM Quantum account and backend
# =============================================================================================
def save_account():
    """Store the API key on this computer (IBM 'hello world' guide, step "Authenticate").

    The key is typed at a hidden prompt (or taken from QISKIT_IBM_TOKEN), so it never appears in
    this file, in the shell history or in git. Qiskit keeps it in ~/.qiskit/qiskit-ibm.json.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("QISKIT_IBM_TOKEN") or getpass.getpass(
        "IBM Quantum API key (44 characters, input hidden): ").strip()
    instance = os.environ.get("QISKIT_IBM_INSTANCE")
    if not instance and sys.stdin.isatty():
        instance = input("Instance CRN (optional, press Enter to skip): ").strip() or None
    if not token:
        sys.exit("no API key given")
    QiskitRuntimeService.save_account(token=token, instance=instance, channel=CHANNEL,
                                      overwrite=True, set_as_default=True)
    print("account saved; from now on `python main.py` signs in automatically")


def connect():
    """service = QiskitRuntimeService(), from the saved account or QISKIT_IBM_TOKEN."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("QISKIT_IBM_TOKEN")
    try:
        if token:
            return QiskitRuntimeService(channel=CHANNEL, token=token,
                                        instance=os.environ.get("QISKIT_IBM_INSTANCE"))
        return QiskitRuntimeService()
    except Exception as err:                                           # noqa: BLE001
        env_hint = ('$env:QISKIT_IBM_TOKEN = "<API key>"   (PowerShell; optionally also '
                    '$env:QISKIT_IBM_INSTANCE = "<CRN>")' if os.name == "nt" else
                    "export QISKIT_IBM_TOKEN=<API key> (and optionally QISKIT_IBM_INSTANCE=<CRN>)")
        sys.exit(f"could not sign in to IBM Quantum: {err}\n"
                 "  run `python main.py --save-account` once (it asks for your API key), or\n"
                 f"  {env_hint}")


def qpu_status(backend) -> tuple[bool, str, object]:
    """(ready, status text, jobs in queue). Ready = online AND status "active"; a QPU that is
    paused, in maintenance or "internal" accepts jobs but does not run them."""
    try:
        st = backend.status()
    except Exception as err:                                           # noqa: BLE001
        return False, f"status unavailable ({err})", "?"
    msg = st.status_msg or ("online" if st.operational else "offline")
    return bool(st.operational) and msg == "active", msg, st.pending_jobs


def pick_qpu(service, prefer) -> object:
    """First QPU in `prefer` that is active right now; otherwise the least-busy active QPU."""
    print("[backend] checking QPU status (preference order: " + " > ".join(prefer) + ")")
    for name in prefer:
        try:
            b = service.backend(name)
        except Exception as err:                                       # noqa: BLE001
            print(f"  {name:16s} not available to this account ({err})")
            continue
        ready, msg, pend = qpu_status(b)
        print(f"  {name:16s} {msg:12s} jobs in queue: {pend}" + ("   <- selected" if ready else ""))
        if ready:
            return b
    try:
        b = service.least_busy(min_num_qubits=10, simulator=False, operational=True,
                               filters=lambda x: qpu_status(x)[0])
    except Exception:                                                  # noqa: BLE001
        sys.exit("no QPU on this account is active right now; nothing was submitted. Try again later,\n"
                 "  or force one with --backend ibm:<name> (the jobs then wait until it is back).")
    print(f"  none of the preferred QPUs is active; falling back to the least-busy active QPU: {b.name}")
    return b


def get_backend(spec: str, service=None, prefer=PREFERRED_QPUS):
    """'aer' | 'fake:<FakeName>' | 'ibm' (preferred QPUs, see PREFERRED_QPUS) | 'ibm:<name>'."""
    if spec == "aer":
        from qiskit_aer import AerSimulator
        return AerSimulator()
    if spec.startswith("fake:"):
        import qiskit_ibm_runtime.fake_provider as fake_provider
        return getattr(fake_provider, spec.split(":", 1)[1])()
    if spec.startswith("ibm"):
        name = spec.split(":", 1)[1] if ":" in spec else None
        if name:                                          # forced: use it, but say if it is down
            b = service.backend(name)
            ready, msg, pend = qpu_status(b)
            print(f"[backend] {name}: {msg}, jobs in queue: {pend}" +
                  ("" if ready else "  (not active: jobs will wait until it is back)"))
            return b
        return pick_qpu(service, prefer)
    raise ValueError(f"unknown backend '{spec}'")


def list_backends(service, prefer=PREFERRED_QPUS):
    print("QPUs available to this account (preference: " + " > ".join(prefer) + "):")
    for b in service.backends(simulator=False):
        ready, msg, pend = qpu_status(b)
        rank = f"preference {prefer.index(b.name) + 1}" if b.name in prefer else ""
        print(f"  {b.name:16s} {b.num_qubits:4d} qubits   {msg:12s} jobs in queue: {str(pend):6s} {rank}")


# =============================================================================================
# 2. The quantum circuits of the project
# =============================================================================================
def cost_hamiltonian(problem: DiagonalProblem) -> SparsePauliOp:
    """H_C = sum_S c_S prod_{i in S} Z_i, scaled so the largest |c_S| is 1 (Modules 1-2).

    The coefficients come from an exact Walsh-Hadamard transform of the cost table, so H_C
    reproduces the planner / IK cost of every bitstring exactly.
    """
    terms = [("Z" * len(sup), list(sup), c / problem.scale) for sup, c in problem.terms]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=problem.n)


def qaoa_circuit(n: int, supports: list, p: int, measure: bool = True) -> QuantumCircuit:
    """QAOA circuit for Module 1 (path HUBO) and Module 2 (inverse-kinematics QUBO).

    |+>^n, then p layers of exp(-i gamma H_C) and exp(-i beta sum_i X_i).
    Every Pauli-Z term gets its own Parameter theta = 2 * gamma * c_S, so ONE transpiled
    circuit serves all problem instances: they only differ in the parameter values.
      1-local term: RZ            2-local term: RZZ
      k-local term: CX ladder, RZ, CX ladder (Module 1 has terms up to 4-local)
    """
    theta = ParameterVector("theta", p * len(supports))
    beta = ParameterVector("beta", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        for j, sup in enumerate(supports):
            th = theta[layer * len(supports) + j]
            if len(sup) == 1:
                qc.rz(th, sup[0])
            elif len(sup) == 2:
                qc.rzz(th, sup[0], sup[1])
            else:
                ladder = list(zip(sup[:-1], sup[1:]))
                for a, b in ladder:
                    qc.cx(a, b)
                qc.rz(th, sup[-1])
                for a, b in reversed(ladder):
                    qc.cx(a, b)
        for q in range(n):
            qc.rx(2 * beta[layer], q)
    if measure:
        qc.measure_all()
    return qc


def qaoa_values(problem: DiagonalProblem, supports: list, gammas, betas) -> dict:
    """Parameter name -> value for one problem instance and one set of QAOA angles."""
    coeff = dict(problem.terms)
    vals = {}
    for layer, g in enumerate(gammas):
        for j, sup in enumerate(supports):
            vals[f"theta[{layer * len(supports) + j}]"] = 2 * g * coeff.get(sup, 0.0) / problem.scale
    vals.update({f"beta[{layer}]": b for layer, b in enumerate(betas)})
    return vals


def kernel_circuit(n_features: int, reps: int, measure: bool = True) -> QuantumCircuit:
    """Fidelity-kernel circuit for Module 3 (gain GP) and Module 4 (residual-dynamics GP).

    U(x) followed by U(y)^dagger, with U = Qiskit's ZZ feature map. The probability of reading
    all zeros equals the kernel k(x, y) = |<phi(y)|phi(x)>|^2.
    """
    fmap = zz_feature_map(n_features, reps=reps, entanglement="linear")
    x = ParameterVector("x", n_features)
    y = ParameterVector("y", n_features)
    ux = fmap.assign_parameters(dict(zip(fmap.parameters, x)))
    uy = fmap.assign_parameters(dict(zip(fmap.parameters, y)))
    qc = ux.compose(uy.inverse())
    if measure:
        qc.measure_all()
    return qc


def kernel_values(xa, xb) -> dict:
    """Parameter name -> value for one kernel entry (already-encoded feature vectors)."""
    vals = {f"x[{k}]": float(v) for k, v in enumerate(xa)}
    vals.update({f"y[{k}]": float(v) for k, v in enumerate(xb)})
    return vals


def policy_circuit(W, Bb, Cz, measure: bool = True) -> QuantumCircuit:
    """Variational policy circuit for Module 5 (data re-uploading, trained weights baked in).

    Each layer: RY(w * x_i + b) and RZ(c) on every qubit, then a ring of CZ gates. The six
    observation values x_i are the circuit Parameters; <Z_0> and <Z_1> give the k_p and k_d
    gain multipliers.
    """
    layers, n = W.shape
    x = ParameterVector("x", n)
    ring = [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    qc = QuantumCircuit(n)
    for layer in range(layers):
        for i in range(n):
            qc.ry(float(W[layer, i]) * x[i] + float(Bb[layer, i]), i)
            qc.rz(float(Cz[layer, i]), i)
        for i, j in ring:
            qc.cz(i, j)
    if measure:
        qc.measure_all()
    return qc


# =============================================================================================
# 3. Workloads: one parameter sweep per module
# =============================================================================================
@dataclass
class Workload:
    key: str                       # e.g. "m1_p1"
    title: str
    circuit: QuantumCircuit        # logical circuit with measurements
    values: list                   # one {parameter name: value} dict per instance
    analyse: Callable              # counts list -> metrics dict
    verify: Callable               # () -> max |Qiskit exact - project simulator|
    isa: QuantumCircuit | None = None
    stats: dict = field(default_factory=dict)

    def value_array(self, circuit=None) -> np.ndarray:
        """(n_instances, n_parameters), columns ordered as circuit.parameters."""
        ref = circuit if circuit is not None else (self.isa if self.isa is not None else self.circuit)
        names = [p.name for p in ref.parameters]
        return np.array([[v[n] for n in names] for v in self.values], float)

    def fingerprint(self) -> str:
        return hashlib.sha1(np.round(self.value_array(self.circuit), 9).tobytes()).hexdigest()[:16]


def _bound_statevector(circuit: QuantumCircuit, values: dict) -> Statevector:
    qc = circuit.remove_final_measurements(inplace=False)
    return Statevector(qc.assign_parameters({p: values[p.name] for p in qc.parameters}))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _need(path: Path):
    if not path.exists():
        sys.exit(f"missing {_rel(path)}: run `python main.py --rebuild` "
                 "(or scripts/run_all.py and scripts/run_learning.py) first")
    return path


def _qaoa_workload(key, title, problems, angles, p) -> Workload:
    supports = term_union(problems)
    n = problems[0].n
    qc = qaoa_circuit(n, supports, p)
    values = [qaoa_values(pr, supports, g, b) for pr, (g, b) in zip(problems, angles)]

    def analyse(counts):
        an = H.analyse_qaoa_counts(problems, counts)
        ideal_p, ideal_ar = [], []
        for pr, (g, b) in zip(problems, angles):
            pe = qaoa_probs(pr, g, b)
            ideal_p.append(pe[pr.ground_index]); ideal_ar.append(pr.approx_ratio(pe))
        return dict(p_opt=float(np.mean([a["p_opt"] for a in an])),
                    approx_ratio=float(np.mean([a["approx_ratio"] for a in an])),
                    opt_in_topk=float(np.mean([a["opt_in_topk"] for a in an])),
                    mean_best_gap=float(np.mean([a["best_gap"] for a in an])),
                    best_sample_is_optimum=int(sum(a["best_gap"] <= 1e-12 for a in an)),
                    ideal=dict(p_opt=float(np.mean(ideal_p)), approx_ratio=float(np.mean(ideal_ar))),
                    uniform_p_opt=1 / 2 ** n)

    def verify():
        pr, (g, b) = problems[0], angles[0]
        ours = _bound_statevector(qc, values[0]).probabilities()
        fast = qaoa_probs(pr, g, b)                                  # project's NumPy simulator
        ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian(pr), reps=p)   # Qiskit's library QAOA
        bind = {par: (g if par.name.startswith("γ") else b)[int(par.name.split("[")[1][:-1])]
                for par in ansatz.parameters}
        lib = Statevector(ansatz.assign_parameters(bind)).probabilities()
        return max(float(np.abs(ours - fast).max()), float(np.abs(ours - lib).max()))

    return Workload(key, title, qc, values, analyse, verify)


def _kernel_workload(key, title, n_features, reps, encode, exact_gram, X, n_train, pairs,
                     gp_check=None) -> Workload:
    qc = kernel_circuit(n_features, reps)
    enc = [np.asarray(encode(x), float).ravel() for x in X]
    values = [kernel_values(enc[i], enc[j]) for i, j in pairs]
    K = exact_gram(X)
    exact = np.array([K[i, j] for i, j in pairs])

    def analyse(counts):
        kv = H.kernel_from_counts(counts, n_features)
        out = dict(kernel_mae=float(np.abs(kv - exact).mean()),
                   kernel_max_err=float(np.abs(kv - exact).max()),
                   exact_offdiag_mean=float(exact.mean()))
        if gp_check is not None:
            Kd = np.eye(len(X))
            for (i, j), v in zip(pairs, kv):
                Kd[i, j] = Kd[j, i] = v
            out.update(gp_check(K, Kd))
        return out

    def verify():
        err = 0.0
        for k in range(0, len(pairs), max(1, len(pairs) // 6)):
            p0 = _bound_statevector(qc, values[k]).probabilities()[0]
            err = max(err, abs(p0 - exact[k]))
        return float(err)

    return Workload(key, title, qc, values, analyse, verify)


def build_workloads(a) -> dict:
    """Load the recorded problem instances and build one Workload per module and depth."""
    depths = [int(x) for x in str(a.depths).split(",")]
    ntr, nte = a.kernel_train, a.kernel_test
    pairs = [(i, j) for i in range(ntr) for j in range(i + 1, ntr)] + \
            [(ntr + i, j) for i in range(nte) for j in range(ntr)]
    W = {}

    # ---- Module 1: QAOA end-effector path planning (one instance per transfer / return leg)
    plan = pickle.load(open(_need(OUT / "plan.pkl"), "rb"))
    legs = [r.problem for r in plan["records"]]
    for p in depths:
        if p == 3:                                    # angles trained in scripts/run_all.py
            ang = [(np.array(r.gammas), np.array(r.betas)) for r in plan["records"]]
        else:
            ang = [optimize_angles([pr], p=p, restarts=4, mode="cvar")[:2] for pr in legs]
        W[f"m1_p{p}"] = _qaoa_workload(f"m1_p{p}", f"Module 1  QAOA path planning, p={p}", legs, ang, p)

    # ---- Module 2: QAOA inverse kinematics (recorded per-step QUBOs, transferred angles)
    metrics = json.load(open(_need(OUT / "metrics.json")))
    if (OUT / "ik_qubos.npy").exists():               # full set written by run_all.py
        Eq = np.load(OUT / "ik_qubos.npy")
        n_total, get_q = len(Eq), (lambda i: Eq[i])
    else:                                             # compact set shipped with the repo
        z = np.load(_need(OUT / "ik_qubos_sel.npz"))
        lut = dict(zip(z["idx"].tolist(), z["E"]))
        n_total, get_q = int(z["n_total"]), (lambda i: lut[int(i)])
    sel = np.linspace(0, n_total - 1, a.ik_instances).astype(int)
    try:
        ikp = [DiagonalProblem.from_energies(get_q(i)) for i in sel]
    except KeyError:
        sys.exit("this --ik-instances count needs results/ik_qubos.npy: run `python main.py "
                 "--rebuild` first, or use 12, 20, 40, 80 or 100")
    tr = metrics["module2"]["transfer"]
    for p in depths:
        t = tr[str(p)] if str(p) in tr else tr[p]
        ang = [(np.array(t["gammas"]), np.array(t["betas"]))] * len(ikp)
        W[f"m2_p{p}"] = _qaoa_workload(f"m2_p{p}", f"Module 2  QAOA inverse kinematics, p={p}", ikp, ang, p)

    # ---- Module 3: quantum-kernel GP for PID gain scheduling
    qkd = pickle.load(open(_need(OUT / "qkgp.pkl"), "rb"))
    qk3 = C.QuantumKernel(4, bandwidth=qkd["bandwidth"])
    Xa, Ya, Xb, Yb = qkd["Xtr"][:ntr], qkd["Ytr"][:ntr], qkd["Xte"][:nte], qkd["Yte"][:nte]

    def gp_check(K, Kd):
        sig = float(np.mean(np.var(Ya, 0))) + 1e-9
        pe = C.GP(noise=qkd["noise"] * sig, signal=sig).fit(K[:ntr, :ntr], Ya).predict(K[ntr:, :ntr])
        pd = C.GP(noise=qkd["noise"] * sig, signal=sig).fit(Kd[:ntr, :ntr], Ya).predict(Kd[ntr:, :ntr])
        return dict(gp_pred_diff_rmse=float(np.sqrt(np.mean((pe - pd) ** 2))),
                    gp_rmse_exact_kernel=float(np.sqrt(np.mean((pe - Yb) ** 2))),
                    gp_rmse_device_kernel=float(np.sqrt(np.mean((pd - Yb) ** 2))))

    W["m3_kernel"] = _kernel_workload("m3_kernel", "Module 3  quantum kernel (gain-scheduling GP)",
                                      4, 2, qk3.encode, qk3.gram, np.vstack([Xa, Xb]), ntr, pairs,
                                      gp_check)

    # ---- Modules 4-5 need scripts/run_learning.py
    lm_f = OUT / "learning_metrics.json"
    lm = json.load(open(lm_f)) if lm_f.exists() else {}
    if "module4" in lm and (OUT / "m4_residual_logs.npz").exists():
        hyp = lm["module4"]["qk_hyper"][0]            # joint 1 (base) residual GP
        zk = ZZKernel(4, reps=hyp["reps"], bandwidth=np.array(hyp["bandwidth"]))
        z = np.load(OUT / "m4_residual_logs.npz")
        rs = np.random.default_rng(0)
        X4 = np.vstack([z["Xtr"][rs.choice(len(z["Xtr"]), ntr, replace=False), 0],
                        z["Xte"][rs.choice(len(z["Xte"]), nte, replace=False), 0]])
        W["m4_kernel"] = _kernel_workload("m4_kernel", "Module 4  quantum kernel (residual-dynamics GP)",
                                          4, hyp["reps"], lambda f: zk.encode(f)[0], zk.gram, X4, ntr, pairs)
    if "module5" in lm and (OUT / "m5_agents.pkl").exists():
        from scipy.ndimage import uniform_filter1d
        name = f"vqc_{a.policy_seed}"
        vq = VQCPolicy(6, 3, 2, seed=a.policy_seed)
        vq.set(np.asarray(pickle.load(open(OUT / "m5_agents.pkl", "rb"))[name]))
        cl = np.load(OUT / "m5_closed_loop.npz", allow_pickle=True)
        tt = np.load(OUT / "test_traj.npz")
        qd, qdd = C.reference_derivatives(tt["Q"])
        feats = uniform_filter1d(C.features(tt["Q"], qd, qdd, 0.6 * tt["carry"]), 21, axis=0)
        ks = np.linspace(10, len(feats) - 10, a.policy_obs).astype(int)
        e = cl["Qref"] - cl["Q"][[str(n) for n in cl["names"]].index(name)]
        ed = np.gradient(e, 0.01, axis=0)
        O5 = np.vstack([qrl.observation(feats[k], e[k][None], ed[k][None]) for k in ks])
        qc5 = policy_circuit(vq.W, vq.Bb, vq.C)
        values5 = [{f"x[{i}]": float(v) for i, v in enumerate(o)} for o in O5]
        Z_exact = vq.forward(O5)

        def analyse5(counts):
            Zd = np.array([z_expectations_from_counts(c, 2) for c in counts])
            gain_ratio = np.exp(qrl.LOG_HALF * (Zd - Z_exact))       # device gain / exact gain
            return dict(z_mae=float(np.abs(Zd - Z_exact).mean()),
                        gain_rel_err=float(np.mean(np.abs(gain_ratio - 1))),
                        z_ideal_std=float(Z_exact.std()), policy=name)

        def verify5():
            err = 0.0
            for k in range(0, len(O5), max(1, len(O5) // 6)):
                pr = _bound_statevector(qc5, values5[k]).probabilities()
                bits = (np.arange(len(pr))[:, None] >> np.arange(2)) & 1
                err = max(err, float(np.abs(pr @ (1 - 2 * bits) - Z_exact[k]).max()))
            return err

        W["m5_policy"] = Workload("m5_policy", f"Module 5  VQC gain policy (seed {a.policy_seed})",
                                  qc5, values5, analyse5, verify5)

    keep = tuple(str(a.modules).split(","))
    return {k: w for k, w in W.items() if k.startswith(keep)}


# =============================================================================================
# 4. Verify, transpile, estimate
# =============================================================================================
def verify_all(W: dict, tol: float = 1e-6):
    """Free pre-flight check: Qiskit's exact Statevector of every circuit against the fast
    simulator used in the closed-loop robot simulation (and, for QAOA, against Qiskit's own
    QAOAAnsatz on the same Hamiltonian)."""
    print("\n[check] each circuit vs exact Statevector, the project's simulator and Qiskit's QAOAAnsatz")
    ok = True
    for w in W.values():
        err = w.verify()
        ok &= err < tol
        print(f"  {w.key:10s} max difference {err:.1e}  {'ok' if err < tol else 'MISMATCH'}")
    if not ok:
        sys.exit("a circuit does not reproduce the exact result; nothing was submitted")


def draw_all(W: dict):
    """Write a diagram of every logical circuit to figures/circuits/ (text; PNG if available)."""
    out = ROOT / "figures" / "circuits"
    out.mkdir(parents=True, exist_ok=True)
    for w in W.values():
        (out / f"{w.key}.txt").write_text(f"{w.title}\n\n{w.circuit.draw('text', fold=160)}\n",
                                          encoding="utf-8")
        try:                                           # needs matplotlib + pylatexenc
            w.circuit.draw("mpl", fold=60).savefig(out / f"{w.key}.png", dpi=110)
        except Exception:                                              # noqa: BLE001
            pass
    print(f"  circuit diagrams written to {_rel(out)}/")


def circuit_stats(isa: QuantumCircuit) -> dict:
    ops = isa.count_ops()
    twoq = sum(v for k, v in ops.items() if k in ("cz", "ecr", "cx", "rzz"))
    return dict(depth=int(isa.depth()), two_qubit_gates=int(twoq), size=int(isa.size()))


def transpile_all(W: dict, backend, seeds: int = 3, level: int = 3):
    """Map every circuit to the backend's qubits and native gates (ISA circuits). Several
    transpiler seeds are tried and the circuit with the fewest two-qubit gates is kept."""
    print(f"\n[transpile] optimisation level {level}, best of {seeds} seeds")
    for w in W.values():
        best = None
        for seed in range(seeds):
            pm = generate_preset_pass_manager(optimization_level=level, backend=backend,
                                              seed_transpiler=seed)
            isa = pm.run(w.circuit)
            st = circuit_stats(isa)
            if best is None or (st["two_qubit_gates"], st["depth"]) < (best[1]["two_qubit_gates"], best[1]["depth"]):
                best = (isa, st)
        w.isa, w.stats = best
        print(f"  {w.key:10s} {w.circuit.num_qubits} qubits  {len(w.values):4d} instances  "
              f"depth {w.stats['depth']:5d}  two-qubit gates {w.stats['two_qubit_gates']:5d}")


def estimate_seconds(w: Workload, shots: int) -> float:
    """Rough QPU time: shots x (repetition delay + circuit) plus a fixed overhead per job."""
    return 2.0 + len(w.values) * shots * 300e-6


# =============================================================================================
# 5. Run: local simulators, or one batch on IBM hardware
# =============================================================================================
def make_sampler(mode, a, hardware_like: bool, cap_seconds: int | None = None):
    if a.server_side:                      # legacy server-side primitive, if you need it
        from qiskit_ibm_runtime.server_side import Sampler
    else:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    sampler = Sampler(mode=mode)
    if hardware_like and a.dd:
        sampler.options.dynamical_decoupling.enable = True
    if hardware_like and a.twirling:
        sampler.options.twirling.enable_gates = True
    if cap_seconds:
        sampler.options.max_execution_time = int(cap_seconds)
    return sampler


def counts_from_result(result, n_instances: int) -> list:
    data = result[0].data
    bits = getattr(data, "meas", None)
    if bits is None:                                   # any other classical register name
        bits = next(v for v in data.values())
    return [bits[i].get_counts() for i in range(n_instances)]


def run_local(W: dict, backend, a) -> dict:
    """Aer / fake backends: the same Runtime Sampler code path, executed on this computer."""
    print("\n[run] local simulation")
    out = {}
    for w in W.values():
        t0 = time.time()
        sampler = make_sampler(backend, a, hardware_like=a.backend.startswith("fake:"))
        job = sampler.run([(w.isa, w.value_array(), a.shots)])
        out[w.key] = dict(counts=counts_from_result(job.result(), len(w.values)),
                          meta=dict(wall_s=time.time() - t0))
        print(f"  {w.key:10s} done in {time.time() - t0:5.0f} s")
    return out


def _status(job) -> str:
    s = job.status()
    return str(getattr(s, "name", s)).upper()


def _qpu_seconds(job):
    try:
        return float(job.metrics()["usage"]["quantum_seconds"])
    except Exception:                                                  # noqa: BLE001
        try:
            return float(job.usage())
        except Exception:                                              # noqa: BLE001
            return None


def submit_ibm(W: dict, backend, service, a, manifest: dict | None = None) -> dict:
    """Submit one Sampler job per module inside a Batch and write the job IDs to disk.

    With `manifest` (a retry), only the given workloads are submitted and their entries in the
    existing manifest are replaced.
    """
    from contextlib import nullcontext

    from qiskit_ibm_runtime import Batch
    est = {k: estimate_seconds(w, a.shots) for k, w in W.items()}
    total = sum(est.values())
    print(f"\n[budget] estimated QPU time {total:.0f} s ({total / 60:.1f} min) for {len(W)} jobs "
          f"on {backend.name}; limit --budget-seconds {a.budget_seconds}")
    try:
        left = service.usage().get("usage_remaining_seconds")
        if left is not None:
            print(f"         remaining on this instance: {left:.0f} s")
            if total > left:
                sys.exit("the estimate exceeds what is left on this instance; reduce --shots, "
                         "--ik-instances or --kernel-train, or drop modules with --modules")
    except Exception:                                                  # noqa: BLE001
        pass
    if total > a.budget_seconds:
        sys.exit("estimate above --budget-seconds; nothing submitted")
    if backend.name not in a.prefer:
        print(f"  note: {backend.name} is not one of your preferred QPUs ({', '.join(a.prefer)})")
    if not a.yes:
        if not sys.stdin.isatty():
            sys.exit("add --yes to submit without the confirmation prompt")
        if input(f"Submit {len(W)} jobs to {backend.name}? This spends QPU time. [y/N] ").strip().lower() != "y":
            sys.exit("nothing submitted")

    if manifest is None:
        manifest = dict(backend=backend.name, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                        args={k: getattr(a, k) for k in WORKLOAD_ARGS}, jobs={})
    manifest["collected"] = False
    jobs = {}
    print("\n[submit]")
    # Batch mode keeps the jobs together on the QPU; --job-mode queues each one separately.
    with (nullcontext(backend) if a.job_mode else Batch(backend=backend)) as mode:
        manifest["batch_id"] = getattr(mode, "session_id", None)
        for w in W.values():
            cap = max(300, int(4 * est[w.key]))        # per-job ceiling on QPU time
            sampler = make_sampler(mode, a, hardware_like=True, cap_seconds=cap)
            job = sampler.run([(w.isa, w.value_array(), a.shots)])
            jobs[w.key] = job
            manifest["jobs"][w.key] = dict(job_id=job.job_id(), backend=backend.name,
                                           n_instances=len(w.values),
                                           fingerprint=w.fingerprint(), circuit=w.stats,
                                           est_qpu_s=est[w.key])
            json.dump(manifest, open(MANIFEST, "w"), indent=1, default=str)   # saved after every job
            print(f"  {w.key:10s} job {job.job_id()}")
    print(f"  job IDs saved to {_rel(MANIFEST)}; the jobs now appear on {WORKLOADS_URL}")

    # the service's own estimate, when it provides one; cancel if it is over budget
    server = []
    for job in jobs.values():
        try:
            server.append(job.usage_estimation.get("quantum_seconds"))
        except Exception:                                              # noqa: BLE001
            server.append(None)
    if server and all(s is not None for s in server):
        tot = float(sum(server))
        print(f"  IBM's own estimate for these jobs: {tot:.0f} s")
        if tot > a.budget_seconds:
            for job in jobs.values():
                try:
                    job.cancel()
                except Exception:                                      # noqa: BLE001
                    pass
            MANIFEST.replace(MANIFEST.with_name("ibm_jobs_cancelled.json"))
            sys.exit("IBM's estimate is above --budget-seconds: the jobs were cancelled before running")
    return jobs


def move_queued_jobs(jobs: dict, manifest: dict, service):
    """--move: cancel the jobs that are still queued on a QPU that is no longer active.
    Cancelling a job that has not started costs no QPU time; --retry then resubmits it."""
    down = {}
    for name in {manifest["jobs"][k].get("backend", manifest["backend"]) for k in jobs}:
        ready, msg, _ = qpu_status(service.backend(name))
        if not ready:
            down[name] = msg
    if not down:
        print("[move] every QPU of this run is active, so nothing is moved; the jobs keep their place in the queue")
        return
    for k, job in jobs.items():
        name = manifest["jobs"][k].get("backend", manifest["backend"])
        st = _status(job)
        if name in down and st in ("QUEUED", "INITIALIZING"):
            job.cancel()
            print(f"[move] {k}: cancelled on {name} ('{down[name]}') before it started")
        elif name in down:
            print(f"[move] {k}: {st} on {name}, left as it is")


def collect_ibm(jobs: dict, W: dict) -> dict:
    """Wait for the jobs (queue + execution) and pull the counts."""
    print("\n[wait] queued on IBM Quantum. You can stop this script with Ctrl-C and run "
          "`python main.py` later: it resumes from the saved job IDs without resubmitting.")
    print(f"        The jobs are also listed, with their results, on {WORKLOADS_URL}")
    last, pending, out = {}, dict(jobs), {}
    warned, next_check = set(), time.time() + 600
    while pending:
        if time.time() > next_check:                  # every 10 min: did a QPU go down?
            next_check = time.time() + 600
            for k, job in pending.items():
                if last.get(k) != "QUEUED":
                    continue
                try:
                    b = job.backend()
                    ready, msg, _ = qpu_status(b)
                    if not ready and b.name not in warned:
                        warned.add(b.name)
                        print(f"  {time.strftime('%H:%M:%S')}  {b.name} is now '{msg}'. Your queued jobs wait "
                              "until it is back. To move them to the next active QPU instead: press Ctrl-C, "
                              "then run `python main.py --move`.")
                except Exception:                                      # noqa: BLE001
                    pass
        for k, job in list(pending.items()):
            try:
                st = _status(job)
            except Exception as err:                                   # noqa: BLE001
                st = f"UNKNOWN ({err})"
            if st != last.get(k):
                print(f"  {time.strftime('%H:%M:%S')}  {k:10s} {st}")
                last[k] = st
            if st in FINAL_STATES:
                pending.pop(k)
                if st == "DONE":
                    result = job.result()
                    try:
                        counts = counts_from_result(result, len(W[k].values))
                    except Exception as err:                           # noqa: BLE001
                        # the QPU time is already spent: keep the raw result so nothing is lost
                        raw = OUT / f"ibm_raw_{k}.pkl"
                        pickle.dump(result, open(raw, "wb"))
                        print(f"    {k}: could not read the counts ({err}); raw result saved to {_rel(raw)}")
                        continue
                    out[k] = dict(counts=counts,
                                  meta=dict(job_id=job.job_id(), qpu_seconds=_qpu_seconds(job)))
                else:
                    msg = ""
                    try:
                        msg = job.error_message() or ""
                    except Exception:                                  # noqa: BLE001
                        pass
                    print(f"    {k} did not finish ({st}) {msg}")
        if pending:
            time.sleep(20)
    return out


# =============================================================================================
# 6. Analyse and report
# =============================================================================================
def tag_of(spec: str) -> str:
    return spec.replace(":", "_")


def analyse_and_save(W: dict, results: dict, a, backend_name: str) -> dict:
    print("\n[results]")
    report = dict(backend=backend_name, shots=a.shots, modules={},
                  options=dict(dynamical_decoupling=bool(a.dd), twirling=bool(a.twirling)))
    for k, w in W.items():
        if k not in results:
            continue
        m = dict(circuit=w.stats, n_instances=len(w.values), meta=results[k]["meta"])
        m.update(w.analyse(results[k]["counts"]))
        report["modules"][k] = m
        if k.startswith(("m1", "m2")):
            print(f"  {k:10s} P(optimum) {m['p_opt']:.4f}  (exact {m['ideal']['p_opt']:.4f}, uniform "
                  f"{m['uniform_p_opt']:.4f})   best sample = optimum in {m['best_sample_is_optimum']}"
                  f"/{len(w.values)} instances")
        elif "kernel" in k:
            extra = (f"   GP error with exact kernel {m['gp_rmse_exact_kernel']:.3f}, with device kernel "
                     f"{m['gp_rmse_device_kernel']:.3f}") if "gp_rmse_exact_kernel" in m else ""
            print(f"  {k:10s} kernel mean abs error {m['kernel_mae']:.4f} "
                  f"(mean entry {m['exact_offdiag_mean']:.2f}){extra}")
        else:
            print(f"  {k:10s} gain error {100 * m['gain_rel_err']:.1f} %  (<Z> mean abs error {m['z_mae']:.4f})")
    used = [r["meta"].get("qpu_seconds") for r in results.values() if r["meta"].get("qpu_seconds")]
    if used:
        report["qpu_seconds_total"] = float(sum(used))
        print(f"  QPU time used: {sum(used):.0f} s")
    tag = tag_of(a.backend)
    fn = OUT / f"hardware_{tag}.json"
    if fn.exists() and not a.modules == DEFAULTS["modules"]:      # partial run: merge
        old = json.load(open(fn))
        old["modules"].update(report["modules"])
        report["modules"] = old["modules"]
    json.dump(report, open(fn, "w"), indent=1, default=float)
    cf = OUT / f"hardware_{tag}_counts.json.gz"          # raw samples, for later analysis
    counts = {k: r["counts"] for k, r in results.items()}
    if cf.exists() and not a.modules == DEFAULTS["modules"]:      # partial run: keep the others
        with gzip.open(cf, "rt") as f:
            counts = {**json.load(f), **counts}
    with gzip.open(cf, "wt") as f:
        json.dump(counts, f)
    print(f"  wrote {_rel(fn)} and the raw counts next to it")
    return report


def print_comparison():
    table = comparison_table()
    if table:
        print("\n" + table)
    return table


def comparison_table() -> str:
    """Ideal vs noisy vs hardware (Markdown), from every results/hardware_*.json that exists."""
    reps = {}
    if (OUT / "hardware_aer.json").exists():
        reps["ideal Aer"] = json.load(open(OUT / "hardware_aer.json"))
    for f in sorted(OUT.glob("hardware_fake_*.json")):
        if "dryrun" not in f.name:
            reps[f"noisy {f.stem.split('fake_', 1)[1]}"] = json.load(open(f))
    for f in sorted(OUT.glob("hardware_ibm*.json")):
        if "dryrun" not in f.name:
            r = json.load(open(f))
            reps[f"IBM {r['backend']}"] = r
    if not reps:
        return ""
    lines = []
    rows = [("m1_p1", "M1 P(opt), p=1", "p_opt"), ("m1_p3", "M1 P(opt), p=3", "p_opt"),
            ("m2_p1", "M2 P(opt), p=1", "p_opt"), ("m2_p3", "M2 P(opt), p=3", "p_opt"),
            ("m3_kernel", "M3 kernel MAE", "kernel_mae"), ("m4_kernel", "M4 kernel MAE", "kernel_mae"),
            ("m5_policy", "M5 gain error", "gain_rel_err")]
    lines.append("| | exact | " + " | ".join(reps) + " | uniform |")
    lines.append("|---|---|" + "---|" * (len(reps) + 1))
    for key, label, fld in rows:
        mods = [r["modules"].get(key) for r in reps.values()]
        ref = next((m for m in mods if m), None)
        if ref is None:
            continue
        exact = f"{ref['ideal']['p_opt']:.4f}" if fld == "p_opt" else "0"
        unif = f"{ref['uniform_p_opt']:.4f}" if fld == "p_opt" else "–"
        fmt = (lambda v: f"{100 * v:.1f} %") if fld == "gain_rel_err" else (lambda v: f"{v:.4f}" if fld == "p_opt" else f"{v:.3f}")
        cells = [fmt(m[fld]) if m and fld in m else "–" for m in mods]
        lines.append(f"| {label} | {exact} | " + " | ".join(cells) + f" | {unif} |")
    return "\n".join(lines)


def write_summary(report: dict, a):
    """results/hardware_summary.md: every result of the IBM run in one readable file."""
    man = json.load(open(MANIFEST)) if MANIFEST.exists() else {"jobs": {}}
    L = ["# IBM Quantum hardware run", "",
         f"* Submitted: {man.get('created', '?')}; analysed: {time.strftime('%Y-%m-%d %H:%M:%S')}",
         f"* QPU: {report['backend']}; shots: {report['shots']}; dynamical decoupling: "
         f"{report['options']['dynamical_decoupling']}; twirling: {report['options']['twirling']}",
         f"* QPU time used: {report.get('qpu_seconds_total', 0):.0f} s",
         f"* Jobs are also on the IBM Quantum Workloads page: {WORKLOADS_URL}", "",
         "## Jobs", "", "| module | job ID | QPU | status | QPU seconds | two-qubit gates |",
         "|---|---|---|---|---|---|"]
    for k, info in man["jobs"].items():
        m = report["modules"].get(k)
        st = "done" if m else "missing (run `python main.py --retry`)"
        qs = (m or {}).get("meta", {}).get("qpu_seconds")
        L.append(f"| {k} | `{info['job_id']}` | {info.get('backend', man.get('backend'))} | {st} | "
                 f"{'' if qs is None else f'{qs:.1f}'} | {info.get('circuit', {}).get('two_qubit_gates', '')} |")
    L += ["", "## Ideal vs noisy vs hardware", "", comparison_table(), "",
          "## Per module", ""]
    for k, m in report["modules"].items():
        if k.startswith(("m1", "m2")):
            L.append(f"* **{k}**: P(optimum) {m['p_opt']:.4f} (exact {m['ideal']['p_opt']:.4f}, uniform "
                     f"{m['uniform_p_opt']:.4f}); approximation ratio {m['approx_ratio']:.3f} (exact "
                     f"{m['ideal']['approx_ratio']:.3f}); best sample = optimum in "
                     f"{m['best_sample_is_optimum']}/{m['n_instances']} instances")
        elif "kernel" in k:
            gp = (f"; GP error with exact kernel {m['gp_rmse_exact_kernel']:.3f}, with device kernel "
                  f"{m['gp_rmse_device_kernel']:.3f}") if "gp_rmse_exact_kernel" in m else ""
            L.append(f"* **{k}**: kernel mean abs error {m['kernel_mae']:.4f}, max {m['kernel_max_err']:.3f} "
                     f"(mean exact entry {m['exact_offdiag_mean']:.2f}){gp}")
        else:
            L.append(f"* **{k}**: gain error {100 * m['gain_rel_err']:.1f} %, <Z> mean abs error {m['z_mae']:.4f}")
    tag = tag_of(a.backend)
    L += ["", "## Files", "",
          f"* `results/hardware_{tag}.json`: all metrics above, plus circuit sizes",
          f"* `results/hardware_{tag}_counts.json.gz`: every measured bitstring count",
          "* `results/ibm_jobs.json`: job IDs (retrieve any job later with `service.job(<id>)`)",
          "* `figures/fig5_hardware.png`: ideal vs noisy vs hardware chart", ""]
    (OUT / "hardware_summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {_rel(OUT / 'hardware_summary.md')} (all results of this run in one file)")


def update_figure():
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import make_figures
        make_figures.fig_hardware()
        print("  updated figures/fig5_hardware.png")
    except Exception as err:                                           # noqa: BLE001
        print(f"  (figure not updated: {err})")


# =============================================================================================
# 7. Command line
# =============================================================================================
DEFAULTS = dict(shots=1024, ik_instances=40, kernel_train=16, kernel_test=8, depths="1,3",
                policy_obs=40, policy_seed=2, modules="m1,m2,m3,m4,m5")
WORKLOAD_ARGS = tuple(DEFAULTS)            # everything that defines what is sent to the QPU


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="ibm",
                    help="ibm (first active QPU of --prefer, default) | ibm:<name> (force one QPU) "
                         "| aer | fake:FakeMarrakesh | fake:FakeKingston | fake:FakeFez")
    ap.add_argument("--prefer", default=",".join(PREFERRED_QPUS),
                    help="QPU preference order for --backend ibm (default: %(default)s)")
    ap.add_argument("--save-account", action="store_true", help="store your IBM Quantum API key and exit")
    ap.add_argument("--list-backends", action="store_true", help="show the QPUs of your account and exit")
    ap.add_argument("--draw", action="store_true", help="also write circuit diagrams to figures/circuits/")
    ap.add_argument("--dry-run", action="store_true", help="build, check, transpile and estimate; submit nothing")
    ap.add_argument("--new", action="store_true", help="ignore results/ibm_jobs.json and submit a fresh run")
    ap.add_argument("--yes", action="store_true", help="do not ask before submitting")
    ap.add_argument("--retry", action="store_true",
                    help="resubmit only the jobs of the saved run that failed or were cancelled")
    ap.add_argument("--move", action="store_true",
                    help="if the QPU of a saved run is no longer active, move its queued jobs to the next active QPU")
    ap.add_argument("--job-mode", action="store_true",
                    help="submit plain jobs instead of one batch (each job queues on its own)")
    ap.add_argument("--budget-seconds", type=float, default=360, help="refuse to submit above this QPU-time estimate")
    ap.add_argument("--rebuild", action="store_true",
                    help="first rerun the classical pipeline (scripts/run_all.py, run_learning.py; ~50 min CPU)")
    ap.add_argument("--shots", type=int, default=DEFAULTS["shots"])
    ap.add_argument("--ik-instances", type=int, default=DEFAULTS["ik_instances"])
    ap.add_argument("--kernel-train", type=int, default=DEFAULTS["kernel_train"])
    ap.add_argument("--kernel-test", type=int, default=DEFAULTS["kernel_test"])
    ap.add_argument("--depths", default=DEFAULTS["depths"], help="QAOA depths p, comma separated")
    ap.add_argument("--policy-obs", type=int, default=DEFAULTS["policy_obs"])
    ap.add_argument("--policy-seed", type=int, default=DEFAULTS["policy_seed"], help="which trained VQC policy (0-2)")
    ap.add_argument("--modules", default=DEFAULTS["modules"], help="comma list, e.g. m1,m3")
    ap.add_argument("--transpile-seeds", type=int, default=3)
    ap.add_argument("--dd", action="store_true", help="dynamical decoupling (error suppression) on hardware")
    ap.add_argument("--twirling", action="store_true", help="Pauli gate twirling on hardware")
    ap.add_argument("--server-side", action="store_true", help="use the legacy server-side Sampler")
    return ap.parse_args()


def main():
    a = parse_args()
    a.prefer = tuple(n.strip() for n in str(a.prefer).split(",") if n.strip())
    if a.save_account:
        return save_account()
    is_ibm = a.backend.startswith("ibm")
    service = None
    if a.list_backends:
        return list_backends(connect(), a.prefer)
    if a.rebuild:
        for script in ("run_all.py", "run_learning.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], check=True)

    # resume an earlier submission instead of spending QPU time twice
    manifest = json.load(open(MANIFEST)) if (is_ibm and MANIFEST.exists() and not a.new) else None
    if manifest and not a.dry_run:
        if manifest.get("collected"):
            print(f"results of the run submitted {manifest['created']} on {manifest['backend']} are already "
                  f"in results/. Use --new to submit again (this spends QPU time).")
            return print_comparison()
        print(f"resuming the run submitted {manifest['created']} on {manifest['backend']}")
        for k, v in manifest["args"].items():
            setattr(a, k, v)

    print("[build] loading problem instances and building the Qiskit circuits")
    W = build_workloads(a)
    for w in W.values():
        print(f"  {w.key:10s} {w.title}: {w.circuit.num_qubits} qubits, {len(w.values)} instances, "
              f"{w.circuit.num_parameters} parameters")
    verify_all(W)
    if a.draw:
        draw_all(W)

    if is_ibm:
        service = connect()
    if manifest and not a.dry_run:
        jobs = {}
        for k, info in manifest["jobs"].items():
            if k in W:
                if info["fingerprint"] != W[k].fingerprint():
                    sys.exit(f"{k}: results/ changed since submission; the saved jobs no longer match")
                jobs[k] = service.job(info["job_id"])
                W[k].stats = info.get("circuit", {})
        if a.move:
            move_queued_jobs(jobs, manifest, service)
            a.retry = True
        results = collect_ibm(jobs, W)
        backend_name = manifest["backend"]
        missing = {k: W[k] for k in manifest["jobs"] if k in W and k not in results}
        if missing and a.retry:
            order = [backend_name] + [n for n in a.prefer if n != backend_name]
            backend = pick_qpu(service, order)
            transpile_all(missing, backend, a.transpile_seeds)
            results.update(collect_ibm(submit_ibm(missing, backend, service, a, manifest), missing))
        elif missing:
            print(f"\n  {', '.join(missing)} did not finish. `python main.py --retry` resubmits only "
                  "those (jobs that never ran cost no QPU time).")
    else:
        backend = get_backend(a.backend, service, a.prefer)
        backend_name = getattr(backend, "name", str(backend))
        print(f"\n[backend] using {backend_name}")
        transpile_all(W, backend, a.transpile_seeds)
        est = sum(estimate_seconds(w, a.shots) for w in W.values())
        if a.dry_run:
            print(f"\n[dry run] {sum(len(w.values) for w in W.values())} circuit executions x {a.shots} shots; "
                  f"estimated QPU time {est:.0f} s ({est / 60:.1f} min). Nothing was submitted.")
            json.dump(dict(backend=backend_name, shots=a.shots, est_qpu_s=est,
                           modules={k: dict(circuit=w.stats, n_instances=len(w.values),
                                            est_qpu_s=estimate_seconds(w, a.shots)) for k, w in W.items()}),
                      open(OUT / f"hardware_{tag_of(a.backend)}_dryrun.json", "w"), indent=1)
            return
        if is_ibm:
            results = collect_ibm(submit_ibm(W, backend, service, a), W)
            if len(results) < len(W):
                print("\n  some jobs did not finish. `python main.py --retry` resubmits only those "
                      "(jobs that never ran cost no QPU time).")
        else:
            results = run_local(W, backend, a)

    if is_ibm and MANIFEST.exists():                  # which QPU ran each module (retries may differ)
        jobs_info = json.load(open(MANIFEST))["jobs"]
        for k, r in results.items():
            r["meta"]["backend"] = jobs_info.get(k, {}).get("backend", backend_name)
        used = sorted({r["meta"]["backend"] for r in results.values()})
        if used:
            backend_name = " + ".join(used)         # the QPU(s) that actually produced the results
        if len(used) > 1:
            print(f"\n  note: the modules ran on different QPUs ({backend_name}); see meta.backend per module")
    analyse_and_save(W, results, a, backend_name)
    if is_ibm and MANIFEST.exists():
        m = json.load(open(MANIFEST))
        m["collected"] = len(results) == len(m["jobs"])
        json.dump(m, open(MANIFEST, "w"), indent=1, default=str)
    report = json.load(open(OUT / f"hardware_{tag_of(a.backend)}.json"))
    update_figure()
    print_comparison()
    if is_ibm:
        write_summary(report, a)
        print(f"  the jobs and their results are also on {WORKLOADS_URL}")


if __name__ == "__main__":
    main()
