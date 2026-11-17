"""Ideal vs noisy vs real-hardware validation of all three quantum modules.

Run scripts/run_all.py first (it writes the problem instances to results/).

    python scripts/hardware_validation.py --backend aer
    python scripts/hardware_validation.py --backend fake:FakeFez --ik-instances 12
    python scripts/hardware_validation.py --backend ibm --dry-run      # transpile + budget only
    python scripts/hardware_validation.py --backend ibm                # least-busy real QPU
    python scripts/hardware_validation.py --backend ibm:ibm_kingston

Credentials: export QISKIT_IBM_TOKEN=<your API key> (and optionally QISKIT_IBM_INSTANCE=<CRN>),
or call QiskitRuntimeService.save_account(...) once. Never put the key in this file.
Each module is ONE SamplerV2 job (a single PUB with a parameter sweep); on IBM the
three jobs go out inside a Batch (Open Plan supports job and batch mode, not sessions).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from qkitchen import control as C                       # noqa: E402
from qkitchen import hardware as H                      # noqa: E402
from qkitchen.qaoa import DiagonalProblem, optimize_angles, qaoa_probs  # noqa: E402

OUT = ROOT / "results"


def ideal_summary(problems, angles):
    po, ar = [], []
    for pr, (g, b) in zip(problems, angles):
        p = qaoa_probs(pr, g, b)
        po.append(p[pr.ground_index]); ar.append(pr.approx_ratio(p))
    return dict(p_opt=float(np.mean(po)), approx_ratio=float(np.mean(ar)))


def summarise(an):
    return dict(p_opt=float(np.mean([a["p_opt"] for a in an])),
                approx_ratio=float(np.mean([a["approx_ratio"] for a in an])),
                opt_in_topk=float(np.mean([a["opt_in_topk"] for a in an])),
                mean_best_gap=float(np.mean([a["best_gap"] for a in an])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="aer")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--ik-instances", type=int, default=40)
    ap.add_argument("--kernel-train", type=int, default=16)
    ap.add_argument("--kernel-test", type=int, default=8)
    ap.add_argument("--depths", default="1,3")
    ap.add_argument("--policy-obs", type=int, default=40)
    ap.add_argument("--modules", default="m1,m2,m3,m4,m5",
                    help="comma list; results are merged into an existing report for this backend")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]

    metrics = json.load(open(OUT / "metrics.json"))
    plan = pickle.load(open(OUT / "plan.pkl", "rb"))
    if (OUT / "ik_qubos.npy").exists():          # full set written by run_all.py
        Eqs = np.load(OUT / "ik_qubos.npy")
        n_total, get_q = len(Eqs), (lambda i: Eqs[i])
    else:                                          # compact set shipped with the repo
        z = np.load(OUT / "ik_qubos_sel.npz")
        lut = dict(zip(z["idx"].tolist(), z["E"]))
        n_total, get_q = int(z["n_total"]), (lambda i: lut[int(i)])
    qkd = pickle.load(open(OUT / "qkgp.pkl", "rb"))

    backend = H.get_backend(args.backend)
    is_ibm = args.backend.startswith("ibm")
    print(f"backend: {getattr(backend, 'name', backend)}")

    # ---------------- build all PUBs
    jobs = {}
    legs = [r.problem for r in plan["records"]]
    for p in depths:
        if p == 3:
            ang = [(np.array(r.gammas), np.array(r.betas)) for r in plan["records"]]
        else:
            ang = [optimize_angles([pr], p=p, restarts=4, mode="cvar")[:2] for pr in legs]
        isa, vals = H.qaoa_sweep_inputs(legs, ang, p, backend)
        jobs[f"m1_p{p}"] = dict(isa=isa, vals=vals, problems=legs, angles=ang)
    sel = np.linspace(0, n_total - 1, args.ik_instances).astype(int)
    try:
        ikp = [DiagonalProblem.from_energies(get_q(i)) for i in sel]
    except KeyError:
        sys.exit("this --ik-instances count needs the full results/ik_qubos.npy: "
                 "run scripts/run_all.py first (or use 12, 20, 40, 80 or 100)")
    tr = metrics["module2"]["transfer"]
    for p in depths:
        key = str(p) if str(p) in tr else p
        g, b = np.array(tr[key]["gammas"]), np.array(tr[key]["betas"])
        isa, vals = H.qaoa_sweep_inputs(ikp, [(g, b)] * len(ikp), p, backend)
        jobs[f"m2_p{p}"] = dict(isa=isa, vals=vals, problems=ikp, angles=[(g, b)] * len(ikp))
    qk = C.QuantumKernel(4, bandwidth=qkd["bandwidth"])
    ntr, nte = args.kernel_train, args.kernel_test
    Xa, Ya = qkd["Xtr"][:ntr], qkd["Ytr"][:ntr]
    Xb, Yb = qkd["Xte"][:nte], qkd["Yte"][:nte]
    XX = np.vstack([Xa, Xb])
    pairs = [(i, j) for i in range(ntr) for j in range(i + 1, ntr)] + \
            [(ntr + i, j) for i in range(nte) for j in range(ntr)]
    kisa = H.kernel_sweep_inputs(qk, backend)
    jobs["m3_kernel"] = dict(isa=kisa, vals=H.kernel_values(kisa, qk, XX, XX, pairs))

    # ---- Modules 4-5 (only if scripts/run_learning.py has been run)
    from types import SimpleNamespace
    from qkitchen.qml import VQCPolicy, ZZKernel, z_expectations_from_counts
    lm_f = OUT / "learning_metrics.json"
    lm = json.load(open(lm_f)) if lm_f.exists() else {}
    if "module4" in lm and (OUT / "m4_residual_logs.npz").exists():
        hyp = lm["module4"]["qk_hyper"][0]                       # joint 1 (base) residual GP
        zk = ZZKernel(4, reps=hyp["reps"], bandwidth=np.array(hyp["bandwidth"]))
        fm = zk.qiskit_circuit()
        qk4 = SimpleNamespace(fmap=fm, params=list(fm.parameters), encode=lambda f: zk.encode(f)[0])
        z = np.load(OUT / "m4_residual_logs.npz")
        rs = np.random.default_rng(0)
        X4 = np.vstack([z["Xtr"][rs.choice(len(z["Xtr"]), ntr, replace=False), 0],
                        z["Xte"][rs.choice(len(z["Xte"]), nte, replace=False), 0]])
        k4isa = H.kernel_sweep_inputs(qk4, backend)
        jobs["m4_kernel"] = dict(isa=k4isa, vals=H.kernel_values(k4isa, qk4, X4, X4, pairs), zk=zk, X=X4)
    if "module5" in lm and (OUT / "m5_agents.pkl").exists():
        from scipy.ndimage import uniform_filter1d
        from qkitchen import control as Cc, qrl
        vq = VQCPolicy(6, 3, 2)
        vq.set(pickle.load(open(OUT / "m5_agents.pkl", "rb"))["vqc_0"])
        cl = np.load(OUT / "m5_closed_loop.npz")
        tt = np.load(OUT / "test_traj.npz")
        qd_, qdd_ = Cc.reference_derivatives(tt["Q"])
        f_ = uniform_filter1d(Cc.features(tt["Q"], qd_, qdd_, 0.6 * tt["carry"]), 21, axis=0)
        ks = np.linspace(10, len(f_) - 10, args.policy_obs).astype(int)
        names = [str(n) for n in cl["names"]]
        b0 = names.index("vqc_0")
        e = cl["Qref"] - cl["Q"][b0]
        ed = np.gradient(e, 0.01, axis=0)
        O5 = np.vstack([qrl.observation(f_[k], e[k][None], ed[k][None]) for k in ks])
        pqc = vq.qiskit_circuit(measure=True)
        pisa = H.transpile_for(pqc, backend)
        order = [int(p.name[2:-1]) for p in pisa.parameters]
        jobs["m5_policy"] = dict(isa=pisa, vals=O5[:, order], vqc=vq, O=O5)

    keep = tuple(args.modules.split(","))
    jobs = {k: v for k, v in jobs.items() if k.startswith(keep)}

    total = 0.0
    report = dict(backend=str(getattr(backend, "name", backend)), shots=args.shots, modules={})
    for k, j in jobs.items():
        st = H.circuit_stats(j["isa"])
        est = H.estimate_qpu_seconds(len(j["vals"]), args.shots)
        total += est
        report["modules"][k] = dict(circuit=st, n_instances=len(j["vals"]), est_qpu_s=est)
        print(f"  {k:10s} {len(j['vals']):4d} instances  depth={st['depth']:5d} "
              f"2q-gates={st['two_qubit_gates']:5d}  est. QPU time ~{est:5.0f}s")
    print(f"  total estimated QPU time ~{total / 60:.1f} min (Open Plan: 10 min / 28 days)")
    if args.dry_run:
        json.dump(report, open(OUT / f"hardware_{args.backend.replace(':', '_')}_dryrun.json", "w"),
                  indent=1)
        return

    # ---------------- execute
    def execute(mode=None):
        res = {}
        for k, j in jobs.items():
            t0 = time.time()
            counts, meta = H.run_sweep(j["isa"], j["vals"], backend, args.shots, mode=mode)
            res[k] = (counts, meta)
            print(f"  ran {k} in {time.time() - t0:.0f}s  job={meta.get('job_id')}")
        return res

    if is_ibm:
        from qiskit_ibm_runtime import Batch
        with Batch(backend=backend) as batch:
            results = execute(mode=batch)
    else:
        results = execute()

    # ---------------- analyse
    for k, j in jobs.items():
        counts, meta = results[k]
        report["modules"][k]["meta"] = {kk: v for kk, v in meta.items() if kk != "usage"}
        if k == "m4_kernel":
            kv = H.kernel_from_counts(counts, 4)
            K = j["zk"].gram(j["X"])
            ex = np.array([K[a, b] for a, b in pairs])
            report["modules"][k].update(kernel_mae=float(np.abs(kv - ex).mean()),
                                        kernel_max_err=float(np.abs(kv - ex).max()),
                                        exact_offdiag_mean=float(ex.mean()))
            print(f"  {k}: residual-GP kernel MAE={np.abs(kv - ex).mean():.4f} "
                  f"(mean exact entry {ex.mean():.3f})")
            continue
        if k == "m5_policy":
            from qkitchen import qrl
            Zd = np.array([z_expectations_from_counts(c, 2) for c in counts])
            Zi = j["vqc"].forward(j["O"])
            gd = np.exp(qrl.LOG_CENTER + qrl.LOG_HALF * Zd)
            gi = np.exp(qrl.LOG_CENTER + qrl.LOG_HALF * Zi)
            report["modules"][k].update(z_mae=float(np.abs(Zd - Zi).mean()),
                                        gain_rel_err=float(np.mean(np.abs(gd / gi - 1))),
                                        z_ideal_std=float(Zi.std()))
            print(f"  {k}: policy <Z> MAE={np.abs(Zd - Zi).mean():.4f}, gain error "
                  f"{100 * np.mean(np.abs(gd / gi - 1)):.1f}% (ideal <Z> spread {Zi.std():.3f})")
            continue
        if k.startswith("m3"):
            kv = H.kernel_from_counts(counts, 4)
            K = qk.gram(XX)
            Kh = np.eye(len(XX))
            for (a, b), v in zip(pairs, kv):
                Kh[a, b] = Kh[b, a] = v
            ex = np.array([K[a, b] for a, b in pairs])
            sig = float(np.mean(np.var(Ya, 0))) + 1e-9
            gp_e = C.GP(noise=qkd["noise"] * sig, signal=sig).fit(K[:ntr, :ntr], Ya)
            gp_h = C.GP(noise=qkd["noise"] * sig, signal=sig).fit(Kh[:ntr, :ntr], Ya)
            pe = gp_e.predict(K[ntr:, :ntr]); ph = gp_h.predict(Kh[ntr:, :ntr])
            report["modules"][k].update(
                kernel_mae=float(np.abs(kv - ex).mean()),
                kernel_max_err=float(np.abs(kv - ex).max()),
                gp_pred_diff_rmse=float(np.sqrt(np.mean((pe - ph) ** 2))),
                gp_rmse_exact_kernel=float(np.sqrt(np.mean((pe - Yb) ** 2))),
                gp_rmse_device_kernel=float(np.sqrt(np.mean((ph - Yb) ** 2))))
            print(f"  {k}: kernel MAE={np.abs(kv - ex).mean():.4f}  GP RMSE exact-kernel="
                  f"{report['modules'][k]['gp_rmse_exact_kernel']:.3f} device-kernel="
                  f"{report['modules'][k]['gp_rmse_device_kernel']:.3f}")
        else:
            an = H.analyse_qaoa_counts(j["problems"], counts)
            s = summarise(an)
            s["ideal"] = ideal_summary(j["problems"], j["angles"])
            n = j["problems"][0].n
            s["uniform_p_opt"] = 1 / 2 ** n
            report["modules"][k].update(s)
            print(f"  {k}: device P(opt)={s['p_opt']:.4f} (ideal {s['ideal']['p_opt']:.4f}, "
                  f"uniform {1 / 2 ** n:.4f})  AR={s['approx_ratio']:.3f} "
                  f"(ideal {s['ideal']['approx_ratio']:.3f})  opt in top-6={s['opt_in_topk']:.2f}")
    fn = OUT / f"hardware_{args.backend.replace(':', '_')}.json"
    if fn.exists():                       # merge with modules run earlier on this backend
        old = json.load(open(fn))
        old["modules"].update(report["modules"])
        report["modules"] = old["modules"]
    json.dump(report, open(fn, "w"), indent=1, default=float)
    print("wrote", fn)


if __name__ == "__main__":
    main()
