"""End-to-end experiment: Module 1 (QAOA path planning), Module 2 (QUBO IK at 0.01 s),
Module 3 (quantum-kernel GP gain scheduling), all on exact / ideal simulation.

    python scripts/run_all.py            # full run (~10 min on a laptop)
    python scripts/run_all.py --quick    # 2 dishes, fewer restarts

Outputs go to results/: metrics.json plus .npz/.pkl traces used by make_figures.py.
Noisy-simulator and IBM-hardware runs are in main.py (repository root).
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

from qkitchen import control as C          # noqa: E402
from qkitchen import env as E              # noqa: E402
from qkitchen import ik as IK              # noqa: E402
from qkitchen import pipeline as PL        # noqa: E402
from qkitchen import planner as P          # noqa: E402
from qkitchen.qaoa import (DiagonalProblem, interp_init, optimize_angles,  # noqa: E402
                           qaoa_probs, sample)

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def uniform_baselines(pr: DiagonalProblem, shots, top_k=6, trials=200, seed=0):
    E_ = pr.energies
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(trials):
        s = np.unique(rng.integers(0, len(E_), shots))
        hit += pr.ground_index in s[np.argsort(E_[s])][:top_k]
    return dict(p_opt=1 / len(E_), approx_ratio=float((E_.max() - E_.mean()) / (E_.max() - E_.min())),
                opt_in_topk=hit / trials)


def module1(kit, q0, args, M):
    print("\n=== Module 1: QAOA end-effector path planning ===")
    t0 = time.time()
    segs, recs = PL.plan_task(kit, q0, planner="qaoa", p=3, shots=2048, mode="cvar",
                              restarts=args.restarts, n_dishes=args.dishes)
    legs = []
    for r in recs:
        u = uniform_baselines(r.problem, 2048)
        legs.append(dict(name=r.name, n_qubits=r.problem.n, n_terms=len(r.problem.terms),
                         max_order=max(len(s) for s, _ in r.problem.terms),
                         dp_cost=r.dp_cost, qaoa_cost=r.chosen_cost,
                         found_optimum=r.found_optimum, p_opt=r.p_opt,
                         approx_ratio=r.approx_ratio, near_opt_mass=r.near_opt_mass,
                         uniform_p_opt=u["p_opt"], uniform_ar=u["approx_ratio"],
                         dp_path=r.dp_path.tolist(), qaoa_path=r.chosen_path.tolist(),
                         gammas=list(r.gammas), betas=list(r.betas)))
    # depth sweep on every leg (per-instance optimised angles)
    sweep = {}
    prev = {}
    for p in (1, 2, 3, 4):
        po, ar = [], []
        for r in recs:
            init = interp_init(*prev[r.name]) if r.name in prev else None
            g, b, _ = optimize_angles([r.problem], p=p, restarts=max(3, args.restarts // 2),
                                      mode="cvar", init=init)
            prev[r.name] = (g, b)
            pr = qaoa_probs(r.problem, g, b)
            po.append(pr[r.problem.ground_index]); ar.append(r.problem.approx_ratio(pr))
        sweep[p] = dict(p_opt_mean=float(np.mean(po)), ar_mean=float(np.mean(ar)))
        print(f"  depth p={p}: mean P(opt)={np.mean(po):.4f}  mean AR={np.mean(ar):.3f}")
    M["module1"] = dict(legs=legs, depth_sweep=sweep, seconds=time.time() - t0)
    with open(OUT / "plan.pkl", "wb") as f:
        pickle.dump(dict(q0=q0, segs=segs, records=recs), f)
    return segs, recs


def module2(kit, q0, segs, args, M):
    print("\n=== Module 2: per-step QUBO inverse kinematics (0.01 s) ===")
    t, Pref, carry, prec, sid = P.sample_reference(segs)
    print(f"  reference: {len(t)} steps, {t[-1]:.2f} s")
    res, traces = {}, {}
    # 1) brute-force grid run also harvests QUBO instances for angle training
    s = IK.StepIK(kit, q0, PL.HALF, method="grid", keep_qubos=True)
    tt = time.time()
    Qg, infg = IK.run_ik(kit, s, q0, t, Pref, carry, prec)
    res["grid"] = IK.evaluate_motion(kit, Qg, t, Pref, carry)[0]
    res["grid"]["seconds"] = time.time() - tt
    traces["grid"] = Qg
    Eqs = np.array([i.qubo[0] for i in infg if i.qubo])
    np.save(OUT / "ik_qubos.npy", Eqs)
    idx = np.unique(np.concatenate([np.linspace(0, len(Eqs) - 1, k).astype(int)
                                    for k in (12, 20, 40, 80, 100)]))
    np.savez(OUT / "ik_qubos_sel.npz", idx=idx, E=Eqs[idx], n_total=len(Eqs))
    # 2) parameter transfer: train on 120 instances, test on 400 others
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Eqs))
    tr = [DiagonalProblem.from_energies(Eqs[i]) for i in idx[:120]]
    te = [DiagonalProblem.from_energies(Eqs[i]) for i in idx[120:520]]
    u = [uniform_baselines(pr, 256, trials=50) for pr in te[:100]]
    transfer = {"uniform": dict(p_opt=float(np.mean([x["p_opt"] for x in u])),
                                approx_ratio=float(np.mean([x["approx_ratio"] for x in u])),
                                opt_in_topk=float(np.mean([x["opt_in_topk"] for x in u])))}
    angles = {}
    for p in (1, 2, 3):
        init = interp_init(*angles[p - 1]) if p > 1 else None
        g, b, _ = optimize_angles(tr, p=p, restarts=args.restarts // 2 + 2, mode="mean",
                                  maxiter=300, init=init)
        po, ar, ins = [], [], []
        for pr in te:
            pb = qaoa_probs(pr, g, b)
            po.append(pb[pr.ground_index]); ar.append(pr.approx_ratio(pb))
            smp = np.unique(sample(pb, 256, rng))
            ins.append(pr.ground_index in smp[np.argsort(pr.energies[smp])][:6])
        transfer[p] = dict(p_opt=float(np.mean(po)), approx_ratio=float(np.mean(ar)),
                           opt_in_topk=float(np.mean(ins)), gammas=list(g), betas=list(b))
        angles[p] = (g, b)
        print(f"  transferred angles p={p}: P(opt)={np.mean(po):.4f} AR={np.mean(ar):.3f} "
              f"opt in top-6 of 256 shots={np.mean(ins):.2f}")
    print(f"  uniform sampling: P(opt)={transfer['uniform']['p_opt']:.4f} "
          f"AR={transfer['uniform']['approx_ratio']:.3f} "
          f"opt in top-6={transfer['uniform']['opt_in_topk']:.2f}")
    # 3) closed IK loop with each solver
    qstats = {}
    for meth in ("dls", "lsq", "qaoa"):
        s = IK.StepIK(kit, q0, PL.HALF, method=meth, angles=angles[3], seed=1)
        tt = time.time()
        Q, inf = IK.run_ik(kit, s, q0, t, Pref, carry, prec)
        res[meth] = IK.evaluate_motion(kit, Q, t, Pref, carry)[0]
        res[meth]["seconds"] = time.time() - tt
        traces[meth] = Q
        if meth == "qaoa":
            po = np.array([i.p_opt for i in inf[1:]])
            qstats = dict(p_opt_mean=float(np.nanmean(po)),
                          opt_sampled_frac=float(np.mean([i.opt_sampled for i in inf[1:]])),
                          chose_grid_opt_frac=float(np.mean([i.gap <= 1e-9 for i in inf[1:]])),
                          qubo_qubits=10, qubos_solved=2 * (len(t) - 1))
    for k, v in res.items():
        print(f"  {k:5s} rms={v['pos_err_rms_mm']:.2f}mm max={v['pos_err_max_mm']:.1f}mm "
              f"tilt={v['tilt_max_deg']:.1f}deg minclear={v['min_clearance_mm']:.1f}mm "
              f"collide_steps={v['steps_in_collision']} travel={v['weighted_travel']:.1f}")
    M["module2"] = dict(solvers=res, transfer=transfer, qaoa_in_loop=qstats, steps=len(t))
    np.savez(OUT / "ik_traces.npz", t=t, Pref=Pref, carry=carry, prec=prec, sid=sid, **traces)
    return t, Pref, carry, prec, sid, traces


def module3(kit, q0, segs, t, carry, sid, Qik, args, M):
    print("\n=== Module 3: quantum-kernel GP gain scheduling for the joint PID ===")
    Qs = C.smooth_reference(Qik)
    qd, qdd = C.reference_derivatives(Qs)
    gains = C.nominal_gains(q0)
    dish_mass = np.array([0.45, 0.80, 0.60, 0.95])
    dish_of = np.array([segs[s].dish for s in sid])
    payload = np.where(carry, dish_mass[dish_of], 0.0)
    # training windows: first half of the task; test windows: second half (held out)
    t_split = t[-1] / 2
    rng = np.random.default_rng(3)
    n_win = 2 * args.windows
    WIN = 150   # 1.5 s: short windows were myopic (they hid slow error build-up)
    k_tr = rng.integers(0, np.searchsorted(t, t_split) - WIN - 1, args.windows)
    k_te = rng.integers(np.searchsorted(t, t_split), len(t) - WIN - 1, args.windows)
    wins = [(int(k), int(k) + WIN) for k in np.r_[k_tr, k_te]]
    tt = time.time()
    # labels use randomised true payloads so the model can't memorise one bowl
    pay_lab = payload.copy()
    pay_lab[carry] = rng.uniform(0.3, 1.0, carry.sum())
    X, Y, J, (KP, KD) = C.label_windows(Qs, carry, gains, wins, pay_lab)
    print(f"  labelled {n_win} windows x {len(KP)} gain pairs in {time.time() - tt:.0f}s")
    Jtr = J[:args.windows]
    safe = C.safe_gain_set(Jtr)
    print(f"  safe gain set: {safe.sum()}/{len(safe)} pairs "
          f"(kp {KP[safe].min()}-{KP[safe].max()}, kd {KD[safe].min()}-{KD[safe].max()})")
    Y = C.soft_labels(J, KP, KD, safe)
    Xtr, Ytr, Xte, Yte = X[:args.windows], Y[:args.windows], X[args.windows:], Y[args.windows:]
    relm = np.where(safe, (Jtr / Jtr.min(1, keepdims=True)).mean(0), np.inf)
    best_fixed = int(np.argmin(relm))
    kp_fix, kd_fix = KP[best_fixed], KD[best_fixed]
    lo = np.log([KP[safe].min(), KD[safe].min()])
    hi = np.log([KP[safe].max(), KD[safe].max()])

    qk = C.fit_qkgp(Xtr, Ytr)
    rbf = C.fit_rbfgp(Xtr, Ytr)
    Kte = qk["kernel"].gram(Xte, SA=None, SB=qk["S_train"])
    y_q = qk["gp"].predict(Kte)
    y_r = rbf.predict(Xte)
    y_c = np.repeat(Ytr.mean(0, keepdims=True), len(Xte), 0)
    y_f = np.repeat(np.log([[kp_fix, kd_fix]]), len(Xte), 0)

    def regret(yhat, Jw):
        """Cost of the nearest safe grid gains to the prediction, relative to the window best."""
        out = []
        for yh, j in zip(yhat, Jw):
            d = (np.log(KP) - yh[0]) ** 2 + (np.log(KD) - yh[1]) ** 2
            k = np.argmin(np.where(safe, d, np.inf))
            out.append(j[k] / j[safe].min() - 1)
        return float(np.median(out))
    Jte = J[args.windows:]
    reg = {name: dict(rmse_log=float(np.sqrt(np.mean((yh - Yte) ** 2))), regret=regret(yh, Jte))
           for name, yh in [("qk_gp", y_q), ("rbf_gp", y_r), ("train_mean", y_c),
                            ("best_fixed", y_f)]}
    for k, v in reg.items():
        print(f"  {k:10s} held-out RMSE(log gains)={v['rmse_log']:.3f}  "
              f"median regret={100 * v['regret']:.1f}%")
    print(f"  QK-GP bandwidth={qk['bandwidth']} noise={qk['noise']} "
          f"mean off-diagonal kernel={qk['offdiag_mean']:.3f}")

    # closed loop on the full task with per-step (10 ms) gain updates
    from scipy.ndimage import uniform_filter1d
    # 0.2 s moving average: labels were made on window-mean features, so the
    # scheduler sees the same smoothed signal (also stops step-to-step gain jitter)
    feats = uniform_filter1d(C.features(Qs, qd, qdd, 0.6 * carry), 21, axis=0)
    S_tr = qk["S_train"]
    t_q = time.time()
    S_all = qk["kernel"].statevectors(feats)
    infer_ms = 1e3 * (time.time() - t_q) / len(feats)
    kq = C.schedule_from_model(lambda F: qk["gp"].predict(np.abs(S_all.conj() @ S_tr.T) ** 2),
                               feats, lo=lo, hi=hi)
    kr = C.schedule_from_model(lambda F: rbf.predict(F), feats, lo=lo, hi=hi)
    N = len(Qs)
    KPm = np.stack([np.ones(N), np.full(N, kp_fix), kr[0], kq[0]])
    KDm = np.stack([np.ones(N), np.full(N, kd_fix), kr[1], kq[1]])
    tt = time.time()
    sim = C.simulate(Qs, qd, payload, KPm, KDm, gains, record=True, seed=11)
    print(f"  closed-loop sim of {t[-1]:.1f}s x 4 controllers in {time.time() - tt:.0f}s")
    names = ["nominal", "best_fixed", "rbf_gp", "qk_gp"]
    held = t >= t_split
    cl = {}
    for b, nm in enumerate(names):
        Qa = sim["Q"][b]
        err = Qa - Qs
        m = IK.evaluate_motion(kit, Qa, t, P.sample_reference(segs)[1], carry)[0]
        dmin = m["min_clearance_mm"]
        cl[nm] = dict(rms_joint_err_mrad=1e3 * float(np.sqrt(np.mean(err ** 2))),
                      rms_joint_err_heldout_mrad=1e3 * float(np.sqrt(np.mean(err[held] ** 2))),
                      max_joint_err_mrad=1e3 * float(np.abs(err).max()),
                      tcp_err_rms_mm=m["pos_err_rms_mm"], tcp_err_max_mm=m["pos_err_max_mm"],
                      min_clearance_mm=dmin, cost=float(sim["cost"][b]),
                      effort=float(sim["eff"][b]), chatter=float(sim["chat"][b]),
                      saturations=int(sim["sat"][b]),
                      mean_kp=float(KPm[b].mean()), mean_kd=float(KDm[b].mean()))
        print(f"  {nm:10s} cost={cl[nm]['cost']:.2f} rms err={cl[nm]['rms_joint_err_mrad']:.3f}mrad "
              f"(held-out {cl[nm]['rms_joint_err_heldout_mrad']:.3f}) tcp max={m['pos_err_max_mm']:.1f}mm "
              f"chatter={cl[nm]['chatter']:.2e} minclear={dmin:.1f}mm")
    M["module3"] = dict(regression=reg, closed_loop=cl, best_fixed=[float(kp_fix), float(kd_fix)],
                        safe_pairs=int(safe.sum()), grid_pairs=int(len(safe)),
                        qk_bandwidth=qk["bandwidth"], qk_noise=qk["noise"],
                        qk_offdiag_kernel_mean=qk["offdiag_mean"],
                        rbf_kernel=str(rbf.kernel_), n_train=len(Xtr), n_test=len(Xte),
                        qk_inference_ms_per_step=infer_ms)
    np.savez(OUT / "pid.npz", t=t, Qs=Qs, Q_act=sim["Q"], KP=KPm, KD=KDm, X=X, Y=Y, J=J,
             KPg=KP, KDg=KD, payload=payload, names=np.array(names), n_train=args.windows,
             TAU=sim["TAU"])
    with open(OUT / "qkgp.pkl", "wb") as f:
        pickle.dump(dict(Xtr=Xtr, Ytr=Ytr, Xte=Xte, Yte=Yte, bandwidth=qk["bandwidth"],
                         noise=qk["noise"]), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-m3", action="store_true", help="rerun Module 3 on saved results")
    args = ap.parse_args()
    args.dishes = 2 if args.quick else 4
    args.restarts = 4 if args.quick else 8
    args.windows = 24 if args.quick else 64
    M = {}
    kit = E.default_kitchen()
    if args.only_m3:
        M = json.load(open(OUT / "metrics.json"))
        plan = pickle.load(open(OUT / "plan.pkl", "rb"))
        tr = np.load(OUT / "ik_traces.npz")
        module3(kit, plan["q0"], plan["segs"], tr["t"], tr["carry"], tr["sid"], tr["qaoa"], args, M)
        with open(OUT / "metrics.json", "w") as f:
            json.dump(M, f, indent=1, default=float)
        return
    q0 = PL.find_start_config(kit, np.array([*kit.dishes[0][:2], kit.sink_approach_z]))
    M["q_start"] = q0.tolist()
    segs, _ = module1(kit, q0, args, M)
    t, Pref, carry, prec, sid, traces = module2(kit, q0, segs, args, M)
    module3(kit, q0, segs, t, carry, sid, traces["qaoa"], args, M)
    with open(OUT / "metrics.json", "w") as f:
        json.dump(M, f, indent=1, default=float)
    print("\nwrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
