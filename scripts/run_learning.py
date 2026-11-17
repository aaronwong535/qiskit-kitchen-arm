"""Modules 4 and 5: quantum learning methods.

  Module 4  Slotine-Li adaptive control (baseline) + residual-dynamics GP with a quantum
            fidelity kernel vs an RBF kernel.
  Module 5  Policy-gradient RL (REINFORCE with baseline) with a variational
            quantum circuit actor vs a classical MLP actor of matched size.

Training uses the task trajectory from run_all.py; testing uses a held-out trajectory
(dishes collected in a different order, different bowl masses, different noise seed).

    python scripts/run_learning.py               # both modules
    python scripts/run_learning.py --only m4     # or --only m5
"""
from __future__ import annotations

import argparse
import copy
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

from qkitchen import control as C                 # noqa: E402
from qkitchen import env as E                     # noqa: E402
from qkitchen import ik as IK                     # noqa: E402
from qkitchen import learning_control as LC       # noqa: E402
from qkitchen import pipeline as PL               # noqa: E402
from qkitchen import planner as P                 # noqa: E402
from qkitchen import robot as R                   # noqa: E402

OUT = ROOT / "results"
MASS_TRAIN = np.array([0.45, 0.80, 0.60, 0.95])
MASS_TEST = np.array([0.90, 0.35, 1.10, 0.55])
TEST_ORDER = [3, 1, 2, 0]           # dish i of the test task = dish TEST_ORDER[i] of the kitchen


# ---------------------------------------------------------------- trajectories
def train_trajectory():
    d = np.load(OUT / "pid.npz", allow_pickle=True)
    ik = np.load(OUT / "ik_traces.npz")
    plan = pickle.load(open(OUT / "plan.pkl", "rb"))
    segs = plan["segs"]
    dish_of = np.array([segs[s].dish for s in ik["sid"]])
    return dict(t=ik["t"], Q=d["Qs"], carry=ik["carry"], dish_of=dish_of, q0=plan["q0"],
                Pref=ik["Pref"])


def test_trajectory():
    f = OUT / "test_traj.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    print("  building held-out test trajectory (different dish order, DP planner + LSQ IK)")
    kit = E.default_kitchen()
    kit.dishes = kit.dishes[TEST_ORDER]
    kit.slots = kit.slots[[2, 0, 3, 1]]
    q0 = PL.find_start_config(kit, np.array([*kit.dishes[0][:2], kit.sink_approach_z]))
    segs, _ = PL.plan_task(kit, q0, planner="dp", verbose=False)
    t, Pref, carry, prec, sid = P.sample_reference(segs)
    s = IK.StepIK(kit, q0, PL.HALF, method="lsq")
    Q, _ = IK.run_ik(kit, s, q0, t, Pref, carry, prec)
    m = IK.evaluate_motion(kit, Q, t, Pref, carry)[0]
    print(f"    test IK: rms {m['pos_err_rms_mm']:.2f} mm, min clearance {m['min_clearance_mm']:.1f} mm, "
          f"collisions {m['steps_in_collision']}")
    dish_of = np.array([segs[x].dish for x in sid])
    out = dict(t=t, Q=C.smooth_reference(Q), carry=carry, dish_of=dish_of, q0=q0, Pref=Pref)
    np.savez(f, **out)
    return out


def tcp_and_clearance(kit, Q, tr):
    kit = kit or E.default_kitchen()
    F = R.fk_frames(Q)
    p = F[:, 6, :3, 3] + R.TOOL_LEN * F[:, 6, :3, 2]
    dmin = np.zeros(len(Q))
    for c in (False, True):
        m = tr["carry"] == c
        dmin[m] = E.min_clearance(kit, Q[m], tr["t"][m], c, F=F[m])
    return np.linalg.norm(p - tr["Pref"], axis=1), dmin


# ---------------------------------------------------------------- Module 4
def module4(M, args):
    print("\n=== Module 4: Slotine-Li adaptive control + quantum-kernel GP residual dynamics ===")
    tr, te = train_trajectory(), test_trajectory()
    pay_tr = np.where(tr["carry"], MASS_TRAIN[tr["dish_of"]], 0.0)
    pay_te = np.where(te["carry"], MASS_TEST[te["dish_of"]], 0.0)
    unm_tr = LC.Unmodelled(cable_q0=tr["q0"].copy())
    unm_te = LC.Unmodelled(cable_q0=tr["q0"].copy())     # same physical cable routing
    t0 = time.time()
    log_tr = LC.simulate_sl(tr["Q"], tr["carry"], pay_tr, [LC.Controller("C1")], unm_tr,
                            seed=1, log_residuals=True)
    log_te = LC.simulate_sl(te["Q"], te["carry"], pay_te, [LC.Controller("C1")], unm_te,
                            seed=2, log_residuals=True)
    print(f"  logged {len(log_tr['X'])} training and {len(log_te['X'])} test residual samples "
          f"({time.time() - t0:.0f}s)")
    np.savez(OUT / "m4_residual_logs.npz", Xtr=log_tr["X"], Ytr=log_tr["Y"], Xte=log_te["X"], Yte=log_te["Y"])

    # regression quality vs support-set size (held-out trajectory)
    def nmse(pred, Y):
        """Pooled over joints (sum MSE / sum variance): joint 6 barely moves, so a plain
        per-joint average would be dominated by its near-zero variance."""
        return float(np.sum(np.mean((pred - Y) ** 2, 0)) / np.sum(np.var(Y, 0)))

    def nmse_joints(pred, Y):
        return (np.mean((pred - Y) ** 2, 0) / np.var(Y, 0))[:3].round(3).tolist()
    eff = {}
    models = {}
    for n_sup in args.support_sizes:
        for kind in ("qk", "rbf"):
            t1 = time.time()
            gp = LC.JointGP(kind, n_support=n_sup, seed=0).fit(log_tr["X"], log_tr["Y"])
            pred = gp.predict(log_te["X"])
            eff[f"{kind}_{n_sup}"] = dict(nmse=nmse(pred, log_te["Y"]),
                                          nmse_joints_1_3=nmse_joints(pred, log_te["Y"]),
                                          fit_s=time.time() - t1)
            models[(kind, n_sup)] = gp
            print(f"  {kind:3s} GP, {n_sup:4d} support points: held-out nMSE "
                  f"{eff[f'{kind}_{n_sup}']['nmse']:.3f}")
    n_main = args.n_support
    qk, rbf = models[("qk", n_main)], models[("rbf", n_main)]

    ctrls = [LC.Controller("C0 nominal model", adapt_payload=False),
             LC.Controller("C1 Slotine-Li (payload)"),
             LC.Controller("C1f Slotine-Li (payload+friction)", adapt_friction=True),
             LC.Controller("C2 C1 + RBF-GP residual", residual=rbf.predict),
             LC.Controller("C3 C1 + QK-GP residual", residual=qk.predict),
             LC.Controller("C4 oracle (true model)", oracle=True)]
    kit = E.default_kitchen()
    t1 = time.time()
    res = LC.simulate_sl(te["Q"], te["carry"], pay_te, ctrls, unm_te, seed=3)
    print(f"  closed loop on held-out task, {len(ctrls)} controllers, {time.time() - t1:.0f}s")
    cl = {}
    carry_idx = np.nonzero(te["carry"])[0]
    for b, c in enumerate(ctrls):
        Qa = res["Q"][b]
        err = Qa - te["Q"]
        tcp, dmin = tcp_and_clearance(kit_for_test(), Qa, te)
        mh = res["m_hat"][b][carry_idx]
        cl[c.name] = dict(rms_joint_err_mrad=1e3 * float(np.sqrt(np.mean(err ** 2))),
                          max_joint_err_mrad=1e3 * float(np.abs(err).max()),
                          tcp_rms_mm=1e3 * float(np.sqrt(np.mean(tcp ** 2))),
                          tcp_max_mm=1e3 * float(tcp.max()),
                          min_clearance_mm=1e3 * float(dmin.min()),
                          chatter=float(res["chatter"][b]), saturations=int(res["sat"][b]),
                          payload_est_err_kg=float(np.mean(np.abs(mh - pay_te[carry_idx]))))
        v = cl[c.name]
        print(f"  {c.name:36s} joint rms {v['rms_joint_err_mrad']:6.3f} mrad  TCP max "
              f"{v['tcp_max_mm']:6.2f} mm  clearance {v['min_clearance_mm']:5.1f} mm  "
              f"|m^-m| {v['payload_est_err_kg']:.2f} kg")
    M["module4"] = dict(closed_loop=cl, data_efficiency=eff, n_support=n_main,
                        qk_hyper=qk.hyper(), rbf_hyper=rbf.hyper(),
                        mass_train=MASS_TRAIN.tolist(), mass_test=MASS_TEST.tolist(),
                        unmodelled=dict(cable_k=unm_tr.cable_k.tolist(), ripple_a=unm_tr.ripple_a.tolist(),
                                        armature_nominal_scale=unm_tr.armature_nominal_scale))
    np.savez(OUT / "m4_closed_loop.npz", t=te["t"], Qref=te["Q"], Q=res["Q"], m_hat=res["m_hat"],
             payload=pay_te, names=np.array([c.name for c in ctrls]))
    with open(OUT / "m4_models.pkl", "wb") as f:
        pickle.dump(dict(qk_hyper=qk.hyper(), support_X=[m["S"].shape for _, m in qk.models]), f)


# ---------------------------------------------------------------- Module 5
def module5(M, args):
    from scipy.ndimage import uniform_filter1d
    from qkitchen import qrl
    from qkitchen.qml import MLPPolicy, VQCPolicy
    print("\n=== Module 5: policy-gradient RL with a variational quantum circuit actor ===")
    tr, te = train_trajectory(), test_trajectory()
    gains = C.nominal_gains(tr["q0"])

    def feats(T):
        qd, qdd = C.reference_derivatives(T["Q"])
        return uniform_filter1d(C.features(T["Q"], qd, qdd, 0.6 * T["carry"]), 21, axis=0)
    f_tr, f_te = feats(tr), feats(te)
    pay_tr = np.where(tr["carry"], MASS_TRAIN[tr["dish_of"]], 0.0)
    qd_tr, _ = C.reference_derivatives(tr["Q"])
    N = len(tr["Q"])
    t0 = time.time()
    nominal_state = C.simulate(tr["Q"], qd_tr, pay_tr, np.ones((1, N)), np.ones((1, N)), gains,
                               seed=5, record=True)
    nominal_state = {k: nominal_state[k][0] for k in ("Q", "QD", "INT")}
    print(f"  nominal rollout for episode start states ({time.time() - t0:.0f}s)")
    cfg = qrl.RLConfig(iters=args.rl_iters, batch=args.rl_batch, lr=0.05, lr_vqc=0.1)
    curves, agents = {}, {}
    for seed in range(args.rl_seeds):
        for kind in ("vqc", "mlp"):
            model = VQCPolicy(6, 3, 2, seed=seed) if kind == "vqc" else MLPPolicy(6, 6, 2, seed=seed)
            t1 = time.time()
            curves[f"{kind}_{seed}"] = qrl.train(model, tr["Q"], tr["carry"], f_tr, nominal_state,
                                                 gains, cfg, seed=seed, name=f"{kind} seed {seed}")
            agents[f"{kind}_{seed}"] = model
            print(f"    trained {kind} seed {seed} ({model.n_params} params) in {time.time() - t1:.0f}s")
    # deterministic deployment on the held-out task
    best_fixed = tuple(json.load(open(OUT / "metrics.json"))["module3"]["best_fixed"])
    names = ["nominal", "best fixed (Module 3)"] + list(agents)
    actors = [(1.0, 1.0), best_fixed] + [agents[k] for k in agents]
    qd_te, _ = C.reference_derivatives(te["Q"])
    pay_te = np.where(te["carry"], MASS_TEST[te["dish_of"]], 0.0)
    Nt = len(te["Q"])
    runner = qrl.PolicyRunner(actors, f_te, sigma=0.0)
    t1 = time.time()
    sim = C.simulate(te["Q"], qd_te, pay_te, np.ones((len(actors), Nt)), np.ones((len(actors), Nt)),
                     gains, seed=7, record=True, gain_fn=runner)
    print(f"  held-out closed loop, {len(actors)} controllers, {time.time() - t1:.0f}s")
    dep = {}
    for b, nm in enumerate(names):
        err = sim["Q"][b] - te["Q"]
        tcp, dmin = tcp_and_clearance(None, sim["Q"][b], te)
        dep[nm] = dict(cost=float(sim["cost"][b]), err=float(sim["err"][b]),
                       chatter_term=float(C.SimConfig().w_chat * sim["chat"][b]),
                       rms_joint_err_mrad=1e3 * float(np.sqrt(np.mean(err ** 2))),
                       tcp_max_mm=1e3 * float(tcp.max()), min_clearance_mm=1e3 * float(dmin.min()),
                       mean_kp=float(sim["kp"][b].mean()), mean_kd=float(sim["kd"][b].mean()))
        print(f"  {nm:24s} cost {dep[nm]['cost']:6.2f}  joint rms {dep[nm]['rms_joint_err_mrad']:.3f} mrad  "
              f"TCP max {dep[nm]['tcp_max_mm']:5.2f} mm  kp {dep[nm]['mean_kp']:.2f} kd {dep[nm]['mean_kd']:.2f}")

    def agg(kind):
        v = [dep[f"{kind}_{s}"]["cost"] for s in range(args.rl_seeds)]
        return dict(mean=float(np.mean(v)), std=float(np.std(v)), all=v)
    M["module5"] = dict(curves=curves, deployment=dep, vqc=agg("vqc"), mlp=agg("mlp"),
                        params=dict(vqc=agents["vqc_0"].n_params, mlp=agents["mlp_0"].n_params),
                        config=dict(iters=cfg.iters, batch=cfg.batch, window_s=cfg.window * 0.01,
                                    decision_s=qrl.DECISION_EVERY * 0.01, lr_mlp=cfg.lr, lr_vqc=cfg.lr_vqc,
                                    gain_smoothing_s=qrl.GAIN_TAU,
                                    sigma=[cfg.sigma0, cfg.sigma1]))
    print(f"  VQC actor: {M['module5']['vqc']['mean']:.2f} +/- {M['module5']['vqc']['std']:.2f}   "
          f"MLP actor: {M['module5']['mlp']['mean']:.2f} +/- {M['module5']['mlp']['std']:.2f}")
    with open(OUT / "m5_agents.pkl", "wb") as f:
        pickle.dump({k: v.get() for k, v in agents.items()}, f)
    with open(OUT / "m5_agent_offsets.pkl", "wb") as f:   # output centring, fixed at init
        pickle.dump({k: np.asarray(getattr(v, "offset", 0.0)) for k, v in agents.items()}, f)
    np.savez(OUT / "m5_closed_loop.npz", t=te["t"], Qref=te["Q"], Q=sim["Q"], kp=sim["kp"],
             kd=sim["kd"], step_cost=sim["step_cost"], tau=sim["TAU"], cost=sim["cost"],
             names=np.array(names))


_KIT_TEST = None


def kit_for_test():
    global _KIT_TEST
    if _KIT_TEST is None:
        _KIT_TEST = E.default_kitchen()
        _KIT_TEST.dishes = _KIT_TEST.dishes[TEST_ORDER]
        _KIT_TEST.slots = _KIT_TEST.slots[[2, 0, 3, 1]]
    return _KIT_TEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["m4", "m5"], default=None)
    ap.add_argument("--n-support", type=int, default=250)
    ap.add_argument("--support-sizes", type=int, nargs="+", default=[50, 100, 250, 500])
    ap.add_argument("--rl-iters", type=int, default=100)
    ap.add_argument("--rl-batch", type=int, default=24)
    ap.add_argument("--rl-seeds", type=int, default=3)
    args = ap.parse_args()
    f = OUT / "learning_metrics.json"
    M = json.load(open(f)) if f.exists() else {}
    if args.only in (None, "m4"):
        module4(M, args)
        json.dump(M, open(f, "w"), indent=1, default=float)
    if args.only in (None, "m5"):
        module5(M, args)
        json.dump(M, open(f, "w"), indent=1, default=float)
    print("wrote", f)


if __name__ == "__main__":
    main()
