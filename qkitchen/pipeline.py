"""Task assembly: start configuration, dish-by-dish segment timeline, Module-1 planning."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import env as E
from . import planner as P
from . import robot as R
from .qaoa import optimize_angles, qaoa_probs, sample

HALF = np.array([np.pi, 1.2, 1.2, 1.5, 1.0, np.pi])   # comfort half-widths [rad]

T_VERT, T_DWELL, T_TRANSFER, T_RETURN = 0.9, 0.3, 2.6, 2.4


def find_start_config(kit: E.Kitchen, p, n_seeds=400, seed=0):
    """Clearance-maximising, elbow-up IK solution for the first approach pose."""
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(-np.pi, np.pi, (n_seeds, 6))
    seeds[:, 2] = rng.uniform(-2.8, 2.8, n_seeds)
    best = None
    for s in seeds:
        q, ep, ea = R.ik_numeric(p, P.DOWN, s, iters=100)
        if ep > 1e-4 or ea > 1e-3:
            continue
        q = R.wrap(q)
        if abs(q[2]) > 0.9 * np.pi:
            continue
        d = E.min_clearance(kit, q[None], 0.0, False)[0]
        elbow_z = R.fk_frames(q[None])[0, 2, 2, 3]
        score = min(d, 0.1) + 0.05 * elbow_z
        if best is None or score > best[0]:
            best = (score, q)
    return best[1]


@dataclass
class LegRecord:
    name: str
    leg: P.LegProblem
    problem: object
    dp_path: np.ndarray
    dp_cost: float
    gammas: np.ndarray = None
    betas: np.ndarray = None
    p_opt: float = np.nan
    approx_ratio: float = np.nan
    near_opt_mass: float = np.nan
    chosen_path: np.ndarray = None
    chosen_cost: float = np.nan
    found_optimum: bool = False
    extra: dict = field(default_factory=dict)


def plan_task(kit: E.Kitchen, q_start, planner="qaoa", p=3, shots=2048, mode="cvar",
              restarts=8, seed=0, n_dishes=None, verbose=True):
    """Build the full segment timeline; transfer/return legs solved by `planner`.

    planner = 'qaoa' (sampled QAOA, best sample used) or 'dp' (exact DP baseline).
    """
    rng = np.random.default_rng(seed)
    nd = len(kit.dishes) if n_dishes is None else n_dishes
    segs, records = [], []
    t = 0.0
    q_cur = np.array(q_start, float)

    def approach(i):
        return np.array([kit.dishes[i, 0], kit.dishes[i, 1], kit.sink_approach_z])

    def rack_app(i):
        return np.array([kit.slots[i, 0], kit.slots[i, 1], kit.rack_approach_z])

    def vertical(kind, a, b, carrying, i):
        nonlocal t, q_cur
        segs.append(P.Segment(kind, t, T_VERT, np.array([a, b]), carrying, True, i))
        t += T_VERT
        q_cur = P.ik_track(np.linspace(a, b, 6)[1:], q_cur)[0][-1]

    def dwell(kind, a, carrying, i):
        nonlocal t
        segs.append(P.Segment(kind, t, T_DWELL, np.array([a, a]), carrying, True, i))
        t += T_DWELL

    def leg(name, a, b, T, carrying, i):
        nonlocal t, q_cur
        lp = P.build_leg(kit, a, b, t, T, carrying, q_cur, q_start, HALF)
        prob = lp.to_problem(name=name)
        ks_dp, c_dp = lp.dp_solve()
        rec = LegRecord(name, lp, prob, ks_dp, c_dp)
        if planner == "dp":
            ks = ks_dp
        else:
            g, b_, _ = optimize_angles([prob], p=p, restarts=restarts, mode=mode, seed=seed)
            pr = qaoa_probs(prob, g, b_)
            s = np.unique(sample(pr, shots, rng))
            best = s[np.argmin(prob.energies[s])]
            ks = lp.decode(int(best))
            Emin = prob.energies.min()
            rec.gammas, rec.betas = g, b_
            rec.p_opt = float(pr[prob.ground_index])
            rec.approx_ratio = prob.approx_ratio(pr)
            rec.near_opt_mass = float(pr[prob.energies <= Emin + 0.5].sum())
        rec.chosen_path = ks
        rec.chosen_cost = lp.path_cost(ks)
        rec.found_optimum = bool(np.isclose(rec.chosen_cost, c_dp))
        records.append(rec)
        if verbose:
            print(f"  {name:22s} DP={c_dp:7.3f} chosen={rec.chosen_cost:7.3f} "
                  f"P(opt)={rec.p_opt:.4f} AR={rec.approx_ratio:.3f} path={ks}")
        segs.append(P.Segment(name.split('_')[0], t, T, lp.via_points(ks), carrying, False, i))
        t += T
        sel = lp.q_cand[np.arange(lp.L), ks]
        q_cur = P.ik_track([b], sel[-1])[0][-1]

    for i in range(nd):
        a, g_pt = approach(i), kit.dishes[i]
        ra, pl = rack_app(i), kit.slots[i]
        vertical("descend", a, g_pt, False, i)
        dwell("grasp", g_pt, False, i)
        vertical("lift", g_pt, a, True, i)
        leg(f"transfer_d{i}", a, ra, T_TRANSFER, True, i)
        vertical("descend", ra, pl, True, i)
        dwell("release", pl, True, i)
        vertical("ascend", pl, ra, False, i)
        if i < nd - 1:
            leg(f"return_d{i}", ra, approach(i + 1), T_RETURN, False, i)
    return segs, records
