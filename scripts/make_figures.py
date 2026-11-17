"""Figures + animation from results/ (run scripts/run_all.py first)."""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from qkitchen import env as E                       # noqa: E402
from qkitchen import robot as R                     # noqa: E402

OUT = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# reference palette (categorical slots 1-4, fixed order) + text/grid tokens
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TXT, TXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
    "text.color": TXT, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2,
    "legend.frameon": False,
})


def box_edges(b):
    lo, hi = b.lo, b.hi
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]],
                  [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [c[list(p)] for p in e]


# translucent fills per scene part: (match prefix, label, face colour, alpha, edge colour)
SCENE_STYLE = [
    (("counter",), "counter top", "#c8b89a", 0.38, "#8f7f63"),
    (("sink",), "sink basin", "#4f9fe0", 0.40, "#1f5f99"),
    (("faucet",), "faucet", "#5d5d5d", 0.75, "#2b2b2b"),
    (("dw_door",), "dishwasher door", "#2fae7a", 0.28, "#1b7a52"),
    (("rack",), "dishwasher rack", "#1baf7a", 0.45, "#0f6e4b"),
    (("upper_cabinet",), "upper cabinet", "#b08968", 0.08, "#b9a58f"),
    (("lower_front",), "lower cabinet front", "#a7a59f", 0.07, "#b9b7b0"),
]


def _style(name):
    for pref, lab, fc, a, ec in SCENE_STYLE:
        if name.startswith(pref):
            return lab, fc, a, ec
    return None


def box_faces(b):
    lo, hi = b.lo, b.hi
    x0, y0, z0 = lo; x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [v[list(f)] for f in idx]


def draw_scene(ax, kit, skip=("back_wall",), legend=True):
    """Translucent, colour-coded solids: counter, sink, faucet, dishwasher, cabinets."""
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    seen = {}
    for b in kit.boxes:
        st = _style(b.name)
        if b.name in skip or st is None:
            continue
        lab, fc, a, ec = st
        pc = Poly3DCollection(box_faces(b), facecolor=fc, alpha=a, edgecolor=ec, linewidths=0.4)
        pc.set_zorder(1)
        ax.add_collection3d(pc)
        seen[lab] = (fc, a, ec)
    # robot base: dark cylinder bolted to the counter at the origin
    th = np.linspace(0, 2 * np.pi, 24)
    zz = np.array([0.0, 0.09])
    TH, ZZ = np.meshgrid(th, zz)
    ax.plot_surface(0.075 * np.cos(TH), 0.075 * np.sin(TH), ZZ, color="#3b3a37", alpha=0.9,
                    linewidth=0, zorder=19)
    if legend:
        handles = [Patch(facecolor=fc, alpha=min(1, a + 0.25), edgecolor=ec, label=lab)
                   for lab, (fc, a, ec) in seen.items()]
        return handles
    return []


def set3d(ax):
    ax.set_xlim(-0.45, 0.85); ax.set_ylim(-0.65, 0.85); ax.set_zlim(-0.88, 0.75)
    ax.set_box_aspect((1.3, 1.5, 1.63))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=24, azim=-58)


def fig_scene(kit, plan, ik):
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    scene_handles = draw_scene(ax, kit)
    ax.scatter(*kit.dishes.T, s=40, color="#ffffff", edgecolor=TXT, marker="o", zorder=5)
    ax.scatter(*kit.slots.T, s=40, color="#ffffff", edgecolor=TXT, marker="s", zorder=5)
    for i, r in enumerate(plan["records"]):
        lp = r.leg
        dp = lp.via_points(r.dp_path); qa = lp.via_points(r.chosen_path)
        ax.plot(*dp.T, color=S1, lw=1.5, ls="--", label="DP optimum (exact)" if i == 0 else None)
        ax.plot(*qa.T, color=S2, lw=1.2, marker="o", ms=3,
                label="QAOA best sample" if i == 0 else None)
        for l in range(lp.L):
            ax.scatter(*lp.cand[l].T, s=6, color="#c9c7c0")
    F = R.fk_frames(ik["qaoa"])
    p = F[:, 6, :3, 3] + R.TOOL_LEN * F[:, 6, :3, 2]
    ax.plot(*p.T, color=S3, lw=1.0, alpha=0.9, label="executed TCP (QAOA IK)")
    A, Bp, _ = R.capsules(ik["qaoa"][[0]])
    for a, b in zip(A[0], Bp[0]):
        ax.plot(*np.c_[a, b], color=TXT2, lw=4, solid_capstyle="round")
    set3d(ax)
    ax.set_title("Kitchen scene, candidate via points, planned and executed paths")
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + scene_handles, loc="upper left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_scene_paths.png", dpi=150)
    plt.close(fig)


def fig_qaoa(M):
    legs = M["module1"]["legs"]
    tr = M["module2"]["transfer"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    ax = axs[0]
    x = np.arange(len(legs))
    w = 0.38
    ax.bar(x - w / 2, [l["p_opt"] for l in legs], w, color=S1, label="QAOA p=3 (exact sim)")
    ax.bar(x + w / 2, [l["uniform_p_opt"] for l in legs], w, color="#b9b7b0", label="uniform random")
    for i, l in enumerate(legs):
        if not l["found_optimum"]:
            ax.annotate("best sample\n+%.1f%%" % (100 * (l["qaoa_cost"] / l["dp_cost"] - 1)),
                        (i, max(l["p_opt"], 2e-3)), ha="center", va="bottom", fontsize=8, color=TXT2)
    ax.set_yscale("log")
    ax.set_xticks(x, [l["name"].replace("_", "\n") for l in legs], fontsize=8)
    ax.set_ylabel("P(sample = exact optimum)")
    ax.set_title("Module 1: path HUBO, 10 qubits, per leg")
    ax.legend(fontsize=8, loc="upper right")
    ax = axs[1]
    ps = [1, 2, 3]
    vals = [tr[str(p)]["opt_in_topk"] if str(p) in tr else tr[p]["opt_in_topk"] for p in ps]
    po = [tr[str(p)]["p_opt"] if str(p) in tr else tr[p]["p_opt"] for p in ps]
    ax.plot(ps, vals, color=S1, marker="o", ms=8, label="optimum in top-6 of 256 shots")
    ax.axhline(tr["uniform"]["opt_in_topk"], color="#8a8984", ls="--", lw=1.5,
               label="same, uniform random sampling")
    for p, v, q in zip(ps, vals, po):
        ax.annotate(f"P(opt)={q:.3f}", (p, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color=TXT2)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0.7, 3.3)
    ax.set_xticks(ps, [f"p={p}" for p in ps])
    ax.set_ylabel("fraction of held-out IK QUBOs")
    ax.set_title("Module 2: transferred angles, 400 held-out IK QUBOs")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_qaoa_quality.png", dpi=150)
    plt.close(fig)


def fig_ik(kit, ik, M):
    t, Pref, carry = ik["t"], ik["Pref"], ik["carry"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    names = [("dls", "DLS, tracking only", S2), ("lsq", "bounded LSQ, full objective", S1),
             ("qaoa", "QAOA QUBO + exact re-rank", S3)]
    for key, lab, col in names:
        Q = ik[key]
        F = R.fk_frames(Q)
        p = F[:, 6, :3, 3] + R.TOOL_LEN * F[:, 6, :3, 2]
        dmin = np.zeros(len(t))
        for c in (False, True):
            m = carry == c
            dmin[m] = E.min_clearance(kit, Q[m], t[m], c, F=F[m])
        axs[0].plot(t, 1e3 * dmin, color=col, lw=1.2, label=lab)
        axs[1].plot(t, 1e3 * np.linalg.norm(p - Pref, axis=1), color=col, lw=1.2, label=lab)
    axs[0].axhline(0, color=TXT, lw=1)
    axs[0].set_ylabel("min clearance [mm]")
    axs[0].set_ylim(-45, 120)
    axs[0].set_title("Module 2: closest approach to any obstacle (below 0 = collision)")
    axs[0].legend(fontsize=8, ncol=3, loc="upper right")
    axs[1].set_ylabel("TCP deviation [mm]")
    axs[1].set_xlabel("time [s]")
    axs[1].set_title("Deviation from the planned TCP path (tolerated in transit, tight at grasp/place)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_ik.png", dpi=150)
    plt.close(fig)


def fig_pid(pid, M):
    t = pid["t"]; Qs = pid["Qs"]; Qa = pid["Q_act"]
    names = [str(n) for n in pid["names"]]
    cols = [S2, "#8a8984", S1, S3]
    labels = {"nominal": "nominal fixed gains", "best_fixed": "best fixed gains (grid)",
              "rbf_gp": "RBF-GP schedule", "qk_gp": "quantum-kernel GP schedule"}
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    win = 25
    ker = np.ones(win) / win
    for b, nm in enumerate(names):
        e = 1e3 * np.linalg.norm(Qa[b] - Qs, axis=1)
        axs[0].plot(t, np.sqrt(np.convolve(e ** 2, ker, "same")), color=cols[b], lw=1.2,
                    label=labels[nm])
        axs[1].plot(t, pid["KP"][b], color=cols[b], lw=1.4, label=labels[nm])
    axs[0].set_ylabel("joint error norm [mrad]\n(0.25 s moving RMS)")
    axs[0].set_title("Module 3: closed-loop tracking of the IK reference (payload unknown to the model)")
    axs[0].legend(fontsize=8, ncol=2, loc="upper right")
    axs[1].set_ylabel("Kp multiplier")
    axs[1].set_xlabel("time [s]")
    axs[1].set_title("Gain schedule updated every 0.01 s")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_pid.png", dpi=150)
    plt.close(fig)


def fig_hardware():
    files = sorted(OUT.glob("hardware_*.json"))
    files = [f for f in files if "dryrun" not in f.name]
    if not files:
        return
    reps = [json.load(open(f)) for f in files]
    keys = [k for k in reps[0]["modules"] if k.startswith(("m1", "m2"))]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(keys))
    w = 0.8 / (len(reps) + 2)
    ideal = [reps[0]["modules"][k]["ideal"]["p_opt"] for k in keys]
    unif = [reps[0]["modules"][k]["uniform_p_opt"] for k in keys]
    axs[0].bar(x - w, ideal, w, color=S1, label="exact statevector")
    cols = [S3, S2, S4]
    for i, r in enumerate(reps):
        axs[0].bar(x + i * w, [r["modules"].get(k, {}).get("p_opt", 0.0) for k in keys], w,
                   color=cols[i % 3], label=r["backend"])
    axs[0].bar(x + len(reps) * w, unif, w, color="#b9b7b0", label="uniform random")
    axs[0].set_xticks(x, keys)
    axs[0].set_ylabel("P(sample = exact optimum)")
    axs[0].set_title("Ideal vs noisy: QAOA sampling quality")
    axs[0].legend(fontsize=8)
    st = [reps[-1]["modules"].get(k, {}).get("circuit", {}).get("two_qubit_gates", 0) for k in keys]
    axs[1].bar(x, st, 0.5, color=S1)
    for i, v in enumerate(st):
        axs[1].annotate(str(v), (i, v), ha="center", va="bottom", fontsize=8, color=TXT2)
    axs[1].set_xticks(x, keys)
    axs[1].set_ylabel("two-qubit gates after transpilation")
    axs[1].set_title(f"Circuit cost on {reps[-1]['backend']}")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_hardware.png", dpi=150)
    plt.close(fig)


def animate(kit, pid, ik, segs, stride=14, fps=14):
    from matplotlib.lines import Line2D
    t = pid["t"]; Q = pid["Q_act"][3]; carry = ik["carry"]
    idx = np.arange(0, len(t), stride)
    fig = plt.figure(figsize=(8.0, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False          # keep the arm drawn on top of translucent solids
    scene_handles = draw_scene(ax, kit)
    set3d(ax)
    ax.view_init(elev=24, azim=18)        # standing in the kitchen, facing the counter
    ax.xaxis.pane.set_alpha(0.0); ax.yaxis.pane.set_alpha(0.0); ax.zaxis.pane.set_alpha(0.0)
    F = R.fk_frames(Q)
    tcp = F[:, 6, :3, 3] + R.TOOL_LEN * F[:, 6, :3, 2]
    A, Bp, _ = R.capsules(Q, F=F)
    dish = R.held_dish_center(Q, F=F)
    # when each bowl leaves the sink and arrives in the rack
    t_pick = np.full(len(kit.dishes), np.inf); t_place = np.full(len(kit.dishes), np.inf)
    for sg in segs:
        if sg.kind == "lift":
            t_pick[sg.dish] = sg.t0
        if sg.kind == "release":
            t_place[sg.dish] = sg.t0 + sg.T
    bowl_off = np.array([0, 0, -R.DISH_OFFSET_Z])
    sink_bowls = kit.dishes + bowl_off
    rack_bowls = kit.slots + bowl_off
    lw = [7, 7, 6, 6, 5, 4.5, 4.5, 4]
    links = [ax.plot([], [], [], color="#3b3a37", lw=w, solid_capstyle="round", zorder=20)[0]
             for w in lw]
    joints, = ax.plot([], [], [], "o", color="#f4f3ef", mec="#3b3a37", ms=4, zorder=21)
    trail, = ax.plot([], [], [], color=S2, lw=1.2, alpha=0.9, zorder=15)
    hand = ax.scatter([], [], [], s=[], color=S2, edgecolor="#9a3d17", zorder=18)
    bottle = ax.scatter([], [], [], s=[], color=S4, edgecolor="#8a5f00", zorder=18)
    held, = ax.plot([], [], [], "o", color="#ffffff", mec=S1, mew=2.5, ms=13, zorder=22)
    in_sink = ax.scatter([], [], [], s=110, color="#ffffff", edgecolor=S1, linewidth=2, zorder=12)
    in_rack = ax.scatter([], [], [], s=110, color="#ffffff", edgecolor=S1, linewidth=2, zorder=12)
    handles = scene_handles + [
        Line2D([], [], color="#3b3a37", lw=5, label="UR5e arm"),
        Line2D([], [], marker="o", ls="", mfc="#ffffff", mec=S1, mew=2, ms=9, label="bowl"),
        Line2D([], [], marker="o", ls="", color=S2, ms=9, label="person's hand (moving)"),
        Line2D([], [], marker="o", ls="", color=S4, ms=8, label="sliding bottle (moving)"),
        Line2D([], [], color=S2, lw=1.2, label="gripper trail (3 s)"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=7.5, ncol=4, frameon=False)
    title = ax.set_title("")

    def update(fi):
        k = idx[fi]
        tk = t[k]
        for j, ln in enumerate(links):
            seg = np.c_[A[k, j], Bp[k, j]]
            ln.set_data(seg[0], seg[1]); ln.set_3d_properties(seg[2])
        J = np.vstack([A[k, 1:], Bp[k, -1:]])
        joints.set_data(J[:, 0], J[:, 1]); joints.set_3d_properties(J[:, 2])
        lo = max(0, k - 300)
        trail.set_data(tcp[lo:k + 1, 0], tcp[lo:k + 1, 1]); trail.set_3d_properties(tcp[lo:k + 1, 2])
        C = kit.mover_centers(tk)
        hand._offsets3d = ([C[0, 0]], [C[0, 1]], [C[0, 2]])
        hand.set_sizes([(kit.movers[0].radius * 1400) ** 1.2])
        bottle._offsets3d = ([C[1, 0]], [C[1, 1]], [C[1, 2]])
        bottle.set_sizes([(kit.movers[1].radius * 1400) ** 1.2])
        s_m = tk < t_pick
        r_m = tk >= t_place
        in_sink._offsets3d = tuple(sink_bowls[s_m].T) if s_m.any() else ([], [], [])
        in_rack._offsets3d = tuple(rack_bowls[r_m].T) if r_m.any() else ([], [], [])
        if carry[k]:
            held.set_data([dish[k, 0]], [dish[k, 1]]); held.set_3d_properties([dish[k, 2]])
        else:
            held.set_data([], []); held.set_3d_properties([])
        n_done = int(r_m.sum())
        title.set_text(f"t = {tk:5.2f} s    bowls in rack: {n_done}/4\n"
                       "QAOA path + QUBO IK + quantum-kernel PID (simulated)")
        return links

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    fig.subplots_adjust(left=-0.08, right=1.05, bottom=0.10, top=0.93)
    anim.save(FIG / "animation.gif", writer=PillowWriter(fps=fps), dpi=72)
    # a few stills for checking / the README
    for k_t in (2.0, 12.0, 30.0):
        update(int(np.argmin(np.abs(t[idx] - k_t))))
        fig.savefig(FIG / f"frame_t{int(k_t):02d}s.png", dpi=110)
    plt.close(fig)


def fig_m4(L):
    m4 = L["module4"]
    cl = m4["closed_loop"]
    names = list(cl)
    short = ["C0 nominal\nmodel", "C1 Slotine-Li\n(baseline)", "C1f + friction\nadaptation",
             "C2 C1 + RBF-GP", "C3 C1 + QK-GP", "C4 oracle"]
    cols = ["#b9b7b0", S2, "#8a8984", S1, S3, "#3b3a37"]
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    v = [cl[n]["rms_joint_err_mrad"] for n in names]
    axs[0].bar(range(len(v)), v, color=cols, width=0.62)
    for i, x in enumerate(v):
        axs[0].annotate(f"{x:.2f}", (i, x), ha="center", va="bottom", fontsize=8, color=TXT2)
    axs[0].set_xticks(range(len(v)), short, fontsize=8)
    axs[0].grid(axis="x", visible=False)
    axs[0].set_ylabel("joint RMS tracking error [mrad]")
    axs[0].set_title("Module 4: held-out task (new dish order and bowl masses)")
    eff = m4["data_efficiency"]
    ns = sorted({int(k.split("_")[1]) for k in eff})
    axs[1].plot(ns, [eff[f"qk_{n}"]["nmse"] for n in ns], color=S3, marker="o", ms=8, label="quantum kernel (ZZ map)")
    axs[1].plot(ns, [eff[f"rbf_{n}"]["nmse"] for n in ns], color=S1, marker="s", ms=8, label="RBF kernel (ARD)")
    from matplotlib.ticker import NullFormatter, NullLocator
    axs[1].set_xscale("log"); axs[1].set_xticks(ns, [str(n) for n in ns])
    axs[1].xaxis.set_minor_locator(NullLocator()); axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].set_xlabel("GP support points per joint")
    axs[1].set_ylabel("held-out residual nMSE (pooled)")
    axs[1].set_ylim(0, None)
    axs[1].set_title("Residual-torque prediction vs data")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_m4_residual_learning.png", dpi=150)
    plt.close(fig)


def fig_m5(L):
    m5 = L["module5"]
    curves = m5["curves"]
    seeds = sorted({int(k.split("_")[1]) for k in curves})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for kind, col, lab in (("vqc", S3, "VQC actor (54 params)"), ("mlp", S1, "MLP actor (56 params)")):
        R_ = np.array([[np.log(c["rel"]) for c in curves[f"{kind}_{s}"]] for s in seeds])
        # windows differ per iteration (shared across agents per seed): smooth with a moving mean
        w = 9
        sm = np.array([np.convolve(r, np.ones(w) / w, "valid") for r in R_])
        x = np.arange(sm.shape[1]) + w // 2
        mu, lo, hi = np.exp(sm.mean(0)), np.exp(sm.min(0)), np.exp(sm.max(0))
        axs[0].plot(x, mu, color=col, label=lab)
        axs[0].fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
    axs[0].axhline(1.0, color="#8a8984", ls="--", lw=1.2, label="nominal gains")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("training iteration (24 rollouts of 1.5 s each)")
    axs[0].set_ylabel("rollout cost / nominal-gain cost\n(9-iteration geometric mean)")
    axs[0].set_title(f"Module 5: REINFORCE training, {len(seeds)} seeds (band = min-max)")
    axs[0].legend(fontsize=8, loc="lower left")
    dep = m5["deployment"]
    labels = ["nominal", "best fixed (Module 3)", "VQC actor", "MLP actor"]
    vals = [dep["nominal"]["cost"], dep["best fixed (Module 3)"]["cost"], m5["vqc"]["mean"], m5["mlp"]["mean"]]
    cols = ["#b9b7b0", "#8a8984", S3, S1]
    axs[1].bar(range(4), vals, color=cols, width=0.6, alpha=0.55)
    for i, kind in ((2, "vqc"), (3, "mlp")):
        pts = m5[kind]["all"]
        axs[1].scatter(np.full(len(pts), i) + np.linspace(-0.12, 0.12, len(pts)), pts, s=46,
                       color=cols[i], edgecolor=TXT, linewidth=0.8, zorder=5,
                       label="individual seeds" if i == 2 else None)
    for i, x in enumerate(vals):
        top = max([x] + (m5[["", "", "vqc", "mlp"][i]]["all"] if i >= 2 else []))
        axs[1].annotate(("mean " if i >= 2 else "") + f"{x:.1f}", (i, top), ha="center", va="bottom",
                        fontsize=8, color=TXT2, xytext=(0, 5), textcoords="offset points")
    axs[1].set_xticks(range(4), labels, fontsize=8)
    axs[1].grid(axis="x", visible=False)
    axs[1].set_ylabel("closed-loop cost, held-out task (lower is better)")
    axs[1].set_title("Deployment on the held-out task")
    axs[1].set_ylim(0, 1.15 * max(vals + m5["vqc"]["all"] + m5["mlp"]["all"]))
    axs[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig7_m5_quantum_rl.png", dpi=150)
    plt.close(fig)


def main():
    kit = E.default_kitchen()
    M = json.load(open(OUT / "metrics.json"))
    plan = pickle.load(open(OUT / "plan.pkl", "rb"))
    ik = dict(np.load(OUT / "ik_traces.npz"))
    pid = dict(np.load(OUT / "pid.npz", allow_pickle=True))
    fig_scene(kit, plan, ik)
    fig_qaoa(M)
    fig_ik(kit, ik, M)
    fig_pid(pid, M)
    fig_hardware()
    lf = OUT / "learning_metrics.json"
    if lf.exists():
        L = json.load(open(lf))
        if "module4" in L:
            fig_m4(L)
        if "module5" in L:
            fig_m5(L)
    if "--no-anim" not in sys.argv:
        animate(kit, pid, ik, plan["segs"])
    print("figures in", FIG)


if __name__ == "__main__":
    main()
