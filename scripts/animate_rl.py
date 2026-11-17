"""Animation of Module 5 on the held-out task: the arm under the VQC gain policy, plus live
panels comparing gain schedules, accumulated closed-loop cost and gripper tracking error.

Needs results/ from scripts/run_learning.py. If m5_closed_loop.npz predates per-step cost
logging, the held-out run is replayed once with the saved policies (same actors, same seed,
so the trajectories are identical) and the extra traces are added to the file.

    python scripts/animate_rl.py                  # VQC seed 2 vs MLP seed 2 (best of each)
    python scripts/animate_rl.py --vqc 0 --mlp 1  # any other seeds
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
import make_figures as MF                           # noqa: E402  (palette, scene drawing)
import run_learning as RL                           # noqa: E402  (trajectories, test kitchen)
from qkitchen import control as C                   # noqa: E402
from qkitchen import qrl                            # noqa: E402
from qkitchen import robot as R                     # noqa: E402
from qkitchen.qml import MLPPolicy, VQCPolicy       # noqa: E402

OUT, FIG = MF.OUT, MF.FIG
GREY = "#8d8c88"


def load_agents():
    """Rebuild the trained actors. The output offset is a function of the initial weights
    only (qrl.center_outputs runs before training), so it is recomputed from the seed."""
    params = pickle.load(open(OUT / "m5_agents.pkl", "rb"))
    off_f = OUT / "m5_agent_offsets.pkl"
    offsets = pickle.load(open(off_f, "rb")) if off_f.exists() else {}
    agents = {}
    for name, v in params.items():
        kind, seed = name.split("_")
        seed = int(seed)
        m = VQCPolicy(6, 3, 2, seed=seed) if kind == "vqc" else MLPPolicy(6, 6, 2, seed=seed)
        if name in offsets:
            m.offset = offsets[name]
        else:
            qrl.center_outputs(m)
        m.set(np.asarray(v))
        agents[name] = m
    return agents


def ensure_traces():
    """Return the held-out closed-loop record, replaying it if step costs are missing."""
    f = OUT / "m5_closed_loop.npz"
    d = dict(np.load(f, allow_pickle=True))
    if "step_cost" in d:
        return d
    print("  replaying the held-out Module 5 run to log per-step cost and torque ...")
    tr, te = RL.train_trajectory(), RL.test_trajectory()
    gains = C.nominal_gains(tr["q0"])
    from scipy.ndimage import uniform_filter1d
    qd, qdd = C.reference_derivatives(te["Q"])
    f_te = uniform_filter1d(C.features(te["Q"], qd, qdd, 0.6 * te["carry"]), 21, axis=0)
    agents = load_agents()
    names = [str(n) for n in d["names"]]
    best_fixed = tuple(json.load(open(OUT / "metrics.json"))["module3"]["best_fixed"])
    actors = [(1.0, 1.0), best_fixed] + [agents[n] for n in names[2:]]
    pay = np.where(te["carry"], RL.MASS_TEST[te["dish_of"]], 0.0)
    N = len(te["Q"])
    t0 = time.time()
    sim = C.simulate(te["Q"], qd, pay, np.ones((len(actors), N)), np.ones((len(actors), N)),
                     gains, seed=7, record=True, gain_fn=qrl.PolicyRunner(actors, f_te, sigma=0.0))
    dev = float(np.abs(sim["Q"] - d["Q"]).max())
    print(f"    done in {time.time() - t0:.0f}s; max deviation from the saved run {dev:.2e} rad")
    if dev > 1e-6:
        raise RuntimeError("replay does not reproduce the saved held-out run")
    d.update(step_cost=sim["step_cost"], tau=sim["TAU"], cost=sim["cost"])
    np.savez(f, **d)
    return d


def animate_rl(vqc_seed=2, mlp_seed=2, stride=14, fps=14):
    d = ensure_traces()
    te = RL.test_trajectory()
    kit = RL.kit_for_test()
    names = [str(n) for n in d["names"]]
    M5 = json.load(open(OUT / "learning_metrics.json"))["module5"]["deployment"]
    vqc_costs = [M5[f"vqc_{s}"]["cost"] for s in range(3)]
    show = [("nominal", "nominal gains", GREY, "-"),
            ("best fixed (Module 3)", "best fixed gains", MF.S3, "-"),
            (f"mlp_{mlp_seed}", f"MLP actor (seed {mlp_seed})", MF.S2, "-"),
            (f"vqc_{vqc_seed}", f"VQC actor (seed {vqc_seed})", MF.S1, "-")]
    bi = {n: names.index(n) for n, *_ in show}
    t = d["t"]
    T = t[-1] - t[0]
    dt = t[1] - t[0]
    acc = np.cumsum(d["step_cost"], axis=1) * dt / T          # ends at the reported cost
    # gripper (TCP) tracking error against the reference
    Fref = R.fk_frames(d["Qref"])
    p_ref = Fref[:, 6, :3, 3] + R.TOOL_LEN * Fref[:, 6, :3, 2]
    tcp_err = {}
    for n, *_ in show:
        F = R.fk_frames(d["Q"][bi[n]])
        p = F[:, 6, :3, 3] + R.TOOL_LEN * F[:, 6, :3, 2]
        tcp_err[n] = 1e3 * np.linalg.norm(p - p_ref, axis=1)
    from scipy.ndimage import uniform_filter1d
    tcp_s = {n: uniform_filter1d(v, 25) for n, v in tcp_err.items()}   # 0.25 s window for legibility

    # ---- arm executed under the VQC policy
    Q = d["Q"][bi[f"vqc_{vqc_seed}"]]
    F = R.fk_frames(Q)
    tcp = F[:, 6, :3, 3] + R.TOOL_LEN * F[:, 6, :3, 2]
    A, Bp, _ = R.capsules(Q, F=F)
    dish = R.held_dish_center(Q, F=F)
    carry = te["carry"].astype(bool)
    dish_of = te["dish_of"]
    nd = len(kit.dishes)
    t_pick = np.full(nd, np.inf); t_place = np.full(nd, np.inf)
    for i in range(nd):
        idx_c = np.flatnonzero(carry & (dish_of == i))
        if len(idx_c):
            t_pick[i], t_place[i] = t[idx_c[0]], t[idx_c[-1]]
    bowl_off = np.array([0, 0, -R.DISH_OFFSET_Z])
    sink_bowls, rack_bowls = kit.dishes + bowl_off, kit.slots + bowl_off

    fig = plt.figure(figsize=(12.8, 7.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1.0], hspace=0.55, wspace=0.12,
                          left=0.0, right=0.93, top=0.88, bottom=0.20)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.computed_zorder = False
    scene_handles = MF.draw_scene(ax, kit)
    MF.set3d(ax)
    ax.view_init(elev=24, azim=18)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
    lw = [7, 7, 6, 6, 5, 4.5, 4.5, 4]
    links = [ax.plot([], [], [], color="#3b3a37", lw=w, solid_capstyle="round", zorder=20)[0]
             for w in lw]
    joints, = ax.plot([], [], [], "o", color="#f4f3ef", mec="#3b3a37", ms=4, zorder=21)
    trail, = ax.plot([], [], [], color=MF.S1, lw=1.2, alpha=0.9, zorder=15)
    hand = ax.scatter([], [], [], s=[], color=MF.S2, edgecolor="#9a3d17", zorder=18)
    bottle = ax.scatter([], [], [], s=[], color=MF.S4, edgecolor="#8a5f00", zorder=18)
    held, = ax.plot([], [], [], "o", color="#ffffff", mec=MF.S1, mew=2.5, ms=13, zorder=22)
    in_sink = ax.scatter([], [], [], s=110, color="#ffffff", edgecolor=MF.S1, linewidth=2, zorder=12)
    in_rack = ax.scatter([], [], [], s=110, color="#ffffff", edgecolor=MF.S1, linewidth=2, zorder=12)
    ax.set_title(f"Arm executed with the VQC gain policy (seed {vqc_seed})", fontsize=10, pad=-4)

    # ---- right-hand panels
    axg = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1]); axe = fig.add_subplot(gs[2, 1])
    for a_ in (axg, axc, axe):
        a_.set_xlim(t[0], t[-1])
        a_.tick_params(labelsize=8)
    axe.set_xlabel("time [s]", fontsize=9)
    axg.set_title("PID gain multipliers chosen every 50 ms", fontsize=10, loc="left")
    gmax = max(d["kp"][bi[n]].max() for n, *_ in show[2:])
    axg.set_ylim(0, max(1.6, 1.08 * gmax))
    axg.axhline(1.0, color=GREY, lw=1.2)
    axg.text(t[-1], 1.03, "nominal = 1", ha="right", va="bottom", fontsize=7.5, color=TXT2)
    axc.set_title("Accumulated closed-loop cost (lower is better)", fontsize=10, loc="left")
    axc.set_ylim(0, 1.08 * max(acc[bi[n], -1] for n, *_ in show))
    axe.set_title("Gripper tracking error [mm] (0.25 s average)", fontsize=10, loc="left")
    axe.set_ylim(0, 1.1 * max(np.percentile(v, 99.5) for v in tcp_s.values()))

    gain_lines = []
    for n, lab, col, _ in show[2:]:
        lk, = axg.plot([], [], color=col, lw=1.6)
        ld, = axg.plot([], [], color=col, lw=1.6, ls=(0, (3, 2)))
        gain_lines.append((n, lk, ld))
    cost_lines, err_lines, cost_tags = [], [], []
    for n, lab, col, ls in show:
        cost_lines.append((n, axc.plot([], [], color=col, lw=1.8, ls=ls)[0]))
        cost_tags.append((n, axc.text(t[-1] + 0.4, 0, "", fontsize=7.5, color=col, va="center",
                                      fontweight="bold", clip_on=False)))
        err_lines.append((n, axe.plot([], [], color=col, lw=1.3, ls=ls)[0]))
    cursors = [a_.axvline(t[0], color=TXT2, lw=0.8, alpha=0.6) for a_ in (axg, axc, axe)]

    handles = scene_handles + [Line2D([], [], color=c, lw=2, ls=s, label=l) for _, l, c, s in show] + [
        Line2D([], [], color=TXT2, lw=1.6, label="k_p multiplier"),
        Line2D([], [], color=TXT2, lw=1.6, ls=(0, (3, 2)), label="k_d multiplier")]
    fig.legend(handles=handles, loc="lower center", fontsize=7.5, ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.012,
             f"Held-out task (new dish order, new bowl masses). VQC seeds 0/1/2 cost "
             f"{vqc_costs[0]:.1f} / {vqc_costs[1]:.1f} / {vqc_costs[2]:.1f}; seed 1 failed to train. "
             "Nominal gains track best but chatter most; the learned policies trade a little "
             "tracking error for much less torque chatter.",
             ha="center", fontsize=7.5, color=TXT2)
    title = fig.suptitle("", fontsize=12, fontweight="bold", y=0.975)

    frames = np.arange(0, len(t), stride)
    if frames[-1] != len(t) - 1:
        frames = np.append(frames, len(t) - 1)

    def update(k):
        tk = t[k]
        for j, ln in enumerate(links):
            seg = np.c_[A[k, j], Bp[k, j]]
            ln.set_data(seg[0], seg[1]); ln.set_3d_properties(seg[2])
        J = np.vstack([A[k, 1:], Bp[k, -1:]])
        joints.set_data(J[:, 0], J[:, 1]); joints.set_3d_properties(J[:, 2])
        lo = max(0, k - 300)
        trail.set_data(tcp[lo:k + 1, 0], tcp[lo:k + 1, 1]); trail.set_3d_properties(tcp[lo:k + 1, 2])
        Cm = kit.mover_centers(tk)
        hand._offsets3d = ([Cm[0, 0]], [Cm[0, 1]], [Cm[0, 2]])
        hand.set_sizes([(kit.movers[0].radius * 1400) ** 1.2])
        bottle._offsets3d = ([Cm[1, 0]], [Cm[1, 1]], [Cm[1, 2]])
        bottle.set_sizes([(kit.movers[1].radius * 1400) ** 1.2])
        s_m, r_m = tk < t_pick, tk > t_place
        in_sink._offsets3d = tuple(sink_bowls[s_m].T) if s_m.any() else ([], [], [])
        in_rack._offsets3d = tuple(rack_bowls[r_m].T) if r_m.any() else ([], [], [])
        if carry[k]:
            held.set_data([dish[k, 0]], [dish[k, 1]]); held.set_3d_properties([dish[k, 2]])
        else:
            held.set_data([], []); held.set_3d_properties([])
        sl = slice(0, k + 1)
        for n, lk, ld in gain_lines:
            lk.set_data(t[sl], d["kp"][bi[n], sl]); ld.set_data(t[sl], d["kd"][bi[n], sl])
        for n, ln in cost_lines:
            ln.set_data(t[sl], acc[bi[n], sl])
        # value tags in the right margin, pushed apart so close values stay readable
        vals = sorted(((acc[bi[n], k], tx) for n, tx in cost_tags), key=lambda x: x[0])
        gap = 0.075 * axc.get_ylim()[1]
        y_prev = -np.inf
        for v, tx in vals:
            y = max(v, y_prev + gap)
            tx.set_position((t[-1] + 0.4, y)); tx.set_text(f"{v:.1f}")
            y_prev = y
        for n, ln in err_lines:
            ln.set_data(t[sl], tcp_s[n][sl])
        for c in cursors:
            c.set_xdata([tk, tk])
        title.set_text(f"Module 5 – quantum RL gain scheduling on the held-out task    "
                       f"t = {tk:5.2f} s    bowls in rack: {int(r_m.sum())}/{nd}")
        return links

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(FIG / "animation_rl.gif", writer=PillowWriter(fps=fps), dpi=72)
    _shrink_gif(FIG / "animation_rl.gif", fps)
    update(frames[-1])
    fig.savefig(FIG / "animation_rl_final.png", dpi=110)
    plt.close(fig)
    print(f"  wrote {FIG / 'animation_rl.gif'} ({len(frames)} frames)")


TXT2 = MF.TXT2


def _shrink_gif(path, fps, colors=96):
    """Re-encode with one shared palette and frame differencing (keeps it under ~8 MB)."""
    from PIL import Image, ImageSequence
    im = Image.open(path)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    pal = frames[-1].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0,
              optimize=True, disposal=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqc", type=int, default=2, help="VQC seed to show (0-2)")
    ap.add_argument("--mlp", type=int, default=2, help="MLP seed to show (0-2)")
    ap.add_argument("--stride", type=int, default=18)
    ap.add_argument("--fps", type=int, default=11)
    a = ap.parse_args()
    animate_rl(a.vqc, a.mlp, a.stride, a.fps)
