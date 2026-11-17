"""Kitchen scene: sink, faucet, counter, upper cabinet, pulled-out dishwasher rack,
plus two moving obstacles (a person's hand reaching toward the sink and a bottle
sliding along the counter). World frame = UR5e base frame, counter top at z = 0,
+x points into the room.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import robot as R


@dataclass
class Box:
    name: str
    lo: np.ndarray
    hi: np.ndarray

    @property
    def center(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def half(self):
        return 0.5 * (self.hi - self.lo)


def _box(name, x, y, z):
    return Box(name, np.array([x[0], y[0], z[0]], float), np.array([x[1], y[1], z[1]], float))


@dataclass
class MovingSphere:
    name: str
    radius: float
    kind: str
    params: dict

    def center(self, t):
        t = np.asarray(t, float)
        if self.kind == "reach":       # smooth periodic reach between two points
            a, b, T = self.params["a"], self.params["b"], self.params["period"]
            s = 0.5 * (1 - np.cos(2 * np.pi * (t + self.params.get("phase", 0.0)) / T))
            return a + s[..., None] * (b - a) if t.ndim else a + s * (b - a)
        if self.kind == "slide_y":     # sinusoidal slide along y
            c, amp, T = self.params["c"], self.params["amp"], self.params["period"]
            y = c[1] + amp * np.sin(2 * np.pi * t / T)
            out = np.broadcast_to(c, np.shape(t) + (3,)).copy()
            out[..., 1] = y
            return out
        raise ValueError(self.kind)


@dataclass
class Kitchen:
    boxes: list = field(default_factory=list)
    movers: list = field(default_factory=list)
    dishes: np.ndarray = None        # TCP grasp positions in the sink (N,3)
    slots: np.ndarray = None         # TCP place positions in the rack (N,3)
    sink_approach_z: float = 0.14
    rack_approach_z: float = -0.08

    def __post_init__(self):
        names = [b.name for b in self.boxes]
        self.support_idx = [names.index(n) for n in ("sink_floor", "rack_floor") if n in names]
        self._lo = np.array([b.lo for b in self.boxes])
        self._hi = np.array([b.hi for b in self.boxes])
        self._c = 0.5 * (self._lo + self._hi)
        self._h = 0.5 * (self._hi - self._lo)

    # ------------------------------------------------------------ distances
    def point_box_sd(self, P):
        """Signed distance from points P (...,3) to every box -> (..., nbox)."""
        d = np.abs(P[..., None, :] - self._c) - self._h
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
        inside = np.minimum(d.max(-1), 0.0)
        return outside + inside

    def mover_centers(self, t):
        return np.stack([m.center(t) for m in self.movers], axis=-2)   # (...,nm,3)

    @property
    def mover_radii(self):
        return np.array([m.radius for m in self.movers])


def default_kitchen() -> Kitchen:
    B = []
    # counter slab (z in [-0.04, 0]) with a sink cut-out x in [-0.12, 0.27], y in [0.22, 0.68]
    B.append(_box("counter_main", (-0.45, 0.30), (-0.90, 0.30), (-0.04, 0.0)))
    B.append(_box("counter_left", (-0.45, 0.30), (0.74, 0.95), (-0.04, 0.0)))
    B.append(_box("counter_back_strip", (-0.45, -0.10), (0.30, 0.74), (-0.04, 0.0)))
    B.append(_box("counter_front_strip", (0.26, 0.30), (0.30, 0.74), (-0.04, 0.0)))
    # sink basin (floor + 4 walls)
    B.append(_box("sink_floor", (-0.10, 0.26), (0.30, 0.74), (-0.23, -0.20)))
    B.append(_box("sink_wall_back", (-0.11, -0.10), (0.29, 0.75), (-0.23, -0.04)))
    B.append(_box("sink_wall_front", (0.26, 0.27), (0.29, 0.75), (-0.23, -0.04)))
    B.append(_box("sink_wall_right", (-0.11, 0.27), (0.29, 0.30), (-0.23, -0.04)))
    B.append(_box("sink_wall_left", (-0.11, 0.27), (0.74, 0.75), (-0.23, -0.04)))
    # faucet column + spout reaching over the back of the sink
    B.append(_box("faucet_column", (-0.21, -0.15), (0.49, 0.55), (0.0, 0.32)))
    B.append(_box("faucet_spout", (-0.21, -0.03), (0.50, 0.54), (0.27, 0.32)))
    # back wall and upper cabinet
    B.append(_box("back_wall", (-0.50, -0.45), (-0.90, 0.90), (-0.90, 0.95)))
    B.append(_box("upper_cabinet", (-0.45, -0.12), (-0.90, 0.90), (0.62, 0.95)))
    # lower cabinet front face below the counter edge
    B.append(_box("lower_front", (0.29, 0.30), (-0.90, 0.90), (-0.85, -0.04)))
    # dishwasher: open door + pulled-out upper rack (wire basket)
    B.append(_box("dw_door", (0.30, 0.78), (-0.54, -0.04), (-0.84, -0.80)))
    B.append(_box("rack_floor", (0.32, 0.72), (-0.52, -0.06), (-0.40, -0.38)))
    B.append(_box("rack_wall_in", (0.315, 0.32), (-0.52, -0.06), (-0.40, -0.28)))
    B.append(_box("rack_wall_out", (0.72, 0.725), (-0.52, -0.06), (-0.40, -0.28)))
    B.append(_box("rack_wall_r", (0.315, 0.725), (-0.525, -0.52), (-0.40, -0.28)))
    B.append(_box("rack_wall_l", (0.315, 0.725), (-0.06, -0.055), (-0.40, -0.28)))
    B.append(_box("rack_tines", (0.51, 0.52), (-0.52, -0.06), (-0.38, -0.33)))

    movers = [
        MovingSphere("hand", 0.07, "reach",
                     dict(a=np.array([0.80, 0.62, 0.25]), b=np.array([0.37, 0.52, 0.13]),
                          period=7.0, phase=0.0)),
        MovingSphere("bottle", 0.05, "slide_y",
                     dict(c=np.array([0.24, -0.10, 0.05]), amp=0.22, period=9.0)),
    ]
    # bowls: grasp TCP so the held-dish sphere bottom is 1 cm above the sink floor
    grasp_z = -0.20 + 0.01 + R.DISH_RADIUS + R.DISH_OFFSET_Z
    place_z = -0.38 + 0.01 + R.DISH_RADIUS + R.DISH_OFFSET_Z
    dishes = np.array([[0.16, 0.41, grasp_z], [0.16, 0.63, grasp_z],
                       [0.00, 0.41, grasp_z], [0.00, 0.63, grasp_z]])
    slots = np.array([[0.415, -0.17, place_z], [0.615, -0.17, place_z],
                      [0.415, -0.39, place_z], [0.615, -0.39, place_z]])
    return Kitchen(boxes=B, movers=movers, dishes=dishes, slots=slots)


# ---------------------------------------------------------------- clearance
N_SAMPLES = 6
_S = np.linspace(0.0, 1.0, N_SAMPLES)


def pair_distances(kit: Kitchen, q: np.ndarray, t, carrying: bool, F=None):
    """All pairwise clearances [m] for configurations q (B,6) at time(s) t.

    Returns (B, P) distances and a list of P pair labels. Negative = penetration.
    Pairs: link capsules x boxes, link capsules x moving spheres, self pairs,
    and (if carrying) the held dish x boxes / movers / proximal links.
    """
    q = np.atleast_2d(q)
    if F is None:
        F = R.fk_frames(q)
    A, E, rad = R.capsules(q, F=F)
    Bn = q.shape[0]
    P = A[:, :, None, :] + _S[None, None, :, None] * (E - A)[:, :, None, :]   # (B,8,S,3)
    # capsules vs boxes
    sd = kit.point_box_sd(P).min(axis=2) - rad[None, :, None]                  # (B,8,nb)
    sd[:, 0, :] = np.inf      # the base column is bolted to the counter: static pairs are moot
    out = [sd.reshape(Bn, -1)]
    # capsules vs moving spheres
    t_arr = np.broadcast_to(np.asarray(t, float), (Bn,))
    C = kit.mover_centers(t_arr)                                                 # (B,nm,3)
    dseg = _seg_point_dist(A[:, :, None, :], E[:, :, None, :], C[:, None, :, :])  # (B,8,nm)
    out.append((dseg - rad[None, :, None] - kit.mover_radii[None, None, :]).reshape(Bn, -1))
    # self collision
    sp = []
    for i, j in R.SELF_PAIRS:
        d = _seg_point_dist(A[:, j, None, :], E[:, j, None, :], P[:, i]).min(axis=1)
        sp.append(d - rad[i] - rad[j])
    out.append(np.stack(sp, axis=1))
    # held dish
    dc = R.held_dish_center(q, F=F)
    if carrying:
        dd_box = kit.point_box_sd(dc) - R.DISH_RADIUS
        dd_box[:, kit.support_idx] = np.inf     # the bowl is meant to touch its support surfaces
        dd_mov = np.linalg.norm(dc[:, None, :] - C, axis=-1) - R.DISH_RADIUS - kit.mover_radii
        dd_self = np.stack([_seg_point_dist(A[:, j], E[:, j], dc) - R.DISH_RADIUS - rad[j]
                            for j in R.DISH_SELF], axis=1)
    else:
        nb, nm = len(kit.boxes), len(kit.movers)
        dd_box = np.full((Bn, nb), np.inf)
        dd_mov = np.full((Bn, nm), np.inf)
        dd_self = np.full((Bn, len(R.DISH_SELF)), np.inf)
    out += [dd_box, dd_mov, dd_self]
    return np.concatenate(out, axis=1)


def pair_labels(kit: Kitchen):
    labels = []
    for c in R.CAPSULE_NAMES:
        labels += [f"{c}|{b.name}" for b in kit.boxes]
    for c in R.CAPSULE_NAMES:
        labels += [f"{c}|{m.name}" for m in kit.movers]
    labels += [f"{R.CAPSULE_NAMES[i]}|{R.CAPSULE_NAMES[j]}" for i, j in R.SELF_PAIRS]
    labels += [f"dish|{b.name}" for b in kit.boxes]
    labels += [f"dish|{m.name}" for m in kit.movers]
    labels += [f"dish|{R.CAPSULE_NAMES[j]}" for j in R.DISH_SELF]
    return labels


def min_clearance(kit, q, t, carrying, F=None):
    return pair_distances(kit, q, t, carrying, F=F).min(axis=1)


def _seg_point_dist(a, b, p):
    ab = b - a
    tt = np.clip(np.sum((p - a) * ab, -1) / np.maximum(np.sum(ab * ab, -1), 1e-12), 0.0, 1.0)
    proj = a + tt[..., None] * ab
    return np.linalg.norm(p - proj, axis=-1)


def point_clearance(kit: Kitchen, P: np.ndarray, t, radius: float):
    """Clearance of spheres of given radius centred at P (N,3) at times t (N,)."""
    d_box = kit.point_box_sd(P).min(-1)
    C = kit.mover_centers(np.asarray(t, float))
    d_mov = (np.linalg.norm(P[:, None, :] - C, axis=-1) - kit.mover_radii).min(-1)
    return np.minimum(d_box, d_mov) - radius
