"""Small checks that can be verified by hand or brute force (run: python -m pytest tests -q)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qkitchen import control as C, env as E, pipeline as PL, planner as P, robot as R  # noqa: E402
from qkitchen.qaoa import (DiagonalProblem, build_qaoa_circuit, parameter_values,  # noqa: E402
                           qaoa_state, term_union)


def test_walsh_expansion_is_exact():
    rng = np.random.default_rng(0)
    E_ = rng.normal(size=2 ** 7)
    pr = DiagonalProblem.from_energies(E_)
    assert np.allclose(pr.pruned_energies(), E_)


def test_fast_simulator_matches_qiskit_statevector():
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(1)
    pr = DiagonalProblem.from_energies(rng.normal(size=2 ** 6))   # includes 6-local terms
    sups = term_union([pr])
    qc, _, _ = build_qaoa_circuit(6, sups, 2, measure=False)
    g, b = rng.uniform(0, 1, 2), rng.uniform(0, 1, 2)
    sv = Statevector(qc.assign_parameters(parameter_values(pr, sups, g, b, circuit=qc))).data
    ps = qaoa_state(pr.normalized_energies(), g, b)
    assert abs(abs(np.vdot(sv, ps)) - 1) < 1e-9


def test_dp_equals_brute_force_on_a_real_leg():
    kit = E.default_kitchen()
    q0 = PL.find_start_config(kit, np.array([*kit.dishes[0][:2], kit.sink_approach_z]), n_seeds=150)
    A = np.array([*kit.dishes[0][:2], kit.sink_approach_z])
    B = np.array([*kit.slots[0][:2], kit.rack_approach_z])
    leg = P.build_leg(kit, A, B, 2.1, 2.6, True, q0, q0, PL.HALF)
    ks, c = leg.dp_solve()
    Eg = leg.energies()
    assert np.isclose(Eg.min(), c) and leg.encode(ks) == int(np.argmin(Eg))


def test_fk_jacobian_by_finite_differences():
    rng = np.random.default_rng(2)
    q = rng.uniform(-2, 2, 6)
    Jv, _ = R.jacobian(q[None])
    num = np.zeros((3, 6))
    for j in range(6):
        d = np.zeros(6); d[j] = 1e-6
        num[:, j] = (R.tcp_pose((q + d)[None])[0][0] - R.tcp_pose((q - d)[None])[0][0]) / 2e-6
    assert np.allclose(Jv[0], num, atol=1e-6)


def test_kernel_compute_uncompute_matches_statevector():
    from qiskit.quantum_info import Statevector
    qk = C.QuantumKernel(4, bandwidth=0.5)
    x, y = np.array([0.1, 0.5, 0.9, 0.3]), np.array([0.7, 0.2, 0.4, 0.8])
    qc = qk.compute_uncompute(x, y).remove_final_measurements(inplace=False)
    p0 = abs(Statevector(qc).data[0]) ** 2
    assert np.isclose(p0, qk.gram(x[None], y[None])[0, 0])


def test_gravity_torque_matches_potential_energy_gradient():
    rng = np.random.default_rng(3)
    q = rng.uniform(-2, 2, 6)

    def V(qq):
        F = R.fk_frames(qq[None])
        Jv, m = R._point_mass_jacobians(F, np.array([0.5]))
        Rr = F[:, 1:, :3, :3]; Pp = F[:, 1:, :3, 3]
        Cc = Pp + np.einsum("blij,lj->bli", Rr, R.LINK_COM)
        z6 = F[:, 6, :3, 2]
        zs = np.r_[Cc[0, :, 2], (F[0, 6, :3, 3] + R.TOOL_COM_Z * z6[0])[2],
                   (F[0, 6, :3, 3] + (R.TOOL_LEN + R.DISH_OFFSET_Z) * z6[0])[2]]
        return R.GRAVITY * float(m[0] @ zs)
    g = R.gravity_torque(q[None], np.array([0.5]))[0]
    num = np.array([(V(q + e) - V(q - e)) / 2e-6 for e in np.eye(6) * 1e-6])
    assert np.allclose(g, num, atol=1e-5)


# ---------------------------------------------------------------- Modules 4-5
def test_zz_kernel_simulator_matches_qiskit():
    from qiskit.quantum_info import Statevector
    from qkitchen.qml import ZZKernel
    rng = np.random.default_rng(4)
    for reps in (1, 2, 3):
        k = ZZKernel(4, reps=reps, bandwidth=np.array([0.2, 0.5, 0.9, 1.3]))
        F = rng.uniform(0, 1, (2, 4))
        fm = k.qiskit_circuit()
        S = k.states(F)
        for b in range(2):
            sv = Statevector(fm.assign_parameters(dict(zip(fm.parameters, k.encode(F[b])[0])))).data
            assert np.allclose(sv, S[b], atol=1e-12)


def test_vqc_policy_matches_qiskit_and_parameter_shift_is_exact():
    from qiskit.quantum_info import Statevector
    from qkitchen.qml import VQCPolicy
    rng = np.random.default_rng(5)
    v = VQCPolicy(6, 3, 2, seed=1)
    v.set(rng.normal(0, 1, v.n_params))
    X = rng.uniform(-1, 1, (3, 6))
    out = v.forward(X)
    qc = v.qiskit_circuit(measure=False)
    idx = np.arange(64)
    for b in range(3):
        p = Statevector(qc.assign_parameters(dict(zip(qc.parameters, X[b])))).probabilities()
        z = [np.sum(p * (1 - 2 * ((idx >> k) & 1))) for k in range(2)]
        assert np.allclose(z, out[b], atol=1e-12)
    G = rng.normal(size=(3, 2))
    g = v.grad(X, G)
    th = v.get()
    num = np.zeros_like(th)
    for j in range(len(th)):
        e = np.zeros_like(th); e[j] = 1e-6
        v.set(th + e); fp = np.sum(v.forward(X) * G)
        v.set(th - e); fm = np.sum(v.forward(X) * G)
        num[j] = (fp - fm) / 2e-6
    v.set(th)
    assert np.allclose(g, num, atol=1e-7)


def test_slotine_li_model_terms_match_rigid_body_dynamics():
    from qkitchen import learning_control as LC
    rng = np.random.default_rng(6)
    q = rng.uniform(-2, 2, (2, 6)); qd = rng.normal(0, 1, (2, 6)); qdd = rng.normal(0, 2, (2, 6))
    pay = np.array([0.7, 0.0])
    tau, Yp = LC.model_terms(q, qd, qd, qdd, pay, R.ARMATURE)
    M = R.mass_matrix(q, pay)
    ref = np.einsum("bij,bj->bi", M, qdd) + R.coriolis(q, qd, pay) + R.gravity_torque(q, pay)
    assert np.allclose(tau, ref, atol=1e-4)
    # payload regressor is exactly the payload's share of the dynamics
    tau0, _ = LC.model_terms(q, qd, qd, qdd, np.zeros(2), R.ARMATURE)
    assert np.allclose(tau - tau0, Yp * pay[:, None], atol=1e-4)
