# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Newton (course-frame) versus Lagrangian form of the quasi-steady trim.

Companion of ``docs/cycle_optimisation/lagrangian_formulation.md``. Builds the
LEI-V3 point-mass model on a periodic B-spline path (the reel-out of
``cycle_configs/helix_spline.yaml`` with a Lissajous shape, as the spline
integration tests do), writes the trim residual once in the course-frame
Newton form the library uses (``SystemModel.force_residual``) and once from
the Euler-Lagrange equations in the generalised coordinates (s, r) plus the
normal constraint row, and reports

- the maximum relative mismatch of the two residuals over random states,
- SX instruction counts of the per-node functions (full spline and the
  local-support variant the NLP uses), split into inertia terms and path
  kinematics,
- with ``--nlp``: IPOPT's timing statistics on a reel-out NLP (shape and
  depower free) so the share of function evaluations in the wall time is
  visible. ``--turn-radius R`` adds the production turn-radius rows,
  ``--kinematics-cache`` shares the ParametrizedKinematics sub-expressions
  (every property built once per instance, smaller SX graph, same
  mathematics), ``--exact-hessian`` swaps L-BFGS for the exact Hessian:
  the follow-up A/B of the write-up.

Run from the repo root:
    python scripts/reduced-order-model/optimization/studies/compare_lagrangian_residual.py [--nlp] [--n-points N]
"""
import argparse
import copy
import functools
import time

import casadi as ca
import numpy as np

from awetrim.environment.Wind import Wind
from awetrim.kinematics import Kinematics as _kinematics_module
from awetrim.kinematics.Kinematics import ParametrizedKinematics
from awetrim.kinematics.parametrized_patterns import (
    create_pattern_from_dict,
    make_bspline_path_parameters_from_named_curve,
)
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.phase import Phase
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.utils.config_paths import LEI_V3_HELIX_SPLINE_CONFIG, LEI_V3_SYSTEM_CONFIG
from awetrim.utils.utils import load_cycle_config_from_yaml

S_INIT, S_FINAL = 0.0, 2.0 * np.pi


def reelout_config(n_points, M=10, r0=230.0):
    config, _ = load_cycle_config_from_yaml(LEI_V3_HELIX_SPLINE_CONFIG)
    config = copy.deepcopy(config)
    config["pattern_type"] = "spline_periodic"
    config["path_parameters"] = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic", M=M, r0=r0, s_init=S_INIT, s_final=S_FINAL,
        n_fit=200, curve_type="lissajous", az_amp0=0.32, beta0=0.3,
        beta_amp0=0.15, downloops=True,
    )
    sim = config["sim_parameters"]
    sim["start_angle"], sim["end_angle"], sim["n_points"] = S_INIT, S_FINAL, n_points
    return config


def v3_system_model(wind_speed=10.0):
    system_model = create_system_model_from_yaml(yaml_path=LEI_V3_SYSTEM_CONFIG)
    wind = Wind(wind_model="uniform", z0=0.03, direction_wind=0)
    wind.speed_wind_ref = wind_speed
    system_model.wind = wind
    return system_model


def start_state(config):
    return {
        "t": 0.0, "s": S_INIT, "s_dot": 3.0, "input_steering": 0.0,
        "tension_tether_ground": 8.4e4, "speed_radial": 0.0,
        "distance_radial": config["path_parameters"]["r0"],
    }


def lagrangian_residual(km, kin):
    """Quasi-steady trim residual from the Euler-Lagrange equations.

    Generalised coordinates s (path parameter) and r, kinetic energy
    T = m/2 (r_dot^2 + r^2 sigma'^2 s_dot^2), generalised forces
    Q_s = F . dR/ds = r sigma' F_x and Q_r = F_z, plus the normal constraint
    row for the lateral force the steering must supply. Under the
    quasi-steady assumption (v_tau_dot = v_r_dot = 0, which fixes s_ddot;
    it is not s_ddot = 0) the rows, divided by their metric factors so each
    is a force in the course frame, are

        F_x - m v_tau v_r / r = 0,   F_y + m k_g v_tau^2 / r = 0,
        F_z + m v_tau^2 / r = 0,

    with k_g the geodesic curvature of the path on the unit sphere.
    """
    m = km.kite.mass_wing + km.kite.mass_kcu
    F = km.force_external
    r, vr, vtau = km.distance_radial, km.speed_radial, kin.vtau
    inertia = ca.vertcat(
        m * vtau * vr / r,
        -m * kin.curvature_geodesic * vtau**2 / r,
        -m * vtau**2 / r,
    )
    return F - inertia


def share_kinematics_subexpressions():
    """Turn every ParametrizedKinematics property into a cached one.

    The properties rebuild the spline sub-graph (phi, dphi_ds, chi, ...) on
    each access and the SX expansion keeps every copy; caching builds each
    once per instance. Same mathematics, 33-64 % fewer node instructions.
    """
    cls = _kinematics_module.ParametrizedKinematics
    for name, attr in list(vars(cls).items()):
        if isinstance(attr, property):
            cached = functools.cached_property(attr.fget)
            cached.__set_name__(cls, name)
            setattr(cls, name, cached)


def n_ops(name, ins, expr):
    return ca.Function(name, ins, [expr]).expand().n_instructions()


def compare_residuals(config, system_model):
    phase = PhaseParameterized(system_model, quasi_steady=True, pattern_config=config)
    pattern = create_pattern_from_dict(config["pattern_type"], config["path_parameters"])
    m = system_model.kite.mass_wing + system_model.kite.mass_kcu

    def build(pat):
        km = phase.substitute_parametrized_kinematics(pattern=pat)
        km.establish_residual()
        kin = ParametrizedKinematics(pat, phase)
        ins = [phase.s, phase.s_dot, km.input_steering, km.tension_tether_ground,
               km.speed_radial, km.distance_radial, km.input_depower]
        return km, kin, ins

    km, kin, ins = build(pattern)
    f_newton = ca.Function("newton", ins, [km.residual]).expand()
    f_lagr = ca.Function("lagrange", ins, [lagrangian_residual(km, kin)]).expand()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        args = [rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 4.0), rng.uniform(-0.3, 0.3),
                rng.uniform(1e3, 1e4), rng.uniform(-3, 3), rng.uniform(150, 400),
                rng.uniform(1.6, 2.2)]
        a = np.asarray(f_newton(*args)).ravel()
        b = np.asarray(f_lagr(*args)).ravel()
        worst = max(worst, float(np.max(np.abs(a - b) / (np.abs(a) + 1.0))))
    print(f"max relative row mismatch Newton vs Lagrange (200 random states): {worst:.1e}")

    print(f"\nSX instructions, full periodic spline (M={pattern.M}):")
    print(f"  Newton residual                         {f_newton.n_instructions():6d}")
    print(f"  Lagrange residual                       {f_lagr.n_instructions():6d}")
    print(f"  inertia, Newton (m a incl. kinematics)  {n_ops('ni', ins, m * km.acceleration):6d}")
    print(f"  path kinematics phi, beta, chi, chi_dot, v_tau "
          f"{n_ops('kin', ins, ca.vertcat(km.angle_azimuth, km.angle_elevation, km.angle_course, km.timeder_angle_course, kin.vtau)):6d}")
    print(f"  force_aerodynamic (ROM)                 {n_ops('aero', ins, km.expression('force_aerodynamic')):6d}")
    # Inertia with the kinematic quantities as plain symbols: the marginal
    # cost of the Newton form once the path quantities exist.
    km2 = copy.deepcopy(system_model)
    for nm in ("angle_course", "timeder_angle_course", "speed_tangential",
               "angle_azimuth", "angle_elevation"):
        setattr(km2, nm, ca.MX.sym(nm))
    km2.timeder_speed_radial = km2.timeder_speed_tangential = 0
    ins2 = [km2.angle_course, km2.timeder_angle_course, km2.speed_tangential,
            km2.angle_elevation, km2.speed_radial, km2.distance_radial]
    print(f"  inertia given the kinematics (Newton)   {n_ops('dyn', ins2, m * km2.acceleration):6d}")

    km_k, kin_k, ins_k = build(pattern.local_support(0))
    x = ca.vertcat(*ins_k)
    res_l = lagrangian_residual(km_k, kin_k)
    print("\nSX instructions, local-support node function (what the NLP calls):")
    print(f"  Newton residual   {n_ops('nk', ins_k, km_k.residual):6d}   jacobian {n_ops('jn', ins_k, ca.jacobian(km_k.residual, x)):6d}")
    print(f"  Lagrange residual {n_ops('lk', ins_k, res_l):6d}   jacobian {n_ops('jl', ins_k, ca.jacobian(res_l, x)):6d}")


def profile_nlp(config, system_model, exact_hessian=False):
    """One reel-out NLP (C_phi, C_beta, input_depower free) with IPOPT timing."""
    phase = Phase(system_model=system_model, pattern_config=config,
                  start_state=start_state(config))
    t0 = time.perf_counter()
    phase.run_simulation(run_plots=False, phase_sim=True)
    print(f"\n[march] {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    opti, _, objective_dict, _ = phase.get_opti_components(
        optimization_params=["C_phi", "C_beta", "input_depower"]
    )
    opti.minimize(-(objective_dict["energy"] / objective_dict["total_time"]
                    / objective_dict["power_scale"]))
    # Same options as Phase.run_opti, plus the timing report.
    ipopt_options = {
        "tol": 1e-5, "acceptable_iter": 5, "acceptable_tol": 1.0,
        "acceptable_constr_viol_tol": 1e-4, "acceptable_obj_change_tol": 1e-4,
        "dual_inf_tol": 1e-4, "hessian_approximation": "limited-memory",
        "mu_strategy": "adaptive", "nlp_scaling_method": "gradient-based",
        "linear_solver": "mumps", "mu_min": 1e-8, "max_iter": 1000,
        "print_timing_statistics": "yes", "print_level": 3,
    }
    if exact_hessian:
        ipopt_options["hessian_approximation"] = "exact"
    opti.solver("ipopt", {"expand": True, "ipopt": ipopt_options})
    print(f"[build] {time.perf_counter() - t0:.1f} s  nx={opti.nx} ng={opti.ng}")
    try:
        opti.solve()
    except RuntimeError as exc:
        print("solve failed:", str(exc)[:200])
    st = opti.stats()
    tot = st["t_wall_total"]
    print(f"\n[solve] {st['iter_count']} iterations, {tot:.2f} s IPOPT wall, "
          f"{st['return_status']}")
    for k in ("t_wall_nlp_g", "t_wall_nlp_jac_g", "t_wall_nlp_f", "t_wall_nlp_grad_f"):
        print(f"  {k:18s} {st[k]:6.3f} s  {st[k] / tot * 100:5.1f} %")
    fe = sum(st[k] for k in st if k.startswith("t_wall_") and k != "t_wall_total")
    print(f"  function evaluations total {fe:.2f} s = {fe / tot * 100:.0f} % of IPOPT wall")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nlp", action="store_true", help="also profile a reel-out NLP")
    parser.add_argument("--n-points", type=int, default=120)
    parser.add_argument("--turn-radius", type=float, default=0.0,
                        help="min_turn_radius [m] rows in the NLP (0 = off)")
    parser.add_argument("--kinematics-cache", action="store_true",
                        help="share the ParametrizedKinematics sub-expressions")
    parser.add_argument("--exact-hessian", action="store_true",
                        help="IPOPT exact Hessian instead of L-BFGS")
    args = parser.parse_args()
    if args.kinematics_cache:
        share_kinematics_subexpressions()
    config = reelout_config(args.n_points)
    if args.turn_radius > 0.0:
        config["sim_parameters"]["min_turn_radius"] = args.turn_radius
    system_model = v3_system_model()
    compare_residuals(config, system_model)
    if args.nlp:
        profile_nlp(config, system_model, exact_hessian=args.exact_hessian)


if __name__ == "__main__":
    main()
