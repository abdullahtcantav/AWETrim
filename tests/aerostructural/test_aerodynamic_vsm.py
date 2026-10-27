"""``aerodynamic_vsm``: the CasADi trim inside the coupled adapter.

``_solve_trim_casadi`` must return the coupled loop's vocabulary (per-panel
loads on a rotated body, ``av_stage``), apply the attached-first rule at the
trim level, and re-use one built graph across calls on the same polars
(cached on the VSM solver object). LEI-V3, 27 panels; ~20 s.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("VSM")

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "examples" / "1_design"))
from common import add_common_arguments, build_body, build_system_model, parsed_common  # noqa: E402

from awetrim.aerodynamics.kcu_drag import KcuDragModel  # noqa: E402
from awetrim.aerostructural import aerodynamic_vsm  # noqa: E402


@pytest.fixture(scope="module")
def lei_v3():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args([])
    body, _ = build_body(args)
    values = parsed_common(args)
    system_model = build_system_model(args)
    return dict(body=body, values=values, system_model=system_model, kcu=KcuDragModel.from_system_model(system_model))


def _solver(av: bool):
    from VSM.core.Solver import Solver

    solver = Solver(
        gamma_loop_type="anderson", allowed_error=1e-8, is_with_artificial_viscosity=av,
        artificial_viscosity_factor=0.035, reference_point=[0.0, 0.0, 0.0],
        # The CasADi trim refuses the quarter-chord law, VSM develop's default.
        is_aoa_corrected=False,
    )
    solver._awetrim_attached_first = True
    return solver


def _call(env, solver, **extra):
    v = env["values"]
    kw = dict(
        body_aero=env["body"], solver=solver, system_model=env["system_model"],
        center_of_gravity=v["center_of_gravity"], reference_point=v["reference_point"],
        x_guess=v["x_guess"], bounds_lower=v["bounds_lower"], bounds_upper=v["bounds_upper"],
        include_gravity=False, kcu_drag=env["kcu"], tether_model=None, gamma_seed=None, tolerance=1e-8,
    )
    kw.update(extra)
    return aerodynamic_vsm._solve_trim_casadi(**kw)


def test_casadi_trim_returns_the_coupled_vocabulary(lei_v3):
    results, body = _call(lei_v3, _solver(av=False))
    n = len(lei_v3["body"].panels)
    assert results["converged"] and results["success"]
    assert np.asarray(results["F_distribution"]).shape == (n, 3)
    assert np.asarray(results["panel_cp_locations"]).shape == (n, 3)
    assert np.asarray(results["alpha_at_ac"]).ravel().shape == (n,)
    assert results["av_stage"] is None  # AV off: no attached-first stage
    assert np.asarray(body.va).shape == (3,)


def test_attached_first_at_the_trim_level(lei_v3):
    """AV on + attached_first: the unsteered LEI-V3 trims attached, the
    predictor is accepted, and the returned body carries the ORIGINAL polars
    (not the continued ones the predictor was built on)."""
    from awetrim.aerodynamics.vsm_quasi_steady import _panel_stall_onsets_rad

    results, body = _call(lei_v3, _solver(av=True))
    assert results["av_stage"] == "attached"
    onsets = _panel_stall_onsets_rad(lei_v3["body"])
    alpha = np.asarray(results["alpha_at_ac"], dtype=float).ravel()
    assert np.all(alpha <= onsets)
    for panel_out, panel_in in zip(body.panels, lei_v3["body"].panels):
        assert panel_out._panel_polar_data is panel_in._panel_polar_data


def test_attached_and_true_polar_graphs_get_separate_cache_entries(lei_v3):
    """The attached-first stage builds its graph on CONTINUED polars; a later
    true-polar solve on the same solver must get its own graph, not trip the
    cached one's polar check (the key was once taken before the swap, so every
    stalled-stage trim raised and fell back to an untrimmed direct solve)."""
    solver = _solver(av=True)
    _call(lei_v3, solver)  # attached-first: graph on the continued polars
    solver._awetrim_attached_first = False  # true polars, same solver object
    results, _ = _call(lei_v3, solver)
    assert results["converged"]
    assert len(solver._awetrim_casadi_trims) == 2


def test_graph_is_cached_on_the_solver_and_reused(lei_v3):
    solver = _solver(av=False)
    first, _ = _call(lei_v3, solver)
    cache = solver._awetrim_casadi_trims
    assert len(cache) == 1
    (system_model, trim), = cache.values()
    assert system_model is lei_v3["system_model"]
    second, _ = _call(lei_v3, solver, gamma_seed=first["gamma_distribution"], x_guess=first["opt_x"])
    assert len(cache) == 1 and next(iter(cache.values()))[1] is trim
    np.testing.assert_allclose(second["opt_x"], first["opt_x"], atol=1e-7)
    # a warm re-solve on the cached graph is far cheaper than the build
    assert second["casadi_trim"]["time_total_s"] < 0.5


def test_pre_loop_trim_config_overrides_only_its_keys():
    config = {
        "is_with_gravity": True,
        "quasi_steady_trim": {
            "max_nfev": 8,
            "yaw_bounds_deg": [-4.0, 4.0],
            "pre_loop": {"yaw_bounds_deg": [-15.0, 15.0]},
        },
    }
    pre = aerodynamic_vsm.pre_loop_trim_config(config)
    assert pre["quasi_steady_trim"]["yaw_bounds_deg"] == [-15.0, 15.0]
    assert pre["quasi_steady_trim"]["max_nfev"] == 8
    assert pre["is_with_gravity"] is True
    # the loop keeps its own bounds
    assert config["quasi_steady_trim"]["yaw_bounds_deg"] == [-4.0, 4.0]


def test_pre_loop_trim_config_without_block_is_unchanged():
    config = {"quasi_steady_trim": {"yaw_bounds_deg": [-4.0, 4.0]}}
    assert aerodynamic_vsm.pre_loop_trim_config(config) is config
    assert aerodynamic_vsm.pre_loop_trim_config(None) is None


def test_trim_just_inside_a_bound_counts_as_pinned():
    """TRF stops strictly inside the box: yaw 3.9987 on [-4, 4] (40 deg
    elevation, 2026-10-22) is pinned, an interior 3.0 is not."""
    lo = [-2.0, -4.0, -4.0, -4.0, -2.5]
    hi = [50.0, 4.0, 4.0, 4.0, 2.5]
    pinned = aerodynamic_vsm._warn_if_trim_on_bounds([20.0, 0.0, 0.0, 3.9987, 0.0], lo, hi)
    assert [(name, which) for name, _value, which in pinned] == [("yaw", "upper")]
    assert aerodynamic_vsm._warn_if_trim_on_bounds([20.0, 0.0, 0.0, 3.0, 0.0], lo, hi) == []
