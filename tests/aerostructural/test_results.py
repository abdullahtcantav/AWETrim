"""Unit tests for the pure helpers in awetrim.aerostructural.results.

Covers the sweep-value generator, the restart directory-candidate logic, the
result-root path builders, and the deformed-geometry rewriter. None of these
touch a solver or the filesystem (path objects are built, not created).
"""

from pathlib import Path

import numpy as np
import pytest

from awetrim.aerostructural.results import (
    aerostructural_results_root,
    build_deformed_struc_geometry,
    candidate_case_dirs,
    legacy_results_root,
    steering_values_from_count_or_step,
)


# --- steering_values_from_count_or_step ------------------------------------


def test_steering_values_single_point():
    out = steering_values_from_count_or_step(0.1, 0.5, n_values=1)
    assert out == pytest.approx([0.1])


def test_steering_values_count_is_inclusive_linspace():
    out = steering_values_from_count_or_step(0.0, 2.0, n_values=3)
    assert out == pytest.approx([0.0, 1.0, 2.0])


def test_steering_values_step_is_inclusive_of_end():
    out = steering_values_from_count_or_step(0.0, 1.0, step_m=0.5)
    assert out == pytest.approx([0.0, 0.5, 1.0])


def test_steering_values_rejects_zero_count():
    with pytest.raises(ValueError):
        steering_values_from_count_or_step(0.0, 1.0, n_values=0)


def test_steering_values_requires_step_or_count():
    with pytest.raises(ValueError):
        steering_values_from_count_or_step(0.0, 1.0)


def test_steering_values_rejects_reversed_range_with_step():
    with pytest.raises(ValueError):
        steering_values_from_count_or_step(1.0, 0.0, step_m=0.5)


# --- result-root path builders ---------------------------------------------


def test_results_roots():
    root = Path("/proj")
    assert aerostructural_results_root(root, "V3") == root / "results" / "V3" / "aerostructural"
    assert legacy_results_root(root, "V3") == root / "results" / "aerostructural" / "V3"


# --- candidate_case_dirs ----------------------------------------------------


def test_candidate_case_dirs_adds_sign_variant():
    dirs = candidate_case_dirs(Path("/proj"), "V3", "depower_p0000mm_steer_p0150mm")
    names = {d.name for d in dirs}
    # The p0000mm token is mirrored to an m0000mm variant.
    assert "depower_p0000mm_steer_p0150mm" in names
    assert "depower_m0000mm_steer_p0150mm" in names
    # Two roots (canonical + legacy) times two variants.
    assert len(dirs) == 4
    assert all(isinstance(d, Path) for d in dirs)


def test_candidate_case_dirs_without_zero_token_has_single_variant():
    dirs = candidate_case_dirs(Path("/proj"), "V3", "depower_p0150mm_steer_p0150mm")
    # No p0000mm/m0000mm token to mirror -> one variant per root.
    assert len(dirs) == 2


# --- build_deformed_struc_geometry -----------------------------------------


def test_build_deformed_struc_geometry_replaces_nodes_by_index():
    struc_geometry = {
        "bridle_point_node": [0.0, 0.0, 0.0],
        "wing_particles": {
            "data": [
                [0, 1.0, 1.0, 1.0],
                [1, 2.0, 2.0, 2.0],
            ]
        },
    }
    struc_nodes = np.array(
        [
            [10.0, 11.0, 12.0],  # node 0
            [20.0, 21.0, 22.0],  # node 1
        ]
    )
    out = build_deformed_struc_geometry(struc_geometry, struc_nodes)

    # Original is not mutated (deepcopy contract).
    assert struc_geometry["wing_particles"]["data"][0][1] == 1.0

    assert out["bridle_point_node"] == [10.0, 11.0, 12.0]
    assert out["wing_particles"]["data"][0][1:] == [10.0, 11.0, 12.0]
    assert out["wing_particles"]["data"][1][1:] == [20.0, 21.0, 22.0]
    # The node-id column is preserved.
    assert out["wing_particles"]["data"][1][0] == 1


# --- live bridle rest lengths in the snapshot ------------------------------

import copy  # noqa: E402

import yaml  # noqa: E402

from awetrim.aerostructural.utils import (  # noqa: E402
    BRIDLE_CONNECTION_REST_LENGTHS,
    bridle_connection_rest_lengths,
)

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


def _toy_geometry():
    """Two steering tapes on one row, a depower tape and one pulley rope."""
    return {
        "bridle_point_node": [0.0, 0.0, 0.0],
        "wing_particles": {"data": [[1, 0.0, 1.0, 5.0], [2, 0.0, -1.0, 5.0]]},
        "bridle_particles": {"data": [[3, 0.0, 0.0, 2.0], [4, 0.0, 0.0, 1.0]]},
        "bridle_lines": {
            "headers": ["name", "l0", "d", "material", "linktype"],
            "data": [
                ["steering_tape", 1.6, 0.005, "dyneema", "noncompressive"],
                ["depower_tape", 1.5, 0.005, "dyneema", "noncompressive"],
                ["M", 9.0, 0.004, "dyneema", "pulley"],
            ],
        },
        "bridle_connections": {
            "headers": ["name", "ci", "cj", "ck"],
            "data": [
                ["steering_tape", 0, 1],
                ["M", 1, 3, 2],
                ["steering_tape", 0, 2],
                ["depower_tape", 0, 4],
            ],
        },
    }


def _nodes():
    return np.array([[0, 0, 0], [0, 1, 5], [0, -1, 5], [0, 0, 2], [0, 0, 1]], dtype=float)


def test_snapshot_writes_the_live_tape_lengths():
    # element order: steering, M arm, M arm, steering, depower
    live = np.array([1.58, 4.4, 4.6, 1.62, 1.58])
    geometry = _toy_geometry()
    out = build_deformed_struc_geometry(geometry, _nodes(), bridle_rest_lengths=live)
    rows = {row[0]: row for row in out["bridle_lines"]["data"]}
    assert rows["depower_tape"][1] == pytest.approx(1.58)
    assert rows["M"][1] == 9.0  # pulley rope: never actuated, kept
    # asymmetric steering: mean in the row, each tape per connection
    assert rows["steering_tape"][1] == pytest.approx(1.60)
    assert bridle_connection_rest_lengths(out) == pytest.approx({0: 1.58, 2: 1.62})
    assert geometry["bridle_lines"]["data"][1][1] == 1.5  # input untouched


def test_symmetric_snapshot_needs_no_per_connection_block():
    live = np.array([1.55, 4.4, 4.6, 1.55, 1.5])
    geometry = _toy_geometry()
    geometry[BRIDLE_CONNECTION_REST_LENGTHS] = {"headers": ["row", "l0"], "data": [[0, 9.9]]}
    out = build_deformed_struc_geometry(geometry, _nodes(), bridle_rest_lengths=live)
    assert BRIDLE_CONNECTION_REST_LENGTHS not in out  # a stale block is dropped
    assert {row[0]: row[1] for row in out["bridle_lines"]["data"]}["steering_tape"] == pytest.approx(1.55)


def test_snapshot_rest_lengths_skip_nan_and_check_the_count():
    live = np.array([np.nan, 4.4, 4.6, np.nan, 1.58])
    out = build_deformed_struc_geometry(_toy_geometry(), _nodes(), bridle_rest_lengths=live)
    assert {row[0]: row[1] for row in out["bridle_lines"]["data"]}["steering_tape"] == 1.6
    with pytest.raises(ValueError):
        build_deformed_struc_geometry(_toy_geometry(), _nodes(), bridle_rest_lengths=live[:4])


def _round_trip(read, geometry, l0_at, bridle_at, power_at, steering_at):
    """Actuate the tapes in a reader's element arrays, write the snapshot,
    read it back: the reader must return the actuated rest lengths."""
    first = read(copy.deepcopy(geometry))
    l0 = np.asarray(first[l0_at], dtype=float).copy()
    n_bridle = len(first[bridle_at])
    left, right = first[steering_at][:2]
    l0[first[power_at]] += 0.08
    l0[left] -= 0.03
    l0[right] += 0.03
    snapshot = build_deformed_struc_geometry(
        geometry, np.asarray(first[0], dtype=float), bridle_rest_lengths=l0[-n_bridle:]
    )
    again = np.asarray(read(copy.deepcopy(snapshot))[l0_at], dtype=float)
    tapes = [first[power_at], left, right]
    np.testing.assert_allclose(again[tapes], l0[tapes], rtol=0, atol=1e-12)
    # Everything else as before; the full-model reader re-splits its pulley
    # arms from the node positions, which round-trip to ~1e-8 m with or
    # without this change.
    np.testing.assert_allclose(again, l0, rtol=0, atol=1e-7)


def _wireframe():
    """The LEI-V3 PSM-format wireframe: the reduction of the FEM geometry."""
    from awetrim.aerostructural.wireframe.reduce_fem import reduce_fem_geometry

    return reduce_fem_geometry(
        yaml.safe_load((KITE_DIR / "struc_geometry_FEM_full.yaml").read_text(encoding="utf-8"))
    )


def test_wireframe_reader_reads_the_snapshot_tapes_back():
    from awetrim.aerostructural.wireframe import structural_geometry_io

    geometry = _wireframe()
    _round_trip(structural_geometry_io.main, geometry, l0_at=10, bridle_at=8, power_at=4, steering_at=5)


def test_full_model_reader_reads_the_snapshot_tapes_back():
    from awetrim.aerostructural.coupled import read_struc_geometry_yaml

    geometry = yaml.safe_load((KITE_DIR / "struc_geometry_FEM_full.yaml").read_text(encoding="utf-8"))
    system = yaml.safe_load((KITE_DIR / "system.yaml").read_text(encoding="utf-8"))
    _round_trip(
        lambda g: read_struc_geometry_yaml.main(g, system_config=system),
        geometry, l0_at=13, bridle_at=11, power_at=4, steering_at=5,
    )
