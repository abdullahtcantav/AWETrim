"""Tests for awetrim.aerostructural.wireframe.reduce_fem (FEM_full -> wireframe).

The reference numbers are frozen from ``struc_geometry_PSM_reduced.yaml``
(removed 2026-10-29; in the v0.1.0 release), an exact reduction of the full
geometry's bridle: same wing nodes (the file sat 0.0466 m lower, a rigid
shift), same line set, same fixed lengths, the tip lines absorbed into their
pulleys.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerostructural.wireframe.reduce_fem import reduce_fem_geometry

DATA = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


def _load(name):
    with (DATA / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def reduced():
    return reduce_fem_geometry(_load("struc_geometry_FEM_full.yaml"))


#: struc_geometry_PSM_reduced.yaml's wing nodes [m], shifted by +0.0466 m in z
#: onto the built FEM geometry: front (odd) and rear (even) attachment per rib.
WING_NODES = {
    1: (-0.183243, 4.085799, 8.602814),
    2: (0.789240, 4.140976, 8.378346),
    3: (-0.692027, 3.967853, 9.239621),
    4: (0.839743, 3.956753, 9.236237),
    5: (-0.906897, 3.134159, 10.244571),
    6: (1.086881, 3.118511, 10.224542),
    7: (-1.050409, 1.959598, 10.846579),
    8: (1.169142, 1.947901, 10.813494),
    9: (-1.122026, 0.664217, 11.106687),
    10: (1.205432, 0.661074, 11.066604),
    11: (-1.122026, -0.664217, 11.106687),
    12: (1.205432, -0.661074, 11.066604),
    13: (-1.050409, -1.959598, 10.846579),
    14: (1.169142, -1.947901, 10.813494),
    15: (-0.906897, -3.134159, 10.244571),
    16: (1.086881, -3.118511, 10.224542),
    17: (-0.692027, -3.967853, 9.239621),
    18: (0.839743, -3.956753, 9.236237),
    19: (-0.183243, -4.085799, 8.602814),
    20: (0.789240, -4.140976, 8.378346),
}

#: struc_geometry_PSM_reduced.yaml's bridle line set (less its extra 0.07 m
#: Br_main_e link before M, the one known difference).
LINES = {
    "A1_equiv", "A2_equiv", "A3_equiv", "A4_equiv", "AI", "AII", "A_main",
    "Br_I", "Br_II", "Br_main_1", "M", "a5_equiv", "br_1", "br_2", "br_3",
    "br_4", "br_5_equiv", "depower_tape", "steering_tape",
}

#: Its fixed rest lengths [m].
FIXED_LENGTHS = {
    "AI": 3.67, "AII": 3.36, "A_main": 3.903, "br_1": 4.407, "br_2": 4.18,
    "br_3": 4.094, "br_4": 3.55, "Br_I": 2.333, "Br_II": 2.045,
    "Br_main_1": 0.61, "M": 4.856, "steering_tape": 1.6,
}


def test_wing_nodes_are_the_reduced_files(reduced):
    ours = {int(r[0]): np.array(r[1:4], float) for r in reduced["wing_particles"]["data"]}
    assert sorted(ours) == list(range(1, 21))
    for i, expected in WING_NODES.items():
        np.testing.assert_allclose(ours[i], expected, atol=1e-6, err_msg=str(i))


def test_bridle_line_set_matches(reduced):
    assert {r[0] for r in reduced["bridle_connections"]["data"]} == LINES


def test_fixed_lengths_and_absorbed_tip_pulleys(reduced):
    ours = {r[0]: float(r[1]) for r in reduced["bridle_lines"]["data"]}
    for name, length in FIXED_LENGTHS.items():
        assert ours[name] == pytest.approx(length), name
    assert ours["a5_equiv"] == pytest.approx(11.928)
    assert ours["br_5_equiv"] == pytest.approx(13.818)


def test_strut_fans_become_one_equivalent_line_each(reduced):
    rows = [r for r in reduced["bridle_connections"]["data"] if r[0].startswith("A") and "_equiv" in r[0]]
    assert len(rows) == 8
    assert all(len(r) == 3 and 1 <= r[1] <= 20 and r[1] % 2 == 1 for r in rows)


def test_wing_topology_and_compressive_tubes(reduced):
    names = [r[0] for r in reduced["wing_elements"]["data"]]
    assert sum(n.startswith("strut_") for n in names) == 10
    assert sum(n.startswith("le_") for n in names) == 9
    assert sum(n.startswith("te_") for n in names) == 9
    assert sum(n.startswith("dia_") for n in names) == 18
    linktype = {r[0]: r[5] for r in reduced["wing_elements"]["data"]}
    assert linktype["le_1"] == linktype["strut_1"] == "default"
    assert linktype["te_1"] == linktype["dia_1a"] == "noncompressive"


def test_masses_are_conserved():
    fem = _load("struc_geometry_FEM_full.yaml")
    ids = [0] + [int(r[0]) for k in ("wing_particles", "bridle_particles") for r in fem[k]["data"]]
    nodes = np.zeros((max(ids) + 1, 3))
    for k in ("wing_particles", "bridle_particles"):
        for r in fem[k]["data"]:
            nodes[int(r[0])] = r[1:4]
    masses = np.full(len(nodes), 0.1)
    masses[0] = 8.0
    reduced = reduce_fem_geometry(fem, nodes=nodes, masses=masses)
    total = reduced["reduction"]["wing_mass"] + reduced["reduction"]["bridle_mass"]
    assert total == pytest.approx(masses[1:].sum(), rel=1e-6)
