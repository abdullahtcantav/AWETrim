"""Tests for the awesIO structure document (coupled/awesio_structure.py)."""

import copy
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from awetrim.aerostructural.coupled import awesio_structure, read_struc_geometry_yaml

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"
KCU_MASS = 8.4


@pytest.fixture(scope="module")
def fem_full_geometry():
    path = KITE_DIR / "struc_geometry_FEM_full.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reader(fem_full_geometry):
    return read_struc_geometry_yaml.main(
        copy.deepcopy(fem_full_geometry), config={"kcu": {"mass": KCU_MASS}}
    )


@pytest.fixture(scope="module")
def written(fem_full_geometry, tmp_path_factory):
    """The document as a reader sees it: written to YAML and loaded back."""
    document = awesio_structure.to_awesio(
        fem_full_geometry, kcu_mass=KCU_MASS, height_kcu=200.0, name="LEI-V3"
    )
    path = tmp_path_factory.mktemp("awesio") / "structure.yml"
    awesio_structure.write_awesio(document, path)
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_document_validates_against_the_schema(written):
    validator = pytest.importorskip("awesio.validator")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validator.validate(written[0])


def test_the_input_geometry_is_not_modified(fem_full_geometry):
    before = copy.deepcopy(fem_full_geometry)
    awesio_structure.to_awesio(fem_full_geometry)
    assert fem_full_geometry == before


def test_the_connectivity_preimage_is_the_schemas_example():
    three_in_a_row = {
        "points": {"data": [["1"], ["2"], ["3"]]},
        "segments": {"data": [["s1", ["1", "2"]], ["s2", ["2", "3"]]]},
        "bodies": {"data": [["a"], ["b"]]},
        "tubes": {"data": [["t", ["a", "b"]]]},
        "canopy_faces": {"data": [["f", "w", ["1", "2", "3"]]]},
    }
    assert awesio_structure.connectivity_preimage(three_in_a_row) == "3;1,2;2,3;2;1,2;1;1,2,3;"


def test_the_written_sha_hashes_the_written_connectivity(written):
    document = written[1]
    assert awesio_structure.connectivity_sha(document) == document["metadata"]["connectivity_sha"]


def test_the_round_trip_rebuilds_the_readers_model(reader, written):
    rebuilt = awesio_structure.from_awesio(written[1])

    np.testing.assert_allclose(rebuilt[0], np.asarray(reader[0]), atol=1e-12)
    for k in (2, 3, 4, 5, 6, 7, 8, 10, 11, 17):
        np.testing.assert_array_equal(np.asarray(rebuilt[k]), np.asarray(reader[k]))
    np.testing.assert_allclose(rebuilt[12], reader[12])  # bridle diameters
    np.testing.assert_allclose(rebuilt[13], reader[13], atol=1e-12)  # rest lengths
    np.testing.assert_allclose(rebuilt[14], reader[14], rtol=1e-6)  # stiffness
    np.testing.assert_allclose(rebuilt[15], reader[15])  # damping / pressure
    assert list(rebuilt[16]) == list(reader[16])  # linktypes
    assert rebuilt[18].keys() == reader[18].keys()
    for key, pair in reader[18].items():
        np.testing.assert_allclose(rebuilt[18][key], pair, atol=1e-8)


def test_the_mass_total_is_the_readers(reader, written):
    masses = awesio_structure.nodal_masses(written[1])
    # The reader weighs a line at its rest length BEFORE the strut projection.
    assert masses.sum() == pytest.approx(np.sum(reader[1]), rel=1e-9)
    assert masses.min() >= 0.0
    # The KCU node also carries half of each tape ending on it.
    assert written[1]["points"]["data"][0][4] == pytest.approx(KCU_MASS)


def test_flat_tapes_keep_their_drag_width(fem_full_geometry, written):
    widths = {
        row[0]: row[5] for row in fem_full_geometry["bridle_lines"]["data"] if len(row) > 5
    }
    segments = awesio_structure._rows(written[1]["segments"])
    for segment in segments:
        assert segment["tape_width"] == widths.get(segment["line"])
