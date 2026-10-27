"""Run one coupled VSM <-> wireframe (PSM) aerostructural case.

The wireframe is Billow's line-system fidelity: tension-only cables,
frictionless pulleys and the wing as a spring net (see ``structural_wireframe``).
:func:`solve_deformation` is the whole pipeline: resolve the kite's config and
geometry, build the VSM body and the structure, run the coupled fixed-point
loop (which converges the quasi-steady trim against the deforming structure)
and write ``sim_output.h5``, the deformed geometry snapshot and ``case.json``.

Moved here from ``examples/2_aerostructural/run_simulation_PSM.py`` on
2026-10-07; that script now only sets the inputs and calls this function.
"""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml as _yaml

from awesio.validator import validate as awesio_validate

from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.case import (
    build_actuation_case_folder,
    build_system_model,
    build_tether,
    DEFAULT_KITE_NAME,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)
from awetrim.aerostructural.case_view import build_case_view, save_case_view
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.results import (
    aerostructural_results_root,
    build_deformed_aero_geometry,
    build_deformed_struc_geometry,
    save_geometry_snapshot,
    save_input_snapshot,
    save_sim_output,
)
from awetrim.aerostructural.utils import load_sim_output, load_yaml, rotate_geometry
from awetrim.aerostructural.wireframe import (
    coupled_solver_qsm,
    structural_geometry_io,
    structural_wireframe,
)
from awetrim.plotting.kite_structure import plot_3d_kite_structure
from awetrim.utils.config_paths import REPO_ROOT
from awetrim.utils.system_config import get_tether


def _resolve_starting_struc_nodes(
    config,
    project_dir,
    kite_name,
    struc_nodes_default,
):
    """
    Optionally override start nodes from a previous simulation result folder.

    Priority:
      1) config["starting_from_sim_subdir"] (new)
      2) config["starting_from_sim_of_date"] (legacy)

    The value is treated as a subdir under results/<kite_name>/, e.g.
    depower_p0100mm_steer_m0020mm/run_003.

    If both keys are empty, return struc_nodes_default.
    """
    sim_subdir = str(config.get("starting_from_sim_subdir", "")).strip()
    if sim_subdir == "":
        sim_subdir = str(config.get("starting_from_sim_of_date", "")).strip()

    if sim_subdir == "":
        return struc_nodes_default

    base_results_dir = Path(project_dir) / "results" / kite_name

    # Candidate 1: exact path from config
    candidates = [base_results_dir / sim_subdir]

    # Candidate 2/3: tolerate zero-sign naming mismatch, e.g. m0000mm vs p0000mm
    sim_subdir_m_to_p = sim_subdir.replace("m0000mm", "p0000mm")
    sim_subdir_p_to_m = sim_subdir.replace("p0000mm", "m0000mm")
    if sim_subdir_m_to_p != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_m_to_p)
    if sim_subdir_p_to_m != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_p_to_m)

    start_dir = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            start_dir = cand
            break

    if start_dir is None:
        raise FileNotFoundError(
            "Configured starting simulation directory does not exist. "
            f"Tried: {', '.join(str(c) for c in candidates)}"
        )

    # Preferred: direct case-folder storage (sim_output.h5 inside start_dir).
    h5_path = start_dir / "sim_output.h5"
    # Backward compatibility: if not found, try legacy run_XXX layout.
    if not h5_path.exists():
        run_dirs = [
            d
            for d in start_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()
        ]
        if len(run_dirs) > 0:
            start_dir = sorted(run_dirs, key=lambda p: int(p.name[4:]))[-1]
            logging.info(
                f"Using latest legacy run folder inside case folder: {start_dir.name}"
            )
            h5_path = start_dir / "sim_output.h5"

    if not h5_path.exists():
        raise FileNotFoundError(
            f"Configured starting simulation has no sim_output.h5: {h5_path}"
        )

    _, tracking_data = load_sim_output(h5_path)
    if "positions" not in tracking_data:
        raise KeyError(f"Expected 'positions' dataset in: {h5_path}")

    positions = np.asarray(tracking_data["positions"])
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"Invalid positions shape in {h5_path}: {positions.shape}. Expected (nt, n_nodes, 3)."
        )

    struc_nodes_loaded = np.array(positions[-1], dtype=float)
    if struc_nodes_loaded.shape != np.asarray(struc_nodes_default).shape:
        raise ValueError(
            "Loaded node shape does not match current geometry. "
            f"loaded={struc_nodes_loaded.shape}, current={np.asarray(struc_nodes_default).shape}"
        )

    logging.info(
        f"Starting from previous simulation final nodes: {start_dir} (n_nodes={len(struc_nodes_loaded)})"
    )
    return struc_nodes_loaded


def _resolve_starting_rest_lengths(
    config,
    project_dir,
    kite_name,
    l0_arr_default,
):
    """
    Optionally load final rest_lengths from a previous simulation result.

    If config["starting_from_sim_subdir"] is set, load the rest_lengths from the
    corresponding H5 file. Otherwise return l0_arr_default.

    Args:
        config: Configuration dictionary
        project_dir: Path to project root
        kite_name: Name of the kite
        l0_arr_default: Default rest length array from YAML geometry

    Returns:
        np.ndarray: Updated rest lengths (or defaults if not recovering)
    """
    sim_subdir = str(config.get("starting_from_sim_subdir", "")).strip()
    if sim_subdir == "":
        sim_subdir = str(config.get("starting_from_sim_of_date", "")).strip()

    if sim_subdir == "":
        return l0_arr_default

    base_results_dir = Path(project_dir) / "results" / kite_name

    # Candidate 1: exact path from config
    candidates = [base_results_dir / sim_subdir]

    # Candidate 2/3: tolerate zero-sign naming mismatch
    sim_subdir_m_to_p = sim_subdir.replace("m0000mm", "p0000mm")
    sim_subdir_p_to_m = sim_subdir.replace("p0000mm", "m0000mm")
    if sim_subdir_m_to_p != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_m_to_p)
    if sim_subdir_p_to_m != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_p_to_m)

    start_dir = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            start_dir = cand
            break

    if start_dir is None:
        # No previous sim found, return defaults
        return l0_arr_default

    # Try to find sim_output.h5
    h5_path = start_dir / "sim_output.h5"
    if not h5_path.exists():
        run_dirs = [
            d
            for d in start_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()
        ]
        if len(run_dirs) > 0:
            start_dir = sorted(run_dirs, key=lambda p: int(p.name[4:]))[-1]
            h5_path = start_dir / "sim_output.h5"

    if not h5_path.exists():
        logging.warning(
            f"No sim_output.h5 found in {start_dir}, using default rest lengths"
        )
        return l0_arr_default

    # Load rest_lengths from metadata
    try:
        metadata, _ = load_sim_output(h5_path)
        if "rest_lengths" in metadata:
            rest_lengths_loaded = np.asarray(metadata["rest_lengths"], dtype=float)
            if rest_lengths_loaded.shape == np.asarray(l0_arr_default).shape:
                logging.info(
                    f"Loaded final rest lengths from previous simulation: {start_dir.name}"
                )
                return rest_lengths_loaded
            else:
                logging.warning(
                    f"Loaded rest_lengths shape {rest_lengths_loaded.shape} "
                    f"does not match current geometry {np.asarray(l0_arr_default).shape}, "
                    f"using defaults"
                )
                return l0_arr_default
        else:
            logging.warning(
                f"No 'rest_lengths' in {h5_path} metadata, using default rest lengths"
            )
            return l0_arr_default
    except Exception as e:
        logging.warning(
            f"Error loading rest_lengths from {h5_path}: {e}, using defaults"
        )
        return l0_arr_default


def _deformed_aero_geometry(aero_geometry, nodes, le_indices, te_indices, mapper):
    """The deformed aero geometry, at the skin's sections when there is one."""
    if mapper is None:
        return build_deformed_aero_geometry(aero_geometry, nodes, le_indices, te_indices)
    edges = mapper.map(nodes)
    step = max(int(mapper.n_panels_per_section), 1)
    le, te = edges.leading_edge_points[::step], edges.trailing_edge_points[::step]
    stacked = np.vstack([nodes, le, te])
    n = len(nodes)
    return build_deformed_aero_geometry(
        aero_geometry, stacked, list(range(n, n + len(le))),
        list(range(n + len(le), n + len(le) + len(te))),
    )


_READER_FIELDS = (
    "struc_nodes", "m_arr", "le_indices", "te_indices", "power_tape_index",
    "steering_tape_indices", "pulley_node_indices", "kite_connectivity",
    "bridle_connectivity", "bridle_diameter", "l0", "k", "c", "linktype",
    "pulley_line_indices", "pulley_line_to_other_node_pair_dict",
)


@dataclass
class WireframeCase:
    """What does not change between coupled solves of one kite and config."""

    kite_name: str
    project_dir: Path
    config: dict
    system_config: dict
    system_config_path: Path
    struc_geometry_path: Path
    struc_geometry: dict
    aero_geometry: dict
    body_aero: Any
    vsm_solver: Any
    vel_app: Any
    initial_polar_data: Any
    tether: Any
    structural_to_aero: Any = None


@dataclass
class PreparedWireframe:
    """One live wireframe structure, ready to be solved (and solved again).

    ``reader`` keeps the geometry reader's arrays by name, with ``l0`` the
    rest lengths the structure was built with. The VSM objects are this
    structure's own copies, carried across solves like the structure itself.
    """

    psystem: Any
    reader: dict
    struc_nodes_initial: np.ndarray
    m_arr: np.ndarray
    aero2struc_mapping: Any
    body_aero: Any
    vsm_solver: Any
    initial_polar_data: Any

    def rest_length(self, element_index: int) -> float:
        """Live rest length [m] of one element (pulley arms split)."""
        return float(structural_wireframe.get_rest_lengths(self.psystem)[int(element_index)])

    @property
    def power_tape_length(self) -> float:
        """Live rest length [m] of the depower tape."""
        return self.rest_length(self.reader["power_tape_index"])

    @property
    def steering_half_difference(self) -> float:
        """Live steering-tape half-difference [m] about the built lengths
        (+ shortens the first tape and lengthens the second)."""
        first, second = self.reader["steering_tape_indices"][:2]
        return 0.5 * (
            (self.reader["l0"][first] - self.rest_length(first))
            + (self.rest_length(second) - self.reader["l0"][second])
        )

    def pin_bridle_modulus(self, factor: float) -> None:
        """Every bridle line (tapes included) at ``factor`` x its own
        stiffness, the wing at its own: the fixed-stiffness model. Pair it
        with ``structural_pss.update_stiffness: false`` so the elongation
        bound does not ratchet the stiffness from solve to solve."""
        k = np.asarray(structural_wireframe.get_stiffnesses(self.psystem), dtype=float).copy()
        n_bridle = len(self.reader["bridle_connectivity"])
        k[len(k) - n_bridle:] *= float(factor)
        structural_wireframe.set_stiffnesses(self.psystem, k)

    def restore(self, sim_output_path) -> None:
        """Put the structure back where a saved solve ended: its final node
        positions, rest lengths and the stiffnesses it ran with.

        The structure IS the wireframe's whole state (a minimum-energy solve
        carries no velocity), so a continuation resumes from a saved step as
        it would have gone on. Unlike ``starting_from_sim_subdir``, the built
        rest lengths in ``reader["l0"]`` are kept: they are the reference the
        steering half-difference is measured from.
        """
        metadata, tracking = load_sim_output(sim_output_path)
        positions = np.asarray(tracking["positions"], dtype=float)[-1]
        rest_lengths = np.asarray(metadata["rest_lengths"], dtype=float)
        current = np.asarray(structural_wireframe.get_rest_lengths(self.psystem), dtype=float)
        if positions.shape != self.psystem.positions.shape or rest_lengths.shape != current.shape:
            raise ValueError(
                f"{sim_output_path} does not fit this structure: positions "
                f"{positions.shape} vs {self.psystem.positions.shape}, rest lengths "
                f"{rest_lengths.shape} vs {current.shape}"
            )
        for particle, position in zip(self.psystem.particles, positions):
            particle.update_pos(position)
        for index in np.flatnonzero(rest_lengths != current):
            self.psystem.update_rest_length(int(index), rest_lengths[index] - current[index])
        if "final_stiffnesses" in metadata:
            structural_wireframe.set_stiffnesses(
                self.psystem, np.asarray(metadata["final_stiffnesses"], dtype=float)
            )


def build_case(
    config: dict | None = None,
    *,
    config_overrides: dict | None = None,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    tether_diameter: float | None = None,
    system_config_path=None,
    struc_geometry_path=None,
    structural_to_aero=None,
    n_panels_aero=None,
) -> WireframeCase:
    """Load the kite's files and build the VSM body and the tether, once.

    ``config`` defaults to the kite's ``as_config.yaml``; ``config_overrides``
    is merged over it at the TOP level only (a nested block replaces the
    file's whole block -- pass a complete ``config`` for nested changes).
    ``system_config_path`` defaults to ``system_flown_2019.yaml``,
    ``struc_geometry_path`` to the kite's PSM geometry. ``tether_diameter``
    overrides the system file's.
    """
    project_dir = Path(project_dir) if project_dir is not None else REPO_ROOT
    config_path, aero_geometry_path, default_struc_geometry_path = resolve_kite_paths(
        project_dir, kite_name
    )
    struc_geometry_path = Path(struc_geometry_path or default_struc_geometry_path)

    # Load and validate the awesIO system config (single source of truth for physical params)
    # ``system_config_path`` lets a caller drive the deformation with a variant
    # of the kite's system.yaml -- e.g. system_flown_2019.yaml, whose KCU mass is the
    # as-flown 22.75 kg rather than the 8.4 kg optimisation value. The KCU mass
    # is read from here (see fem.read_struc_geometry_yaml._resolve_kcu_mass) and
    # again by build_system_model, so overriding the path is the only way
    # to keep the structural cloud and the coupled QSM trim on the same mass.
    system_config_path = (
        Path(system_config_path)
        if system_config_path is not None
        else Path(project_dir) / "data" / kite_name / "system_flown_2019.yaml"
    )
    with system_config_path.open("r", encoding="utf-8") as _f:
        system_config = _yaml.safe_load(_f)
    awesio_validate(system_config, restrictive=False)

    if config is None:
        config = load_yaml(config_path)
    if config_overrides:
        config = {**config, **config_overrides}
    struc_geometry = load_yaml(struc_geometry_path)
    aero_geometry = load_yaml(aero_geometry_path)

    n_wing_struc_nodes = len(struc_geometry["wing_particles"]["data"])
    n_struc_ribs = n_wing_struc_nodes / 2
    if n_panels_aero is None:
        n_panels_aero = (n_struc_ribs - 1) * config["aerodynamic"][
            "n_aero_panels_per_struc_section"
        ]
    bridle_path = (
        struc_geometry_path if config.get("is_with_aero_bridle", False) else None
    )
    body_aero, vsm_solver, vel_app, initial_polar_data = aerodynamic_vsm.initialize(
        aero_geometry_path,
        config,
        n_panels_aero,
        bridle_path=bridle_path,
    )
    # ``tether_diameter`` lets a caller analyse the deformation with the same
    # tether it will use downstream; without it the deformation silently runs on
    # system.yaml's diameter while the caller's trim uses another one, and the
    # shape is then produced under loads that do not match the analysis.
    tether = build_tether(config, system_config, diameter=tether_diameter)
    return WireframeCase(
        kite_name=kite_name,
        project_dir=Path(project_dir),
        config=config,
        system_config=system_config,
        system_config_path=system_config_path,
        struc_geometry_path=struc_geometry_path,
        struc_geometry=struc_geometry,
        aero_geometry=aero_geometry,
        body_aero=body_aero,
        vsm_solver=vsm_solver,
        vel_app=vel_app,
        initial_polar_data=initial_polar_data,
        tether=tether,
        structural_to_aero=structural_to_aero,
    )


def prepare_structure(case: WireframeCase, config: dict | None = None) -> PreparedWireframe:
    """Read the geometry and build one wireframe structure, ready to be solved.

    ``config`` defaults to the case's. Honours the restart keys
    (``starting_from_sim_subdir``) for the start nodes and rest lengths.
    """
    config = case.config if config is None else config
    reader = dict(
        zip(
            _READER_FIELDS,
            structural_geometry_io.main(
                case.struc_geometry, config=config, system_config=case.system_config
            ),
        )
    )

    #####################################################
    ### rotating the initial geometry by some angle,
    ### to enable the wind to be horizontal
    #####################################################
    struc_nodes = rotate_geometry(
        reader["struc_nodes"],
        **resolve_initial_geometry_rotation_kwargs(config),
    )
    struc_nodes = _resolve_starting_struc_nodes(
        config=config,
        project_dir=case.project_dir,
        kite_name=case.kite_name,
        struc_nodes_default=struc_nodes,
    )
    # Also recover the final rest_lengths (element l0 values) from previous simulation if available
    reader["l0"] = _resolve_starting_rest_lengths(
        config=config,
        project_dir=case.project_dir,
        kite_name=case.kite_name,
        l0_arr_default=reader["l0"],
    )
    m_arr = reader["m_arr"]

    # logging initial conditions
    logging.info(f"\n\nINITIAL CONDITIONS, NODES \n")
    for idx, (node_i, m_i) in enumerate(zip(struc_nodes, m_arr)):
        logging.info(f"node_idx: {idx}: node: {node_i}, mass: {m_i}")

    logging.info(f"\n\nINITIAL CONDITIONS, ELEMENTS \n")
    for idx, conn in enumerate(reader["kite_connectivity"]):
        logging.info(
            f"conn_idx: {idx}: conn: {conn}, l0: {reader['l0'][idx]}, k: {reader['k'][idx]}, c: {reader['c'][idx]}, linktype: {reader['linktype'][idx]}"
        )

    psystem, _pss_initial_conditions, _pss_params, struc_nodes_initial = (
        structural_wireframe.instantiate(
            config,
            struc_nodes,
            m_arr,
            reader["kite_connectivity"],
            reader["l0"],
            reader["k"],
            reader["c"],
            reader["linktype"],
            reader["pulley_line_to_other_node_pair_dict"],
        )
    )
    if config["is_with_initial_structure_plot"]:
        plot_3d_kite_structure(
            struc_nodes,
            reader["kite_connectivity"],
            reader["power_tape_index"],
            k_arr=reader["k"],
            c_arr=reader["c"],
            linktype_arr=reader["linktype"],
            pulley_nodes=reader["pulley_node_indices"],
        )

    ##################
    ### AERO2STRUC ###
    ##################
    aero2struc_mapping = (
        BilinearAeroToStructuralLoadMapper()
        .initialize(
            case.body_aero.panels,
            struc_nodes,
            reader["le_indices"],
            reader["te_indices"],
        )
        .panel_corner_map
    )
    return PreparedWireframe(
        psystem=psystem,
        reader=reader,
        struc_nodes_initial=struc_nodes_initial,
        m_arr=m_arr,
        aero2struc_mapping=aero2struc_mapping,
        body_aero=copy.deepcopy(case.body_aero),
        vsm_solver=copy.deepcopy(case.vsm_solver),
        initial_polar_data=copy.deepcopy(case.initial_polar_data),
    )


def solve_step(
    case: WireframeCase,
    prepared: PreparedWireframe,
    config: dict | None = None,
    *,
    final_state: dict | None = None,
):
    """One coupled solve on ``prepared``, from wherever it stands now.

    ``config`` (default the case's) sets the flight condition and the
    actuation of THIS solve: the depower tape walks
    ``power_tape_final_extension`` [m] from its current rest length in
    ``power_tape_extension_step`` increments, the steering half-difference
    ``steering_tape_final_extension`` [m] from the current tape lengths in
    ``steering_tape_extension_step`` increments (0 = no steering walk).
    ``final_state`` is passed on to ``coupled_solver_qsm.main`` and also
    receives the ``system_model`` the solve was trimmed with.

    Returns ``(tracking_data, meta)`` of ``coupled_solver_qsm.main``.
    """
    config = case.config if config is None else config
    reader = prepared.reader
    psystem = prepared.psystem
    power_tape_index = reader["power_tape_index"]
    steering_tape_indices = reader["steering_tape_indices"]

    #################
    ### ACTUATION ###
    #################
    initial_length_power_tape = prepared.power_tape_length
    power_tape_extension_step = config["power_tape_extension_step"]
    power_tape_final_extension = config["power_tape_final_extension"]
    if power_tape_extension_step != 0:
        n_power_tape_steps = int(power_tape_final_extension / power_tape_extension_step)
    else:
        n_power_tape_steps = 0
    logging.info(f"Initial depower tape length: {initial_length_power_tape:.3f}m")
    logging.info(
        f"Desired depower tape length: {initial_length_power_tape + power_tape_final_extension:.3f}m"
    )

    initial_length_steering_left = prepared.rest_length(steering_tape_indices[0])
    initial_length_steering_right = prepared.rest_length(steering_tape_indices[1])
    steering_tape_extension_step = config["steering_tape_extension_step"]
    steering_tape_final_extension = config["steering_tape_final_extension"]
    logging.info(
        f"Initial steering tape lengths: left={initial_length_steering_left:.3f}m, "
        f"right={initial_length_steering_right:.3f}m"
    )
    logging.info(
        f"Desired steering extension target: {steering_tape_final_extension:.3f}m "
        f"with internal step {steering_tape_extension_step:.3f}m"
    )

    ########################################
    # AWETRIM SYSTEM MODEL
    ########################################
    tether = case.tether
    _tether_cfg = config.get("tether", {}) or {}
    _tether_d = tether.diameter_tether
    _tether_rho = get_tether(case.system_config)["structure"].get("density", 970.0)
    logging.info(
        "Tether model for the deformation: %s (d=%.4f m, rho=%.1f kg/m3), "
        "include_in_trim=%s",
        type(tether).__name__,
        _tether_d,
        _tether_rho,
        bool(_tether_cfg.get("include_in_trim", False)),
    )
    m_arr = prepared.m_arr
    mass_total = float(np.sum(m_arr))
    print(f"Total structural mass (sum of particle masses): {mass_total:.3f} kg")
    system_model = build_system_model(case.system_config_path, tether, m_arr, config)
    # Report the tether the system model ACTUALLY ends up with. The factory
    # inside build_system_model prints the tether it built from system.yaml and
    # is then overwritten by ``tether`` -- so that earlier line is stale and
    # reads as if rigid-lumped were in use even when it is not.
    print(
        f"  -> system model tether in use: {type(system_model.tether).__name__} "
        f"(diameter={system_model.tether.diameter_tether:g}, "
        f"in trim={bool(_tether_cfg.get('include_in_trim', False))})"
    )

    ########################################
    ### AEROSTUCTURAL COUPLED SIMULATION ###
    ########################################
    tracking_data, meta = coupled_solver_qsm.main(
        m_arr=m_arr,
        struc_nodes=np.asarray(psystem.positions, dtype=float).copy(),
        struc_nodes_initial=prepared.struc_nodes_initial,
        system_model=system_model,
        config=config,
        ### ACTUATION
        initial_length_power_tape=initial_length_power_tape,
        n_power_tape_steps=n_power_tape_steps,
        power_tape_final_extension=power_tape_final_extension,
        power_tape_extension_step=power_tape_extension_step,
        initial_length_steering_left=initial_length_steering_left,
        initial_length_steering_right=initial_length_steering_right,
        steering_tape_indices=steering_tape_indices,
        steering_tape_final_extension=steering_tape_final_extension,
        steering_tape_extension_step=steering_tape_extension_step,
        ### CONNECTIVITY
        kite_connectivity_arr=reader["kite_connectivity"],
        bridle_connectivity_arr=reader["bridle_connectivity"],
        pulley_line_indices=reader["pulley_line_indices"],
        pulley_line_to_other_node_pair_dict=reader["pulley_line_to_other_node_pair_dict"],
        ### STRUC --> AERO
        struc_node_le_indices=reader["le_indices"],
        struc_node_te_indices=reader["te_indices"],
        ### AERO
        body_aero=prepared.body_aero,
        vsm_solver=prepared.vsm_solver,
        vel_app=case.vel_app,
        initial_polar_data=prepared.initial_polar_data,
        bridle_diameter_arr=reader["bridle_diameter"],
        # Keep the VSM bridle-line drag tracking the deforming/actuated bridle
        # (the initial static bridle mis-trims roll on steered shapes).
        bridle_line_specs=aerodynamic_vsm.parse_bridle_line_specs(
            case.struc_geometry, config
        ),
        ### AERO --> STRUC
        aero2struc_mapping=prepared.aero2struc_mapping,
        power_tape_index=power_tape_index,
        ### STRUC
        psystem=psystem,
        structural_to_aero=case.structural_to_aero,
        final_state=final_state,
    )
    if final_state is not None:
        final_state["system_model"] = system_model
    return tracking_data, meta


def solve_deformation(
    config: dict | None = None,
    *,
    config_overrides: dict | None = None,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    results_dir=None,
    tether_diameter: float | None = None,
    system_config_path=None,
    struc_geometry_path=None,
    structural_to_aero=None,
    n_panels_aero=None,
) -> dict:
    """Run one PSS/QSM aerostructural deformation and snapshot the result.

    This is the whole coupled pipeline: resolve the kite's config/geometry,
    initialise the VSM body and the PSS structure, run the fixed-point
    aero-structural loop (which converges the VSM quasi-steady trim against the
    deforming structure), then write ``sim_output.h5`` and -- when the config
    sets ``is_save_geometry_snapshots`` -- the deformed ``struc_geometry.yaml``
    / ``aero_geometry.yaml`` / ``system.yaml`` into the results directory.
    It is :func:`build_case`, :func:`prepare_structure` and one
    :func:`solve_step`; a continuation calls :func:`solve_step` again on the
    same prepared structure.

    ``config_overrides`` is merged over the kite's ``config.yaml`` -- use it to
    set the flight condition (``angle_elevation_deg``, ``angle_azimuth_deg``,
    ``angle_course_deg``, ``speed_radial``, ``distance_radial``,
    ``wind_speed_wind_ref``) and ``is_save_geometry_snapshots``.

    ``struc_geometry_path`` overrides the kite's default PSM geometry (e.g.
    the reduction :func:`solve_deformation_reduced_fem` writes).
    ``structural_to_aero`` (a ``SkinStructuralToAeroMapper``) and
    ``n_panels_aero`` fly an aerodynamic mesh other than the ribs' own.

    ``system_config_path`` overrides the kite's ``system.yaml`` (default
    ``data/<kite_name>/system_flown_2019.yaml``), e.g. to deform with the
    as-flown KCU mass.

    Returns a dict with the results directory, the deformed nodes, the solver
    tracking data/meta, and the pieces the caller needs for reporting.
    """
    case = build_case(
        config,
        config_overrides=config_overrides,
        kite_name=kite_name,
        project_dir=project_dir,
        tether_diameter=tether_diameter,
        system_config_path=system_config_path,
        struc_geometry_path=struc_geometry_path,
        structural_to_aero=structural_to_aero,
        n_panels_aero=n_panels_aero,
    )
    config = case.config
    case_folder = build_actuation_case_folder(config)
    results_root = aerostructural_results_root(case.project_dir, kite_name)
    results_dir = (
        Path(results_dir) if results_dir is not None else results_root / case_folder
    )
    results_dir = save_input_snapshot(
        config=config,
        results_dir=results_dir,
    )
    logging.info(f"config files saved in {results_dir}\n")

    prepared = prepare_structure(case)
    final_state = {}
    tracking_data, meta = solve_step(case, prepared, final_state=final_state)
    system_model = final_state["system_model"]
    reader = prepared.reader

    # Save results
    h5_path = save_sim_output(tracking_data, meta, results_dir)
    # The final coupled trim's converged circulation, as a plain .npy next to
    # the geometry snapshot: snapshot re-solves seed their gamma loop with it
    # (branch selection near stall) without having to open the h5.
    gamma_final = np.asarray(meta.get("gamma_distribution", []), dtype=float)
    if gamma_final.size:
        np.save(Path(results_dir) / "gamma_distribution.npy", gamma_final)
    final_nodes = np.asarray(tracking_data["positions"][meta["n_iter"] - 1])
    save_geometry_snapshot(
        config,
        build_deformed_struc_geometry(case.struc_geometry, final_nodes),
        _deformed_aero_geometry(
            case.aero_geometry, final_nodes, reader["le_indices"],
            reader["te_indices"], structural_to_aero,
        ),
        results_dir,
        system_yaml_path=case.system_config_path,
    )

    save_case_view(
        results_dir,
        build_case_view(
            backend="wireframe",
            kite_name=kite_name,
            config=config,
            system_config_path=case.system_config_path,
            struc_geometry=case.struc_geometry,
            nodes=final_nodes,
            masses=prepared.m_arr,
            connectivity=reader["kite_connectivity"],
            line_tension=prepared.psystem.tensions(),
            le_indices=reader["le_indices"],
            te_indices=reader["te_indices"],
            power_tape_index=reader["power_tape_index"],
            steering_tape_indices=reader["steering_tape_indices"],
            pulley_line_indices=reader["pulley_line_indices"],
            rest_lengths=meta.get("rest_lengths"),
            fixed=config.get("structural_pss", {}).get("fixed_point_indices", [0]),
        ),
    )

    return {
        "results_dir": results_dir,
        "results_root": results_root,
        "case_folder": case_folder,
        "config": config,
        "h5_path": h5_path,
        "tracking_data": tracking_data,
        "meta": meta,
        "final_nodes": final_nodes,
        "m_arr": prepared.m_arr,
        "struc_geometry": case.struc_geometry,
        "aero_geometry": case.aero_geometry,
        "kite_connectivity_arr": reader["kite_connectivity"],
        "l0_arr": reader["l0"],
        "k_arr": reader["k"],
        "struc_node_le_indices": reader["le_indices"],
        "struc_node_te_indices": reader["te_indices"],
        "power_tape_index": reader["power_tape_index"],
        "steering_tape_indices": reader["steering_tape_indices"],
        "system_model": system_model,
        "system_config_path": case.system_config_path,
    }


def solve_deformation_reduced_fem(
    config_overrides: dict | None = None,
    *,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    results_dir=None,
    system_config_path=None,
    fem_geometry_filename: str = "struc_geometry_FEM_full.yaml",
) -> dict:
    """One wireframe case on the wireframe REDUCED from the full FEM geometry.

    The same input file as the Billow full model
    (``billow.driver.solve_deformation``), so the two differ only in the
    structural model. The full geometry is read, its bridle relaxed onto the
    held wing exactly as Billow relaxes it
    (``structural_billow.relax_bridle_nodes``), and reduced to a PSM-format
    wireframe (``reduce_fem.reduce_fem_geometry``: one rib per strut, two wing
    nodes per rib, the per-strut line fans collapsed to one equivalent line,
    tube and canopy stiffnesses from Billow's materials). The reduction is
    written to ``<results_dir>/struc_geometry_wireframe.yaml`` and solved with
    :func:`solve_deformation`.

    ``config_overrides`` is merged key by key (nested dicts too) over the
    kite's ``as_config.yaml``; ``system_config_path`` defaults to the kite's
    ``system.yaml``.
    """
    from awetrim.aerostructural.billow.driver import merge_config
    from awetrim.aerostructural.billow.structural_billow import relax_bridle_nodes
    from awetrim.aerostructural.coupled import read_struc_geometry_yaml
    from awetrim.aerostructural.wireframe.reduce_fem import reduce_fem_geometry

    project_dir = Path(project_dir) if project_dir is not None else REPO_ROOT
    config_path, _, _ = resolve_kite_paths(project_dir, kite_name)
    system_config_path = Path(
        system_config_path or project_dir / "data" / kite_name / "system.yaml"
    )
    with system_config_path.open("r", encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)
    config = merge_config(load_yaml(config_path), config_overrides)

    fem_path = project_dir / "data" / kite_name / fem_geometry_filename
    fem = load_yaml(fem_path)
    reader = read_struc_geometry_yaml.main(
        load_yaml(fem_path), config=config, system_config=system_config
    )
    nodes, masses = reader[0], reader[1]
    relaxed = relax_bridle_nodes(
        config,
        fem,
        nodes,
        reader[10],  # kite_connectivity_arr
        reader[13],  # l0_arr
        reader[14],  # k_arr
        reader[16],  # linktype_arr
        reader[17],  # pulley_line_indices
        reader[7],  # canopy_sections
        reader[8],  # strut_sections
    )
    reduced = reduce_fem_geometry(fem, nodes=relaxed, masses=masses, config=config)

    # The aerodynamic mesh of the FULL geometry -- its true leading and
    # trailing edges, every section -- carried on the reduced ribs, so the
    # wireframe and Billow fly the same wing. Without it the wireframe would
    # fly the PSM convention: a wing whose edges ARE the bridle attachment
    # nodes (x/c 0.01 to 0.90, 11 % short of the chord, on 10 ribs).
    from awetrim.aerostructural.mapping import SkinStructuralToAeroMapper

    per_section = int(config["aerodynamic"]["n_aero_panels_per_struc_section"])
    n_ribs = len(reduced["wing_particles"]["data"]) // 2
    reduced_nodes = np.zeros((1 + n_ribs * 2 + len(reduced["bridle_particles"]["data"]), 3))
    reduced_nodes[0] = reduced["bridle_point_node"]
    for key in ("wing_particles", "bridle_particles"):
        for row in reduced[key]["data"]:
            reduced_nodes[int(row[0])] = row[1:4]
    reduced_nodes = rotate_geometry(
        reduced_nodes, **resolve_initial_geometry_rotation_kwargs(config)
    )
    fem_le, fem_te = np.asarray(reader[2], dtype=int), np.asarray(reader[3], dtype=int)
    full_nodes = rotate_geometry(
        np.asarray(relaxed, dtype=float), **resolve_initial_geometry_rotation_kwargs(config)
    )
    skin = SkinStructuralToAeroMapper().initialize(
        reduced_nodes,
        front_indices=[2 * r + 1 for r in range(n_ribs)],
        rear_indices=[2 * r + 2 for r in range(n_ribs)],
        le_points=full_nodes[fem_le],
        te_points=full_nodes[fem_te],
        n_panels_per_section=per_section,
    )
    n_panels_aero = (len(fem_le) - 1) * per_section

    results_dir = Path(
        results_dir
        or aerostructural_results_root(project_dir, kite_name)
        / "wireframe"
        / build_actuation_case_folder(config)
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    reduced_path = results_dir / "struc_geometry_wireframe.yaml"
    with reduced_path.open("w", encoding="utf-8") as handle:
        _yaml.safe_dump(reduced, handle, sort_keys=False)
    logging.info(
        "wireframe reduced from %s: %d wing + %d bridle nodes, wing %.3f kg, "
        "bridle %.3f kg -> %s",
        fem_path.name,
        len(reduced["wing_particles"]["data"]),
        len(reduced["bridle_particles"]["data"]),
        reduced["reduction"]["wing_mass"],
        reduced["reduction"]["bridle_mass"],
        reduced_path,
    )
    return solve_deformation(
        config=config,
        kite_name=kite_name,
        project_dir=project_dir,
        results_dir=results_dir,
        system_config_path=system_config_path,
        struc_geometry_path=reduced_path,
        structural_to_aero=skin,
        n_panels_aero=n_panels_aero,
    )
