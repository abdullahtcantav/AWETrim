"""Solve the aerostructural dataset the ROM is identified on (step 3, part 1).

Coupled VSM + Billow states at the centre of the wind window, gravity off,
along continuation chains, at either structural fidelity (`fidelity` in
identification.yaml, or --fidelity): `billow`, the full model (tube beams,
membrane canopy, bridle), or `wireframe`, the bridle line system on the PSM
wireframe geometry (the model the shipped ROM was identified on):

* depower chains: from the tape length the geometry is built at, the power
  tape walks to each target length, one converged state per step;
* steering chains: the tape is first walked to a set length, then the
  steering half-difference grows, one converged state per step.

Each step warm-starts from the previous one on the SAME structure (the
drivers' solve_step, awetrim.aerostructural.billow.driver and
awetrim.aerostructural.wireframe.driver), so a chain costs about one tape
walk. On every converged state (an anchor) a frozen-shape angle-of-attack
polar is solved (aerodynamic_vsm.run_frozen_alpha_polar_about_trim): the
trims alone tie alpha to the depower almost one to one, the polars separate
them. Everything is read from data/<kite>/identification.yaml (`sweep:`).

Output, results/<kite>/identification/<fidelity>/sweep/ (format in
awetrim.identification.aerostructural_dataset):
    trims.csv     one row per anchor (the coupled trim)
    polars.csv    one row per polar sample
    <chain>/step_<n>/sim_output.h5   the coupled solve of each anchor

Resumable: a finished chain is skipped. An unfinished wireframe chain goes
on after its last complete anchor, from the structure that anchor's
sim_output.h5 saved; an unfinished Billow chain is solved again from its
start (its beam frames are not saved). A state that does not converge ends
its chain. About a minute per anchor.

Usage (project root):
    python examples/3_rom_identification/run_identification_sweep.py
    python examples/3_rom_identification/run_identification_sweep.py --fidelity wireframe
    python examples/3_rom_identification/run_identification_sweep.py --chains w4.0_depower_2.60
    python examples/3_rom_identification/run_identification_sweep.py --list
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow.driver import merge_config
from awetrim.aerostructural.case import resolve_kite_paths
from awetrim.aerostructural.results import save_sim_output
from awetrim.aerostructural.utils import load_yaml
from awetrim.identification import aerostructural_dataset as ds

PROJECT_DIR = Path(__file__).resolve().parents[2]
DONE_MARKER = "chain_done"


@dataclass
class Step:
    """One converged state of a chain: where the tapes go before it is solved."""

    power_tape_length: float | None  # None: leave the tape where it is
    steering_half_difference: float


@dataclass
class Chain:
    name: str
    wind_speed: float
    steps: list[Step]


def _walk(start: float, target: float, step: float) -> list[float]:
    """Tape lengths from ``start`` (excluded) to ``target`` (included)."""
    n = int(np.ceil(abs(target - start) / step - 1e-9))
    return [float(v) for v in np.linspace(start, target, n + 1)[1:]] if n else []


def build_chains(spec: dict, built_tape_length: float) -> list[Chain]:
    """The chains of ``identification.yaml``'s ``sweep`` block."""
    depower, steering = spec["depower"], spec.get("steering") or {}
    chains = []
    for wind in spec["wind_speeds"]:
        # The built state itself opens the first depower chain only.
        for i, target in enumerate(depower["targets"]):
            lengths = _walk(built_tape_length, target, depower["step"])
            steps = ([Step(None, 0.0)] if i == 0 else []) + [Step(l, 0.0) for l in lengths]
            chains.append(Chain(f"w{wind:.1f}_depower_{target:.2f}", wind, steps))
        for length in steering.get("power_tape_lengths", []):
            halves = _walk(0.0, steering["max"], steering["step"])
            steps = [Step(length, h) for h in halves]
            chains.append(Chain(f"w{wind:.1f}_steering_{length:.2f}", wind, steps))
    return chains


def open_case(fidelity: str, spec: dict, kite: str):
    """``(driver, case)``: the fidelity's driver module and its built case."""
    kite_dir = PROJECT_DIR / "data" / kite
    system_path = kite_dir / spec.get("system_file", "system.yaml")
    if fidelity == "billow":
        from awetrim.aerostructural.billow import driver

        case = driver.build_case(
            spec.get("config_overrides"), kite_name=kite, project_dir=PROJECT_DIR,
            system_config_path=system_path,
        )
    else:
        from awetrim.aerostructural.wireframe import driver

        config_path, _, _ = resolve_kite_paths(PROJECT_DIR, kite)
        case = driver.build_case(
            merge_config(load_yaml(config_path), spec.get("config_overrides")),
            kite_name=kite, project_dir=PROJECT_DIR, system_config_path=system_path,
            struc_geometry_path=(
                kite_dir / spec["struc_geometry"] if spec.get("struc_geometry") else None
            ),
        )
    return driver, case


def prepare(driver, case, spec: dict, config: dict):
    """A fresh structure; the wireframe with its bridle modulus pinned."""
    prepared = driver.prepare_structure(case, config)
    if spec.get("bridle_modulus_factor"):
        prepared.pin_bridle_modulus(float(spec["bridle_modulus_factor"]))
    return prepared


def solve_chain(driver, case, chain: Chain, spec: dict, out: Path, start: int = 0) -> int:
    """Solve one chain from step ``start``, appending its anchors to the
    sweep files. A chain resumed past its first step starts from the
    structure saved with step ``start - 1``. Returns the chain's number of
    anchors."""
    config = copy.deepcopy(case.config)
    config["wind_speed_wind_ref"] = float(chain.wind_speed)
    prepared = prepare(driver, case, spec, config)
    if start:
        prepared.restore(out / chain.name / f"step_{start - 1:02d}" / "sim_output.h5")
    lo, hi, inc = spec["polar_d_alpha_deg"]
    d_alpha = np.round(np.arange(lo, hi + 0.5 * inc, inc), 9)
    m_arr = np.asarray(prepared.m_arr, dtype=float)
    n_written = start
    for index, step in enumerate(chain.steps[start:], start=start):
        started = time.perf_counter()
        # The depower walk comes first and the steering after it, both from
        # the structure's current state (coupled_solver.main's order).
        tape_reach = (
            0.0 if step.power_tape_length is None
            else step.power_tape_length - prepared.power_tape_length
        )
        config["power_tape_final_extension"] = float(tape_reach)
        config["power_tape_extension_step"] = float(spec["depower"]["step"])
        config["steering_tape_final_extension"] = float(
            step.steering_half_difference - prepared.steering_half_difference
        )
        config["steering_tape_extension_step"] = float(spec["steering"]["step"])
        final_state = {}
        tracking, meta = driver.solve_step(case, prepared, config, final_state=final_state)
        step_dir = out / chain.name / f"step_{index:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        save_sim_output(tracking, meta, step_dir)

        trim, body = final_state["results_aero"], final_state["body_aero"]
        # An unconverged state ends the chain before its polar: the fit drops
        # it anyway, and a diverged (NaN) shape has no polar to take.
        if not meta["converged"] or not np.all(
            np.isfinite(np.asarray(trim["total_aero_force_vec"], dtype=float))
        ):
            print(
                f"  {chain.name}/{index:02d}: NOT CONVERGED, chain stopped "
                f"| {(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )
            break
        polar = aerodynamic_vsm.run_frozen_alpha_polar_about_trim(
            body, prepared.vsm_solver, d_alpha
        )
        f_trim = np.asarray(trim["total_aero_force_vec"], dtype=float)
        f_zero = next(s["force_world"] for s in polar if s["d_alpha_deg"] == 0.0)
        anchor = f"{chain.name}/{index:02d}"
        ds.append_rows(out / ds.TRIMS_FILE, [ds.trim_row(
            anchor=anchor, chain=chain.name, step=index,
            wind_speed=chain.wind_speed,
            power_tape_length=prepared.power_tape_length,
            steering_half_difference=prepared.steering_half_difference,
            rho=float(config["rho"]), mass_kcu=float(m_arr[0]),
            mass_total=float(m_arr.sum()), trim=trim, meta=meta,
            body_z=aerodynamic_vsm.body_centre_panel_axes(body)[2],
            polar_check=float(np.linalg.norm(f_zero - f_trim) / np.linalg.norm(f_trim)),
        )])
        ds.append_rows(out / ds.POLARS_FILE, ds.polar_rows(anchor, polar))
        n_written += 1

        stalled = float(trim.get("stalled_fraction", 0.0) or 0.0)
        print(
            f"  {anchor}: l_dp {prepared.power_tape_length:.3f} m, steer "
            f"{prepared.steering_half_difference:+.3f} m | v_a {np.linalg.norm(trim['va_vel_world']):5.2f} "
            f"m/s, alpha {trim['aoa_deg']:5.2f} deg, F_t {trim['tether_force']:6.0f} N, "
            f"chi_dot {np.ravel(trim['opt_x'])[4]:+.3f} rad/s, stalled {stalled:.2f} | "
            f"{meta['n_iter']} it, {'converged' if meta['converged'] else 'NOT CONVERGED'}"
            f" | {(time.perf_counter() - started) / 60:.1f} min",
            flush=True,
        )
        if stalled > spec["max_stalled_fraction"]:
            print(f"  {chain.name}: stopped (stalled)", flush=True)
            break
    (out / chain.name).mkdir(parents=True, exist_ok=True)
    (out / chain.name / DONE_MARKER).write_text(f"{n_written} anchors\n", encoding="utf-8")
    return n_written


def _read_rows(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=["nan", "NaN"])


def _step_of(anchors: pd.Series) -> pd.Series:
    return anchors.str.rsplit("/", n=1).str[-1].astype(int)


def resume_point(out: Path, chain_name: str) -> int:
    """The steps of an unfinished chain already complete -- trim row and
    polar rows written -- in an unbroken run from step 0."""
    trims, polars = _read_rows(out / ds.TRIMS_FILE), _read_rows(out / ds.POLARS_FILE)
    if trims is None or polars is None:
        return 0
    prefix = chain_name + "/"
    with_trim = set(_step_of(trims.anchor[trims.anchor.str.startswith(prefix)]))
    with_polar = set(_step_of(polars.anchor[polars.anchor.str.startswith(prefix)]))
    n = 0
    while n in with_trim and n in with_polar:
        n += 1
    return n


def _drop_chain_rows(out: Path, chain_name: str, keep_steps: int = 0) -> None:
    """Remove an unfinished chain's rows from step ``keep_steps`` on, so its
    rerun replaces them."""
    for name in (ds.TRIMS_FILE, ds.POLARS_FILE):
        path = out / name
        table = _read_rows(path)
        if table is not None:
            ours = table["anchor"].str.startswith(chain_name + "/")
            drop = ours & (_step_of(table["anchor"].where(ours, "x/0")) >= keep_steps)
            if drop.any():
                table[~drop].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kite", default="LEI-V3-KITE", help="folder under data/")
    parser.add_argument("--fidelity", choices=ds.FIDELITIES, default=None,
                        help="structural model (default: identification.yaml's)")
    parser.add_argument("--chains", nargs="+", help="solve only these chains")
    parser.add_argument("--list", action="store_true", help="list the chains and exit")
    args = parser.parse_args()

    fidelity, spec = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "sweep", args.fidelity
    )
    out = PROJECT_DIR / "results" / args.kite / "identification" / fidelity / "sweep"
    driver, case = open_case(fidelity, spec, args.kite)
    built = prepare(driver, case, spec, case.config).power_tape_length
    chains = build_chains(spec, built)
    if args.list:
        for chain in chains:
            print(f"{chain.name}: {len(chain.steps)} states, wind {chain.wind_speed} m/s")
        return
    if args.chains:
        unknown = set(args.chains) - {c.name for c in chains}
        if unknown:
            raise SystemExit(f"unknown chains {sorted(unknown)}; see --list")
        chains = [c for c in chains if c.name in args.chains]

    print(f"{args.kite} ({fidelity}): built power-tape length {built:.3f} m, {len(chains)} chains "
          f"-> {out}", flush=True)
    for chain in chains:
        if (out / chain.name / DONE_MARKER).exists():
            print(f"{chain.name}: done, skipped", flush=True)
            continue
        # The wireframe resumes after its last complete anchor; Billow's
        # beam frames are not in sim_output.h5, so its chain starts over.
        start = resume_point(out, chain.name) if fidelity == "wireframe" else 0
        _drop_chain_rows(out, chain.name, keep_steps=start)
        resumed = f", resumed at step {start}" if start else ""
        print(f"{chain.name}: {len(chain.steps)} states{resumed}", flush=True)
        solve_chain(driver, case, chain, spec, out, start=start)


if __name__ == "__main__":
    main()
