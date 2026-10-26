# AWETrim examples

A short tour of what AWETrim does, one runnable example per capability. Run
every example **from the project root**. They all use the TU Delft LEI-V3
kite in `data/LEI-V3-KITE/`.

| Capability | Example | What it shows |
|------------|---------|---------------|
| Aerodynamic trim | [`aerodynamics/solve_single_state.py`](aerodynamics/solve_single_state.py) | One quasi-steady VSM trim state of the kite and tether |
| Static stability | [`aerodynamics/compute_stability_derivatives.py`](aerodynamics/compute_stability_derivatives.py) | Stability derivatives about a trim and the six-channel static-stability verdict |
| Aerostructural, wireframe | [`aerostructural/run_simulation_PSM.py`](aerostructural/run_simulation_PSM.py) | Coupled VSM and structure, converged deformed shape (under a minute) |
| Aerostructural, full Billow | [`aerostructural/run_simulation_BILLOW.py`](aerostructural/run_simulation_BILLOW.py) | The same with inflatable tube beams and a wrinkling canopy (about a minute) |
| Result viewer | [`aerostructural/plot_simulation.py`](aerostructural/plot_simulation.py) | Interactive 3-D page of a converged case with its trim table |
| Reel-out patterns | [`reduced-order-model/optimization/reelout/`](reduced-order-model/optimization/reelout/) | Down-loop and helix reel-out loops simulated with the reduced-order model |
| Full pumping cycle | [`reduced-order-model/optimization/cycle/run_full_cycle_opti.py`](reduced-order-model/optimization/cycle/run_full_cycle_opti.py) | Optimise a whole pumping cycle for mean power |
| Flight validation | [`reduced-order-model/validation/validate_quasi_steady_state_v3.py`](reduced-order-model/validation/validate_quasi_steady_state_v3.py) | Reduced-order model against the 2019 LEI-V3 flight |
| REST server | [`server/`](server/) | The reel-out optimiser as a web service, with Python, Julia and MATLAB clients |

Quick start:

```bash
python examples/aerodynamics/solve_single_state.py
python examples/aerostructural/run_simulation_PSM.py
python examples/reduced-order-model/optimization/cycle/fit_periodic_cycle_config.py --kite LEI-V3-KITE --auto
python examples/reduced-order-model/optimization/cycle/run_full_cycle_opti.py --kite LEI-V3-KITE
```

Each folder has its own README with the options and outputs.

## Flight data

The validation examples compare against the 2019-10-08 LEI-V3 flight as
reconstructed by the extended Kalman filter in
[EKF-AWE](https://github.com/awegroup/EKF-AWE), which ships the raw flight
log. Run EKF-AWE on that log and place its output at
`results/LEI-V3-KITE/ekf/LEI-V3 Kite_2019-10-08.h5`.

## Conventions

- Results go to `results/<kite>/<analysis>/` (aerodynamics examples:
  `results/aerodynamics/<script>/`, override with `--output-dir`).
- Everything about a kite comes from the YAML files in `data/<kite>/`
  (`system.yaml`, `aero_geometry.yaml`, `struc_geometry*.yaml`,
  `as_config.yaml`, `cycle_profile.yaml`, `cycle_configs/`).

The research studies behind the papers and notes in `docs/` (parameter
identification, Billow studies, solver comparisons) are not part of the
examples; their scripts are available on request.
