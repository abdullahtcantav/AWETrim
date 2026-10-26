# Step 3: ROM identification

Identify the reduced-order model (ROM) of the kite from aerostructural
simulations only, no flight data. The ROM is what makes trajectory
optimisation (step 4) fast: a quasi-steady kite whose aerodynamics are a few
fitted functions instead of a VSM solve.

| Script | What it does | Writes |
|--------|--------------|--------|
| [`identify_rom_aerostructural.py`](identify_rom_aerostructural.py) | Fits every ROM parameter on the step-2 sweep at the centre of the wind window: the bridle pitch `theta_b(u_p)` from the coupled trims, the lift and drag polynomials in `(alpha, u_p, u_s, v_a)` from frozen-shape alpha polars (terms chosen on cross-validation), and the steering roll gain. | `data/<kite>/rom_config_aerostructural.yaml`, plus the tidy datasets under `results/<kite>/identification/rom_aerostructural/` |

```bash
python examples/3_rom_identification/identify_rom_aerostructural.py --results-root <sweep folder>
```

`--no-write` fits and reports without writing the ROM file.

What goes in and comes out:

- **Input**: per operating point, the coupled trim (`point.json`) and a
  frozen-shape alpha sweep about it (`alpha_polar.json`, optionally
  `alpha_polar_stall.json` past the stall), as written by the step-2 sweep.
- **Output**: a ROM file in the format `awetrim.system.kite` reads. Select it
  in the kite's `system.yaml` (`models.reduced_order.aerodynamics`) to fly
  it in step 4.
- The drag coefficient is the wing's, bridle-line drag included. The KCU drag
  is not in it: the ROM adds it from the system file's KCU hardware
  (`awetrim.aerodynamics.kcu_drag`), so a KCU change needs no
  re-identification.

With flight data, [step 5](../5_flight_data/) corrects this ROM's depower
pitch and drag on a flown cycle.
