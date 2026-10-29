# LEI-V3 structural geometry

`struc_geometry_FEM_full.yaml` is the kite's ONE structural geometry. Both
structural fidelities start from it:

- **Billow full** (`awetrim.aerostructural.billow.driver`) reads it as it is:
  inflatable strut and leading-edge tubes, the membrane canopy between them
  (28 sections), the inflation pressure, and the complete bridle down to the
  individual a/b/c/d lines that fan onto each strut.
- **Wireframe** (`awetrim.aerostructural.wireframe.driver`) reduces it on the
  fly (`wireframe.reduce_fem.reduce_fem_geometry`). The reduction has one rib
  per strut with two wing nodes, where the front and rear bridle attach, and
  each per-strut line fan collapsed to one equivalent line. It flies the full
  geometry's aerodynamic mesh. The reduction is written next to each case as
  `struc_geometry_wireframe.yaml`.

`awetrim.aerostructural.case.DEFAULT_STRUC_GEOMETRY_FILENAME` names the file.
The step-1 aerodynamics (bridle drag, mass and inertia) read it too.

A deformed results folder holds a snapshot of the solved shape as
`struc_geometry.yaml`. The snapshot is a PSM-format wireframe, and
`case.find_struc_geometry` picks it up there.

## Removed geometries

The PSM wireframe files were removed on 2026-10-29:

- `struc_geometry.yaml`
- `struc_geometry_PSM_reduced.yaml`
- `struc_geometry_PSM_reduced_photogrammetry_adjusted.yaml`

They are in git history and in the v0.1.0 release. They reproduce the
wes-quasi-steady paper, whose model they are: the wing nodes sit at the bridle
attachments (a short chord, 17.2 m2 projected), and the trailing-edge rest
lengths were cut by hand to match the photogrammetry. The current
identification replaces those hand corrections with physical ones (canopy
billowing, the full chord and the strut-lever load transfer). See
`docs/aerostructural/lei_v3_wing_corrections.md`.

The tape lengths that match the 2019 flight are in `identification.yaml`
(`fit.fidelities.wireframe.flight_match`).
