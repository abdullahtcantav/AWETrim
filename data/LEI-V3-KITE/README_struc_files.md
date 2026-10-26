# Notes to URI

### Struc Geometry file
- **Default for PSM simulations:** `struc_geometry_PSM_reduced_photogrammetry_adjusted.yaml`. The PSM/PSS aerostructural scripts (`run_simulation_level_qsm.py` and the sweeps, via `resolve_kite_paths` / `DEFAULT_STRUC_GEOMETRY_FILENAME` in `examples/aerostructural/common.py`) use this file by default. Don't bother with the non-adjusted version.
- for a semi-complete account of how this was obtained, read the parent directory.

### 2019 powered and depowered state
- 2019 powered and depowered state estimations are VERY APPROXIMATE and most definitely wrong    -> 

- From EKF-AWE obtain: 22deg powered and 26deg power signal

Convert using the same procedure as described below, but adjust for min_signal = 21.4deg and max_signal = 26.92deg
```
The 2019 value of $u_{\mathrm{dp}}$ was obtained by transforming the reported depower signal in degrees\footnote{\label{fn:udp_2019} The 2019 dataset reported a depower signal $x^{2019}$ in degrees~\citep{Oehler2019}, which was normalised using the campaign bounds $x_{\min}=0.02~\unit{\degree}$ and $x_{\max}=22.68~\unit{\degree}$ to obtain $u_{\mathrm{p}}^{2019}$, and then converted to the 2025 convention by equating physical tape lengths. The 2019 kinematic depower-tape relation $l_{\mathrm{dp}}=1.098+0.384\left(1-u^{2019}_{\mathrm{p}}\right)$~\citep{Poland2023} was combined with the 2025 linear calibration $l_{\mathrm{dp}}=0.2+5u_{\mathrm{dp}}$ to yield $u_{\mathrm{dp}}=0.2564-0.0768u_{\mathrm{p}}^{2019}$. This conversion assumes that the underlying tape-length-to-depower mapping remained consistent between campaigns.} 
into the 2025 representation $u_{\mathrm{dp}}$. 
This conversion makes it possible to express a $u_{\mathrm{dp}}$ number in the 2025 linear calibration mapping, but should not be considered absolute as it depends on the assumed tape-length relation and on the unknown 2019 bridle-line geometry.
```

We then find:
- ldp_2019_power = 1.14
- ldp_depower = 1.42

Which converted to udp 2025 non-linear relation:
```
\begin{equation}
    l_{\mathrm{dp},2025}
    =
    -1.724u_{\mathrm{dp}}^2
    +6.624u_{\mathrm{dp}}
    +0.192 .
    \label{eq:dp_relation_2025}
\end{equation}
```
gives:
udp_power = 0.146
udp_depower = 0.190

- dynamic aero-structural coupled steady circular flight simulations suggest that Gk will match with udp_powered ~ 0.23/24

- **Default powered (2019): $u_{\mathrm{dp}} = 0.24$, $l_{\mathrm{dp}} = 1.7$**
- **Default depowered (2019): $u_{\mathrm{dp}} = 0.284$, $l_{\mathrm{dp}} = 1.98$**


## 2025 powered values

- Torque2026 paper is outdated, reasoning from dissertation simulation insights we find
- quasi-steady aero-structural coupled symmetric simulations give a force-match using -> udp_powered ~ 0.23/0.24 
- quasi-steady aero-structural coupled steady circular flight simulations suggest that Gk will match with udp_powered ~ 0.21/0.22
- dynamic aero-structural coupled steady circular flight simulations suggest that Gk will match with udp_powered ~ 0.21

- **Default (2025): $u_{\mathrm{dp}} = 0.22$**
