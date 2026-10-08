# Spiral galaxies in SONGS

`SONGS(..., morphology='spiral')` and `SONGSPhy(..., morphology='spiral')`
turn the central galaxy of each cube into a tightly wound grand-design
spiral (at most 3 arms) with a bulge and a bright centre. The arms are not
drawn by hand. Their shape, extent, cross-section, star-forming complexes,
spurs and kinematics all follow from density-wave theory applied to an
explicit mass model. The physics lives in
[`src/songs/spiral.py`](../src/songs/spiral.py).

```python
from songs.core import SONGS

g = SONGS(n_cubes=4, morphology='spiral', seed=1)              # everything sampled
g = SONGS(n_cubes=1, morphology='spiral',
          spiral_params={'m': 3, 'm2': None, 'c_s': 9.0})       # pin some parameters
cube, params = g.generate_cubes()[0]
params['spiral']        # pitch, pattern speed, resonances, streaming, bulge, ...
```

## The model, step by step

| Step | Physics | Reference |
|---|---|---|
| Mass model | Exponential (Freeman) disk + Hernquist bulge + pseudo-isothermal halo. B/T = 0.05–0.3 and R_e,bulge = 0.12–0.3 R_d, as measured by GALFIT/IMFIT decompositions of Sab–Sc spirals | Freeman 1970; Gadotti 2009; Salo+ 2015 |
| Stability | σ_R ∝ e^(−R/2R_d), normalised to a minimum Toomre Q of 1.0–1.3 | van der Kruit & Freeman 2011 |
| Arm shape | Short-wave trailing branch of the WKB dispersion relation (ω−mΩ)² = κ² − 2πGΣ\|k\| + σ²k²; pitch tan p = m/kR (7–15°, about one turn) | Lin & Shu 1964; BT08 §6.2 |
| Arm extent | Arms start at the ILR created by the bulge, decay through the Q-barrier forbidden zone around corotation, and are absorbed at the OLR | Contopoulos & Grosbøl 1986 |
| Gas shocks | Gas crosses the arms at u₀ = m(Ω−Ω_p)/k. Supersonic flow shocks into narrow, skewed arms (steep upstream, slow post-shock decline); near corotation the arms are weak and broad | Roberts 1969 |
| Streaming | Mass and angular-momentum conservation across a steady arm: v_R = u₀(1/ρ − 1), v_φ = κ²/(2Ωk)∫(ρ−1)dχ. Reduces to linear Lin–Shu for weak arms; 20–70 km/s in practice | Shu, Milione & Roberts 1973 |
| Arm segments | Arm strength modulated along each arm by a smooth lognormal field in (ln R, φ) (rms of ln η 0.4–0.8, ~3 kpc segments, capped at 1.5× the shocked contrast); streaming and star formation follow the segments | Honig & Reid 2015 |
| Second mode | An optional weaker mode with its own m and Ω_p gives branches and lopsidedness | e.g. M51 (m=2+3) |
| Star formation | Hierarchical complexes spaced by the Jeans length λ = 2c_s²/(GΣ_arm), ~0.1 kpc downstream of the shock. dN/dL ∝ L⁻², size ∝ L^⅓ | Elmegreen & Elmegreen 1983; Elmegreen & Efremov 1996; Kennicutt+ 1989; Egusa+ 2009 |
| Spurs | Gas feathers (30–80% density enhancement, 100–300 pc wide) sheared out of the complexes at tan p = 1/(ΓΩΔt) | Balbus 1988; Kim & Ostriker 2002; La Vigne+ 2006 |
| Turbulent ISM | Lognormal density field, P(k) ∝ k^−2.7, steepening to k^−3.7 below the 0.25 kpc disk thickness; fed on ~3 kpc scales | Elmegreen, Kim & Staveley-Smith 2001 |
| Turbulent motions | Correlated isotropic velocity field, P(k) ∝ k^−3, rms 0.5–0.7 c_s (5–9 km/s of non-circular motion); the per-voxel dispersion keeps the rest so the total stays c_s | Stanimirović & Lazarian 2001; Trachternach+ 2008 |
| Centre | Gas inflow builds a central concentration (5–25% of the flux) and a ring at the ILR; the gas dispersion rises to 20–50 km/s in the centre | Buta & Combes 1996; Regan+ 2001; Sun+ 2020 |
| Emission | ∝ Σ_gas^N (N = 1 for CO/HI, ~1.4 for Hα). Arms, turbulence and clumps redistribute the disk's flux around each annulus | Kennicutt 1998 |

Arm contrasts and the gas dispersion (c_s = 8–15 km/s, used as the
line-of-sight dispersion) are calibrated on the ALMA CO(2–1) cube of NGC
7469.

Only the central galaxy becomes a spiral. Satellites and the diffuse
components are unchanged, and `morphology='disk'` (the default) draws the
same random numbers as before.

## Rotation sense and arm handedness

The arms always **trail** the rotation: the WKB solution is the trailing
(k > 0) branch in the rotating frame, and the streaming motions, the shock
side of the arm profile and the spurs all take their orientation from the
same rotation. SONGS disks spin clockwise in their own (x, y) plane, so the
arms wind counter-clockwise outward there (`test_arms_are_trailing`).

On the sky this also needs the velocity vectors to be rotated by exactly
the rotation that moves the voxels. The original `rotated_system` projected
velocities with a `Rotation.from_euler('yxz', ...)` matrix that did not
match the two `ndimage.rotate` calls: the inclination had the opposite sign
(spin reversed, so arms would *lead*), and with both angles non-zero the
projected velocity field was essentially unrelated to the projected
geometry (correlation −0.2 with the true field). Both generators now use
the matrix of the image rotation itself (`_image_rotation_matrix`), and
`test_projected_velocities_follow_the_projected_geometry` checks that every
sky voxel's v_LOS equals its galaxy-frame velocity rotated the same way
(correlation +0.999). This fix also changes the line-of-sight velocities of
plain disks: they now spin consistently with their orientation, and their
kinematic and photometric major axes agree.

## Saved systems (3D numpy arrays)

Each system is in `systems/<name>/`: the fiducial spiral, its axisymmetric
twin (same mass model, centre and ring, no arms), and 8 gallery spirals
sampled exactly as `SONGS(morphology='spiral')` samples them.

| File | Contents | Shape |
|---|---|---|
| `cube.npy` | beam-convolved spectral cube | (n_v, n_y, n_x) |
| `velocities_km_s.npy` | channel centres | (n_v,) |
| `emissivity_galaxy_frame.npy` | intrinsic 3D emission before rotation | (z, y, x) |
| `emissivity_sky.npy` | 3D emission after rotation (z = line of sight) | (z, y, x) |
| `vlos_sky.npy` | line-of-sight velocity of every voxel (km/s) | (z, y, x) |
| `params.json` | all spiral parameters, derived quantities, grid metadata | |

```python
import numpy as np, json
cube = np.load('spiral_galaxies/systems/fiducial_spiral/cube.npy')
meta = json.load(open('spiral_galaxies/systems/fiducial_spiral/params.json'))
```

## Figures

Regenerate everything with `python spiral_galaxies/make_spiral_figures.py`
(about 10 minutes; `--skip-gallery` skips the gallery systems and figures
22–23). Figures 24–25 need the NGC 7469 cube from the IFU-Source-ID
repository (`--ngc7469 PATH`).

| # | File | What it shows |
|---|---|---|
| 01 | `01_mass_model.png` | Rotation curve (disk, bulge, halo), Σ(R), Ω and κ |
| 02 | `02_lindblad_resonances.png` | Lindblad diagram, pattern speeds, ILR / 4:1 / CR / OLR |
| 03 | `03_stability_Q_barrier.png` | σ(R), Toomre Q(R), propagation criterion and forbidden zones |
| 04 | `04_dispersion_relation.png` | Dimensionless dispersion relation; k(R) branches and Im k |
| 05 | `05_pitch_angle_winding.png` | Pitch angle p(R), arm turns, shear rate, radial arm spacing |
| 06 | `06_envelope_and_shocks.png` | Amplitude envelope, Q-barrier transmission, Mach number, arm contrast |
| 07 | `07_arm_geometry.png` | Arm crests with resonance circles; φ vs ln R |
| 08 | `08_arm_profile_streaming.png` | Skewed shock profile and the nonlinear vs linear streaming |
| 09 | `09_knots_and_spurs.png` | Complexes and spurs, luminosity function, Jeans spacing, spur pitch, zoom |
| 10 | `10_turbulence.png` | Turbulent field, lognormal PDF, broken power spectrum |
| 11 | `11_bulge_and_center.png` | Inner rotation curve, emission profile by component, centre, σ(R) |
| 12 | `12_cartesian_grid_3d.png` | The 3D Cartesian voxel grid and the emission on it; edge-on cut |
| 13 | `13_emissivity_slices.png` | Midplane, off-plane and vertical slices; vertical profiles |
| 14 | `14_face_on_buildup.png` | Face-on emission built up one physical component at a time |
| 15 | `15_spiral_vs_twin.png` | Spiral vs axisymmetric twin, and where the flux moved |
| 16 | `16_velocity_field_face_on.png` | v_R and v_φ − v_c maps and streaming vectors |
| 17 | `17_rotation_to_observer_frame.png` | 3D render before and after rotation to the sky |
| 18 | `18_los_velocity_residual.png` | LOS velocity field, kinked isovelocity contours, residual |
| 19 | `19_channel_maps.png` | Channel maps of the final cube |
| 20 | `20_moment_maps_spectrum.png` | Moments 0/1/2 and the integrated spectrum vs the twin |
| 21 | `21_pv_diagram.png` | Major-axis PV diagram, spiral vs twin |
| 22 | `22_gallery_sampled_spirals.png` | 8 sampled spirals: face-on emission and inclined-cube moments |
| 23 | `23_parameter_distributions.png` | Distributions of m, pitch, turns, ILR, streaming, complexes |
| 24 | `24_ngc7469_vs_songs_moments.png` | NGC 7469 vs SONGS: moments and spectra |
| 25 | `25_ngc7469_vs_songs_channels.png` | NGC 7469 vs SONGS: channel maps, on-arm vs interarm spectra |

## Limitations

- The arm profile is prescribed (a skewed von Mises shape with a
  Mach-dependent contrast), not obtained by solving the Roberts shock
  equations.
- No bar. A cuspy bulge gives a single ILR, so the ring is the inner ring
  at the arm roots rather than a bar-driven nuclear ring like NGC 7469's.
- The fluid dispersion relation is used for the stellar disk, so the
  stellar reduction factor and disk-thickness corrections are neglected.
- The turbulent field and complexes are 2D; they are extruded along z with
  the disk's vertical profile.
- The velocity dispersion is one random draw per voxel, as for SONGS disks.
