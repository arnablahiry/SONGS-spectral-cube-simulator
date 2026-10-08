"""Physically motivated spiral structure for SONGS disk galaxies.

This module turns the axisymmetric SONGS disk into a grand-design spiral
whose arm *shape*, *extent*, *profile*, *clumps* and *kinematics* all follow
from density-wave theory applied to an explicit mass model, rather than
from hand-drawn logarithmic arms. Every quantity is derived from a handful
of physical parameters (disk and bulge masses, halo core, Toomre Q, pattern
speed, number of arms, gas sound speed):

1. **Mass model.** An exponential (Freeman 1970) stellar disk, a Hernquist
   (1990) bulge and a pseudo-isothermal dark halo. The disk mass is fixed by
   the disk's share of the circular speed at 2.2 R_d (``disc_fraction``;
   ~0.85 is a "maximal" disk, Sackett 1997; ~0.6 is the DiskMass value,
   Bershady et al. 2011), the bulge by its bulge-to-total mass ratio and
   effective radius (``bulge_fraction``, ``bulge_size``; calibrated on
   GALFIT/IMFIT bulge-disk decompositions, B/T ≈ 0.05-0.3 and
   R_e,bulge ≈ 0.1-0.3 R_d for Sab-Sc spirals, Gadotti 2009; Salo et al.
   2015), and the halo is normalised so that v_c(2.2 R_d) = ``v_0``. This
   gives v_c(R), Ω(R) = v_c/R, κ²(R) = 2Ω(Ω + dv_c/dR) and Σ(R). The bulge
   makes the rotation curve rise steeply, which raises Ω - κ/2 in the
   centre and creates an inner Lindblad resonance.

2. **Stability.** The radial velocity dispersion follows the
   constant-scale-height disk expectation σ(R) = σ_0 exp(-R / 2R_d)
   (van der Kruit & Freeman 2011), normalised so that the minimum of the
   fluid Toomre parameter Q = σκ / (πGΣ) equals ``Q_min``.

3. **Arm shape.** Arms are the trailing, short-wave branch of the WKB
   (tight-winding) dispersion relation of a fluid disk,

   ``(ω - mΩ)² = κ² - 2πGΣ|k| + σ²k²``  (Lin & Shu 1964; BT08 eq. 6.55)

   with ω = mΩ_p. Writing ν = m(Ω_p - Ω)/κ, the radial wavenumber is
   ``|k| = (πGΣ/σ²)[1 + sqrt(1 - Q²(1 - ν²))]``. The arm phase is
   χ = m(φ - φ_0) + f(R) with f(R) = ∫ k dR, and the pitch angle is
   tan p = m / (kR). Pitch angles therefore vary with radius and are set by
   Σ, κ, σ and Ω_p: heavier disks give more open arms, hotter disks tighter
   ones.

4. **Arm extent.** Waves propagate where Q²(1 - ν²) ≤ 1. Around corotation
   this fails (the "Q-barrier" forbidden zone). There the wavenumber becomes
   complex and the amplitude decays as exp(-∫ Im k dR), so arms fade out
   before corotation, close to the inner 4:1 resonance, as observed
   (Contopoulos & Grosbøl 1986). Arms start at the outermost of the inner
   Lindblad resonance, the inner Q-barrier and the radius inside which the
   tight-winding approximation itself fails (pitch > ``wkb_max_pitch_deg``),
   and are absorbed at the outer Lindblad resonance.

5. **Shocked gas arms.** Gas flows through the arms at the normal speed
   u₀ = m(Ω - Ω_p)/k. Where u₀ exceeds the gas sound speed c_s it shocks
   into narrow, high-contrast arms (Roberts 1969); near corotation the flow
   is subsonic and the response stays weak and broad. The arm cross-section
   is a skewed von Mises profile,

   ``ρ(χ) ∝ exp(η cos ψ)``,  ψ = χ + δ(1 - cos χ),  normalised to mean 1,

   whose crest stays at χ = 0 while the monotone phase map ψ(χ)
   (dψ/dχ = 1 + δ sin χ) compresses one flank and stretches the other: a
   steep (shock) side facing upstream and a slow post-shock decline
   downstream when δ = -``skew`` · sign(Ω - Ω_p). Its peak-to-trough contrast e^{2η} rises
   from a quarter of the shocked contrast ``arm_contrast`` (subsonic, linear
   response) to the full value (supersonic), times the propagation
   envelope.

6. **Streaming motions.** For a steady, tightly wound flow, mass
   conservation across the arm gives the normal velocity u = u₀ / ρ(χ),
   and the azimuthal equation of motion gives the velocity along the arm,
   v = κ²/(2Ωk) ∫₀^χ (ρ - 1) dχ' (Roberts 1969; Shu, Milione & Roberts
   1973). In tight winding the normal and parallel directions are the
   radial and azimuthal ones, so

   ``v_R = u₀ (1/ρ - 1)``,  ``v_φ = κ²/(2Ωk) ∫₀^χ (ρ - 1) dχ'``.

   For weak arms (ρ = 1 + A cos χ) these reduce to the linear Lin–Shu
   results v_R = m(Ω_p - Ω)/k · A cos χ and v_φ = κ²/(2Ωk) · A sin χ.

7. **Arm segments.** Real arms are chains of segments a few kpc long that
   brighten, fade and occasionally break (e.g. Honig & Reid 2015), from
   transient swing-amplified pieces, mode interference and the local gas
   supply. The local arm strength η is multiplied by a smooth, mean-one
   lognormal field in (ln R, φ) with rms ``arm_segmentation`` and a
   correlation length of ~0.5 rad (~3 kpc along an arm at 6 kpc). Because
   it depends on φ rather than on the arm phase, each arm gets its own
   segments, and the streaming motions (computed from the local η) become
   patchy too.

8. **Secondary mode.** Real disks host several coexisting modes (e.g. the
   m = 2 + m = 3 superposition in M51). A weaker second mode with its own m
   and pattern speed interferes with the main one, producing arm branches
   and lopsidedness. Densities combine multiplicatively and streaming
   velocities add.

9. **Star-forming knots and spurs.** Shocked arm gas fragments at the Jeans
   length of the compressed layer, λ = 2c_s² / (GΣ_gas,arm) (Elmegreen &
   Elmegreen 1983), into star-forming complexes that sit ~0.1 kpc
   downstream of the shock (stars form a few Myr later; Egusa et al. 2009).
   Complex luminosities follow the HII-region luminosity function
   dN/dL ∝ L^-2 (Kennicutt, Edgar & Hodge 1989), sizes scale as L^{1/3}, and
   each complex is a two-level hierarchy of sub-clumps (≥ 50 pc; the
   clustered, fractal-like distribution of young stars; Elmegreen & Efremov
   1996) embedded in its parent cloud, a smooth envelope the size of the
   Jeans fragment that carries half of the luminosity. Many
   complexes are sheared downstream into spurs (feathers): modest gas-density
   enhancements (30-80% over the local gas, 100-300 pc wide) that leave the
   arm on its convex side at the pitch a radial filament reaches after a
   quarter of the interarm crossing time, tan p_s = 1/(Γ Ω Δt) (Balbus 1988;
   Kim & Ostriker 2002; La Vigne et al. 2006).

10. **Turbulent ISM.** The gas surface density carries a lognormal field
   (the PDF produced by supersonic turbulence) with a power spectrum
   P(k) ∝ k^-2.7 on scales larger than the gas-disk thickness, steepening to
   k^-3.7 below it where structure averages out along the line of sight
   (both measured for the LMC HI; Elmegreen, Kim & Staveley-Smith 2001),
   fed on galactic (~3 kpc) scales. It breaks the arms into patchy
   fragments and populates the interarm regions with flocculent debris.
   The same turbulence moves the gas: each velocity component carries a
   correlated Gaussian field with the k^-3 spectrum of observed velocity
   centroids (Stanimirović & Lazarian 2001) and an rms of
   ``turb_velocity_fraction`` × c_s, i.e. the 5-9 km/s of non-circular
   motions measured in nearby disks (Trachternach et al. 2008). The
   per-voxel random draw keeps the rest of the dispersion, so the total
   stays c_s.

11. **Galaxy centre.** Spiral torques drive gas inwards. Gas stalls near the
    inner ILR, forming a nuclear ring (Combes & Gerin 1985;
    Buta & Combes 1996; cf. the circumnuclear ring of NGC 7469), and the rest
    builds a central molecular concentration on the bulge scale (Regan et al.
    2001; Sakamoto et al. 1999). The gas velocity dispersion rises towards
    the centre (e.g. Sun et al. 2020), modelled as
    σ(R)² = c_s² + σ_c² exp(-(R / R_e,bulge)²).

12. **Emission.** The line emission scales as Σ_gas^N. The default index
    N = 1 suits a pure gas tracer (CO, HI); use N ≈ 1.4 (Kennicutt 1998)
    for a star-formation tracer such as Hα. Arms and turbulence only
    *redistribute* the disk emission around each annulus; the nuclear ring
    and central concentration carry the fractions ``ring_fraction`` and
    ``center_fraction`` of the total.

Conventions
-----------
Lengths are in kpc, velocities in km/s and frequencies in km/s/kpc. Physics
is computed in a frame where the disk rotates towards increasing φ' (the
usual Binney & Tremaine convention, in which k > 0 means trailing arms).
SONGS disks rotate clockwise in their own (x, y) plane (tangent vector
(y, -x)), so the public grid methods map φ' = atan2(-y, x) and flip the y
velocity component. Mirroring both the arms and the rotation keeps the arms
trailing.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import i0e, i1e, k0e, k1e

# Gravitational constant in kpc (km/s)^2 / Msun.
G_KPC = 4.30091e-6

# Default sampling ranges for sample_spiral_params. Each entry is a
# (low, high) uniform range or, for the arm multiplicities, a
# {m: probability} mapping. The observed arm/interarm contrast (moment 0 of
# NGC 7469 in ALMA CO(2-1): ~3 to ~9 between 0.8 and 2 kpc) is the product
# of the smooth shocked arm, the knots and the turbulence, so the smooth-arm
# contrast alone is lower; this also keeps the streaming motions at the
# observed 20-60 km/s. Bulge parameters are calibrated on GALFIT/IMFIT
# decompositions of Sab-Sc spirals (Gadotti 2009; Salo et al. 2015).
DEFAULT_SPIRAL_RANGES = {
    'm': {1: 0.1, 2: 0.6, 3: 0.3},     # grand-design 2-armed spirals dominate
    'disc_fraction': (0.55, 0.85),     # sub-maximal .. maximal disks
    'halo_core': (0.5, 2.0),           # pseudo-isothermal core radius / R_d
    'bulge_fraction': (0.05, 0.3),     # bulge-to-total mass ratio B/T
    'bulge_size': (0.12, 0.3),         # bulge effective radius / R_d
    'Q_min': (1.0, 1.3),               # marginally stable disks
    'corotation': (3.5, 5.5),          # R_CR / R_d; arms end near the inner 4:1 / Q-barrier
    'arm_contrast': (2.0, 4.5),        # smooth shocked-arm peak/trough density ratio
    'skew': (0.3, 0.6),                # shock asymmetry of the arm profile
    'arm_segmentation': (0.4, 0.8),    # rms of ln(arm strength) along the arms
    'c_s': (8.0, 15.0),                # effective (turbulent) gas sound speed, km/s
    'gas_fraction': (0.1, 0.3),        # Σ_gas / Σ_disk
    'knot_fraction': (0.05, 0.2),      # share of the disk emission in star-forming complexes
    'turb_sigma': (0.25, 0.45),        # rms of ln Σ_gas from turbulence
    'turb_velocity_fraction': (0.5, 0.7),  # coherent turbulent velocity rms / c_s
    'center_fraction': (0.05, 0.25),   # central molecular concentration
    'ring_fraction': (0.0, 0.15),      # nuclear ring at the ILR (0 if there is no ILR)
    'sigma_center': (20.0, 50.0),      # central rise of the gas dispersion, km/s
    'emissivity_index': (1.0, 1.0),    # Σ_gas^N; 1 for CO/HI, ~1.4 for Hα
    'secondary_probability': 0.6,      # chance of a second, weaker mode
    'm2': {1: 0.3, 2: 0.3, 3: 0.4},    # (m2 == m is excluded when drawn)
    'corotation2': (3.5, 6.5),
    'secondary_strength': (0.25, 0.5), # (arm_contrast2 - 1) / (arm_contrast - 1)
}

TURB_SLOPE = 2.7          # power-spectrum slope of the turbulent column density
TURB_VELOCITY_SLOPE = 3.0 # power-spectrum slope of turbulent velocity centroids
TURB_DRIVING_KPC = 3.0    # turbulence fed on galactic (shear, superbubble, spiral-shock) scales
TURB_THICKNESS_KPC = 0.25 # gas-disk thickness: below it the projected spectrum steepens by k^-1
SPUR_PROBABILITY = 0.6    # fraction of knots that grow a spur
SEGMENT_SCALE_RAD = 0.5   # correlation length of the arm-segment field in (ln R, φ)


def segment_field(seed, sigma, scale=SEGMENT_SCALE_RAD, n_waves=64):
    """Mean-one lognormal modulation M(ln R, φ), smooth on ``scale``.

    Built from random plane waves in (ln R, φ) with integer azimuthal
    wavenumbers, so it is periodic in φ. Returns a function of (R, φ).
    """
    rng = np.random.RandomState(seed)
    n = np.round(rng.normal(0, 1.0 / scale, n_waves))
    q = rng.normal(0, 1.0 / scale, n_waves)
    psi = rng.uniform(0, 2 * np.pi, n_waves)

    def field(R, phi):
        u = np.log(np.maximum(np.asarray(R, dtype=float), 1e-3))
        g = np.sqrt(2.0 / n_waves) * np.sum(
            np.cos(np.multiply.outer(u, q) + np.multiply.outer(phi, n) + psi), axis=-1)
        return np.exp(sigma * g - 0.5 * sigma**2)
    return field


def freeman_disc_vcirc(R, Sigma0, R_d):
    """Circular speed of a razor-thin exponential disk (Freeman 1970).

    v² = 4πGΣ₀R_d y² [I₀(y)K₀(y) - I₁(y)K₁(y)] with y = R / (2R_d). The
    exponentially scaled Bessel functions are used so that the products
    stay finite at large y.
    """
    y = np.maximum(np.asarray(R, dtype=float) / (2.0 * R_d), 1e-8)
    bessel = i0e(y) * k0e(y) - i1e(y) * k1e(y)
    return np.sqrt(np.maximum(4 * np.pi * G_KPC * Sigma0 * R_d * y**2 * bessel, 0.0))


def hernquist_vcirc(R, M, a):
    """Circular speed of a Hernquist (1990) sphere of mass M and scale a."""
    R = np.asarray(R, dtype=float)
    return np.sqrt(G_KPC * M * R) / (R + a)


def pseudo_isothermal_vcirc(R, v_inf, r_c):
    """Circular speed of a pseudo-isothermal halo with core radius r_c."""
    x = np.maximum(np.asarray(R, dtype=float) / r_c, 1e-8)
    return v_inf * np.sqrt(np.maximum(1.0 - np.arctan(x) / x, 0.0))


def _raw_profile(chi, eta, delta):
    psi = chi + delta * (1.0 - np.cos(chi))
    return np.exp(eta * (np.cos(psi) - 1.0))


def arm_profile(chi, eta, delta=0.0):
    """Skewed von Mises arm profile with azimuthal mean 1 (see module docs).

    ``eta`` must be a scalar here; :class:`SpiralDensityWave` uses tabulated
    normalisations for array-valued η.
    """
    if delta == 0.0:
        return _raw_profile(chi, eta, 0.0) / i0e(eta)
    t = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    return _raw_profile(chi, eta, delta) / _raw_profile(t, eta, delta).mean()


def gaussian_field(shape, dx, seed, slope, driving=TURB_DRIVING_KPC, thickness=TURB_THICKNESS_KPC):
    """Zero-mean, unit-variance Gaussian random field on a 2D grid.

    P(k) ∝ (k² + k₀²)^(-slope/2) / (1 + (k·thickness)²): the 2D slope on
    scales larger than the disk thickness, steepening by one power of k
    below it, where structure is averaged along the line of sight (the
    break seen at the LMC disk thickness by Elmegreen, Kim & Staveley-Smith
    2001). ``shape`` is the 2D grid shape and ``dx`` its spacing (kpc); k₀
    is 1/``driving``. The field is periodic over the grid.
    """
    white = np.random.RandomState(seed).normal(size=shape)
    kx = np.fft.fftfreq(shape[0], d=dx)
    ky = np.fft.fftfreq(shape[1], d=dx)
    K2 = kx[:, None] ** 2 + ky[None, :] ** 2
    amp = (K2 + driving ** -2) ** (-slope / 4) / np.sqrt(1.0 + K2 * thickness**2)
    g = np.fft.ifft2(np.fft.fft2(white) * amp).real
    return (g - g.mean()) / g.std()


def lognormal_field(shape, dx, sigma, seed, slope=TURB_SLOPE, driving=TURB_DRIVING_KPC,
                    thickness=TURB_THICKNESS_KPC):
    """Mean-1 lognormal random field of projected (column) density (see
    :func:`gaussian_field` for the power spectrum)."""
    if sigma <= 0:
        return np.ones(shape)
    g = gaussian_field(shape, dx, seed, slope, driving, thickness)
    return np.exp(sigma * g - 0.5 * sigma**2)


# Tabulation grids for the along-arm velocity integral.
_G_ETA = np.linspace(0.0, 4.0, 161)
_G_CHI = np.linspace(0.0, 2 * np.pi, 1441)


def _cumulative(y, x):
    return np.concatenate([np.zeros(y.shape[:-1] + (1,)),
                           np.cumsum(0.5 * (y[..., 1:] + y[..., :-1]) * np.diff(x), axis=-1)], axis=-1)


def _lookup(table, chi, eta):
    """Bilinear lookup of a (η, χ) table at arbitrary χ (2π-periodic)."""
    chi = np.mod(chi, 2 * np.pi)
    shape = np.broadcast(chi, eta).shape
    i = np.clip(np.broadcast_to(eta, shape), _G_ETA[0], _G_ETA[-1]) / (_G_ETA[1] - _G_ETA[0])
    j = np.broadcast_to(chi, shape) / (_G_CHI[1] - _G_CHI[0])
    coords = np.vstack([np.ravel(i), np.ravel(j)])
    return map_coordinates(table, coords, order=1, mode='nearest').reshape(shape)


def _smooth_step(x, x0, width):
    return 0.5 * (1.0 + np.tanh((x - x0) / width))


def _crossings(x, y, level):
    """All x where y crosses `level` (linear interpolation between samples)."""
    s = np.sign(y - level)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return [float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])) for i in idx]


def _nan_if_none(x):
    return np.nan if x is None else float(x)


class SpiralDensityWave:
    """One tightly wound m-armed density-wave mode in a bulge + disk + halo galaxy.

    Parameters
    ----------
    R_d : float
        Exponential disk scale length (kpc).
    v_0 : float
        Total circular speed at 2.2 R_d (km/s).
    m : int
        Number of arms (1, 2 or 3).
    disc_fraction : float
        v_disk / v_c at 2.2 R_d. Sets the disk mass.
    halo_core : float
        Pseudo-isothermal halo core radius in units of R_d.
    bulge_fraction : float
        Bulge-to-total (bulge + disk) mass ratio.
    bulge_size : float
        Bulge effective radius in units of R_d.
    Q_min : float
        Minimum of the fluid Toomre Q(R) profile.
    corotation : float
        Corotation radius in units of R_d; fixes the pattern speed Ω_p.
    arm_contrast : float
        Peak-to-trough gas density ratio of fully shocked arms (> 1).
    skew : float
        Shock asymmetry of the arm profile, in [0, 1).
    c_s : float
        Effective gas sound speed (km/s); sets the Mach number of the flow
        through the arms.
    emissivity_index : float
        Exponent N of the emissivity ∝ Σ_gas^N.
    phase0 : float
        Azimuthal orientation of the pattern (radians).
    wkb_max_pitch_deg : float
        Arms are only launched where the WKB solution has a pitch angle below
        this value (tan p = m/kR must be small for the theory to hold).
    segmentation : float
        rms of ln(arm strength) along the arms (0: perfectly coherent arms).
    segment_seed : int
        Seed of the arm-segment field.
    r_max : float
        Outer edge of the radial tables in units of R_d.
    n_r : int
        Number of radial table samples.
    """

    def __init__(self, R_d, v_0, m=2, disc_fraction=0.7, halo_core=1.0, bulge_fraction=0.0,
                 bulge_size=0.2, Q_min=1.1, corotation=4.5, arm_contrast=4.0, skew=0.0,
                 c_s=10.0, emissivity_index=1.0, phase0=0.0, wkb_max_pitch_deg=20.0,
                 segmentation=0.0, segment_seed=0, r_max=40.0, n_r=8000):
        if int(m) not in (1, 2, 3):
            raise ValueError(f"m must be 1, 2 or 3 (got {m})")
        if not 0.0 < disc_fraction < 1.0:
            raise ValueError("disc_fraction must lie in (0, 1)")
        if not 0.0 <= bulge_fraction < 1.0:
            raise ValueError("bulge_fraction must lie in [0, 1)")
        if arm_contrast < 1.0:
            raise ValueError("arm_contrast must be >= 1")
        if not 0.0 <= skew < 1.0:
            raise ValueError("skew must lie in [0, 1)")

        self.R_d = R_d = float(R_d)
        self.v_0 = float(v_0)
        self.m = m = int(m)
        self.disc_fraction = float(disc_fraction)
        self.Q_min = float(Q_min)
        self.arm_contrast = float(arm_contrast)
        self.skew = float(skew)
        self.c_s = float(c_s)
        self.emissivity_index = float(emissivity_index)
        self.phase0 = float(phase0)
        self.segmentation = float(segmentation)
        self._segments = segment_field(segment_seed, self.segmentation) if segmentation > 0 else None

        R = np.linspace(r_max * R_d / n_r, r_max * R_d, n_r)
        self.R = R

        # ---- mass model -------------------------------------------------
        y22 = 1.1  # 2.2 R_d / (2 R_d)
        b22 = i0e(y22) * k0e(y22) - i1e(y22) * k1e(y22)
        self.Sigma0 = (disc_fraction * v_0) ** 2 / (4 * np.pi * G_KPC * R_d * y22**2 * b22)
        self.M_disc = 2 * np.pi * self.Sigma0 * R_d**2
        self.M_bulge = bulge_fraction / (1.0 - bulge_fraction) * self.M_disc
        self.R_e_bulge = bulge_size * R_d
        self.a_bulge = self.R_e_bulge / 1.8153          # Hernquist: R_e = 1.8153 a
        v_b22 = hernquist_vcirc(2.2 * R_d, self.M_bulge, self.a_bulge)
        v_h22_sq = v_0**2 - (disc_fraction * v_0) ** 2 - v_b22**2
        if v_h22_sq <= 0.05 * v_0**2:
            raise ValueError("disk + bulge exceed the circular speed at 2.2 R_d; lower "
                             "disc_fraction or bulge_fraction")
        self.r_c = halo_core * R_d
        x22 = 2.2 * R_d / self.r_c
        self.v_inf = np.sqrt(v_h22_sq / (1 - np.arctan(x22) / x22))

        self.v_disc = freeman_disc_vcirc(R, self.Sigma0, R_d)
        self.v_bulge = hernquist_vcirc(R, self.M_bulge, self.a_bulge)
        self.v_halo = pseudo_isothermal_vcirc(R, self.v_inf, self.r_c)
        self.v_c = np.sqrt(self.v_disc**2 + self.v_bulge**2 + self.v_halo**2)
        self.Omega = self.v_c / R
        dvdR = np.gradient(self.v_c, R)
        self.kappa = np.sqrt(np.maximum(2 * self.Omega * (self.Omega + dvdR), 0.0))
        self.shear = 1.0 - dvdR * R / self.v_c          # Γ = -dlnΩ/dlnR
        self.Sigma = self.Sigma0 * np.exp(-R / R_d)

        # ---- stability --------------------------------------------------
        q_shape = self.kappa * np.exp(-R / (2 * R_d)) / (np.pi * G_KPC * self.Sigma)
        window = (R >= 0.5 * R_d) & (R <= 5.0 * R_d)
        self.sigma0 = Q_min / q_shape[window].min()
        self.sigma = self.sigma0 * np.exp(-R / (2 * R_d))
        self.Q = self.sigma * self.kappa / (np.pi * G_KPC * self.Sigma)

        # ---- pattern speed and resonances -------------------------------
        self.R_CR = corotation * R_d
        self.Omega_p = float(np.interp(self.R_CR, R, self.Omega))
        self.nu = m * (self.Omega_p - self.Omega) / self.kappa
        ilr = [r for r in _crossings(R, self.nu, -1.0) if r < self.R_CR]
        olr = [r for r in _crossings(R, self.nu, 1.0) if r > self.R_CR]
        uhr_in = [r for r in _crossings(R, self.nu, -0.5) if r < self.R_CR]
        self.R_ILR = max(ilr) if ilr else None
        self.R_iILR = min(ilr) if len(ilr) > 1 else None   # inner ILR (bulge-dominated centre)
        self.R_OLR = min(olr) if olr else None
        self.R_UHR = max(uhr_in) if uhr_in else None    # inner ultraharmonic (4:1 for m=2)

        # ---- WKB dispersion relation ------------------------------------
        self.D = self.Q**2 * (1.0 - self.nu**2)
        disc = 1.0 - self.D
        k_scale = np.pi * G_KPC * self.Sigma / self.sigma**2
        self.k_crit = self.kappa**2 / (2 * np.pi * G_KPC * self.Sigma)
        self.k_r = k_scale * (1.0 + np.sqrt(np.maximum(disc, 0.0)))
        self.k_i = k_scale * np.sqrt(np.maximum(-disc, 0.0))
        self.k_long = k_scale * (1.0 - np.sqrt(np.clip(disc, 0.0, 1.0)))
        self.allowed = disc >= 0.0
        self.phase = _cumulative(self.k_r, R)
        self.pitch_deg = np.degrees(np.arctan(m / (self.k_r * R)))

        # ---- propagation region and amplitude envelope ------------------
        i_cr = int(np.searchsorted(R, self.R_CR))
        i = i_cr
        while i > 0 and not self.allowed[i]:
            i -= 1
        if self.allowed[i] and i > 0:
            self.R_barrier_in = float(R[i])
            while (i > 0 and self.allowed[i - 1] and self.nu[i - 1] > -1.0
                   and self.pitch_deg[i - 1] <= wkb_max_pitch_deg):
                i -= 1
            self.R_in = max(float(R[i]), 0.5 * R_d)
        else:
            self.R_barrier_in = None
            self.R_in = None
        j = i_cr
        while j < n_r - 1 and not self.allowed[j]:
            j += 1
        self.R_barrier_out = float(R[j]) if self.allowed[j] else None
        self.R_out = self.R_OLR if self.R_OLR is not None else float(R[-1])

        width = 0.25 * R_d
        if self.R_in is None:
            start = self.R_barrier_out if self.R_barrier_out is not None else np.inf
        else:
            start = self.R_in
        k_i_active = np.where(R >= start, self.k_i, 0.0)
        self.transmission = np.exp(-_cumulative(k_i_active, R))
        self.envelope = (_smooth_step(R, start, width)
                         * (1.0 - _smooth_step(R, self.R_out, width))
                         * self.transmission)

        arms = self.envelope > 0.5 * max(self.envelope.max(), 1e-12)
        self.arm_mask = arms
        self.arm_extent = float(R[arms].max() - R[arms].min()) if arms.any() else 0.0

        # ---- gas response: shocks and arm profile -----------------------
        self.u0 = m * (self.Omega - self.Omega_p) / self.k_r    # normal inflow speed
        self.mach = np.abs(self.u0) / self.c_s
        with np.errstate(divide='ignore'):
            shocked = 1.0 / (1.0 + self.mach ** -4.0)
        self.shock_efficiency = 0.25 + 0.75 * shocked
        self.contrast = 1.0 + (self.arm_contrast - 1.0) * self.envelope * self.shock_efficiency
        self.eta = 0.5 * np.log(self.contrast)                    # von Mises concentration

        # Profile tables for δ₀ = -skew (inside corotation, where gas overtakes
        # the arms and the shock faces χ < 0); outside corotation the profile
        # is mirrored: ρ(χ; -δ₀) = ρ(-χ; δ₀).
        self.delta0 = -self.skew
        raw = _raw_profile(_G_CHI[None, :], _G_ETA[:, None], self.delta0)
        self._mean_table = raw[:, :-1].mean(axis=1)
        prof = raw / self._mean_table[:, None]
        self._G_table = _cumulative(prof - 1.0, _G_CHI)
        self._norm_table = (prof[:, :-1] ** self.emissivity_index).mean(axis=1)

    # ------------------------------------------------------------------
    # Radial interpolation helpers
    # ------------------------------------------------------------------
    def _at(self, table, R):
        return np.interp(R, self.R, table)

    def _side(self, R):
        """+1 inside corotation (gas overtakes the arms), -1 outside."""
        return np.where(np.asarray(R) < self.R_CR, 1.0, -1.0)

    def pitch_stats(self):
        """Median and 90th-percentile pitch angle (deg) over the arm region."""
        if not self.arm_mask.any():
            return np.nan, np.nan
        p = self.pitch_deg[self.arm_mask]
        return float(np.median(p)), float(np.percentile(p, 90))

    # ------------------------------------------------------------------
    # Fields in the physics frame (R, φ'), rotation towards +φ'
    # ------------------------------------------------------------------
    def chi(self, R, phi):
        """Spiral phase χ = m(φ' - φ₀) + f(R)."""
        return self.m * (phi - self.phase0) + self._at(self.phase, R)

    def segment_factor(self, R, phi):
        """Mean-one modulation of the arm strength along the arms."""
        if self._segments is None:
            return np.ones(np.broadcast(R, phi).shape)
        return self._segments(R, phi)

    def local_eta(self, R, phi):
        """Arm concentration η at (R, φ'), including the arm segments.

        Strong segments are capped at 1.5× the shocked-arm contrast: a shock
        cannot compress the gas arbitrarily, and mass conservation would
        otherwise demand unphysically fast interarm flow.
        """
        eta_max = 0.5 * np.log(max(1.5 * self.arm_contrast, 1.0))
        return np.minimum(self._at(self.eta, R) * self.segment_factor(R, phi), eta_max)

    def _profile(self, chi, eta):
        return _raw_profile(chi, eta, self.delta0) / np.interp(eta, _G_ETA, self._mean_table)

    def density(self, R, phi):
        """Gas surface density relative to the axisymmetric disk, ρ(χ)."""
        s = self._side(R)
        return self._profile(s * self.chi(R, phi), self.local_eta(R, phi))

    def emissivity_factor(self, R, phi):
        """ρ^N divided by its azimuthal mean (flux-redistributing)."""
        eta = self.local_eta(R, phi)
        norm = np.interp(eta, _G_ETA, self._norm_table)
        return self.density(R, phi) ** self.emissivity_index / norm

    def streaming(self, R, phi):
        """Velocity perturbations (v_R, v_φ') in km/s from mass and angular-momentum conservation."""
        s = self._side(R)
        chi = s * self.chi(R, phi)
        eta = self.local_eta(R, phi)
        rho = self._profile(chi, eta)
        k = self._at(self.k_r, R)
        Om = self._at(self.Omega, R)
        kap = self._at(self.kappa, R)
        v_R = self._at(self.u0, R) * (1.0 / rho - 1.0)
        v_phi = kap**2 / (2 * Om * k) * s * _lookup(self._G_table, chi, eta)
        return v_R, v_phi

    def crest_phi(self, R, j=0):
        """Azimuth φ' of the crest (χ = 2πj) of arm j at radius R."""
        return self.phase0 + (2 * np.pi * j - self._at(self.phase, R)) / self.m

    def summary(self, prefix=''):
        """Scalar description of the mode, suitable for metadata."""
        p_med, p_90 = self.pitch_stats()
        Rs = np.linspace(0.5 * self.R_d, 5 * self.R_d, 300)
        chis = np.linspace(0, 2 * np.pi, 181)
        RR, CC = np.meshgrid(Rs, chis, indexing='ij')
        phi = (CC - self._at(self.phase, RR)) / self.m + self.phase0
        v_R, v_phi = self.streaming(RR, phi)
        out = {
            'm': self.m,
            'Omega_p_km_s_kpc': self.Omega_p,
            'R_CR_kpc': self.R_CR,
            'R_ILR_kpc': _nan_if_none(self.R_ILR),
            'R_iILR_kpc': _nan_if_none(self.R_iILR),
            'R_OLR_kpc': _nan_if_none(self.R_OLR),
            'R_arms_in_kpc': _nan_if_none(self.R_in),
            'arm_extent_kpc': self.arm_extent,
            'arm_turns': float(np.ptp(self.phase[self.arm_mask]) / (2 * np.pi * self.m))
            if self.arm_mask.any() else 0.0,
            'pitch_median_deg': p_med,
            'pitch_p90_deg': p_90,
            'arm_contrast': self.arm_contrast,
            'peak_contrast': float(self.contrast.max()),
            'v_R_stream_max_km_s': float(np.abs(v_R).max()),
            'v_phi_stream_max_km_s': float(np.abs(v_phi).max()),
            'phase0': self.phase0,
        }
        return {prefix + k: v for k, v in out.items()}


class SpiralGalaxy:
    """Gas structure of one spiral galaxy: modes, knots, spurs, turbulence, centre.

    All arguments are plain scalars, so a parameter dict produced by
    :func:`sample_spiral_params` fully reproduces the galaxy (stochastic
    parts are seeded by ``seed``).

    Parameters
    ----------
    R_d, v_0, disc_fraction, halo_core, bulge_fraction, bulge_size, Q_min
        Mass model and stability (shared by all modes); see
        :class:`SpiralDensityWave`.
    m, corotation, arm_contrast, phase0
        Main mode.
    m2, corotation2, arm_contrast2, phase0_2
        Optional second mode (``m2=None`` disables it).
    skew : float
        Shock asymmetry of the arm profiles.
    c_s : float
        Effective gas sound speed (km/s); the disk gas velocity dispersion.
    emissivity_index : float
        Emission ∝ Σ_gas^N.
    gas_fraction : float
        Σ_gas / Σ_disk, used for the Jeans length of the knots.
    knot_fraction : float
        Fraction of the disk emission carried by knots and spurs (0 disables).
    turb_sigma : float
        rms of ln Σ_gas in the turbulent field (0 disables).
    turb_velocity_fraction : float
        rms of each coherent turbulent velocity component as a fraction of
        c_s (0 disables); the per-voxel dispersion takes the remainder.
    center_fraction, ring_fraction : float
        Fractions of the total emission in the central concentration and in
        the nuclear ring at the ILR (the ring is folded into the centre when
        the main mode has no ILR).
    sigma_center : float
        Central rise of the gas velocity dispersion (km/s).
    arm_segmentation : float
        rms of ln(arm strength) along the arms (0: perfectly coherent arms).
    seed : int
        Seed for knots, spurs and turbulence.
    wkb_max_pitch_deg : float
        See :class:`SpiralDensityWave`.
    """

    def __init__(self, R_d, v_0, m=2, disc_fraction=0.7, halo_core=1.0, bulge_fraction=0.15,
                 bulge_size=0.2, Q_min=1.1, corotation=4.5, arm_contrast=4.0, phase0=0.0,
                 m2=None, corotation2=5.0, arm_contrast2=2.0, phase0_2=0.0, skew=0.6,
                 c_s=10.0, emissivity_index=1.0, gas_fraction=0.2, knot_fraction=0.2,
                 turb_sigma=0.45, turb_velocity_fraction=0.6, center_fraction=0.15,
                 ring_fraction=0.08, sigma_center=30.0,
                 arm_segmentation=0.6, seed=0, wkb_max_pitch_deg=20.0):
        self.params = dict(R_d=float(R_d), v_0=float(v_0), m=int(m),
                           disc_fraction=float(disc_fraction), halo_core=float(halo_core),
                           bulge_fraction=float(bulge_fraction), bulge_size=float(bulge_size),
                           Q_min=float(Q_min), corotation=float(corotation),
                           arm_contrast=float(arm_contrast), phase0=float(phase0),
                           m2=None if m2 is None else int(m2), corotation2=float(corotation2),
                           arm_contrast2=float(arm_contrast2), phase0_2=float(phase0_2),
                           skew=float(skew), c_s=float(c_s), emissivity_index=float(emissivity_index),
                           gas_fraction=float(gas_fraction), knot_fraction=float(knot_fraction),
                           turb_sigma=float(turb_sigma),
                           turb_velocity_fraction=float(turb_velocity_fraction),
                           center_fraction=float(center_fraction),
                           ring_fraction=float(ring_fraction), sigma_center=float(sigma_center),
                           arm_segmentation=float(arm_segmentation), seed=int(seed),
                           wkb_max_pitch_deg=float(wkb_max_pitch_deg))
        self.R_d = float(R_d)
        self.c_s = float(c_s)
        self.gas_fraction = float(gas_fraction)
        self.knot_fraction = float(knot_fraction)
        self.turb_sigma = float(turb_sigma)
        if not 0.0 <= turb_velocity_fraction < 1.0:
            raise ValueError("turb_velocity_fraction must lie in [0, 1)")
        self.turb_velocity_fraction = float(turb_velocity_fraction)
        self.emissivity_index = float(emissivity_index)
        self.sigma_center = float(sigma_center)
        self.seed = int(seed)

        shared = dict(R_d=R_d, v_0=v_0, disc_fraction=disc_fraction, halo_core=halo_core,
                      bulge_fraction=bulge_fraction, bulge_size=bulge_size, Q_min=Q_min,
                      skew=skew, c_s=c_s, emissivity_index=emissivity_index,
                      wkb_max_pitch_deg=wkb_max_pitch_deg)
        self.primary = SpiralDensityWave(m=m, corotation=corotation,
                                         arm_contrast=arm_contrast, phase0=phase0,
                                         segmentation=arm_segmentation, segment_seed=int(seed) + 2,
                                         **shared)
        self.secondary = None
        if m2 is not None:
            self.secondary = SpiralDensityWave(m=m2, corotation=corotation2,
                                               arm_contrast=arm_contrast2, phase0=phase0_2,
                                               segmentation=arm_segmentation,
                                               segment_seed=int(seed) + 3, **shared)
        self.modes = [w for w in (self.primary, self.secondary) if w is not None]
        self.m = self.primary.m

        # Centre: inflowing gas stalls at the ILR and forms a ring there
        # (Buta & Combes 1996). With a core there may be an inner ILR, which
        # hosts a nuclear ring; with a cuspy bulge there is a single ILR,
        # where the arms start, and the ring is the inner (pseudo-)ring the
        # arms emanate from, kept only within 1.5 R_d.
        self.R_e_bulge = self.primary.R_e_bulge
        self.h_center = self.R_e_bulge / 1.678
        w = self.primary
        if w.R_iILR is not None:
            self.R_ring = w.R_iILR
        elif w.R_ILR is not None and w.R_ILR < 1.5 * w.R_d:
            self.R_ring = w.R_ILR
        else:
            self.R_ring = None
        if self.R_ring is not None and self.R_ring < 2 * self.h_center:
            self.R_ring = None
        self.ring_fraction = float(ring_fraction) if self.R_ring is not None else 0.0
        self.center_fraction = float(center_fraction) + (float(ring_fraction) - self.ring_fraction)

        rng = np.random.RandomState(self.seed)
        self.knots, self.spurs = self._place_knots(rng)

    # ------------------------------------------------------------------
    # Star-forming knots and spurs
    # ------------------------------------------------------------------
    def jeans_spacing(self, R):
        """Fragmentation length λ = 2c_s² / (GΣ_gas,arm) along the main arms (kpc)."""
        w = self.primary
        sigma_arm = self.gas_fraction * w._at(w.Sigma, R) / i0e(w._at(w.eta, R))
        return 2 * self.c_s**2 / (G_KPC * sigma_arm)

    def spur_pitch_deg(self, R):
        """Pitch of a radial filament sheared for a quarter of the interarm crossing time."""
        w = self.primary
        Om = w._at(w.Omega, R)
        dt = 0.25 * 2 * np.pi / (w.m * np.abs(Om - w.Omega_p))
        tan_p = 1.0 / (np.maximum(w._at(w.shear, R), 0.1) * Om * dt)
        return np.clip(np.degrees(np.arctan(tan_p)), 20.0, 60.0)

    def _place_knots(self, rng, L_range=(1.0, 20.0)):
        """Star-forming complexes along the main-mode arms, and the spurs grown from them.

        Complexes are spaced by the local Jeans length and sit slightly
        downstream of the shock. Each is a two-level hierarchy of sub-clumps
        (the clustered, fractal-like distribution of young stars and HII
        regions, Elmegreen & Efremov 1996). Spurs are gas-density features
        (see :meth:`spur_field`), described by points along their spines.
        """
        knot_keys = ('x', 'y', 'R', 'L', 'size')
        spur_keys = ('x', 'y', 'amp', 'width')
        knots = {k: [] for k in knot_keys}
        spurs = {k: [] for k in spur_keys}
        complexes = {k: [] for k in ('x', 'y', 'R', 'L')}
        self.n_complexes = 0
        self.n_spurs = 0
        w = self.primary
        dense = w.contrast >= 1.0 + 0.3 * (w.arm_contrast - 1.0)   # shocked, dense arm gas
        if self.knot_fraction > 0 and w.arm_mask.any() and dense.any():
            Rs = np.linspace(w.R[dense].min(), w.R[dense].max(), 4000)
            lam = self.jeans_spacing(Rs)
            L_lo, L_hi = L_range
            for j in range(w.m):
                phi = w.crest_phi(Rs, j)
                s = np.cumsum(np.hypot(np.gradient(Rs), Rs * np.gradient(phi)))
                s_next = rng.uniform(0, 1) * lam[0]
                for idx in range(Rs.size):
                    if s[idx] < s_next:
                        continue
                    s_next = s[idx] + lam[idx] * rng.uniform(0.6, 1.4)
                    r = Rs[idx] + rng.normal(0, 0.04) * lam[idx]
                    # Stars form a few Myr after the shock: shift downstream by
                    # u_post * t_SF (~0.1 kpc; Egusa et al. 2009)
                    side = 1.0 if r < w.R_CR else -1.0
                    p = phi[idx] + side * rng.uniform(0.05, 0.15) / r
                    # dN/dL ∝ L^-2 between L_lo and L_hi (inverse-CDF sample)
                    L = 1.0 / (1.0 / L_lo - rng.uniform() * (1.0 / L_lo - 1.0 / L_hi))
                    # Strong arm segments host more star formation
                    if rng.uniform() > min(1.0, float(w.segment_factor(r, p))):
                        continue
                    x0, y0 = r * np.cos(p), -r * np.sin(p)
                    self._add_complex(rng, knots, x0, y0, L, lam[idx])
                    for k, v in zip(('x', 'y', 'R', 'L'), (x0, y0, r, L)):
                        complexes[k].append(v)
                    self.n_complexes += 1
                    if r < w.R_CR and rng.uniform() < SPUR_PROBABILITY:
                        self._add_spur(rng, spurs, r, p)
                        self.n_spurs += 1
        self.complexes = {k: np.asarray(v, dtype=float) for k, v in complexes.items()}
        return ({k: np.asarray(v, dtype=float) for k, v in knots.items()},
                {k: np.asarray(v, dtype=float) for k, v in spurs.items()})

    @staticmethod
    def _add_complex(rng, knots, x0, y0, L, lam):
        """Hierarchical complex of total luminosity L inside its parent gas cloud.

        Half the luminosity is in the parent giant molecular association, a
        smooth envelope the size of the Jeans fragment (λ/4); the rest is in
        a two-level hierarchy of sub-clumps (sizes ∝ L^{1/3}, at least 50 pc,
        the scale of giant HII regions and GMC complexes).
        """
        env = float(np.clip(lam / 4.0, 0.15, 0.6))
        for k, v in zip(('x', 'y', 'R', 'L', 'size'), (x0, y0, np.hypot(x0, y0), 0.5 * L, env)):
            knots[k].append(v)
        L = 0.5 * L
        size = 0.12 * L ** (1.0 / 3.0)            # complex radius, kpc (Strömgren-like scaling)
        n1 = 1 + rng.poisson(2.5)
        L1 = L * rng.dirichlet(np.ones(n1))
        for l1 in L1:
            x1, y1 = (x0, y0) if n1 == 1 else (x0 + rng.normal(0, 0.5 * size), y0 + rng.normal(0, 0.5 * size))
            n2 = 1 + rng.poisson(2.0)
            L2 = l1 * rng.dirichlet(np.ones(n2))
            for l2 in L2:
                s2 = max(0.35 * size * (l2 / L) ** (1.0 / 3.0), 0.05)
                x2 = x1 + rng.normal(0, 0.25 * size)
                y2 = y1 + rng.normal(0, 0.25 * size)
                for k, v in zip(('x', 'y', 'R', 'L', 'size'), (x2, y2, np.hypot(x2, y2), l2, s2)):
                    knots[k].append(v)

    def _add_spur(self, rng, spurs, r, p):
        """Spine of a spur leaving the arm on its convex side at the shear pitch."""
        w = self.primary
        tan_ps = np.tan(np.radians(self.spur_pitch_deg(r)))
        length = rng.uniform(0.3, 0.7) * 2 * np.pi / w._at(w.k_r, r)
        width = rng.uniform(0.08, 0.2)             # observed feathers: ~100-300 pc wide
        amp = rng.uniform(0.3, 0.8)                # density enhancement over the local gas
        ell = np.arange(0.0, length, 0.5 * width)
        rr = r + ell * tan_ps / np.hypot(1, tan_ps)
        pp = p - np.log(rr / r) / tan_ps           # dφ'/dR = -1/(R tan p_s)
        # Grows out of the arm, then fades downstream
        profile = amp * (1 - np.exp(-ell / (0.15 * length))) * np.exp(-ell / (0.6 * length))
        for k, v in (('x', rr * np.cos(pp)), ('y', -rr * np.sin(pp)), ('amp', profile),
                     ('width', np.full(ell.size, width))):
            spurs[k].extend(v)

    @staticmethod
    def _grid_dx(x):
        x = np.asarray(x)
        return abs(float(x[1, 0] - x[0, 0])) if x.ndim == 2 and x.shape[0] > 1 else 0.0

    def _blob_field(self, x, y, blobs, dx):
        """Sum of pixel-integrated Gaussians, each integrating to its luminosity L (kpc^-2)."""
        out = np.zeros(np.shape(x))
        for xk, yk, L, s in zip(blobs['x'], blobs['y'], blobs['L'], blobs['size']):
            s2 = s**2 + dx**2 / 12.0               # a Gaussian integrated over a pixel
            d2 = (x - xk) ** 2 + (y - yk) ** 2
            near = d2 < 25 * s2
            out[near] += L / (2 * np.pi * s2) * np.exp(-0.5 * d2[near] / s2)
        return out

    def clump_field(self, x, y):
        """Star-forming complexes (unnormalised, kpc^-2)."""
        return self._blob_field(x, y, self.knots, self._grid_dx(x))

    def spur_field(self, x, y):
        """Fractional gas-density enhancement of the spurs, S ≥ 0 (density × (1 + S))."""
        out = np.zeros(np.shape(x))
        dx = self._grid_dx(x)
        sp = self.spurs
        for xs, ys, a, wd in zip(sp['x'], sp['y'], sp['amp'], sp['width']):
            s2 = wd**2 + dx**2 / 12.0
            d2 = (x - xs) ** 2 + (y - ys) ** 2
            near = d2 < 25 * s2
            # Spine points every half width: the sum along the spine is a ridge of height a
            out[near] += a * 0.5 * wd / np.sqrt(2 * np.pi * s2) * np.exp(-0.5 * d2[near] / s2)
        return out

    # ------------------------------------------------------------------
    # Fields on a SONGS Cartesian grid (clockwise rotation)
    # ------------------------------------------------------------------
    @staticmethod
    def _polar(x, y):
        return np.hypot(x, y), np.arctan2(-y, x)

    def _raw_arm_factor(self, R, phi):
        f = np.ones(np.broadcast(R, phi).shape)
        for w in self.modes:
            f = f * w.emissivity_factor(R, phi)
        return f

    def _arm_norm(self, R):
        """Azimuthal mean of the raw arm factor at radius R (tabulated once).

        Each mode's profile has mean 1 at fixed η, but arm segments and
        mode interference correlate with the arm phase, so the product is
        renormalised to keep the arms purely flux-redistributing.
        """
        if not hasattr(self, '_norm_R'):
            self._norm_R = np.linspace(0.0, self.primary.R[-1], 600)
            phi = np.linspace(-np.pi, np.pi, 720, endpoint=False)
            self._norm_val = self._raw_arm_factor(self._norm_R[:, None], phi[None, :]).mean(axis=1)
        return np.interp(R, self._norm_R, self._norm_val)

    def arm_factor_on_grid(self, x, y):
        """Product of the modes' emissivity factors, with azimuthal mean 1 at every
        radius (no clumps, no turbulence)."""
        R, phi = self._polar(x, y)
        return self._raw_arm_factor(R, phi) / self._arm_norm(R)

    def turbulence_on_grid(self, x, y):
        """Mean-1 lognormal turbulent factor for the emission on a regular 2D grid."""
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[0] < 2 or self.turb_sigma <= 0:
            return np.ones(np.shape(x))
        dx = abs(float(x[1, 0] - x[0, 0]))
        return lognormal_field(x.shape, dx, self.emissivity_index * self.turb_sigma, self.seed + 1)

    def emissivity_on_grid(self, x, y, return_parts=False):
        """Disk emissivity multiplier: arms × turbulence × spurs × (1 + clumps).

        The clump amplitude is set so that the star-forming complexes carry
        ``knot_fraction`` of the emission of an exponential disk with these
        arms on this grid.
        """
        arms = self.arm_factor_on_grid(x, y)
        turb = self.turbulence_on_grid(x, y)
        spurs = (1.0 + self.spur_field(x, y)) ** self.emissivity_index
        clumps = self.clump_field(x, y)
        base = np.exp(-np.hypot(x, y) / self.R_d) * arms * turb * spurs
        scale = 0.0
        if self.knot_fraction > 0 and np.sum(base * clumps) > 0:
            scale = (self.knot_fraction / (1.0 - self.knot_fraction)
                     * np.sum(base) / np.sum(base * clumps))
        total = arms * turb * spurs * (1.0 + scale * clumps)
        if return_parts:
            return total, arms, turb, spurs, scale * clumps
        return total

    def emission_cube(self, X, Y, Z, disk, return_parts=False):
        """Full 3D emission: structured disk + central concentration + nuclear ring.

        ``disk`` is the axisymmetric disk emissivity on the (X, Y, Z) grid
        (indexing='ij', kpc). The disk keeps its own total; the centre and
        ring are added so that they carry ``center_fraction`` and
        ``ring_fraction`` of the result.
        """
        x2, y2 = X[:, :, 0], Y[:, :, 0]
        turb = self.turbulence_on_grid(x2, y2)
        disk_s = disk * self.emissivity_on_grid(x2, y2)[:, :, None]
        S = disk_s.sum()
        rest = 1.0 - self.center_fraction - self.ring_fraction
        center = np.zeros_like(disk)
        ring = np.zeros_like(disk)
        if self.center_fraction > 0:
            # Flattened (q = 0.5) exponential concentration on the bulge scale
            center = np.exp(-np.sqrt(X**2 + Y**2 + (Z / 0.5) ** 2) / self.h_center) * turb[:, :, None]
            center *= S * self.center_fraction / rest / center.sum()
        if self.ring_fraction > 0:
            R2 = np.hypot(x2, y2)
            width = 0.12 * self.R_ring
            ring2 = np.exp(-0.5 * ((R2 - self.R_ring) / width) ** 2) * turb
            c = disk.shape[0] // 2
            vertical = disk[c, c, :] / max(disk[c, c, :].max(), 1e-300)   # the disk's own z profile
            ring = ring2[:, :, None] * vertical[None, None, :]
            if ring.sum() > 0:
                ring *= S * self.ring_fraction / rest / ring.sum()
        total = disk_s + center + ring
        if return_parts:
            return total, disk_s, center, ring
        return total

    def dispersion_on_grid(self, x, y):
        """Small-scale (per-voxel) gas velocity dispersion (km/s), rising towards the centre.

        The coherent turbulent field (:meth:`turbulent_velocity_on_grid`)
        carries ``turb_velocity_fraction`` of c_s; this is the remainder, so
        that the two add up to c_s in the disk.
        """
        R = np.hypot(x, y)
        small = self.c_s**2 * (1.0 - self.turb_velocity_fraction**2)
        return np.sqrt(small + self.sigma_center**2 * np.exp(-(R / self.R_e_bulge) ** 2))

    def turbulent_velocity_on_grid(self, x, y):
        """Coherent turbulent velocity (v_x, v_y, v_z) in km/s on a regular 2D grid.

        Isotropic: three independent Gaussian fields with the velocity-centroid
        spectrum, each with rms ``turb_velocity_fraction`` × c_s.
        """
        x = np.asarray(x)
        zero = np.zeros(np.shape(x))
        if x.ndim != 2 or x.shape[0] < 2 or self.turb_velocity_fraction <= 0:
            return zero, zero, zero
        dx = abs(float(x[1, 0] - x[0, 0]))
        rms = self.turb_velocity_fraction * self.c_s
        return tuple(rms * gaussian_field(x.shape, dx, self.seed + 5 + i, TURB_VELOCITY_SLOPE)
                     for i in range(3))

    def velocity_on_grid(self, x, y, perturbed=True):
        """In-plane gas velocity (v_x, v_y) on a SONGS grid (kpc → km/s)."""
        R, phi = self._polar(x, y)
        v_phi = self.primary._at(self.primary.v_c, R)
        v_R = np.zeros_like(R)
        if perturbed:
            for w in self.modes:
                dv_R, dv_phi = w.streaming(R, phi)
                v_R = v_R + dv_R
                v_phi = v_phi + dv_phi
        vx = v_R * np.cos(phi) - v_phi * np.sin(phi)
        vy_prime = v_R * np.sin(phi) + v_phi * np.cos(phi)
        return vx, -vy_prime

    def arm_crests(self, mode=0, n=400):
        """Arm-crest loci (χ = 2πj) in SONGS (x, y) kpc, one (x, y) pair per arm."""
        w = self.modes[mode]
        if not w.arm_mask.any():
            return []
        Rs = np.linspace(w.R[w.arm_mask].min(), w.R[w.arm_mask].max(), n)
        crests = []
        for j in range(w.m):
            phi = w.crest_phi(Rs, j)
            crests.append((Rs * np.cos(phi), -Rs * np.sin(phi)))
        return crests

    def summary(self):
        """Flat scalar description (HDF5-attribute friendly; no None values)."""
        w = self.primary
        out = {k: (np.nan if v is None else v) for k, v in self.params.items()}
        out.update(w.summary())
        if self.secondary is not None:
            out.update(self.secondary.summary(prefix='mode2_'))
        out.update(
            M_disc_Msun=w.M_disc,
            M_bulge_Msun=w.M_bulge,
            R_e_bulge_kpc=w.R_e_bulge,
            Sigma0_Msun_pc2=w.Sigma0 / 1e6,
            sigma0_km_s=w.sigma0,
            R_ring_kpc=_nan_if_none(self.R_ring),
            ring_fraction_used=self.ring_fraction,
            n_knots=int(self.n_complexes),
            n_knot_clumps=int(self.knots['x'].size),
            n_spurs=int(self.n_spurs),
        )
        return out


def sample_spiral_params(R_d, v_0, overrides=None, ranges=None, max_pitch_deg=15.0,
                         min_pitch_deg=7.0, min_arm_extent=1.5, max_tries=1000, rng=None):
    """Draw physical spiral parameters that yield tightly wound arms.

    Parameters are sampled from ``ranges`` (default
    :data:`DEFAULT_SPIRAL_RANGES`) and kept only if the main mode has 90% of
    its arm region below ``max_pitch_deg``, a median pitch above
    ``min_pitch_deg``, and arms spanning at least ``min_arm_extent`` R_d. A
    second mode is kept only if it too has arms (it is otherwise dropped).
    Keys present in ``overrides`` are held fixed; ``overrides={'m2': None}``
    disables the second mode.

    Returns
    -------
    dict
        Keyword arguments for :class:`SpiralGalaxy`.
    """
    rng = np.random if rng is None else rng
    overrides = dict(overrides or {})
    spec = dict(DEFAULT_SPIRAL_RANGES)
    if ranges:
        spec.update(ranges)

    def draw(key, exclude=None):
        if key in overrides:
            return overrides[key]
        s = spec[key]
        if isinstance(s, dict):
            keys = [k for k in s if k != exclude]
            p = np.asarray([s[k] for k in keys], dtype=float)
            return int(keys[rng.choice(len(keys), p=p / p.sum())])
        lo, hi = s
        return float(rng.uniform(lo, hi))

    mass_keys = ('disc_fraction', 'halo_core', 'bulge_fraction', 'bulge_size', 'Q_min')
    gas_keys = ('arm_contrast', 'skew', 'arm_segmentation', 'c_s', 'emissivity_index', 'gas_fraction', 'knot_fraction',
                'turb_sigma', 'turb_velocity_fraction', 'center_fraction', 'ring_fraction', 'sigma_center')
    for _ in range(max_tries):
        params = dict(R_d=float(R_d), v_0=float(v_0))
        for key in ('m',) + mass_keys + ('corotation',) + gas_keys:
            params[key] = draw(key)
        params['phase0'] = overrides.get('phase0', float(rng.uniform(0, 2 * np.pi)))
        try:
            main = SpiralDensityWave(**{k: params[k] for k in
                                        ('R_d', 'v_0', 'm', 'corotation', 'arm_contrast', 'skew',
                                         'c_s', 'phase0') + mass_keys})
        except ValueError:
            continue                       # unphysical mass model (disk + bulge > v_c)
        p_med, p_90 = main.pitch_stats()
        if not (main.arm_extent >= min_arm_extent * R_d and np.isfinite(p_90)
                and p_90 <= max_pitch_deg and p_med >= min_pitch_deg):
            continue

        # Optional second, weaker mode with its own pattern speed
        if 'm2' in overrides:
            want_second = overrides['m2'] is not None
        else:
            want_second = rng.uniform() < spec['secondary_probability']
        params['m2'] = None
        if want_second:
            for _ in range(50):
                m2 = draw('m2', exclude=params['m'])
                cr2 = draw('corotation2')
                if 'arm_contrast2' in overrides:
                    c2 = overrides['arm_contrast2']
                else:
                    c2 = 1.0 + (params['arm_contrast'] - 1.0) * draw('secondary_strength')
                ph2 = overrides.get('phase0_2', float(rng.uniform(0, 2 * np.pi)))
                second = SpiralDensityWave(R_d=R_d, v_0=v_0, m=m2, corotation=cr2,
                                           arm_contrast=c2, skew=params['skew'], c_s=params['c_s'],
                                           phase0=ph2, **{k: params[k] for k in mass_keys})
                if second.arm_extent >= 1.0 * R_d:
                    params.update(m2=m2, corotation2=cr2, arm_contrast2=c2, phase0_2=ph2)
                    break
        params['seed'] = int(overrides.get('seed', rng.randint(0, 2**31 - 1)))
        return params
    raise RuntimeError(
        f"No tightly wound spiral ({min_pitch_deg} <= pitch <= {max_pitch_deg} deg) found in "
        f"{max_tries} draws for R_d={R_d:.2f} kpc, v_0={v_0:.0f} km/s with overrides {overrides}.")
