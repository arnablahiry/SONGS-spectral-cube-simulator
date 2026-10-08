"""Step-by-step figures for the SONGS density-wave spiral model.

Builds one fiducial spiral galaxy through the real SONGS pipeline
(``SONGSPhy(morphology='spiral')``), records every intermediate stage, and
writes a figure for each step to ``spiral_galaxies/figures/``:

    01-11  the physics: mass model with bulge, resonances, stability,
           dispersion relation, pitch angles, arm extent and shocks, arm
           geometry, shock profile and streaming, knots and spurs,
           turbulence, the galaxy centre (bulge, central gas, ring)
    12-18  the 3D Cartesian grid, emissivity slices, the face-on build-up of
           every component, spiral vs axisymmetric twin, velocity field,
           the rotation to the observer frame, LOS velocity residuals
    19-21  the spectral cube: channel maps, moments + spectrum, PV diagram
    22-23  a gallery of randomly sampled spirals and the sampled parameter
           distributions
    24-25  comparison with the ALMA CO(2-1) cube of NGC 7469

Every system it builds (the fiducial spiral, its axisymmetric twin and the
gallery spirals) is also saved as 3D numpy arrays under
``spiral_galaxies/systems/<name>/``:

    cube.npy                     spectral cube, float32 (n_v, n_y, n_x)
    velocities_km_s.npy          channel centres (n_v,)
    emissivity_galaxy_frame.npy  intrinsic 3D emission before rotation (z, y, x)
    emissivity_sky.npy           3D emission after rotation, z = line of sight (z, y, x)
    vlos_sky.npy                 line-of-sight velocity of every voxel (z, y, x), km/s
    params.json                  spiral parameters, derived quantities and grid metadata

Usage::

    python spiral_galaxies/make_spiral_figures.py [--ngc7469 PATH] [--skip-gallery]

The NGC 7469 cube is read from the IFU-Source-ID repository by default; the
comparison figures are skipped if it is not found.
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.ndimage import binary_dilation, gaussian_filter

from songs.core import SONGS, SONGSPhy
from songs.spiral import SpiralGalaxy, sample_spiral_params

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
SYSTEMS = os.path.join(HERE, 'systems')
NGC7469_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'IFU-Source-ID', 'data',
                               'observed_cubes', 'CO21_image_pbcor-degrad.fits')

# Categorical slots (validated reference palette, slots 1-3) + neutral ink
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#8a8984', '#d9d8d4'
SEQ, DIV, WIDTH = 'magma', 'coolwarm', 'viridis'

plt.rcParams.update({
    'figure.dpi': 110, 'savefig.dpi': 150, 'figure.constrained_layout.use': True,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.alpha': 0.7,
    'lines.linewidth': 2.0, 'legend.frameon': False, 'image.origin': 'lower',
})

# Fiducial galaxy: a two-armed grand-design spiral with a weaker m = 3 mode,
# an Sb-like bulge, a central gas concentration and a ring at the ILR.
FIDUCIAL = dict(
    m=2, disc_fraction=0.75, halo_core=1.0, bulge_fraction=0.2, bulge_size=0.2, Q_min=1.2,
    corotation=4.5, arm_contrast=3.5, phase0=0.6, skew=0.5,
    m2=3, corotation2=5.5, arm_contrast2=1.8, phase0_2=1.0,
    c_s=10.0, emissivity_index=1.0, gas_fraction=0.2, knot_fraction=0.2, turb_sigma=0.35,
    center_fraction=0.15, ring_fraction=0.08, sigma_center=30.0, arm_segmentation=0.6,
    turb_velocity_fraction=0.6, seed=11,
)
TWIN = dict(FIDUCIAL, arm_contrast=1.0, m2=None, knot_fraction=0.0, turb_sigma=0.0, turb_velocity_fraction=0.0)
PIX_KPC = 0.2          # spatial resolution
FOV_KPC = 36.0
CHAN_KMS = 10.0        # as in the NGC 7469 cube
BEAM_KPC = 0.4
INCLINATION = 50.0
HZ_KPC = 0.25
VIEW = 14.0            # half-width of the zoomed maps, kpc
GALLERY_SEED = 5       # sampler seed of the gallery spirals
GALLERY_INCLINATIONS = (15, 30, 40, 50, 55, 60, 70, 35)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print('  wrote', os.path.relpath(path, os.path.dirname(HERE)), flush=True)


def title(fig, text):
    fig.suptitle(text, x=0.01, ha='left', fontsize=12)


def colorbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    cb.outline.set_visible(False)
    return cb


def sky(ax, img, extent, ttl, xy=True, **kw):
    """Show an (x, y)-indexed map with x horizontal."""
    im = ax.imshow(np.asarray(img).T, extent=extent, **kw)
    ax.set_title(ttl)
    if xy:
        ax.set_xlabel('x [kpc]')
        ax.set_ylabel('y [kpc]')
    ax.grid(False)
    return im


def zoom(ax, lim=VIEW):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def resonance_lines(ax, wave, color, prefix=''):
    for R, name in ((wave.R_ILR, 'ILR'), (wave.R_UHR, '4:1'), (wave.R_CR, 'CR'), (wave.R_OLR, 'OLR')):
        if R is not None and R < 25:
            ax.axvline(R, color=color, lw=0.9, ls=':', zorder=0)
            ax.text(R, 0.98, f' {prefix}{name}', color=color, fontsize=8, rotation=90,
                    va='top', ha='right', transform=ax.get_xaxis_transform())


def radial_profile(img, axis_kpc, edges):
    X, Y = np.meshgrid(axis_kpc, axis_kpc, indexing='ij')
    R = np.hypot(X, Y)
    idx = np.digitize(R.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1)
    s = np.bincount(idx[ok], weights=img.ravel()[ok], minlength=len(edges) - 1)
    n = np.bincount(idx[ok], minlength=len(edges) - 1)
    return s / np.maximum(n, 1)


# ---------------------------------------------------------------------------
# Building the fiducial galaxy through SONGS
# ---------------------------------------------------------------------------

def build(spiral_params, label, inclination=INCLINATION, pix=PIX_KPC):
    """Run SONGSPhy end to end and keep every intermediate stage."""
    t0 = time.time()
    g = SONGSPhy(n_gals=1, n_cubes=1, spatial_resolution=pix, spectral_resolution=CHAN_KMS,
                 fov=FOV_KPC, beam_info=[BEAM_KPC, BEAM_KPC, 0], seed=4, verbose=False,
                 morphology='spiral', spiral_params=spiral_params,
                 diffuse_params={'enabled': False})
    assert g.all_spiral_params[0][0]['m2'] == spiral_params['m2']
    # Thin gas disk, viewed at a single clean inclination
    g.all_hz = [np.array([HZ_KPC / pix])]
    g.all_gal_x_angles = [np.array([float(inclination)])]
    g.all_gal_y_angles = [np.array([0.0])]
    g.keep_intermediates = True
    cube, params = g.generate_cubes()[0]
    params['inclination_deg'] = float(inclination)
    params['pix_kpc'] = float(pix)
    print(f'  built {label} cube {cube.shape} in {time.time() - t0:.0f} s', flush=True)
    return cube, params, g.intermediates[0]


def _jsonable(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def save_system(name, cube, params, stage):
    """Write one system as 3D numpy arrays (astronomical (v|z, y, x) order) + JSON metadata."""
    d = os.path.join(SYSTEMS, name)
    os.makedirs(d, exist_ok=True)
    # SONGS arrays are indexed (.., x, y); store them as (.., y, x) like a FITS cube
    np.save(os.path.join(d, 'cube.npy'), cube.transpose(0, 2, 1).astype(np.float32))
    np.save(os.path.join(d, 'velocities_km_s.npy'), np.asarray(params['average_vels'], dtype=np.float64))
    np.save(os.path.join(d, 'emissivity_galaxy_frame.npy'), stage['disk'].transpose(2, 1, 0).astype(np.float32))
    np.save(os.path.join(d, 'emissivity_sky.npy'), stage['rotated_disk'].transpose(2, 1, 0).astype(np.float32))
    np.save(os.path.join(d, 'vlos_sky.npy'), stage['rotated_vel_los'].transpose(2, 1, 0).astype(np.float32))
    meta = {
        'spiral': {k: _jsonable(v) for k, v in params['spiral'].items()},
        'inclination_deg': params['inclination_deg'],
        'pixel_kpc': params['pix_kpc'],
        'channel_width_km_s': CHAN_KMS,
        'beam_fwhm_kpc': BEAM_KPC,
        'scale_height_kpc': HZ_KPC,
        'grid_axis_kpc': [float(a) for a in stage['axis_kpc'][[0, -1]]],
        'arrays': {
            'cube.npy': 'beam-convolved spectral cube, (n_v, n_y, n_x), flux per pixel per channel',
            'velocities_km_s.npy': 'channel centres, km/s',
            'emissivity_galaxy_frame.npy': 'intrinsic 3D emission in the disk frame, (z, y, x)',
            'emissivity_sky.npy': '3D emission after rotation to the sky, (z = line of sight, y, x)',
            'vlos_sky.npy': 'line-of-sight velocity of each voxel incl. dispersion draw, (z, y, x), km/s',
        },
    }
    with open(os.path.join(d, 'params.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    size = sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d)) / 1e6
    print(f'  saved {os.path.relpath(d, os.path.dirname(HERE))} ({size:.0f} MB)', flush=True)


def masked_moments(cube, v, free=None, rms=None):
    """Smooth-and-mask moments, as for an interferometric cube.

    cube is (n_v, a, b). The noise of the smoothed cube is measured in the
    line-free channels ``free`` (per pixel; this handles beam-correlated and
    primary-beam-corrected noise), or derived from a white-noise ``rms``.
    """
    s = gaussian_filter(cube, (1, 2, 2))
    if free is not None:
        s_rms = s[free].std(0)
        s_rms[s_rms == 0] = np.inf
    else:
        white = gaussian_filter(np.random.RandomState(0).normal(size=(32, 64, 64)), (1, 2, 2))
        s_rms = white.std() * rms
    mask = binary_dilation(s > 5 * s_rms, iterations=2) & (s > 2 * s_rms)
    dd = np.where(mask, cube, 0.0)
    W = dd.sum(0)
    good = W > 0
    m0 = W * abs(v[1] - v[0])
    m1 = np.where(good, (dd * v[:, None, None]).sum(0) / np.where(good, W, 1), np.nan)
    var = (dd * (v[:, None, None] - np.nan_to_num(m1)) ** 2).sum(0) / np.where(good, W, 1)
    m2 = np.where(good, np.sqrt(np.maximum(var, 0)), np.nan)
    return m0, m1, m2, good


def add_noise(cube, beam_px, peak_snr, seed=0):
    """Beam-correlated Gaussian noise with a given peak S/N."""
    n = np.random.RandomState(seed).normal(size=cube.shape)
    s = beam_px / 2.355
    n = gaussian_filter(n, (0, s, s))
    rms = cube.max() / peak_snr
    return cube + n / n.std() * rms, rms


# ---------------------------------------------------------------------------
# 01-11: the physics
# ---------------------------------------------------------------------------

def fig_mass_model(gal):
    w = gal.primary
    R = w.R
    sel = R <= 8 * w.R_d
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(R[sel], w.v_c[sel], color=INK, label='total')
    ax[0].plot(R[sel], w.v_disc[sel], color=C1, label=f'exponential disk ({w.M_disc:.1e} M$_\\odot$)')
    ax[0].plot(R[sel], w.v_bulge[sel], color=C3, label=f'Hernquist bulge ({w.M_bulge:.1e} M$_\\odot$)')
    ax[0].plot(R[sel], w.v_halo[sel], color=C2, label='pseudo-isothermal halo')
    ax[0].axvline(2.2 * w.R_d, color=MUTED, ls=':', lw=1)
    ax[0].text(2.2 * w.R_d, 15, ' 2.2 R$_d$', color=MUTED, fontsize=8)
    ax[0].set(xlabel='R [kpc]', ylabel='v$_c$ [km/s]', title='Rotation curve decomposition')
    ax[0].legend(loc='lower right', fontsize=8)

    ax[1].semilogy(R[sel], w.Sigma[sel] / 1e6, color=C1, label='stellar disk')
    ax[1].semilogy(R[sel], gal.gas_fraction * w.Sigma[sel] / 1e6, color=C3, label='gas disk')
    ax[1].set(xlabel='R [kpc]', ylabel='Σ [M$_\\odot$ pc$^{-2}$]', title='Disk surface density')
    ax[1].legend(fontsize=8)

    ax[2].plot(R[sel], w.Omega[sel], color=INK, label='Ω = v$_c$/R')
    ax[2].plot(R[sel], w.kappa[sel], color=C1, label='κ (epicyclic)')
    ax[2].set(xlabel='R [kpc]', ylabel='frequency [km/s/kpc]', ylim=(0, 400),
              title='Orbital and epicyclic frequencies')
    ax[2].legend(fontsize=8)
    title(fig, 'Step 1: mass model (Freeman disk + Hernquist bulge + dark halo; B/T and R$_{e,bulge}$ '
               'from GALFIT/IMFIT decompositions)')
    save(fig, '01_mass_model.png')


def fig_lindblad(gal):
    w = gal.primary
    R = w.R
    sel = (R > 0.3) & (R <= 8 * w.R_d)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(R[sel], w.Omega[sel], color=INK, label='Ω')
    for wave, col in ((gal.primary, C1), (gal.secondary, C2)):
        if wave is None:
            continue
        m = wave.m
        ax.plot(R[sel], (w.Omega - w.kappa / m)[sel], color=col, lw=1.4, label=f'Ω ∓ κ/{m}  (m={m} Lindblad)')
        ax.plot(R[sel], (w.Omega + w.kappa / m)[sel], color=col, lw=1.4)
        ax.plot(R[sel], (w.Omega - w.kappa / (2 * m))[sel], color=col, lw=1.0, ls='--',
                label=f'Ω − κ/{2 * m}  (inner {2 * m}:1)')
        ax.axhline(wave.Omega_p, color=col, lw=1.2, ls='-.')
        ax.text(R[sel][-1], wave.Omega_p, f'Ω$_p$ (m={m}) = {wave.Omega_p:.1f}', color=col,
                fontsize=8, va='bottom', ha='right')
        for Rr, name in ((wave.R_ILR, 'ILR'), (wave.R_UHR, '4:1'), (wave.R_CR, 'CR'), (wave.R_OLR, 'OLR')):
            if Rr is not None and Rr < R[sel][-1]:
                ax.plot([Rr], [wave.Omega_p], 'o', ms=8, color=col, mec='white', mew=2)
                ax.annotate(name, (Rr, wave.Omega_p), textcoords='offset points', xytext=(4, 6),
                            fontsize=8, color=col)
    ax.set(xlabel='R [kpc]', ylabel='frequency [km/s/kpc]', ylim=(0, 120))
    ax.legend(fontsize=8, loc='upper right')
    title(fig, 'Step 2: resonances set by the pattern speeds\n(the bulge lifts Ω − κ/2 in the centre and '
               'creates the ILR where the arms start)')
    save(fig, '02_lindblad_resonances.png')


def fig_stability(gal):
    w = gal.primary
    R = w.R
    sel = (R > 0.3) & (R <= 8 * w.R_d)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].plot(R[sel], w.sigma[sel], color=C1, label='stellar σ$_R$ ∝ e$^{-R/2R_d}$')
    ax[0].axhline(gal.c_s, color=C3, label=f'gas c$_s$ = {gal.c_s:.0f} km/s')
    ax[0].set(xlabel='R [kpc]', ylabel='km/s', title='Velocity dispersion')
    ax[0].legend(fontsize=8)
    ax[1].plot(R[sel], w.Q[sel], color=INK)
    ax[1].axhline(1, color=MUTED, ls=':')
    ax[1].set(xlabel='R [kpc]', ylabel='Q = σκ/(πGΣ)', ylim=(0, 4), title=f'Toomre Q (min {w.Q_min:.2f})')
    for wave, col in ((gal.primary, C1), (gal.secondary, C2)):
        if wave is None:
            continue
        ax[2].plot(R[sel], wave.D[sel], color=col, label=f'm={wave.m}')
        ax[2].fill_between(R[sel], 0, 1, where=(wave.D > 1)[sel], color=col, alpha=0.12, lw=0)
    ax[2].axhline(1, color=MUTED, ls=':')
    ax[2].set(xlabel='R [kpc]', ylabel='Q²(1 − ν²)', ylim=(-1, 3),
              title='Waves need Q²(1−ν²) ≤ 1\n(shaded: forbidden zone around corotation)')
    ax[2].legend(fontsize=8)
    title(fig, 'Step 3: disk stability and the Q-barrier')
    save(fig, '03_stability_Q_barrier.png')


def fig_dispersion(gal):
    w = gal.primary
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.linspace(0, 4, 400)
    for Q, col in ((0.8, C3), (1.0, INK), (1.2, C1), (1.5, C2)):
        ax[0].plot(x, 1 - x + Q**2 * x**2 / 4, color=col, label=f'Q = {Q}')
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].fill_between(x, -0.5, 0, color=GRID, alpha=0.6, lw=0)
    ax[0].set(xlabel='|k| / k$_{crit}$', ylabel='ν² = [(ω − mΩ)/κ]²', ylim=(-0.5, 1.3),
              title='Fluid WKB dispersion relation\nν² = 1 − |k|/k$_{crit}$ + Q²(k/k$_{crit}$)²/4')
    ax[0].text(3.9, -0.25, 'ν² < 0: axisymmetric instability', ha='right', fontsize=8, color=MUTED)
    ax[0].legend(fontsize=8)

    R = w.R
    sel = (R > 0.3) & (R <= 7 * w.R_d)
    ax[1].plot(R[sel], w.k_r[sel], color=C1, label='short-wave branch (trailing arms)')
    long = np.where(w.allowed, w.k_long, np.nan)
    ax[1].plot(R[sel], long[sel], color=C1, ls='--', lw=1.2, label='long-wave branch')
    ax[1].plot(R[sel], w.k_crit[sel], color=MUTED, lw=1.2, label='k$_{crit}$ = κ²/(2πGΣ)')
    ax[1].plot(R[sel], w.k_i[sel], color=C2, label='Im k (evanescent)')
    ax[1].fill_between(R[sel], 0, 1, where=w.arm_mask[sel], color=C1, alpha=0.1, lw=0,
                       transform=ax[1].get_xaxis_transform())
    resonance_lines(ax[1], w, MUTED)
    ax[1].set(xlabel='R [kpc]', ylabel='k [kpc$^{-1}$]', ylim=(0, 6),
              title=f'Radial wavenumber of the m={w.m} wave (shaded: arms)')
    ax[1].legend(fontsize=8, loc='upper right')
    title(fig, 'Step 4: the arm shape comes from the dispersion relation')
    save(fig, '04_dispersion_relation.png')


def fig_pitch(gal):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for wave, col in ((gal.primary, C1), (gal.secondary, C2)):
        if wave is None:
            continue
        R = wave.R
        sel = (R > 0.5) & (R <= 7 * wave.R_d)
        ax[0].plot(R[sel], wave.pitch_deg[sel], color=col, lw=1, alpha=0.5)
        arm = sel & wave.arm_mask
        p_med, p_90 = wave.pitch_stats()
        turns = wave.summary()['arm_turns']
        ax[0].plot(R[arm], wave.pitch_deg[arm], color=col, lw=3,
                   label=f'm={wave.m}: median {p_med:.1f}°, 90% < {p_90:.1f}°, {turns:.1f} turns')
        ax[2].plot(R[sel], (2 * np.pi / wave.k_r)[sel], color=col, label=f'm={wave.m}')
    ax[0].axhline(15, color=MUTED, ls=':')
    ax[0].text(0.6, 15.5, 'tight-winding limit (15°)', color=MUTED, fontsize=8)
    ax[0].set(xlabel='R [kpc]', ylabel='pitch angle p [deg]', ylim=(0, 40),
              title='tan p = m / (kR)   (thick: arm region)')
    ax[0].legend(fontsize=8)
    w = gal.primary
    sel = (w.R > 0.3) & (w.R <= 7 * w.R_d)
    ax[1].plot(w.R[sel], w.shear[sel], color=INK)
    ax[1].axhline(1, color=MUTED, ls=':')
    ax[1].text(w.R[sel][-1], 1.02, 'flat rotation curve', ha='right', fontsize=8, color=MUTED)
    ax[1].set(xlabel='R [kpc]', ylabel='Γ = −dlnΩ/dlnR', title='Shear rate', ylim=(0, 1.8))
    ax[2].set(xlabel='R [kpc]', ylabel='λ = 2π/k [kpc]', title='Radial spacing between arm crossings')
    ax[2].legend(fontsize=8)
    title(fig, 'Step 5: tightly wound arms, with pitch varying with radius')
    save(fig, '05_pitch_angle_winding.png')


def fig_envelope(gal):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for wave, col in ((gal.primary, C1), (gal.secondary, C2)):
        if wave is None:
            continue
        R = wave.R
        sel = (R > 0.3) & (R <= 7 * wave.R_d)
        ax[0].plot(R[sel], wave.envelope[sel], color=col, label=f'm={wave.m} envelope')
        ax[0].plot(R[sel], wave.transmission[sel], color=col, lw=1, ls='--',
                   label=f'm={wave.m} Q-barrier transmission')
        ax[1].semilogy(R[sel], wave.mach[sel], color=col, label=f'm={wave.m}')
        ax[2].plot(R[sel], wave.contrast[sel], color=col, label=f'm={wave.m} (shocked limit {wave.arm_contrast:.1f})')
    resonance_lines(ax[0], gal.primary, MUTED)
    ax[0].set(xlabel='R [kpc]', ylabel='relative amplitude', title='Where the wave lives', ylim=(0, 1.1))
    ax[0].legend(fontsize=7, loc='upper right')
    ax[1].axhline(1, color=MUTED, ls=':')
    ax[1].text(0.4, 1.1, 'sonic', color=MUTED, fontsize=8)
    ax[1].set(xlabel='R [kpc]', ylabel='M = |u₀|/c$_s$', title='Mach number of the flow through the arms\n'
              'u₀ = m(Ω − Ω$_p$)/k')
    ax[1].legend(fontsize=8)
    ax[2].set(xlabel='R [kpc]', ylabel='peak / trough gas density',
              title='Smooth-arm contrast: shocks where supersonic,\nweak response near corotation')
    ax[2].legend(fontsize=8)
    title(fig, 'Step 6: arm extent and gas shocks')
    save(fig, '06_envelope_and_shocks.png')


def fig_geometry(gal):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    lim = 4.6 * gal.R_d
    x = np.linspace(-lim, lim, 400)
    X, Y = np.meshgrid(x, x, indexing='ij')
    base = np.exp(-np.hypot(X, Y) / gal.R_d) * gal.arm_factor_on_grid(X, Y)
    ax[0].imshow(np.log10(base.T + 1e-4), extent=[-lim, lim, -lim, lim], cmap='Greys', vmin=-2.5, alpha=0.6)
    t = np.linspace(0, 2 * np.pi, 300)
    for mode, (wave, col) in enumerate(((gal.primary, C1), (gal.secondary, C2))):
        if wave is None:
            continue
        for j, (cx, cy) in enumerate(gal.arm_crests(mode=mode)):
            lw = 2.2 if mode == 0 else 1.4
            ax[0].plot(cx, cy, color=col, lw=lw, label=f'm={wave.m} crests' if j == 0 else None)
            ax[1].plot(np.log(np.hypot(cx, cy)), np.degrees(np.unwrap(np.arctan2(cy, cx))),
                       color=col, lw=lw, label=f'm={wave.m}' if j == 0 else None)
        for Rr, name in ((wave.R_ILR, 'ILR'), (wave.R_CR, 'CR')):
            if Rr is not None and Rr < lim:
                ax[0].plot(Rr * np.cos(t), Rr * np.sin(t), color=col, lw=0.8, ls=':')
                ax[0].text(Rr / np.sqrt(2), Rr / np.sqrt(2), f' {name}', color=col, fontsize=8)
    ax[0].annotate('', xy=(0.7 * lim, -0.1 * lim), xytext=(0.7 * lim, 0.1 * lim),
                   arrowprops=dict(arrowstyle='->', color=INK, lw=1.5))
    ax[0].text(0.73 * lim, 0, 'rotation', fontsize=8, color=INK, va='center')
    ax[0].set(xlabel='x [kpc]', ylabel='y [kpc]', aspect='equal', title='Arm crests (χ = 2πj): trailing, ~1 turn')
    ax[0].grid(False)
    ax[0].legend(fontsize=8, loc='lower left')
    ax[1].set(xlabel='ln R [kpc]', ylabel='crest azimuth [deg, unwrapped]',
              title='A logarithmic spiral would be a straight line:\ncurvature = pitch varying with R')
    ax[1].legend(fontsize=8)
    title(fig, 'Step 7: arm geometry')
    save(fig, '07_arm_geometry.png')


def fig_streaming_profile(gal):
    w = gal.primary
    R0 = 2.0 * w.R_d
    chi = np.linspace(-np.pi, np.pi, 721)
    R = np.full_like(chi, R0)
    phi = (chi - w._at(w.phase, R)) / w.m + w.phase0
    rho = w.density(R, phi)
    v_R, v_phi = w.streaming(R, phi)
    A1 = 2 * np.mean(rho[:-1] * np.cos(chi[:-1]))          # first Fourier harmonic of ρ
    Om, k, kap = w._at(w.Omega, R0), w._at(w.k_r, R0), w._at(w.kappa, R0)
    lin_R = w.m * (w.Omega_p - Om) / k * A1 * np.cos(chi)
    lin_phi = kap**2 / (2 * Om * k) * A1 * np.sin(chi)
    deg = np.degrees(chi)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), sharex=True)
    ax[0].plot(deg, rho, color=C3, label='skewed shock profile ρ(χ)')
    ax[0].plot(deg, 1 + A1 * np.cos(chi), color=MUTED, ls='--', lw=1.2, label='linear (first harmonic)')
    ax[0].annotate('shock\n(upstream)', xy=(-25, rho[np.argmin(np.abs(deg + 25))]), xytext=(-150, rho.max() * 0.85),
                   arrowprops=dict(arrowstyle='->', color=MUTED), fontsize=8, color=INK)
    ax[0].annotate('post-shock\ndecline', xy=(70, rho[np.argmin(np.abs(deg - 70))]), xytext=(95, rho.max() * 0.8),
                   arrowprops=dict(arrowstyle='->', color=MUTED), fontsize=8, color=INK)
    ax[0].set(ylabel='Σ$_{gas}$ / ⟨Σ$_{gas}$⟩', title=f'Gas density across the arm at R = {R0:.1f} kpc')
    ax[1].plot(deg, v_R, color=C1, label='u₀(1/ρ − 1)  (mass flux)')
    ax[1].plot(deg, lin_R, color=MUTED, ls='--', lw=1.2, label='linear Lin–Shu')
    ax[1].set(ylabel='v$_R$ [km/s]', title='Radial streaming')
    ax[2].plot(deg, v_phi, color=C2, label='κ²/(2Ωk) ∫(ρ−1)dχ  (ang. mom.)')
    ax[2].plot(deg, lin_phi, color=MUTED, ls='--', lw=1.2, label='linear Lin–Shu')
    ax[2].set(ylabel='v$_φ$ − v$_c$ [km/s]', title='Azimuthal streaming')
    for a in ax:
        a.axvline(0, color=MUTED, lw=0.8, ls=':')
        a.set_xlabel('arm phase χ [deg]  (0 = crest; gas flows towards +χ)')
        a.legend(fontsize=8, loc='lower left')
    title(fig, 'Step 8: shocked-arm profile and the streaming it implies (inside corotation: gas slows '
               'and flows inward in the arm)')
    save(fig, '08_arm_profile_streaming.png')


def fig_knots(gal):
    k = gal.complexes
    lim = 4.6 * gal.R_d
    x = np.linspace(-lim, lim, 600)
    X, Y = np.meshgrid(x, x, indexing='ij')
    total, arms, turb, spurs, clumps = gal.emissivity_on_grid(X, Y, return_parts=True)
    base = np.exp(-np.hypot(X, Y) / gal.R_d)
    fig, ax = plt.subplots(1, 5, figsize=(26, 5))
    img = base * arms * spurs * (1 + clumps)
    im = sky(ax[0], img, [-lim, lim, -lim, lim],
             f'{gal.n_complexes} complexes ({gal.knots["x"].size} sub-clumps), {gal.n_spurs} spurs',
             cmap=SEQ, norm=LogNorm(vmin=3e-3, vmax=img.max()))
    colorbar(fig, im, ax[0], 'relative emission (no turbulence)')
    # Zoom on one complex and its spur
    i = int(np.argmax(k['L']))
    zx, zy = k['x'][i], k['y'][i]
    im = sky(ax[4], img, [-lim, lim, -lim, lim], 'Zoom: brightest complex (hierarchical sub-clumps)',
             cmap=SEQ, norm=LogNorm(vmin=img.max() * 3e-3, vmax=img.max()))
    ax[4].set(xlim=(zx - 1.5, zx + 1.5), ylim=(zy - 1.5, zy + 1.5))

    L = np.sort(k['L'])[::-1]
    ax[1].loglog(L, np.arange(1, L.size + 1), 'o', ms=4, color=C1, label='sampled knots')
    L_lo, L_hi = 1.0, 20.0
    Lg = np.logspace(0, np.log10(L_hi), 100)
    ax[1].loglog(Lg, L.size * (1 / Lg - 1 / L_hi) / (1 / L_lo - 1 / L_hi), color=MUTED, ls='--', lw=1.2,
                 label='dN/dL ∝ L$^{-2}$ on [1, 20] L$_{min}$')
    ax[1].set(xlabel='complex luminosity L / L$_{min}$', ylabel='N(> L)', title='HII-region luminosity function')
    ax[1].legend(fontsize=8)

    d = np.hypot(np.diff(k['x']), np.diff(k['y']))
    Rm = 0.5 * (k['R'][1:] + k['R'][:-1])
    ok = d < 2.5 * gal.jeans_spacing(Rm)            # consecutive knots on the same arm
    Rs = np.linspace(k['R'].min(), k['R'].max(), 200)
    ax[2].plot(Rm[ok], d[ok], 'o', ms=4, color=C1, label='spacing between neighbouring complexes')
    ax[2].plot(Rs, gal.jeans_spacing(Rs), color=INK, label='λ = 2c$_s$²/(GΣ$_{gas,arm}$)')
    ax[2].set(xlabel='R [kpc]', ylabel='spacing along the arm [kpc]', title='"Beads on a string": Jeans fragmentation')
    ax[2].legend(fontsize=8)

    w = gal.primary
    Rs = np.linspace(w.R[w.arm_mask].min(), min(w.R[w.arm_mask].max(), 0.95 * w.R_CR), 200)
    ax[3].plot(Rs, w._at(w.pitch_deg, Rs), color=C1, label='arm pitch')
    ax[3].plot(Rs, gal.spur_pitch_deg(Rs), color=C2, label='spur pitch: tan p = 1/(Γ Ω Δt)')
    ax[3].set(xlabel='R [kpc]', ylabel='pitch [deg]', ylim=(0, 65),
              title='Spurs: knots sheared downstream\nfor a quarter interarm crossing time')
    ax[3].legend(fontsize=8)
    title(fig, 'Step 9: star-forming complexes ("beads on a string", just downstream of the shock) and '
               'the spurs sheared out of them')
    save(fig, '09_knots_and_spurs.png')


def fig_turbulence(gal):
    n, dx = 512, 0.05
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    f = gal.turbulence_on_grid(X, Y)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    im = sky(ax[0], f, [x[0], x[-1], x[0], x[-1]], f'Turbulent factor (σ$_{{ln Σ}}$ = {gal.turb_sigma:.2f})',
             cmap=SEQ, norm=LogNorm(vmin=0.2, vmax=5))
    colorbar(fig, im, ax[0], 'Σ$_{gas}$ / ⟨Σ$_{gas}$⟩')
    lnf = np.log(f).ravel()
    ax[1].hist(lnf, bins=80, density=True, color=C1, edgecolor='white', linewidth=0.5)
    s = gal.turb_sigma
    g = np.linspace(lnf.min(), lnf.max(), 200)
    ax[1].plot(g, np.exp(-0.5 * ((g + s**2 / 2) / s) ** 2) / (s * np.sqrt(2 * np.pi)), color=INK, lw=1.4,
               label='lognormal (supersonic turbulence)')
    ax[1].set(xlabel='ln (Σ / ⟨Σ⟩)', ylabel='PDF', title='Density PDF')
    ax[1].legend(fontsize=8)
    P = np.abs(np.fft.fft2(np.log(f))) ** 2
    kx = np.fft.fftfreq(n, d=dx)
    K = np.hypot(kx[:, None], kx[None, :]).ravel()
    edges = np.logspace(np.log10(kx[1]), np.log10(np.abs(kx).max()), 30)
    idx = np.digitize(K, edges)
    Pk = np.array([P.ravel()[idx == i].mean() for i in range(1, len(edges))])
    kc = np.sqrt(edges[1:] * edges[:-1])
    ax[2].loglog(kc, Pk, 'o', ms=4, color=C1, label='measured')
    from songs.spiral import TURB_THICKNESS_KPC as h
    ref = (kc**2 + (1 / 3.0) ** 2) ** (-2.7 / 2) / (1 + kc**2 * h**2)
    ax[2].loglog(kc, ref * Pk[3] / ref[3], color=INK, lw=1.2, ls='--',
                 label='k$^{-2.7}$ → k$^{-3.7}$ below the disk thickness\n(LMC HI, Elmegreen+ 2001)')
    from songs.spiral import TURB_DRIVING_KPC as L0
    for kk, lab in ((1 / L0, f'driving ({L0:.0f} kpc)'), (1 / h, f'disk thickness ({h} kpc)')):
        ax[2].axvline(kk, color=MUTED, ls=':')
        ax[2].text(kk, 0.05, ' ' + lab, transform=ax[2].get_xaxis_transform(), fontsize=8, color=MUTED)
    ax[2].set(xlabel='k [kpc$^{-1}$]', ylabel='P(k)', title='Power spectrum')
    ax[2].legend(fontsize=8)
    title(fig, 'Step 10: turbulent ISM: a lognormal field with the observed power spectrum')
    save(fig, '10_turbulence.png')


def fig_center(gal, stage):
    w = gal.primary
    R = w.R
    sel = R <= 2.5 * w.R_d
    ax_kpc = stage['axis_kpc']
    X, Y, Z = np.meshgrid(ax_kpc, ax_kpc, np.linspace(-1.5, 1.5, 31), indexing='ij')
    disk = np.exp(-np.hypot(X, Y) / gal.R_d) * np.exp(-np.abs(Z) / HZ_KPC)
    total, disk_s, center, ring = gal.emission_cube(X, Y, Z, disk, return_parts=True)
    edges = np.arange(0, 3 * gal.R_d, 1.5 * (ax_kpc[1] - ax_kpc[0]))
    rc = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    ax[0].plot(R[sel], w.v_c[sel], color=INK, label='total')
    ax[0].plot(R[sel], w.v_bulge[sel], color=C3, label='bulge')
    ax[0].plot(R[sel], w.v_disc[sel], color=C1, label='disk')
    ax[0].plot(R[sel], w.v_halo[sel], color=C2, label='halo')
    if w.R_ILR is not None:
        ax[0].axvline(w.R_ILR, color=MUTED, ls=':')
        ax[0].text(w.R_ILR, 20, ' ILR', fontsize=8, color=MUTED)
    ax[0].set(xlabel='R [kpc]', ylabel='v$_c$ [km/s]', title='Inner rotation curve: the bulge dominates')
    ax[0].legend(fontsize=8)
    for comp, col, lab in ((total, INK, 'total'), (disk_s, C1, 'disk (arms, knots)'),
                           (center, C3, f'central concentration ({gal.center_fraction:.0%})'),
                           (ring, C2, f'ring at the ILR ({gal.ring_fraction:.0%})')):
        ax[1].semilogy(rc, radial_profile(comp.sum(2), ax_kpc, edges), color=col, label=lab,
                       lw=2.4 if col == INK else 1.6)
    ax[1].set(xlabel='R [kpc]', ylabel='azimuthally averaged emission', ylim=(1e-4 * total.sum(2).max(), None),
              title='Radial emission profile by component')
    ax[1].legend(fontsize=8)
    S = total.sum(2)
    im = sky(ax[2], S, [ax_kpc[0], ax_kpc[-1]] * 2, 'Face-on centre', cmap=SEQ,
             norm=LogNorm(vmin=S.max() * 1e-3, vmax=S.max()))
    zoom(ax[2], 1.6 * gal.R_d)
    colorbar(fig, im, ax[2], 'emission')
    ax[3].plot(rc, gal.dispersion_on_grid(rc, 0 * rc), color=C1, label='σ$_{gas}$(R)')
    ax[3].axhline(gal.c_s, color=MUTED, ls=':', label=f'disk c$_s$ = {gal.c_s:.0f} km/s')
    ax[3].set(xlabel='R [kpc]', ylabel='km/s', ylim=(0, None),
              title='Gas dispersion rises in the centre\nσ² = c$_s$² + σ$_c$² exp(−(R/R$_{e,b}$)²)')
    ax[3].legend(fontsize=8)
    title(fig, 'Step 11: the galaxy centre: bulge gravity, inflowing gas piling up at the ILR ring and '
               'in a central concentration')
    save(fig, '11_bulge_and_center.png')


# ---------------------------------------------------------------------------
# 12-18: the 3D grid and the rotation into the observer frame
# ---------------------------------------------------------------------------

def fig_grid_3d(stage):
    ax_kpc = stage['axis_kpc']
    disk = stage['disk']
    n = ax_kpc.size
    fig = plt.figure(figsize=(16, 6.6))
    a = fig.add_subplot(1, 2, 1, projection='3d')
    step = max(n // 10, 1)
    idx = np.arange(0, n, step)
    gx, gy, gz = np.meshgrid(ax_kpc[idx], ax_kpc[idx], ax_kpc[idx], indexing='ij')
    a.scatter(gx.ravel(), gy.ravel(), gz.ravel(), s=1.5, color=MUTED, alpha=0.35, depthshade=False)
    thr = np.percentile(disk, 99.7)
    ii, jj, kk = np.nonzero(disk > thr)
    sub = np.random.RandomState(0).choice(ii.size, min(ii.size, 15000), replace=False)
    sc = a.scatter(ax_kpc[ii[sub]], ax_kpc[jj[sub]], ax_kpc[kk[sub]],
                   c=np.log10(disk[ii[sub], jj[sub], kk[sub]]), s=2, cmap=SEQ, depthshade=False)
    a.set(xlabel='x [kpc]', ylabel='y [kpc]', zlabel='z [kpc]', zlim=(ax_kpc[0], ax_kpc[-1]))
    a.set_title(f'{n}³ Cartesian voxel grid, {ax_kpc[1] - ax_kpc[0]:.2f} kpc/voxel\n'
                f'(grey: every {step}th voxel; colour: brightest 0.3% of the emissivity)')
    a.view_init(elev=28, azim=-60)
    colorbar(fig, sc, a, 'log emissivity per voxel')

    b = fig.add_subplot(1, 2, 2)
    c = n // 2
    nz = int(3 / (ax_kpc[1] - ax_kpc[0]))
    zz = slice(c - nz, c + nz)
    im = b.imshow(np.log10(disk[:, c, zz].T + 1e-12), extent=[ax_kpc[0], ax_kpc[-1], ax_kpc[zz][0], ax_kpc[zz][-1]],
                  cmap=SEQ, aspect='auto', vmin=np.log10(disk.max()) - 4)
    b.set(xlabel='x [kpc]', ylabel='z [kpc]',
          title=f'Edge-on cut (y = 0): thin disk (h$_z$ = {HZ_KPC} kpc) + thicker central concentration')
    b.grid(False)
    colorbar(fig, im, b, 'log emissivity')
    title(fig, 'Step 12: the 3D Cartesian grid in the galaxy frame')
    save(fig, '12_cartesian_grid_3d.png')


def fig_slices(stage):
    ax_kpc = stage['axis_kpc']
    disk = stage['disk']
    n = ax_kpc.size
    c = n // 2
    ext = [ax_kpc[0], ax_kpc[-1], ax_kpc[0], ax_kpc[-1]]
    vmin = disk.max() * 1e-4
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.8))
    norm = LogNorm(vmin=vmin, vmax=disk.max())
    for a, img, ttl, yl in ((ax[0], disk[:, :, c], 'Midplane (z = 0)', 'y'),
                            (ax[1], disk[:, :, c + 3], f'z = {ax_kpc[c + 3]:.1f} kpc', 'y'),
                            (ax[2], disk[:, c, :], 'y = 0 slice', 'z')):
        im = a.imshow(np.maximum(img, vmin).T, extent=ext, cmap=SEQ, norm=norm)
        a.set(xlabel='x [kpc]', ylabel=f'{yl} [kpc]', title=ttl)
        a.grid(False)
        zoom(a)
    ax[2].set_ylim(-4, 4)
    colorbar(fig, im, ax[2], 'emissivity per voxel')
    i, j = np.unravel_index(disk[:, :, c].argmax(), disk[:, :, c].shape)
    r_on = np.hypot(ax_kpc[i], ax_kpc[j])
    ax[3].semilogy(ax_kpc, disk[i, j, :] / disk[i, j, c], color=C1, label=f'brightest column (R = {r_on:.1f} kpc)')
    k = np.argmin(np.abs(ax_kpc - 6.0))
    ax[3].semilogy(ax_kpc, disk[k, c, :] / disk[k, c, c], color=C3, label='disk column (R = 6 kpc)')
    ax[3].semilogy(ax_kpc, np.exp(-np.abs(ax_kpc) / HZ_KPC), color=MUTED, ls='--', lw=1.2,
                   label=f'exp(−|z|/{HZ_KPC} kpc)')
    ax[3].set(xlim=(-2, 2), ylim=(1e-3, 1.5), xlabel='z [kpc]', ylabel='relative emissivity', title='Vertical profiles')
    ax[3].legend(fontsize=8)
    title(fig, 'Step 13: slices through the 3D emissivity cube')
    save(fig, '13_emissivity_slices.png')


def fig_buildup(gal, stage):
    ax_kpc = stage['axis_kpc']
    ext = [ax_kpc[0], ax_kpc[-1]] * 2
    X, Y = np.meshgrid(ax_kpc, ax_kpc, indexing='ij')
    total2d, arms, turb, spurs, clumps = gal.emissivity_on_grid(X, Y, return_parts=True)
    disk = np.exp(-np.hypot(X, Y) / gal.R_d)
    S = stage['disk'].sum(2)
    steps = ((disk, '1. axisymmetric exponential disk'),
             (disk * arms, '2. + shocked density-wave arms'),
             (disk * arms * turb, '3. + turbulent ISM'),
             (disk * arms * turb * spurs, '4. + spurs (feathers)'),
             (disk * arms * turb * spurs * (1 + clumps), '5. + star-forming complexes'),
             (S, '6. + central concentration and ILR ring\n(final 3D cube, summed along z)'))
    fig, ax = plt.subplots(1, 6, figsize=(30, 5.4))
    for a, (img, ttl) in zip(ax, steps):
        img = img / img.max()
        im = sky(a, img, ext, ttl, cmap=SEQ, norm=LogNorm(vmin=2e-4, vmax=1))
        zoom(a)
    colorbar(fig, im, ax[-1], 'relative surface brightness')
    title(fig, 'Step 14: building up the face-on emission, component by component')
    save(fig, '14_face_on_buildup.png')


def fig_face_on(stage, stage_twin):
    ax_kpc = stage['axis_kpc']
    ext = [ax_kpc[0], ax_kpc[-1]] * 2
    S = stage['disk'].sum(2)
    T = stage_twin['disk'].sum(2)
    vmin = S.max() * 3e-4
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    norm = LogNorm(vmin=vmin, vmax=S.max())
    im = sky(ax[0], np.maximum(S, vmin), ext, 'Spiral: face-on surface brightness', cmap=SEQ, norm=norm)
    sky(ax[1], np.maximum(T, vmin), ext, 'Axisymmetric twin (same mass model, centre and ring)',
        cmap=SEQ, norm=norm)
    colorbar(fig, im, ax[1], 'flux per pixel')
    ratio = np.log10(np.maximum(S, vmin) / np.maximum(T, vmin))
    im = sky(ax[2], ratio, ext, 'log(spiral / twin): flux moved into arms and knots', cmap=DIV,
             norm=TwoSlopeNorm(0, -1, 1))
    colorbar(fig, im, ax[2], 'dex')
    for a in ax:
        zoom(a)
    title(fig, 'Step 15: face-on view (sum along z)')
    save(fig, '15_spiral_vs_twin.png')


def fig_velocity_field(stage, gal):
    ax_kpc = stage['axis_kpc']
    X, Y = np.meshgrid(ax_kpc, ax_kpc, indexing='ij')
    vx, vy = stage['vel_x_2d'], stage['vel_y_2d']
    vx0, vy0 = gal.velocity_on_grid(X, Y, perturbed=False)
    R = np.hypot(X, Y) + 1e-9
    v_R = vx * X / R + vy * Y / R
    dv_phi = (vx - vx0) * Y / R - (vy - vy0) * X / R      # along the SONGS rotation (y, -x)
    ext = [ax_kpc[0], ax_kpc[-1]] * 2
    S = stage['disk'].sum(2)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    lim = 40
    im = sky(ax[0], v_R, ext, 'Radial streaming v$_R$', cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    colorbar(fig, im, ax[0], 'km/s')
    im = sky(ax[1], dv_phi, ext, 'Azimuthal streaming v$_φ$ − v$_c$', cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim))
    colorbar(fig, im, ax[1], 'km/s')
    sky(ax[2], np.log10(np.maximum(S, S.max() * 3e-4)), ext, 'Streaming + turbulent motions (v − v$_{circ}$)',
        cmap='Greys')
    sl = (slice(None, None, 6), slice(None, None, 6))
    ax[2].quiver(X[sl], Y[sl], (vx - vx0)[sl], (vy - vy0)[sl], color=C2, scale=900, width=0.003)
    for a in ax:
        zoom(a)
        for cx, cy in gal.arm_crests():
            a.plot(cx, cy, color=INK, lw=0.6, alpha=0.6)
    title(fig, 'Step 16: velocity field in the disk plane (black lines: arm crests)')
    save(fig, '16_velocity_field_face_on.png')


def fig_rotation_3d(stage):
    ax_kpc = stage['axis_kpc']
    rng = np.random.RandomState(1)
    fig = plt.figure(figsize=(16, 6.6))
    vx = stage['vel_x_2d']
    panels = ((stage['disk'], np.repeat(vx[:, :, None], ax_kpc.size, axis=2),
               'Galaxy frame (colour: v$_x$, which tilts into the LOS)'),
              (stage['rotated_disk'], stage['rotated_vel_los'],
               f'Observer frame after the {INCLINATION:.0f}° rotation (colour: v$_{{LOS}}$)'))
    for col, (disk, vel, ttl) in enumerate(panels):
        a = fig.add_subplot(1, 2, col + 1, projection='3d')
        ii, jj, kk = np.nonzero(disk > np.percentile(disk, 99.6))
        sub = rng.choice(ii.size, min(ii.size, 15000), replace=False)
        sc = a.scatter(ax_kpc[ii[sub]], ax_kpc[jj[sub]], ax_kpc[kk[sub]], c=vel[ii[sub], jj[sub], kk[sub]],
                       cmap=DIV, vmin=-250, vmax=250, s=2, depthshade=False)
        a.set(xlabel='x [kpc]', ylabel='y [kpc]', zlabel='z = LOS [kpc]', xlim=(-VIEW, VIEW), ylim=(-VIEW, VIEW),
              zlim=(-VIEW, VIEW))
        a.set_title(ttl)
        a.view_init(elev=22, azim=-55)
        colorbar(fig, sc, a, 'km/s')
    title(fig, 'Step 17: rotating the emissivity and velocity vectors into the observer frame')
    save(fig, '17_rotation_to_observer_frame.png')


def fig_los_velocity(stage, stage_twin):
    ax_kpc = stage['axis_kpc']
    ext = [ax_kpc[0], ax_kpc[-1]] * 2

    def mom1(st):
        f = st['rotated_disk']
        W = f.sum(2)
        m1 = (f * st['rotated_vel_los']).sum(2) / np.where(W > 0, W, 1)
        return np.where(W > W.max() * 1e-3, m1, np.nan), W

    m1, W = mom1(stage)
    m1t, _ = mom1(stage_twin)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))
    levels = np.arange(-250, 251, 25)
    for a, img, ttl in ((ax[0], m1, 'Spiral: intensity-weighted v$_{LOS}$'), (ax[1], m1t, 'Axisymmetric twin')):
        im = sky(a, img, ext, ttl, cmap=DIV, vmin=-220, vmax=220)
        a.contour(ax_kpc, ax_kpc, np.nan_to_num(img).T, levels=levels, colors=INK, linewidths=0.5)
    colorbar(fig, im, ax[1], 'km/s')
    im = sky(ax[2], m1 - m1t, ext, 'Residual: spiral − twin (streaming + turbulence)', cmap=DIV, norm=TwoSlopeNorm(0, -30, 30))
    ax[2].contour(ax_kpc, ax_kpc, np.log10(np.maximum(W, W.max() * 1e-3)).T, levels=6, colors=INK,
                  linewidths=0.4, alpha=0.6)
    colorbar(fig, im, ax[2], 'km/s')
    for a in ax:
        zoom(a)
    title(fig, f'Step 18: line-of-sight velocity at i = {INCLINATION:.0f}° (before channel binning and beam); '
               'isovelocity contours kink across the arms')
    save(fig, '18_los_velocity_residual.png')


# ---------------------------------------------------------------------------
# 19-21: the spectral cube
# ---------------------------------------------------------------------------

def fig_channels(cube, params):
    v = params['average_vels']
    ext = [-FOV_KPC / 2, FOV_KPC / 2] * 2
    spec = cube.sum((1, 2))
    on = np.nonzero(spec > spec.max() * 0.02)[0]
    chans = np.linspace(on[0], on[-1], 16).round().astype(int)
    vmax = np.percentile(cube[chans], 99.9)
    fig, axs = plt.subplots(4, 4, figsize=(14, 13.5), sharex=True, sharey=True)
    for a, c in zip(axs.flat, chans):
        im = a.imshow(cube[c].T, extent=ext, cmap=SEQ, vmin=0, vmax=vmax)
        a.text(0.04, 0.92, f'{v[c]:+.0f} km/s', transform=a.transAxes, color='white', fontsize=9)
        zoom(a)
        a.grid(False)
    for a in axs[-1]:
        a.set_xlabel('x [kpc]')
    for a in axs[:, 0]:
        a.set_ylabel('y [kpc]')
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01, label='flux per pixel per channel')
    title(fig, f'Step 19: channel maps ({CHAN_KMS:.0f} km/s channels, {BEAM_KPC} kpc beam): arms and knots '
               'trace the butterfly pattern')
    save(fig, '19_channel_maps.png')


def fig_moments(cube, params, cube_twin, params_twin):
    v = params['average_vels']
    ext = [-FOV_KPC / 2, FOV_KPC / 2] * 2
    m0, m1, m2, good = masked_moments(cube, v, rms=cube.max() * 1e-3)
    fig, ax = plt.subplots(1, 4, figsize=(23, 5))
    im = sky(ax[0], np.where(good, m0, np.nan), ext, 'Moment 0', cmap=SEQ,
             norm=LogNorm(vmin=np.nanmax(m0) * 1e-3, vmax=np.nanmax(m0)))
    colorbar(fig, im, ax[0], 'flux · km/s')
    im = sky(ax[1], m1, ext, 'Moment 1', cmap=DIV, vmin=-220, vmax=220)
    colorbar(fig, im, ax[1], 'km/s')
    im = sky(ax[2], m2, ext, f'Moment 2 (median {np.nanmedian(m2):.0f} km/s)', cmap=WIDTH, vmin=0, vmax=50)
    colorbar(fig, im, ax[2], 'km/s')
    for a in ax[:3]:
        zoom(a)
    ax[3].plot(v, cube.sum((1, 2)), color=C1, label='spiral')
    ax[3].plot(params_twin['average_vels'], cube_twin.sum((1, 2)), color=MUTED, lw=1.2, ls='--',
               label='axisymmetric twin')
    ax[3].set(xlabel='v$_{LOS}$ [km/s]', ylabel='integrated flux', title='Integrated spectrum', xlim=(-350, 350))
    ax[3].legend(fontsize=8)
    title(fig, 'Step 20: moments of the final (beam-convolved) cube')
    save(fig, '20_moment_maps_spectrum.png')


def fig_pv(cube, params, cube_twin):
    v = params['average_vels']
    n = cube.shape[1]
    c = n // 2
    y = (np.arange(n) - (n - 1) / 2) * PIX_KPC
    band = slice(c - 1, c + 2)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for a, cb, ttl in ((ax[0], cube, 'Spiral'), (ax[1], cube_twin, 'Axisymmetric twin')):
        pv = cb[:, band, :].sum(1)
        im = a.imshow(pv, extent=[y[0], y[-1], v[0], v[-1]], aspect='auto', cmap=SEQ,
                      norm=LogNorm(vmin=pv.max() * 1e-3, vmax=pv.max()))
        a.set(xlabel='offset along the major axis [kpc]', title=ttl, xlim=(-16, 16), ylim=(-320, 320))
        a.grid(False)
    ax[0].set_ylabel('v$_{LOS}$ [km/s]')
    colorbar(fig, im, ax[1], 'flux')
    title(fig, 'Step 21: position–velocity diagram along the major axis: steep bulge-driven inner rise, '
               'arms as bright kinked segments')
    save(fig, '21_pv_diagram.png')


# ---------------------------------------------------------------------------
# 22-23: population view
# ---------------------------------------------------------------------------

def build_gallery():
    """Sample spirals exactly as SONGS(morphology='spiral') does and build each one."""
    rng = np.random.RandomState(GALLERY_SEED)
    systems = []
    for i, incl in enumerate(GALLERY_INCLINATIONS):
        p = sample_spiral_params(5.0 / 1.678, 200.0, rng=rng)
        cube, params, stage = build(p, f'gallery_{i + 1:02d}', inclination=incl, pix=0.25)
        save_system(f'gallery_{i + 1:02d}', cube, params, stage)
        systems.append((cube, params, stage))
    return systems


def fig_gallery(systems):
    n = len(systems)
    fig, axs = plt.subplots(3, n, figsize=(3.4 * n, 10.6))
    for col, (cube, params, stage) in enumerate(systems):
        sp = params['spiral']
        ax_kpc = stage['axis_kpc']
        ext = [ax_kpc[0], ax_kpc[-1]] * 2
        S = stage['disk'].sum(2)
        m2txt = '' if np.isnan(sp['m2']) else f'+{int(sp["m2"])}'
        sky(axs[0, col], S / S.max(), ext, f'm={int(sp["m"])}{m2txt}, p={sp["pitch_median_deg"]:.0f}°, '
            f'B/T={sp["bulge_fraction"]:.2f}', xy=False, cmap=SEQ, norm=LogNorm(vmin=5e-4, vmax=1))
        v = params['average_vels']
        m0, m1, _, good = masked_moments(cube, v, rms=cube.max() * 1e-3)
        sky(axs[1, col], np.where(good, m0, np.nan), ext, f'i = {params["inclination_deg"]:.0f}°: moment 0',
            xy=False, cmap=SEQ, norm=LogNorm(vmin=np.nanmax(m0) * 1e-3, vmax=np.nanmax(m0)))
        sky(axs[2, col], m1, ext, f'moment 1 (|v$_R$| ≤ {sp["v_R_stream_max_km_s"]:.0f} km/s)', xy=False,
            cmap=DIV, vmin=-250, vmax=250)
        for a in axs[:, col]:
            zoom(a)
            a.set_xticks([])
            a.set_yticks([])
    axs[0, 0].set_ylabel('face-on emission')
    axs[1, 0].set_ylabel('observed cube')
    title(fig, 'Step 22: spirals sampled as SONGS(morphology="spiral") does (top: intrinsic face-on emission; '
               'middle/bottom: moments of the inclined, beam-convolved cube). Saved in spiral_galaxies/systems/')
    save(fig, '22_gallery_sampled_spirals.png')


def fig_distributions(n=300):
    rng = np.random.RandomState(0)
    rows = [SpiralGalaxy(**sample_spiral_params(2.98, 200.0, rng=rng)).summary() for _ in range(n)]

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    fig, ax = plt.subplots(2, 4, figsize=(20, 8.4))
    ms, m2 = col('m'), col('m2')
    lab = ['m=1', 'm=2', 'm=3']
    ax[0, 0].bar(lab, [np.sum(ms == k) for k in (1, 2, 3)], color=C1, width=0.6)
    ax[0, 0].set(title='Arms of the main mode (after physical cuts)', ylabel='count')
    ax[0, 1].bar(['none'] + lab, [np.sum(np.isnan(m2))] + [np.sum(m2 == k) for k in (1, 2, 3)], color=C2, width=0.6)
    ax[0, 1].set(title='Second mode')
    panels = ((ax[0, 2], 'pitch_median_deg', 'median pitch [deg]'),
              (ax[0, 3], 'arm_turns', 'arm winding [turns]'),
              (ax[1, 0], 'R_arms_in_kpc', 'arm start (ILR) [kpc]'),
              (ax[1, 1], 'v_R_stream_max_km_s', 'max |v$_R$| streaming [km/s]'),
              (ax[1, 2], 'v_phi_stream_max_km_s', 'max |v$_φ$| streaming [km/s]'),
              (ax[1, 3], 'n_knots', 'number of knots'))
    for a, k, label in panels:
        a.hist(col(k), bins=25, color=C1, edgecolor='white', linewidth=1)
        a.set(xlabel=label, ylabel='count')
    ax[0, 2].axvline(15, color=MUTED, ls=':')
    title(fig, f'Step 23: distributions over {n} sampled galaxies (R$_d$ = 3 kpc, v$_0$ = 200 km/s)')
    save(fig, '23_parameter_distributions.png')


# ---------------------------------------------------------------------------
# 24-25: NGC 7469
# ---------------------------------------------------------------------------

def load_ngc7469(path, half=300):
    from astropy.io import fits
    with fits.open(path, memmap=True) as hd:
        h = hd[0].header
        data = np.nan_to_num(np.asarray(hd[0].data[0], dtype=np.float32))
    nu = h['CRVAL3'] + (np.arange(h['NAXIS3']) + 1 - h['CRPIX3']) * h['CDELT3']
    v = 299792.458 * (h['RESTFRQ'] - nu) / h['RESTFRQ']
    order = np.argsort(v)
    v, data = v[order], data[order]
    free = np.r_[0:35, v.size - 35:v.size]
    m0 = data[35:-35].sum(0)
    cy, cx = np.unravel_index(np.argmax(gaussian_filter(m0, 10)), m0.shape)
    data = data[:, cy - half:cy + half, cx - half:cx + half]
    m0, m1, m2, good = masked_moments(data, v, free=free)
    vsys = np.nansum(m0 * m1) / np.nansum(m0)
    kpc_per_px = abs(h['CDELT1']) * 3600 * 0.339       # 1" = 0.339 kpc at D = 70 Mpc
    return dict(cube=data, v=v - vsys, m0=m0, m1=m1 - vsys, m2=m2, good=good, kpc_per_px=kpc_per_px)


def fig_ngc7469(ngc, cube, params):
    v = params['average_vels']
    noisy, _ = add_noise(cube, BEAM_KPC / PIX_KPC, peak_snr=60)
    noisy = noisy.transpose(0, 2, 1)          # (v, y, x), the FITS layout
    free = np.nonzero(np.abs(v) > 350)[0]
    sm0, sm1, sm2, sgood = masked_moments(noisy, v, free=free)
    half_ngc = ngc['m0'].shape[0] / 2 * ngc['kpc_per_px']
    ext_ngc = [-half_ngc, half_ngc] * 2
    ext_sim = [-FOV_KPC / 2, FOV_KPC / 2] * 2
    fig, ax = plt.subplots(2, 4, figsize=(23, 10.5))
    rows = ((ngc['m0'], ngc['m1'], ngc['m2'], ngc['good'], ext_ngc, 'NGC 7469, ALMA CO(2−1)',
             ngc['v'], ngc['cube'].sum((1, 2)), 4.0),
            (sm0, sm1, sm2, sgood, ext_sim, 'SONGS spiral (+ noise, peak S/N 60)', v, noisy.sum((1, 2)), VIEW))
    for r, (m0, m1, m2, good, ext, name, vv, spec, lim) in enumerate(rows):
        im = ax[r, 0].imshow(np.where(good, m0, np.nan), extent=ext, cmap=SEQ,
                             norm=LogNorm(vmin=np.nanpercentile(m0[good], 5), vmax=np.nanmax(m0)))
        ax[r, 0].set_title(f'{name}\nmoment 0')
        colorbar(fig, im, ax[r, 0], 'integrated intensity')
        im = ax[r, 1].imshow(m1, extent=ext, cmap=DIV, vmin=-200, vmax=200)
        ax[r, 1].set_title('moment 1 (relative to systemic)')
        colorbar(fig, im, ax[r, 1], 'km/s')
        im = ax[r, 2].imshow(m2, extent=ext, cmap=WIDTH, vmin=0, vmax=60)
        ax[r, 2].set_title(f'moment 2 (median {np.nanmedian(m2):.0f} km/s)')
        colorbar(fig, im, ax[r, 2], 'km/s')
        for a in ax[r, :3]:
            a.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='kpc', ylabel='kpc')
            a.grid(False)
        ax[r, 3].plot(vv, spec / spec.max(), color=C1 if r else C2)
        ax[r, 3].set(xlim=(-350, 350), xlabel='v − v$_{sys}$ [km/s]', ylabel='normalised flux',
                     title='integrated spectrum')
    title(fig, 'Step 24: a real spiral (NGC 7469, i ≈ 45°) next to the simulated one (i = 50°). NGC 7469\'s CO '
               'arms lie within ~4 kpc, so the panels use different scales')
    save(fig, '24_ngc7469_vs_songs_moments.png')

    fig, ax = plt.subplots(2, 6, figsize=(27, 9.5))
    for r, (cb, vv, ext, lim, name, m0, pix, r0) in enumerate((
            (ngc['cube'], ngc['v'], ext_ngc, 4.0, 'NGC 7469', ngc['m0'], ngc['kpc_per_px'], 2.0),
            (noisy, v, ext_sim, VIEW, 'SONGS', sm0, PIX_KPC, 6.0))):
        vmax = np.percentile(cb, 99.95)
        for a, vt in zip(ax[r, :5], (-150, -75, 0, 75, 150)):
            c = int(np.argmin(np.abs(vv - vt)))
            a.imshow(cb[c], extent=ext, cmap=SEQ, vmin=0, vmax=vmax)
            a.set(xlim=(-lim, lim), ylim=(-lim, lim), title=f'{name}: {vv[c]:+.0f} km/s')
            a.grid(False)
        n = m0.shape[0]
        yy, xx = np.indices(m0.shape)
        ring = np.abs(np.hypot(xx - n / 2, yy - n / 2) * pix - r0) < 0.075 * r0
        vals = np.where(ring, np.nan_to_num(m0), np.nan)
        on = np.unravel_index(np.nanargmax(vals), m0.shape)
        off = np.unravel_index(np.nanargmin(np.where(ring & (m0 > 0), m0, np.nan)), m0.shape)
        for (iy, ix), label, col in ((on, 'on arm', C2), (off, 'interarm', C1)):
            ax[r, 5].plot(vv, cb[:, iy, ix], color=col, label=label)
        ax[r, 5].set(xlim=(-350, 350), xlabel='v [km/s]', title=f'{name}: spectra at R = {r0:.0f} kpc')
        ax[r, 5].legend(fontsize=8)
    title(fig, 'Step 25: channel maps and single-pixel spectra, real vs simulated')
    save(fig, '25_ngc7469_vs_songs_channels.png')


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ngc7469', default=NGC7469_DEFAULT, help='path to the NGC 7469 CO(2-1) FITS cube')
    ap.add_argument('--skip-gallery', action='store_true',
                    help='skip the gallery systems and population figures (22-23)')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    print('Building the fiducial spiral and its axisymmetric twin through SONGSPhy ...', flush=True)
    cube, params, stage = build(FIDUCIAL, 'spiral')
    cube_t, params_t, stage_t = build(TWIN, 'twin')
    save_system('fiducial_spiral', cube, params, stage)
    save_system('fiducial_axisymmetric_twin', cube_t, params_t, stage_t)
    gal = stage['wave']
    print('Fiducial galaxy:')
    for k, v in params['spiral'].items():
        print(f'    {k:28s} {v}')

    print('Physics figures ...', flush=True)
    fig_mass_model(gal)
    fig_lindblad(gal)
    fig_stability(gal)
    fig_dispersion(gal)
    fig_pitch(gal)
    fig_envelope(gal)
    fig_geometry(gal)
    fig_streaming_profile(gal)
    fig_knots(gal)
    fig_turbulence(gal)
    fig_center(gal, stage)

    print('Grid and rotation figures ...', flush=True)
    fig_grid_3d(stage)
    fig_slices(stage)
    fig_buildup(gal, stage)
    fig_face_on(stage, stage_t)
    fig_velocity_field(stage, gal)
    fig_rotation_3d(stage)
    fig_los_velocity(stage, stage_t)

    print('Cube figures ...', flush=True)
    fig_channels(cube, params)
    fig_moments(cube, params, cube_t, params_t)
    fig_pv(cube, params, cube_t)

    if not args.skip_gallery:
        print('Population figures ...', flush=True)
        fig_gallery(build_gallery())
        fig_distributions()

    if os.path.exists(args.ngc7469):
        print('NGC 7469 comparison ...', flush=True)
        fig_ngc7469(load_ngc7469(args.ngc7469), cube, params)
    else:
        print('NGC 7469 cube not found, skipping figures 24-25:', args.ngc7469)


if __name__ == '__main__':
    main()
