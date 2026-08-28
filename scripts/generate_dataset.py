#!/usr/bin/env python3
"""Headless (no-GUI) SONGS Large-Dataset Mode generator.

Reproduces exactly what the GUI's "Generate Dataset" button does
(``SONGSGUI._sample_cube_params`` / ``_run_generate_dataset`` /
``_build_cube_manifest`` in ``src/songs/gui.py``) as a standalone CLI, with
NO dependency on tkinter — safe to run on a headless cluster node with no
display.

Output layout matches the GUI exactly, so it drops straight into
``SongsCubeDataset`` (``src/data/songs_cube_dataset.py`` in
Denoiser2D1D-improved) with no changes:

    <out>/dataset.json     manifest (one entry per cube; sn_peak, all
                           physical params, filename, index)
    <out>/raw/cube_00001.h5, cube_00002.h5, ...
                           one HDF5 file per cube: /clean_cube, /noisy_cube,
                           /galaxies/*, /beam, /diffuse_params, plus a
                           parameters_json attr identical to its manifest
                           entry (see songs.core._save_cube_hdf5).

Every [min, max] range below is drawn independently and UNIFORMLY per cube
except Re (physically fixed across the whole dataset — see the "Physical
size, not distance" note in --re-kpc's help) — this mirrors the GUI's
Large-Dataset Mode defaults exactly. All defaults below match the GUI's
factory defaults (src/songs/gui.py, the `_dv`/`*_min_var`/`*_max_var`
initializers), so running with no flags at all reproduces what "Generate
Dataset" does out of the box.

Cluster-friendly behaviour:

- SIGINT/SIGTERM (Ctrl-C, or a scheduler's preemption/time-limit signal)
  stops the loop cleanly after the in-flight cube and writes dataset.json
  with whatever was generated so far — same as pressing the GUI's Stop
  button. You lose nothing by requesting an ambitious --n-samples and
  letting a job time limit cut it short.
- ``--resume`` continues an existing --out directory: already-written cube
  files are skipped (by filename) and new ones are appended to a merged
  manifest, so a job that got preempted can be resubmitted as-is.

Examples
--------
    # Reproduce the GUI's defaults exactly, 20000 cubes
    python scripts/generate_dataset.py --out data/songs_run --n-samples 20000

    # Bigger cubes, wider resolution range, resumable across job resubmits
    python scripts/generate_dataset.py --out /scratch/$USER/songs_run \\
        --n-samples 50000 --grid-size 128 \\
        --spatial-resolution-range 0.5 3.0 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import h5py
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..', 'src'))

from songs.core import SONGSPhy, DEFAULT_DIFFUSE_PARAMS, _save_cube_hdf5, _sersic_total_flux_3d  # noqa: E402
from songs.utils import apply_and_convolve_noise  # noqa: E402


# ---------------------------------------------------------------------------
# Graceful stop — SIGINT/SIGTERM sets a flag the loop checks between cubes,
# exactly like the GUI's Stop button setting self._stop_requested.
# ---------------------------------------------------------------------------
_stop_requested = False


def _handle_stop_signal(signum, _frame):
    global _stop_requested
    print(f"\n[generate_dataset] received signal {signum} — finishing current "
          f"cube, then stopping and writing dataset.json...", flush=True)
    _stop_requested = True


def _diffuse_range(key, spread=0.2):
    """(default*(1-spread), default*(1+spread)) around a DEFAULT_DIFFUSE_PARAMS
    value — matches the GUI's `_dv()` initializer exactly (±20%)."""
    base = float(DEFAULT_DIFFUSE_PARAMS[key])
    return (base * (1.0 - spread), base * (1.0 + spread))


def build_base_config(args):
    """Equivalent of SONGSGUI._collect_large_dataset_base(), but reading
    argparse results instead of Tk vars — same keys, same defaults."""
    return dict(
        bmin_px=args.bmin_px,
        bmaj_px=args.bmaj_px,
        bpa=args.bpa,
        grid_size=args.grid_size,
        spectral_resolution=args.spectral_resolution,
        spatial_resolution_range=tuple(args.spatial_resolution_range),
        offset_gals_px=tuple(args.offset_px_range),
        convolve_beam=not args.raw_beam,
        allow_overlap=args.allow_overlap,
        hz_range=tuple(args.hz_range),
        Se_range=tuple(args.se_range),
        Re_fixed=args.re_kpc,
        sigma_v_range=tuple(args.sigma_v_range),
        sat_brightness_frac_range=tuple(args.sat_brightness_frac_range),
        sat_Re_frac_range=tuple(args.sat_re_frac_range),
        max_gals=args.max_gals,
        angle_x_range=tuple(args.angle_x_range),
        angle_y_range=tuple(args.angle_y_range),
        n_sersic_range=tuple(args.n_sersic_range),
        use_noise=not args.no_noise,
        sn_range=tuple(args.sn_peak_range),
        diffuse_ranges=dict(
            halo_Se_factor=tuple(args.halo_se_factor_range),
            halo_Re_factor=tuple(args.halo_re_factor_range),
            halo_sigma_vz=tuple(args.halo_sigma_vz_range),
            bridge_Se_factor=tuple(args.bridge_se_factor_range),
            bridge_width_start_factor=tuple(args.bridge_width_start_factor_range),
            bridge_width_end_factor=tuple(args.bridge_width_end_factor_range),
            bridge_sigma_vz=tuple(args.bridge_sigma_vz_range),
            tail_Se_factor=tuple(args.tail_se_factor_range),
            tail_vel_gradient=tuple(args.tail_vel_gradient_range),
            tail_length_factor=tuple(args.tail_length_factor_range),
            tail_width_factor=tuple(args.tail_width_factor_range),
            tail_sigma_vz=tuple(args.tail_sigma_vz_range),
        ),
        n_samples=args.n_samples,
        save_folder=args.out,
    )


def sample_cube_params(base, rng):
    """Exact port of SONGSGUI._sample_cube_params — see src/songs/gui.py for
    the full derivation notes (fov-rounding, hz unit conversion, and the
    exact-total-flux-fraction satellite solve). Kept byte-for-byte identical
    in logic so headless and GUI runs are statistically the same generator."""

    def u(bounds):
        lo, hi = bounds
        if hi <= lo:
            return float(lo)
        return float(rng.uniform(lo, hi))

    spatial_resolution = max(u(base['spatial_resolution_range']), 0.01)
    fov = max(1, int(round(base['grid_size'] * spatial_resolution)))
    spatial_resolution = fov / (base['grid_size'] + 1e-9)

    bmin_kpc = base['bmin_px'] * spatial_resolution
    bmaj_kpc = base['bmaj_px'] * spatial_resolution
    offset_gals = (base['offset_gals_px'][0] * spatial_resolution,
                   base['offset_gals_px'][1] * spatial_resolution)

    n_gals = int(rng.integers(1, base['max_gals'] + 1))
    central_n = u(base['n_sersic_range'])
    central_hz = u(base['hz_range'])
    central_Se = u(base['Se_range'])
    central_Re_kpc = base['Re_fixed']
    central_gal_x_angle = u(base['angle_x_range'])
    central_gal_y_angle = u(base['angle_y_range'])
    sigma_v = u(base['sigma_v_range'])

    all_Re = [central_Re_kpc / spatial_resolution]
    all_hz = [central_hz / spatial_resolution]
    all_Se = [central_Se]
    all_gal_x_angles = [central_gal_x_angle]
    all_gal_y_angles = [central_gal_y_angle]
    all_n = [central_n]

    if n_gals > 1:
        n_sat = n_gals - 1
        _re_frac = float(np.clip(u(base['sat_Re_frac_range']), 0.05, 1.0))
        _re_lo = max(_re_frac - 0.1, 0.05) * all_Re[0]
        _re_hi = min(_re_frac + 0.1, 1.0) * all_Re[0]
        sat_Re = list(rng.uniform(_re_lo, _re_hi, n_sat))
        sat_hz = list(rng.uniform(all_hz[0] / 3, all_hz[0] / 2, n_sat))
        _b = float(np.clip(u(base['sat_brightness_frac_range']), 0.0, 2.0))
        sat_n = list(rng.uniform(base['n_sersic_range'][0], base['n_sersic_range'][1], n_sat))
        _central_F_total = _sersic_total_flux_3d(all_Se[0], all_Re[0], central_n, central_hz)
        sat_Se = []
        for re_sat, hz_sat, n_sat_i in zip(sat_Re, sat_hz, sat_n):
            _F_target = _b * _central_F_total * rng.uniform(0.85, 1.15)
            _F_unit_Se = _sersic_total_flux_3d(1.0, re_sat, n_sat_i, hz_sat)
            sat_Se.append(float(_F_target / _F_unit_Se) if _F_unit_Se > 0 else 0.0)
        sat_x_angles = list(rng.uniform(-180.0, 180.0, n_sat))
        sat_y_angles = list(rng.uniform(-180.0, 180.0, n_sat))
        all_Re += sat_Re
        all_hz += sat_hz
        all_Se += sat_Se
        all_n += sat_n
        all_gal_x_angles += sat_x_angles
        all_gal_y_angles += sat_y_angles

    diffuse_params = dict(DEFAULT_DIFFUSE_PARAMS)
    diffuse_params['enabled'] = True
    for k, bounds in base['diffuse_ranges'].items():
        diffuse_params[k] = u(bounds)

    sn_peak = u(base['sn_range']) if base['use_noise'] else None

    return dict(
        beam_info=[bmin_kpc, bmaj_kpc, base['bpa']],
        n_gals=n_gals,
        fov=fov,
        spectral_resolution=base['spectral_resolution'],
        spatial_resolution=spatial_resolution,
        all_Re=np.array(all_Re),
        all_hz=np.array(all_hz),
        all_Se=np.array(all_Se),
        all_n=np.array(all_n),
        all_gal_x_angles=np.array(all_gal_x_angles),
        all_gal_y_angles=np.array(all_gal_y_angles),
        sigma_v=sigma_v,
        offset_gals=offset_gals,
        diffuse_params=diffuse_params,
        sn_peak=sn_peak,
    )


def build_cube_manifest(params, sn_peak=None):
    """Exact port of SONGSGUI._build_cube_manifest."""
    sr = float(params['spatial_resolution'])
    sigma_v = params['sigma_v']
    sigma_v = float(np.mean(sigma_v)) if not np.isscalar(sigma_v) else float(sigma_v)
    return dict(
        n_gals=int(params['n_gals']),
        sersic_n=[float(v) for v in np.atleast_1d(params['all_n'])],
        Re_kpc=[float(v) * sr for v in np.atleast_1d(params['all_Re'])],
        hz=[float(v) * sr for v in np.atleast_1d(params['all_hz'])],
        Se=[float(v) for v in np.atleast_1d(params['all_Se'])],
        inclination_x_deg=[float(v) for v in np.atleast_1d(params['all_gal_x_angles'])],
        inclination_y_deg=[float(v) for v in np.atleast_1d(params['all_gal_y_angles'])],
        sigma_v_km_s=sigma_v,
        beam_info_kpc=list(params['beam_info']),
        fov_kpc=int(params['fov']),
        spatial_resolution_kpc_per_px=sr,
        spectral_resolution_km_s=int(params['spectral_resolution']),
        diffuse_params=dict(params.get('diffuse_params', {})),
        sn_peak=(float(sn_peak) if sn_peak is not None else None),
    )


def load_replay_manifest(path):
    """Load a previously written dataset.json and return its manifest
    entries, sorted by ``index``. Used by ``--replay-json`` to regenerate
    cubes with the exact same physical parameters as an existing dataset
    instead of freshly sampling them."""
    with open(path, 'r') as fh:
        d = json.load(fh)
    entries = d.get('manifest', [])
    return sorted(entries, key=lambda e: e['index'])


def cube_params_from_entry(entry, offset_gals_px_range):
    """Inverse of build_cube_manifest(): reconstruct a sample_cube_params()
    -shaped dict from one manifest entry, so an existing cube's exact
    physical parameters (Re, hz, Se, angles, sigma_v, beam, diffuse params,
    sn_peak, ...) can be replayed through the same generation code path.

    NOTE: the manifest does not record ``offset_gals`` (the per-cube
    galaxy-placement offset), because it's a placement detail, not a
    physical galaxy property — so it's redrawn here from
    ``offset_gals_px_range`` (scaled to this entry's spatial resolution),
    same as a fresh run would. This means replayed cubes reproduce every
    stored physical parameter exactly, but satellite *positions* are not
    guaranteed to be pixel-identical to the original run.
    """
    sr = float(entry['spatial_resolution_kpc_per_px'])
    return dict(
        beam_info=list(entry['beam_info_kpc']),
        n_gals=int(entry['n_gals']),
        fov=int(entry['fov_kpc']),
        spectral_resolution=entry['spectral_resolution_km_s'],
        spatial_resolution=sr,
        all_Re=np.array(entry['Re_kpc'], dtype=float) / sr,
        all_hz=np.array(entry['hz'], dtype=float) / sr,
        all_Se=np.array(entry['Se'], dtype=float),
        all_n=np.array(entry['sersic_n'], dtype=float),
        all_gal_x_angles=np.array(entry['inclination_x_deg'], dtype=float),
        all_gal_y_angles=np.array(entry['inclination_y_deg'], dtype=float),
        sigma_v=float(entry['sigma_v_km_s']),
        offset_gals=(offset_gals_px_range[0] * sr, offset_gals_px_range[1] * sr),
        diffuse_params=dict(entry['diffuse_params']),
        sn_peak=(float(entry['sn_peak']) if entry.get('sn_peak') is not None else None),
    )


def run(base, seed=None, resume=False, log_every=10, replay_entries=None):
    """Exact port of SONGSGUI._run_generate_dataset, minus the Tk
    progress-bar/thread plumbing — prints progress to stdout instead.

    If ``replay_entries`` is given (a list of manifest entries loaded via
    ``load_replay_manifest``), cube parameters are taken from those entries
    instead of being freshly sampled — see ``cube_params_from_entry``."""
    raw_dir = os.path.join(base['save_folder'], 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    dataset_path = os.path.join(base['save_folder'], 'dataset.json')

    manifest = []
    existing_filenames = set()
    start_index = 0
    if resume and os.path.exists(dataset_path):
        with open(dataset_path, 'r') as fh:
            prior = json.load(fh)
        manifest = prior.get('manifest', [])
        existing_filenames = {e['filename'] for e in manifest}
        start_index = max((e['index'] for e in manifest), default=0)
        print(f"[generate_dataset] --resume: {len(manifest)} cubes already present "
             f"in {dataset_path}, continuing from index {start_index + 1}")

    rng = np.random.default_rng(seed)

    if replay_entries is not None:
        # Replay order need not be contiguous by index, so (unlike the fresh
        # RNG-sampled path) rely solely on existing_filenames below to skip
        # already-generated cubes rather than start_index.
        n = len(replay_entries)
        iterator = [(e['index'], e['filename'], e) for e in replay_entries]
    else:
        n = base['n_samples']
        iterator = [(i + 1, f'cube_{i + 1:05d}.h5', None) for i in range(start_index, n)]

    n_done_this_run = 0
    t_run_start = time.time()

    for cube_index, cube_filename, replay_entry in iterator:
        if _stop_requested:
            print(f"[generate_dataset] stopped after {len(manifest)}/{n} cubes.")
            break

        if cube_filename in existing_filenames:
            continue  # --resume: already generated in a previous invocation

        cube_t0 = time.time()
        if replay_entry is not None:
            cube_params = cube_params_from_entry(replay_entry, base['offset_gals_px'])
        else:
            cube_params = sample_cube_params(base, rng)
        try:
            g = SONGSPhy(
                n_gals=cube_params['n_gals'],
                n_cubes=1,
                spatial_resolution=cube_params['spatial_resolution'],
                spectral_resolution=cube_params['spectral_resolution'],
                offset_gals=cube_params['offset_gals'],
                beam_info=cube_params['beam_info'],
                fov=cube_params['fov'],
                verbose=False,
                diffuse_params=cube_params['diffuse_params'],
            )
            n_g = cube_params['n_gals']
            g.all_Re = [cube_params['all_Re']]
            g.all_hz = [cube_params['all_hz']]
            g.all_Se = [cube_params['all_Se']]
            g.all_n = [cube_params['all_n']]
            g.all_gal_x_angles = [cube_params['all_gal_x_angles']]
            g.all_gal_y_angles = [cube_params['all_gal_y_angles']]
            g.all_gal_vz_sigmas = [np.full(n_g, cube_params['sigma_v'])]
            g.all_gal_v_0 = [np.full(n_g, 200.0)]
            g.convolve_beam = base['convolve_beam']
            g.allow_overlap = base['allow_overlap']

            results = g.generate_cubes()
            cube, core_params = results[0]
        except Exception as e:
            print(f"[generate_dataset] cube {cube_index} failed: {e}")
            continue

        manifest_entry = build_cube_manifest(cube_params, sn_peak=cube_params['sn_peak'])
        manifest_entry['index'] = cube_index
        manifest_entry['filename'] = cube_filename

        noisy_cube = None
        if base['use_noise'] and cube_params['sn_peak'] is not None:
            beam_px = [float(g.beam_info[0]), float(g.beam_info[1]), float(g.beam_info[2])]
            noisy_cube = apply_and_convolve_noise(cube, beam_px, cube_params['sn_peak'])

        cube_path = os.path.join(raw_dir, cube_filename)
        _save_cube_hdf5(cube_path, cube, core_params, g, 0, noisy_cube=noisy_cube)
        with h5py.File(cube_path, 'a') as f:
            f.attrs['parameters_json'] = json.dumps(manifest_entry)

        manifest.append(manifest_entry)
        n_done_this_run += 1

        dt = time.time() - cube_t0
        if n_done_this_run % log_every == 0 or cube_index == n:
            elapsed = time.time() - t_run_start
            rate = n_done_this_run / elapsed if elapsed > 0 else 0.0
            eta_s = (n - cube_index) / rate if rate > 0 else float('nan')
            print(f"[generate_dataset] {cube_index}/{n} cubes "
                 f"({len(manifest)} total in manifest) — last {dt:.1f}s/cube, "
                 f"avg {1/rate:.1f}s/cube, ETA {eta_s/60:.1f} min", flush=True)

    with open(dataset_path, 'w') as fh:
        json.dump(dict(
            n_samples=n,
            n_generated=len(manifest),
            use_noise=base['use_noise'],
            manifest=manifest,
        ), fh, indent=2)
    print(f"[generate_dataset] wrote {dataset_path} "
         f"({len(manifest)}/{n} cubes total)")
    return dataset_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('--out', required=True,
                   help='Output directory. Written as <out>/dataset.json + <out>/raw/*.h5 '
                        '(created if missing).')
    p.add_argument('--n-samples', type=int, default=20000)
    p.add_argument('--seed', type=int, default=None,
                   help='RNG seed for reproducibility. Omit for a fresh random dataset '
                        '(matches the GUI, which never seeds this).')
    p.add_argument('--resume', action='store_true',
                   help='Continue an existing --out directory: skip cube_XXXXX.h5 files '
                        'already present and append new ones. Safe to resubmit after '
                        'preemption/a time-limit kill.')
    p.add_argument('--log-every', type=int, default=10, help='Print progress every N cubes.')
    p.add_argument('--replay-json', type=str, default=None,
                   help='Path to an existing dataset.json whose manifest entries should be '
                        'replayed instead of freshly sampled: every cube is regenerated with '
                        'that entry\'s exact n_gals/Re/hz/Se/sersic-index/inclinations/'
                        'sigma_v/beam/fov/spatial+spectral resolution/diffuse-params/sn_peak, '
                        'and written under the same index/filename as in that manifest. '
                        '--n-samples, --seed and all the per-parameter --*-range flags are '
                        'ignored for replayed cubes (they only matter for the RNG-sampled '
                        'path). The one thing NOT stored in a manifest — and so NOT '
                        'reproduced exactly — is each cube\'s satellite placement offset; it '
                        'is redrawn from --offset-px-range same as a fresh run. Combine with '
                        '--out pointing at a fresh directory (or with --resume) to only '
                        'regenerate cubes missing their raw/*.h5 file.')

    # --- Initialisation (shared by every cube) ---
    p.add_argument('--grid-size', type=int, default=96,
                   help='Spatial pixel grid size (both axes) shared by every cube; FOV is '
                        'derived per cube from spatial_resolution to keep this fixed.')
    p.add_argument('--spectral-resolution', type=int, default=20, help='km/s per channel.')
    p.add_argument('--spatial-resolution-range', type=float, nargs=2, default=(0.625, 2.5),
                   metavar=('MIN', 'MAX'), help='kpc/px, drawn per cube.')
    p.add_argument('--bmin-px', type=float, default=4.0, help='Beam minor axis, pixels (fixed).')
    p.add_argument('--bmaj-px', type=float, default=4.7, help='Beam major axis, pixels (fixed).')
    p.add_argument('--bpa', type=float, default=20.0, help='Beam position angle, degrees.')
    p.add_argument('--raw-beam', action='store_true',
                   help='Skip beam convolution entirely (spectral smoothing still applies). '
                        'Default is convolved, matching the GUI.')
    p.add_argument('--no-noise', action='store_true',
                   help='Disable noise. NOTE: SongsCubeDataset requires noise-enabled data '
                        '(it pairs clean/noisy) — only pass this if you have your own '
                        'downstream use for clean-only cubes.')
    p.add_argument('--sn-peak-range', type=float, nargs=2, default=(3.0, 100.0), metavar=('MIN', 'MAX'))

    # --- Central galaxy ---
    p.add_argument('--re-kpc', type=float, default=5.0,
                   help='Central effective radius in kpc — PHYSICALLY FIXED across the whole '
                        'dataset (not a range). Resolved/unresolved diversity comes entirely '
                        'from --spatial-resolution-range changing how many pixels this same '
                        'physical size spans, not from varying Re itself.')
    p.add_argument('--n-sersic-range', type=float, nargs=2, default=(0.5, 1.5), metavar=('MIN', 'MAX'))
    p.add_argument('--hz-range', type=float, nargs=2, default=(0.64, 0.96), metavar=('MIN', 'MAX'),
                   help='Scale height, kpc.')
    p.add_argument('--se-range', type=float, nargs=2, default=(0.08, 0.12), metavar=('MIN', 'MAX'),
                   help='Surface brightness at Re.')
    p.add_argument('--sigma-v-range', type=float, nargs=2, default=(32.0, 48.0), metavar=('MIN', 'MAX'),
                   help='LOS velocity dispersion, km/s.')
    p.add_argument('--angle-x-range', type=float, nargs=2, default=(0, 359), metavar=('MIN', 'MAX'))
    p.add_argument('--angle-y-range', type=float, nargs=2, default=(0, 359), metavar=('MIN', 'MAX'))

    # --- Satellites ---
    p.add_argument('--max-gals', type=int, default=3, help='1..max_gals galaxies per cube, uniform.')
    p.add_argument('--sat-brightness-frac-range', type=float, nargs=2, default=(0.37, 0.50),
                   metavar=('MIN', 'MAX'),
                   help='Each satellite total flux, as a fraction of the central total flux.')
    p.add_argument('--sat-re-frac-range', type=float, nargs=2, default=(0.32, 0.48), metavar=('MIN', 'MAX'),
                   help='Satellite Re as a fraction of the central Re.')
    p.add_argument('--offset-px-range', type=float, nargs=2, default=(29.0, 46.0), metavar=('MIN', 'MAX'),
                   help='Central-to-satellite 3D separation, pixels (fixed footprint; physical '
                        'separation scales with the sampled resolution).')
    p.add_argument('--allow-overlap', action='store_true',
                   help='Allow satellites to overlap the central/each other (default: kept distinct).')

    # --- Diffuse structure (halo / bridges / tails); every range defaults to
    # DEFAULT_DIFFUSE_PARAMS[key] * (0.8, 1.2), matching the GUI's _dv(). ---
    p.add_argument('--halo-se-factor-range', type=float, nargs=2, default=_diffuse_range('halo_Se_factor'))
    p.add_argument('--halo-re-factor-range', type=float, nargs=2, default=_diffuse_range('halo_Re_factor'))
    p.add_argument('--halo-sigma-vz-range', type=float, nargs=2, default=_diffuse_range('halo_sigma_vz'))
    p.add_argument('--bridge-se-factor-range', type=float, nargs=2, default=_diffuse_range('bridge_Se_factor'))
    p.add_argument('--bridge-width-start-factor-range', type=float, nargs=2,
                   default=_diffuse_range('bridge_width_start_factor'))
    p.add_argument('--bridge-width-end-factor-range', type=float, nargs=2,
                   default=_diffuse_range('bridge_width_end_factor'))
    p.add_argument('--bridge-sigma-vz-range', type=float, nargs=2, default=_diffuse_range('bridge_sigma_vz'))
    p.add_argument('--tail-se-factor-range', type=float, nargs=2, default=_diffuse_range('tail_Se_factor'))
    p.add_argument('--tail-vel-gradient-range', type=float, nargs=2, default=_diffuse_range('tail_vel_gradient'))
    p.add_argument('--tail-length-factor-range', type=float, nargs=2, default=_diffuse_range('tail_length_factor'))
    p.add_argument('--tail-width-factor-range', type=float, nargs=2, default=_diffuse_range('tail_width_factor'))
    p.add_argument('--tail-sigma-vz-range', type=float, nargs=2, default=_diffuse_range('tail_sigma_vz'))

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)  # scheduler preemption/time-limit kill

    os.makedirs(args.out, exist_ok=True)
    base = build_base_config(args)

    replay_entries = None
    if args.replay_json:
        replay_entries = load_replay_manifest(args.replay_json)
        base['n_samples'] = len(replay_entries)
        print(f"[generate_dataset] replaying {len(replay_entries)} cube params from "
             f"{args.replay_json} into {args.out}")
    else:
        print(f"[generate_dataset] generating up to {base['n_samples']} cubes into {args.out}")
        print(f"[generate_dataset] grid_size={base['grid_size']} "
             f"spatial_resolution_range={base['spatial_resolution_range']} "
             f"Re_fixed={base['Re_fixed']} kpc  use_noise={base['use_noise']}")

    run(base, seed=args.seed, resume=args.resume, log_every=args.log_every,
        replay_entries=replay_entries)


if __name__ == '__main__':
    main()
