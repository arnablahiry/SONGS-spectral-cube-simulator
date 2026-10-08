"""Tests for the density-wave spiral model (songs.spiral) and its wiring into
the SONGS generators."""
import numpy as np
import pytest

from songs.spiral import (SpiralDensityWave, SpiralGalaxy, sample_spiral_params,
                          arm_profile, lognormal_field)


@pytest.fixture(scope="module")
def galaxy():
    return SpiralGalaxy(R_d=3.0, v_0=200.0, m=2, Q_min=1.1, corotation=4.5,
                        arm_contrast=4.0, m2=None, knot_fraction=0.0, turb_sigma=0.0,
                        arm_segmentation=0.0)


def test_weak_arms_reduce_to_linear_lin_shu():
    w = SpiralDensityWave(R_d=3.0, v_0=200.0, m=2, Q_min=1.1, corotation=4.5,
                          arm_contrast=1.02)
    R = np.full(360, 5.0)
    chi = np.linspace(0, 2 * np.pi, 360)
    phi = (chi - w._at(w.phase, R)) / w.m + w.phase0
    v_R, v_phi = w.streaming(R, phi)
    A = w._at(w.eta, 5.0)
    Om, k, kap = w._at(w.Omega, 5.0), w._at(w.k_r, 5.0), w._at(w.kappa, 5.0)
    lin_R = w.m * (w.Omega_p - Om) / k * A * np.cos(chi)
    lin_phi = kap**2 / (2 * Om * k) * A * np.sin(chi)
    assert np.abs(v_R - lin_R).max() < 0.02 * np.abs(lin_R).max()
    assert np.abs(v_phi - lin_phi).max() < 0.02 * np.abs(lin_phi).max()


def test_arm_profile_and_integral_are_normalised():
    chi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    for eta in (0.1, 0.8, 2.0):
        for delta in (0.0, -0.5, 0.7):
            assert arm_profile(chi, eta, delta).mean() == pytest.approx(1.0, abs=1e-6)
    # ∫(ρ-1) over a full wavelength vanishes, so v_φ is periodic
    w = SpiralDensityWave(R_d=3.0, v_0=200.0, skew=0.5)
    assert np.abs(w._G_table[:, -1]).max() < 1e-4


def test_shock_side_faces_upstream_and_flips_at_corotation():
    w = SpiralDensityWave(R_d=3.0, v_0=200.0, m=2, Q_min=1.1, corotation=4.5,
                          arm_contrast=4.0, skew=0.6)
    d = np.radians(20)
    for R, upstream in ((0.6 * w.R_CR, -1), (1.2 * w.R_CR, +1)):
        # Gas moves towards +χ inside corotation and -χ outside, so the
        # steep (shocked) side of the crest faces χ < 0 inside, χ > 0 outside.
        chi0 = -w._at(w.phase, R)
        rho = lambda c: w.density(R, (chi0 + c) / w.m + w.phase0)
        drop_up, drop_down = rho(0) - rho(upstream * d), rho(0) - rho(-upstream * d)
        if w._at(w.eta, R) > 0.05:
            assert drop_up > drop_down


def test_arms_redistribute_flux_around_annuli(galaxy):
    phi = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    for R in (3.0, 6.0, 9.0):
        x, y = R * np.cos(phi), R * np.sin(phi)
        assert galaxy.arm_factor_on_grid(x, y).mean() == pytest.approx(1.0, rel=1e-3)


def test_arms_are_trailing(galaxy):
    # SONGS disks rotate clockwise, so trailing arms wind counter-clockwise
    # outward: the crest's standard azimuth must increase with radius.
    for x, y in galaxy.arm_crests():
        ang = np.unwrap(np.arctan2(y, x))
        assert np.all(np.diff(ang) > 0)


def test_unperturbed_velocity_matches_songs_rotation_sense(galaxy):
    x, y = np.array([4.0, 0.0]), np.array([0.0, 4.0])
    vx, vy = galaxy.velocity_on_grid(x, y, perturbed=False)
    vc = galaxy.primary._at(galaxy.primary.v_c, 4.0)
    # SONGS tangent vector is (y, -x)
    np.testing.assert_allclose(vx, [0.0, vc], atol=1e-6 * vc)
    np.testing.assert_allclose(vy, [-vc, 0.0], atol=1e-6 * vc)


def test_sampler_gives_tightly_wound_arms():
    rng = np.random.RandomState(0)
    for _ in range(10):
        p = sample_spiral_params(3.0, 200.0, rng=rng)
        g = SpiralGalaxy(**p)
        p_med, p_90 = g.primary.pitch_stats()
        assert p['m'] in (1, 2, 3)
        assert p['m2'] in (None, 1, 2, 3) and p['m2'] != p['m']
        assert 3.0 <= p_med and p_90 <= 15.0


def test_overrides_are_respected():
    p = sample_spiral_params(3.0, 200.0, overrides={'m': 3, 'm2': None, 'c_s': 9.0},
                             rng=np.random.RandomState(1))
    assert p['m'] == 3 and p['m2'] is None and p['c_s'] == 9.0


def test_knots_follow_jeans_spacing():
    g = SpiralGalaxy(R_d=3.0, v_0=200.0, m=2, Q_min=1.1, corotation=4.5,
                     arm_contrast=5.0, knot_fraction=0.2, seed=3)
    assert g.knots['x'].size > 0 and g.n_spurs > 0
    lam = g.jeans_spacing(g.knots['R'])
    assert np.all((lam > 0.05) & (lam < 10.0))


def test_bulge_creates_ilr_and_steep_inner_rotation():
    no_b = SpiralDensityWave(R_d=3.0, v_0=200.0, bulge_fraction=0.0)
    bulge = SpiralDensityWave(R_d=3.0, v_0=200.0, bulge_fraction=0.25, bulge_size=0.2)
    assert bulge._at(bulge.v_c, 1.0) > 1.3 * no_b._at(no_b.v_c, 1.0)
    assert bulge.R_ILR is not None
    with pytest.raises(ValueError):
        SpiralDensityWave(R_d=3.0, v_0=200.0, disc_fraction=0.95, bulge_fraction=0.5)


def test_center_and_ring_carry_their_flux_fractions():
    g = SpiralGalaxy(R_d=3.0, v_0=200.0, m=2, Q_min=1.1, corotation=4.5, bulge_fraction=0.2,
                     center_fraction=0.15, ring_fraction=0.1, turb_sigma=0.0, knot_fraction=0.0)
    x = np.linspace(-20, 20, 201)
    z = np.linspace(-2, 2, 21)
    X, Y, Z = np.meshgrid(x, x, z, indexing='ij')
    disk = np.exp(-np.hypot(X, Y) / 3.0) * np.exp(-np.abs(Z) / 0.3)
    total, disk_s, center, ring = g.emission_cube(X, Y, Z, disk, return_parts=True)
    assert center.sum() / total.sum() == pytest.approx(g.center_fraction, rel=1e-6)
    assert ring.sum() / total.sum() == pytest.approx(g.ring_fraction, rel=1e-6)
    assert disk_s.sum() == pytest.approx(disk.sum(), rel=0.02)


def test_turbulence_is_mean_one_lognormal():
    f = lognormal_field((256, 256), 0.1, 0.4, seed=1)
    assert f.mean() == pytest.approx(1.0, rel=0.05)
    assert np.log(f).std() == pytest.approx(0.4, rel=1e-6)


def test_dispersion_rises_to_the_centre():
    g = SpiralGalaxy(R_d=3.0, v_0=200.0, c_s=10.0, sigma_center=40.0, turb_velocity_fraction=0.6)
    s = g.dispersion_on_grid(np.array([0.0, 10.0]), np.array([0.0, 0.0]))
    small = 10.0 * np.sqrt(1 - 0.6**2)
    assert s[0] == pytest.approx(np.hypot(small, 40.0)) and s[1] == pytest.approx(small, rel=1e-3)
    # coherent part + small-scale part add up to c_s in the disk
    x = np.linspace(-20, 20, 256)
    X, Y = np.meshgrid(x, x, indexing='ij')
    vx, vy, vz = g.turbulent_velocity_on_grid(X, Y)
    assert np.hypot(vz.std(), small) == pytest.approx(10.0, rel=1e-6)
    assert abs(np.corrcoef(vx[:, :-1].ravel(), vx[:, 1:].ravel())[0, 1]) > 0.7   # coherent (white noise: ~0)


def test_songs_spiral_end_to_end(tmp_path):
    import h5py
    from songs.core import SONGS
    g = SONGS(n_gals=1, n_cubes=1, resolution='resolved', grid_size=48, n_spectral_slices=20,
              seed=2, verbose=False, morphology='spiral', save=True, fname=str(tmp_path),
              diffuse_params={'enabled': False})
    cube, params = g.generate_cubes()[0]
    assert cube.shape == (20, 48, 48) and np.isfinite(cube).all()
    assert params['morphology'] == 'spiral'
    assert params['spiral']['m'] in (1, 2, 3)
    assert g.all_gal_vz_sigmas[0][0] == pytest.approx(g.all_spiral_params[0][0]['c_s'])
    with h5py.File(tmp_path / 'cube_1.h5') as f:
        assert f.attrs['morphology'] == 'spiral'
        assert 'pitch_median_deg' in f['spiral'].attrs


def test_disk_morphology_is_default_and_unchanged():
    from songs.core import SONGS
    g = SONGS(n_gals=1, n_cubes=1, grid_size=32, n_spectral_slices=10, seed=5, verbose=False)
    assert g.morphology == 'disk' and g.all_spiral_params == [[None]]


def test_bad_morphology_rejected():
    from songs.core import SONGS
    with pytest.raises(ValueError):
        SONGS(n_cubes=1, grid_size=32, morphology='barred', verbose=False)


@pytest.mark.parametrize("angles", [(40.0, 0.0), (35.0, 25.0), (60.0, -30.0)])
def test_projected_velocities_follow_the_projected_geometry(angles):
    # Every voxel moved to the sky must carry its own velocity rotated by the
    # same rotation; otherwise spin sense, kinematic axis and arm handedness
    # (trailing vs leading) disagree on the sky.
    from songs.core import SONGSPhy, _image_rotation_matrix
    g = SONGSPhy(n_gals=1, n_cubes=1, spatial_resolution=0.5, spectral_resolution=20, fov=40,
                 beam_info=[1, 1, 0], seed=1, verbose=False, morphology='spiral',
                 spiral_params={'m2': None, 'c_s': 8.0, 'sigma_center': 20.0},
                 diffuse_params={'enabled': False})
    g.all_hz = [np.array([0.5])]
    g.all_gal_x_angles = [np.array([angles[0]])]
    g.all_gal_y_angles = [np.array([angles[1]])]
    g.keep_intermediates = True
    g.generate_cubes()
    st = g.intermediates[0]
    ax = st['axis_kpc']
    M = _image_rotation_matrix(*angles)
    P = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), axis=-1)
    Q = P @ M                                   # galaxy-frame position of each sky voxel (Mᵀp)
    gal = st['wave']
    vx, vy = gal.velocity_on_grid(Q[..., 0], Q[..., 1])
    expected = M[2, 0] * vx + M[2, 1] * vy
    bright = st['rotated_disk'] > np.percentile(st['rotated_disk'], 95)
    got = st['rotated_vel_los'][bright]
    assert np.corrcoef(got, expected[bright])[0, 1] > 0.97
    assert np.median(np.abs(got - expected[bright])) < 15.0


def test_arm_segments_vary_strength_along_arms():
    g = SpiralGalaxy(R_d=3.0, v_0=200.0, m=2, Q_min=1.1, corotation=4.5, arm_contrast=4.0,
                     m2=None, knot_fraction=0.0, turb_sigma=0.0, arm_segmentation=0.7, seed=4)
    w = g.primary
    Rs = np.linspace(w.R[w.arm_mask].min(), w.R[w.arm_mask].max(), 400)
    eta_crest = [w.local_eta(Rs, w.crest_phi(Rs, j)) / w._at(w.eta, Rs) for j in range(w.m)]
    # strength varies along each arm, differently for the two arms
    assert all(np.std(np.log(e)) > 0.3 for e in eta_crest)
    assert np.corrcoef(eta_crest[0], eta_crest[1])[0, 1] < 0.9
    # still only redistributes flux around annuli
    phi = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
    for R in (4.0, 7.0):
        x, y = R * np.cos(phi), R * np.sin(phi)
        assert g.arm_factor_on_grid(x, y).mean() == pytest.approx(1.0, rel=0.01)
