"""Behavioural tests for SONGS core pipeline.

These tests import the actual package and run real (but small) computations
to catch regressions in the generation pipeline, diffuse parameter contracts,
and galaxy placement logic.
"""
import ast
import os
import py_compile
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _src(name):
    return os.path.join(ROOT, "src", "songs", name)


# ---------------------------------------------------------------------------
# Syntax checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["core.py", "gui.py", "visualise.py", "utils.py"])
def test_syntax_clean(module):
    py_compile.compile(_src(module), doraise=True)


# ---------------------------------------------------------------------------
# DEFAULT_DIFFUSE_PARAMS contract
# ---------------------------------------------------------------------------

# Keys the GUI reads via _collect_parameters — adding/removing here will
# catch mismatches between core defaults and GUI expectations.
_REQUIRED_DIFFUSE_KEYS = {
    "halo_Re_factor", "halo_Se_factor", "halo_sigma_vz",
    "bridge_Se_factor", "bridge_width_start_factor", "bridge_width_end_factor",
    "bridge_sigma_vz",
    "tail_Se_factor", "tail_vel_gradient", "tail_curvature",
    "tail_width_factor", "tail_sigma_vz", "tail_length_factor",
    "tail_decay_scale", "tail_n_samples",
}

def test_default_diffuse_params_keys():
    from songs.core import DEFAULT_DIFFUSE_PARAMS
    missing = _REQUIRED_DIFFUSE_KEYS - set(DEFAULT_DIFFUSE_PARAMS)
    assert not missing, f"Missing keys in DEFAULT_DIFFUSE_PARAMS: {missing}"


# ---------------------------------------------------------------------------
# place_galaxies
# ---------------------------------------------------------------------------

def test_place_galaxies_count():
    from songs.core import place_galaxies
    centers = place_galaxies(n_galaxies=3, grid_size=64, init_grid_size=40, offset_gals=10)
    assert len(centers) == 3


def test_place_galaxies_within_grid():
    from songs.core import place_galaxies
    grid_size = 64
    centers = place_galaxies(n_galaxies=4, grid_size=grid_size, init_grid_size=40, offset_gals=10)
    for c in centers:
        assert all(0 <= v < grid_size for v in c), f"Center out of bounds: {c}"


def test_place_galaxies_minimum_separation():
    """Satellites must be placed at least _min_sep away from all other galaxies.

    Use a large grid with generous offset so the placement algorithm can always
    satisfy the separation constraint without hitting the degenerate fallback.
    """
    from songs.core import place_galaxies
    grid_size = 125
    init_grid_size = 30   # kept small so min_pos/max_pos leave plenty of room
    # Mirror the formula in place_galaxies
    min_sep = max(4, int(init_grid_size // 1.5))
    centers = place_galaxies(n_galaxies=3, grid_size=grid_size,
                             init_grid_size=init_grid_size, offset_gals=40)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d3 = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            assert d3 >= min_sep, f"Galaxies {i} and {j} too close: {d3:.1f} < {min_sep}"


# ---------------------------------------------------------------------------
# songs.init — small cube generation
# ---------------------------------------------------------------------------

def test_init_returns_generator():
    import songs
    g = songs.init(n_gals=1, n_cubes=1, grid_size=24, n_spectral_slices=10, seed=0)
    assert hasattr(g, "generate_cubes")


def test_generate_cubes_output_shape():
    import songs
    g = songs.init(n_gals=1, n_cubes=1, grid_size=24, n_spectral_slices=10, seed=1)
    results = g.generate_cubes()
    assert len(results) == 1
    cube, meta = results[0]
    assert cube.ndim == 3
    # spectral axis length may differ from n_spectral_slices due to oversampling trim
    assert cube.shape[1] == cube.shape[2]  # square spatial grid


def test_generate_cubes_metadata_keys():
    import songs
    g = songs.init(n_gals=1, n_cubes=1, grid_size=24, n_spectral_slices=10, seed=2)
    _, meta = g.generate_cubes()[0]
    for key in ("average_vels", "beam_info", "pix_spatial_scale"):
        assert key in meta, f"Missing metadata key: {key}"


def test_generate_cubes_no_nan():
    import songs
    g = songs.init(n_gals=1, n_cubes=1, grid_size=24, n_spectral_slices=10, seed=3)
    cube, _ = g.generate_cubes()[0]
    assert not np.any(np.isnan(cube)), "Cube contains NaN values"


def test_generate_cubes_nonnegative():
    import songs
    g = songs.init(n_gals=1, n_cubes=1, grid_size=24, n_spectral_slices=10, seed=4)
    cube, _ = g.generate_cubes()[0]
    assert np.all(cube >= 0), "Cube contains negative flux values"


# ---------------------------------------------------------------------------
# songs.init — multi-galaxy
# ---------------------------------------------------------------------------

def test_generate_multi_galaxy():
    import songs
    # Use a larger grid so 3 galaxies can be placed without collision
    g = songs.init(n_gals=3, n_cubes=1, grid_size=48, n_spectral_slices=10, seed=5)
    results = g.generate_cubes()
    assert len(results) == 1
    cube, meta = results[0]
    assert cube.ndim == 3
    pg = meta.get("per_galaxy_cubes")
    if pg is not None:
        pg = np.array(pg)
        assert pg.shape[0] == 3


# ---------------------------------------------------------------------------
# songs.init_phy — physical unit mode
# ---------------------------------------------------------------------------

def test_init_phy_returns_generator():
    import songs
    g = songs.init_phy(n_gals=1, fov=30, spatial_resolution=2.0,
                       spectral_resolution=20, seed=6)
    assert hasattr(g, "generate_cubes")


def test_init_phy_generate_shape():
    import songs
    g = songs.init_phy(n_gals=1, fov=30, spatial_resolution=3.0,
                       spectral_resolution=20, seed=7)
    results = g.generate_cubes()
    assert len(results) >= 1
    cube, _ = results[0]
    assert cube.ndim == 3


# ---------------------------------------------------------------------------
# Diffuse params override
# ---------------------------------------------------------------------------

def test_diffuse_params_override_accepted():
    import songs
    g = songs.init(
        n_gals=2, n_cubes=1, grid_size=48, n_spectral_slices=10, seed=8,
        diffuse_params={"halo_Se_factor": 0.05, "tail_Se_factor": 0.5},
    )
    results = g.generate_cubes()
    assert len(results) == 1
