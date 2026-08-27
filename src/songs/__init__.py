"""SONGS — Spectral Observations of Non-stationary Galactic Structures

Public API for the ``songs`` package. Exposes two complementary generators
for different use cases:

- Population-based generator in pixel units (``SONGS``): designed for
    statistical sampling and batch generation of synthetic galaxy cubes.
- Physically parameterised generator in explicit units (``SONGSPhy``):
    designed for interactive construction of single systems, e.g. via the GUI.

Thin convenience wrappers (``init`` and ``init_phy``) are provided::

    import songs
    g = songs.init(n_gals=1, n_cubes=1, grid_size=125, n_spectral_slices=40)
"""

from .core import SONGS, SONGSPhy

__all__ = [
    "SONGS",
    "SONGSPhy",
    "init",
    "init_phy",
]


def init_phy(
    n_gals=None,
    n_cubes=1,
    spatial_resolution=4.5,
    spectral_resolution=10,
    offset_gals=20,
    beam_info=[18, 18, 0],
    fov=125,
    n_sersic=None,
    save=False,
    fname=None,
    verbose=True,
    seed=None,
    diffuse_params=None,
    allow_overlap=False,
):
    """
    Create and return a physically parameterised :class:`SONGSPhy` instance.

    This entry point is designed for **explicit physical-unit parametrisation**
    (e.g. kpc, km s⁻¹) and is primarily intended for interactive usage through
    the graphical user interface. In this mode, galaxies are specified via
    directly interpretable physical quantities, enabling the construction of
    highly controlled, single systems.

    Unlike the population-based generator, parameters are not drawn from
    statistical distributions but are instead provided explicitly, making this
    interface well suited for exploratory modelling, demonstrations, and
    pedagogical use.

    Parameters
    ----------
    All parameters are forwarded directly to the :class:`SONGSPhy`
    constructor. Refer to the class documentation for detailed descriptions
    of each physical quantity. Notably:

    offset_gals : float or (float, float)
        Distance of each satellite from the central galaxy, in kpc. A
        scalar is the max offset (min 0); a ``(min, max)`` tuple bounds it
        on both ends. This is always a true 3D distance, not the on-sky
        projected distance.
    allow_overlap : bool, default False
        If ``False`` (default), satellites are kept apart from the central
        galaxy and each other by a small separation floor so their images
        don't literally coincide. If ``True``, that floor is dropped and
        satellites may overlap, while still respecting ``offset_gals``.

    Returns
    -------
    SONGSPhy
        An initialized SONGSPhy generator configured for physical-unit
        parametrisation.

    Example
    -------
    >>> import songs
    >>> g = songs.init_phy(n_gals=3, offset_gals=(20, 60), allow_overlap=True)
    """
    return SONGSPhy(
        n_gals,
        n_cubes,
        spatial_resolution,
        spectral_resolution,
        offset_gals,
        beam_info,
        fov,
        n_sersic,
        save,
        fname,
        verbose,
        seed,
        diffuse_params,
        allow_overlap=allow_overlap,
    )


def init(
    n_gals=None,
    n_cubes=1,
    resolution="all",
    offset_gals=5,
    beam_info=[4, 4, 0],
    grid_size=125,
    n_spectral_slices=40,
    n_sersic=None,
    save=False,
    fname=None,
    verbose=True,
    seed=None,
    diffuse_params=None,
    allow_overlap=False,
):
    """
    Create and return a population-based :class:`SONGS` instance.

    This entry point corresponds to the **standard SONGS generator**,
    operating primarily in **pixel units**. It is designed to generate
    ensembles of synthetic galaxy cubes in which parameter values are drawn
    from uniform distributions over physically viable ranges.

    This mode is intended for statistical studies, machine-learning training
    set generation, and large-scale benchmarking, where diversity across
    many realisations is more important than fine control over individual
    systems.

    Parameters
    ----------
    All parameters are forwarded directly to the :class:`SONGS`
    constructor. Refer to the class documentation for details on the sampling
    strategy and parameter ranges. Notably:

    offset_gals : float or (float, float)
        Distance of each satellite from the central galaxy, in pixels. A
        scalar is the max offset (min 0); a ``(min, max)`` tuple bounds it
        on both ends. This is always a true 3D distance, not the on-sky
        projected distance.
    allow_overlap : bool, default False
        If ``False`` (default), satellites are kept apart from the central
        galaxy and each other by a small separation floor so their images
        don't literally coincide. If ``True``, that floor is dropped and
        satellites may overlap, while still respecting ``offset_gals``.

    Returns
    -------
    SONGS
        An initialized SONGS generator configured for population-based
        sampling in pixel space.

    Example
    -------
    >>> import songs
    >>> g = songs.init(n_gals=3, offset_gals=(30, 90), allow_overlap=True)
    """
    return SONGS(
        n_gals,
        n_cubes,
        resolution,
        offset_gals,
        beam_info,
        grid_size,
        n_spectral_slices,
        n_sersic,
        save,
        fname,
        verbose,
        seed,
        diffuse_params,
        allow_overlap=allow_overlap,
    )
