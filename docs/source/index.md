# SONGS — Spectral Observations of Non-stationary Galactic Structures

<p align="center">
  <img class="only-light"
       src="https://raw.githubusercontent.com/arnablahiry/SONGS-spectral-cube-simulator/main/assets/songs_banner.png"
       alt="SONGS banner"
       width="100%" />
  <img class="only-dark"
       src="https://raw.githubusercontent.com/arnablahiry/SONGS-spectral-cube-simulator/main/assets/songs_banner_dark.png"
       alt="SONGS banner"
       width="100%" />
</p>

**SONGS** is a high-fidelity simulator for synthetic IFU (Integral Field Unit) spectral cubes of disk galaxies,
interacting systems, and diffuse low-surface-brightness features including halos, bridges, and tidal tails.

It combines analytic Sérsic profiles, exponential vertical structure, empirical rotation curves, viewing-angle
projections, instrument beam convolution, and a physically motivated diffuse emission model to produce
3-D spectral datacubes suitable for algorithm development, denoising benchmarks, and transfer-learning
pre-training.

---

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 1
:caption: User Guide

gui
```

```{toctree}
:maxdepth: 1
:caption: Physics & Model Reference

physics
```

```{toctree}
:maxdepth: 2
:caption: API Reference

generator
api_gui
visualise
utilities
```
