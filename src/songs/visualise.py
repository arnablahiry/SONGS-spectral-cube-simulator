"""Plotting helpers for SONGS results.

This module provides small, focused plotting utilities that operate on the
``results`` list produced by :meth:`~SONGS.core.SONGS.generate_cubes`.
Each public helper accepts the ``results`` container and an index selecting
which generated cube to visualise. The functions are intentionally lightweight
and return a Matplotlib ``(fig, ax)`` pair so callers (GUIs, scripts, tests)
can further customise or save figures.

Dependencies
------------
- matplotlib (this module sets the ``TkAgg`` backend by default)
- astrodendro (used to compute a crude mask for visual guides)

Notes
-----
- These helpers call :func:`_prepare_cube` to extract cube / metadata and to
    compute a simple dendrogram-based mask used for moment maps. The dendrogram
    parameters are deliberately conservative and may be tuned for different
    signal-to-noise regimes.
- The plotting functions attempt to save to ``figures/<shape>/`` when
    ``save=True`` is passed; save failures are intentionally ignored to keep
    UI flows robust.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
import matplotlib.pyplot as plt

# Only force TkAgg if we aren't already in an inline environment
if not matplotlib.get_backend().lower().startswith('inline'):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astrodendro import Dendrogram
from .utils import convolve_beam, add_beam
import os
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def _prepare_cube(data, idx):
    """Internal helper: extract cube and derived metadata for plotting.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
        Each element should be a ``(cube, meta)`` tuple where ``cube`` is a
        NumPy array of shape ``(n_vel, ny, nx)`` and ``meta`` is a dict with
        keys including ``'beam_info'``, ``'average_vels'``, and
        ``'pix_spatial_scale'``.
    idx : int
        Index of the cube to extract.

    Returns
    -------
    cube : ndarray
        The spectral cube selected (shape ``n_vel x ny x nx``).
    meta : dict
        The metadata dictionary stored alongside the cube.
    beam_info : sequence
        Beam description (bmin_px, bmaj_px, bpa) as provided in ``meta``.
    vels : ndarray
        Velocity axis (km/s) corresponding to the spectral channels.
    pix_spatial_scale : float
        Physical scale per pixel (kpc/pixel).
    del_V : float
        Mean width of a spectral channel in km/s (used when computing
        moment0 units).
    moment_cube : ndarray
        The cube multiplied by the velocity axis (useful when computing
        first moment / intensity-weighted velocity).
    mask : ndarray (bool)
        A conservative mask derived from a dendrogram computed on the cube;
        intended to highlight contiguous emission regions for plotting.

    Notes
    -----
    - The dendrogram mask is produced with a heuristic threshold (0.25 x
      cube.std()) and may be noisy for very low S/N cubes. The mask is used
      to focus moment computations and to identify significant structures.
    """

    cube, meta = data[idx]
    beam_info = meta['beam_info']
    vels = meta['average_vels']
    pix_spatial_scale = meta['pix_spatial_scale']
    del_V = np.diff(vels).mean()
    moment_cube = cube * vels[:, np.newaxis, np.newaxis]

    # Create a conservative mask from a dendrogram to help moment maps
    mask = np.zeros(cube.shape, dtype=bool)
    dendro = Dendrogram.compute(cube,
                                min_value=0.25 * cube.std(),
                                min_delta=cube.std(),
                                verbose=False)
    for trunk in dendro.trunk:
        mask |= trunk.get_mask()

    return cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask

def moment0(data, idx, save=False, fname_save=None, inline=False):
    """Plot the zeroth moment (integrated intensity) of a spectral cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
    idx : int
        Index selecting which cube to plot.
    save : bool, optional
        If True, attempt to save the figure to ``figures/<shape>/moment0.pdf``.
    fname_save : str or None, optional
        Optional directory to save the figure. If None a path under the
        current working directory is chosen automatically.

    Returns
    -------
    fig, ax : (Figure, Axes)
        Matplotlib figure and axes objects containing the rendered moment map.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)
    ny, nx = cube.shape[1], cube.shape[2]
    extent = [0, nx, 0, ny]

    # Compute fixed colour limits from the integrated (moment0) map so the
    # slice viewer uses a consistent scale across channels.
    integrated = cube.sum(axis=0) * del_V
    vmin = float(np.nanmin(integrated))
    vmax = float(np.nanmax(integrated))

    fig, ax = plt.subplots(figsize=(5,5))
    # Set a descriptive window title where the backend/window manager
    # exposes a canvas manager (e.g., TkAgg). Wrap in try/except for
    # environments where this attribute is not available.
    if not inline:
        try:
            fig.canvas.manager.set_window_title('Moment 0')
        except Exception:
            try:
                # Older matplotlib versions expose a different attribute
                fig.canvas.set_window_title('Moment 0')
            except Exception:
                pass
        except Exception:
            pass
    im = ax.imshow(cube.sum(axis=0) * del_V, cmap='RdBu_r', origin='lower', extent=extent, vmin=0, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.2)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label=r'$\rm Jy\;beam^{-1} \cdot km\;s^{-1}$', format='%.2f')
    # Place label and ticks on the top and draw ticks outward from the
    # colorbar so they appear above the bar (consistent with Moment0).
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.label.set_size(14)
    # Make ticks point outwards and add a small pad so label is above ticks
    cb.ax.tick_params(labelsize=12, direction='out', pad=6)
    cb.ax.xaxis.labelpad = 12
    ax.text(nx*0.05, ny*0.89, 'Moment 0', color='white', fontsize=13, weight='bold')
    add_beam(ax, beam_info[0], beam_info[1], beam_info[2], xy_offset=(6,6), color='white')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Scalebar
    scalebar = (25/72)*cube.shape[2]
    x0, y0 = nx*0.6, ny*0.07
    ax.plot([x0, x0+scalebar], [y0, y0], color='white', lw=2)
    ax.text(x0+scalebar/2, y0 + ny*0.03, f'{scalebar*pix_spatial_scale:.1f} kpc',
            color='white', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()

    if save:
        if fname_save is None:
            fname_save = os.path.join(os.getcwd(), 'figures', f'{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}')
        os.makedirs(fname_save, exist_ok=True)
        try:
            fig.savefig(os.path.join(fname_save, 'moment0.pdf'), bbox_inches='tight')
        except Exception:
            pass

    # Final interactive trigger
    if not inline:
        fig.show()
    return fig, ax


def moment1(data, idx, save=False, fname_save=None):
    """Plot the first moment (intensity-weighted velocity) of a spectral cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
    idx : int
        Index selecting which cube to plot.
    save : bool, optional
        If True, attempt to save the figure to ``figures/<shape>/moment1.pdf``.
    fname_save : str or None, optional
        Optional directory to save the figure. If None a path under the
        current working directory is chosen automatically.

    Returns
    -------
    fig, ax : (Figure, Axes)
        Matplotlib figure and axes objects containing the rendered moment map.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)
    ny, nx = cube.shape[1], cube.shape[2]
    extent = [0, nx, 0, ny]

    numerator = (mask * moment_cube).sum(axis=0)
    denominator = (mask * cube).sum(axis=0)
    ratio = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator != 0)
    ratio = np.clip(ratio, float(vels.min()), float(vels.max()))
    vmax = np.max([np.abs(np.nanmin(ratio)), np.abs(np.nanmax(ratio))])

    fig, ax = plt.subplots(figsize=(5,5))
    try:
        fig.canvas.manager.set_window_title('Moment 1')
    except Exception:
        try:
            fig.canvas.set_window_title('Moment 1')
        except Exception:
            pass
    im = ax.imshow(ratio, cmap='RdBu_r', origin='lower', extent=extent, vmin=-vmax, vmax=vmax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.2)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label=r'$\rm km\;s^{-1}$', format='%.0f')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize=12)
    cb.ax.xaxis.label.set_size(14)
    cb.ax.xaxis.labelpad = 10
    ax.text(nx*0.05, ny*0.89, 'Moment 1', color='black', fontsize=13, weight='bold')
    add_beam(ax, beam_info[0], beam_info[1], beam_info[2], xy_offset=(6*cube.shape[1]/72,6*cube.shape[1]/72), color='black')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    # Scalebar
    scalebar = (25/72)*cube.shape[2]
    x0, y0 = nx*0.6, ny*0.07
    ax.plot([x0, x0+scalebar], [y0, y0], color='black', lw=2)
    ax.text(x0+scalebar/2, y0 + ny*0.03, f'{scalebar*pix_spatial_scale:.1f} kpc',
            color='black', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()
    if save:
        if fname_save is None:
            fname_save = os.path.join(os.getcwd(), 'figures', f'{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}')
        os.makedirs(fname_save, exist_ok=True)
        try:
            fig.savefig(os.path.join(fname_save, 'moment1.pdf'), bbox_inches='tight')
        except Exception:
            pass

    fig.show()
    return fig, ax


def spectrum(data, idx, save=False, fname_save=None):
    """Plot the integrated spectrum (total flux vs velocity) for a cube.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
    idx : int
        Index selecting which cube to plot.
    save : bool, optional
        If True, save the figure as ``spectrum.pdf`` under
        ``figures/<shape>/`` unless ``fname_save`` overrides the path.
    fname_save : str or None, optional
        Optional directory to save the figure.

    Returns
    -------
    fig, ax : (Figure, Axes)
        The Matplotlib figure and axes containing the spectrum.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)

    fig, ax = plt.subplots(figsize=(7,4.5))
    try:
        fig.canvas.manager.set_window_title('Integrated LOS Spectrum')
    except Exception:
        try:
            fig.canvas.set_window_title('Integrated LOS Spectrum')
        except Exception:
            pass
    ax.plot(vels, np.sum(cube, axis=(1,2)), color='xkcd:blue', linewidth=1.2)
    ax.set_ylabel(r'Flux Density ($\rm Jy\;beam^{-1}$)', fontsize=15, labelpad=10)
    ax.set_xlabel(r'(Line-of-sight) Velocity ($\rm km\;s^{-1}   $)', fontsize=15, labelpad=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True)
    plt.tight_layout()

    if save:
        if fname_save is None:
            fname_save = os.path.join(os.getcwd(), 'figures', f'{cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}')
        os.makedirs(fname_save, exist_ok=True)
        try:
            fig.savefig(os.path.join(fname_save, 'spectrum.pdf'), bbox_inches='tight')
        except Exception:
            pass

    fig.show()
    return fig, ax


# ---------------------------------------------------------------------------
# Colour constants — SONGS theme (black + faint yellow)
# ---------------------------------------------------------------------------
_BG           = "#0a0a0a"   # window background
_CARD_BG      = "#111111"   # control card / sidebar background
_ACCENT       = "#b8960a"   # faint yellow — headings, active elements
_ACCENT_HOV   = "#f0c040"   # bright yellow on hover
_DIM          = "#2e2000"   # very dark yellow — borders
_DIM_TXT      = "#3a3010"   # dimmed text
_LOG_BG       = "#0a0a0a"   # matplotlib figure background
_STEP_LBL     = "#999999"   # secondary label text

# Current viewer theme — toggled by the GUI's theme button
_VIEWER_THEME = 'light'

# Palette mirrors the main GUI's _THEMES exactly so the viewer's sliders,
# pills, and accents match the main cards 1:1.
_VIEWER_PALETTES = {
    'dark': dict(
        bg='#0a0a0a', card_bg='#111111', accent='#d4aa40', accent_hov='#f0c040',
        dim='#5e4200', dim_txt='#3a3010', log_bg='#0a0a0a', step_lbl='#999999',
        slider_border='#5e4200', fg_on_accent='#000000', entry_bg='#1a1a1a',
        trough='#111111', pill_nor='#1e1e1e', pill_hov='#2e2700',
        sym_fg='white', sm_border='#201800',
    ),
    'light': dict(
        bg='#f0ede6', card_bg='#ffffff', accent='#9a7200', accent_hov='#e8c040',
        dim='#c8a030', dim_txt='#9a8050', log_bg='#ffffff', step_lbl='#444444',
        slider_border='#c8a030', fg_on_accent='#ffffff', entry_bg='#ffffff',
        trough='#ffffff', pill_nor='#dedad0', pill_hov='#f5e898',
        sym_fg='#222222', sm_border='#e0d5b0',
    ),
}

def _apply_window_appearance(win, theme=None):
    """Force a window's macOS title bar to match the app theme.

    Mirrors SONGSGUI._set_window_appearance so Toplevel viewers get the same
    themed title bar as the main window instead of following the system
    light/dark setting. No-op off macOS or without PyObjC.
    """
    try:
        import AppKit
        # Deliberately NO update_idletasks() here: forcing idle processing
        # realises and MAPS the window, which made blank placeholder windows
        # flash on screen before they were ready (the log window is created
        # then immediately withdrawn; the main window is themed before its
        # cards are built). If the NSWindow doesn't exist yet we simply do
        # nothing — every caller also schedules a delayed retry, by which
        # point the mainloop has realised the window on its own.
        title = str(win.title())
        nswin = next((w for w in AppKit.NSApplication.sharedApplication().windows()
                      if str(w.title()) == title), None)
        if nswin is None:
            return
        theme = theme or _VIEWER_THEME
        name = (AppKit.NSAppearanceNameAqua if theme == 'light'
                else AppKit.NSAppearanceNameDarkAqua)
        nswin.setAppearance_(AppKit.NSAppearance.appearanceNamed_(name))
    except Exception:
        # Non-macOS / PyObjC unavailable — leave the title bar as-is.
        pass


_CMAPS = ["inferno", "viridis", "magma", "plasma", "cividis",
          "gray", "hot", "afmhot", "YlOrRd", "cubehelix"]

# Distinct palette for up to 8 sources (matches matplotlib tab10 first 8)
_SRC_PALETTE = [
    (0.122, 0.467, 0.706),  # blue      — central
    (1.000, 0.498, 0.055),  # orange    — sat 1
    (0.173, 0.627, 0.173),  # green     — sat 2
    (0.839, 0.153, 0.157),  # red       — sat 3
    (0.580, 0.404, 0.741),  # purple    — sat 4
    (0.549, 0.337, 0.294),  # brown     — sat 5
    (0.890, 0.467, 0.761),  # pink      — sat 6
    (0.498, 0.498, 0.498),  # grey      — sat 7
]


def _src_label(i: int) -> str:
    return "Central Galaxy" if i == 0 else f"Satellite {i}"


def _rgb_to_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))


def _lighten(rgb, amount=0.3):
    return tuple(min(c + amount, 1.0) for c in rgb)


def _main_slider(parent, label, var, from_, to, pal,
                 resolution=0.01, fmt="{:.2f}", integer=False):
    """Replica of SONGSGUI.make_slider so viewer sliders match the main cards
    exactly (soft-accent border, label above, editable value entry, snapping)."""
    bg          = pal['card_bg']
    fg          = pal['step_lbl']
    acc         = pal['accent']
    trough      = pal['trough']
    _entry_bg   = pal['entry_bg']
    _border_col = pal['slider_border']
    _thumb      = pal['accent']
    _thumbh     = pal['accent_hov']

    _wrap = tk.Frame(parent, bg=_border_col, padx=1, pady=1)
    fr = tk.Frame(_wrap, bg=bg)
    fr.pack(fill='both', expand=True)
    if label:
        tk.Label(fr, text=label, bg=bg, fg=fg,
                 font=("Helvetica", 8)).pack(anchor='w', pady=(0, 2))
    slider_row = tk.Frame(fr, bg=bg)
    slider_row.pack(fill='x')

    entry_var = tk.StringVar(value=fmt.format(var.get()) if not integer
                             else str(int(var.get())))
    entry = tk.Entry(slider_row, textvariable=entry_var, width=6, justify='right',
                     bg=_entry_bg, fg=acc, insertbackground=acc,
                     relief='flat', highlightthickness=1,
                     highlightbackground=_border_col, highlightcolor=acc,
                     font=("Helvetica", 8), bd=2)
    entry.pack(side='right', padx=(4, 0))

    scale = tk.Scale(slider_row, from_=from_, to=to, orient='horizontal',
                     resolution=resolution, bg=_thumb, fg=fg, troughcolor=trough,
                     activebackground=_thumbh, highlightthickness=0,
                     sliderrelief='flat', bd=0, showvalue=False, relief='flat', width=6)
    scale.pack(side='left', fill='x', expand=True)
    step = resolution if resolution else 0.01
    busy = {'val': False}

    def snap(v):
        if integer:
            return int(round(float(v)))
        nsteps = round((float(v) - from_) / step)
        return from_ + nsteps * step

    def _fmt(v):
        try:    return fmt.format(v)
        except Exception: return str(v)

    def update(v):
        if busy['val']:
            return
        busy['val'] = True
        v_snap = snap(v)
        try: var.set(v_snap)
        except Exception: pass
        entry_var.set(_fmt(v_snap))
        try: scale.set(v_snap)
        except Exception: pass
        busy['val'] = False

    scale.configure(command=update)
    try: scale.set(var.get())
    except Exception: scale.set(from_)

    def _commit_entry(*_):
        if busy['val']:
            return
        try:
            v = max(from_, min(to, snap(float(entry_var.get().strip()))))
            busy['val'] = True
            var.set(v); entry_var.set(_fmt(v)); scale.set(v)
            busy['val'] = False
        except (ValueError, tk.TclError):
            pass

    entry.bind('<Return>',   _commit_entry)
    entry.bind('<FocusOut>', _commit_entry)

    def _var_trace(*_):
        if busy['val']:
            return
        busy['val'] = True
        v = var.get()
        entry_var.set(_fmt(v))
        try: scale.set(v)
        except Exception: pass
        busy['val'] = False
    try:
        var.trace_add('write', _var_trace)
    except Exception:
        var.trace('w', _var_trace)
    return _wrap


# ---------------------------------------------------------------------------
# Shared viewer widgets — used by both SliceViewer and AnalysisViewer so the
# two windows render cards, symbol sliders, and source pills identically.
# ---------------------------------------------------------------------------
_PILL_H = 21


def _find_scale(widget):
    """Recursively find the first tk.Scale inside a widget tree — used to
    reconfigure a slider's from_/to bounds after construction (e.g. when
    the underlying data range changes, as with the Clean/Noisy toggle)."""
    if isinstance(widget, tk.Scale):
        return widget
    for c in widget.winfo_children():
        found = _find_scale(c)
        if found is not None:
            return found
    return None


def _viewer_slider(parent, master, pal, label, var, lo, hi,
                   resolution=0.01, fmt="{:.2f}", integer=False):
    """Prefer the real SONGSGUI.make_slider when the master is the app; else
    fall back to the module-level replica so the slider matches the main cards."""
    _gui_make = getattr(master, "make_slider", None)
    if callable(_gui_make):
        return _gui_make(parent, label, var, lo, hi,
                         resolution=resolution, fmt=fmt, integer=integer)
    return _main_slider(parent, label, var, lo, hi, pal,
                        resolution=resolution, fmt=fmt, integer=integer)


def _viewer_small_card(parent, pal, title=None):
    """Replica of the main GUI's small_card (soft border + card body + heading)."""
    outer = tk.Frame(parent, bg=pal['sm_border'], padx=1, pady=1)
    outer.pack(fill='x', padx=8, pady=4)
    inner = tk.Frame(outer, bg=pal['card_bg'], padx=5, pady=6)
    inner.pack(fill='both', expand=True)
    if title:
        tk.Label(inner, text=title, bg=pal['card_bg'], fg=pal['step_lbl'],
                 font=("Helvetica", 8), anchor='w',
                 justify='left').pack(anchor='w', pady=(2, 5))
    return inner


def _viewer_sym_slider_row(parent, master, pal, segs, var, lo, hi,
                           resolution=0.01, fmt="{:.2f}", integer=False):
    """A rich_label symbol on the left + a SONGS-style slider on the right."""
    try:
        from .gui import rich_label as _rich
    except Exception:
        _rich = None
    row = tk.Frame(parent, bg=pal['card_bg'])
    row.pack(fill='x', pady=(0, 4))
    if _rich is not None:
        sym = _rich(row, segs, bg=pal['card_bg'], fg=pal['sym_fg'])
    else:
        sym = tk.Label(row, text="".join(t for t, _ in segs),
                       bg=pal['card_bg'], fg=pal['sym_fg'],
                       font=("Georgia", 10, "italic"))
    sym.pack(side='left', padx=(0, 4))
    _viewer_slider(row, master, pal, "", var, lo, hi, resolution=resolution,
                   fmt=fmt, integer=integer).pack(side='left', fill='x', expand=True)
    return row


def _viewer_pill(parent, pal, label, var, on_toggle, *, swatch="■",
                 swatch_color=None, pill_w=None, pill_font=None):
    """Clickable pill (accent = active) with a coloured swatch to its left.

    Mirrors the SliceViewer source pills: fixed-width canvas, pointing-hand
    cursor, hover highlight, toggles ``var`` then calls ``on_toggle``. Returns
    the pill's ``_redraw`` callable so external state changes can refresh it."""
    from tkinter import font as _tkfont
    _bg       = pal['bg']
    _accent   = pal['accent']
    _pill_nor = pal['pill_nor']
    _pill_hov = pal['pill_hov']
    _fg_on    = pal['fg_on_accent']
    _step     = pal['step_lbl']
    f = pill_font or _tkfont.Font(family="Helvetica", size=8, weight="bold")
    w = pill_w or (f.measure(label) + 16)

    prow = tk.Frame(parent, bg=_bg)
    prow.pack(fill=tk.X, pady=2, anchor="w")
    tk.Label(prow, text=swatch, bg=_bg, fg=swatch_color or _accent,
             font=("Helvetica", 9, "bold")).pack(side=tk.LEFT, padx=(0, 3))
    cv = tk.Canvas(prow, width=w, height=_PILL_H, highlightthickness=0,
                   bd=0, bg=_bg, cursor="pointinghand")
    cv.pack(side=tk.LEFT)

    def _redraw(hover=False):
        cv.delete("all")
        active  = var.get()
        fill    = _accent if active else (_pill_hov if hover else _pill_nor)
        txt_col = _fg_on if active else _step
        cv.create_rectangle(0, 0, w, _PILL_H, fill=fill, outline=fill)
        cv.create_text(w // 2, _PILL_H // 2, text=label, fill=txt_col, font=f)

    def _toggle(_e=None):
        var.set(not var.get())
        _redraw()
        on_toggle()

    cv.bind("<ButtonRelease-1>", _toggle)
    cv.bind("<Enter>", lambda _e: _redraw(hover=True) if not var.get() else None)
    cv.bind("<Leave>", lambda _e: _redraw())
    _redraw()
    return _redraw


def _viewer_pill_group(parent, pal, values, var, on_select, *, pill_w=None,
                       pill_font=None, bg=None):
    """Mutually-exclusive row of pills (accent = active), one per entry in
    ``values``. ``var`` is a ``tk.StringVar``; clicking a pill sets it and
    calls ``on_select()``. Generalizes the inline norm-mode pill pattern
    (SliceViewer) so it can be reused for e.g. a Clean/Noisy toggle in both
    SliceViewer and AnalysisViewer."""
    from tkinter import font as _tkfont
    _accent   = pal['accent']
    _pill_nor = pal['pill_nor']
    _pill_hov = pal['pill_hov']
    _fg_on    = pal['fg_on_accent']
    _step     = pal['step_lbl']
    _bg       = bg if bg is not None else pal['card_bg']
    f = pill_font or _tkfont.Font(family="Helvetica", size=8, weight="bold")
    w = pill_w or (max(f.measure(v) for v in values) + 16)
    redraws: list = []

    def _make_pill(value):
        # Nested function call (not a bare for-loop body) so each pill gets
        # its own closure scope — a for-loop body would rebind `_redraw`/
        # `_select` in the *shared* enclosing scope each iteration, leaving
        # every <Enter>/<Leave> handler pointing at the last pill only.
        cv = tk.Canvas(parent, width=w, height=_PILL_H, highlightthickness=0,
                       bd=0, bg=_bg, cursor="pointinghand")
        cv.pack(side=tk.LEFT, padx=2)

        def _redraw(hover=False):
            cv.delete("all")
            active  = var.get() == value
            fill    = _accent if active else (_pill_hov if hover else _pill_nor)
            txt_col = _fg_on if active else _step
            cv.create_rectangle(0, 0, w, _PILL_H, fill=fill, outline=fill)
            cv.create_text(w // 2, _PILL_H // 2, text=value, fill=txt_col, font=f)

        def _select(_e=None):
            var.set(value)
            for r in redraws:
                r()
            on_select()

        cv.bind("<ButtonRelease-1>", _select)
        cv.bind("<Enter>", lambda _e: _redraw(hover=True) if var.get() != value else None)
        cv.bind("<Leave>", lambda _e: _redraw())
        redraws.append(_redraw)
        _redraw()

    for value in values:
        _make_pill(value)
    return redraws


class SliceViewer(tk.Toplevel):
    """Channel-by-channel IFU slice viewer matching nemo aesthetics.

    Displays the full spectral cube with colormap / normalization controls,
    vmin/vmax sliders, and (when per-galaxy cubes are available) a sources
    sidebar with per-source contours, bounding boxes, and intensity-threshold
    masks whose threshold percentage is tunable via a dedicated slider.
    """

    def __init__(self, master, data, idx: int = 0, noisy_data=None):
        super().__init__(master)

        # Resolve palette from current viewer theme
        _p          = _VIEWER_PALETTES[_VIEWER_THEME]
        _bg         = _p['bg']
        _card_bg    = _p['card_bg']
        _accent     = _p['accent']
        _accent_hov = _p['accent_hov']
        _dim        = _p['dim']
        _dim_txt    = _p['dim_txt']
        _log_bg     = _p['log_bg']
        _step_lbl   = _p['step_lbl']

        self.configure(bg=_bg)
        self.resizable(True, True)
        self.title("SONGS — IFU Slice Viewer")
        # Themed title bar, matching the main window. Re-applied on a short
        # delay too, since the NSWindow may not exist yet on the first call.
        _apply_window_appearance(self)
        self.after(250, lambda: _apply_window_appearance(self))

        # ── Unpack data ──────────────────────────────────────────────────────
        # ``noisy_data`` (same shape/idx convention as ``data``) is optional —
        # only present when "Use Noise?" is Yes. Both versions of the TOTAL
        # cube are kept; per-source component arrays (per_galaxy_cubes) stay
        # the clean ground truth in both cases, since noise is an instrument
        # effect on the observed total, not a per-source quantity.
        cube, meta = data[idx]
        self._cube_clean  = cube.astype(np.float32)
        self._cube_noisy  = None
        if noisy_data:
            noisy_cube, _noisy_meta = noisy_data[idx]
            self._cube_noisy = noisy_cube.astype(np.float32)
        # Default to noisy when available — matches what would actually be
        # observed/saved.
        self._cube         = self._cube_noisy if self._cube_noisy is not None else self._cube_clean
        self._vels        = np.asarray(meta.get("average_vels", np.arange(cube.shape[0])))
        self._beam        = meta.get("beam_info")
        self._pix_scale   = float(meta.get("pix_spatial_scale", 1.0))
        pg = meta.get("per_galaxy_cubes")
        self._per_gal     = np.asarray(pg) if pg is not None else None
        self._n_gals      = int(self._per_gal.shape[0]) if self._per_gal is not None else 0

        n_ch, ny, nx = self._cube.shape
        self._channels = list(range(n_ch))
        VW = 560

        flat             = self._cube.ravel()
        self._data_min   = float(np.nanmin(flat))
        self._data_max   = float(np.nanmax(flat))

        # ── Matplotlib figure (colorbar via axes_locatable, like NEMO) ────────
        # Extra width (+100, was +60) and a narrower main-axes fraction
        # (0.86, was 0.90) give the vertical colorbar's rotated unit label
        # enough room that it doesn't get clipped at the figure's right edge.
        self._fig   = plt.Figure(figsize=((VW+100)/96, VW/96), dpi=96, facecolor=_log_bg)
        self._ax    = self._fig.add_axes([0.02, 0.02, 0.86, 0.96])
        self._ax.set_xticks([]); self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_dim_txt)
            spine.set_linewidth(0.8)
        _div        = make_axes_locatable(self._ax)
        self._ax_cb = _div.append_axes("right", size="4%", pad=0.1)
        self._ax_cb.set_facecolor(_log_bg)

        # ── Layout: sidebar + canvas ─────────────────────────────────────────
        top = tk.Frame(self, bg=_bg)
        top.pack(fill=tk.BOTH, expand=True)

        # Store palette for use in _draw and helpers
        self._pal = _p

        # Slider factory — delegates to the shared helper so the viewer's
        # sliders match the main GUI's cards exactly.
        from tkinter import font as _tkfont
        def _slider(parent, label, var, lo, hi, resolution=0.01,
                    fmt="{:.2f}", integer=False):
            return _viewer_slider(parent, self.master, _p, label, var, lo, hi,
                                  resolution=resolution, fmt=fmt, integer=integer)

        # Threshold var initialised before sidebar so left panel can reference it
        self._thresh_var = tk.DoubleVar(value=5.0)
        # Power-law exponent for the "power" normalization mode
        self._gamma_var  = tk.DoubleVar(value=0.5)

        # ── Sources sidebar: fixed-width clickable pills (accent = active) ────
        self._src_visible: dict[int, tk.BooleanVar] = {}
        self._src_pill_redraw: list = []
        if self._per_gal is not None:
            sb = tk.Frame(top, bg=_bg, width=140)
            sb.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 2), pady=4)
            sb.pack_propagate(False)

            tk.Label(sb, text="Sources", bg=_bg, fg=_accent,
                     font=("Helvetica", 9, "bold")).pack(pady=(4, 6), anchor="w")

            _pill_font = _tkfont.Font(family="Helvetica", size=8, weight="bold")
            # Fixed pill width sized to the longest label so all pills match
            _all_labels = [_src_label(i) for i in range(self._n_gals)]
            _PILL_W = max((_pill_font.measure(lbl) for lbl in _all_labels), default=80) + 16

            def _on_src_toggle():
                self._draw()
                self._update_spectrum_data()

            for i in range(self._n_gals):
                hex_col = _rgb_to_hex(_SRC_PALETTE[i % len(_SRC_PALETTE)])
                self._src_visible[i] = tk.BooleanVar(value=True)
                _redraw = _viewer_pill(sb, _p, _src_label(i), self._src_visible[i],
                                       _on_src_toggle, swatch="■", swatch_color=hex_col,
                                       pill_w=_PILL_W, pill_font=_pill_font)
                self._src_pill_redraw.append(_redraw)

        self._canvas = FigureCanvasTkAgg(self._fig, master=top)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── Control bar: colormap + invert + norm ────────────────────────────
        ctrl = tk.Frame(self, bg=_card_bg)
        ctrl.pack(fill=tk.X, padx=0, pady=0)
        tk.Frame(ctrl, bg=_dim, height=1).pack(fill=tk.X)
        inner_ctrl = tk.Frame(ctrl, bg=_card_bg)
        inner_ctrl.pack(fill=tk.X, padx=10, pady=4)

        # Clean/Noisy toggle — only shown when a noisy preview is available
        # ("Use Noise?" was Yes at generation time). Swaps which TOTAL cube
        # self._cube points to and refreshes both the channel image and the
        # spectrum panel (whose curves are cached, not recomputed by _draw()).
        if self._cube_noisy is not None:
            tk.Label(inner_ctrl, text="View:", bg=_card_bg, fg=_step_lbl,
                     font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 4))
            self._view_mode = tk.StringVar(value="Noisy")

            def _on_view_mode_change():
                self._cube = (self._cube_noisy if self._view_mode.get() == "Noisy"
                             else self._cube_clean)
                # Re-derive the colour-scale range from whichever cube is
                # now active — clean and noisy cubes have different peak/
                # noise-floor values, so both the V_min/V_max sliders'
                # bounds and their current values need to follow the
                # switch, not stay pinned to the cube that was active at
                # __init__ time.
                flat = self._cube.ravel()
                self._data_min = float(np.nanmin(flat))
                self._data_max = float(np.nanmax(flat))
                for scale in (self._vmin_scale, self._vmax_scale):
                    if scale is not None:
                        scale.configure(from_=self._data_min, to=self._data_max)
                self._vmin_var.set(self._data_min)
                self._vmax_var.set(self._data_max)
                self._spec_total = np.nansum(self._cube, axis=(1, 2))
                self._redraw_spectrum()
                self._draw()

            _viewer_pill_group(inner_ctrl, _p, ("Clean", "Noisy"), self._view_mode,
                               _on_view_mode_change, bg=_card_bg)
            tk.Frame(inner_ctrl, bg=_card_bg, width=8).pack(side=tk.LEFT)

        tk.Label(inner_ctrl, text="Colormap:", bg=_card_bg, fg=_step_lbl,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 3))
        self._cmap     = tk.StringVar(value="inferno")
        self._inverted = tk.BooleanVar(value=False)
        om = tk.OptionMenu(inner_ctrl, self._cmap, *_CMAPS, command=lambda _v: self._draw())
        om.configure(bg=_card_bg, fg=_accent,
                     activebackground=_dim, activeforeground=_accent_hov,
                     highlightthickness=1, highlightbackground=_dim,
                     relief=tk.FLAT, font=("Helvetica", 8), width=8)
        om["menu"].configure(bg=_card_bg, fg=_accent,
                             activebackground=_dim, activeforeground=_accent_hov,
                             font=("Helvetica", 8))
        om.pack(side=tk.LEFT, padx=(0, 6))

        tk.Checkbutton(inner_ctrl, text="Invert", variable=self._inverted,
                       command=self._draw,
                       bg=_card_bg, fg=_step_lbl, selectcolor=_accent,
                       activebackground=_card_bg, activeforeground=_accent_hov,
                       font=("Helvetica", 8), relief=tk.FLAT).pack(side=tk.LEFT, padx=(0, 12))

        # Normalization — single-select pills (NEMO-style), accent = active
        self._norm_mode = tk.StringVar(value="linear")
        self._norm_pill_redraw: list = []
        tk.Label(inner_ctrl, text="Norm:", bg=_card_bg, fg=_step_lbl,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 4))

        _norm_font = _tkfont.Font(family="Helvetica", size=8, weight="bold")
        _NP_W = max(_norm_font.measure(l) for l in ("linear", "log", "power")) + 16

        def _make_norm_pill(value):
            cv = tk.Canvas(inner_ctrl, width=_NP_W, height=_PILL_H,
                           highlightthickness=0, bd=0, bg=_card_bg,
                           cursor="pointinghand")
            cv.pack(side=tk.LEFT, padx=2)

            def _redraw(hover=False):
                cv.delete("all")
                active  = self._norm_mode.get() == value
                fill    = _accent if active else (_p['pill_hov'] if hover else _p['pill_nor'])
                txt_col = _p['fg_on_accent'] if active else _step_lbl
                cv.create_rectangle(0, 0, _NP_W, _PILL_H, fill=fill, outline=fill)
                cv.create_text(_NP_W // 2, _PILL_H // 2, text=value,
                               fill=txt_col, font=_norm_font)

            def _select(_e=None):
                self._norm_mode.set(value)
                for r in self._norm_pill_redraw:
                    r()
                self._draw()

            cv.bind("<ButtonRelease-1>", _select)
            cv.bind("<Enter>", lambda _e: _redraw(hover=True)
                    if self._norm_mode.get() != value else None)
            cv.bind("<Leave>", lambda _e: _redraw())
            self._norm_pill_redraw.append(_redraw)
            _redraw()

        for _nm in ("linear", "log", "power"):
            _make_norm_pill(_nm)

        # Gamma (power-law exponent) — only meaningful in "power" mode, so
        # it sits right next to the pills, styled exactly like every other
        # slider in the app (same make_slider() output, no cramped box),
        # and its opacity is reduced (blended toward the card background,
        # not just state=disabled) whenever a different norm mode is active.
        def _blend(c1, c2, t):
            def _rgb(h):
                h = h.lstrip('#')
                return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
            r1, g1, b1 = _rgb(c1)
            r2, g2, b2 = _rgb(c2)
            return '#{:02x}{:02x}{:02x}'.format(
                round(r1 + (r2 - r1) * t), round(g1 + (g2 - g1) * t), round(b1 + (b2 - b1) * t))

        _gamma_lbl = tk.Label(inner_ctrl, text="γ:", bg=_card_bg, fg=_step_lbl,
                              font=("Helvetica", 8))
        _gamma_lbl.pack(side=tk.LEFT, padx=(8, 2))
        _gamma_wrap = _slider(inner_ctrl, "", self._gamma_var, 0.1, 2.0,
                              resolution=0.05, fmt="{:.2f}")
        _gamma_wrap.pack(side=tk.LEFT, padx=(0, 6))

        def _find_widgets(root, cls):
            out = []
            if isinstance(root, cls):
                out.append(root)
            for c in root.winfo_children():
                out.extend(_find_widgets(c, cls))
            return out

        _gamma_scales  = _find_widgets(_gamma_wrap, tk.Scale)
        _gamma_entries = _find_widgets(_gamma_wrap, tk.Entry)
        # Capture each Scale's real colours once, so dimming/restoring is a
        # blend toward the card background rather than a hardcoded guess.
        _gamma_scale_orig = [
            dict(bg=w.cget('bg'), fg=w.cget('fg'), troughcolor=w.cget('troughcolor'),
                activebackground=w.cget('activebackground'))
            for w in _gamma_scales
        ]
        _gamma_entry_orig_fg = [w.cget('fg') for w in _gamma_entries]
        _DIM_T = 0.75  # 0 = full colour, 1 = fully background (opacity ~25% when dimmed)

        def _update_gamma_enabled(*_):
            active = self._norm_mode.get() == "power"
            state = tk.NORMAL if active else tk.DISABLED
            _gamma_lbl.configure(fg=_step_lbl if active else _blend(_step_lbl, _card_bg, _DIM_T))
            for w, orig in zip(_gamma_scales, _gamma_scale_orig):
                try:
                    w.configure(state=state)
                    for opt, col in orig.items():
                        w.configure(**{opt: col if active else _blend(col, _card_bg, _DIM_T)})
                except Exception: pass
            for w, orig_fg in zip(_gamma_entries, _gamma_entry_orig_fg):
                try:
                    w.configure(state=state,
                               fg=orig_fg if active else _blend(orig_fg, _card_bg, _DIM_T),
                               disabledforeground=_blend(orig_fg, _card_bg, _DIM_T),
                               disabledbackground=_card_bg)
                except Exception: pass
        self._norm_mode.trace_add('write', _update_gamma_enabled)
        _update_gamma_enabled()

        # ── Body: sliders (left) + dynamic spectrum (right), two equal cols ───
        body = tk.Frame(self, bg=_bg)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=1, uniform="half")
        body.columnconfigure(1, weight=1, uniform="half")
        body.rowconfigure(0, weight=1)
        left  = tk.Frame(body, bg=_bg)
        left.grid(row=0, column=0, sticky="nsew")
        right = tk.Frame(body, bg=_bg)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 4), pady=(4, 2))

        def _small_card(parent, title=None):
            return _viewer_small_card(parent, _p, title)

        def _sym_slider_row(parent, segs, var, lo, hi, resolution=0.01,
                            fmt="{:.2f}", integer=False):
            return _viewer_sym_slider_row(parent, self.master, _p, segs, var,
                                          lo, hi, resolution=resolution,
                                          fmt=fmt, integer=integer)

        _rng = (self._data_max - self._data_min) or 1.0
        self._vmin_var    = tk.DoubleVar(value=self._data_min)
        self._vmax_var    = tk.DoubleVar(value=self._data_max)
        self._channel_var = tk.IntVar(value=len(self._channels) // 2)

        # Threshold card — λ(S_max): contour-mask threshold (% of source peak)
        _thresh_card = _small_card(left, "Source contour threshold (% of peak)")
        _sym_slider_row(_thresh_card, [("λ(S", "n"), ("max", "s"), (")", "n")],
                        self._thresh_var, 0.1, 50.0, resolution=0.1, fmt="{:.1f}")
        self._thresh_var.trace_add('write', lambda *_: self._draw())

        # Colour-scale card — V_min / V_max / γ (power-law exponent), no heading
        _vrange_card = _small_card(left)
        _vmin_row = _sym_slider_row(_vrange_card, [("V", "n"), ("min", "s")],
                        self._vmin_var, self._data_min, self._data_max,
                        resolution=_rng / 500, fmt="{:.2e}")
        _vmax_row = _sym_slider_row(_vrange_card, [("V", "n"), ("max", "s")],
                        self._vmax_var, self._data_min, self._data_max,
                        resolution=_rng / 500, fmt="{:.2e}")
        # Kept so the Clean/Noisy toggle (see the "View:" pills above) can
        # reconfigure these sliders' from_/to bounds to match whichever
        # cube is active — without this, dragging past the OTHER cube's
        # range would be silently clamped, and the colour scale wouldn't
        # reflect what's actually being displayed.
        self._vmin_scale = _find_scale(_vmin_row)
        self._vmax_scale = _find_scale(_vmax_row)
        # γ (power-law exponent) now lives next to the "power" norm pill in
        # the control bar, not here — see _gamma_wrap above.

        # Channel card — Play/Pause + FPS on the left, the channel slider
        # filling the rest of the same row, with the channel/velocity
        # readout below.
        _chan_card = _small_card(left, "Channel")
        _chan_row = tk.Frame(_chan_card, bg=_card_bg)
        _chan_row.pack(fill=tk.X, pady=(0, 4))

        self._playing = False
        self._play_after_id = None
        self._fps_var = tk.DoubleVar(value=8.0)

        _play_font = _tkfont.Font(family="Helvetica", size=10, weight="bold")
        _play_w = max(_play_font.measure("⏸"), _play_font.measure("▶")) + 16

        _play_cv = tk.Canvas(_chan_row, width=_play_w, height=_PILL_H,
                             highlightthickness=0, bd=0, bg=_card_bg,
                             cursor="pointinghand")
        _play_cv.pack(side=tk.LEFT, padx=(0, 6))

        def _draw_play_btn(hover=False):
            _play_cv.delete("all")
            fill = _accent if self._playing else (_p['pill_hov'] if hover else _p['pill_nor'])
            txt_col = _p['fg_on_accent'] if self._playing else _step_lbl
            _play_cv.create_rectangle(0, 0, _play_w, _PILL_H, fill=fill, outline=fill)
            symbol = "⏸" if self._playing else "▶"
            _play_cv.create_text(_play_w // 2, _PILL_H // 2, text=symbol,
                                 fill=txt_col, font=_play_font)

        def _advance_channel():
            if not self._playing or not self.winfo_exists():
                return
            n = len(self._channels)
            nxt = (int(self._channel_var.get()) + 1) % n
            self._channel_var.set(nxt)
            fps = max(float(self._fps_var.get()), 0.1)
            self._play_after_id = self.after(int(1000 / fps), _advance_channel)

        def _toggle_play(_e=None):
            self._playing = not self._playing
            _draw_play_btn()
            if self._playing:
                _advance_channel()
            elif self._play_after_id is not None:
                self.after_cancel(self._play_after_id)
                self._play_after_id = None

        _play_cv.bind("<ButtonRelease-1>", _toggle_play)
        _play_cv.bind("<Enter>", lambda _e: _draw_play_btn(hover=True) if not self._playing else None)
        _play_cv.bind("<Leave>", lambda _e: _draw_play_btn())
        _draw_play_btn()

        tk.Label(_chan_row, text="FPS:", bg=_card_bg, fg=_step_lbl,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0, 4))

        # +/- stepper (accent square buttons flanking a numeric readout),
        # same visual language as the app's other pill buttons, instead of
        # a cramped mini-slider.
        _FPS_MIN, _FPS_MAX = 1, 30
        _step_font = _tkfont.Font(family="Helvetica", size=9, weight="bold")
        _step_btn_w = _step_font.measure("−") + 12
        _fps_val_w = _step_font.measure("30") + 12

        def _make_step_btn(parent, symbol, on_click):
            cv = tk.Canvas(parent, width=_step_btn_w, height=_PILL_H,
                           highlightthickness=0, bd=0, bg=_card_bg,
                           cursor="pointinghand")

            def _redraw(hover=False):
                cv.delete("all")
                fill = _p['pill_hov'] if hover else _p['pill_nor']
                cv.create_rectangle(0, 0, _step_btn_w, _PILL_H, fill=fill, outline=fill)
                cv.create_text(_step_btn_w // 2, _PILL_H // 2, text=symbol,
                               fill=_accent, font=_step_font)

            cv.bind("<ButtonRelease-1>", lambda _e: on_click())
            cv.bind("<Enter>", lambda _e: _redraw(hover=True))
            cv.bind("<Leave>", lambda _e: _redraw())
            _redraw()
            return cv

        _fps_val_lbl = tk.Label(_chan_row, text=str(int(self._fps_var.get())),
                                bg=_card_bg, fg=_accent, font=_step_font,
                                width=2, anchor="center")

        def _set_fps(v):
            self._fps_var.set(max(_FPS_MIN, min(_FPS_MAX, v)))
            _fps_val_lbl.configure(text=str(int(self._fps_var.get())))

        _make_step_btn(_chan_row, "−", lambda: _set_fps(int(self._fps_var.get()) - 1)
                      ).pack(side=tk.LEFT)
        _fps_val_lbl.pack(side=tk.LEFT, padx=2)
        _make_step_btn(_chan_row, "+", lambda: _set_fps(int(self._fps_var.get()) + 1)
                      ).pack(side=tk.LEFT, padx=(0, 10))

        _slider(_chan_row, "", self._channel_var, 0, len(self._channels) - 1,
                resolution=1, fmt="{:d}", integer=True).pack(
                side=tk.LEFT, fill=tk.X, expand=True)

        self._ch_lbl = tk.Label(_chan_card, text="", bg=_card_bg, fg=_accent,
                                font=("Helvetica", 8, "italic"), anchor="w",
                                justify="left")
        self._ch_lbl.pack(fill=tk.X, anchor="w")

        for _v in (self._vmin_var, self._vmax_var, self._channel_var, self._gamma_var):
            _v.trace_add('write', lambda *_: self._draw())

        self._build_spectrum(right)

        self._draw()
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        self.geometry(f"{w}x{h}")
        self.minsize(w, h)

    # ── Normalization ─────────────────────────────────────────────────────────
    def _norm(self):
        from matplotlib.colors import Normalize, LogNorm, PowerNorm
        vmin = float(self._vmin_var.get())
        vmax = float(self._vmax_var.get())
        if vmin >= vmax:
            vmax = vmin + 1e-9
        mode = self._norm_mode.get()
        if mode == "log":
            vmin = max(vmin, 1e-12)
            vmax = max(vmax, vmin + 1e-12)
            return LogNorm(vmin=vmin, vmax=vmax)
        elif mode == "power":
            gamma = float(self._gamma_var.get())
            return PowerNorm(gamma=max(gamma, 1e-3), vmin=max(vmin, 0), vmax=vmax)
        return Normalize(vmin=vmin, vmax=vmax)

    def _fmt_val(self, v: float) -> str:
        if self._norm_mode.get() == "log":
            return f"10^{np.log10(max(abs(v), 1e-30)):.2f}"
        return f"{v:.2e}"

    # ── Dynamic per-source spectrum (bottom-right) ──────────────────────────────
    def _build_spectrum(self, parent):
        """Spectrum panel: a dashed whole-field "Total" curve plus one curve per
        currently-selected source, each in its source colour, redrawn on toggle.
        A dashed marker tracks the current channel."""
        _p = self._pal
        nchan = self._cube.shape[0]
        if len(self._vels) == nchan and np.ptp(self._vels) > 0:
            self._spec_x = np.asarray(self._vels, dtype=float)
            self._spec_xlabel = r'Velocity (km s$^{-1}$)'
        else:
            self._spec_x = np.arange(nchan, dtype=float)
            self._spec_xlabel = "Channel"

        self._spec_total = np.nansum(self._cube, axis=(1, 2))
        self._spec_curve: dict = {}
        if self._per_gal is not None:
            for i in range(self._n_gals):
                self._spec_curve[i] = np.nansum(self._per_gal[i], axis=(1, 2))

        fig = plt.Figure(figsize=(2.6, 2.2), dpi=96, facecolor=_p['log_bg'])
        ax  = fig.add_subplot(111)
        self._spec_fig    = fig
        self._spec_ax     = ax
        self._spec_canvas = FigureCanvasTkAgg(fig, master=parent)
        self._spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._last_vis_key = None
        self._redraw_spectrum()

    def _spec_visible_ids(self):
        return frozenset(i for i, v in self._src_visible.items() if v.get())

    def _spec_visible_items(self):
        """[(label, i, hex_rgb)] for visible sources, coloured to match contours."""
        items = []
        for i in range(self._n_gals):
            v = self._src_visible.get(i)
            if v is not None and not v.get():
                continue
            items.append((_src_label(i), i, _SRC_PALETTE[i % len(_SRC_PALETTE)]))
        return items

    def _redraw_spectrum(self):
        _p  = self._pal
        ax  = self._spec_ax
        ax.clear()
        ax.set_facecolor(_p['log_bg'])
        xs  = self._spec_x
        txt = _p['step_lbl']

        if self._per_gal is not None:
            ax.plot(xs, self._spec_total, color=_p['dim_txt'], lw=1.0, ls="--",
                    label="Total")
            items = self._spec_visible_items()
            for label, i, rgb in items:
                ax.plot(xs, self._spec_curve[i], color=rgb, lw=1.2, label=label)
            if items:
                leg = ax.legend(fontsize=6, ncol=1, framealpha=0.6,
                                facecolor=_p['log_bg'], edgecolor=_p['dim'],
                                labelcolor=txt, handlelength=1.2,
                                handletextpad=0.4, borderpad=0.3, loc="best")
                leg.get_frame().set_linewidth(0.6)
            self._last_vis_key = self._spec_visible_ids()
        else:
            ax.plot(xs, self._spec_total, color=_p['accent'], lw=1.1)

        ax.set_xlabel(self._spec_xlabel, color=txt, fontsize=9)
        ax.set_ylabel(r'Jy beam$^{-1}$', color=txt, fontsize=9)
        ax.tick_params(colors=txt, labelsize=8, length=3)
        ax.margins(x=0.02)
        for sp in ax.spines.values():
            sp.set_edgecolor(_p['dim_txt'])
            sp.set_linewidth(0.8)
        ch = self._channels[int(self._channel_var.get())]
        self._spec_vline = ax.axvline(self._spec_x[ch],
                                      color=_p['accent_hov'], ls="--", lw=1.1)
        self._spec_fig.tight_layout(pad=1.0)
        self._spec_canvas.draw_idle()

    def _update_spectrum_data(self):
        """Replot per-source spectra when the source selection changes."""
        if self._per_gal is None or not hasattr(self, "_spec_ax"):
            return
        if self._spec_visible_ids() == self._last_vis_key:
            return
        self._redraw_spectrum()

    def _update_spectrum_marker(self, ch):
        if not hasattr(self, "_spec_vline"):
            return
        x = float(self._spec_x[ch])
        self._spec_vline.set_xdata([x, x])
        self._spec_canvas.draw_idle()

    # ── Main draw ─────────────────────────────────────────────────────────────
    def _draw(self):
        idx = int(self._channel_var.get())
        ch  = self._channels[idx]
        img = self._cube[ch]

        norm = self._norm()
        cmap = self._cmap.get() + ("_r" if self._inverted.get() else "")

        _pal = self._pal
        self._ax.clear()
        self._ax.set_xticks([]); self._ax.set_yticks([])
        for spine in self._ax.spines.values():
            spine.set_edgecolor(_pal['dim_txt'])
            spine.set_linewidth(0.8)
        self._ax.imshow(img, cmap=cmap, norm=norm, origin="lower")
        # Pin the axes to the image extent so source bboxes / labels that fall
        # near or past the FOV edge spill outside instead of shrinking the image.
        self._ax.set_xlim(-0.5, img.shape[1] - 0.5)
        self._ax.set_ylim(-0.5, img.shape[0] - 0.5)
        self._ax.set_autoscale_on(False)

        # Colorbar — axes_locatable cax, styled exactly like NEMO
        self._fig.set_facecolor(_pal['log_bg'])
        self._ax_cb.set_facecolor(_pal['log_bg'])
        self._ax_cb.clear()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = self._fig.colorbar(sm, cax=self._ax_cb)
        cb.ax.tick_params(colors=_pal['step_lbl'], labelsize=9, length=3)
        cb.outline.set_edgecolor(_pal['dim'])
        plt.setp(plt.getp(cb.ax, "yticklabels"), color=_pal['step_lbl'], fontsize=9)
        cb.set_label(r'Jy beam$^{-1}$', color=_pal['step_lbl'], fontsize=9, labelpad=6)

        # Per-source contours + bboxes
        if self._per_gal is not None:
            thresh_frac = float(self._thresh_var.get()) / 100.0
            from matplotlib.patches import Rectangle as _Rect
            PAD = 4
            for i in range(self._n_gals):
                if not self._src_visible[i].get():
                    continue
                if ch >= self._per_gal.shape[1]:
                    continue
                gal_ch = self._per_gal[i, ch]
                cube_max = float(np.nanmax(self._per_gal[i])) if np.nanmax(self._per_gal[i]) > 0 else 1e-9
                thresh = thresh_frac * cube_max
                mask = gal_ch >= thresh
                if not mask.any():
                    continue
                col  = _SRC_PALETTE[i % len(_SRC_PALETTE)]
                lcol = _lighten(col)
                self._ax.contour(mask.astype(float), [0.5],
                                 colors=[col], linewidths=0.8)
                rows, cols = np.where(mask)
                r0, r1 = int(rows.min()), int(rows.max())
                c0, c1 = int(cols.min()), int(cols.max())
                self._ax.add_patch(_Rect(
                    (c0 - PAD, r0 - PAD),
                    c1 - c0 + 2*PAD, r1 - r0 + 2*PAD,
                    linewidth=0.8, edgecolor=lcol, facecolor="none", zorder=4,
                    clip_on=False,
                ))
                label = "C" if i == 0 else f"S{i}"
                self._ax.text(
                    c1 + PAD, r1 + PAD, label,
                    ha="center", va="center", fontsize=7,
                    color="black", fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.22", fc=lcol, ec=lcol, lw=1.2),
                    zorder=6, clip_on=False,
                )

        # Beam + scalebar
        if self._beam is not None:
            try:
                add_beam(self._ax, self._beam[0], self._beam[1], self._beam[2],
                         xy_offset=(6*img.shape[0]/72, 6*img.shape[0]/72), color=_pal['accent'])
            except Exception:
                pass
        ny, nx = img.shape
        scalebar_px = (25/72) * nx
        x0, y0 = nx * 0.6, ny * 0.07
        self._ax.plot([x0, x0 + scalebar_px], [y0, y0], color=_pal['accent'], lw=1.5)
        self._ax.text(x0 + scalebar_px/2, y0 + ny*0.03,
                      f"{scalebar_px * self._pix_scale:.1f} kpc",
                      color=_pal['accent'], ha="center", va="bottom",
                      fontsize=7, weight="bold")

        self._canvas.draw()

        # Keep the spectrum marker on the current channel
        self._update_spectrum_marker(ch)

        # Channel status label
        v = self._vels[ch] if ch < len(self._vels) else 0.0
        n_active = 0
        if self._per_gal is not None:
            thresh_frac = float(self._thresh_var.get()) / 100.0
            for i in range(self._n_gals):
                if not self._src_visible[i].get():
                    continue
                if ch >= self._per_gal.shape[1]:
                    continue
                cube_max = float(np.nanmax(self._per_gal[i])) if np.nanmax(self._per_gal[i]) > 0 else 1e-9
                if (self._per_gal[i, ch] >= thresh_frac * cube_max).any():
                    n_active += 1
        parts = [f"{ch}  ·  {v:.1f} km s⁻¹"]
        if n_active:
            parts.append(f"{n_active} source(s) visible")
        self._ch_lbl.configure(text="  ·  ".join(parts))


class AnalysisViewer(tk.Toplevel):
    """Combined analysis viewer: Moment 0, Moment 1, and integrated spectrum.

    Source checkboxes (one per galaxy) and a Diffuse checkbox let the user
    select which components contribute to all three panels. The layout mirrors
    the nemo analysis viewer: two moment maps side-by-side on top, spectrum
    spanning the full width below.
    """

    def __init__(self, master, data, idx: int = 0, noisy_data=None):
        super().__init__(master)

        _p          = _VIEWER_PALETTES[_VIEWER_THEME]
        _bg         = _p['bg']
        _card_bg    = _p['card_bg']
        _accent     = _p['accent']
        _accent_hov = _p['accent_hov']
        _dim        = _p['dim']
        _step_lbl   = _p['step_lbl']
        _log_bg     = _p['log_bg']

        self.configure(bg=_bg)
        self.resizable(True, True)
        self.title("SONGS — Analysis")
        # Themed title bar, matching the main window. Re-applied on a short
        # delay too, since the NSWindow may not exist yet on the first call.
        _apply_window_appearance(self)
        self.after(250, lambda: _apply_window_appearance(self))
        self._pal = _p

        # ``noisy_data`` (optional, same shape/idx convention as ``data``) is
        # only present when "Use Noise?" was Yes. Both TOTAL cube versions
        # are kept; per-source/diffuse component arrays stay the clean
        # ground truth either way (noise is an instrument effect on the
        # observed total, not a per-source quantity) — see
        # _build_display_cube(), which uses self._cube directly (whichever
        # version is active) only when every component is selected.
        cube, meta = data[idx]
        self._cube_clean = cube.astype(np.float32)
        self._cube_noisy = None
        if noisy_data:
            noisy_cube, _noisy_meta = noisy_data[idx]
            self._cube_noisy = noisy_cube.astype(np.float32)
        # Default to noisy when available — matches what would actually be
        # observed/saved.
        self._cube = self._cube_noisy if self._cube_noisy is not None else self._cube_clean
        self._vels      = np.asarray(meta.get('average_vels', np.arange(cube.shape[0])))
        self._beam      = meta.get('beam_info')
        self._pix_scale = float(meta.get('pix_spatial_scale', 1.0))

        pg = meta.get('per_galaxy_cubes')
        self._per_gal = np.asarray(pg) if pg is not None else None
        self._n_gals  = int(self._per_gal.shape[0]) if self._per_gal is not None else 0

        hc = meta.get('halo_cube')
        bc = meta.get('bridges_cube')
        self._halo_cube    = np.asarray(hc).astype(np.float32) if hc is not None else None
        self._bridges_cube = np.asarray(bc).astype(np.float32) if bc is not None else None
        # Fallback: treat total minus per-galaxy as diffuse when component cubes absent
        self._diffuse_cube = (
            self._cube - self._per_gal.sum(axis=0)
            if (self._per_gal is not None and self._n_gals > 0
                and self._halo_cube is None and self._bridges_cube is None)
            else None
        )

        # ── Layout: sidebar left, figure right ────────────────────────────────
        top = tk.Frame(self, bg=_bg)
        top.pack(fill=tk.BOTH, expand=True)

        sb = tk.Frame(top, bg=_bg, width=178)
        sb.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 2), pady=6)
        sb.pack_propagate(False)

        tk.Label(sb, text="Sources", bg=_bg, fg=_accent,
                 font=("Helvetica", 9, "bold")).pack(pady=(4, 6), anchor="w")

        # ── Source / component pills (clickable, pointing-hand cursor) ─────────
        from tkinter import font as _tkfont
        _pill_font = _tkfont.Font(family="Helvetica", size=8, weight="bold")

        self._show_total   = tk.BooleanVar(value=True)
        self._src_visible: dict[int, tk.BooleanVar] = {}
        self._show_halo    = tk.BooleanVar(value=True)
        self._show_bridges = tk.BooleanVar(value=True)
        self._show_diffuse = tk.BooleanVar(value=True)  # fallback legacy

        # Collect every pill label up front so one fixed width keeps the column
        # uniform (matches the SliceViewer source pills).
        _labels = ["Total"] + [_src_label(i) for i in range(self._n_gals)]
        if self._halo_cube is not None:
            _labels.append("Diffuse Halo")
        if self._bridges_cube is not None:
            _labels.append("Bridges")
        if self._diffuse_cube is not None:
            _labels.append("Diffuse")
        _PILL_W = max((_pill_font.measure(lbl) for lbl in _labels), default=80) + 16

        def _pill(label, var, swatch, swatch_color):
            _viewer_pill(sb, _p, label, var, self._draw, swatch=swatch,
                         swatch_color=swatch_color, pill_w=_PILL_W,
                         pill_font=_pill_font)

        _pill("Total", self._show_total, "—", _accent)
        tk.Frame(sb, bg=_accent, height=1).pack(fill=tk.X, pady=(3, 5))

        for i in range(self._n_gals):
            self._src_visible[i] = tk.BooleanVar(value=True)
            _pill(_src_label(i), self._src_visible[i], "■",
                  _rgb_to_hex(_SRC_PALETTE[i % len(_SRC_PALETTE)]))

        _has_components = (self._halo_cube is not None or self._bridges_cube is not None
                           or self._diffuse_cube is not None)
        if _has_components:
            tk.Frame(sb, bg=_dim, height=1).pack(fill=tk.X, pady=(10, 5))
            if self._halo_cube is not None:
                _pill("Diffuse Halo", self._show_halo, "◉", _step_lbl)
            if self._bridges_cube is not None:
                _pill("Bridges", self._show_bridges, "▒", _step_lbl)
            if self._diffuse_cube is not None:
                _pill("Diffuse", self._show_diffuse, "▒", _step_lbl)

        # ── Clean/Noisy toggle — only shown when a noisy preview is
        # available ("Use Noise?" was Yes at generation time). ─────────────
        if self._cube_noisy is not None:
            tk.Frame(sb, bg=_dim, height=1).pack(fill=tk.X, pady=(10, 5))
            tk.Label(sb, text="View", bg=_bg, fg=_accent,
                     font=("Helvetica", 9, "bold")).pack(pady=(0, 4), anchor="w")
            self._view_mode = tk.StringVar(value="Noisy")
            _view_row = tk.Frame(sb, bg=_bg)
            _view_row.pack(anchor="w")

            def _on_view_mode_change():
                self._cube = (self._cube_noisy if self._view_mode.get() == "Noisy"
                             else self._cube_clean)
                self._draw()

            _viewer_pill_group(_view_row, _p, ("Clean", "Noisy"), self._view_mode,
                               _on_view_mode_change, bg=_bg)

        # ── Threshold — same card + symbol style as the SliceViewer ────────────
        self._thresh_var = tk.DoubleVar(value=5.0)
        tk.Frame(sb, bg=_dim, height=1).pack(fill=tk.X, pady=(10, 5))
        _thresh_card = _viewer_small_card(sb, _p, "Moment-1 mask (% of peak)")
        _viewer_sym_slider_row(_thresh_card, self.master, _p,
                               [("λ(S", "n"), ("max", "s"), (")", "n")],
                               self._thresh_var, 0.1, 50.0,
                               resolution=0.1, fmt="{:.1f}")
        self._thresh_var.trace_add('write', lambda *_: self._draw())

        # ── Matplotlib figure ─────────────────────────────────────────────────
        self._fig = plt.Figure(figsize=(9, 7), dpi=96, facecolor=_log_bg)
        self._canvas = FigureCanvasTkAgg(self._fig, master=top)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._draw()
        self.update_idletasks()
        self.resizable(True, True)
        self.geometry("960x800")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_display_cube(self) -> np.ndarray:
        """Sum selected per-galaxy cubes plus selected diffuse components.

        When every component is selected (the default), this returns
        ``self._cube`` (the active clean-or-noisy TOTAL) directly instead
        of re-summing the per-source arrays — those stay clean-only always
        (noise isn't decomposable per source), so re-summing them would
        silently drop the noise from the moment maps / "Total" spectrum
        line whenever "Noisy" is selected. Any subset selection still falls
        back to summing the (necessarily clean) selected components."""
        if self._per_gal is None:
            return self._cube
        all_selected = (
            all(v.get() for v in self._src_visible.values())
            and (self._show_halo.get() if self._halo_cube is not None else True)
            and (self._show_bridges.get() if self._bridges_cube is not None else True)
            and (self._show_diffuse.get() if self._diffuse_cube is not None else True)
        )
        if all_selected:
            return self._cube
        display = np.zeros_like(self._cube)
        for i, var in self._src_visible.items():
            if var.get():
                display = display + self._per_gal[i]
        if self._show_halo.get() and self._halo_cube is not None:
            display = display + self._halo_cube
        if self._show_bridges.get() and self._bridges_cube is not None:
            display = display + self._bridges_cube
        if self._show_diffuse.get() and self._diffuse_cube is not None:
            display = display + self._diffuse_cube
        return display

    def _draw(self):
        _pal      = self._pal
        _bg       = _pal['log_bg']
        _accent   = _pal['accent']
        _step_lbl = _pal['step_lbl']
        _dim      = _pal['dim']

        _white = '#ffffff'          # text/markers drawn over the (dark) images
        _txt   = _pal['sym_fg']     # text on the figure background (black in light)

        self._fig.clf()

        # Grid: thin colorbar row on top, moment maps, spectrum. top=0.90
        # (rather than 0.96) leaves headroom above the colorbars — their
        # tick/unit labels sit above the bars themselves (set_label_position
        # ('top') below) and were getting clipped by the figure edge with
        # only a 4% margin.
        gs = self._fig.add_gridspec(
            3, 2,
            height_ratios=[0.08, 1.1, 0.9],
            hspace=0.18, wspace=0.12,
            left=0.13, right=0.97, top=0.90, bottom=0.10,
        )
        cax0  = self._fig.add_subplot(gs[0, 0])
        cax1  = self._fig.add_subplot(gs[0, 1])
        ax_m0 = self._fig.add_subplot(gs[1, 0])
        ax_m1 = self._fig.add_subplot(gs[1, 1])
        ax_sp = self._fig.add_subplot(gs[2, :])

        cube = self._build_display_cube()
        vels = self._vels
        del_V = float(np.diff(vels).mean()) if len(vels) > 1 else 1.0

        ny, nx = cube.shape[1], cube.shape[2]
        extent = [0, nx, 0, ny]
        scalebar_px = (25 / 72) * nx
        sb_x0, sb_y0 = nx * 0.6, ny * 0.07

        def _style_ax(ax):
            ax.set_facecolor(_bg)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(_dim); sp.set_linewidth(0.6)

        def _scalebar(ax):
            ax.plot([sb_x0, sb_x0 + scalebar_px], [sb_y0, sb_y0],
                    color=_white, lw=1.5)
            ax.text(sb_x0 + scalebar_px / 2, sb_y0 + ny * 0.03,
                    f"{scalebar_px * self._pix_scale:.1f} kpc",
                    color=_white, ha='center', va='bottom',
                    fontsize=8, weight='bold')

        def _beam(ax):
            if self._beam is not None:
                try:
                    add_beam(ax, self._beam[0], self._beam[1], self._beam[2],
                             xy_offset=(6 * ny / 72, 6 * ny / 72), color=_white)
                except Exception:
                    pass

        def _inner_title(ax, text):
            ax.text(0.03, 0.97, text, transform=ax.transAxes,
                    color=_txt, fontsize=10, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=_bg,
                              edgecolor=_dim, alpha=0.75, linewidth=0.6))

        # ── Moment 0 ──────────────────────────────────────────────────────────
        mom0 = cube.sum(axis=0) * del_V
        vmax0 = float(np.nanmax(mom0)) if np.nanmax(mom0) > 0 else 1.0
        _style_ax(ax_m0)
        im0 = ax_m0.imshow(mom0, cmap='inferno', origin='lower',
                           extent=extent, vmin=0, vmax=vmax0)
        _inner_title(ax_m0, "Moment 0")
        cb0 = self._fig.colorbar(im0, cax=cax0, orientation='horizontal')
        cb0.ax.tick_params(colors=_txt, labelsize=7, pad=3)
        cb0.outline.set_edgecolor(_dim)
        cb0.set_label(r'Jy beam$^{-1}$ km s$^{-1}$', color=_txt, fontsize=8, labelpad=6)
        cax0.xaxis.set_ticks_position('top')
        cax0.xaxis.set_label_position('top')
        _beam(ax_m0); _scalebar(ax_m0)

        # ── Moment 1 — masked by threshold % of peak moment-0 flux ───────────
        thresh_frac = float(self._thresh_var.get()) / 100.0
        flux_mask = mom0 >= thresh_frac * vmax0

        moment_cube = cube * vels[:, np.newaxis, np.newaxis]
        denom = cube.sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            mom1_raw = np.where(denom > 0, moment_cube.sum(axis=0) / denom, np.nan)
        mom1_raw = np.clip(mom1_raw, float(vels.min()), float(vels.max()))
        mom1 = np.where(flux_mask, mom1_raw, np.nan)

        finite = mom1[np.isfinite(mom1)]
        vm1 = max(float(np.nanmax(np.abs(finite))) if finite.size > 0 else 1.0, 1.0)

        _style_ax(ax_m1)
        im1 = ax_m1.imshow(mom1, cmap='RdBu_r', origin='lower',
                           extent=extent, vmin=-vm1, vmax=vm1)
        _inner_title(ax_m1, "Moment 1")
        cb1 = self._fig.colorbar(im1, cax=cax1, orientation='horizontal')
        cb1.ax.tick_params(colors=_txt, labelsize=7, pad=3)
        cb1.outline.set_edgecolor(_dim)
        cb1.set_label(r'km s$^{-1}$', color=_txt, fontsize=8, labelpad=6)
        cax1.xaxis.set_ticks_position('top')
        cax1.xaxis.set_label_position('top')
        _beam(ax_m1); _scalebar(ax_m1)

        # ── Spectrum ──────────────────────────────────────────────────────────
        ax_sp.set_facecolor(_bg)
        for sp in ax_sp.spines.values():
            sp.set_edgecolor(_dim); sp.set_linewidth(0.6)

        if self._show_total.get():
            ax_sp.plot(vels, cube.sum(axis=(1, 2)),
                       color=_accent, lw=1.8, label="Total", zorder=3)

        if self._per_gal is not None:
            for i, var in self._src_visible.items():
                if not var.get():
                    continue
                col = _SRC_PALETTE[i % len(_SRC_PALETTE)]
                ax_sp.plot(vels, self._per_gal[i].sum(axis=(1, 2)),
                           color=col, lw=0.9, alpha=0.75,
                           label=_src_label(i), zorder=2)

        if self._show_halo.get() and self._halo_cube is not None:
            ax_sp.plot(vels, self._halo_cube.sum(axis=(1, 2)),
                       color=_step_lbl, lw=0.9, alpha=0.55,
                       linestyle=':', label="Diffuse Halo", zorder=2)
        if self._show_bridges.get() and self._bridges_cube is not None:
            ax_sp.plot(vels, self._bridges_cube.sum(axis=(1, 2)),
                       color=_dim, lw=0.9, alpha=0.65,
                       linestyle='--', label="Bridges", zorder=2)
        if self._show_diffuse.get() and self._diffuse_cube is not None:
            ax_sp.plot(vels, self._diffuse_cube.sum(axis=(1, 2)),
                       color=_step_lbl, lw=0.9, alpha=0.6,
                       linestyle='--', label="Diffuse", zorder=2)

        ax_sp.set_xlabel(r'Velocity (km s$^{-1}$)', color=_txt, fontsize=10)
        ax_sp.set_ylabel(r'Flux Density (Jy beam$^{-1}$)', color=_txt, fontsize=10)
        ax_sp.tick_params(colors=_txt, labelsize=9)
        leg = ax_sp.legend(fontsize=9, facecolor=_pal['card_bg'],
                           edgecolor=_dim, labelcolor=_txt)

        self._fig.set_facecolor(_pal['log_bg'])
        self._canvas.draw()


def slice_view(data, idx=0, channel=None, cmap='viridis', parent=None):
    """Show a per-channel slice viewer embedded in a Tk window.

    This viewer embeds the Matplotlib figure into a Tk Toplevel and uses a
    ttk-styled slider to step through spectral channels. The viewer keeps the
    lower colour limit fixed at 0 and computes an upper limit per slice so
    contrast adapts to the currently displayed channel.

    Parameters
    ----------
    data : sequence
        The ``results`` container produced by ``SONGS.generate_cubes``.
    idx : int, optional
        Index of the cube within ``data`` to display (default 0).
    channel : int, optional
        Initial spectral channel index to show. If ``None`` (the default)
        the viewer will open on the central spectral channel (``int(n/2)``).
    cmap : str, optional
        Matplotlib colormap to use for imshow.
    parent : tkinter widget, optional
        If provided, the slice viewer will be a child Toplevel of this
        widget. Otherwise a new Toplevel (or root) is used.

    Returns
    -------
    fig, ax : (Figure, Axes)
        The Matplotlib figure and axes used by the embedded viewer.
    """

    cube, meta, beam_info, vels, pix_spatial_scale, del_V, moment_cube, mask = _prepare_cube(data, idx)

    n_chan = int(cube.shape[0])
    # Default spectral index: middle channel
    if channel is None:
        # Use the 1-based middle slice formula int((n_slices+1)/2) then
        # convert to 0-based index by subtracting 1. This matches the
        # user's requested behaviour for odd/even slice counts.
        channel = int((n_chan + 1) / 2) - 1
    channel = int(max(0, min(int(channel), n_chan - 1)))

    # Precompute fixed colour limits from the integrated (moment0) map so the
    # fixed option has a consistent reference scale. vmin is fixed at 0.
    fixed_vmin = 0.0
    fixed_vmax = float(np.nanmax(cube))

    # Create a Tk Toplevel to host the canvas. If there's an existing Tk
    # root, make a Toplevel so we don't create a second main window.
    if parent is not None:
        win = tk.Toplevel(master=parent)
    else:
        if tk._default_root is None:
            win = tk.Tk()
        else:
            win = tk.Toplevel()
    win.title(f"IFU viewer")

    ny, nx = cube.shape[1], cube.shape[2]
    extent = [0, nx, 0, ny]

    from matplotlib.figure import Figure as _Figure
    fig = _Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    # Shift the subplot region slightly up so title, figure and colorbar sit
    # a bit higher in the Toplevel window by default.
    fig.subplots_adjust(top=0.95, bottom=0.12)
    # Use the same colormap and units styling as moment0. Multiply single
    im = ax.imshow(cube[channel, :, :], cmap='RdBu_r', origin='lower', extent=extent, vmin=0.0, vmax=fixed_vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    # Put the colorbar below the image
    cax = divider.append_axes("bottom", size="5%", pad=0.3)
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', label=r'$\rm Jy\;beam^{-1}$', format='%.2f')
    # Place label and ticks on the bottom and draw ticks outward
    cb.ax.xaxis.set_label_position('bottom')
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.xaxis.label.set_size(14)
    cb.ax.tick_params(labelsize=12, direction='out', pad=6)
    cb.ax.xaxis.labelpad = 6

    # Initialize color limits according to the autoscale default (per-slice)
    try:
        sl0 = cube[channel, :, :]
        v1 = float(np.nanmax(sl0))
        im.set_clim(0.0, v1)
        cb.set_clim(0.0, v1)
        cb.draw_all()
    except Exception:
        # fall back to fixed integrated limits
        try:
            im.set_clim(fixed_vmin, fixed_vmax)
            cb.set_clim(fixed_vmin, fixed_vmax)
            cb.draw_all()
        except Exception:
            pass
    # (we will show the channel/velocity description below the figure as
    # LaTeX text; keep the axes area free of a title overlay)
    add_beam(ax, beam_info[0], beam_info[1], beam_info[2], xy_offset=(6*cube.shape[1]/72,6*cube.shape[1]/72), color='white')

    ax.set_aspect('equal')

    # Scalebar (match moment0 style)
    scalebar = (25/72)*cube.shape[2]
    x0, y0 = nx*0.6, ny*0.07
    ax.plot([x0, x0+scalebar], [y0, y0], color='white', lw=2)
    ax.text(x0+scalebar/2, y0 + ny*0.03, f'{scalebar*pix_spatial_scale:.1f} kpc',
        color='white', ha='center', va='bottom', fontsize=12, weight='bold')

    # Embed the Matplotlib figure in the Tk window. We will draw once and
    # compute sizes so we can fix the Toplevel geometry; this prevents the
    # window from resizing when the controls (scale/labels) update.
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas_widget = canvas.get_tk_widget()
    # Pack without expansion so the geometry we set stays stable
    canvas_widget.pack(side=tk.TOP)

    # Optional navigation toolbar
    toolbar = None
    try:
        toolbar = NavigationToolbar2Tk(canvas, win)
        # Place toolbar above the canvas (it will request its own height)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.update()
    except Exception:
        toolbar = None

    # Controls frame with a native Tk scale for robust interaction
    ctrl = tk.Frame(win)
    ctrl.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)

    # Use a ttk-styled slider that matches the rest of the GUI. We create a
    # small slider row with a right-aligned numeric label like the main app's
    # `make_slider` helper so appearance is consistent.
    label = ttk.Label(ctrl, text=f"Channel: {channel+1} : v = {vels[channel]:.1f} km/s")
    label.pack(side=tk.LEFT, padx=(0, 12))

    slider_row = ttk.Frame(ctrl)
    slider_row.pack(side=tk.LEFT, fill='x', expand=1)
    # Show the displayed channel as 1-based to match user expectation
    val_lbl = ttk.Label(slider_row, text=f"{channel+1}", width=6, anchor="e")
    val_lbl.pack(side='right', padx=(4, 0))

    scale = ttk.Scale(slider_row, from_=0, to=n_chan - 1, orient='horizontal')
    scale.pack(side='left', fill='x', expand=1)

    # No autoscale checkbox: always use vmin=0 and per-slice vmax by default.

    # Snapping/busy guard to avoid recursive updates and ensure integer steps
    busy = {'val': False}
    # Create a LaTeX title as the Axes title (so it appears above the image)
    def _latex_for(ci, v):
        return r"$\rm Channel\ %d\;:\;v=%.1f\;km\;s^{-1}$" % (ci, v)

    # Set the initial title on the axes (matplotlib mathtext will render it)
    ax.set_title(_latex_for(channel+1, vels[channel]), fontsize=13)

    def _on_scale(val):
        if busy['val']:
            return
        busy['val'] = True
        try:
            ci = int(round(float(val)))
        except Exception:
            busy['val'] = False
            return
        # update widgets and image
        # Display channel number as 1-based
        val_lbl.config(text=str(ci + 1))
        sl = cube[ci, :, :]
        im.set_data(sl)
        # Update the Axes title (LaTeX) and the left-side label
        ax.set_title(_latex_for(ci+1, vels[ci]))
        label.config(text=f"Channel: {ci+1} : v = {vels[ci]:.1f} km/s")
        # Always update the displayed data and compute a per-slice vmax.
        try:
            try:
                v1 = float(np.nanmax(sl))
            except Exception:
                v1 = fixed_vmax
            try:
                im.set_clim(0.0, v1)
                cb.set_clim(0.0, v1)
                cb.draw_all()
            except Exception:
                try:
                    im.set_clim(0.0, fixed_vmax)
                    cb.set_clim(0.0, fixed_vmax)
                    cb.draw_all()
                except Exception:
                    pass
            try:
                canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            # Keep function robust: if anything unexpected fails, continue
            # without crashing the UI.
            pass
        busy['val'] = False

    scale.configure(command=_on_scale)
    try:
        scale.set(channel)
    except Exception:
        pass

    # Force an initial draw so geometry measurements are reliable
    try:
        canvas.draw()
        win.update_idletasks()
    except Exception:
        pass

    # Measure sizes
    try:
        c_w, c_h = canvas.get_width_height()
    except Exception:
        # Fallback to widget requested size
        c_w = canvas_widget.winfo_reqwidth()
        c_h = canvas_widget.winfo_reqheight()

    toolbar_h = toolbar.winfo_height() if toolbar is not None else 0
    ctrl_h = ctrl.winfo_reqheight()

    total_w = max(c_w, 480)
    total_h = c_h + toolbar_h + ctrl_h + 10

    # Set fixed geometry and prevent resizing to keep the window stable
    try:
        win.geometry(f"{total_w}x{total_h}")
        win.minsize(total_w, total_h)
        win.maxsize(total_w, total_h)
        win.resizable(False, False)
    except Exception:
        # If any of the geometry calls fail, continue without locking
        pass

    return fig, ax
