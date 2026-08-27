"""SONGS GUI

Compact Tkinter-based GUI to interactively configure and run the
``SONGS`` generator. Provides a three-column layout of parameter
frames, crisp LaTeX-rendered labels, convenience sliders, and utility
buttons (Generate, Slice, Moments, Spectrum, Save, New). Plotting and file
I/O are intentionally kept out of the generator core; the GUI imports
top-level visualisation helpers (``moment0``, ``moment1``, ``spectrum``,
``slice_view``) to display results.

Design notes
------------
- Lightweight: the GUI focuses on inspection and quick interactive
    experimentation, not production batch runs.
- Threading: generation runs in a background thread so the UI remains
    responsive; generated figures are produced by the visualise helpers.
- Cleanup: LaTeX labels are rendered to temporary PNG files (via
    matplotlib) and tracked in ``_MATH_TEMPFILES`` for removal when the
    application exits.

Usage
-----
Run the module as a script to display the GUI::

    python -m songs.gui

Or instantiate :class:`SONGSGUI` and call ``mainloop()``. The GUI
expects the package to be importable (it will try a fallback path insertion
when executed as a script)."""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pickle
import threading
import numpy as np
import matplotlib
# Use Agg backend to avoid Tkinter threading issues
# Figures will still display properly when show() is called
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
import sys
import platform
from PIL import Image, ImageTk

# Track latex PNG tempfiles for cleanup
_MATH_TEMPFILES = []

# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------
_THEMES = {
    'dark': dict(
        BG='#0a0a0a', CARD_BG='#111111', BIG_BG='#0d0d0d',
        BIG_BORDER='#403100', SM_BORDER='#201800',
        TEXT='#999999', ACCENT='#d4aa40', ENTRY_BG='#1a1a1a',
        SLIDER_BORDER='#5e4200', SLIDER_TROUGH='#111111',
        SLIDER_THUMBHOV='#f0c040',
        PILL_NOR='#1e1e1e', PILL_HOV='#2e2700',
        BTN_DIS_FG='#333322',
        BTN_NOR_BG='#1a1400', BTN_NOR_FG='#d4aa40', BTN_NOR_HOV='#2e2400',
        REGEN_HOV='#1e1e1e',
        SYM_FG='white',
        LOGO='songs_dark_vertical.png',
        PLOT_BG='#0a0a0a',
    ),
    'light': dict(
        BG='#f0ede6', CARD_BG='#ffffff', BIG_BG='#faf8f3',
        BIG_BORDER='#c8a030', SM_BORDER='#e0d5b0',
        TEXT='#444444', ACCENT='#9a7200', ENTRY_BG='#ffffff',
        SLIDER_BORDER='#c8a030', SLIDER_TROUGH='#ffffff',
        SLIDER_THUMBHOV='#e8c040',
        PILL_NOR='#dedad0', PILL_HOV='#f5e898',
        BTN_DIS_FG='#aaa890',
        BTN_NOR_BG='#ede8da', BTN_NOR_FG='#444444', BTN_NOR_HOV='#ddd5c0',
        REGEN_HOV='#ebebeb',
        SYM_FG='#222222',
        LOGO='songs_light_vertical.png',
        PLOT_BG='#ffffff',
    ),
}

import warnings

# Or suppress ALL UserWarnings if you prefer a cleaner log
warnings.filterwarnings("ignore", category=UserWarning)


def _enable_hidpi_macos():
    """macOS equivalent of Windows SetProcessDpiAwareness.

    Sets NSHighResolutionCapable on the running process via the ObjC runtime
    so that Tkinter renders at native Retina (physical) resolution instead of
    being upscaled 2× by the OS.  Must be called *before* the Tk root window
    is created.  Safe no-op on non-macOS or if the ObjC bridge is unavailable.
    """
    if platform.system() != 'Darwin':
        return
    try:
        import ctypes, ctypes.util
        libobjc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')

        libobjc.objc_getClass.restype        = ctypes.c_void_p
        libobjc.objc_getClass.argtypes       = [ctypes.c_char_p]
        libobjc.sel_registerName.restype     = ctypes.c_void_p
        libobjc.sel_registerName.argtypes    = [ctypes.c_char_p]
        libobjc.objc_msgSend.restype         = ctypes.c_void_p
        libobjc.objc_msgSend.argtypes        = [ctypes.c_void_p, ctypes.c_void_p]

        def _cls(name):  return libobjc.objc_getClass(name.encode())
        def _sel(name):  return libobjc.sel_registerName(name.encode())
        def _msg(obj, sel, *args):
            libobjc.objc_msgSend.argtypes = (
                [ctypes.c_void_p, ctypes.c_void_p] + [type(a) for a in args]
            )
            return libobjc.objc_msgSend(obj, sel, *args)

        bundle    = _msg(_cls('NSBundle'), _sel('mainBundle'))
        info_dict = _msg(bundle, _sel('infoDictionary'))

        # Build key NSString and value NSNumber(YES)
        key = _msg(_cls('NSString'), _sel('stringWithUTF8String:'), b'NSHighResolutionCapable')
        val = _msg(_cls('NSNumber'), _sel('numberWithBool:'), ctypes.c_bool(True))

        # infoDictionary is typically an NSMutableDictionary at runtime
        _msg(info_dict, _sel('setValue:forKey:'), key, val)   # type: ignore[arg-type]
    except Exception:
        pass   # non-fatal — worst case: non-Retina rendering


def _get_display_scale(widget):
    """Return the integer HiDPI scale factor (1 or 2) for the display."""
    try:
        ppi = widget.winfo_fpixels('1i')   # actual pixels per inch
        return 2 if ppi > 120 else 1
    except Exception:
        return 1

# ---------------------------
# Tweakable parameter frames 
# ---------------------------
def param_frame(parent, padding=8, border_color="#797979", bg="#303030", width=None, height=80, do_pack=True):
    """Create a framed parameter panel used throughout the GUI."""
    outer = tk.Frame(parent, bg=border_color)
    if do_pack:
        outer.pack(padx=4, pady=4)
    inner = tk.Frame(outer, bg=bg, padx=padding, pady=padding)
    if width or height:
        inner.config(width=width, height=height)
        inner.pack_propagate(False)
    inner.pack(fill='both', expand=True)
    return outer, inner




def rich_label(parent, segments, bg=None, fg="white"):
    """Render a symbol with superscript/subscript on a tk.Canvas.

    Uses exact pixel placement so subscript descenders are never clipped.

    Parameters
    ----------
    parent : tk.Widget
    segments : list of (str, str) where the second element is one of:
        ``'n'`` — normal baseline
        ``'s'`` — subscript (small, lowered)
        ``'p'`` — superscript (small, raised)
    bg : str or None   Background colour; defaults to parent's background.
    fg : str           Foreground (text) colour.

    Returns
    -------
    tk.Canvas   Sized exactly to the rendered content.
    """
    from tkinter import font as tkfont
    bg = bg or parent.cget('bg')

    base_f  = tkfont.Font(family="Georgia", size=11,
                          weight="bold", slant="italic")
    small_f = tkfont.Font(family="Georgia", size=8, slant="italic")
    tiny_f  = tkfont.Font(family="Georgia", size=6, slant="italic")

    # Baseline sits here (px from top of canvas).
    # Normal text hangs below it; superscripts rise above; subscripts drop further.
    BASELINE   = 13
    SUB_DROP   =  3   # extra pixels below baseline for subscript anchor
    SUBSUB_DROP = 4   # deeper drop for sub-subscript (e.g. the 'z' in v_z)
    SUP_LIFT   =  5   # pixels above baseline for superscript anchor
    CANVAS_H   = BASELINE + SUBSUB_DROP + tiny_f.metrics("linespace") // 2 + 2

    def _font(style):
        return base_f if style == 'n' else (tiny_f if style == 'ss' else small_f)

    # Measure total width
    total_w = 4
    for text, style in segments:
        total_w += _font(style).measure(text)
    total_w += 4

    cv = tk.Canvas(parent, width=total_w, height=CANVAS_H,
                   bg=bg, highlightthickness=0, bd=0)

    x = 2
    for text, style in segments:
        if style == 'n':
            cv.create_text(x, BASELINE, text=text,
                           font=base_f, fill=fg, anchor='sw')
            x += base_f.measure(text)
        elif style == 's':
            cv.create_text(x, BASELINE + SUB_DROP, text=text,
                           font=small_f, fill=fg, anchor='sw')
            x += small_f.measure(text)
        elif style == 'ss':
            cv.create_text(x, BASELINE + SUBSUB_DROP, text=text,
                           font=tiny_f, fill=fg, anchor='sw')
            x += tiny_f.measure(text)
        elif style == 'p':
            cv.create_text(x, BASELINE - SUP_LIFT, text=text,
                           font=small_f, fill=fg, anchor='sw')
            x += small_f.measure(text)

    return cv


# Import core
try:
    from .core import (SONGSPhy, DEFAULT_DIFFUSE_PARAMS, place_galaxies,
                       _save_cube_hdf5, _sersic_total_flux_3d)
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from songs.core import (SONGSPhy, DEFAULT_DIFFUSE_PARAMS, place_galaxies,
                            _save_cube_hdf5, _sersic_total_flux_3d)

try:
    from .utils import apply_and_convolve_noise
except Exception:
    from songs.utils import apply_and_convolve_noise

import json
import h5py
import time

# Import visualise helpers (module provides moment0, moment1, spectrum)
try:
    from .visualise import *
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from songs.visualise import *

import sys
import tkinter as tk
from tkinter import ttk

class TextRedirector:
    """Redirect writes into a Tk ``Text`` widget behaving like a stream.

    Use this helper to capture and display program output inside the GUI
    (for example, to show progress logs, exceptions, or print() output).
    ``TextRedirector`` implements a minimal stream interface (``write`` and
    ``flush``) so it can be assigned directly to ``sys.stdout`` or
    ``sys.stderr``; written text is inserted into the provided Tk Text
    widget and scrolled to the end so the latest output is visible.

    Threading note
    --------------
    - The class itself is not thread-safe: writes coming from background
      threads should be marshalled to the Tk mainloop (e.g. via
      ``widget.after(...)``) if there is a risk of concurrent access.

    Parameters
    ----------
    widget : tk.Text
        The Tk Text widget where text will be appended.
    tag : str, optional
        Optional text tag name to apply to inserted text (default ``'stdout'``).

    Example
    -------
    Redirect stdout into a Text widget::

        txt = tk.Text(root)
        txt.pack()
        sys.stdout = TextRedirector(txt, tag='log')

    """

    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, string):
        try:
            if not self.widget.winfo_exists():
                sys.__stdout__.write(string)
                return
            self.widget.configure(state="normal")
            self.widget.insert("end", string, (self.tag,))
            self.widget.see("end")
            self.widget.configure(state="disabled")
        except Exception:
            try:
                sys.__stdout__.write(string)
            except Exception:
                pass

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

class LogWindow(tk.Toplevel):
    """Top-level log window that captures and displays stdout/stderr.

    ``LogWindow`` creates a simple resizable Toplevel containing a Tk
    ``Text`` widget and installs ``TextRedirector`` instances on
    ``sys.stdout`` and ``sys.stderr`` so that all subsequent ``print``
    output and uncaught exception tracebacks are visible in the GUI. The
    window restores the original streams when closed.

    Behaviour
    ---------
    - Creating an instance replaces ``sys.stdout`` and ``sys.stderr`` in
        the running interpreter until the window is closed (``on_close``).
    - The window configures a separate text tag for ``stderr`` so error
        messages are coloured differently.

    Example
    -------
    >>> log = LogWindow(root)
    >>> log.deiconify()  # show the window

    """

    def __init__(self, master):
        super().__init__(master)
        # Hide before anything can map this window. It is created eagerly at
        # startup purely to capture stdout/stderr and is only shown on demand
        # (see SONGSGUI._show_log_window, which also applies the themed title
        # bar once it is actually visible) — without this it briefly appeared
        # as a small blank "Logs" window during startup.
        self.withdraw()
        self.title("Logs")
        self.geometry("700x320")

        _theme = getattr(master, '_theme', 'light')
        t = _THEMES.get(_theme, _THEMES['light'])
        _acc    = t['ACCENT']
        _big_bg = t['BIG_BG']

        self.configure(bg=_big_bg)
        self.text = tk.Text(
            self, bg=_big_bg, fg=_acc, insertbackground=_acc,
            font=('Courier', 12), relief='flat', bd=0,
            selectbackground=_acc, selectforeground=_big_bg,
            padx=8, pady=6,
        )
        self.text.pack(fill="both", expand=True, padx=4, pady=4)
        self.text.tag_configure("stderr", foreground="#e55b5b")
        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # Optionally restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


_NATIVE_IV_CLS = None


def _native_imageview_class():
    """Lazily define (exactly once) a click-through NSImageView subclass.

    ObjC classes are registered globally, so defining this twice raises.
    """
    global _NATIVE_IV_CLS
    if _NATIVE_IV_CLS is None:
        import AppKit

        class _SongsBannerImageView(AppKit.NSImageView):
            def hitTest_(self, point):
                return None  # click-through: never intercept mouse events

        _NATIVE_IV_CLS = _SongsBannerImageView
    return _NATIVE_IV_CLS


class SONGSGUI(tk.Tk):
    """Main GUI application for interactively configuring and running
    SONGS simulations.

    This class implements a compact, self-contained Tk application that
    exposes the most commonly-used parameters of the generator via a
    three-column layout of parameter panels. Controls include numeric
    sliders, textual inputs and convenience buttons that invoke high-level
    visualisation helpers (``moment0``, ``moment1``, ``spectrum``) or
    persist generated results to disk.

    Key behaviour
    --------------
    - The generator is constructed from the current UI values and stored
        on ``self.generator``. Calling ``Generate`` runs the generator in a
        background daemon thread so the UI remains responsive; generated
        results become available via ``self.generator.results``.
    - Visualisation buttons call into functions defined in
        :mod:`songs.visualise` which create Matplotlib figures; these
        functions are intentionally separate from the generator core so the
        GUI remains a thin orchestration layer.
    - Temporary files created by :func:`latex_label` are tracked in the
        module-level ``_MATH_TEMPFILES`` list and cleaned up when the GUI is
        closed via ``_on_close``.

    Threading and shutdown
    ----------------------
    - Generation and save operations spawn background daemon threads. The
        UI schedules finalisation callbacks back on the main thread using
        ``self.after(...)`` when worker threads complete.
    - Closing the main window triggers a cleanup of temporary files and
        forces process termination to avoid orphaned interpreters. If you
        prefer a softer shutdown that joins worker threads, modify
        ``_on_close`` accordingly.

    Usage example
    -------------
    Run the GUI as a script::

            python -m songs.gui

    Or instantiate from Python::

            from songs.gui import SONGSGUI
            app = SONGSGUI()
            app.mainloop()

    """

    def __init__(self, theme: str = 'light'):
        super().__init__()
        self.title('SONGS GUI')
        self.WINDOW_HEIGHT = 840
        self._theme = theme if theme in ('dark', 'light') else 'light'
        self._galaxy_centers = None   # pre-sampled positions, refreshed by Regenerate
        # Bumped every time fresh results become available (successful
        # generate or load) — lets show_analysis/show_slice tell whether an
        # already-open viewer window is showing this run's data or a stale
        # one from a previous generate, so regenerating actually refreshes
        # the view instead of just re-raising the old window untouched.
        self._results_generation = 0
        # Handshake between the background generation worker and
        # _poll_generation_done() on the main thread (see _run_generate).
        self._gen_done = False
        self._gen_error = None
        self._gen_preview_results = None
        self._gen_in_progress = False
        self._large_dataset_mode = False
        self.resizable(False, False)

        # Window icon
        try:
            _icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'songs_icon.png'))
            if not os.path.exists(_icon_path):
                _icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'assets', 'songs_icon.png'))
            _icon_img = Image.open(_icon_path).convert('RGBA')
            self._icon_photo = ImageTk.PhotoImage(_icon_img)
            self.iconphoto(True, self._icon_photo)
        except Exception:
            pass

        # Create a hidden log window immediately
        self.log_window = LogWindow(self)
        self.log_window.withdraw()
        self._is_closing = False
        self._stop_requested = False   # set by the Stop button; checked by the generation workers
        self._preview_results = None   # noisy version of self.generator.results, for Slice/Analysis (see _run_generate)
        self._slice_win = None      # currently-open SliceViewer, if any (see show_slice/_raise_existing)
        self._analysis_win = None   # currently-open AnalysisViewer, if any (see show_analysis/_raise_existing)

        t = _THEMES[self._theme]
        self.configure(bg=t['BG'])

        # Horizontal root layout: logo strip on left, right column (cards + buttons) on right
        self._root_frame = tk.Frame(self, bg=t['BG'])
        self._root_frame.pack(fill='both', expand=True)

        # Left: vertical logo strip — spans full window height
        self._logo_strip = tk.Frame(self._root_frame, bg=t['BG'])
        self._logo_strip.pack(side='left', fill='y')

        # Logo fills full height; buttons are overlaid with place
        self._logo_lbl = None
        self._load_logo()
        self._banner_btns = []   # (canvas, redraw_fn) — rebuilt in _place_banner_btns
        self._place_banner_btns()

        # Right column: cards on top, buttons at bottom (logo not included)
        self._right_col = tk.Frame(self._root_frame, bg=t['BG'])
        self._right_col.pack(side='left', fill='both', expand=True)

        # Button bar at the bottom of the right column only
        self._btn_area = tk.Frame(self._right_col, bg=t['BG'])
        self._btn_area.pack(side='bottom', fill='x')

        # Cards container fills remaining space above buttons
        self.container = tk.Frame(self._right_col, bg=t['BG'])
        self.container.pack(side='top', fill='both', expand=True, padx=4, pady=4)

        self.generator = None
        self._has_results = False

        self._build_widgets()
        self._apply_large_dataset_mode()

        # Fix height to WINDOW_HEIGHT; width = logo + content (measured after layout).
        self.update_idletasks()
        total_w = self.winfo_reqwidth()
        self.geometry(f"{total_w}x{self.WINDOW_HEIGHT}")
        self.resizable(False, False)

        # Retina-sharp native logo overlay (no-op if PyObjC unavailable);
        # delayed so the NSWindow exists and layout has settled
        self.after(250, self._add_native_logo_overlay)
        self.after(250, self._sync_toggle_height_to_generate)

        # Force the title bar to match the theme (overrides the system theme)
        self._set_window_appearance()
        self.after(250, self._set_window_appearance)

        self.after(100, self._raise_to_front)

        self.protocol('WM_DELETE_WINDOW', self._on_close)



    # ---------------------------
    # Theme helpers
    # ---------------------------
    def _set_window_appearance(self):
        """Force the macOS title bar to match the app theme, overriding the
        system (light/dark) appearance."""
        try:
            import AppKit
            # No update_idletasks() — see visualise._apply_window_appearance:
            # forcing it here mapped the main window before its cards were
            # built, showing a small blank 'SONGS GUI' window at startup. The
            # after(250, ...) retry in __init__ covers the real case.
            title = str(self.title())
            nswin = next((w for w in AppKit.NSApplication.sharedApplication().windows()
                          if str(w.title()) == title), None)
            if nswin is None:
                return
            name = (AppKit.NSAppearanceNameAqua if self._theme == 'light'
                    else AppKit.NSAppearanceNameDarkAqua)
            nswin.setAppearance_(AppKit.NSAppearance.appearanceNamed_(name))
        except Exception:
            # Non-macOS / PyObjC unavailable — leave the title bar as-is.
            pass

    def _load_logo(self):
        t = _THEMES[self._theme]
        bg = t['BG']
        try:
            logo_file = t['LOGO']
            logo_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'assets', logo_file))
            if not os.path.exists(logo_path):
                logo_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), 'assets', logo_file))
            _img = Image.open(logo_path).convert('RGBA')
            _logo_h = self.WINDOW_HEIGHT
            _w = int(_logo_h * _img.width / _img.height)
            _img = _img.resize((_w, _logo_h), Image.LANCZOS)
            self._logo_photo = ImageTk.PhotoImage(_img)
            if self._logo_lbl is None:
                # padx/pady=0: Labels default to 1px internal padding, which
                # would offset this image 1pt from the native overlay above it
                self._logo_lbl = tk.Label(
                    self._logo_strip, image=self._logo_photo,
                    bg=bg, borderwidth=0, padx=0, pady=0, highlightthickness=0)
                self._logo_lbl.pack(side='top')
            else:
                self._logo_lbl.configure(image=self._logo_photo, bg=bg)
            self._logo_lbl.image = self._logo_photo
        except Exception:
            if self._logo_lbl is None:
                self._logo_lbl = tk.Label(
                    self._logo_strip, text='SONGS', bg=bg, fg=t['ACCENT'],
                    font=('Helvetica', 11, 'bold'), wraplength=80)
                self._logo_lbl.pack(side='top', pady=20)

    def _raise_to_front(self):
        """Bring the window to the front after the event loop has started."""
        self.lift()
        self.attributes("-topmost", True)
        self.after(200, lambda: self.attributes("-topmost", False))
        try:
            import AppKit
            AppKit.NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass

    def _add_native_logo_overlay(self):
        """Overlay a Retina-aware NSImageView over the vertical logo strip.

        Tk 8.6 draws PhotoImages point-per-pixel, which macOS stretches 2x
        onto HiDPI screens (blurry). NSImageView renders at the native
        backing scale, so the logo stays crisp. The overlay starts below the
        top button row and stops above the bottom toggle row, and is
        click-through; on any failure the Tk-rendered logo simply remains
        visible as the fallback.
        """
        try:
            import AppKit

            t = _THEMES[self._theme]
            logo_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', '..', 'assets', t['LOGO']))
            if not os.path.exists(logo_path):
                logo_path = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), 'assets', t['LOGO']))
            if not os.path.exists(logo_path) or self._logo_lbl is None:
                return

            title = self.title()
            nswin = next((w for w in AppKit.NSApplication.sharedApplication().windows()
                          if str(w.title()) == title), None)
            if nswin is None:
                # window not mapped yet (startup) — retry once shortly
                if not getattr(self, '_logo_overlay_retried', False):
                    self._logo_overlay_retried = True
                    self.after(300, self._add_native_logo_overlay)
                return
            self._logo_overlay_retried = False
            content = nswin.contentView()

            if getattr(self, '_logo_overlay_view', None) is not None:
                self._logo_overlay_view.removeFromSuperview()
                self._logo_overlay_view = None

            self.update_idletasks()
            fx = self._logo_lbl.winfo_rootx() - self.winfo_rootx()
            fy = self._logo_lbl.winfo_rooty() - self.winfo_rooty()
            disp_w = self._logo_lbl.winfo_width()
            disp_h = self._logo_lbl.winfo_height()
            if disp_w < 10 or disp_h < 10:
                return

            # leave the top button row and bottom toggle row (+ the
            # Large-Dataset Mode note, when shown) uncovered
            overlay_w = disp_w
            top_row_h = 50   # PAD + BH + PAD
            bot_row_h = getattr(self, '_banner_bottom_reserved_h', 50)
            overlay_h = max(1, disp_h - top_row_h - bot_row_h)

            # crop the matching middle fraction of the FULL-RES source
            src = Image.open(logo_path)
            top_crop_px = round(src.height * top_row_h / disp_h)
            bot_crop_px = round(src.height * bot_row_h / disp_h)
            tmp = os.path.join(tempfile.gettempdir(), '_songs_logo_overlay.png')
            src.crop((0, top_crop_px, src.width, src.height - bot_crop_px)).save(tmp)
            nsimg = AppKit.NSImage.alloc().initWithContentsOfFile_(tmp)
            if nsimg is None:
                return

            if content.isFlipped():
                oy = fy + top_row_h
            else:
                oy = content.frame().size.height - fy - overlay_h - top_row_h
            iv = _native_imageview_class().alloc().initWithFrame_(
                ((fx, oy), (overlay_w, overlay_h)))
            iv.setImage_(nsimg)
            iv.setImageScaling_(AppKit.NSImageScaleAxesIndependently)
            iv.setImageFrameStyle_(AppKit.NSImageFrameNone)
            content.addSubview_(iv)
            self._logo_overlay_view = iv
        except Exception:
            pass

    def _place_banner_btns(self):
        """Create (or recreate) the three overlay buttons on the logo label."""
        import webbrowser
        t = _THEMES[self._theme]
        # Destroy previous banner buttons if any
        for cv, _ in self._banner_btns:
            try: cv.destroy()
            except Exception: pass
        self._banner_btns.clear()
        _old_note = getattr(self, '_large_dataset_note_wrap', None)
        if _old_note is not None:
            try: _old_note.destroy()
            except Exception: pass

        # Dimensions — all three buttons same height, horizontal row
        BH  = 30   # button height px
        PAD = 10   # margin from banner edges
        GAP = 6    # gap between buttons

        # Banner display width (image was resized proportionally)
        try:
            banner_w = self._logo_lbl.winfo_reqwidth()
            if banner_w < 10:
                raise ValueError
        except Exception:
            banner_w = 200
        usable   = banner_w - 2 * PAD
        THEME_W  = 36          # theme button a bit narrower (icon only)
        LINK_W   = (usable - THEME_W - 2 * GAP) // 2   # split remainder

        def _make_btn(text, w, cmd, is_theme=False):
            cv = tk.Canvas(self._logo_lbl, width=w, height=BH,
                           highlightthickness=0, bd=0, cursor='pointinghand')

            def _redraw(cv=cv, text=text, w=w, is_theme=is_theme):
                _t = _THEMES[self._theme]
                cv.configure(bg=_t['BG'])
                cv.delete('all')
                # dark bg + accent text in dark mode; light bg + accent text in light
                cv.create_rectangle(1, 1, w-1, BH-1, fill=_t['BG'], outline=_t['ACCENT'], width=1)
                lbl = text if not is_theme else ('☀' if self._theme == 'dark' else '☾')
                fnt = ('Helvetica', 13, 'bold') if is_theme else ('Courier', 9, 'bold')
                cv.create_text(w // 2, BH // 2,
                               text=lbl, fill=_t['ACCENT'], font=fnt)

            _redraw()
            cv.bind('<ButtonRelease-1>', lambda e, c=cmd: c())
            cv.bind('<Enter>', lambda e, cv=cv, fn=_redraw:
                    (cv.delete('all'),
                     cv.create_rectangle(1, 1, cv.winfo_reqwidth()-1, BH-1,
                                         fill=_THEMES[self._theme]['ACCENT'],
                                         outline=_THEMES[self._theme]['ACCENT'], width=1),
                     cv.create_text(cv.winfo_reqwidth()//2, BH//2,
                                    text=cv._lbl_text,
                                    fill=_THEMES[self._theme]['BG'],
                                    font=cv._lbl_font)))
            cv.bind('<Leave>', lambda e, fn=_redraw: fn())
            self._banner_btns.append((cv, _redraw))
            # store for hover handler
            cv._lbl_text = text if not is_theme else ('☀' if self._theme == 'dark' else '☾')
            cv._lbl_font = ('Helvetica', 13, 'bold') if is_theme else ('Courier', 9, 'bold')
            cv._is_theme = is_theme
            return cv

        btns = [
            _make_btn('☀' if self._theme == 'dark' else '☾',
                      THEME_W, self._toggle_theme, is_theme=True),
            _make_btn('GitHub', LINK_W,
                      lambda: webbrowser.open('https://github.com/arnablahiry/GalCubeCraft')),
            _make_btn('API Docs', LINK_W,
                      lambda: webbrowser.open('https://arnablahiry.github.io/software/songs')),
        ]
        self._theme_btn = btns[0]

        # Place in a horizontal row, top of banner
        widths = [THEME_W, LINK_W, LINK_W]
        x = PAD
        for cv, w in zip(btns, widths):
            cv.place(x=x, y=PAD)
            x += w + GAP

        # Bottom-of-banner toggle — full width of the three buttons above,
        # height/bottom-margin synced to the Generate button (see
        # _sync_toggle_height_to_generate) so the two align on screen.
        TOGGLE_W = usable
        BH_T     = getattr(self, '_toggle_bh', BH)
        MARGIN_T = getattr(self, '_toggle_margin', PAD)
        toggle_cv = tk.Canvas(self._logo_lbl, width=TOGGLE_W, height=BH_T,
                              highlightthickness=0, bd=0, cursor='pointinghand')

        def _toggle_label():
            return 'Disable Large-Dataset Mode' if self._large_dataset_mode else 'Toggle Large-Dataset Mode'

        def _toggle_redraw(cv=toggle_cv):
            _t = _THEMES[self._theme]
            on = self._large_dataset_mode
            fill    = _t['ACCENT'] if on else _t['BG']
            txt_fill = _t['BG'] if on else _t['ACCENT']
            cv.configure(bg=_t['BG'])
            cv.delete('all')
            cv.create_rectangle(1, 1, TOGGLE_W - 1, BH_T - 1,
                                fill=fill, outline=_t['ACCENT'], width=1)
            cv.create_text(TOGGLE_W // 2, BH_T // 2, text=_toggle_label(),
                           fill=txt_fill, font=('Courier', 9, 'bold'))

        def _toggle_hover_enter(e):
            if self._large_dataset_mode:
                return
            _t = _THEMES[self._theme]
            toggle_cv.delete('all')
            toggle_cv.create_rectangle(1, 1, TOGGLE_W - 1, BH_T - 1,
                                       fill=_t['ACCENT'], outline=_t['ACCENT'], width=1)
            toggle_cv.create_text(TOGGLE_W // 2, BH_T // 2, text=_toggle_label(),
                                  fill=_t['BG'], font=('Courier', 9, 'bold'))

        def _toggle_hover_leave(e):
            _toggle_redraw()

        _toggle_redraw()
        toggle_cv.bind('<ButtonRelease-1>', lambda e: self._toggle_large_dataset_mode())
        toggle_cv.bind('<Enter>', _toggle_hover_enter)
        toggle_cv.bind('<Leave>', _toggle_hover_leave)
        toggle_cv.place(x=PAD, y=-(MARGIN_T + BH_T), rely=1.0)

        self._banner_btns.append((toggle_cv, _toggle_redraw))
        self._large_dataset_btn = toggle_cv
        self._large_dataset_redraw = _toggle_redraw
        self._large_dataset_toggle_w = TOGGLE_W

        # Note box — shown only in Large-Dataset Mode, right above the toggle
        _note_wrap = tk.Frame(self._logo_lbl, bg=t['ACCENT'], padx=1, pady=1)
        _note_lbl = tk.Label(
            _note_wrap,
            text=("This mode is used to generate a large dataset of synthetic "
                 "spectral cubes with randomly sampled physical parameters. "
                 "Overall ranges/mean values are controllable by the user."),
            bg=t['BG'], fg=('white' if self._theme == 'dark' else 'black'),
            font=('Helvetica', 10),
            justify='left', anchor='w', padx=6, pady=6,
            wraplength=max(20, TOGGLE_W - 14),
        )
        _note_lbl.pack(fill='both', expand=True)
        self._large_dataset_note_wrap = _note_wrap
        self._large_dataset_note_lbl = _note_lbl

        self._position_large_dataset_note()

    def _position_large_dataset_note(self):
        """Show/hide + position the Large-Dataset Mode note box right above
        the banner toggle button, and let the native logo overlay know how
        much bottom space to leave uncovered."""
        wrap = getattr(self, '_large_dataset_note_wrap', None)
        toggle_w = getattr(self, '_large_dataset_toggle_w', None)
        if wrap is None or toggle_w is None:
            return
        PAD, GAP = 10, 6
        BH = getattr(self, '_toggle_bh', 30)
        MARGIN = getattr(self, '_toggle_margin', PAD)
        if self._large_dataset_mode:
            self.update_idletasks()
            wrap_h = wrap.winfo_reqheight()
            wrap.place(x=PAD, rely=1.0, y=-(MARGIN + BH + GAP + wrap_h), width=toggle_w)
            self._banner_bottom_reserved_h = MARGIN + BH + GAP + wrap_h + PAD
        else:
            wrap.place_forget()
            self._banner_bottom_reserved_h = MARGIN + BH + PAD
        self._add_native_logo_overlay()

    def _sync_toggle_height_to_generate(self):
        """Measure the Generate button's actual on-screen height and its
        margin to the window bottom, and apply the same values to the
        banner toggle button so the two are the same height and align."""
        gen = getattr(self, 'generate_btn', None)
        if gen is None:
            return
        self.update_idletasks()
        gen_h = gen.winfo_height()
        if gen_h < 4:
            self.after(100, self._sync_toggle_height_to_generate)
            return
        margin = max(0, (self.winfo_rooty() + self.winfo_height())
                        - (gen.winfo_rooty() + gen_h))
        if gen_h == getattr(self, '_toggle_bh', None) and margin == getattr(self, '_toggle_margin', None):
            return
        self._toggle_bh = gen_h
        self._toggle_margin = margin
        self._place_banner_btns()
        self.after(50, self._add_native_logo_overlay)

    def _toggle_large_dataset_mode(self):
        self._large_dataset_mode = not self._large_dataset_mode
        if hasattr(self, '_large_dataset_redraw'):
            self._large_dataset_redraw()
        self._apply_large_dataset_mode()

    def _apply_large_dataset_mode(self):
        """Swap every mode-specific card/slider between its single-cube and
        Large-Dataset Mode form, position the banner note, swap Generate's
        label/action, and show/hide the dataset-generation progress bar.
        The window is always the same fixed 3-column layout — no column is
        added/removed and no resizing happens here."""
        self._swap_mode_specific_cards()
        self._position_large_dataset_note()

        gen_btn = getattr(self, 'generate_btn', None)
        if gen_btn is not None and not getattr(self, '_ld_generating', False):
            if self._large_dataset_mode:
                gen_btn.configure(text='Generate Dataset')
                gen_btn._btn_cmd = self.generate_dataset
            else:
                gen_btn.configure(text='Generate')
                gen_btn._btn_cmd = self.generate
            gen_btn.unbind('<ButtonRelease-1>')
            gen_btn.bind('<ButtonRelease-1>', lambda e, b=gen_btn: b._btn_cmd())

        # Progress bar takes Slice/Analysis/Save's place in the button bar
        # (meaningless for a batch run); Reset stays put on the right.
        prog = getattr(self, '_ld_progress_frame', None)
        if prog is not None:
            others = (self.load_btn, self.slice_btn, self.analysis_btn, self.save_btn)
            if self._large_dataset_mode:
                for b in others:
                    b.pack_forget()
                prog.pack(side='left', padx=4, expand=True, fill='x',
                         before=self.new_instance_btn)
                if not getattr(self, '_ld_generating', False):
                    self._set_ld_progress(0, int(self.n_samples_var.get()))
            else:
                prog.pack_forget()
                for b in others:
                    b.pack(side='left', padx=4, expand=True, fill='x',
                          before=self.new_instance_btn)

        # Stop is only meaningful in Large-Dataset Mode: a single cube's
        # generate_cubes() call is one atomic step that can't actually be
        # interrupted mid-flight, so Stop was a no-op button there — only
        # the Large-Dataset Mode per-cube loop can genuinely be cancelled.
        stop_sq = getattr(self, 'stop_btn', None)
        stop_sq = stop_sq.master if stop_sq is not None else None
        if stop_sq is not None:
            if self._large_dataset_mode:
                stop_sq.pack(side='left', padx=4, before=self.new_instance_btn)
            else:
                stop_sq.pack_forget()

    # ---------------------------
    # Generic widget greyout (used by Large-Dataset Mode / Noise toggle)
    # ---------------------------
    def _blend_hex(self, color, target, t):
        """Blend ``color`` a fraction ``t`` of the way toward ``target``
        (0 -> color, 1 -> target). Used to simulate reduced opacity, since
        Tk widgets have no real alpha channel."""
        try:
            c1 = self.winfo_rgb(color)
            c2 = self.winfo_rgb(target)
            r = (round(c1[0] + (c2[0] - c1[0]) * t) >> 8) & 0xff
            g = (round(c1[1] + (c2[1] - c1[1]) * t) >> 8) & 0xff
            b = (round(c1[2] + (c2[2] - c1[2]) * t) >> 8) & 0xff
            return f'#{r:02x}{g:02x}{b:02x}'
        except Exception:
            return color

    _DIM_COLOR_OPTS = ('bg', 'background', 'fg', 'foreground', 'troughcolor',
                       'activebackground', 'insertbackground',
                       'highlightbackground', 'highlightcolor',
                       'selectbackground', 'selectforeground',
                       'readonlybackground')

    def _dim_walk(self, root, dim):
        """Recursively grey out (``dim=True``) or restore (``dim=False``)
        ``root`` and all its descendants: text, slider troughs/pills,
        borders, and canvas-drawn labels/pill-buttons are all faded toward
        the card background, and every interactive widget (entries,
        sliders, pill buttons, hover-styled labels) is made unclickable."""
        _t = _THEMES[self._theme]
        fade_target = _t['BIG_BG']
        factor = 0.8
        factor_heading = 0.92  # card headings fade further than the rest of the card

        def blend(color, heading=False):
            return self._blend_hex(color, fade_target, factor_heading if heading else factor)

        def walk(w):
            if dim:
                if not hasattr(w, '_dim_orig'):
                    orig = {}
                    try:
                        opts = w.keys()
                    except Exception:
                        opts = []
                    for o in self._DIM_COLOR_OPTS:
                        if o in opts:
                            try:
                                cur = w.cget(o)
                            except Exception:
                                continue
                            if cur:
                                orig[o] = cur
                    w._dim_orig = orig
                    is_heading = getattr(w, '_is_card_heading', False)
                    for o, cur in orig.items():
                        _use_heading = is_heading and o in ('fg', 'foreground')
                        try: w.configure(**{o: blend(cur, heading=_use_heading)})
                        except Exception: pass
                    if isinstance(w, tk.Entry):
                        try:
                            w.configure(disabledforeground=blend(orig.get('fg', w.cget('fg'))),
                                       disabledbackground=blend(orig.get('bg', w.cget('bg'))))
                        except Exception: pass
                    try:
                        if 'state' in w.keys():
                            w._dim_orig_state = w.cget('state')
                            w.configure(state='disabled')
                    except Exception: pass
                    try:
                        if 'cursor' in w.keys():
                            w._dim_orig_cursor = w.cget('cursor')
                            w.configure(cursor='arrow')
                    except Exception: pass
                    if isinstance(w, tk.Canvas):
                        dim_items = {}
                        for item in w.find_all():
                            for io in ('fill', 'outline'):
                                try:
                                    cur = w.itemcget(item, io)
                                except Exception:
                                    continue
                                if cur:
                                    dim_items.setdefault(item, {})[io] = cur
                                    try: w.itemconfigure(item, **{io: blend(cur)})
                                    except Exception: pass
                        w._dim_items = dim_items
                    # unbind any directly-bound interaction handlers (custom
                    # canvas buttons, hover-styled Labels, etc.) — class-level
                    # bindings on Entry/Scale are untouched by this
                    dim_binds = {}
                    for seq in ('<ButtonRelease-1>', '<Button-1>', '<B1-Motion>', '<Motion>', '<Enter>', '<Leave>'):
                        try:
                            dim_binds[seq] = w.bind(seq)
                            w.unbind(seq)
                        except Exception: pass
                    w._dim_binds = dim_binds
            else:
                if hasattr(w, '_dim_orig'):
                    for o, cur in w._dim_orig.items():
                        try: w.configure(**{o: cur})
                        except Exception: pass
                    if isinstance(w, tk.Entry):
                        try:
                            w.configure(disabledforeground=w._dim_orig.get('fg', ''),
                                       disabledbackground=w._dim_orig.get('bg', ''))
                        except Exception: pass
                    del w._dim_orig
                if hasattr(w, '_dim_orig_state'):
                    try: w.configure(state=w._dim_orig_state)
                    except Exception: pass
                    del w._dim_orig_state
                if hasattr(w, '_dim_orig_cursor'):
                    try: w.configure(cursor=w._dim_orig_cursor)
                    except Exception: pass
                    del w._dim_orig_cursor
                if isinstance(w, tk.Canvas):
                    for item, io_map in getattr(w, '_dim_items', {}).items():
                        for io, cur in io_map.items():
                            try: w.itemconfigure(item, **{io: cur})
                            except Exception: pass
                    if hasattr(w, '_dim_items'):
                        del w._dim_items
                for seq, fn in getattr(w, '_dim_binds', {}).items():
                    try:
                        w.bind(seq, fn) if fn else w.unbind(seq)
                    except Exception: pass
                if hasattr(w, '_dim_binds'):
                    del w._dim_binds
            for child in w.winfo_children():
                walk(child)

        walk(root)

    def _lock_walk(self, root, lock):
        """Recursively make every descendant of ``root`` unclickable
        (``lock=True``) or restore it (``lock=False``) — no colour changes
        at all, unlike ``_dim_walk``. Used when viewing a loaded cube: the
        controls stay fully legible, they just can't be edited."""
        def walk(w):
            if lock:
                try:
                    opts = w.keys()
                except Exception:
                    opts = []
                # tk.Scale can spuriously re-invoke its 'command' as a side
                # effect of the redraw triggered by state='disabled' (it
                # recomputes the thumb position from the value on reconfig).
                # That would call make_slider()'s update(), which re-.set()s
                # the bound var and re-fires every trace on it — including
                # _auto_update_generator, which clobbers the loaded-view
                # generator stub. Clear the command before disabling so
                # nothing fires while locked.
                if isinstance(w, tk.Scale) and 'command' in opts:
                    try:
                        if not hasattr(w, '_lock_orig_command'):
                            w._lock_orig_command = w.cget('command')
                        w.configure(command='')
                    except Exception: pass
                # Skip 'state' on plain Labels: they're not draggable/typeable
                # like Scale/Entry, so unbinding their events is enough to
                # make them inert — and on macOS, state='disabled' makes Tk
                # render Label text in the system dim colour regardless of
                # 'fg', which is exactly the fade-out this method is meant
                # to avoid.
                if 'state' in opts and not isinstance(w, tk.Label):
                    try:
                        if not hasattr(w, '_lock_orig_state'):
                            w._lock_orig_state = w.cget('state')
                        w.configure(state='disabled')
                    except Exception: pass
                if 'cursor' in opts:
                    try:
                        if not hasattr(w, '_lock_orig_cursor'):
                            w._lock_orig_cursor = w.cget('cursor')
                        w.configure(cursor='arrow')
                    except Exception: pass
                lock_binds = {}
                for seq in ('<ButtonRelease-1>', '<Button-1>', '<B1-Motion>', '<Motion>', '<Enter>', '<Leave>'):
                    try:
                        lock_binds[seq] = w.bind(seq)
                        w.unbind(seq)
                    except Exception: pass
                w._lock_binds = lock_binds
            else:
                if hasattr(w, '_lock_orig_state'):
                    try: w.configure(state=w._lock_orig_state)
                    except Exception: pass
                    del w._lock_orig_state
                if hasattr(w, '_lock_orig_command'):
                    try: w.configure(command=w._lock_orig_command)
                    except Exception: pass
                    del w._lock_orig_command
                if hasattr(w, '_lock_orig_cursor'):
                    try: w.configure(cursor=w._lock_orig_cursor)
                    except Exception: pass
                    del w._lock_orig_cursor
                for seq, fn in getattr(w, '_lock_binds', {}).items():
                    try:
                        w.bind(seq, fn) if fn else w.unbind(seq)
                    except Exception: pass
                if hasattr(w, '_lock_binds'):
                    del w._lock_binds
            for child in w.winfo_children():
                walk(child)

        walk(root)

    def _swap_mode_specific_cards(self):
        """Swap every mode-specific bit of UI between its single-cube and
        Large-Dataset Mode form:
          - bc1 top (Number of samples / Max galaxies / Spatial pixel grid
            size) and bc1 tail (Use Noise? double-slider + Choose Save
            Folder) are shown only in Large-Dataset Mode; Number of
            galaxies / classic Use Noise? / the FOV preview are shown only
            in single-cube mode — swapped in place (no dimming).
          - Every Central Galaxy / Satellite / Diffuse Features slider is a
            make_dual_slider(): same card, single value vs. min/max range,
            toggled via self._dual_sliders.
        """
        ld = self._large_dataset_mode

        ld_only      = getattr(self, '_bc1_ld_only', [])
        ld_only_tail = getattr(self, '_bc1_ld_only_tail', [])
        bc1_single   = getattr(self, '_bc1_single_only', [])
        anchor_top   = getattr(self, '_card_spatial_res', None)  # bc1 top-of-card anchor

        if ld:
            for card in bc1_single:
                card.pack_forget()
            if anchor_top is not None:
                # Packing a previously-unmapped tk.Scale can spuriously
                # re-invoke its command (it recomputes the thumb's value
                # from pixel position once it gets a real width), silently
                # corrupting the var — save/restore across the pack calls.
                _saved_n_samples = self.n_samples_var.get()
                _saved_grid = self.spatial_pixel_dim_var.get()
                for card in ld_only:
                    card.pack(fill='x', padx=6, pady=3, before=anchor_top)
                self.update_idletasks()
                self.n_samples_var.set(_saved_n_samples)
                self.spatial_pixel_dim_var.set(_saved_grid)
            for card in ld_only_tail:
                card.pack(fill='x', padx=6, pady=3)
        else:
            for card in ld_only:
                card.pack_forget()
            for card in ld_only_tail:
                card.pack_forget()
            # Restore single-cube cards in their original relative order.
            if anchor_top is not None and self._card_ngals in bc1_single:
                self._card_ngals.pack(fill='x', padx=6, pady=3, before=anchor_top)
            # Use Noise? (classic) and the FOV preview are the trailing
            # items in bc1 — plain re-pack (append) restores them in order.
            if self._card_noise in bc1_single:
                self._card_noise.pack(fill='x', padx=6, pady=3)
            if self._prev_cv in bc1_single:
                self._prev_cv.pack(fill='x', padx=6, pady=(4, 6))

            # the pack_forget/pack above doesn't touch colours, but the
            # Use-Noise-dependent dim on the classic noise card's slider
            # could be stale after a rebuild — reassert it.
            updater = getattr(self, '_update_noise_dependent', None)
            if updater is not None:
                updater()

        # Swap every Central Galaxy / Satellite / Diffuse Features slider
        # between its single-value and min/max-range form.
        for apply_mode in getattr(self, '_dual_sliders', []):
            apply_mode()

        # The FOV preview draws its own colours dynamically each redraw,
        # and its content depends on fov/resolution — refresh it now.
        preview_redraw = getattr(self, '_draw_fov_preview_ref', None)
        if preview_redraw is not None:
            preview_redraw()

        if ld:
            sync = getattr(self, '_sync_dynamic_fov', None)
            if sync is not None:
                sync()

    def _redraw_banner_btns(self):
        for _, fn in self._banner_btns:
            fn()

    def _disable_theme_btn(self):
        cv = getattr(self, '_theme_btn', None)
        if cv is None:
            return
        _t = _THEMES[self._theme]
        _dim = _t.get('BTN_DIS_FG', '#555555')
        cv.delete('all')
        cv.create_rectangle(1, 1, cv.winfo_reqwidth()-1, cv.winfo_reqheight()-1,
                            fill=_t['BG'], outline=_dim, width=1)
        cv.create_text(cv.winfo_reqwidth()//2, cv.winfo_reqheight()//2,
                       text=cv._lbl_text, fill=_dim, font=cv._lbl_font)
        cv.unbind('<ButtonRelease-1>')
        cv.unbind('<Enter>')
        cv.unbind('<Leave>')
        cv.configure(cursor='arrow')

    def _enable_theme_btn(self):
        cv = getattr(self, '_theme_btn', None)
        if cv is None:
            return
        # Re-bind via _place_banner_btns (simplest: just redraw the banner)
        self._place_banner_btns()

    # Names of every tk.Var that must survive a theme switch
    _VAR_NAMES = [
        'bmin_var', 'bmaj_var', 'bpa_var', 'bmin_px_var', 'bmaj_px_var',
        'spatial_resolution', 'spatial_resolution_min_var', 'spatial_resolution_max_var',
        'n_var', 'hz_var', 'Se_var', 'Re_var', 'sigma_v_var',
        'fov', 'fov_min_var', 'fov_max_var', 'spectral_resolution', 'angle_x_var', 'angle_y_var', 'n_gals_var',
        'halo_Se_factor_var', 'halo_Re_factor_var', 'halo_sigma_vz_var',
        'bridge_Se_factor_var', 'bridge_width_start_factor_var',
        'bridge_width_end_factor_var', 'bridge_sigma_vz_var',
        'tail_Se_factor_var', 'tail_vel_gradient_var', 'tail_length_var',
        'tail_width_factor_var', 'tail_sigma_vz_var',
        'sat_brightness_frac_var', 'sat_Re_frac_var', 'sat_offset_min_var', 'sat_offset_max_var',
        'sat_offset_min_px_var', 'sat_offset_max_px_var',
        'beam_mode_var', 'allow_overlap_var',
        'n_samples_var', 'spatial_pixel_dim_var', 'max_gals_per_cube_var',
        'angle_x_min_var', 'angle_x_max_var', 'angle_y_min_var', 'angle_y_max_var',
        'n_min_var', 'n_max_var', 'use_noise_var', 'use_noise_ld_var', 'sn_peak_var',
        'sn_peak_min_var', 'sn_peak_max_var', 'save_folder_var',
        'hz_min_var', 'hz_max_var', 'Se_min_var', 'Se_max_var',
        'sigma_v_min_var', 'sigma_v_max_var',
        'sat_brightness_frac_min_var', 'sat_brightness_frac_max_var',
        'sat_Re_frac_min_var', 'sat_Re_frac_max_var',
        'halo_Se_factor_min_var', 'halo_Se_factor_max_var',
        'halo_Re_factor_min_var', 'halo_Re_factor_max_var',
        'halo_sigma_vz_min_var', 'halo_sigma_vz_max_var',
        'bridge_Se_factor_min_var', 'bridge_Se_factor_max_var',
        'bridge_width_start_factor_min_var', 'bridge_width_start_factor_max_var',
        'bridge_width_end_factor_min_var', 'bridge_width_end_factor_max_var',
        'bridge_sigma_vz_min_var', 'bridge_sigma_vz_max_var',
        'tail_Se_factor_min_var', 'tail_Se_factor_max_var',
        'tail_vel_gradient_min_var', 'tail_vel_gradient_max_var',
        'tail_length_min_var', 'tail_length_max_var',
        'tail_width_factor_min_var', 'tail_width_factor_max_var',
        'tail_sigma_vz_min_var', 'tail_sigma_vz_max_var',
    ]

    def _save_var_state(self) -> dict:
        return {name: getattr(self, name).get()
                for name in self._VAR_NAMES
                if hasattr(self, name)}

    def _restore_var_state(self, state: dict):
        for name, value in state.items():
            var = getattr(self, name, None)
            if var is not None:
                try:
                    var.set(value)
                except Exception:
                    pass

    def _toggle_theme(self):
        saved = self._save_var_state()
        self._theme = 'light' if self._theme == 'dark' else 'dark'
        # drop the stale native overlay immediately so the old theme's logo
        # never lingers on top while the Tk widgets beneath are re-themed
        if getattr(self, '_logo_overlay_view', None) is not None:
            try:
                self._logo_overlay_view.removeFromSuperview()
            except Exception:
                pass
            self._logo_overlay_view = None
        t = _THEMES[self._theme]
        self.configure(bg=t['BG'])
        self._root_frame.configure(bg=t['BG'])
        self._logo_strip.configure(bg=t['BG'])
        self._right_col.configure(bg=t['BG'])
        self._load_logo()
        self._place_banner_btns()
        self.after(50, self._add_native_logo_overlay)
        self._set_window_appearance()
        self._rebuild_widgets()
        self.after(150, self._sync_toggle_height_to_generate)
        self._restore_var_state(saved)
        import songs.visualise as _vis
        _vis._VIEWER_THEME = self._theme

    def _rebuild_widgets(self):
        for w in list(self.container.winfo_children()):
            w.destroy()
        for w in list(self._btn_area.winfo_children()):
            w.destroy()
        t = _THEMES[self._theme]
        self.container.configure(bg=t['BG'])
        self._btn_area.configure(bg=t['BG'])
        self._build_widgets()
        self._apply_large_dataset_mode()
        self.update_idletasks()
        total_w = self.winfo_reqwidth()
        self.geometry(f"{total_w}x{self.WINDOW_HEIGHT}")

    # ---------------------------
    # Slider helper
    # ---------------------------
    def make_slider(self, parent, label, var, from_, to,
                    resolution=0.01, fmt="{:.2f}", integer=False):
        """Create a labelled slider widget with snapping and a value label."""
        # Colours — fall back to safe defaults before _build_widgets sets them.
        bg      = getattr(self, '_slider_bg',     "#111111")
        fg      = getattr(self, '_slider_fg',     "#999999")
        acc     = getattr(self, '_slider_accent',  "#b8960a")
        trough  = getattr(self, '_slider_trough',  "#111111")

        _entry_bg   = getattr(self, '_entry_bg',    "#1a1a1a")
        _border_col = getattr(self, '_slider_border', "#785605FF")
        _wrap = tk.Frame(parent, bg=_border_col, padx=1, pady=1)
        fr = tk.Frame(_wrap, bg=bg)
        fr.pack(fill='both', expand=True)
        if label:
            tk.Label(fr, text=label, bg=bg, fg=fg,
                     font=("Helvetica", 8)).pack(anchor='w', pady=(0,2))
        slider_row = tk.Frame(fr, bg=bg)
        slider_row.pack(fill='x')

        # Editable entry that shows and accepts the current value
        entry_var = tk.StringVar(value=fmt.format(var.get()) if not integer
                                 else str(int(var.get())))
        entry = tk.Entry(slider_row, textvariable=entry_var,
                         width=6, justify='right',
                         bg=_entry_bg, fg=acc, insertbackground=acc,
                         relief='flat', highlightthickness=1,
                         highlightbackground=_border_col,
                         highlightcolor=acc,
                         font=("Helvetica", 8),
                         bd=2)
        entry.pack(side='right', padx=(4, 0))

        _thumb  = getattr(self, '_slider_thumb',  "#b8960a")
        _thumbh = getattr(self, '_slider_thumbhover', "#f0c040")
        scale = tk.Scale(slider_row, from_=from_, to=to, orient='horizontal',
                         resolution=resolution,
                         bg=_thumb, fg=fg, troughcolor=trough,
                         activebackground=_thumbh, highlightthickness=0,
                         sliderrelief='flat', bd=0, showvalue=False,
                         relief='flat', width=6)
        scale.pack(side='left', fill='x', expand=True, padx=(4, 0))
        step = resolution if resolution else 0.01
        busy = {'val': False}

        def snap(v):
            if integer:
                return int(round(float(v)))
            nsteps = round((float(v) - from_) / step)
            return from_ + nsteps * step

        def _fmt(v):
            try:    return fmt.format(v)
            except: return str(v)

        def update(v):
            if busy['val']: return
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

        # Commit entry value on Return or focus-out
        def _commit_entry(*_):
            if busy['val']: return
            try:
                raw = entry_var.get().strip()
                v = float(raw)
                v = max(from_, min(to, snap(v)))
                busy['val'] = True
                var.set(v)
                entry_var.set(_fmt(v))
                scale.set(v)
                busy['val'] = False
            except (ValueError, tk.TclError):
                pass  # leave entry as-is; user may still be typing

        entry.bind('<Return>',    _commit_entry)
        entry.bind('<FocusOut>',  _commit_entry)

        try:
            def _var_trace(*_):
                if busy['val']: return
                busy['val'] = True
                v = var.get()
                entry_var.set(_fmt(v))
                try: scale.set(v)
                except Exception: pass
                busy['val'] = False
            if hasattr(var, 'trace_add'):
                var.trace_add('write', _var_trace)
            else:
                var.trace('w', _var_trace)
        except Exception: pass
        return _wrap

    def make_range_slider(self, parent, var_lo, var_hi, from_, to,
                          resolution=0.01, fmt="{:.2f}", integer=False):
        """Single-row dual-handle range slider: an editable entry for each
        bound flanking a draggable two-handle track — replaces the old
        pattern of two stacked make_slider() rows (one for min, one for
        max) with one compact widget. lo <= hi is enforced by construction
        (dragging/typing one handle past the other clamps it there)."""
        bg      = getattr(self, '_slider_bg',     "#111111")
        fg      = getattr(self, '_slider_fg',     "#999999")
        acc     = getattr(self, '_slider_accent',  "#b8960a")
        trough  = getattr(self, '_slider_trough',  "#111111")
        # Same lighter thumb colour make_slider()'s tk.Scale shows via
        # activebackground on hover/drag — applied here to whichever handle
        # is currently hovered or being dragged (see redraw()/on_motion()).
        thumbh  = getattr(self, '_slider_thumbhover', "#f0c040")

        _entry_bg   = getattr(self, '_entry_bg',    "#1a1a1a")
        _border_col = getattr(self, '_slider_border', "#785605FF")
        _wrap = tk.Frame(parent, bg=_border_col, padx=1, pady=1)
        fr = tk.Frame(_wrap, bg=bg)
        fr.pack(fill='both', expand=True)
        row = tk.Frame(fr, bg=bg)
        row.pack(fill='x')

        step = resolution if resolution else 0.01

        def snap(v):
            if integer:
                return int(round(float(v)))
            nsteps = round((float(v) - from_) / step)
            return from_ + nsteps * step

        def _fmt(v):
            try:    return fmt.format(v)
            except: return str(v)

        lo_entry_var = tk.StringVar(value=_fmt(var_lo.get()))
        hi_entry_var = tk.StringVar(value=_fmt(var_hi.get()))

        lo_entry = tk.Entry(row, textvariable=lo_entry_var, width=5, justify='right',
                            bg=_entry_bg, fg=acc, insertbackground=acc,
                            relief='flat', highlightthickness=1,
                            highlightbackground=_border_col, highlightcolor=acc,
                            font=("Helvetica", 8), bd=2)
        hi_entry = tk.Entry(row, textvariable=hi_entry_var, width=5, justify='left',
                            bg=_entry_bg, fg=acc, insertbackground=acc,
                            relief='flat', highlightthickness=1,
                            highlightbackground=_border_col, highlightcolor=acc,
                            font=("Helvetica", 8), bd=2)
        R = 6  # handle radius, px
        # Explicit (small) initial width — tk.Canvas defaults to 200px,
        # which would otherwise inflate every card's natural/requested
        # width; fill='x'+expand=True below still lets it grow to fill
        # whatever space the card actually ends up with.
        # No cursor override — default, matching make_slider()'s tk.Scale
        # (which also leaves the cursor at its Tk default).
        cv = tk.Canvas(row, height=20, width=60, bg=bg, highlightthickness=0, bd=0)

        # Both flanking entries must be packed (from their own side) BEFORE
        # the expand=True canvas — pack carves cavity in registration
        # order, so an expand widget registered before a fixed-width one
        # greedily claims the whole remaining cavity, leaving the other
        # widget 0px wide.
        # ipady brings the entry's rendered height up to match the track
        # canvas (height=20 above) — without it the entry (natural height
        # ~17px from font/border alone) looks noticeably shorter than the
        # slider box next to it, same fix applied to make_slider()'s single
        # entry (there it's a non-issue since tk.Scale's own trough is
        # thinner than the entry, so the entry already dominates the row).
        lo_entry.pack(side='left', ipady=2)
        hi_entry.pack(side='right', ipady=2)
        cv.pack(side='left', fill='x', expand=True, padx=4)

        busy = {'val': False}
        drag = {'handle': None}
        hover = {'handle': None}

        def value_to_x(v, w):
            usable = max(1, w - 2 * R)
            frac = 0.0 if to == from_ else (v - from_) / (to - from_)
            frac = min(1.0, max(0.0, frac))
            return R + frac * usable

        def x_to_value(x, w):
            usable = max(1, w - 2 * R)
            frac = (x - R) / usable
            frac = min(1.0, max(0.0, frac))
            return from_ + frac * (to - from_)

        def redraw():
            w = cv.winfo_width()
            if w < 10:
                return
            h = int(cv.cget('height'))
            ty = h // 2
            cv.delete('all')
            lo_v = float(var_lo.get())
            hi_v = float(var_hi.get())
            x_lo = value_to_x(lo_v, w)
            x_hi = value_to_x(hi_v, w)
            # Selected-range line is a bit lower-opacity than the solid
            # handle pills (simulated by blending toward the background,
            # since canvas colours have no real alpha channel).
            line_acc = self._blend_hex(acc, bg, 0.35)
            handle_border = 'white' if self._theme == 'light' else 'black'
            cv.create_line(R, ty, w - R, ty, fill=trough, width=4, capstyle='round')
            cv.create_line(x_lo, ty, x_hi, ty, fill=line_acc, width=4, capstyle='round')
            active_handle = drag['handle'] or hover['handle']
            for name, x in (('lo', x_lo), ('hi', x_hi)):
                fill = thumbh if name == active_handle else acc
                cv.create_rectangle(x - R, ty - R, x + R, ty + R,
                                    fill=fill, outline=handle_border, width=2)

        def commit(lo_v=None, hi_v=None):
            if busy['val']: return
            busy['val'] = True
            if lo_v is not None:
                lo_v = max(from_, min(to, snap(lo_v)))
                lo_v = min(lo_v, float(var_hi.get()))
                var_lo.set(lo_v)
                lo_entry_var.set(_fmt(lo_v))
            if hi_v is not None:
                hi_v = max(from_, min(to, snap(hi_v)))
                hi_v = max(hi_v, float(var_lo.get()))
                var_hi.set(hi_v)
                hi_entry_var.set(_fmt(hi_v))
            redraw()
            busy['val'] = False

        def on_press(e):
            w = cv.winfo_width()
            x_lo = value_to_x(float(var_lo.get()), w)
            x_hi = value_to_x(float(var_hi.get()), w)
            drag['handle'] = 'lo' if abs(e.x - x_lo) <= abs(e.x - x_hi) else 'hi'
            on_drag(e)

        def on_drag(e):
            if drag['handle'] is None:
                return
            v = x_to_value(e.x, cv.winfo_width())
            if drag['handle'] == 'lo':
                commit(lo_v=v)
            else:
                commit(hi_v=v)

        def on_release(_e):
            drag['handle'] = None
            redraw()

        def on_motion(e):
            if drag['handle'] is not None:
                return  # dragging already keeps a handle highlighted
            w = cv.winfo_width()
            x_lo = value_to_x(float(var_lo.get()), w)
            x_hi = value_to_x(float(var_hi.get()), w)
            d_lo, d_hi = abs(e.x - x_lo), abs(e.x - x_hi)
            nearest = 'lo' if d_lo <= d_hi else 'hi'
            new_hover = nearest if min(d_lo, d_hi) <= R + 2 else None
            if new_hover != hover['handle']:
                hover['handle'] = new_hover
                redraw()

        def on_leave(_e):
            if hover['handle'] is not None:
                hover['handle'] = None
                redraw()

        cv.bind('<Button-1>', on_press)
        cv.bind('<B1-Motion>', on_drag)
        cv.bind('<ButtonRelease-1>', on_release)
        cv.bind('<Motion>', on_motion)
        cv.bind('<Leave>', on_leave)
        cv.bind('<Configure>', lambda e: redraw())

        def _commit_lo_entry(*_):
            if busy['val']: return
            try:
                commit(lo_v=float(lo_entry_var.get().strip()))
            except (ValueError, tk.TclError):
                pass

        def _commit_hi_entry(*_):
            if busy['val']: return
            try:
                commit(hi_v=float(hi_entry_var.get().strip()))
            except (ValueError, tk.TclError):
                pass

        lo_entry.bind('<Return>',   _commit_lo_entry)
        lo_entry.bind('<FocusOut>', _commit_lo_entry)
        hi_entry.bind('<Return>',   _commit_hi_entry)
        hi_entry.bind('<FocusOut>', _commit_hi_entry)

        def _on_var_change(*_):
            if busy['val']: return
            busy['val'] = True
            lo_entry_var.set(_fmt(float(var_lo.get())))
            hi_entry_var.set(_fmt(float(var_hi.get())))
            busy['val'] = False
            redraw()

        for _v in (var_lo, var_hi):
            if hasattr(_v, 'trace_add'):
                _v.trace_add('write', _on_var_change)
            else:
                _v.trace('w', _on_var_change)

        cv.after_idle(redraw)
        return _wrap

    def make_dual_slider(self, parent, segs, var, var_min, var_max, from_, to,
                         resolution=0.01, fmt="{:.2f}", integer=False):
        """A single card row that shows a classic make_slider() (single
        value, ``var``) in single-cube mode, or a make_range_slider()
        (``var_min``/``var_max``) in Large-Dataset Mode — swapped in place
        whenever the mode toggles (see self._dual_sliders, refreshed by
        _apply_large_dataset_mode). Every parameter in Central Galaxy /
        Satellite / Diffuse Features uses this so the same card works in
        both modes."""
        bg     = getattr(self, '_slider_bg',     "#111111")
        sym_fg = getattr(self, '_slider_sym_fg', "#ffffff")
        row = tk.Frame(parent, bg=bg)
        row.pack(fill='x', pady=(0, 8))
        rich_label(row, segs, bg=bg, fg=sym_fg).pack(side='left', padx=(0, 4))

        single_wrap = self.make_slider(row, "", var, from_, to,
                                       resolution=resolution, fmt=fmt, integer=integer)
        range_wrap = self.make_range_slider(row, var_min, var_max, from_, to,
                                            resolution=resolution, fmt=fmt, integer=integer)

        def _apply_mode():
            if self._large_dataset_mode:
                single_wrap.pack_forget()
                range_wrap.pack(side='left', fill='x', expand=True)
            else:
                range_wrap.pack_forget()
                single_wrap.pack(side='left', fill='x', expand=True)

        # self._dual_sliders is reset once per _build_widgets() call (see
        # its top) — NOT lazily here, since a hasattr guard would leave
        # stale closures from a previous build (pointing at now-destroyed
        # widgets) mixed in after _rebuild_widgets() (theme toggle).
        self._dual_sliders.append(_apply_mode)
        _apply_mode()
        return row

    # ---------------------------
    # Button callback methods
    # ---------------------------
    def show_logs(self):
        # Shares _show_log_window with generate() — which also deiconifies,
        # needed because the log window is withdraw()n at startup, so a bare
        # lift() left it invisible.
        self._show_log_window()



    def _popup_figure(self, title, fig):
        """Utility to put a matplotlib figure into a new popup window"""
        new_win = tk.Toplevel(self)
        new_win.title(title)
        
        # Use the FigureCanvasTkAgg to embed the plot
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=new_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _view_results(self):
        """(clean_results, noisy_results) to display in Slice/Analysis.
        ``clean_results`` is always self.generator.results (never mutated,
        so Save can still build a correct clean/noisy pair from it);
        ``noisy_results`` is the preview computed in _run_generate when
        "Use Noise?" is Yes, or None otherwise. Both are handed to the
        viewer so it can offer a Clean/Noisy toggle instead of gui.py
        deciding once which one to show."""
        clean = getattr(self.generator, 'results', None) if self.generator else None
        noisy = getattr(self, '_preview_results', None)
        return clean, (noisy or None)

    def _raise_existing(self, attr):
        """If a previously-opened window tracked under ``attr`` is still
        alive AND still showing the current generation's results, lift it
        to the front and return True. If it's alive but stale (a newer
        generate/load happened since it was opened — tracked via
        ``_results_generation``), destroy it instead so the caller opens a
        fresh one with current data rather than silently re-raising old
        results. Returns False when the caller should open a fresh
        window (nothing was open, or the stale one was just destroyed)."""
        win = getattr(self, attr, None)
        if win is None or not win.winfo_exists():
            return False
        if getattr(win, '_songs_results_generation', None) != self._results_generation:
            win.destroy()
            setattr(self, attr, None)
            return False
        win.deiconify()
        win.lift()
        win.focus_force()
        return True

    def show_analysis(self):
        """Open the combined Analysis viewer (moments + spectrum + source
        checkboxes) — or, if one is already open and still current, bring
        it to front instead of stacking a second window."""
        if self._raise_existing('_analysis_win'):
            return
        clean, noisy = self._view_results()
        if not clean:
            return
        try:
            self._analysis_win = AnalysisViewer(self, clean, noisy_data=noisy, idx=0)
            self._analysis_win._songs_results_generation = self._results_generation
        except Exception as e:
            messagebox.showerror('Analysis viewer error', str(e))


    def show_slice(self):
        """Open the SONGS SliceViewer for the first generated cube — or, if
        one is already open and still current, bring it to front instead of stacking a
        second window."""
        if self._raise_existing('_slice_win'):
            return
        clean, noisy = self._view_results()
        if clean:
            try:
                self._slice_win = SliceViewer(self, clean, noisy_data=noisy, idx=0)
                self._slice_win._songs_results_generation = self._results_generation
            except Exception as e:
                messagebox.showerror('Slice viewer error', str(e))

    def reset_instance(self):
        """Reset the GUI to a fresh state and disable visualisation/save.

        This clears the in-memory ``self.generator`` reference so that the
        next generate action will create a new instance from current UI
        values. Buttons that depend on generated results are disabled. If
        we were viewing a cube loaded via ``load_cube()``, this also
        re-enables every parameter card that load dimmed, restores every
        slider/toggle to the app's hardcoded defaults (undoing
        ``_apply_loaded_manifest``), and restores Generate to its normal
        clickable state. If a Large-Dataset Mode batch run is currently in
        progress, this also requests it stop (same as clicking Stop) — the
        background worker unwinds cooperatively at its next safe point.
        """
        if getattr(self, '_ld_generating', False):
            self.stop_generation()

        # Disable all result buttons
        for _b in (self.analysis_btn, self.slice_btn,
                   self.save_btn, self.new_instance_btn):
            try:
                self._disable_btn(_b)
            except Exception:
                pass
        for child in self.winfo_children():
            if isinstance(child, tk.Toplevel):
                child.destroy()

        if getattr(self, '_loaded_view_active', False):
            for root_name in ('_bc1_outer', '_bc2_outer', '_bc3_outer', '_bc4_outer'):
                root = getattr(self, root_name, None)
                if root is not None:
                    self._lock_walk(root, False)
            self._loaded_view_active = False
            default_state = getattr(self, '_default_var_state', None)
            if default_state is not None:
                self._restore_var_state(default_state)
            if not self._large_dataset_mode:
                _gen_fg = "#000000" if self._theme == 'dark' else "#ffffff"
                self._enable_btn(self.generate_btn, fg=_gen_fg)

        self.generator = None
        self._has_results = False
        self._preview_results = None
        self._enable_theme_btn()

        # Re-assert the satellite card's own greyout: the broad restore above
        # can un-dim it even when n_gals is still 1.
        _sat_update = getattr(self, '_update_sat_dependent', None)
        if _sat_update is not None:
            try: _sat_update()
            except Exception: pass

        if self._large_dataset_mode and not getattr(self, '_ld_generating', False):
            self._set_ld_progress(0, int(self.n_samples_var.get()))

    def _find_scale_in(self, widget):
        """Recursively find a ttk.Scale inside a widget tree.

        Returns the first found Scale or None.
        """
        if isinstance(widget, ttk.Scale):
            return widget
        for c in widget.winfo_children():
            found = self._find_scale_in(c)
            if found is not None:
                return found
        return None

    def _set_sliders_enabled(self, enabled=True):
        """Enable or disable all slider widgets present in the GUI.

        This toggles the internal ttk.Scale widget state for each slider
        frame we create in :meth:`_build_widgets`.
        """
        names = [
            'r_slider', 'n_slider', 'hz_slider', 'sigma_slider',
            'grid_slider', 'spec_slider', 'angle_x_slider', 'angle_y_slider',
            'sat_offset_slider_frame'
        ]
        for name in names:
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                scale = self._find_scale_in(w)
                if scale is None:
                    continue
                if enabled:
                    try:
                        scale.state(['!disabled'])
                    except Exception:
                        scale.configure(state=tk.NORMAL)
                else:
                    try:
                        scale.state(['disabled'])
                    except Exception:
                        scale.configure(state=tk.DISABLED)
            except Exception:
                # Best-effort: ignore any widget-specific errors
                pass
        

   

    # ---------------------------
    # Build all widgets
    # ---------------------------
    def _build_widgets(self):

        """Build and layout all GUI widgets.

        This method assembles the complete UI inside the scrollable
        container: it defines Tk variables, creates the three-column
        parameter panels (rows 1--6), the slider widgets, and the bottom
        utility buttons (Generate, Moment0, Moment1, Spectra, Save, New).

        The method also hooks variable traces to an auto-update helper so
        that changing parameters in the UI will keep an internal
        ``SONGS`` generator in sync for quick inspection.

        Notes
        -----
        - This method focuses on layout and widget creation; no heavy
            computation is performed here.
        - For clarity we keep layout logic (pack) local to this helper so
            other methods can assume the widgets exist after this call.
        """
        
        # Reset once per build — make_dual_slider() appends to this, and a
        # rebuild (theme toggle) must not accumulate stale closures that
        # point at widgets from the previous build (now destroyed).
        self._dual_sliders = []

        # ---------------------------
        # Variables
        # ---------------------------
        self.bmin_var = tk.DoubleVar(value=2.5)
        self.bmaj_var = tk.DoubleVar(value=2.9)
        self.bpa_var = tk.DoubleVar(value=20.0)
        # Large-Dataset Mode only: beam expressed in pixels instead of kpc,
        # so every cube gets the same beam FOOTPRINT (in px) despite each
        # cube's own sampled Spatial Resolution — see _sample_cube_params,
        # which converts back to kpc per-cube (beam_kpc = beam_px * that
        # cube's spatial_resolution). Single-cube mode is untouched and
        # always uses bmin_var/bmaj_var (kpc) above.
        self.bmin_px_var = tk.DoubleVar(value=4.0)
        self.bmaj_px_var = tk.DoubleVar(value=4.7)
        self.spatial_resolution = tk.DoubleVar(value=0.62)
        self.spatial_resolution_min_var = tk.DoubleVar(value=0.625)  # Large-Dataset Mode
        self.spatial_resolution_max_var = tk.DoubleVar(value=2.5)    # Large-Dataset Mode
        self.n_var = tk.DoubleVar(value=1.0)
        self.hz_var = tk.DoubleVar(value=0.8)
        self.hz_min_var = tk.DoubleVar(value=0.64)            # Large-Dataset Mode (-20%)
        self.hz_max_var = tk.DoubleVar(value=0.96)            # Large-Dataset Mode (+20%)
        self.Se_var = tk.DoubleVar(value=0.1)
        self.Se_min_var = tk.DoubleVar(value=0.08)            # Large-Dataset Mode (-20%)
        self.Se_max_var = tk.DoubleVar(value=0.12)            # Large-Dataset Mode (+20%)
        self.Re_var = tk.DoubleVar(value=5.0)   # kpc — same default in both modes
        self.sigma_v_var = tk.DoubleVar(value=40.0)
        self.sigma_v_min_var = tk.DoubleVar(value=32.0)       # Large-Dataset Mode (-20%)
        self.sigma_v_max_var = tk.DoubleVar(value=48.0)       # Large-Dataset Mode (+20%)
        self.fov = tk.DoubleVar(value=60.0)   # in kpc
        # Large-Dataset Mode only: FOV min/max, kept dynamically in sync
        # with Spatial Resolution's min/max and the fixed pixel grid size
        # (fov = grid_size * spatial_resolution, applied per endpoint) —
        # see the sync traces set up alongside the "Spatial Resolution" and
        # "Field of View" cards below.
        self.fov_min_var = tk.DoubleVar(value=60.0)
        self.fov_max_var = tk.DoubleVar(value=240.0)
        self.spectral_resolution = tk.IntVar(value=20)
        self.angle_x_var = tk.IntVar(value=45)
        self.angle_y_var = tk.IntVar(value=30)
        self.n_gals_var = tk.IntVar(value=4)
        self.n_samples_var = tk.IntVar(value=1000)          # Large-Dataset Mode
        self.spatial_pixel_dim_var = tk.IntVar(value=96)    # Large-Dataset Mode
        self.max_gals_per_cube_var = tk.IntVar(value=3)      # Large-Dataset Mode
        self.angle_x_min_var = tk.IntVar(value=0)            # Large-Dataset Mode
        self.angle_x_max_var = tk.IntVar(value=359)          # Large-Dataset Mode
        self.angle_y_min_var = tk.IntVar(value=0)            # Large-Dataset Mode
        self.angle_y_max_var = tk.IntVar(value=359)          # Large-Dataset Mode
        self.n_min_var = tk.DoubleVar(value=0.5)              # Large-Dataset Mode
        self.n_max_var = tk.DoubleVar(value=1.5)              # Large-Dataset Mode
        self.use_noise_var = tk.StringVar(value='Yes')        # Noise Parameters
        self.use_noise_ld_var = tk.StringVar(value='Yes')      # Large-Dataset Mode noise card
        self.sn_peak_var = tk.DoubleVar(value=45.0)             # Noise Parameters
        self.sn_peak_min_var = tk.DoubleVar(value=3.0)         # Large-Dataset Mode
        self.sn_peak_max_var = tk.DoubleVar(value=100.0)       # Large-Dataset Mode
        self.save_folder_var = tk.StringVar(value='')          # Large-Dataset Mode

        # --- Diffuse-emission knobs (defaults pulled from core's DEFAULT_DIFFUSE_PARAMS) ---
        # Each also gets a _min_var/_max_var pair (Large-Dataset Mode dual
        # slider), defaulting to a zero-width range at the single-cube value.
        dp = DEFAULT_DIFFUSE_PARAMS
        def _dv(key, default):
            v = float(dp.get(key, default))
            return tk.DoubleVar(value=v), tk.DoubleVar(value=v * 0.8), tk.DoubleVar(value=v * 1.2)
        # Halo
        self.halo_Se_factor_var, self.halo_Se_factor_min_var, self.halo_Se_factor_max_var = _dv('halo_Se_factor', 0.03)
        self.halo_Re_factor_var, self.halo_Re_factor_min_var, self.halo_Re_factor_max_var = _dv('halo_Re_factor', 2.0)
        self.halo_sigma_vz_var, self.halo_sigma_vz_min_var, self.halo_sigma_vz_max_var = _dv('halo_sigma_vz', 70.0)
        # Bridges
        self.bridge_Se_factor_var, self.bridge_Se_factor_min_var, self.bridge_Se_factor_max_var = _dv('bridge_Se_factor', 0.05)
        self.bridge_width_start_factor_var, self.bridge_width_start_factor_min_var, self.bridge_width_start_factor_max_var = _dv('bridge_width_start_factor', 1.0)
        self.bridge_width_end_factor_var, self.bridge_width_end_factor_min_var, self.bridge_width_end_factor_max_var = _dv('bridge_width_end_factor', 0.8)
        self.bridge_sigma_vz_var, self.bridge_sigma_vz_min_var, self.bridge_sigma_vz_max_var = _dv('bridge_sigma_vz', 70.0)
        # Tails / streamers
        self.tail_Se_factor_var, self.tail_Se_factor_min_var, self.tail_Se_factor_max_var = _dv('tail_Se_factor', 0.4)
        self.tail_length_var, self.tail_length_min_var, self.tail_length_max_var = _dv('tail_length_factor', 6.0)
        self.tail_width_factor_var, self.tail_width_factor_min_var, self.tail_width_factor_max_var = _dv('tail_width_factor', 1.2)
        self.tail_sigma_vz_var, self.tail_sigma_vz_min_var, self.tail_sigma_vz_max_var = _dv('tail_sigma_vz', 80.0)
        # Streamer (channel-traversing trajectory) extras
        self.tail_vel_gradient_var, self.tail_vel_gradient_min_var, self.tail_vel_gradient_max_var = _dv('tail_vel_gradient', 80.0)

        # Large-Dataset Mode defaults for a few diffuse knobs that deviate
        # from the standard +-20% derived above.
        self.halo_Se_factor_min_var.set(0.035)
        self.halo_Se_factor_max_var.set(0.085)
        self.halo_Re_factor_min_var.set(1.3)
        self.halo_Re_factor_max_var.set(1.7)
        self.bridge_Se_factor_min_var.set(0.032)
        self.bridge_Se_factor_max_var.set(0.055)
        self.bridge_width_end_factor_min_var.set(0.6)
        self.bridge_width_end_factor_max_var.set(1.0)
        self.tail_Se_factor_min_var.set(0.21)
        self.tail_Se_factor_max_var.set(0.31)


        # New: satellite size fraction (max satellite-to-central ratio for Re,
        # hz, Se). Greyed out when only one galaxy is requested.
        self.sat_brightness_frac_var = tk.DoubleVar(value=0.36)
        self.sat_brightness_frac_min_var = tk.DoubleVar(value=0.37)   # Large-Dataset Mode
        self.sat_brightness_frac_max_var = tk.DoubleVar(value=0.50)   # Large-Dataset Mode
        self.sat_Re_frac_var = tk.DoubleVar(value=0.4)  # avg satellite Re as fraction of central Re
        self.sat_Re_frac_min_var = tk.DoubleVar(value=0.32)          # Large-Dataset Mode (-20%)
        self.sat_Re_frac_max_var = tk.DoubleVar(value=0.48)          # Large-Dataset Mode (+20%)

        # ── Colour scheme — pulled from current theme ──────────────────────────
        _t          = _THEMES[self._theme]
        _BG         = _t['BG']
        _CARD_BG    = _t['CARD_BG']
        _BIG_BG     = _t['BIG_BG']
        _BIG_BORDER = _t['BIG_BORDER']
        _SM_BORDER  = _t['SM_BORDER']
        _TEXT       = _t['TEXT']
        _ACCENT     = _t['ACCENT']
        _ENTRY_BG   = _t['ENTRY_BG']
        # Small-card headings: proper white in dark mode (matching the
        # slider-symbol colour) rather than the dimmer grey TEXT colour;
        # light mode is unaffected.
        _TITLE_FG   = _t['SYM_FG'] if self._theme == 'dark' else _TEXT
        _FONT_SM    = ("Helvetica", 10)
        _FONT_HDR   = ("Helvetica", 9, "bold")

        # Expose colors to make_slider via instance attrs
        self._slider_bg         = _CARD_BG
        self._slider_fg         = _TEXT
        self._slider_sym_fg     = _t['SYM_FG']
        self._slider_accent     = _ACCENT
        self._slider_trough     = _t['SLIDER_TROUGH']
        self._slider_thumb      = _ACCENT
        self._slider_thumbhover = _t['SLIDER_THUMBHOV']
        self._slider_border     = _t['SLIDER_BORDER']
        self._entry_bg          = _ENTRY_BG

        # Make ttk slider trough match the small card background
        _style = ttk.Style()
        _style.configure('Horizontal.TScale', troughcolor=_CARD_BG, background=_CARD_BG)
        _style.configure('TScale', troughcolor=_CARD_BG, background=_CARD_BG)

        col_width = 200  # small card fixed width

        # Helper used multiple times below to find the underlying ttk.Scale
        # inside a slider frame (so we can grey it out when n_gals == 1).
        def find_scale(widget):
            if isinstance(widget, (tk.Scale, ttk.Scale)):
                return widget
            for child in widget.winfo_children():
                result = find_scale(child)
                if result is not None:
                    return result
            return None

        def big_card(parent, title, stack=False, expand=False):
            """Bordered card with thin, low-opacity yellow outline and title."""
            outer = tk.Frame(parent, bg=_BIG_BORDER, padx=1, pady=1)
            if stack:
                outer.pack(fill='both', expand=expand, padx=6, pady=6)
            else:
                # expand=False: a sibling column's pack_forget() (Large-
                # Dataset Mode toggle) must never let this stretch into the
                # freed space — that stretch-then-snap-back is what caused
                # the disable-time flicker.
                outer.pack(side='left', fill='both', expand=False, padx=6, pady=6)
            inner = tk.Frame(outer, bg=_BIG_BG)
            inner.pack(fill='both', expand=True)
            inner._card_outer = outer  # lets callers match/measure the full card (incl. border)
            _hdr = tk.Label(inner, text=title, bg=_BIG_BG, fg=_ACCENT,
                            font=_FONT_HDR)
            _hdr._orig_fg = _ACCENT
            _hdr.pack(anchor='w', padx=8, pady=(6,2))
            sep = tk.Frame(inner, bg=_BIG_BORDER, height=1)
            sep.pack(fill='x', padx=6, pady=(0,6))
            return inner

        def small_card(parent, title=None):
            outer = tk.Frame(parent, bg=_SM_BORDER, padx=1, pady=1)
            outer.pack(fill='x', padx=6, pady=3)
            inner = tk.Frame(outer, bg=_CARD_BG, padx=5, pady=4)
            inner.pack(fill='both', expand=True)
            inner._card_outer = outer  # lets callers grey out the whole card (incl. border)
            if title:
                _lbl = tk.Label(inner, text=title, bg=_CARD_BG, fg=_TITLE_FG,
                                font=_FONT_SM)
                _lbl._orig_fg = _TITLE_FG
                _lbl._is_card_heading = True  # dimmed more aggressively than the rest of the card
                _lbl.pack(anchor='w', pady=(2, 5))
            return inner

        def slider_with_symbol(parent, segs, var, from_, to,
                               resolution=0.01, fmt="{:.2f}", integer=False,
                               symbol_side='left'):
            """Return a row frame containing a rich-text symbol label + slider."""
            row = tk.Frame(parent, bg=_CARD_BG)
            row.pack(fill='x', pady=(0, 8))
            sym = rich_label(row, segs, bg=_CARD_BG, fg=_t['SYM_FG'])
            sl  = self.make_slider(row, "", var, from_, to,
                                   resolution=resolution, fmt=fmt, integer=integer)
            if symbol_side == 'left':
                sym.pack(side='left', padx=(0, 4))
                sl.pack(side='left', fill='x', expand=True)
            else:
                sl.pack(side='left', fill='x', expand=True)
                sym.pack(side='left', padx=(4, 0))
            return row

        # ── Horizontal big-card row ─────────────────────────────────────────
        cards_row = tk.Frame(self.container, bg=_BG)
        cards_row.pack(fill='both', expand=True)

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 1: Initialisation Parameters
        # ──────────────────────────────────────────────────────────────────────
        bc1 = big_card(cards_row, "Initialisation Parameters")
        self._bc1_outer = bc1._card_outer  # used to match the Large-Dataset Mode column width

        # Pill-button constants + factory — reused for "Number of galaxies"
        # (below) and "Max number of galaxies per data cube" (Large-
        # Dataset Mode column).
        _PB_W, _PB_H = 28, 22
        _PB_SEL_BG   = _ACCENT
        _PB_SEL_FG   = "#000000" if self._theme == 'dark' else "#ffffff"
        _PB_NOR_BG   = _t['PILL_NOR']
        _PB_NOR_FG   = _TEXT
        _PB_HOV_BG   = _t['PILL_HOV']

        def _make_pill_selector(parent, var, values, width=None):
            """Build a row of pill buttons bound to ``var``; clicking a pill
            sets ``var`` to its value. Pill width defaults to ``_PB_W`` but
            grows to fit longer labels (e.g. 'Yes'/'No'). Returns the list
            of pill canvases."""
            from tkinter import font as _tkfont
            _pill_canvases = []
            _pb_font = ("Helvetica", 9, "bold")
            if width is None:
                _measure_font = _tkfont.Font(family="Helvetica", size=9, weight="bold")
                width = max([_PB_W] + [_measure_font.measure(str(v)) + 16 for v in values])

            def _draw_pill(cv, selected, hover=False):
                cv.delete("all")
                fill = _PB_SEL_BG if selected else (_PB_HOV_BG if hover else _PB_NOR_BG)
                cv.create_rectangle(0, 0, width, _PB_H, fill=fill, outline=fill)
                fg = _PB_SEL_FG if selected else _PB_NOR_FG
                cv.create_text(width//2, _PB_H//2, text=cv._val_str,
                               fill=fg, font=_pb_font)

            def _make_pill(val):
                cv = tk.Canvas(parent, width=width, height=_PB_H,
                               bg=_CARD_BG, highlightthickness=0, bd=0, cursor='pointinghand')
                cv._val = val
                cv._val_str = str(val)
                _pill_canvases.append(cv)

                def _select():
                    var.set(val)

                def _on_enter(e):
                    if var.get() != val:
                        _draw_pill(cv, False, hover=True)

                def _on_leave(e):
                    _draw_pill(cv, var.get() == val)

                cv.bind("<ButtonRelease-1>", lambda e: _select())
                cv.bind("<Enter>", _on_enter)
                cv.bind("<Leave>", _on_leave)
                return cv

            def _refresh_pills(*_):
                sel = var.get()
                for cv in _pill_canvases:
                    _draw_pill(cv, cv._val == sel)

            for val in values:
                cv = _make_pill(val)
                cv.pack(side='left', padx=2)

            _refresh_pills()
            var.trace_add('write', _refresh_pills)
            return _pill_canvases

        # ── Large-Dataset-Mode-only controls — top of the card, hidden in
        # single-cube mode. Order preserved via before=self._card_spatial_res
        # each time they're re-shown (see _apply_large_dataset_mode).
        self._bc1_ld_only = []

        sc = small_card(bc1, "Number of samples")
        self._bc1_ld_only.append(sc._card_outer)
        self.n_samples_slider = self.make_slider(sc, "", self.n_samples_var,
                                                  2, 40000, resolution=1,
                                                  fmt="{:d}", integer=True)
        self.n_samples_slider.pack(fill='x', expand=True)

        sc = small_card(bc1, "Max number of galaxies per data cube")
        self._bc1_ld_only.append(sc._card_outer)
        _mg_frame = tk.Frame(sc, bg=_CARD_BG)
        _mg_frame.pack(anchor='w', pady=(2, 4))
        _make_pill_selector(_mg_frame, self.max_gals_per_cube_var, range(1, 7))

        sc = small_card(bc1, "Spatial pixel grid size")
        self._bc1_ld_only.append(sc._card_outer)
        _spd_row = tk.Frame(sc, bg=_CARD_BG)
        _spd_row.pack(fill='x', pady=(0, 2))
        rich_label(_spd_row, [("N","n"),("px","s")], bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
        self.spatial_pixel_dim_slider = self.make_slider(_spd_row, "", self.spatial_pixel_dim_var,
                                                          32, 256, resolution=1, fmt="{:d}", integer=True)
        self.spatial_pixel_dim_slider.pack(side='left', fill='x', expand=True)

        # FOV's min/max (see the "Field of View" card below) stay two-way in
        # sync with Spatial Resolution's min/max — per endpoint — and the
        # fixed pixel grid size above, so grid_size = fov/spatial_resolution
        # always holds at both ends of the range, whichever slider (or its
        # min/max entry) the user actually drags. Single-cube mode is
        # unaffected — this only fires in Large-Dataset Mode, where both
        # cards show their range (dual-slider) form.
        _fov_range_sync_busy = {'val': False}

        def _sync_fov_range_from_grid_or_res(*_):
            if _fov_range_sync_busy['val'] or not self._large_dataset_mode:
                return
            _fov_range_sync_busy['val'] = True
            try:
                grid = float(self.spatial_pixel_dim_var.get())
                self.fov_min_var.set(grid * float(self.spatial_resolution_min_var.get()))
                self.fov_max_var.set(grid * float(self.spatial_resolution_max_var.get()))
            finally:
                _fov_range_sync_busy['val'] = False

        def _sync_res_range_from_fov(*_):
            if _fov_range_sync_busy['val'] or not self._large_dataset_mode:
                return
            grid = float(self.spatial_pixel_dim_var.get())
            if grid <= 0:
                return
            _fov_range_sync_busy['val'] = True
            try:
                self.spatial_resolution_min_var.set(float(self.fov_min_var.get()) / grid)
                self.spatial_resolution_max_var.set(float(self.fov_max_var.get()) / grid)
            finally:
                _fov_range_sync_busy['val'] = False

        self.spatial_pixel_dim_var.trace_add('write', _sync_fov_range_from_grid_or_res)
        self.spatial_resolution_min_var.trace_add('write', _sync_fov_range_from_grid_or_res)
        self.spatial_resolution_max_var.trace_add('write', _sync_fov_range_from_grid_or_res)
        self.fov_min_var.trace_add('write', _sync_res_range_from_fov)
        self.fov_max_var.trace_add('write', _sync_res_range_from_fov)
        self._sync_dynamic_fov = _sync_fov_range_from_grid_or_res

        # ── Single-cube-mode-only controls ──────────────────────────────────
        self._bc1_single_only = []

        sc = small_card(bc1, "Number of galaxies")
        self._card_ngals = sc._card_outer
        self._bc1_single_only.append(self._card_ngals)
        rb_frame = tk.Frame(sc, bg=_CARD_BG)
        rb_frame.pack(anchor='w', pady=(2, 4))
        _make_pill_selector(rb_frame, self.n_gals_var, range(1, 7))

        sc = small_card(bc1, "Spatial Resolution [kpc/px]")
        self._card_spatial_res = sc._card_outer  # always visible — anchors LD-only cards above it
        self.spatial_res_dual = self.make_dual_slider(sc, [("Δ","n"),("X,Y","s")], self.spatial_resolution,
                                                       self.spatial_resolution_min_var, self.spatial_resolution_max_var,
                                                       0.5, 9.0, resolution=0.01, fmt="{:.2f}")

        sc = small_card(bc1, "Spectral Resolution [km/s]")
        slider_with_symbol(sc, [("Δ","n"),("v","s"),("z","ss")], self.spectral_resolution, 5, 40, resolution=5, fmt="{:d}", integer=True)
        self.spec_slider = sc.winfo_children()[-1]

        sc = small_card(bc1, "Field of View [kpc]")
        self._card_fov = sc._card_outer  # always visible — kept dynamic with Spatial Resolution in LD mode
        self.fov_dual = self.make_dual_slider(sc, [("N","n"),("kpc","s")], self.fov,
                                              self.fov_min_var, self.fov_max_var, 60.0, 512.0,
                                              resolution=1.0, fmt="{:.0f}")

        sc = small_card(bc1, "Beam [kpc, kpc, deg]")
        self._card_beam = sc._card_outer  # always visible — anchors the Field of View card above it
        self._beam_hdr_lbl = sc.winfo_children()[0]  # title label — text swaps units with mode

        # Convolved / Raw toggle — same pill aesthetic & colours as
        # "Number of galaxies" (accent = active, hover highlight, pointing hand).
        from tkinter import font as _tkfont
        self.beam_mode_var = tk.StringVar(value='Convolved')
        _bm_row = tk.Frame(sc, bg=_CARD_BG)
        _bm_row.pack(anchor='w', pady=(0, 8))

        _bm_font = _tkfont.Font(family="Helvetica", size=9, weight="bold")
        _bm_w = max(_bm_font.measure(v) for v in ("Convolved", "Raw")) + 16
        _bm_pill_canvases = []

        def _draw_bm_pill(cv, selected, hover=False):
            cv.delete("all")
            fill = _PB_SEL_BG if selected else (_PB_HOV_BG if hover else _PB_NOR_BG)
            cv.create_rectangle(0, 0, _bm_w, _PB_H, fill=fill, outline=fill)
            fg = _PB_SEL_FG if selected else _PB_NOR_FG
            cv.create_text(_bm_w // 2, _PB_H // 2, text=cv._val,
                           fill=fg, font=_bm_font)

        def _make_bm_pill(parent, val):
            cv = tk.Canvas(parent, width=_bm_w, height=_PB_H,
                           bg=_CARD_BG, highlightthickness=0, bd=0, cursor='pointinghand')
            cv._val = val
            _bm_pill_canvases.append(cv)

            def _select():
                self.beam_mode_var.set(val)

            def _on_enter(e):
                if self.beam_mode_var.get() != val:
                    _draw_bm_pill(cv, False, hover=True)

            def _on_leave(e):
                _draw_bm_pill(cv, self.beam_mode_var.get() == val)

            cv.bind("<ButtonRelease-1>", lambda e: _select())
            cv.bind("<Enter>", _on_enter)
            cv.bind("<Leave>", _on_leave)
            return cv

        def _refresh_bm_pills(*_):
            sel = self.beam_mode_var.get()
            for cv in _bm_pill_canvases:
                _draw_bm_pill(cv, cv._val == sel)

        for val in ("Convolved", "Raw"):
            cv = _make_bm_pill(_bm_row, val)
            cv.pack(side='left', padx=2)

        _refresh_bm_pills()

        # Beam parameter rows — symbols, sliders, their border "dividers", and
        # value entry boxes all live in this container so a single recursive
        # pass can dim/disable them when "Raw" is selected.
        _beam_content = tk.Frame(sc, bg=_CARD_BG)
        _beam_content.pack(fill='x')

        # bmin/bmaj: kpc sliders in single-cube mode (untouched), pixel
        # sliders in Large-Dataset Mode — every generated cube then keeps
        # the same beam FOOTPRINT IN PIXELS despite its own sampled Spatial
        # Resolution (converted back to kpc per-cube in _sample_cube_params).
        # BPA is unitless (an angle) so it's shared, unaffected by mode.
        _beam_kpc_frame = tk.Frame(_beam_content, bg=_CARD_BG)
        _beam_px_frame = tk.Frame(_beam_content, bg=_CARD_BG)

        for sym_segs, var, lo, hi, res in [
            ([("B","n"),("min","s")], self.bmin_var, 1.0, 30.0, 0.1),
            ([("B","n"),("maj","s")], self.bmaj_var, 1.0, 30.0, 0.1),
        ]:
            beam_row = tk.Frame(_beam_kpc_frame, bg=_CARD_BG)
            beam_row.pack(fill='x', pady=(0, 8))
            rich_label(beam_row, sym_segs, bg=_CARD_BG, fg=_t['SYM_FG']).pack(
                side='left', padx=(0, 4))
            sl = self.make_slider(beam_row, "", var, lo, hi, resolution=res, fmt="{:.1f}")
            sl.pack(side='left', fill='x', expand=True)

        for sym_segs, var, lo, hi, res in [
            ([("B","n"),("min","s")], self.bmin_px_var, 0.5, 20.0, 0.1),
            ([("B","n"),("maj","s")], self.bmaj_px_var, 0.5, 20.0, 0.1),
        ]:
            beam_row = tk.Frame(_beam_px_frame, bg=_CARD_BG)
            beam_row.pack(fill='x', pady=(0, 8))
            rich_label(beam_row, sym_segs, bg=_CARD_BG, fg=_t['SYM_FG']).pack(
                side='left', padx=(0, 4))
            sl = self.make_slider(beam_row, "", var, lo, hi, resolution=res, fmt="{:.1f}")
            sl.pack(side='left', fill='x', expand=True)

        _bpa_row = tk.Frame(_beam_content, bg=_CARD_BG)
        _bpa_row.pack(fill='x', pady=(0, 8))
        rich_label(_bpa_row, [("BPA","n")], bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
        _bpa_slider = self.make_slider(_bpa_row, "", self.bpa_var, 0.0, 90.0, resolution=1.0, fmt="{:.1f}")
        _bpa_slider.pack(side='left', fill='x', expand=True)

        def _apply_beam_unit_mode():
            if self._large_dataset_mode:
                _beam_kpc_frame.pack_forget()
                _beam_px_frame.pack(fill='x', before=_bpa_row)
                self._beam_hdr_lbl.configure(text="Beam [px, px, deg]")
            else:
                _beam_px_frame.pack_forget()
                _beam_kpc_frame.pack(fill='x', before=_bpa_row)
                self._beam_hdr_lbl.configure(text="Beam [kpc, kpc, deg]")
        self._dual_sliders.append(_apply_beam_unit_mode)
        _apply_beam_unit_mode()

        def _update_beam_dependent(*_args):
            active = self.beam_mode_var.get() == 'Convolved'
            self._dim_walk(_beam_content, not active)

        _update_beam_dependent()
        self.beam_mode_var.trace_add('write', _update_beam_dependent)

        # allow_overlap_var lives on the generator/preview from here on, but the
        # toggle widget itself sits in the Satellite Properties card (bc3),
        # right above the min/max offset controls it governs.
        self.allow_overlap_var = tk.StringVar(value='Allow overlap')

        # ── Use Noise? — right under the Beam card (single-cube mode only;
        # the double-slider min/max version in the Large-Dataset Mode
        # column replaces it in that mode) ──────────────────────────────────
        sc = small_card(bc1)
        self._card_noise = sc._card_outer
        self._bc1_single_only.append(self._card_noise)
        _noise_row = tk.Frame(sc, bg=_CARD_BG)
        _noise_row.pack(fill='x', pady=(2, 4))
        self._noise_hdr_lbl = tk.Label(_noise_row, text="Use Noise?", bg=_CARD_BG, fg=_TITLE_FG,
                                       font=_FONT_SM)
        self._noise_hdr_lbl._is_card_heading = True
        self._noise_hdr_lbl.pack(side='left')
        _noise_pill_frame = tk.Frame(_noise_row, bg=_CARD_BG)
        _noise_pill_frame.pack(side='left', padx=(10, 0))
        _make_pill_selector(_noise_pill_frame, self.use_noise_var, ('Yes', 'No'))

        self._noise_sn_row = tk.Frame(sc, bg=_CARD_BG)
        self._noise_sn_row.pack(fill='x', pady=(4, 2))
        rich_label(self._noise_sn_row, [("S/N","n"),("peak","s")],
                  bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
        self.sn_peak_slider = self.make_slider(self._noise_sn_row, "", self.sn_peak_var,
                                               3, 50, resolution=0.5, fmt="{:.1f}")
        self.sn_peak_slider.pack(side='left', fill='x', expand=True)

        def _update_noise_dependent(*_):
            active = (self.use_noise_var.get() == 'Yes')
            self._dim_walk(self._noise_sn_row, not active)
        _update_noise_dependent()
        self.use_noise_var.trace_add('write', _update_noise_dependent)
        self._update_noise_dependent = _update_noise_dependent

        # ── Use Noise? (Large-Dataset Mode double-slider version) + Choose
        # Save Folder — same spot as the classic Use Noise? card above,
        # shown only in Large-Dataset Mode ────────────────────────────────
        self._bc1_ld_only_tail = []

        sc = small_card(bc1)
        self._bc1_ld_only_tail.append(sc._card_outer)
        _noise_ld_row = tk.Frame(sc, bg=_CARD_BG)
        _noise_ld_row.pack(fill='x', pady=(2, 4))
        self._noise_ld_hdr_lbl = tk.Label(_noise_ld_row, text="Use Noise?", bg=_CARD_BG, fg=_TITLE_FG,
                                          font=_FONT_SM)
        self._noise_ld_hdr_lbl._is_card_heading = True
        self._noise_ld_hdr_lbl.pack(side='left')
        _noise_ld_pill_frame = tk.Frame(_noise_ld_row, bg=_CARD_BG)
        _noise_ld_pill_frame.pack(side='left', padx=(10, 0))
        _make_pill_selector(_noise_ld_pill_frame, self.use_noise_ld_var, ('Yes', 'No'))

        self._noise_ld_sn_row = tk.Frame(sc, bg=_CARD_BG)
        self._noise_ld_sn_row.pack(fill='x', pady=(4, 2))
        rich_label(self._noise_ld_sn_row, [("S/N","n"),("peak","s")],
                  bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
        self.sn_peak_range_slider = self.make_range_slider(
            self._noise_ld_sn_row, self.sn_peak_min_var, self.sn_peak_max_var,
            3, 100, resolution=0.5, fmt="{:.1f}")
        self.sn_peak_range_slider.pack(side='left', fill='x', expand=True)

        def _update_noise_ld_dependent(*_):
            active = (self.use_noise_ld_var.get() == 'Yes')
            self._dim_walk(self._noise_ld_sn_row, not active)
        _update_noise_ld_dependent()
        self.use_noise_ld_var.trace_add('write', _update_noise_ld_dependent)
        self._update_noise_ld_dependent = _update_noise_ld_dependent

        # ---- Choose Save Folder — its own card, with a restored caption
        # (path shown in a monospace/code font) below the button ----------
        sc = small_card(bc1)
        self._bc1_ld_only_tail.append(sc._card_outer)
        _save_gen_fg  = "#000000" if self._theme == 'dark' else "#ffffff"
        _save_gen_hov = "#e8c040" if self._theme == 'dark' else "#7a5800"
        self._save_folder_btn = tk.Label(
            sc, text='Choose Save Folder', bg=_ACCENT, fg=_save_gen_fg,
            font=('Helvetica', 9, 'bold'), padx=8, pady=8, cursor='pointinghand',
        )
        self._save_folder_btn.pack(fill='x', pady=(0, 4))
        self._save_folder_btn.bind('<Enter>', lambda _: self._save_folder_btn.configure(bg=_save_gen_hov))
        self._save_folder_btn.bind('<Leave>', lambda _: self._save_folder_btn.configure(bg=_ACCENT))

        self._save_folder_note = tk.Label(
            sc, text=("Final Dataset pickle file will be saved here, and raw "
                     "hdf5 files will be saved in this path's /raw folder:"),
            bg=_CARD_BG, fg=_TEXT, font=('Helvetica', 8),
            justify='left', anchor='w', wraplength=200,
        )
        self._save_folder_note.pack(anchor='w', fill='x', pady=(0, 2))

        self._save_folder_path_lbl = tk.Label(
            sc, text=(self.save_folder_var.get() or 'chosen_path'),
            bg=_CARD_BG, fg=_ACCENT, font=('Courier', 8, 'bold'),
            justify='left', anchor='w', wraplength=200,
        )
        self._save_folder_path_lbl.pack(anchor='w', fill='x', pady=(0, 2))

        def _choose_save_folder(*_):
            path = filedialog.askdirectory(title='Choose Save Folder')
            if path:
                self.save_folder_var.set(path)
                self._save_folder_path_lbl.configure(text=path)

        self._save_folder_btn.bind('<ButtonRelease-1>', _choose_save_folder)

        # ── FOV preview canvas — forced square, same width as card ───────────
        import math as _math
        _PREV_BG = '#252520' if self._theme == 'dark' else '#ede8dc'

        _prev_cv = tk.Canvas(bc1, bg=_PREV_BG,
                             highlightthickness=2, highlightbackground=_ACCENT,
                             width=10, height=10)
        _prev_cv.pack(fill='x', padx=6, pady=(4, 6))
        self._prev_cv = _prev_cv  # single-cube mode only
        self._bc1_single_only.append(_prev_cv)

        # Track last width so Configure from height-change doesn't redraw twice
        _prev_last_w = [-1]

        def _draw_fov_preview(*_):
            S = _prev_cv.winfo_width()
            H = _prev_cv.winfo_height()
            if S < 20 or H < 20:
                return
            _prev_cv.delete('all')

            _dim = getattr(self, '_large_dataset_mode', False)
            _prev_acc = self._blend_hex(_ACCENT, _PREV_BG, 0.6) if _dim else _ACCENT
            _prev_gal = self._blend_hex(_t['SYM_FG'], _PREV_BG, 0.6) if _dim else _t['SYM_FG']

            fov_kpc  = max(float(self.fov.get()), 1.0)
            res_kpc  = max(float(self.spatial_resolution.get()), 0.01)
            bmin_kpc = max(float(self.bmin_var.get()), 0.01)
            bmaj_kpc = max(float(self.bmaj_var.get()), 0.01)
            bpa_deg  = float(self.bpa_var.get())

            fov_px   = int(fov_kpc / res_kpc)         # grid pixels — matches SONGSPhy (truncation)
            scale    = S / fov_px                    # canvas px per grid px
            bmin_px  = bmin_kpc / res_kpc            # beam minor in grid px
            bmaj_px  = bmaj_kpc / res_kpc            # beam major in grid px

            # ── beam ellipse + crosshairs (bottom-left, mirrors add_beam) ────
            marg   = max(int(S * 0.09), 10)
            semi_a = max(bmin_px * scale / 2, 2.0)  # minor semi-axis in canvas px
            semi_b = max(bmaj_px * scale / 2, 2.0)  # major semi-axis in canvas px
            ecx    = marg + semi_b + 2
            ecy    = S - marg - semi_b + 10          # slightly lower than default

            # Ellipse polygon (180 pts, no spline → crisp)
            theta_rot = _math.radians(-bpa_deg)   # Tkinter y-down convention
            n_pts = 180
            pts = []
            for k in range(n_pts):
                ang = 2 * _math.pi * k / n_pts
                ex  = semi_b * _math.cos(ang)
                ey  = semi_a * _math.sin(ang)
                rx  = ex * _math.cos(theta_rot) - ey * _math.sin(theta_rot)
                ry  = ex * _math.sin(theta_rot) + ey * _math.cos(theta_rot)
                pts.extend([ecx + rx, ecy + ry])
            _prev_cv.create_polygon(pts, outline=_prev_acc, fill=_PREV_BG,
                                    width=1, smooth=False)

            # Crosshairs: major dir = [cos(θ), sin(θ)], minor = [-sin(θ), cos(θ)]
            dx_maj =  semi_b * _math.cos(theta_rot)
            dy_maj =  semi_b * _math.sin(theta_rot)
            dx_min = -semi_a * _math.sin(theta_rot)
            dy_min =  semi_a * _math.cos(theta_rot)
            for dx, dy in [(dx_maj, dy_maj), (dx_min, dy_min)]:
                _prev_cv.create_line(ecx - dx, ecy - dy, ecx + dx, ecy + dy,
                                     fill=_prev_acc, width=1)

            # Beam pixel label
            _fnt = ('Helvetica', 9)
            _prev_cv.create_text(ecx + semi_b + 5, ecy,
                                 text=f"{bmin_px:.1f}×{bmaj_px:.1f} px",
                                 fill=_prev_acc, font=_fnt, anchor='w')

            # ── scalebar (top-right) ─────────────────────────────────────────
            bar_px_grid = fov_px / 2.0
            bar_kpc_val = fov_kpc / 2
            bar_cv_px   = S / 2

            bx1 = S - marg
            bx0 = bx1 - bar_cv_px
            by  = marg + 8

            for bx in (bx0, bx1):
                _prev_cv.create_line(bx, by - 4, bx, by + 4,
                                     fill=_prev_acc, width=1.5)
            _prev_cv.create_line(bx0, by, bx1, by, fill=_prev_acc, width=1.5)

            _prev_cv.create_text((bx0 + bx1) / 2, by - 10,
                                 text=f"{bar_px_grid:.0f} px",
                                 fill=_prev_acc, font=_fnt)
            _prev_cv.create_text((bx0 + bx1) / 2, by + 10,
                                 text=f"{bar_kpc_val:.0f} kpc",
                                 fill=_prev_acc, font=_fnt)

            # ── galaxy markers ───────────────────────────────────────────────
            _gal_col = _prev_gal
            _lbl_fnt = ('Helvetica', 7, 'bold')
            if self._galaxy_centers is not None and len(self._galaxy_centers) > 0:
                for _gi, _gc in enumerate(self._galaxy_centers):
                    _gx = (_gc[1] / fov_px) * S
                    _gy = S - (_gc[0] / fov_px) * S
                    if _gi == 0:
                        # Central: hollow circle then filled dot on top (both centred on _gx, _gy)
                        _cr, _dr = 7, 2
                        _prev_cv.create_oval(_gx - _cr, _gy - _cr, _gx + _cr, _gy + _cr,
                                             fill='', outline=_gal_col, width=1.5, tags='gal')
                        _prev_cv.create_oval(_gx - _dr, _gy - _dr, _gx + _dr, _gy + _dr,
                                             fill=_gal_col, outline='', tags='gal')
                        _prev_cv.create_text(_gx + _cr + 4, _gy,
                                             text='C', fill=_gal_col, font=_lbl_fnt,
                                             anchor='w', tags='gal')
                    else:
                        # Satellite: hollow circle
                        _sr = 4
                        _prev_cv.create_oval(_gx - _sr, _gy - _sr, _gx + _sr, _gy + _sr,
                                             fill='', outline=_gal_col, width=1.2, tags='gal')
                        _prev_cv.create_text(_gx + _sr + 4, _gy,
                                             text=f'S{_gi}', fill=_gal_col, font=_lbl_fnt,
                                             anchor='w', tags='gal')
            else:
                _prev_cv.create_text(S / 2, S / 2,
                                     text="Spatial preview of\nthe Spectral Cube\n(in pixel units)",
                                     fill=_prev_acc, font=('Helvetica', 8),
                                     justify='center')

        def _on_prev_configure(e):
            w = e.width
            if w == _prev_last_w[0]:   # height-only change — skip
                return
            _prev_last_w[0] = w
            if w > 20:
                _prev_cv.configure(height=w)   # force square directly on canvas
            _prev_cv.after_idle(_draw_fov_preview)

        # Keep a reference so _sample_positions can trigger a redraw
        self._draw_fov_preview_ref = _draw_fov_preview

        _prev_cv.bind('<Configure>', _on_prev_configure)
        for _pv in (self.fov, self.spatial_resolution,
                    self.bmin_var, self.bmaj_var, self.bpa_var):
            _pv.trace_add('write', _draw_fov_preview)

        # Snapshot of every value _resample_and_redraw cares about, so a
        # spurious no-op trigger (see below) can be told apart from a real
        # change.
        _resample_snapshot = {'val': None}

        def _resample_and_redraw(*_):
            """Re-draw positions when n_gals, FOV/resolution, satellite
            offset range, or overlap mode changes.

            tk.Scale can spuriously re-invoke its command with the SAME
            value it already had, as a side effect of some unrelated
            widget elsewhere in the window being reconfigured/remapped
            (observed: happening while a background generation is
            running) — and since Tk fires 'write' traces even for a
            no-op .set(), that would otherwise silently re-randomise the
            galaxy positions with no actual input from the user. Guard by
            only resampling when the tracked values actually changed.
            """
            current = (
                int(self.n_gals_var.get()),
                float(self.fov.get()),
                float(self.spatial_resolution.get()),
                float(self.sat_offset_min_var.get()),
                float(self.sat_offset_max_var.get()),
                self.allow_overlap_var.get(),
            )
            if current == _resample_snapshot['val']:
                return
            _resample_snapshot['val'] = current
            self._sample_positions(redraw=True)

        self.n_gals_var.trace_add('write', _resample_and_redraw)
        self._resample_and_redraw = _resample_and_redraw  # wired after sat vars exist
        # Re-sample whenever anything that affects grid_size changes
        self.fov.trace_add('write', _resample_and_redraw)
        self.spatial_resolution.trace_add('write', _resample_and_redraw)

        # Re-randomise button — small overlay, bottom-right corner of the preview
        _regen_hov = _t['REGEN_HOV']
        _regen_cmd = lambda: self._sample_positions(redraw=True)
        _regen_btn = tk.Label(
            _prev_cv, text='Re-randomise',
            bg=_PREV_BG, fg=_ACCENT,
            font=('Helvetica', 8, 'bold'), padx=6, pady=3, cursor='pointinghand',
            highlightthickness=1, highlightbackground=_ACCENT,
        )
        _regen_btn.place(relx=1.0, rely=1.0, anchor='se', x=-6, y=-6)
        _regen_btn.bind('<Enter>', lambda _: _regen_btn.configure(bg=_regen_hov))
        _regen_btn.bind('<Leave>', lambda _: _regen_btn.configure(bg=_PREV_BG))
        _regen_btn.bind('<ButtonRelease-1>', lambda _: _regen_cmd())
        self._regen_btn = _regen_btn  # greyed out + unclickable in Large-Dataset Mode

        _prev_cv.after(150, lambda: (self._sample_positions(redraw=False),
                                     _draw_fov_preview()))

        # ──────────────────────────────────────────────────────────────────────
        # MIDDLE COLUMN: Central Galaxy (top) + Satellite Properties (below)
        # ──────────────────────────────────────────────────────────────────────
        mid_col = tk.Frame(cards_row, bg=_BG)
        mid_col.pack(side='left', fill='both', expand=False)
        self._mid_col = mid_col  # used to match the fourth column's width

        bc2 = big_card(mid_col, "Central Galaxy Properties", stack=True)
        self._bc2_outer = bc2._card_outer  # dimmed while viewing a loaded cube

        # Every card below uses make_dual_slider(): a single make_slider()
        # in single-cube mode, or a make_range_slider() (min/max) in
        # Large-Dataset Mode, swapped in place by _apply_large_dataset_mode.
        sc = small_card(bc2, "Sérsic index")
        self.n_dual = self.make_dual_slider(sc, [("n","n")], self.n_var, self.n_min_var, self.n_max_var,
                                            0.5, 1.5, resolution=0.01, fmt="{:.3f}")

        sc = small_card(bc2, "Effective radius and scale height [kpc, kpc]")
        # Re is fixed (single slider, same in both modes) — see the "which
        # to vary" discussion: physical galaxy size stays representative of
        # a real population while Spatial Resolution/FOV vary per-cube
        # instead, so the same objects are sampled at different resolutions.
        slider_with_symbol(sc, [("R","n"),("e","s")], self.Re_var, 1.0, 60.0, resolution=0.5, fmt="{:.1f}")
        self.hz_dual = self.make_dual_slider(sc, [("h","n"),("z","s")], self.hz_var,
                                             self.hz_min_var, self.hz_max_var, 0.4, 9.0, resolution=0.01, fmt="{:.3f}")

        sc = small_card(bc2, "Surface brightness [Jy]")
        self.Se_dual = self.make_dual_slider(sc, [("S","n"),("e","s")], self.Se_var,
                                             self.Se_min_var, self.Se_max_var, 0.01, 1.0, resolution=0.01, fmt="{:.3f}")

        sc = small_card(bc2, "Velocity dispersion [km/s]")
        self.sigma_v_dual = self.make_dual_slider(sc, [("σ","n"),("v","s"),("z","ss")], self.sigma_v_var,
                                                  self.sigma_v_min_var, self.sigma_v_max_var, 30.0, 60.0,
                                                  resolution=0.1, fmt="{:.1f}")

        sc = small_card(bc2, "X and Y Inclination Angle Ranges [deg, deg]")
        self.angle_x_dual = self.make_dual_slider(sc, [("θ","n"),("X","s")], self.angle_x_var,
                                                   self.angle_x_min_var, self.angle_x_max_var, 0, 359,
                                                   resolution=1, fmt="{:d}", integer=True)
        self.angle_y_dual = self.make_dual_slider(sc, [("φ","n"),("Y","s")], self.angle_y_var,
                                                   self.angle_y_min_var, self.angle_y_max_var, 0, 359,
                                                   resolution=1, fmt="{:d}", integer=True)

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 3: Satellite & Halo  (stacked under Central Galaxy)
        # ──────────────────────────────────────────────────────────────────────
        bc3 = big_card(mid_col, "Satellite Properties", stack=True, expand=True)
        self._bc3 = bc3  # kept for greyout
        self._bc3_outer = bc3._card_outer  # dimmed while viewing a loaded cube

        sc = small_card(bc3, "Satellite flux & Re fraction")
        self.sat_frac_dual = self.make_dual_slider(sc, [("f","n"),("sat","s")], self.sat_brightness_frac_var,
                                                    self.sat_brightness_frac_min_var, self.sat_brightness_frac_max_var,
                                                    0.0, 0.5, resolution=0.01, fmt="{:.2f}")
        self.sat_re_frac_dual = self.make_dual_slider(sc, [("R","n"),("e,sat","s")], self.sat_Re_frac_var,
                                                       self.sat_Re_frac_min_var, self.sat_Re_frac_max_var,
                                                       0.1, 0.9, resolution=0.05, fmt="{:.2f}")

        self.sat_offset_max_var = tk.DoubleVar(value=110.0)
        self.sat_offset_min_var = tk.DoubleVar(value=75.0)
        self.sat_offset_min_var.trace_add('write', self._resample_and_redraw)
        self.sat_offset_max_var.trace_add('write', self._resample_and_redraw)
        self.allow_overlap_var.trace_add('write', self._resample_and_redraw)
        # Large-Dataset Mode only: satellite offset expressed in pixels
        # instead of kpc — same reasoning as the beam, but for offset the
        # opposite physical choice applies (see the "same Re, but not the
        # same object" discussion): every cube keeps the same PIXEL offset
        # range, so physical separation is allowed to scale with each
        # cube's own sampled Spatial Resolution rather than staying fixed.
        # Single-cube mode is untouched — always uses the kpc vars above.
        self.sat_offset_min_px_var = tk.DoubleVar(value=29.0)
        self.sat_offset_max_px_var = tk.DoubleVar(value=46.0)

        # ── Allow overlap / Distinct positions toggle — same pill aesthetic
        # as "Number of galaxies" and "Convolved/Raw", placed directly above
        # the min/max offset controls it governs.
        sc = small_card(bc3, "Satellite Positions")
        _ov_row = tk.Frame(sc, bg=_CARD_BG)
        _ov_row.pack(anchor='w', pady=(0, 8))

        _ov_font = _tkfont.Font(family="Helvetica", size=9, weight="bold")
        _ov_w = max(_ov_font.measure(v) for v in ("Distinct", "Allow overlap")) + 16
        _ov_pill_canvases = []

        def _draw_ov_pill(cv, selected, hover=False):
            cv.delete("all")
            fill = _PB_SEL_BG if selected else (_PB_HOV_BG if hover else _PB_NOR_BG)
            cv.create_rectangle(0, 0, _ov_w, _PB_H, fill=fill, outline=fill)
            fg = _PB_SEL_FG if selected else _PB_NOR_FG
            cv.create_text(_ov_w // 2, _PB_H // 2, text=cv._val,
                           fill=fg, font=_ov_font)

        def _make_ov_pill(parent, val):
            cv = tk.Canvas(parent, width=_ov_w, height=_PB_H,
                           bg=_CARD_BG, highlightthickness=0, bd=0, cursor='hand2')
            cv._val = val
            _ov_pill_canvases.append(cv)

            def _select():
                self.allow_overlap_var.set(val)

            def _on_enter(e):
                if self.allow_overlap_var.get() != val:
                    _draw_ov_pill(cv, False, hover=True)

            def _on_leave(e):
                _draw_ov_pill(cv, self.allow_overlap_var.get() == val)

            cv.bind("<ButtonRelease-1>", lambda e: _select())
            cv.bind("<Enter>", _on_enter)
            cv.bind("<Leave>", _on_leave)
            return cv

        for val in ("Distinct", "Allow overlap"):
            cv = _make_ov_pill(_ov_row, val)
            cv.pack(side='left', padx=2)

        def _refresh_ov_pills(*_):
            sel = self.allow_overlap_var.get()
            for cv in _ov_pill_canvases:
                _draw_ov_pill(cv, cv._val == sel)

        _refresh_ov_pills()
        self.allow_overlap_var.trace_add('write', _refresh_ov_pills)

        sc = small_card(bc3, "Min & Max Offset from Central [kpc, kpc]")
        self._offset_hdr_lbl = sc.winfo_children()[0]  # title label — text swaps units with mode

        _offset_kpc_frame = tk.Frame(sc, bg=_CARD_BG)
        _offset_px_frame = tk.Frame(sc, bg=_CARD_BG)

        for _segs, _var, _f_attr, _s_attr, _lo, _hi in [
            ([("d","n"),("min","s")], self.sat_offset_min_var,
             'sat_offset_min_frame', 'sat_offset_min_scale', 0.0,  333.0),
            ([("d","n"),("max","s")], self.sat_offset_max_var,
             'sat_offset_max_frame', 'sat_offset_max_scale', 10.0, 500.0),
        ]:
            _orow = tk.Frame(_offset_kpc_frame, bg=_CARD_BG)
            _orow.pack(fill='x', pady=(0, 8))
            rich_label(_orow, _segs, bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
            _sl = self.make_slider(_orow, "", _var, _lo, _hi, resolution=5.0, fmt="{:.0f}")
            _sl.pack(side='left', fill='x', expand=True)
            setattr(self, _f_attr, _orow)
            setattr(self, _s_attr, find_scale(_orow))

        for _segs, _var, _lo, _hi in [
            ([("d","n"),("min","s")], self.sat_offset_min_px_var, 0.0, 100.0),
            ([("d","n"),("max","s")], self.sat_offset_max_px_var, 2.0, 150.0),
        ]:
            _orow = tk.Frame(_offset_px_frame, bg=_CARD_BG)
            _orow.pack(fill='x', pady=(0, 8))
            rich_label(_orow, _segs, bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
            _sl = self.make_slider(_orow, "", _var, _lo, _hi, resolution=1.0, fmt="{:.0f}")
            _sl.pack(side='left', fill='x', expand=True)

        def _apply_offset_unit_mode():
            if self._large_dataset_mode:
                _offset_kpc_frame.pack_forget()
                _offset_px_frame.pack(fill='x')
                self._offset_hdr_lbl.configure(text="Min & Max Offset from Central [px, px]")
            else:
                _offset_px_frame.pack_forget()
                _offset_kpc_frame.pack(fill='x')
                self._offset_hdr_lbl.configure(text="Min & Max Offset from Central [kpc, kpc]")
        self._dual_sliders.append(_apply_offset_unit_mode)
        _apply_offset_unit_mode()

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 4: Diffuse Features (halo + bridge + streamers)
        # ──────────────────────────────────────────────────────────────────────
        bc4 = big_card(cards_row, "Diffuse Features")
        self._bc4_outer = bc4._card_outer  # dimmed while viewing a loaded cube

        sc = small_card(bc4, "Diffuse halo flux (fraction of central)")
        self.halo_Se_dual = self.make_dual_slider(sc, [("S","n"),("e,halo","s"),(" / S","n"),("e,c","s")],
                                                   self.halo_Se_factor_var, self.halo_Se_factor_min_var,
                                                   self.halo_Se_factor_max_var, 0.0, 0.3, resolution=0.005, fmt="{:.3f}")

        sc = small_card(bc4, "Diffuse halo effective radius (fraction of central)")
        self.halo_Re_dual = self.make_dual_slider(sc, [("R","n"),("e,halo","s"),(" / R","n"),("e,c","s")],
                                                   self.halo_Re_factor_var, self.halo_Re_factor_min_var,
                                                   self.halo_Re_factor_max_var, 1.0, 5.0, resolution=0.1, fmt="{:.1f}")

        sc = small_card(bc4, "Bridge flux amplitude")
        self.bridge_Se_dual = self.make_dual_slider(sc, [("S","n"),("e,br","s"),(" / S","n"),("e,s","s")],
                                                     self.bridge_Se_factor_var, self.bridge_Se_factor_min_var,
                                                     self.bridge_Se_factor_max_var, 0.0, 0.3, resolution=0.005, fmt="{:.3f}")

        sc = small_card(bc4, "Bridge width fractions (central & satellite ends)")
        self.bridge_w0_dual = self.make_dual_slider(sc, [("σ","n"),("br,h","s"),(" / R","n"),("e,c","s")],
                                                     self.bridge_width_start_factor_var, self.bridge_width_start_factor_min_var,
                                                     self.bridge_width_start_factor_max_var, 0.5, 4.0, resolution=0.1, fmt="{:.1f}")
        self.bridge_w1_dual = self.make_dual_slider(sc, [("σ","n"),("br,s","s"),(" / R","n"),("e,s","s")],
                                                     self.bridge_width_end_factor_var, self.bridge_width_end_factor_min_var,
                                                     self.bridge_width_end_factor_max_var, 0.3, 3.0, resolution=0.1, fmt="{:.1f}")

        sc = small_card(bc4, "Tidal tail flux fraction")
        self.tail_Se_dual = self.make_dual_slider(sc, [("S","n"),("e,tail","s"),(" / S","n"),("e,s","c|s")],
                                                   self.tail_Se_factor_var, self.tail_Se_factor_min_var,
                                                   self.tail_Se_factor_max_var, 0.0, 0.9, resolution=0.02, fmt="{:.2f}")

        sc = small_card(bc4, "Tidal tail velocity scale [km/s]")
        self.tail_vel_grad_dual = self.make_dual_slider(sc, [("Δv","n"),("tail","s")],
                                                         self.tail_vel_gradient_var, self.tail_vel_gradient_min_var,
                                                         self.tail_vel_gradient_max_var, 0.0, 300.0, resolution=5.0, fmt="{:.0f}")

        sc = small_card(bc4, "Tidal tail length (× Re of the tail's own galaxy)")
        self.tail_length_dual = self.make_dual_slider(sc, [("L","n"),("tail","s")],
                                                       self.tail_length_var, self.tail_length_min_var,
                                                       self.tail_length_max_var, 1.0, 15.0, resolution=0.5, fmt="{:.1f}")

        sc = small_card(bc4, "Tidal tail width (× Re)")
        self.tail_width_dual = self.make_dual_slider(sc, [("σ","n"),("tail","s"),(" / R","n"),("e","s")],
                                                      self.tail_width_factor_var, self.tail_width_factor_min_var,
                                                      self.tail_width_factor_max_var, 0.1, 5.0, resolution=0.1, fmt="{:.1f}")

        sc = small_card(bc4, "Diffuse velocity dispersion [km/s]")
        for _segs, _var, _var_min, _var_max in [
            ([("σ","n"),("v","s"),("z","ss"),(", halo","s")], self.halo_sigma_vz_var, self.halo_sigma_vz_min_var, self.halo_sigma_vz_max_var),
            ([("σ","n"),("v","s"),("z","ss"),(", br","s")],   self.bridge_sigma_vz_var, self.bridge_sigma_vz_min_var, self.bridge_sigma_vz_max_var),
            ([("σ","n"),("v","s"),("z","ss"),(", tail","s")], self.tail_sigma_vz_var, self.tail_sigma_vz_min_var, self.tail_sigma_vz_max_var),
        ]:
            self.make_dual_slider(sc, _segs, _var, _var_min, _var_max, 0.0, 200.0, resolution=5.0, fmt="{:.0f}")

        self._cards_row = cards_row

        # ──────────────────────────────────────────────────────────────────────
        # ── Satellite-dependent greyout ──────────────────────────────────────
        def _reconfigure_scale_to(scale, to):
            # tk.Scale can spuriously re-invoke its 'command' as a side
            # effect of a to=/from_= reconfigure (it recomputes the thumb's
            # value from its current pixel position against the NEW range,
            # which can be a wrong/different value than what's actually
            # stored) — same quirk _lock_walk works around for state=. The
            # spurious re-fire can happen on a DEFERRED redraw, not
            # synchronously within configure(to=...) itself, so command
            # must stay cleared through an explicit update_idletasks()
            # before being restored — clearing and immediately restoring
            # in the same call isn't enough to avoid it.
            try:
                orig_cmd = scale.cget('command')
                scale.configure(command='')
                scale.configure(to=to)
                scale.update_idletasks()
            finally:
                try: scale.configure(command=orig_cmd)
                except Exception: pass

        def _update_min_range(*args):
            new_upper = max(5.0, float(self.sat_offset_max_var.get()) / 1.5)
            _reconfigure_scale_to(self.sat_offset_min_scale, new_upper)
            if float(self.sat_offset_min_var.get()) > new_upper:
                self.sat_offset_min_var.set(round(new_upper / 5) * 5)
        self.sat_offset_max_var.trace_add('write', _update_min_range)
        _update_min_range()

        def _compute_max_offset_kpc():
            # Mirrors the relaxed edge margin in core.place_galaxies (half of
            # init_grid_size // 8, not the full footprint) so the slider cap
            # matches what placement will actually allow.
            res_kpc = max(float(self.spatial_resolution.get()), 0.01)
            fov_px  = max(int(self.fov.get()), 4)
            gs      = max(fov_px // res_kpc, 4)
            igs     = int((31 / 64) * gs)
            if igs % 2 != 0:
                igs -= 1
            half    = max(2, igs // 8)
            center  = (gs + 1) // 2
            max_off_px = max(min(gs - half - 1 - center, center - half) - 1, 1)
            return max_off_px * res_kpc

        def _update_offset_range(*_args):
            cap = _compute_max_offset_kpc()
            new_max_upper = float(cap)
            new_min_upper = max(1.0, new_max_upper / 1.5)
            _reconfigure_scale_to(self.sat_offset_max_scale, new_max_upper)
            _reconfigure_scale_to(self.sat_offset_min_scale, new_min_upper)
            if float(self.sat_offset_max_var.get()) > new_max_upper:
                self.sat_offset_max_var.set(round(new_max_upper * 0.8 / 5) * 5)
            if float(self.sat_offset_min_var.get()) > new_min_upper:
                self.sat_offset_min_var.set(round(new_min_upper * 0.5 / 5) * 5)

        _update_offset_range()
        for _v in (self.fov, self.spatial_resolution):
            if hasattr(_v, 'trace_add'):
                _v.trace_add('write', _update_offset_range)
            else:
                _v.trace('w', _update_offset_range)

        def _update_sat_dependent(*args):
            # With one galaxy there are no satellites, so the whole Satellite
            # Properties card (flux/Re fractions, Satellite Positions'
            # Distinct/Allow-overlap pills, offset range) is inapplicable.
            #
            # Use the shared _dim_walk rather than a bespoke greyout: it fades
            # every colour proportionally toward the card background (the same
            # low-opacity look as the Beam/Noise cards) instead of slamming
            # text to a flat disabled grey on the window background, and it
            # also dims canvas-drawn items — which is what fades the rich-text
            # maths symbols and the pill buttons that the old code left at
            # full brightness. It disables and unbinds interaction too, so the
            # card is genuinely unclickable while faded.
            active = self.n_gals_var.get() > 1
            self._dim_walk(self._bc3, not active)

        # Exposed so paths that restore a broad region (loaded-cube view,
        # Large-Dataset Mode) can re-assert the satellite card's own state
        # afterwards — a blanket restore would otherwise un-dim it even when
        # n_gals is still 1.
        self._update_sat_dependent = _update_sat_dependent

        _update_sat_dependent()
        if hasattr(self.n_gals_var, 'trace_add'):
            self.n_gals_var.trace_add('write', _update_sat_dependent)
        else:
            self.n_gals_var.trace('w', _update_sat_dependent)



        # ---------------------------
        # Generate & utility buttons (Generate, Slice, Moments, Spectrum, Save, New)
        # ---------------------------
        btn_frame = self._btn_area
        btn_frame.configure(padx=8, pady=6)

        # tk.Label buttons — Labels always respect bg/fg on macOS unlike tk.Button
        _btn_dis_fg  = _t['BTN_DIS_FG']
        _btn_nor_bg  = _t['BTN_NOR_BG']
        _btn_nor_fg  = _t['BTN_NOR_FG']
        _btn_nor_hov = _t['BTN_NOR_HOV']

        def _mk_btn(parent, text, cmd, bg=None, fg=None,
                    hov=None, disabled=False, font=("Helvetica", 10, "bold"),
                    square=False):
            bg  = bg  or _btn_nor_bg
            fg  = fg  or _btn_nor_fg
            hov = hov or _btn_nor_hov
            if square:
                # Fixed-size icon button (e.g. Stop) — doesn't stretch to
                # share the row's width like the text buttons. Label's
                # own width/height options are character units (not equal
                # in pixels), so an exact square needs a fixed-pixel outer
                # frame with propagation off; the label just fills it, so
                # hover/click still cover the whole square.
                _side = 35  # matches the text buttons' natural row height
                _sq = tk.Frame(parent, width=_side, height=_side, bg=bg)
                _sq.pack_propagate(False)
                _sq.pack(side='left', padx=4)
                lbl = tk.Label(_sq, text=text, bg=bg, fg=fg,
                               font=font, cursor='pointinghand')
                lbl.pack(fill='both', expand=True)
            else:
                lbl = tk.Label(parent, text=text, bg=bg, fg=fg,
                               font=font, padx=14, pady=10, cursor='pointinghand')
                lbl.pack(side='left', padx=4, expand=True, fill='x')
            if disabled:
                lbl.configure(fg=_btn_dis_fg, cursor='arrow')
            else:
                lbl.bind('<Enter>', lambda e, b=lbl, h=hov: b.configure(bg=h))
                lbl.bind('<Leave>', lambda e, b=lbl, n=bg: b.configure(bg=n))
                lbl.bind('<ButtonRelease-1>', lambda e: cmd())
            lbl._btn_bg   = bg
            lbl._btn_hov  = hov
            lbl._btn_cmd  = cmd
            lbl._disabled = disabled
            return lbl

        def _enable_btn(lbl, bg=None, hov=None, fg=None):
            bg  = bg  or lbl._btn_bg
            hov = hov or lbl._btn_hov
            fg  = fg  or _btn_nor_fg
            lbl.configure(bg=bg, fg=fg, cursor='pointinghand')
            lbl._disabled = False
            lbl.bind('<Enter>', lambda e, b=lbl, h=hov: b.configure(bg=h))
            lbl.bind('<Leave>', lambda e, b=lbl, n=bg:  b.configure(bg=n))
            lbl.bind('<ButtonRelease-1>', lambda e: lbl._btn_cmd())

        def _disable_btn(lbl):
            # bg reset to the normal (non-accent) button colour too, so an
            # accent-styled button like Generate looks exactly like the
            # other deactivated buttons (Load/Slice/Analysis/Save/Reset)
            # instead of keeping its gold background with dimmed text.
            lbl.configure(bg=_btn_nor_bg, fg=_btn_dis_fg, cursor='arrow')
            lbl._disabled = True
            lbl.unbind('<Enter>')
            lbl.unbind('<Leave>')
            lbl.unbind('<ButtonRelease-1>')

        self._enable_btn  = _enable_btn
        self._disable_btn = _disable_btn

        _gen_fg  = "#000000" if self._theme == 'dark' else "#ffffff"
        _gen_hov = "#e8c040" if self._theme == 'dark' else "#7a5800"
        self.generate_btn     = _mk_btn(btn_frame, 'Generate', self.generate,
                                        bg=_ACCENT, fg=_gen_fg, hov=_gen_hov)
        self.load_btn          = _mk_btn(btn_frame, 'Load',     self.load_cube)
        self.slice_btn        = _mk_btn(btn_frame, 'Slice',    self.show_slice,     disabled=True)
        self.analysis_btn     = _mk_btn(btn_frame, 'Analysis', self.show_analysis,  disabled=True)
        self.save_btn         = _mk_btn(btn_frame, 'Save',     self.save_sim,       disabled=True)
        self.stop_btn         = _mk_btn(btn_frame, '■', self.stop_generation, disabled=True, square=True)
        self.new_instance_btn = _mk_btn(btn_frame, 'Reset',    self.reset_instance, disabled=True)

        # Dataset-generation progress bar — takes the place of Slice /
        # Analysis / Save (hidden) in Large-Dataset Mode; those three are
        # meaningless for a batch run. Not packed here — swapped in/out by
        # _apply_large_dataset_mode()/_set_generate_dataset_enabled().
        self._ld_progress_frame = tk.Frame(btn_frame, bg=_btn_nor_bg, padx=14, pady=10)
        _ld_inner = tk.Frame(self._ld_progress_frame, bg=_btn_nor_bg)
        _ld_inner.pack(fill='both', expand=True)

        self._ld_progress_lbl = tk.Label(_ld_inner, text='Generated cube 0/0', bg=_btn_nor_bg,
                                         fg=_btn_nor_fg, font=_FONT_SM)
        self._ld_progress_lbl.pack(side='left', padx=(0, 8))

        self._ld_progress_pct_lbl = tk.Label(_ld_inner, text='0%', bg=_btn_nor_bg,
                                             fg=_btn_nor_fg, font=_FONT_SM)
        self._ld_progress_pct_lbl.pack(side='right', padx=(8, 0))

        # The blocky bar itself stretches to fill whatever's left between
        # the two labels, redrawing its blocks to match on every resize.
        self._ld_progress_nblocks = 40
        self._ld_progress_cv = tk.Canvas(
            _ld_inner, height=14, bg=_btn_nor_bg, highlightthickness=0, bd=0,
        )
        self._ld_progress_cv.pack(side='left', fill='x', expand=True)
        self._ld_progress_last_frac = 0.0
        self._ld_progress_cv.bind(
            '<Configure>', lambda e: self._set_ld_progress(
                *self._ld_progress_last_iv) if hasattr(self, '_ld_progress_last_iv') else None)

        self._set_ld_progress(0, int(self.n_samples_var.get()))

        def _on_n_samples_change(*_):
            if not getattr(self, '_ld_generating', False):
                self._set_ld_progress(0, int(self.n_samples_var.get()))
        self.n_samples_var.trace_add('write', _on_n_samples_change)

        # Auto-create/refresh generator when variables change (fast preview)
        _auto_update_vars = [
            self.bmin_var, self.bmaj_var, self.bpa_var, self.spatial_resolution, self.n_var,
            self.hz_var, self.Se_var, self.sigma_v_var, self.fov,
            self.spectral_resolution, self.angle_x_var, self.angle_y_var,
            self.sat_brightness_frac_var, self.sat_offset_min_var, self.sat_offset_max_var,
            # Diffuse-emission knobs
            self.halo_Se_factor_var, self.halo_Re_factor_var,
            self.halo_sigma_vz_var,
            self.bridge_Se_factor_var, self.bridge_width_start_factor_var,
            self.bridge_width_end_factor_var,
            self.tail_Se_factor_var,
            self.tail_vel_gradient_var,
            self.tail_length_var,
        ]
        _auto_update_snapshot = {'val': None}

        def _auto_update_generator(*args):
            # Guard against SPURIOUS re-fires, exactly like
            # _resample_and_redraw does. tk.Scale re-invokes its write trace
            # with the value it already had whenever some unrelated widget in
            # the window is reconfigured/remapped — which includes switching
            # away from the window and back. Without this guard that no-op
            # re-fire would rebuild self.generator (wiping the results of the
            # run that just finished) and grey out Slice/Analysis/Save, so the
            # buttons went dead and — because self.generator.results was now
            # empty — even clicking an enabled one did nothing at all.
            try:
                current = tuple(v.get() for v in _auto_update_vars)
            except Exception:
                current = None
            if current is not None and current == _auto_update_snapshot['val']:
                return
            _auto_update_snapshot['val'] = current

            # Never rebuild the generator out from under an in-flight run —
            # _run_generate holds a reference to it and publishes into it.
            if getattr(self, '_gen_in_progress', False) or getattr(self, '_ld_generating', False):
                return

            try:
                self.create_generator()
            except Exception as e:
                print("Auto-create generator failed:", e)
                return
            # The freshly (re)built generator has no results yet — a
            # slider change invalidates whatever was previously generated,
            # so Slice/Analysis/Save must go back to disabled until
            # Generate is run again, instead of staying enabled and
            # silently doing nothing when clicked. Loaded-cube view is
            # exempt (sliders are locked read-only there, so this
            # shouldn't legitimately fire, but guard anyway).
            if not getattr(self, '_loaded_view_active', False):
                self._has_results = False
                self._preview_results = None
                for _b in (self.analysis_btn, self.slice_btn, self.save_btn):
                    try: self._disable_btn(_b)
                    except Exception: pass

        # Prime the snapshot with the current values so the very first
        # spurious re-fire (before any real edit) is recognised as a no-op.
        try:
            _auto_update_snapshot['val'] = tuple(v.get() for v in _auto_update_vars)
        except Exception:
            pass

        for var in _auto_update_vars:
            if hasattr(var, 'trace_add'):
                var.trace_add('write', _auto_update_generator)
            else:
                var.trace('w', _auto_update_generator)

        # One-time snapshot of every slider/toggle's hardcoded default —
        # taken only on the very first build (a theme toggle re-enters
        # _build_widgets() via _rebuild_widgets(), which must not overwrite
        # this with the user's current values). reset_instance() restores
        # it after leaving a loaded-cube view, so Reset actually returns to
        # the app's default state instead of leaving the loaded cube's
        # values in place.
        if not hasattr(self, '_default_var_state'):
            self._default_var_state = self._save_var_state()

    # ---------------------------
    # Parameter collection & generator
    # ---------------------------

    
    def _collect_parameters(self):
        """Read current UI controls and return a parameter dict.

        The returned dictionary mirrors the small set of fields used by the
        :class:`SONGS` constructor and the GUI. Values are converted
        to plain Python / NumPy types where appropriate.

        Returns
        -------
        params : dict
            Dictionary containing keys like ``beam_info``, ``n_gals``,
            ``grid_size``, ``n_spectral_slices``, ``all_Re``, ``all_hz``,
            ``all_Se``, ``all_n``, and ``sigma_v``. This dict is consumed by
            :meth:`create_generator` and used when saving.
        """

        bmin = float(self.bmin_var.get())
        bmaj = float(self.bmaj_var.get())
        bpa = float(self.bpa_var.get())
        n_gals = int(self.n_gals_var.get())
        fov = int(self.fov.get())
        spectral_resolution = int(self.spectral_resolution.get())
        spatial_resolution = max(float(self.spatial_resolution.get()), 0.01)
        central_n = float(self.n_var.get())
        central_hz = float(self.hz_var.get())
        central_Se = float(self.Se_var.get())
        central_Re_kpc = float(self.Re_var.get())
        central_gal_x_angle = int(self.angle_x_var.get())
        central_gal_y_angle = int(self.angle_y_var.get())
        offset_gals = (float(self.sat_offset_min_var.get()), float(self.sat_offset_max_var.get()))
        sigma_v = float(self.sigma_v_var.get())

        # Create per-galaxy lists. For a single galaxy we keep the
        # specified central values. For multiple galaxies we generate
        # satellite properties using simple random draws so the
        # generator receives arrays of length ``n_gals`` (primary + satellites).
        all_Re = [central_Re_kpc / spatial_resolution]
        # hz slider is in kpc (like Re) but rotated_system treats all_hz as
        # PIXELS (it does hz_kpc = params['hz'] * pix_spatial_scale), so it
        # must be converted here exactly like Re — otherwise the same slider
        # value means a different physical thickness at every resolution,
        # and since total flux scales with hz (the vertical exponential
        # integrates to 2*hz), the identical galaxy observed at a coarser
        # resolution came out proportionally brighter.
        all_hz = [central_hz / spatial_resolution]
        all_Se = [central_Se]
        all_gal_x_angles = [central_gal_x_angle]
        all_gal_y_angles = [central_gal_y_angle]
        all_n = [central_n]

        if n_gals > 1:
            n_sat = n_gals - 1
            rng = np.random.default_rng()

            # Satellites Re: mean fraction of central Re, ±20% scatter
            _re_frac = float(np.clip(self.sat_Re_frac_var.get(), 0.05, 1.0))
            _re_lo = max(_re_frac - 0.1, 0.05) * all_Re[0]
            _re_hi = min(_re_frac + 0.1, 1.0)  * all_Re[0]
            sat_Re = list(rng.uniform(_re_lo, _re_hi, n_sat))
            sat_hz = list(rng.uniform(all_hz[0] / 3, all_hz[0] / 2, n_sat))

            # sat_brightness_frac (f_sat) is a TOTAL-FLUX fraction: each
            # satellite's total integrated flux is f_sat x the central
            # galaxy's total integrated flux. Se is then solved backwards
            # from that target given the satellite's own Re/n/hz.
            #
            # This is the intuitive knob ("this companion has 36% of the
            # central's light") and, unlike an Se ratio, it means the same
            # thing at every satellite size: shrinking Re_e,sat no longer
            # dims the satellite, it just concentrates the same light into a
            # smaller area. An Se-ratio parameterisation instead made small
            # satellites vanishingly faint, because total flux falls off with
            # Re^2 on top of the fraction.
            #
            # Solved exactly via the analytic Sersic total-flux formula
            # rather than an (Re_c/Re_sat)**2 shortcut, since that shortcut
            # silently assumes the satellite shares the central's n and hz —
            # both of which are sampled independently here. Total flux is
            # linear in Se, so the inverse is a single division:
            # Se_sat = F_target / F(Se=1). Re/hz are in pixels for both
            # galaxies, so the pixel-scale factors cancel in the ratio.
            #
            # Consequence worth knowing: a compact satellite carrying f_sat
            # of the central's light can have a HIGHER peak surface
            # brightness than the central, since the same flux is packed into
            # a smaller area. That is real Sersic behaviour, not an artefact.
            _b = float(np.clip(self.sat_brightness_frac_var.get(), 0.0, 2.0))

            # Random Sérsic indices for satellites
            sat_n = list(rng.uniform(0.5, 1.5, n_sat))

            _central_F_total = _sersic_total_flux_3d(all_Se[0], all_Re[0],
                                                     all_n[0], all_hz[0])
            sat_Se = []
            for re_sat, hz_sat, n_sat_i in zip(sat_Re, sat_hz, sat_n):
                _F_target = _b * _central_F_total * rng.uniform(0.85, 1.15)
                _F_unit_Se = _sersic_total_flux_3d(1.0, re_sat, n_sat_i, hz_sat)
                sat_Se.append(float(_F_target / _F_unit_Se) if _F_unit_Se > 0 else 0.0)

            # Random orientations for satellites (degrees)
            sat_x_angles = list(rng.uniform(-180.0, 180.0, n_sat))
            sat_y_angles = list(rng.uniform(-180.0, 180.0, n_sat))

            all_Re += sat_Re
            all_hz += sat_hz
            all_Se += sat_Se
            all_n += sat_n
            all_gal_x_angles += sat_x_angles
            all_gal_y_angles += sat_y_angles

        # Convert lists to NumPy arrays to match generator expectations
        all_Re = np.array(all_Re)
        all_hz = np.array(all_hz)
        all_Se = np.array(all_Se)
        all_n = np.array(all_n)
        all_gal_x_angles = np.array(all_gal_x_angles)
        all_gal_y_angles = np.array(all_gal_y_angles)
        
        # Compose a `diffuse_params` dict from the GUI controls, layered on
        # top of the package defaults so we never silently drop any key the
        # core helper expects.
        diffuse_params = dict(DEFAULT_DIFFUSE_PARAMS)
        diffuse_params.update({
            'enabled': True,
            'halo_Se_factor': float(self.halo_Se_factor_var.get()),
            'halo_Re_factor': float(self.halo_Re_factor_var.get()),
            'halo_sigma_vz': float(self.halo_sigma_vz_var.get()),
            'bridge_Se_factor': float(self.bridge_Se_factor_var.get()),
            'bridge_width_start_factor': float(self.bridge_width_start_factor_var.get()),
            'bridge_width_end_factor': float(self.bridge_width_end_factor_var.get()),
            'bridge_sigma_vz': float(self.bridge_sigma_vz_var.get()),
            'tail_Se_factor': float(self.tail_Se_factor_var.get()),
            'tail_vel_gradient': float(self.tail_vel_gradient_var.get()),
            'tail_length_factor': float(self.tail_length_var.get()),
            'tail_width_factor': float(self.tail_width_factor_var.get()),
            'tail_sigma_vz': float(self.tail_sigma_vz_var.get()),
        })

        params = dict(
                    beam_info=[bmin,bmaj,bpa],
                    n_gals=n_gals,
                    fov=fov,
                    spectral_resolution=spectral_resolution,
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
                )
        return params

    def _sample_positions(self, redraw=True):
        """Draw new random satellite positions from current offset/FOV settings."""
        n_gals = int(self.n_gals_var.get())
        fov = int(self.fov.get())
        spatial_resolution = max(float(self.spatial_resolution.get()), 0.01)
        grid_size = int(fov / spatial_resolution)   # matches SONGSPhy: truncation, not round
        # init_grid_size mirrors core.py logic
        _igs = (31 / 64) * grid_size
        init_grid_size = int(_igs) - 1 if int(_igs) % 2 != 0 else int(_igs)
        offset_gals = (float(self.sat_offset_min_var.get()) / spatial_resolution,
                       float(self.sat_offset_max_var.get()) / spatial_resolution)
        allow_overlap = self.allow_overlap_var.get() == 'Allow overlap'
        self._galaxy_centers = place_galaxies(n_gals, grid_size, init_grid_size, offset_gals,
                                              allow_overlap=allow_overlap)
        if redraw:
            self._draw_fov_preview_ref()

    def create_generator(self):
        """Instantiate a :class:`SONGS` object from current UI values.

        The method calls :meth:`_collect_parameters` to assemble a parameter
        dictionary and then constructs a single-cube generator instance with
        sensible defaults for fields not exposed directly in the GUI. After
        construction the per-galaxy attributes on the generator are filled
        from the collected parameters so the generator is ready to run.
        """

        params = self._collect_parameters()
        _instance_seed = getattr(self, '_pending_seed', None)
        try:
            g = SONGSPhy(
                n_gals=params['n_gals'],
                n_cubes=1,
                spatial_resolution=params['spatial_resolution'],
                spectral_resolution=params['spectral_resolution'],
                offset_gals=params['offset_gals'],
                beam_info=params['beam_info'],
                fov=params['fov'],
                verbose=True,
                seed=_instance_seed,
                diffuse_params=params['diffuse_params'],
            )
        except Exception as e:
            messagebox.showerror('Error', f'Failed to create SONGS: {e}')
            return

        # Fill the galaxy-specific properties
        n_g = params['n_gals']
        g.all_Re = [params['all_Re']]
        g.all_hz = [params['all_hz']]
        g.all_Se = [params['all_Se']]
        g.all_n = [params['all_n']]
        g.all_gal_x_angles = [params['all_gal_x_angles']]
        g.all_gal_y_angles = [params['all_gal_y_angles']]
        g.all_gal_vz_sigmas = [np.full(n_g, params['sigma_v'])]
        #g.all_pix_spatial_scales = [np.full(n_g, params['spatial_resolution'])]
        g.all_gal_v_0 = [np.full(n_g, 200.0)]  # default systemic velocity

        # "Raw" mode in the Beam card skips spatial beam convolution entirely
        # (spectral smoothing still applies); "Convolved" is the normal default.
        g.convolve_beam = (self.beam_mode_var.get() == 'Convolved')
        g.allow_overlap = (self.allow_overlap_var.get() == 'Allow overlap')

        self.generator = g


    def _run_generate(self, use_noise=False, sn_peak=None):
        # Disable garbage collection in this thread to prevent cleanup
        # of Tkinter objects from the wrong thread
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()

        try:
            # Check if closing before doing expensive work
            if self._is_closing:
                return

            # NOTE: the log window is raised/created on the MAIN thread by
            # generate() before this worker starts — doing it here touched Tk
            # from a background thread ("main thread is not in main loop"),
            # and because that call sat in this outer try (which has only a
            # finally, no except) the exception escaped the thread silently:
            # no popup, generation never ran, and the Slice/Analysis/Save
            # buttons disabled at the start of generate() were never
            # re-enabled. Only bit on the SECOND generate onward, when
            # log_window already existed and the deiconify/lift path ran.

            try:
                results = self.generator.generate_cubes()
                # Check again before publishing results
                if self._is_closing:
                    return

                # Slice/Analysis view self.generator.results directly, which
                # generate_cubes() always leaves as the CLEAN cube (Save
                # relies on that staying clean so it can write a proper
                # clean/noisy pair) — so "Use Noise? Yes" had no visible
                # effect until you saved. Precompute a separate noisy
                # preview here (background thread — apply_and_convolve_noise
                # is pure numpy, no Tk calls) and show it in Slice/Analysis
                # instead, without touching self.generator.results.
                preview_results = None
                if use_noise and results:
                    try:
                        cube0, meta0 = results[0]
                        beam_px = [float(self.generator.beam_info[0]),
                                  float(self.generator.beam_info[1]),
                                  float(self.generator.beam_info[2])]
                        noisy_cube0 = apply_and_convolve_noise(cube0, beam_px, sn_peak)
                        preview_results = [(noisy_cube0, meta0)]
                    except Exception as e:
                        print(f"[SONGS] Noisy preview failed, showing clean cube instead: {e}")

                # Hand the outcome back to the MAIN thread as plain Python
                # attributes. This worker must not touch Tk at all — not even
                # self.after(), which is itself a Tk call (it does
                # tk.createcommand under the hood) and can raise
                # "main thread is not in main loop". When that happened the
                # exception escaped the thread with no dialog and the buttons
                # disabled at the start of generate() were never re-enabled —
                # the window looked permanently dead. _poll_generation_done(),
                # scheduled on the main thread by generate(), picks these up.
                self._gen_preview_results = preview_results
                self._gen_error = None
                self._gen_done = True
            except Exception as e:
                self._gen_preview_results = None
                self._gen_error = e
                self._gen_done = True
        except Exception as e:
            # Safety net: anything unexpected outside the generation block
            # itself must still be reported and must still release the UI.
            self._gen_preview_results = None
            self._gen_error = e
            self._gen_done = True
        finally:
            # Re-enable garbage collection if it was enabled
            if gc_was_enabled:
                gc.enable()

    def _poll_generation_done(self):
        """Main-thread poller for the background generation worker.

        Runs entirely on the main thread (rescheduled via ``after``), so every
        Tk call here is safe. The worker communicates only through plain
        attributes (``_gen_done`` / ``_gen_error`` / ``_gen_preview_results``)
        — see the note in :meth:`_run_generate` for why it must not call
        ``after`` itself.
        """
        if self._is_closing:
            return
        if not getattr(self, '_gen_done', False):
            self.after(150, self._poll_generation_done)
            return

        self._gen_done = False          # consume, so a later run re-arms cleanly
        self._gen_in_progress = False
        err = getattr(self, '_gen_error', None)
        self._gen_error = None

        try:
            self._enable_theme_btn()
        except Exception:
            pass

        if err is not None:
            messagebox.showerror('Error during generation', str(err))
            # A failed run produced no results of its own; only re-offer the
            # result buttons if an earlier run left usable ones behind.
            if getattr(self, '_has_results', False):
                for _b in (self.analysis_btn, self.slice_btn,
                           self.save_btn, self.new_instance_btn):
                    try: self._enable_btn(_b)
                    except Exception: pass
            return

        self._has_results = True
        self._results_generation += 1
        self._preview_results = getattr(self, '_gen_preview_results', None)
        for _b in (self.analysis_btn, self.slice_btn,
                   self.save_btn, self.new_instance_btn):
            try: self._enable_btn(_b)
            except Exception: pass

    def generate(self):
        import random as _random
        self._pending_seed = _random.randint(0, 2**31 - 1)
        print(f"[SONGS] Instance seed: {self._pending_seed}")
        self.create_generator()

        if self.generator is None:
            return

        # Disable theme button and result buttons while generating
        self._disable_theme_btn()
        for _b in (self.analysis_btn, self.slice_btn, self.save_btn, self.new_instance_btn):
            try: self._disable_btn(_b)
            except Exception: pass
        self._preview_results = None   # drop any stale noisy preview from a previous run

        # Raise/create the log window HERE, on the main thread — it used to
        # be done inside _run_generate, i.e. from the worker thread, which is
        # not safe for Tk and blew up (silently, killing the worker) from the
        # second generate onward. See the note in _run_generate.
        self._show_log_window()

        # Stamp the pre-sampled positions onto the generator right before launch,
        # so they reflect the current UI state and are not overwritten by traces.
        self.generator._preset_centers = self._galaxy_centers
        # Read Tk vars here (main thread) — _run_generate runs in a
        # background thread and must never touch Tk Vars directly.
        use_noise = (self.use_noise_var.get() == 'Yes')
        sn_peak = float(self.sn_peak_var.get()) if use_noise else None

        # Re-arm the worker handshake, then start polling on the main thread.
        # The worker never touches Tk (see _run_generate); this poller is what
        # actually re-enables the buttons when it finishes.
        self._gen_done = False
        self._gen_error = None
        self._gen_preview_results = None
        # Blocks _auto_update_generator from rebuilding self.generator (and
        # wiping its results) while the worker is using it.
        self._gen_in_progress = True

        t = threading.Thread(target=self._run_generate, args=(use_noise, sn_peak), daemon=True)
        t.start()
        self.after(150, self._poll_generation_done)

    def _show_log_window(self):
        """Raise the log window (creating it if needed). MAIN THREAD ONLY —
        every call here touches Tk, so background workers must never invoke
        it directly; schedule it with self.after(0, ...) instead."""
        try:
            win = getattr(self, 'log_window', None)
            if win is not None and win.winfo_exists():
                win.deiconify()
                win.lift()
            else:
                win = self.log_window = LogWindow(self)
                win.deiconify()
                win.lift()
            # Theme the title bar now that it is actually on screen (it is
            # created withdrawn, so this can't be done in its constructor).
            try:
                from .visualise import _apply_window_appearance as _apply_appearance
            except Exception:
                from songs.visualise import _apply_window_appearance as _apply_appearance
            _apply_appearance(win, self._theme)
            self.after(250, lambda w=win: _apply_appearance(w, self._theme))
        except Exception as e:
            # A log window is a convenience, never a reason to abort a run.
            print(f"[SONGS] Could not open log window: {e}")

    def stop_generation(self):
        """Request cancellation of whatever generation is currently
        running (single-cube or Large-Dataset Mode batch). Cooperative —
        the background worker checks ``self._stop_requested`` at its next
        safe point and unwinds itself; this just raises the flag and gives
        immediate feedback (disabling the Stop button so a second click
        can't double-fire)."""
        if self._stop_requested:
            return
        self._stop_requested = True
        self._disable_btn(self.stop_btn)

    # ---------------------------
    # Large-Dataset Mode: batch generation
    # ---------------------------
    def _set_ld_progress(self, i, total):
        """Update the 'Generated cube i/total' label, blocky progress bar,
        and percentage. ``i`` may be fractional (e.g. 3.4) so the bar can
        creep smoothly toward the next cube instead of jumping once per
        cube — see ``_ld_progress_ticker``; the label always shows the
        floored integer count. Safe to call from the main thread only —
        background workers must go through ``self.after(0, ...)``."""
        lbl = getattr(self, '_ld_progress_lbl', None)
        cv = getattr(self, '_ld_progress_cv', None)
        pct_lbl = getattr(self, '_ld_progress_pct_lbl', None)
        if lbl is None or cv is None or pct_lbl is None:
            return
        self._ld_progress_last_iv = (i, total)
        lbl.configure(text=f'Generated cube {int(i)}/{total}')
        frac = 0.0 if total <= 0 else min(1.0, i / total)
        t = _THEMES[self._theme]
        cv.configure(bg=t['BTN_NOR_BG'])
        cv.delete('all')
        n = getattr(self, '_ld_progress_nblocks', 20)
        filled = int(round(frac * n))
        w = cv.winfo_width()
        if w < n:
            w = n * 10  # not laid out yet — fall back to a sane default
        gap = 2
        bw = max(1, (w - gap * (n - 1)) / n)
        h = int(cv.cget('height'))
        for k in range(n):
            x0 = k * (bw + gap)
            fill = t['ACCENT'] if k < filled else t['PILL_NOR']
            cv.create_rectangle(x0, 0, x0 + bw, h, fill=fill, outline=fill)
        pct_lbl.configure(text=f'{int(round(frac * 100))}%')

    def _ld_progress_ticker(self):
        """While generating, creep the bar smoothly toward the next cube
        between the discrete per-cube updates, using a running average of
        how long previous cubes in this run took. Self-reschedules on the
        main thread; stops as soon as ``_ld_generating`` goes False."""
        if not getattr(self, '_ld_generating', False):
            return
        done = getattr(self, '_ld_done', 0)
        total = getattr(self, '_ld_total', 0)
        avg = getattr(self, '_ld_avg_cube_time', None)
        start = getattr(self, '_ld_cube_start_time', None)
        if avg and start and avg > 0 and done < total:
            intra = min(0.95, (time.time() - start) / avg)
            self._set_ld_progress(done + intra, total)
        self.after(120, self._ld_progress_ticker)

    def _set_generate_dataset_enabled(self, enabled):
        """Toggle the (repurposed) Generate button while a batch dataset
        run is in progress. Only meaningful while Large-Dataset Mode is on
        — ``_apply_large_dataset_mode`` restores the normal 'Generate'
        label/action as soon as the mode is switched off. While disabled,
        it's styled exactly like the other deactivated buttons
        (Load/Slice/Analysis/Save/Reset) rather than keeping its gold
        background."""
        btn = getattr(self, 'generate_btn', None)
        if btn is None:
            return
        t = _THEMES[self._theme]
        if enabled:
            gen_fg = "#000000" if self._theme == 'dark' else "#ffffff"
            btn.configure(bg=t['ACCENT'], fg=gen_fg, cursor='pointinghand', text='Generate Dataset')
            btn.unbind('<ButtonRelease-1>')
            btn.bind('<ButtonRelease-1>', lambda e, b=btn: b._btn_cmd())
        else:
            btn.configure(bg=t['BTN_NOR_BG'], fg=t['BTN_DIS_FG'], cursor='arrow', text='Generating…')
            btn.unbind('<ButtonRelease-1>')

    def _build_cube_manifest(self, params, sn_peak=None):
        """Build the per-cube parameter manifest saved as the HDF5
        ``parameters_json`` header attribute. ``params`` accepts either a
        large-dataset ``_sample_cube_params()`` dict or a single-cube
        ``_collect_parameters()`` dict — both share the same key names
        (``all_Re`` is in pixels in both, converted to kpc here) — so both
        generation paths produce an identical manifest schema."""
        sr = float(params['spatial_resolution'])
        sigma_v = params['sigma_v']
        sigma_v = float(np.mean(sigma_v)) if not np.isscalar(sigma_v) else float(sigma_v)
        return dict(
            n_gals=int(params['n_gals']),
            sersic_n=[float(v) for v in np.atleast_1d(params['all_n'])],
            Re_kpc=[float(v) * sr for v in np.atleast_1d(params['all_Re'])],
            # all_hz is in pixels (same convention as all_Re) — convert to
            # kpc so the manifest is resolution-independent and round-trips
            # back through _apply_loaded_manifest's hz_var (also kpc).
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

    def _collect_large_dataset_base(self):
        """Read every Tk var needed for batch generation ONCE, on the main
        thread, into a plain dict. The background worker must never touch
        Tk Vars directly (not thread-safe)."""
        return dict(
            # Fixed "initialisation" parameters — shared by every cube.
            # Beam is a pixel footprint in Large-Dataset Mode (see the
            # "Beam [px, px, deg]" card) — converted back to kpc per-cube
            # in _sample_cube_params using that cube's own spatial_resolution,
            # so every cube shares the same beam width in pixels.
            bmin_px=float(self.bmin_px_var.get()),
            bmaj_px=float(self.bmaj_px_var.get()),
            bpa=float(self.bpa_var.get()),
            grid_size=int(self.spatial_pixel_dim_var.get()),
            spectral_resolution=int(self.spectral_resolution.get()),
            # Spatial resolution (and therefore FOV, kept dynamically in
            # sync with it and the fixed pixel grid size — see the dual
            # sliders below) is drawn per-cube from its explicit range, so
            # the same objects (fixed Re, see Re_fixed) end up sampled at
            # different resolutions across the dataset.
            spatial_resolution_range=(max(float(self.spatial_resolution_min_var.get()), 0.01),
                                      max(float(self.spatial_resolution_max_var.get()), 0.01)),
            # Satellite offset is a pixel footprint in Large-Dataset Mode
            # (see the "Min & Max Offset from Central [px, px]" card) —
            # converted to kpc per-cube in _sample_cube_params using that
            # cube's own spatial_resolution, so every cube keeps the same
            # offset range in pixels (physical separation scales with
            # resolution instead, unlike Re which stays physically fixed).
            offset_gals_px=(float(self.sat_offset_min_px_var.get()), float(self.sat_offset_max_px_var.get())),
            convolve_beam=(self.beam_mode_var.get() == 'Convolved'),
            allow_overlap=(self.allow_overlap_var.get() == 'Allow overlap'),

            # Every physical parameter is drawn uniformly from its explicit
            # [min, max] range (the dual-mode sliders in Large-Dataset Mode),
            # except Re, which stays fixed (single slider, see Re_var) so
            # the same physical galaxy size is sampled at varying resolutions.
            hz_range=(float(self.hz_min_var.get()), float(self.hz_max_var.get())),
            Se_range=(float(self.Se_min_var.get()), float(self.Se_max_var.get())),
            Re_fixed=float(self.Re_var.get()),
            sigma_v_range=(float(self.sigma_v_min_var.get()), float(self.sigma_v_max_var.get())),
            sat_brightness_frac_range=(float(self.sat_brightness_frac_min_var.get()),
                                       float(self.sat_brightness_frac_max_var.get())),
            sat_Re_frac_range=(float(self.sat_Re_frac_min_var.get()), float(self.sat_Re_frac_max_var.get())),

            max_gals=int(self.max_gals_per_cube_var.get()),
            angle_x_range=(float(self.angle_x_min_var.get()), float(self.angle_x_max_var.get())),
            angle_y_range=(float(self.angle_y_min_var.get()), float(self.angle_y_max_var.get())),
            n_sersic_range=(float(self.n_min_var.get()), float(self.n_max_var.get())),

            # Noise (correlated, beam-convolved — src/songs/utils.py:apply_and_convolve_noise).
            use_noise=(self.use_noise_ld_var.get() == 'Yes'),
            sn_range=(float(self.sn_peak_min_var.get()), float(self.sn_peak_max_var.get())),

            diffuse_ranges=dict(
                halo_Se_factor=(float(self.halo_Se_factor_min_var.get()), float(self.halo_Se_factor_max_var.get())),
                halo_Re_factor=(float(self.halo_Re_factor_min_var.get()), float(self.halo_Re_factor_max_var.get())),
                halo_sigma_vz=(float(self.halo_sigma_vz_min_var.get()), float(self.halo_sigma_vz_max_var.get())),
                bridge_Se_factor=(float(self.bridge_Se_factor_min_var.get()), float(self.bridge_Se_factor_max_var.get())),
                bridge_width_start_factor=(float(self.bridge_width_start_factor_min_var.get()),
                                           float(self.bridge_width_start_factor_max_var.get())),
                bridge_width_end_factor=(float(self.bridge_width_end_factor_min_var.get()),
                                         float(self.bridge_width_end_factor_max_var.get())),
                bridge_sigma_vz=(float(self.bridge_sigma_vz_min_var.get()), float(self.bridge_sigma_vz_max_var.get())),
                tail_Se_factor=(float(self.tail_Se_factor_min_var.get()), float(self.tail_Se_factor_max_var.get())),
                tail_vel_gradient=(float(self.tail_vel_gradient_min_var.get()), float(self.tail_vel_gradient_max_var.get())),
                tail_length_factor=(float(self.tail_length_min_var.get()), float(self.tail_length_max_var.get())),
                tail_width_factor=(float(self.tail_width_factor_min_var.get()), float(self.tail_width_factor_max_var.get())),
                tail_sigma_vz=(float(self.tail_sigma_vz_min_var.get()), float(self.tail_sigma_vz_max_var.get())),
            ),

            n_samples=int(self.n_samples_var.get()),
            save_folder=self.save_folder_var.get().strip(),
        )

    def _sample_cube_params(self, base, rng):
        """Pure-Python per-cube parameter sampler (thread-safe — no Tk var
        access). Mirrors ``_collect_parameters()``'s shape/satellite logic,
        but every physical parameter is drawn uniformly from its explicit
        [min, max] range, except Re, which stays fixed (see
        base['Re_fixed']) across the whole dataset."""

        def u(bounds):
            lo, hi = bounds
            if hi <= lo:
                return float(lo)
            return float(rng.uniform(lo, hi))

        # Spatial resolution is drawn per-cube from its range; FOV is then
        # derived so grid_size = fov/spatial_resolution stays fixed at the
        # dataset's "Spatial pixel grid size" for every cube, keeping pixel
        # dimensions constant across varying physical resolution. SONGSPhy
        # recomputes grid_size = int(fov/spatial_resolution) internally
        # (truncation, not round), and fov must be a whole number of kpc.
        #
        # Nudging fov upward until int(fov/sr) >= grid_size does NOT work:
        # when sr < 1 each +1 kpc of fov adds 1/sr > 1 pixels, so the loop
        # can jump straight PAST grid_size and settle on grid_size + 1 (seen
        # in practice: sr=0.8975, fov=44 -> int(44/0.8975) = 49 for a
        # requested 48). That yields mismatched cube shapes in one dataset,
        # which breaks batch collation when training.
        #
        # Instead pick the nearest integer fov and then re-derive
        # spatial_resolution from it, so the division lands on grid_size by
        # construction rather than by search. The tiny epsilon keeps
        # floating-point error on the exact quotient from truncating down to
        # grid_size - 1. sr shifts by at most 0.5/grid_size kpc/px, which is
        # immaterial for a value that was a random draw to begin with, and
        # the manifest records the adjusted value actually used.
        spatial_resolution = max(u(base['spatial_resolution_range']), 0.01)
        fov = max(1, int(round(base['grid_size'] * spatial_resolution)))
        spatial_resolution = fov / (base['grid_size'] + 1e-9)
        # Beam footprint stays fixed in pixels across the dataset; convert
        # to this cube's kpc using its own sampled spatial_resolution.
        bmin_kpc = base['bmin_px'] * spatial_resolution
        bmaj_kpc = base['bmaj_px'] * spatial_resolution
        # Same for satellite offset — fixed in pixels, converted to this
        # cube's kpc so physical separation scales with resolution.
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
        # hz slider is in kpc (like Re) but rotated_system treats all_hz as
        # PIXELS (it does hz_kpc = params['hz'] * pix_spatial_scale), so it
        # must be converted here exactly like Re — otherwise the same slider
        # value means a different physical thickness at every resolution,
        # and since total flux scales with hz (the vertical exponential
        # integrates to 2*hz), the identical galaxy observed at a coarser
        # resolution came out proportionally brighter.
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
            # Total-flux fraction of the central, solved exactly for Se via
            # the analytic Sersic total-flux formula (accounts for each
            # satellite's own n/hz) — see the matching comment in
            # _collect_parameters().
            _central_F_total = _sersic_total_flux_3d(all_Se[0], all_Re[0],
                                                     central_n, central_hz)
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

    def generate_dataset(self):
        """Kick off batch generation for Large-Dataset Mode: samples fresh
        parameters per cube, generates it, optionally adds correlated
        beam-convolved noise, and writes clean/noisy HDF5 files (with the
        full per-cube parameter manifest as a JSON header attribute) plus
        one dataset-level pickle summarising every cube. Runs in a
        background thread; the GUI only reads/writes Tk state before the
        thread starts and via ``self.after(0, ...)`` afterwards."""
        if getattr(self, '_ld_generating', False):
            return
        if not self._large_dataset_mode:
            return
        base = self._collect_large_dataset_base()
        if not base['save_folder']:
            messagebox.showerror('No save folder', 'Choose a save folder first (Choose Save Folder button).')
            return
        if base['n_samples'] < 1:
            return

        self._ld_generating = True
        self._ld_done = 0
        self._ld_total = base['n_samples']
        self._ld_avg_cube_time = None
        self._ld_cube_start_time = time.time()
        self._set_generate_dataset_enabled(False)
        self._set_ld_progress(0, base['n_samples'])
        self._stop_requested = False
        self._enable_btn(self.stop_btn)

        t = threading.Thread(target=self._run_generate_dataset, args=(base,), daemon=True)
        t.start()
        self.after(120, self._ld_progress_ticker)

    def _run_generate_dataset(self, base):
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        n = base['n_samples']
        manifest = []
        stopped = False
        try:
            raw_dir = os.path.join(base['save_folder'], 'raw')
            os.makedirs(raw_dir, exist_ok=True)
            rng = np.random.default_rng()

            for i in range(n):
                if self._is_closing:
                    return
                if self._stop_requested:
                    stopped = True
                    break
                cube_t0 = time.time()
                cube_params = self._sample_cube_params(base, rng)
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
                    print(f"[Large-Dataset] cube {i + 1} failed: {e}")
                    self._ld_cube_start_time = time.time()
                    continue

                manifest_entry = self._build_cube_manifest(cube_params, sn_peak=cube_params['sn_peak'])
                manifest_entry['index'] = i + 1
                # Self-describing: any downstream tool can locate the file
                # from the manifest entry alone, without having to know/
                # reconstruct the "cube_{index:05d}.h5" naming convention.
                cube_filename = f'cube_{i + 1:05d}.h5'
                manifest_entry['filename'] = cube_filename

                # A noise-enabled cube gets both versions in ONE file, as
                # /clean_cube and /noisy_cube (see _save_cube_hdf5) — no more
                # separate _clean.h5/_noisy.h5 pair.
                noisy_cube = None
                if base['use_noise'] and cube_params['sn_peak'] is not None:
                    beam_px = [float(g.beam_info[0]), float(g.beam_info[1]), float(g.beam_info[2])]
                    noisy_cube = apply_and_convolve_noise(cube, beam_px, cube_params['sn_peak'])

                cube_path = os.path.join(raw_dir, cube_filename)
                _save_cube_hdf5(cube_path, cube, core_params, g, 0, noisy_cube=noisy_cube)
                with h5py.File(cube_path, 'a') as f:
                    f.attrs['parameters_json'] = json.dumps(manifest_entry)

                manifest.append(manifest_entry)

                dt = time.time() - cube_t0
                self._ld_avg_cube_time = (dt if self._ld_avg_cube_time is None
                                          else 0.5 * self._ld_avg_cube_time + 0.5 * dt)
                self._ld_done = i + 1
                self._ld_cube_start_time = time.time()

                _i = i + 1
                self.after(0, lambda _i=_i, _n=n: self._set_ld_progress(_i, _n))

            # JSON, not pickle — every field here is already a plain
            # float/int/list/dict/None (each manifest_entry is proven
            # JSON-safe already, since it's independently json.dumps()'d
            # into every cube's HDF5 parameters_json attribute above), so
            # there's no reason to pay pickle's Python-version/security
            # cost for what is genuinely just a manifest.
            dataset_path = os.path.join(base['save_folder'], 'dataset.json')
            with open(dataset_path, 'w') as fh:
                json.dump(dict(
                    n_samples=n,
                    n_generated=len(manifest),
                    use_noise=base['use_noise'],
                    manifest=manifest,
                ), fh, indent=2)

            if not self._is_closing:
                _done = len(manifest)
                _title = 'Dataset stopped' if stopped else 'Dataset complete'
                _verb = 'Stopped after generating' if stopped else 'Generated'
                self.after(0, lambda: messagebox.showinfo(
                    _title, f'{_verb} {_done}/{n} cubes.\nSaved to {base["save_folder"]}'))
        except Exception as e:
            if not self._is_closing:
                self.after(0, lambda e=e: messagebox.showerror('Error generating dataset', str(e)))
        finally:
            if gc_was_enabled:
                gc.enable()
            self._ld_generating = False
            if not self._is_closing:
                self.after(0, lambda: self._set_generate_dataset_enabled(True))
                self.after(0, lambda: self._enable_btn(self.new_instance_btn))
                self.after(0, lambda: self._disable_btn(self.stop_btn))

    # ---------------------------
    # Save simulation (cube + params)
    # ---------------------------
    # ---------------------------
    # Load a pre-made SONGS cube (single-cube mode only) — read-only viewing
    # ---------------------------
    def load_cube(self):
        """Open a SONGS HDF5 cube for viewing only (Slice/Analysis), without
        running the generator. Populates every slider from the file's
        ``parameters_json`` header and locks the UI read-only until Reset is
        pressed. If the file has both a clean and a noisy version of the
        cube (see _save_cube_hdf5), both are loaded and the Slice/Analysis
        viewers get the same Clean/Noisy toggle used for live generation —
        no more up-front "which one?" prompt."""
        path = filedialog.askopenfilename(
            title='Load SONGS cube',
            filetypes=[('SONGS HDF5', '*.h5 *.hdf5'), ('All files', '*.*')],
        )
        if not path:
            return

        try:
            clean_cube, noisy_cube, meta, manifest = self._load_songs_hdf5(path)
            self._apply_loaded_manifest(manifest)
            self._enter_loaded_view(clean_cube, meta, noisy_cube=noisy_cube)
        except Exception as e:
            messagebox.showerror('Load error', f'Failed to load {path}:\n{e}')
            return

    def _load_songs_hdf5(self, path):
        """Read a SONGS HDF5 cube (as written by ``_save_cube_hdf5``) and
        return ``(clean_cube, noisy_cube, meta, manifest)`` — ``noisy_cube``
        is None unless the file has a noise-enabled ``/clean_cube`` +
        ``/noisy_cube`` pair (older single-``/cube`` files load as
        clean-only). ``meta`` is shaped for SliceViewer/AnalysisViewer,
        ``manifest`` is the ``parameters_json`` header dict (empty if the
        file predates that attribute)."""
        with h5py.File(path, 'r') as f:
            if 'clean_cube' in f and 'noisy_cube' in f:
                clean_cube = np.asarray(f['clean_cube'][:], dtype=np.float32)
                noisy_cube = np.asarray(f['noisy_cube'][:], dtype=np.float32)
            else:
                clean_cube = np.asarray(f['cube'][:], dtype=np.float32)
                noisy_cube = None
            avg_v = (np.asarray(f['channel_velocities_km_s'][:], dtype=np.float64)
                     if 'channel_velocities_km_s' in f else np.arange(clean_cube.shape[0], dtype=np.float64))
            beam_info = [1.0, 1.0, 0.0]
            if 'beam' in f:
                bg = f['beam']
                beam_info = [float(bg.attrs.get('bmin_px', 1.0)),
                            float(bg.attrs.get('bmaj_px', 1.0)),
                            float(bg.attrs.get('bpa_deg', 0.0))]
            pix_scale = float(f.attrs.get('spatial_resolution_kpc_per_px', 1.0))
            per_gal = None
            if 'galaxies' in f and 'cubes' in f['galaxies']:
                per_gal = np.asarray(f['galaxies']['cubes'][:])
            raw = f.attrs.get('parameters_json')
            manifest = json.loads(raw) if raw else {}

        meta = dict(average_vels=avg_v, beam_info=beam_info,
                    pix_spatial_scale=pix_scale, per_galaxy_cubes=per_gal)
        return clean_cube, noisy_cube, meta, manifest

    def _apply_loaded_manifest(self, manifest):
        """Set every GUI slider/toggle to the values recorded in a loaded
        cube's parameter manifest (central/first galaxy's values)."""
        if not manifest:
            return

        def _get(key, default=None):
            return manifest.get(key, default)

        self.n_gals_var.set(int(_get('n_gals', 1)))

        sersic = _get('sersic_n') or [1.0]
        Re_kpc = _get('Re_kpc') or [10.0]
        hz = _get('hz') or [1.0]
        Se = _get('Se') or [0.1]
        incl_x = _get('inclination_x_deg') or [0.0]
        incl_y = _get('inclination_y_deg') or [0.0]

        self.n_var.set(float(sersic[0]))
        self.Re_var.set(float(Re_kpc[0]))
        self.hz_var.set(float(hz[0]))
        self.Se_var.set(float(Se[0]))
        self.angle_x_var.set(int(round(incl_x[0])))
        self.angle_y_var.set(int(round(incl_y[0])))
        self.sigma_v_var.set(float(_get('sigma_v_km_s', 40.0)))

        beam = _get('beam_info_kpc')
        if beam and len(beam) == 3:
            self.bmin_var.set(float(beam[0]))
            self.bmaj_var.set(float(beam[1]))
            self.bpa_var.set(float(beam[2]))

        if _get('fov_kpc') is not None:
            self.fov.set(int(_get('fov_kpc')))
        if _get('spatial_resolution_kpc_per_px') is not None:
            self.spatial_resolution.set(float(_get('spatial_resolution_kpc_per_px')))
        if _get('spectral_resolution_km_s') is not None:
            self.spectral_resolution.set(int(_get('spectral_resolution_km_s')))

        dp = _get('diffuse_params') or {}
        _dp_map = dict(
            halo_Se_factor='halo_Se_factor_var', halo_Re_factor='halo_Re_factor_var',
            halo_sigma_vz='halo_sigma_vz_var', bridge_Se_factor='bridge_Se_factor_var',
            bridge_width_start_factor='bridge_width_start_factor_var',
            bridge_width_end_factor='bridge_width_end_factor_var',
            bridge_sigma_vz='bridge_sigma_vz_var', tail_Se_factor='tail_Se_factor_var',
            tail_vel_gradient='tail_vel_gradient_var', tail_length_factor='tail_length_var',
            tail_width_factor='tail_width_factor_var', tail_sigma_vz='tail_sigma_vz_var',
        )
        for k, varname in _dp_map.items():
            if k in dp and hasattr(self, varname):
                try:
                    getattr(self, varname).set(float(dp[k]))
                except Exception:
                    pass

        sn_peak = _get('sn_peak')
        if sn_peak is not None:
            self.use_noise_var.set('Yes')
            self.sn_peak_var.set(float(sn_peak))
        else:
            self.use_noise_var.set('No')

    def _enter_loaded_view(self, cube, meta, noisy_cube=None):
        """Wire a loaded (cube, meta) pair up as if it were a fresh
        generator result, then lock every parameter card read-only —
        loaded cubes are for visualising, not editing. ``noisy_cube`` (from
        a file with both /clean_cube and /noisy_cube) is stored the same
        way _run_generate stores its noisy preview, so show_slice/
        show_analysis's existing Clean/Noisy toggle wiring picks it up with
        no further changes."""
        import types
        self.generator = types.SimpleNamespace(results=[(cube, meta)])
        self._has_results = True
        self._results_generation += 1
        self._preview_results = [(noisy_cube, meta)] if noisy_cube is not None else None
        self._loaded_view_active = True

        for root_name in ('_bc1_outer', '_bc2_outer', '_bc3_outer', '_bc4_outer'):
            root = getattr(self, root_name, None)
            if root is not None:
                self._lock_walk(root, True)

        self._disable_btn(self.generate_btn)
        self._enable_btn(self.slice_btn)
        self._enable_btn(self.analysis_btn)
        self._enable_btn(self.new_instance_btn)

    def save_sim(self):
        """Generate (if needed) and save the sim tuple (cube, params).

        This runs generation in a background thread and then opens a
        Save-As dialog on the main thread to let the user choose where
        to store the result. We support .npz (numpy savez) and .pkl
        (pickle) formats; complex parameter dicts fall back to pickle.
        """
        # If we already have generated results, save them directly without
        # re-running the (potentially expensive) generation. Otherwise,
        # fall back to running generation in background and then prompting
        # the user to save.
        try:
            has_results = bool(self.generator and getattr(self.generator, 'results', None))
        except Exception:
            has_results = False

        if has_results:
            # Use existing results (do not re-run generation)
            results = self.generator.results
            # extract first cube/meta
            cube = None
            meta = None
            if isinstance(results, (list, tuple)) and len(results) > 0:
                first = results[0]
                if isinstance(first, tuple) and len(first) >= 2:
                    cube, meta = first[0], first[1]
                else:
                    cube = first
            else:
                cube = results

            params = self._collect_parameters()
            # Prompt on main thread
            self.after(0, lambda: self._save_sim_prompt(cube, params, meta))
            return

        # No existing results: run generation in background then prompt to save
        if self.generator is None:
            # create generator from current GUI values
            self.create_generator()
            if self.generator is None:
                return

        self.generator._preset_centers = self._galaxy_centers
        t = threading.Thread(target=self._save_sim_thread, daemon=True)
        t.start()

    def _save_sim_thread(self):
        """Background worker that runs generation and then prompts to save.

        Runs ``self.generator.generate_cubes()`` in the background thread and
        then schedules :meth:`_save_sim_prompt` on the main thread to show the
        Save-As dialog. Errors are displayed via a messagebox scheduled on
        the main thread.
        """
        # Disable garbage collection in this thread to prevent cleanup
        # of Tkinter objects from the wrong thread
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        
        try:
            # Check if closing before doing expensive work
            if self._is_closing:
                return

            try:
                results = self.generator.generate_cubes()
            except Exception as e:
                if not self._is_closing:
                    self.after(0, lambda e=e: messagebox.showerror('Error during generation', str(e)))
                return

            # Check again after generation completes
            if self._is_closing:
                return

            # extract first cube and params
            cube = None
            meta = None
            if isinstance(results, (list, tuple)) and len(results) > 0:
                first = results[0]
                if isinstance(first, tuple) and len(first) >= 2:
                    cube, meta = first[0], first[1]
                else:
                    cube = first
            else:
                cube = results

            params = self._collect_parameters()

            # prompt/save on main thread
            if not self._is_closing:
                self.after(0, lambda: self._save_sim_prompt(cube, params, meta))
        finally:
            # Re-enable garbage collection if it was enabled
            if gc_was_enabled:
                gc.enable()

    def _save_sim_prompt(self, cube, params, meta=None):
        """Prompt the user for a filename and save the provided cube/params.

        Parameters
        ----------
        cube : ndarray
            Spectral cube array to save.
        params : dict
            Parameters dictionary produced by :meth:`_collect_parameters`.
        meta : dict or None
            Optional metadata returned by the generator.
        """

        # Ask for filename
        fname = filedialog.asksaveasfilename(
            defaultextension='.h5',
            filetypes=[
                ('HDF5 file', '.h5'),
                ('NumPy archive', '.npz'),
                ('Pickled Python object', '.pkl'),
            ],
        )
        if not fname:
            return

        try:
            if fname.lower().endswith('.h5') or fname.lower().endswith('.hdf5'):
                # Same writer + manifest schema as the Large-Dataset Mode
                # per-cube save, so every SONGS HDF5 (single-cube or batch)
                # carries an identical, fully self-describing header.
                use_noise = getattr(self, 'use_noise_var', None) and self.use_noise_var.get() == 'Yes'
                sn_peak = float(self.sn_peak_var.get()) if use_noise else None
                manifest = self._build_cube_manifest(params, sn_peak=sn_peak)

                # A noise-enabled cube gets both versions in ONE file, as
                # /clean_cube and /noisy_cube (see _save_cube_hdf5) — no more
                # separate _clean.h5/_noisy.h5 pair.
                noisy_cube = None
                if use_noise:
                    beam_px = [float(self.generator.beam_info[0]),
                              float(self.generator.beam_info[1]),
                              float(self.generator.beam_info[2])]
                    noisy_cube = apply_and_convolve_noise(cube, beam_px, sn_peak)

                _save_cube_hdf5(fname, cube, meta or {}, self.generator, 0, noisy_cube=noisy_cube)
                with h5py.File(fname, 'a') as f:
                    f.attrs['parameters_json'] = json.dumps(manifest)
            elif fname.lower().endswith('.npz'):
                # try to prepare a flat dict for savez
                save_dict = {}
                save_dict['cube'] = cube
                # flatten params into arrays where possible
                for k, v in params.items():
                    try:
                        if isinstance(v, (list, tuple)):
                            save_dict[k] = np.array(v)
                        else:
                            save_dict[k] = v
                    except Exception:
                        save_dict[k] = v
                # include meta if available
                if meta is not None:
                    try:
                        save_dict['meta'] = meta
                    except Exception:
                        pass
                np.savez(fname, **save_dict)
            else:
                with open(fname, 'wb') as fh:
                    pickle.dump((cube, params, meta), fh)
        except Exception as e:
            messagebox.showerror('Save error', f'Failed to save simulation: {e}')
            return

        messagebox.showinfo('Saved', f'Simulation saved to {fname}')

    # ---------------------------
    # Cleanup
    # ---------------------------
    def _on_close(self):
        """Cleanup temporary files created for LaTeX rendering and exit.

        Sets a flag to stop background threads from scheduling UI updates,
        removes any temporary PNG files recorded in ``_MATH_TEMPFILES``,
        and performs a graceful shutdown of the Tkinter application.
        """
        # Signal threads to stop scheduling UI updates
        self._is_closing = True
        
        # Clean up temporary files
        for p in list(_MATH_TEMPFILES):
            try: 
                os.remove(p)
            except: 
                pass
        
        # Graceful Tkinter shutdown
        try:
            self.quit()  # Stop the mainloop
        except Exception:
            pass
        
        try:
            self.destroy()  # Destroy all widgets
        except Exception:
            pass


def main():
    import argparse
    parser = argparse.ArgumentParser(prog='songs.gui', description='SONGS spectral cube simulator GUI')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dark',  action='store_true', help='Launch in dark mode')
    group.add_argument('--light', action='store_true', help='Launch in light mode (default)')
    args = parser.parse_args()
    theme = 'dark' if args.dark else 'light'

    _enable_hidpi_macos()   # must run before Tk() is instantiated
    app = SONGSGUI(theme=theme)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure cleanup happens
        try:
            app._is_closing = True
            app.quit()
        except:
            pass
        try:
            app.destroy()
        except:
            pass

if __name__ == '__main__':
    main()