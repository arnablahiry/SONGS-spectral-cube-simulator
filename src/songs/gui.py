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

    # Baseline sits here (px from top of canvas).
    # Normal text hangs below it; superscripts rise above; subscripts drop further.
    BASELINE   = 13
    SUB_DROP   =  3   # extra pixels below baseline for subscript anchor
    SUP_LIFT   =  5   # pixels above baseline for superscript anchor
    CANVAS_H   = BASELINE + SUB_DROP + small_f.metrics("linespace") // 2 + 2

    # Measure total width
    total_w = 4
    for text, style in segments:
        total_w += (base_f if style == 'n' else small_f).measure(text)
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
        elif style == 'p':
            cv.create_text(x, BASELINE - SUP_LIFT, text=text,
                           font=small_f, fill=fg, anchor='sw')
            x += small_f.measure(text)

    return cv


# Import core
try:
    from .core import SONGSPhy, DEFAULT_DIFFUSE_PARAMS
except Exception:
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from songs.core import SONGSPhy, DEFAULT_DIFFUSE_PARAMS

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
        self.widget.configure(state="normal")
        self.widget.insert("end", string, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")

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
        self.title("Logs")
        self.text = tk.Text(self)
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("stderr", foreground="#e55b5b")
        # Redirect stdout and stderr
        sys.stdout = TextRedirector(self.text, "stdout")
        sys.stderr = TextRedirector(self.text, "stderr")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # Optionally restore stdout/stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()


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

    def __init__(self):
        super().__init__()
        self.title('SONGS GUI')
        self.WINDOW_HEIGHT = 780
        self._theme = 'dark'
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

        self._build_widgets()

        # Fix height to WINDOW_HEIGHT; width = logo + content (measured after layout).
        self.update_idletasks()
        total_w = self.winfo_reqwidth()
        self.geometry(f"{total_w}x{self.WINDOW_HEIGHT}")
        self.resizable(False, False)

        self.protocol('WM_DELETE_WINDOW', self._on_close)



    # ---------------------------
    # Theme helpers
    # ---------------------------
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
                self._logo_lbl = tk.Label(
                    self._logo_strip, image=self._logo_photo,
                    bg=bg, borderwidth=0)
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

    def _place_banner_btns(self):
        """Create (or recreate) the three overlay buttons on the logo label."""
        import webbrowser
        t = _THEMES[self._theme]
        # Destroy previous banner buttons if any
        for cv, _ in self._banner_btns:
            try: cv.destroy()
            except Exception: pass
        self._banner_btns.clear()

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

        # Place in a horizontal row, bottom of banner
        widths = [THEME_W, LINK_W, LINK_W]
        x = PAD
        for cv, w in zip(btns, widths):
            cv.place(x=x, y=-(PAD + BH), rely=1.0)
            x += w + GAP

    def _redraw_banner_btns(self):
        for _, fn in self._banner_btns:
            fn()

    def _toggle_theme(self):
        self._theme = 'light' if self._theme == 'dark' else 'dark'
        t = _THEMES[self._theme]
        self.configure(bg=t['BG'])
        self._root_frame.configure(bg=t['BG'])
        self._logo_strip.configure(bg=t['BG'])
        self._right_col.configure(bg=t['BG'])
        self._load_logo()
        self._place_banner_btns()
        self._rebuild_widgets()
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


    # ---------------------------
    # Button callback methods
    # ---------------------------
    def show_logs(self):
        if hasattr(self, 'log_window') and self.log_window.winfo_exists():
            self.log_window.lift()
        else:
            self.log_window = LogWindow(self)



    def _popup_figure(self, title, fig):
        """Utility to put a matplotlib figure into a new popup window"""
        new_win = tk.Toplevel(self)
        new_win.title(title)
        
        # Use the FigureCanvasTkAgg to embed the plot
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=new_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_moments(self):
        if not self.generator:
            return
        try:
            # Generate the figures using the 'Agg' backend (already set)
            fig0, _ = moment0(self.generator.results, idx=0, save=False)
            self._popup_figure("Moment 0", fig0)
            
            fig1, _ = moment1(self.generator.results, idx=0, save=False)
            self._popup_figure("Moment 1", fig1)
        except Exception as e:
            print(f"Error displaying moments: {e}")

    def show_spectra(self):
        if not self.generator:
            return
        try:
            fig, _ = spectrum(self.generator.results, idx=0, save=False)
            self._popup_figure("Integrated Spectrum", fig)
        except Exception as e:
            print(f"Error displaying spectrum: {e}")


    def show_slice(self):
        """Open the SONGS SliceViewer for the first generated cube."""
        if self.generator and self.generator.results:
            try:
                SliceViewer(self, self.generator.results, idx=0)
            except Exception as e:
                messagebox.showerror('Slice viewer error', str(e))

    def show_mom1(self):
        if self.generator:
            fig, ax = moment1(self.generator.results, idx=0, save=False)
            try: 
                import matplotlib
                matplotlib.use('TkAgg')
                plt.figure(fig.number)
                plt.show(block=False)
                matplotlib.use('Agg')
            except Exception: 
                pass

    '''def show_spectra(self):
        if self.generator:
            fig, ax = spectrum(self.generator.results, idx=0, save=False)
            try: 
                import matplotlib
                matplotlib.use('TkAgg')
                plt.figure(fig.number)
                plt.show(block=False)
                matplotlib.use('Agg')
            except Exception: 
                pass'''

    def reset_instance(self):
        """Reset the GUI to a fresh state and disable visualisation/save.

        This clears the in-memory ``self.generator`` reference so that the
        next generate action will create a new instance from current UI
        values. Buttons that depend on generated results are disabled.
        """
        # Disable all result buttons
        for _b in (self.moments_btn, self.spectra_btn, self.slice_btn,
                   self.save_btn, self.new_instance_btn):
            try:
                self._disable_btn(_b)
            except Exception:
                pass
        for child in self.winfo_children():
            if isinstance(child, tk.Toplevel):
                child.destroy()

        self.generator = None

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
        
        # ---------------------------
        # Variables
        # ---------------------------
        self.bmin_var = tk.DoubleVar(value=11.0)
        self.bmaj_var = tk.DoubleVar(value=13.0)
        self.bpa_var = tk.DoubleVar(value=20.0)
        self.spatial_resolution = tk.DoubleVar(value=3.8)
        self.n_var = tk.DoubleVar(value=1.0)
        self.hz_var = tk.DoubleVar(value=0.8)
        self.Se_var = tk.DoubleVar(value=0.1)
        self.sigma_v_var = tk.DoubleVar(value=40.0)
        self.fov = tk.DoubleVar(value=275.0)   # in kpc
        self.spectral_resolution = tk.IntVar(value=20)
        self.angle_x_var = tk.IntVar(value=45)
        self.angle_y_var = tk.IntVar(value=30)
        self.n_gals_var = tk.IntVar(value=3)

        # --- Diffuse-emission knobs (defaults pulled from core's DEFAULT_DIFFUSE_PARAMS) ---
        dp = DEFAULT_DIFFUSE_PARAMS
        # Halo
        self.halo_Se_factor_var = tk.DoubleVar(value=float(dp.get('halo_Se_factor', 0.065)))
        self.halo_Re_factor_var = tk.DoubleVar(value=float(dp.get('halo_Re_factor', 3.0)))
        self.halo_sigma_vz_var  = tk.DoubleVar(value=float(dp.get('halo_sigma_vz', 70.0)))
        # Bridges
        self.bridge_Se_factor_var          = tk.DoubleVar(value=float(dp.get('bridge_Se_factor', 0.05)))
        self.bridge_width_start_factor_var = tk.DoubleVar(value=float(dp.get('bridge_width_start_factor', 1.5)))
        self.bridge_width_end_factor_var   = tk.DoubleVar(value=float(dp.get('bridge_width_end_factor', 1.0)))
        # Tails / streamers
        self.tail_Se_factor_var     = tk.DoubleVar(value=float(dp.get('tail_Se_factor', 0.4)))
        # Streamer (channel-traversing trajectory) extras
        self.tail_vel_gradient_var          = tk.DoubleVar(value=float(dp.get('tail_vel_gradient', 0.5)))


        # New: satellite size fraction (max satellite-to-central ratio for Re,
        # hz, Se). Greyed out when only one galaxy is requested.
        self.sat_brightness_frac_var = tk.DoubleVar(value=0.15)

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
        _FONT_SM    = ("Helvetica", 10)
        _FONT_HDR   = ("Helvetica", 9, "bold")

        # Expose colors to make_slider via instance attrs
        self._slider_bg         = _CARD_BG
        self._slider_fg         = _TEXT
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
                outer.pack(side='left', fill='both', expand=True, padx=6, pady=6)
            inner = tk.Frame(outer, bg=_BIG_BG)
            inner.pack(fill='both', expand=True)
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
            inner = tk.Frame(outer, bg=_CARD_BG, padx=6, pady=10)
            inner.pack(fill='both', expand=True)
            if title:
                _lbl = tk.Label(inner, text=title, bg=_CARD_BG, fg=_TEXT,
                                font=_FONT_SM)
                _lbl._orig_fg = _TEXT
                _lbl.pack(anchor='w', pady=(0, 6))
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

        sc = small_card(bc1, "Number of galaxies")
        rb_frame = tk.Frame(sc, bg=_CARD_BG)
        rb_frame.pack(anchor='w', pady=(2, 4))

        # Pill-button constants
        _PB_W, _PB_H = 28, 22
        _PB_SEL_BG   = _ACCENT
        _PB_SEL_FG   = "#000000" if self._theme == 'dark' else "#ffffff"
        _PB_NOR_BG   = _t['PILL_NOR']
        _PB_NOR_FG   = _TEXT
        _PB_HOV_BG   = _t['PILL_HOV']
        _pill_canvases = []

        def _draw_pill(cv, selected, hover=False):
            cv.delete("all")
            fill = _PB_SEL_BG if selected else (_PB_HOV_BG if hover else _PB_NOR_BG)
            cv.create_rectangle(0, 0, _PB_W, _PB_H, fill=fill, outline=fill)
            fg = _PB_SEL_FG if selected else _PB_NOR_FG
            cv.create_text(_PB_W//2, _PB_H//2, text=cv._val_str,
                           fill=fg, font=("Helvetica", 9, "bold"))

        def _make_pill(parent, val):
            cv = tk.Canvas(parent, width=_PB_W, height=_PB_H,
                           bg=_CARD_BG, highlightthickness=0, bd=0, cursor='pointinghand')
            cv._val = val
            cv._val_str = str(val)
            _pill_canvases.append(cv)

            def _select():
                self.n_gals_var.set(val)

            def _on_enter(e):
                if self.n_gals_var.get() != val:
                    _draw_pill(cv, False, hover=True)

            def _on_leave(e):
                _draw_pill(cv, self.n_gals_var.get() == val)

            cv.bind("<ButtonRelease-1>", lambda e: _select())
            cv.bind("<Enter>", _on_enter)
            cv.bind("<Leave>", _on_leave)
            return cv

        def _refresh_pills(*_):
            sel = self.n_gals_var.get()
            for cv in _pill_canvases:
                _draw_pill(cv, cv._val == sel)

        for val in range(1, 7):
            cv = _make_pill(rb_frame, val)
            cv.pack(side='left', padx=2)

        _refresh_pills()
        self.n_gals_var.trace_add('write', _refresh_pills)

        sc = small_card(bc1, "Spatial Resolution [kpc/px]")
        slider_with_symbol(sc, [("Δ","n"),("X,Y","s")], self.spatial_resolution, 0.72, 9.0, resolution=0.01, fmt="{:.2f}")
        self.pix_scale_var_slider = sc.winfo_children()[-1]

        sc = small_card(bc1, "Spectral Resolution [km/s]")
        slider_with_symbol(sc, [("Δ","n"),("vz","s")], self.spectral_resolution, 5, 40, resolution=5, fmt="{:d}", integer=True)
        self.spec_slider = sc.winfo_children()[-1]

        sc = small_card(bc1, "Field of View [kpc]")
        slider_with_symbol(sc, [("N","n"),("kpc","s")], self.fov, 64.0, 512.0, resolution=1.0, fmt="{:.0f}")
        self.fov_slider = sc.winfo_children()[-1]

        sc = small_card(bc1, "Beam [kpc, kpc, deg]")
        for sym_segs, var, lo, hi, res in [
            ([("B","n"),("min","s")], self.bmin_var, 1.0, 30.0, 0.1),
            ([("B","n"),("maj","s")], self.bmaj_var, 1.0, 30.0, 0.1),
            ([("BPA","n")],          self.bpa_var,   0.0, 90.0, 1.0),
        ]:
            beam_row = tk.Frame(sc, bg=_CARD_BG)
            beam_row.pack(fill='x', pady=(0, 8))
            rich_label(beam_row, sym_segs, bg=_CARD_BG, fg=_t['SYM_FG']).pack(
                side='left', padx=(0, 4))
            sl = self.make_slider(beam_row, "", var, lo, hi, resolution=res, fmt="{:.1f}")
            sl.pack(side='left', fill='x', expand=True)

        # ── FOV preview canvas — forced square, same width as card ───────────
        import math as _math
        _PREV_BG = '#252520' if self._theme == 'dark' else '#ede8dc'

        _prev_cv = tk.Canvas(bc1, bg=_PREV_BG,
                             highlightthickness=2, highlightbackground=_ACCENT,
                             width=10, height=10)
        _prev_cv.pack(fill='x', padx=6, pady=(4, 8))

        # Track last width so Configure from height-change doesn't redraw twice
        _prev_last_w = [-1]

        def _draw_fov_preview(*_):
            S = _prev_cv.winfo_width()
            H = _prev_cv.winfo_height()
            if S < 20 or H < 20:
                return
            _prev_cv.delete('all')

            fov_kpc  = max(float(self.fov.get()), 1.0)
            res_kpc  = max(float(self.spatial_resolution.get()), 0.01)
            bmin_kpc = max(float(self.bmin_var.get()), 0.01)
            bmaj_kpc = max(float(self.bmaj_var.get()), 0.01)
            bpa_deg  = float(self.bpa_var.get())

            fov_px   = fov_kpc / res_kpc             # grid pixels for this resolution
            scale    = S / fov_px                    # canvas px per grid px
            bmin_px  = bmin_kpc / res_kpc            # beam minor in grid px
            bmaj_px  = bmaj_kpc / res_kpc            # beam major in grid px

            # ── beam ellipse + crosshairs (bottom-left, mirrors add_beam) ────
            marg   = max(int(S * 0.09), 10)
            semi_a = max(bmin_px * scale / 2, 2.0)  # minor semi-axis in canvas px
            semi_b = max(bmaj_px * scale / 2, 2.0)  # major semi-axis in canvas px
            ecx    = marg + semi_b + 2
            ecy    = S - marg - semi_b - 2

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
            _prev_cv.create_polygon(pts, outline=_ACCENT, fill=_PREV_BG,
                                    width=1, smooth=False)

            # Crosshairs: major dir = [cos(θ), sin(θ)], minor = [-sin(θ), cos(θ)]
            dx_maj =  semi_b * _math.cos(theta_rot)
            dy_maj =  semi_b * _math.sin(theta_rot)
            dx_min = -semi_a * _math.sin(theta_rot)
            dy_min =  semi_a * _math.cos(theta_rot)
            for dx, dy in [(dx_maj, dy_maj), (dx_min, dy_min)]:
                _prev_cv.create_line(ecx - dx, ecy - dy, ecx + dx, ecy + dy,
                                     fill=_ACCENT, width=1)

            # Beam pixel label
            _fnt = ('Helvetica', 9)
            _prev_cv.create_text(ecx + semi_b + 5, ecy,
                                 text=f"{bmin_px:.1f}×{bmaj_px:.1f} px",
                                 fill=_ACCENT, font=_fnt, anchor='w')

            # ── scalebar (top-right, always half the pixel FOV = S/2 wide) ───
            bar_px_grid = fov_px / 2    # grid pixels — changes with res
            bar_kpc_val = fov_kpc / 2   # kpc — changes with fov slider
            bar_cv_px   = S / 2         # always half the canvas

            bx1 = S - marg
            bx0 = bx1 - bar_cv_px
            by  = marg + 26

            for bx in (bx0, bx1):
                _prev_cv.create_line(bx, by - 5, bx, by + 5,
                                     fill=_ACCENT, width=1.5)
            _prev_cv.create_line(bx0, by, bx1, by, fill=_ACCENT, width=1.5)

            _prev_cv.create_text((bx0 + bx1) / 2, by - 13,
                                 text=f"{bar_px_grid:.0f} px",
                                 fill=_ACCENT, font=_fnt)
            _prev_cv.create_text((bx0 + bx1) / 2, by + 13,
                                 text=f"{bar_kpc_val:.0f} kpc",
                                 fill=_ACCENT, font=_fnt)

            # ── center watermark ─────────────────────────────────────────────
            _prev_cv.create_text(S / 2, S / 2,
                                 text="Spatial preview of\nthe Spectral Cube\n(in pixel units)",
                                 fill=_ACCENT, font=('Helvetica', 8),
                                 justify='center')

        def _on_prev_configure(e):
            w = e.width
            if w == _prev_last_w[0]:   # height-only change — skip
                return
            _prev_last_w[0] = w
            if w > 20:
                _prev_cv.configure(height=w)   # force square directly on canvas
            _prev_cv.after_idle(_draw_fov_preview)

        _prev_cv.bind('<Configure>', _on_prev_configure)
        for _pv in (self.fov, self.spatial_resolution,
                    self.bmin_var, self.bmaj_var, self.bpa_var):
            _pv.trace_add('write', _draw_fov_preview)
        _prev_cv.after(150, _draw_fov_preview)

        # ──────────────────────────────────────────────────────────────────────
        # MIDDLE COLUMN: Central Galaxy (top) + Satellite Properties (below)
        # ──────────────────────────────────────────────────────────────────────
        mid_col = tk.Frame(cards_row, bg=_BG)
        mid_col.pack(side='left', fill='both', expand=True)

        bc2 = big_card(mid_col, "Central Galaxy Properties", stack=True)

        sc = small_card(bc2, "Sérsic index")
        slider_with_symbol(sc, [("n","n")], self.n_var, 0.5, 1.5, resolution=0.01, fmt="{:.3f}")
        self.n_slider = sc.winfo_children()[-1]

        sc = small_card(bc2, "Scale height [kpc]")
        slider_with_symbol(sc, [("h","n"),("z","s")], self.hz_var, 0.4, 9.0, resolution=0.01, fmt="{:.3f}")
        self.hz_slider = sc.winfo_children()[-1]

        sc = small_card(bc2, "Surface brightness [Jy]")
        slider_with_symbol(sc, [("S","n"),("e","s")], self.Se_var, 0.01, 1.0, resolution=0.01, fmt="{:.3f}")
        self.Se_slider = sc.winfo_children()[-1]

        sc = small_card(bc2, "Velocity dispersion [km/s]")
        slider_with_symbol(sc, [("σ","n"),("vz","s")], self.sigma_v_var, 30.0, 60.0, resolution=0.1, fmt="{:.1f}")
        self.sigma_slider = sc.winfo_children()[-1]

        sc = small_card(bc2, "X & Y Inclination Angles [deg, deg]")
        for _segs, _var, _attr in [
            ([("θ","n"),("X","s")], self.angle_x_var, 'angle_x_slider'),
            ([("φ","n"),("Y","s")], self.angle_y_var, 'angle_y_slider'),
        ]:
            _arow = tk.Frame(sc, bg=_CARD_BG)
            _arow.pack(fill='x', pady=(0, 8))
            rich_label(_arow, _segs, bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
            _sl = self.make_slider(_arow, "", _var, 0, 359, resolution=1, fmt="{:d}", integer=True)
            _sl.pack(side='left', fill='x', expand=True)
            setattr(self, _attr, _arow)

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 3: Satellite & Halo  (stacked under Central Galaxy)
        # ──────────────────────────────────────────────────────────────────────
        bc3 = big_card(mid_col, "Satellite Properties", stack=True, expand=True)
        self._bc3 = bc3  # kept for greyout

        sc = small_card(bc3, "Satellite flux fraction")
        slider_with_symbol(sc, [("f","n"),("sat","s")], self.sat_brightness_frac_var, 0.0, 0.5, resolution=0.01, fmt="{:.2f}")
        self.sat_frac_slider_frame = sc.winfo_children()[-1]
        self.sat_frac_scale = find_scale(self.sat_frac_slider_frame)

        self.sat_offset_max_var = tk.DoubleVar(value=180.0)
        self.sat_offset_min_var = tk.DoubleVar(value=80.0)

        sc = small_card(bc3, "Min & Max Offset from Central [kpc, kpc]")
        for _segs, _var, _f_attr, _s_attr, _lo, _hi in [
            ([("d","n"),("min","s")], self.sat_offset_min_var,
             'sat_offset_min_frame', 'sat_offset_min_scale', 0.0,  333.0),
            ([("d","n"),("max","s")], self.sat_offset_max_var,
             'sat_offset_max_frame', 'sat_offset_max_scale', 10.0, 500.0),
        ]:
            _orow = tk.Frame(sc, bg=_CARD_BG)
            _orow.pack(fill='x', pady=(0, 8))
            rich_label(_orow, _segs, bg=_CARD_BG, fg=_t['SYM_FG']).pack(side='left', padx=(0, 4))
            _sl = self.make_slider(_orow, "", _var, _lo, _hi, resolution=5.0, fmt="{:.0f}")
            _sl.pack(side='left', fill='x', expand=True)
            setattr(self, _f_attr, _orow)
            setattr(self, _s_attr, find_scale(_orow))

        # ──────────────────────────────────────────────────────────────────────
        # BIG CARD 4: Diffuse Features (halo + bridge + streamers)
        # ──────────────────────────────────────────────────────────────────────
        bc4 = big_card(cards_row, "Diffuse Features")

        sc = small_card(bc4, "Halo flux amplitude")
        slider_with_symbol(sc, [("S","n"),("e,halo","s"),(" / S","n"),("e,c","s")],
                           self.halo_Se_factor_var, 0.0, 0.3, resolution=0.005, fmt="{:.3f}")
        self.halo_Se_slider = sc.winfo_children()[-1]

        sc = small_card(bc4, "Halo effective radius")
        slider_with_symbol(sc, [("R","n"),("e,halo","s"),(" / R","n"),("e,c","s")],
                           self.halo_Re_factor_var, 1.0, 5.0, resolution=0.1, fmt="{:.1f}")
        self.halo_Re_slider = sc.winfo_children()[-1]

        sc = small_card(bc4, "Halo vel. dispersion [km/s]")
        slider_with_symbol(sc, [("σ","n"),("vz,halo","s")],
                           self.halo_sigma_vz_var, 0.0, 150.0, resolution=5.0, fmt="{:.0f}")
        self.halo_sigma_slider = sc.winfo_children()[-1]

        sc = small_card(bc4, "Bridge flux amplitude")
        slider_with_symbol(sc, [("S","n"),("e,br","s"),(" / S","n"),("e,s","s")],
                           self.bridge_Se_factor_var, 0.0, 0.3, resolution=0.005, fmt="{:.3f}")
        self.bridge_Se_slider = sc.winfo_children()[-1]

        sc = small_card(bc4, "Bridge width (halo end)")
        slider_with_symbol(sc, [("σ","n"),("br,h","s"),(" / R","n"),("e,c","s")],
                           self.bridge_width_start_factor_var, 0.5, 4.0, resolution=0.1, fmt="{:.1f}")
        self.bridge_w0_slider = sc.winfo_children()[-1]

        sc = small_card(bc4, "Bridge width (sat. end)")
        slider_with_symbol(sc, [("σ","n"),("br,s","s"),(" / R","n"),("e,s","s")],
                           self.bridge_width_end_factor_var, 0.3, 3.0, resolution=0.1, fmt="{:.1f}")
        self.bridge_w1_slider = sc.winfo_children()[-1]

        sc = small_card(bc4, "Streamer flux amplitude")
        slider_with_symbol(sc, [("S","n"),("e,tail","s"),(" / S","n"),("e,s","s")],
                           self.tail_Se_factor_var, 0.0, 1.0, resolution=0.02, fmt="{:.2f}")
        self.tail_Se_slider = sc.winfo_children()[-1]

        sc = small_card(bc4, "Streamer velocity scale")
        slider_with_symbol(sc, [(" × Δv","n"),("sys","s")],
                           self.tail_vel_gradient_var, 0.0, 2.0, resolution=0.05, fmt="{:.2f}",
                           symbol_side='right')
        self.tail_vel_grad_slider = sc.winfo_children()[-1]

        # ── Satellite-dependent greyout ──────────────────────────────────────
        def _update_min_range(*args):
            new_upper = max(5.0, float(self.sat_offset_max_var.get()) / 1.5)
            self.sat_offset_min_scale.configure(to=new_upper)
            if float(self.sat_offset_min_var.get()) > new_upper:
                self.sat_offset_min_var.set(round(new_upper / 5) * 5)
        self.sat_offset_max_var.trace_add('write', _update_min_range)
        _update_min_range()

        def _compute_max_offset_kpc():
            res_kpc = max(float(self.spatial_resolution.get()), 0.01)
            fov_px  = max(int(self.fov.get()), 4)
            gs      = max(fov_px // res_kpc, 4)
            igs     = int((31 / 64) * gs)
            if igs % 2 != 0:
                igs -= 1
            half    = igs // 2
            center  = (gs + 1) // 2
            max_off_px = max(min(gs - half - 1 - center, center - half) - 1, 1)
            return max_off_px * res_kpc

        def _update_offset_range(*_args):
            cap = _compute_max_offset_kpc()
            new_max_upper = float(cap)
            new_min_upper = max(1.0, new_max_upper / 1.5)
            self.sat_offset_max_scale.configure(to=new_max_upper)
            self.sat_offset_min_scale.configure(to=new_min_upper)
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
            active = self.n_gals_var.get() > 1
            state  = tk.NORMAL if active else tk.DISABLED
            dim_fg = "#2a2a1a"   # dimmed text — keep bg unchanged so design stays intact

            def _set_state(w):
                # Only dim foreground text when inactive; leave backgrounds alone.
                try:
                    w.configure(state=state)
                except Exception: pass
                if not active:
                    try: w.configure(fg=dim_fg)
                    except Exception: pass
                else:
                    # Restore fg — use _orig_fg if stored, otherwise infer by type
                    if isinstance(w, tk.Scale):
                        try: w.configure(fg=_TEXT, bg=_ACCENT, activebackground="#f0c040")
                        except Exception: pass
                    elif isinstance(w, tk.Label):
                        orig = getattr(w, '_orig_fg', None)
                        try: w.configure(fg=orig if orig else _TEXT)
                        except Exception: pass
                for child in w.winfo_children():
                    _set_state(child)

            _set_state(self._bc3)

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
                    hov=None, disabled=False, font=("Helvetica", 10, "bold")):
            bg  = bg  or _btn_nor_bg
            fg  = fg  or _btn_nor_fg
            hov = hov or _btn_nor_hov
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

        def _enable_btn(lbl, bg=None, hov=None):
            bg  = bg  or lbl._btn_bg
            hov = hov or lbl._btn_hov
            lbl.configure(bg=bg, fg=_btn_nor_fg, cursor='pointinghand')
            lbl._disabled = False
            lbl.bind('<Enter>', lambda e, b=lbl, h=hov: b.configure(bg=h))
            lbl.bind('<Leave>', lambda e, b=lbl, n=bg:  b.configure(bg=n))
            lbl.bind('<ButtonRelease-1>', lambda e: lbl._btn_cmd())

        def _disable_btn(lbl):
            lbl.configure(fg=_btn_dis_fg, cursor='arrow')
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
        self.slice_btn        = _mk_btn(btn_frame, 'Slice',    self.show_slice,    disabled=True)
        self.moments_btn      = _mk_btn(btn_frame, 'Moments',  self.show_moments,  disabled=True)
        self.spectra_btn      = _mk_btn(btn_frame, 'Spectrum', self.show_spectra,  disabled=True)
        self.save_btn         = _mk_btn(btn_frame, 'Save',     self.save_sim,      disabled=True)
        self.new_instance_btn = _mk_btn(btn_frame, 'Reset',    self.reset_instance, disabled=True)



       

        # Auto-create/refresh generator when variables change (fast preview)
        def _auto_update_generator(*args):
            try:
                self.create_generator()
            except Exception as e:
                print("Auto-create generator failed:", e)

        for var in [self.bmin_var, self.bmaj_var, self.bpa_var, self.spatial_resolution, self.n_var,
                    self.hz_var, self.Se_var, self.sigma_v_var, self.fov,
                    self.spectral_resolution, self.angle_x_var, self.angle_y_var,
                    self.sat_brightness_frac_var, self.sat_offset_min_var, self.sat_offset_max_var,
                    # Diffuse-emission knobs
                    self.halo_Se_factor_var, self.halo_Re_factor_var,
                    self.halo_sigma_vz_var,
                    self.bridge_Se_factor_var, self.bridge_width_start_factor_var,
                    self.bridge_width_end_factor_var,
                    self.tail_Se_factor_var,
                    self.tail_vel_gradient_var]:
            if hasattr(var, 'trace_add'):
                var.trace_add('write', _auto_update_generator)
            else:
                var.trace('w', _auto_update_generator)


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
        spatial_resolution = int(self.spatial_resolution.get())
        central_n = float(self.n_var.get())
        central_hz = float(self.hz_var.get())
        central_Se = float(self.Se_var.get())
        central_gal_x_angle = int(self.angle_x_var.get())
        central_gal_y_angle = int(self.angle_y_var.get())
        offset_gals = (float(self.sat_offset_min_var.get()), float(self.sat_offset_max_var.get()))
        sigma_v = float(self.sigma_v_var.get())

        # Create per-galaxy lists. For a single galaxy we keep the
        # specified central values. For multiple galaxies we generate
        # satellite properties using simple random draws so the
        # generator receives arrays of length ``n_gals`` (primary + satellites).
        all_Re = [5/spatial_resolution]
        all_hz = [central_hz]
        all_Se = [central_Se]
        all_gal_x_angles = [central_gal_x_angle]
        all_gal_y_angles = [central_gal_y_angle]
        all_n = [central_n]

        if n_gals > 1:
            n_sat = n_gals - 1
            rng = np.random.default_rng()

            # Satellites are physically smaller (Re, hz fixed ratio).
            sat_Re = list(rng.uniform(all_Re[0] / 3, all_Re[0] / 2, n_sat))
            sat_hz = list(rng.uniform(all_hz[0] / 3, all_hz[0] / 2, n_sat))

            # sat_brightness_frac scales Se relative to central, compensating for
            # the smaller satellite Re so that frac=1 ≈ similar surface brightness.
            _b = float(np.clip(self.sat_brightness_frac_var.get(), 0.0, 2.0))

            # Random Sérsic indices for satellites
            sat_n = list(rng.uniform(0.5, 1.5, n_sat))

            sat_Se = [
                float(all_Se[0] * _b * (all_Re[0] / re_sat) ** 2
                      * rng.uniform(0.85, 1.15))
                for re_sat in sat_Re
            ]

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
            'tail_Se_factor': float(self.tail_Se_factor_var.get()),
            # Streamer knobs
            'tail_vel_gradient': float(self.tail_vel_gradient_var.get()),
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

        self.generator = g


    def _run_generate(self):
        # Disable garbage collection in this thread to prevent cleanup
        # of Tkinter objects from the wrong thread
        import gc
        gc_was_enabled = gc.isenabled()
        gc.disable()
        
        try:
            # Check if closing before doing expensive work
            if self._is_closing:
                return
                
            # Auto-show log window
            if hasattr(self, 'log_window') and self.log_window.winfo_exists():
                self.log_window.deiconify()
                self.log_window.lift()
            else:
                self.log_window = LogWindow(self)

            try:
                results = self.generator.generate_cubes()
                # Check again before scheduling UI updates
                if self._is_closing:
                    return
                # Enable buttons on main thread
                def _enable_all():
                    for _b in (self.moments_btn, self.spectra_btn, self.slice_btn,
                               self.save_btn, self.new_instance_btn):
                        try: self._enable_btn(_b)
                        except Exception: pass
                self.after(0, _enable_all)
            except Exception as e:
                if not self._is_closing:
                    self.after(0, lambda e=e: messagebox.showerror('Error during generation', str(e)))
        finally:
            # Re-enable garbage collection if it was enabled
            if gc_was_enabled:
                gc.enable()
    
    
    def generate(self):
        import random as _random
        self._pending_seed = _random.randint(0, 2**31 - 1)
        print(f"[SONGS] Instance seed: {self._pending_seed}")
        self.create_generator()

        if self.generator is None:
            return

        t = threading.Thread(target=self._run_generate, daemon=True)
        t.start()

    # ---------------------------
    # Save simulation (cube + params)
    # ---------------------------
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
                import h5py
                with h5py.File(fname, 'w') as f:
                    f.create_dataset('cube', data=cube)
                    g = f.create_group('galaxies')
                    if meta is not None and 'per_galaxy_cubes' in meta:
                        try:
                            g.create_dataset('cubes', data=np.array(meta['per_galaxy_cubes']))
                        except Exception:
                            pass
                    if meta is not None and 'galaxy_centers' in meta:
                        try:
                            g.create_dataset('positions_xyz_px', data=np.array(meta['galaxy_centers']))
                        except Exception:
                            pass
                    n_gals = int(params['n_gals'])
                    types = np.array(['central'] + ['satellite'] * (n_gals - 1), dtype='S10')
                    g.create_dataset('types', data=types)
                    g.create_dataset('Re_px', data=np.asarray(params['all_Re']))
                    g.create_dataset('Se', data=np.asarray(params['all_Se']))
                    g.create_dataset('hz_px', data=np.asarray(params['all_hz']))
                    f.attrs['n_gals'] = n_gals
                    f.attrs['n_satellites'] = n_gals - 1
                    f.attrs['spatial_resolution_kpc_per_px'] = float(params['spatial_resolution'])
                    f.attrs['spectral_resolution_km_s'] = float(params['spectral_resolution'])
                    f.attrs['fov_kpc'] = float(params['fov'])
                    dp_grp = f.create_group('diffuse_params')
                    for k, v in params.get('diffuse_params', {}).items():
                        try:
                            dp_grp.attrs[k] = v
                        except Exception:
                            pass
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
    _enable_hidpi_macos()   # must run before Tk() is instantiated
    app = SONGSGUI()
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