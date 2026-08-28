# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SONGS'
copyright = '2025, Arnab Lahiry'
author = 'Arnab Lahiry'
release = '1.0.1'

# Landing page (index.md) has no visible heading — the banner image serves as
# one — so the browser tab / window title is set explicitly here instead of
# being derived from a first-heading node.
html_title = 'SONGS'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))  # points to SONGS-spectral-cube-simulator/src

# Path for static files
html_static_path = ['_static']

# Logo in the top left of the sidebar. html_logo is the fallback used
# wherever a theme-specific variant isn't given (non-JS contexts, print);
# html_theme_options.logo below swaps to image_dark when dark mode is active.
html_logo = '../../assets/songs_icon.png'

html_theme_options = {
    "logo": {
        "image_light": "../../assets/songs_icon.png",
        "image_dark": "../../assets/songs_icon_dark.png",
    },
}

# Favicon: kept on the small square icon (not theme-swapped — Sphinx only
# supports a single html_favicon, and browsers don't apply prefers-color-scheme
# to favicons consistently anyway).
html_favicon = '../../assets/songs_icon.png'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # for Google/Numpy-style docstrings
    'sphinx.ext.viewcode',  # add links to highlighted source code
    'myst_parser',
    'sphinx.ext.mathjax',
]


myst_enable_extensions = [
    "amsmath",    # support for display math
    "dollarmath", # support for $...$ inline and $$...$$ display math
]

html_css_files = ['custom.css']

intersphinx_mapping = {
    "python": ("http://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


html_theme = 'sphinx_book_theme'