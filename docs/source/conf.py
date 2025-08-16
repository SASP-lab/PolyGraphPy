import os
import sys

sys.path.insert(0, os.path.abspath('../..'))  # Points to PolyGraphPy root

# -- Project information -----------------------------------------------------
project = 'PolyGraphPy'
copyright = '2025, Joao Gabriel Duarte'
author = 'Joao Gabriel Duarte'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    'myst_parser',
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autosummary_generate = True