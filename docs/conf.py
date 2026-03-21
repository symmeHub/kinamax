"""Sphinx configuration for the kinamax documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import kinamax

project = "kinamax"
author = "kinamax contributors"
copyright = f"{datetime.now().year}, {author}"
release = kinamax.__version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

html_theme = "alabaster"
html_title = f"{project} {release}"
html_static_path = ["_static"]
