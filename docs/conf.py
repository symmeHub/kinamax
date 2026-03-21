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
author = "Ludovic Charleux, Cloé Léglise, Adrien Morel, and the kinamax contributors"
copyright = f"{datetime.now().year}, {author}"
release = kinamax.__version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "conf.py",
    "examples/batched_time_integration/models.py",
    "examples/batched_time_integration/run_steps_1_to_3.sh",
    "examples/batched_time_integration/wip/**",
    "examples/batched_time_integration/outputs/**",
    "examples/single_time_integration/README.md",
    "examples/single_time_integration/run.sh",
    "examples/single_time_integration/outputs/**",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".py": "myst-nb",
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

nb_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}
nb_execution_mode = "auto"
nb_execution_raise_on_error = True
nb_execution_timeout = 180

html_theme = "alabaster"
html_title = f"{project} {release}"
html_static_path = ["_static"]
