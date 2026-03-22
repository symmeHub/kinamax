"""Sphinx configuration for the kinamax documentation."""

from __future__ import annotations

import sys
import doctest
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
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "conf.py",
    "examples/integration/batched_time_integration/run_steps_1_to_3.sh",
    "examples/integration/batched_time_integration/wip/**",
    "examples/integration/batched_time_integration/outputs/**",
    "examples/integration/single_time_integration/README.md",
    "examples/integration/single_time_integration/run.sh",
    "examples/integration/single_time_integration/outputs/**",
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
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]

nb_custom_formats = {
    ".py": ["jupytext.reads", {"fmt": "py:percent"}],
}
is_doctest_build = (
    ("-b" in sys.argv and sys.argv[sys.argv.index("-b") + 1] == "doctest")
    or ("-M" in sys.argv and sys.argv[sys.argv.index("-M") + 1] == "doctest")
)
nb_execution_mode = "off" if is_doctest_build else "auto"
nb_execution_raise_on_error = True
nb_execution_timeout = 180

doctest_default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
doctest_global_setup = """
import jax.numpy as jnp
from diffrax import PIDController, Tsit5

from kinamax.hbm import (
    FourierCoeffs,
    SampledSignal,
    coeffs_derivative,
    coeffs_to_time_signal,
    time_to_coeffs,
)
from kinamax.integration.core import (
    AttractorFinder,
    AttractorFinderConfig,
    post_process_attractor_finder_results,
)
from kinamax.integration.models import H46Problem, H46_EM_Problem
"""

html_theme = "alabaster"
html_title = f"{project} {release}"
html_static_path = ["_static"]
