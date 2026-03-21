# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Step 2: Detect Orbits From The Converged Attractors
#
# This notebook reads the attractor table generated in Step 1 and groups the
# converged attractor samples into orbit labels.
#
# Two parquet files are produced:
#
# - `outputs/orbits.parquet` with attractor and orbit metadata,
# - `outputs/sim_orbit.parquet` mapping each simulation to one orbit label.

# %%
from pathlib import Path

import polars as pl

from kinamax.core import detect_orbits
from models import H46_EM_Problem

problem_class = H46_EM_Problem


# %% [markdown]
# ## Load The Batched Simulation Table
#
# This file is the output of Step 1. The attractor-state labels are the original
# state names with an `a` suffix because they correspond to attractor samples
# rather than initial conditions.

# %%
example_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
working_dir = example_dir / "outputs"

simulations = pl.read_parquet(working_dir / "simulations.parquet")
state_vec_labels = problem_class.state_vector_labels
attractor_state_vec_labels = [f"{k}a" for k in state_vec_labels]
ode_params_labels = problem_class.params_labels


# %% [markdown]
# ## Cluster Attractors Into Orbits
#
# `detect_orbits(...)` groups simulations that converge to the same attractor
# sequence, up to cyclic permutation along the orbit.

# %%
attractors, sim_orbit = detect_orbits(
    problem_class=problem_class,
    simulations=simulations,
    ode_params_labels=ode_params_labels,
    attractor_state_vec_labels=attractor_state_vec_labels,
    state_vec_labels=state_vec_labels,
)


# %% [markdown]
# ## Clean The Energy-Like States And Save
#
# Orbit detection focuses on the dynamical variables. The accumulated energy
# states are therefore reset to zero before storing the orbit representatives.

# %%
attractors = attractors.with_columns(
    Ev=pl.lit(0.0),
    Eh=pl.lit(0.0),
)
attractors.write_parquet(working_dir / "orbits.parquet")
sim_orbit.write_parquet(working_dir / "sim_orbit.parquet")

attractors.head()
