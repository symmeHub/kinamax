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
# # Single Time Integration Until Convergence
#
# This tutorial shows the smallest complete attractor-search workflow in
# `kinamax`.
#
# It mirrors the batched example, but restricts the computation to one problem
# instance and one initial condition:
#
# - one driven oscillator, `kinamax.integration.models.H46Problem`
# - one drive frequency, `fd`
# - one initial state, `X0`
#
# The goal is to make each moving part visible before introducing batching and
# orbit clustering.

# %% [markdown]
# ## What The Script Does
#
# The workflow has four stages:
#
# 1. define the physical problem and the attractor-finder settings,
# 2. generate one initial condition,
# 3. integrate until one candidate subharmonic satisfies the shooting residual
#    tolerance,
# 4. flatten the structured result into a table that is easy to inspect and save.
#
# Because this file uses Jupytext percent format, it remains a normal Python
# script while also behaving like a notebook in the documentation.

# %%
from __future__ import annotations

from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import polars as pl
from diffrax import PIDController, Tsit5

from kinamax.integration.core import (
    AttractorFinder,
    AttractorFinderConfig,
    post_process_attractor_finder_results,
)
from kinamax.integration.models import H46Problem

config.update("jax_enable_x64", True)  # Use double precision for improved accuracy


# %% [markdown]
# ## Set Up The Output Directory
#
# The example writes its results next to the source file so the generated
# parquet file stays attached to the tutorial.

# %%
example_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
output_dir = example_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## Define The Physical Problem
#
# In this first parameter block we choose the forcing frequency and instantiate
# one H46 oscillator. The problem namespace also exposes the state ordering,
# which is reused later when we format the output table.

# %%
fd = 50.0
problem_class = H46Problem
state_vector_labels = problem_class.state_vector_labels
problem = H46Problem.Params(fd=jnp.array(fd), Ad=jnp.array(2.5))


# %% [markdown]
# ## Define The Attractor-Finder Settings
#
# The attractor search has two levels of configuration:
#
# - `AttractorFinderConfig` controls the integration start time, initial time
#   step, target frequency, and convergence threshold.
# - `AttractorFinder.Params` controls the overall shooting strategy:
#   how many residual samples are used per period, which subharmonics are tested,
#   and how much total integration effort is allowed.

# %%
finder_config = AttractorFinderConfig(
    convergence_tol=jnp.array(1.0e-10),
    target_frequency=fd,
    init_time=jnp.array(0.0),
    init_time_step=jnp.array(1.0e-3),
    subharmonic_factor=10.0,
)

solver = Tsit5()
controller = PIDController(rtol=1e-8, atol=1e-9)
target_subharmonics = np.array([1, 2, 3, 5], dtype=int)
attractor_finder = AttractorFinder.Params(
    residuals_per_period=np.array(20, int),
    targetted_subharmonics=target_subharmonics,
    max_periods=2000,
    controller=controller,
    solver=solver,
)


# %% [markdown]
# ## Generate One Initial Condition
#
# The state is sampled in a box scaled by the characteristic displacement and
# velocity of the oscillator. The last state corresponds to cumulative energy,
# so it starts from zero.

# %%
key = jax.random.PRNGKey(758493)
init_condition = (
    (jax.random.uniform(key, shape=(len(state_vector_labels),)) - 0.5)
    * 2.0
    * jnp.array([5.0 * problem.xw, 10.0 * problem.xw * problem.w0, 0.0])
)


# %% [markdown]
# ## Integrate Until Convergence
#
# `AttractorFinder.find_attractors(...)` returns the original inputs together
# with a structured solution object. The call is JIT-compiled here because even
# a single search performs repeated ODE solves internally.

# %%
find_one = jax.jit(
    lambda p, x0, cfg: AttractorFinder.find_attractors(
        attractor_finder, problem_class, p, x0, cfg
    )
)
problem, finder_config, init_condition, solution = find_one(
    problem, init_condition, finder_config
)


# %% [markdown]
# ## Why The Post-Processing Step Matters
#
# `AttractorFinder.find_attractors(...)` returns structured arrays designed for
# computation. `post_process_attractor_finder_results(...)` converts that result
# into a tidy table with one row per detected attractor sample and with the
# problem/configuration metadata duplicated alongside it. That table is much
# easier to save, filter, plot, and compare across examples.

# %%
processed = post_process_attractor_finder_results(
    problem_class=problem_class,
    problems=problem,
    finder_configs=finder_config,
    init_conditions=init_condition,
    solutions=solution,
    target_subharmonics=target_subharmonics,
    solution_state_labels=[lab + "a" for lab in state_vector_labels],
)

out_path = output_dir / "simulations.parquet"
processed.write_parquet(out_path)
pl.Config.set_tbl_rows(8)
print(processed)
print(f"Wrote: {out_path}")
