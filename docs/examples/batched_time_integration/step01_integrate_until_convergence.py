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
# # Step 1: Integrate A Batched Frequency Sweep Until Convergence
#
# This notebook is the batched counterpart of the single-point tutorial.
#
# Instead of solving one problem from one initial condition, it builds:
#
# - a frequency sweep,
# - a cloud of initial conditions,
# - a vectorized attractor search over both axes.
#
# The output is a flat table of converged attractor samples saved to
# `outputs/simulations.parquet`.

# %%
from pathlib import Path
from jax import numpy as jnp
import jax
from jax import config, vmap
from diffrax import Tsit5, PIDController
import numpy as np
import polars as pl
import equinox as eqx
from kinamax.core import (
    AttractorFinder,
    AttractorFinderConfig,
    post_process_attractor_finder_results,
)

# from kinamax.problems import H46Problem
from models import H46_EM_Problem

config.update("jax_enable_x64", True)  # Use double precision for improved accuracy

problem_class = H46_EM_Problem


# %% [markdown]
# ## Build A Two-Axis Batched Attractor Finder
#
# The scalar routine `AttractorFinder.find_attractors(...)` works on one problem
# and one initial condition at a time.
#
# Here we wrap it twice with `vmap`:
#
# - the inner `vmap` scans over the starting states,
# - the outer `vmap` scans over the frequency-dependent problem instances.
#
# The result is then JIT-compiled once with Equinox.

# %%
def build_batched_finder(find_attractors_fn):
    """Return a JIT-compiled attractor finder that runs on a batch of problems.

    The raw attractor-finder routine expects a single dynamical
    system, a single initial condition, and a single configuration. When exploring
    basins of attraction, we typically want to evaluate many initial conditions for
    the same problem (or multiple frequency points) and execute the search on
    accelerator hardware. This helper function wraps the user provided
    ``find_attractors_fn`` in two layers of `jax.vmap` to broadcast over:

    - the ``H46Problem`` instances that capture the drive frequency sweep,
    - the initial conditions to seed the integration,
    - the ``AttractorFinderConfig`` objects that contain per-frequency metadata.

    Once vectorised, the whole routine is `jax.jit` compiled through Equinox
    (`eqx.filter_jit`) so that arrays are traced efficiently and PyTrees coming
    from dataclasses are handled automatically.

    Parameters
    ----------
    find_attractors_fn:
        Callable matching the signature of
        ``AttractorFinder.find_attractors``.

    Returns
    -------
    Callable
        A compiled function that simultaneously integrates every combination of
        problem, initial condition, and configuration provided.
    """

    return eqx.filter_jit(
        vmap(
            # Inner vmap: for one problem/frequency, scan over all initial states.
            vmap(find_attractors_fn, in_axes=(None, 0, None)),
            in_axes=(
                # Outer vmap: advance problem parameters and matching configuration
                # together across the frequency sweep.
                problem_class(
                    fd=0,
                    xw=None,
                    Q=None,
                    Ad=None,
                    w0=None,
                    alpha=None,
                    C0=None,
                    R=None,
                    L=None,
                    M=None,
                ),
                None,
                AttractorFinderConfig(
                    init_time=None,
                    init_time_step=None,
                    convergence_tol=None,
                    target_frequency=0,
                    subharmonic_factor=None,
                ),
            ),
        )
    )


# %% [markdown]
# ## Save Helper
#
# The integration stage writes one parquet file that the next notebooks will
# consume.

# %%
def save_results(
    data: pl.DataFrame,
    output_dir: Path,
    filename: str = "simulations.parquet",
):
    """Persist processed attractor finder results to disk.

    Parameters
    ----------
    data:
        Polars DataFrame containing the flattened attractor metadata and summary
        statistics produced by :func:`post_process_attractor_finder_results`.
    output_dir:
        Directory (relative to this script) where the parquet file will be written.
        The directory is created automatically if it does not exist.
    filename:
        Name of the parquet file. Use the default when following this tutorial so
        the downstream notebooks can discover the expected artifact.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(str(output_dir / filename))


# %% [markdown]
# ## Choose The Sweep And Finder Parameters
#
# The frequency axis and the attractor-finder settings are broadcast together.
# Every frequency point gets its own problem parameters and matching
# `AttractorFinderConfig`.

# %%
example_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
output_dir = example_dir / "outputs"

fd = jnp.linspace(20, 50.0, 11)
finder_config = AttractorFinderConfig(
    convergence_tol=1.0e-10,
    target_frequency=fd,
    init_time=0.0,
    init_time_step=1.0e-3,
    subharmonic_factor=10,
)
solver = Tsit5()
controller = PIDController(rtol=1e-8, atol=1e-9)
target_subharmonics = np.array([1, 2, 3, 5], dtype=int)
attractor_finder = AttractorFinder.Params(
    residuals_per_period=20,
    targetted_subharmonics=target_subharmonics,
    max_periods=2000,
    controller=controller,
    solver=solver,
)


# %% [markdown]
# ## Generate The Batched Problems And Initial Conditions
#
# The electromechanical model has five states. Each row of `init_conditions`
# defines one starting point, and each column matches one state component of the
# model.

# %%
problem = problem_class(fd=fd, Ad=2.5)
key = jax.random.PRNGKey(758493)
Nstart = 20
init_conditions = (
    (jax.random.uniform(key, shape=(Nstart, 5)) - 0.5)
    * 2.0
    * jnp.array([5.0 * problem.xw, 10.0 * problem.xw * problem.w0, 1.0, 0.0, 0.0])
)


# %% [markdown]
# ## Run The Batched Attractor Search
#
# The returned arrays keep the full batched structure:
# frequency point, initial condition, and attractor sample along the detected
# orbit.

# %%
batched_find = build_batched_finder(
    lambda problem, x0, cfg: AttractorFinder.find_attractors(
        attractor_finder, problem, x0, cfg
    )
)
problems, finder_configs, vmaped_init, solutions = batched_find(
    problem,
    init_conditions,
    finder_config,
)


# %% [markdown]
# ## Flatten And Save The Result
#
# `post_process_attractor_finder_results(...)` turns the batched PyTree into one
# tidy table that downstream steps can consume directly.

# %%
processed = post_process_attractor_finder_results(
    problem_class=problem_class,
    problems=problems,
    finder_configs=finder_configs,
    init_conditions=vmaped_init,
    solutions=solutions,
    target_subharmonics=target_subharmonics,
    solution_state_labels=[lab + "a" for lab in problem_class.state_vector_labels],
)
save_results(processed, output_dir=output_dir)
processed
