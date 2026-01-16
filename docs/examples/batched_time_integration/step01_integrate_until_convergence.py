"""Time integration tutorial for the H46 problem.

This script demonstrates how to:

1. Construct a driven oscillator (`H46Problem`) with multiple start points and a frequency sweep.
2. Configure an `AttractorFinder` that integrates trajectories until they converge.
3. Vectorise the attractor search across both initial conditions and frequency values.
4. Persist the processed results to disk for downstream analysis.

Each function below contains detailed docstrings so the file can be read top to
bottom as a self-contained tutorial. The code is intentionally explicit about
what gets configured and why, mirroring the steps required when adapting the
workflow to a new problem.
"""

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


def build_batched_finder(find_attractors_fn):
    """Return a JIT-compiled attractor finder that runs on a batch of problems.

    The raw ``AttractorFinder.find_attractors`` routine expects a single dynamical
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
        Callable matching the signature of ``AttractorFinder.find_attractors``.

    Returns
    -------
    Callable
        A compiled function that simultaneously integrates every combination of
        problem, initial condition, and configuration provided.
    """

    return eqx.filter_jit(
        vmap(
            vmap(find_attractors_fn, in_axes=(None, 0, None)),
            in_axes=(
                problem_class(fd=0, xw=None, Q=None, Ad=None, w0=None, alpha=None, C0=None, R=None, L=None, M=None),
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


def save_results(
    data: pl.DataFrame,
    output_dir: str = "outputs",
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

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    data.write_parquet(str(target_dir / filename))


def main():
    """Execute the full integration workflow for the tutorial example.

    The steps are deliberately spelled out to emphasise how each component fits
    together:

    1. Define a single-point frequency sweep and populate the attractor finder
       configuration with tolerances and integration spacing.
    2. Instantiate the `Tsit5` solver and `PIDController` that control the adaptive
       time stepping during integration.
    3. Sample a grid of random initial conditions inside a bounding box scaled by
       the problem's characteristic width (`xw`) so that we explore a range of
       possible basins.
    4. Call :func:`build_batched_finder` to obtain a vectorised attractor finder,
       run it, and post-process the results into a tidy table.
    5. Save the tidy table to ``docs/examples/time_integration/outputs`` so the
       companion tutorial in the documentation can visualise the attractors.
    """
    
    fd = jnp.linspace(10.0, 60.0, 51) # Frequency sweep (Hz)
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
    attractor_finder = AttractorFinder(
        residuals_per_period=20,
        targetted_subharmonics=target_subharmonics,
        max_periods=2000,
        controller=controller,
        solver=solver,
    )

    
    problem = problem_class(fd=fd, Ad=2.5)
    key = jax.random.PRNGKey(758493)
    Nstart = 40
    init_conditions = (
        (jax.random.uniform(key, shape=(Nstart, 5)) - 0.5)
        * 2.0
        * jnp.array([5.0 * problem.xw, 10.0 * problem.xw * problem.w0, 1.0, 0.0, 0.0])
    )

    batched_find = build_batched_finder(attractor_finder.find_attractors)
    problems, finder_configs, vmaped_init, solutions = batched_find(
        problem,
        init_conditions,
        finder_config,
    )

    processed = post_process_attractor_finder_results(
        problem_class=problem_class,
        problems=problems,
        finder_configs=finder_configs,
        init_conditions=vmaped_init,
        solutions=solutions,
        target_subharmonics=target_subharmonics,
        solution_state_labels=[lab + "a" for lab in problem_class.state_vector_labels],
    )
    save_results(processed)


if __name__ == "__main__":
    main()
