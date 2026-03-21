"""Single (non-batched) time integration until convergence for the H46 problem.

This example mirrors the batched tutorial in
`docs/examples/batched_time_integration/`, but runs a *single* simulation:

- one driven oscillator (`kinamax.problems.H46Problem`)
- one drive frequency (`fd`)
- one initial condition (`X0`)

It integrates until the subharmonic shooting residuals satisfy the convergence
tolerance, then writes a tidy results table to `outputs/simulations.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np
import polars as pl
from diffrax import PIDController, Tsit5

from kinamax.core import (
    AttractorFinder,
    AttractorFinderConfig,
    post_process_attractor_finder_results,
)
from kinamax.problems import H46Problem

config.update("jax_enable_x64", True)  # Use double precision for improved accuracy


def main() -> None:
    # Store outputs next to the example so it is self-contained.
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fd = 50.0
    problem_class = H46Problem
    # The problem namespace exposes fixed labels directly on the class.
    state_vector_labels = problem_class.state_vector_labels
    # Build one physical parameter set for the H46 oscillator.
    problem = H46Problem.Params(fd=jnp.array(fd), Ad=jnp.array(2.5))

    # Finder configuration controls when integration starts, the initial step size,
    # and the convergence criterion used by the shooting residuals.
    finder_config = AttractorFinderConfig(
        convergence_tol=jnp.array(1.0e-10),
        target_frequency=fd,
        init_time=jnp.array(0.0),
        init_time_step=jnp.array(1.0e-3),
        subharmonic_factor=10.0,
    )

    solver = Tsit5()
    controller = PIDController(rtol=1e-8, atol=1e-9)
    # Test the fundamental response together with a few candidate subharmonics.
    target_subharmonics = np.array([1, 2, 3, 5], dtype=int)
    attractor_finder = AttractorFinder.Params(
        residuals_per_period=np.array(20, int),
        targetted_subharmonics=target_subharmonics,
        max_periods=2000,
        controller=controller,
        solver=solver,
    )

    key = jax.random.PRNGKey(758493)
    # Draw one random initial condition in a box scaled by the characteristic
    # displacement and velocity of the system. The energy state starts at zero.
    init_condition = (
        (jax.random.uniform(key, shape=(len(state_vector_labels),)) - 0.5)
        * 2.0
        * jnp.array([5.0 * problem.xw, 10.0 * problem.xw * problem.w0, 0.0])
    )

    # JIT compile the full attractor-search pipeline for this one problem.
    find_one = jax.jit(
        lambda p, x0, cfg: AttractorFinder.find_attractors(
            attractor_finder, problem_class, p, x0, cfg
        )
    )
    problem, finder_config, init_condition, solution = find_one(
        problem, init_condition, finder_config
    )

    # Convert the structured solver output to a flat table that is convenient to
    # inspect, save, and post-process in later steps.
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


if __name__ == "__main__":
    main()
