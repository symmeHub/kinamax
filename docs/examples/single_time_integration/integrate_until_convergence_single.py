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
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fd = 50.0
    problem_class = H46Problem
    problem = problem_class(fd=jnp.array(fd), Ad=jnp.array(2.5))

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
    attractor_finder = AttractorFinder(
        residuals_per_period=np.array(20, int),
        targetted_subharmonics=target_subharmonics,
        max_periods=2000,
        controller=controller,
        solver=solver,
    )

    key = jax.random.PRNGKey(758493)
    init_condition = (
        (jax.random.uniform(key, shape=(len(problem_class.state_vector_labels),)) - 0.5)
        * 2.0
        * jnp.array([5.0 * problem.xw, 10.0 * problem.xw * problem.w0, 0.0])
    )

    find_one = jax.jit(
        lambda p, x0, cfg: attractor_finder.find_attractors(p, x0, cfg)
    )
    problem, finder_config, init_condition, solution = find_one(
        problem, init_condition, finder_config
    )

    processed = post_process_attractor_finder_results(
        problem_class=problem_class,
        problems=problem,
        finder_configs=finder_config,
        init_conditions=init_condition,
        solutions=solution,
        target_subharmonics=target_subharmonics,
        solution_state_labels=[lab + "a" for lab in problem_class.state_vector_labels],
    )

    out_path = output_dir / "simulations.parquet"
    processed.write_parquet(out_path)
    pl.Config.set_tbl_rows(8)
    print(processed)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

