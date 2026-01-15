# Kinamax

## Overview
kinamax is a JAX/Diffrax powered toolkit for analyzing periodic and subharmonic responses of driven nonlinear ODEs. It includes:

- A reference nonlinear oscillator model (H46) with a forced, damped dynamics.
- A vectorized orbit-finding routine that integrates the ODE and detects subharmonic periodic orbits via residual-based shooting.
- Utilities to convert simulation results to Polars DataFrames, cluster attractors, and derive orbit labels.

### Core Components

- Problem model: `H46Problem` defining the ODE right‑hand side and state weighting for normalization.
- Orbit search: `OrbitFinder` with Diffrax `Tsit5` and a PID stepsize controller; detects subharmonics across candidate periods.
- Results container: `OrbitFinderSolution` with helpers to flatten results to tables.
- Post‑processing: `cluster_points` (DBSCAN/Agglomerative/cuML) and `detect_attractors_orbits` to turn trajectories into attractor and orbit assignments.

### Examples

- Time integration demo: `docs/examples/time_integration/ode_run_sim.py` writes `data/simulations.parquet` for downstream analysis and plotting.
- Plotting apps: `docs/examples/time_integration/ode_structure_plot.py` and `docs/examples/time_integration/ode_structure_orbit_plot.py` visualize attractors/orbits.

### Requirements

- Python 3.11+
- Core libraries: JAX, Diffrax, Equinox, Polars, NumPy, scikit‑learn (optional: cuML for GPU DBSCAN, Plotly + Marimo for plots/apps)

Install (editable)

- From the repo root, run: `pip install -e .`
- To uninstall: `pip uninstall kinamax`

Tests

- Run with pytest: `pytest`
- Optional: install extras for tests: `pip install -e .[test]`

Docs

- Build HTML docs: `make -C docs html` (output in `docs/_build/html`)
- Optional: install extras for docs: `pip install -e .[docs]`
 
Examples (local)

- Ensure runtime deps are installed (e.g., `pip install jax diffrax equinox polars scikit-learn plotly marimo`).
- Run simulation: `python docs/examples/time_integration/ode_run_sim.py`.
- Explore plots: run the plotting apps in `docs/examples/time_integration/`.
