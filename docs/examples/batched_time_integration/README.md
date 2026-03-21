# Batched time integration demo

This example shows how to use `kinamax` to run **many time integrations in parallel** (batched over initial conditions and a frequency sweep), detect steady-state **orbits/attractors**, and then reconstruct representative **limit cycles** for post-processing and plotting.

The dynamical system used here is `kinamax.integration.models.H46_EM_Problem`, a driven nonlinear oscillator with an electromechanical (EM) coupling term. The workflow is generic: you can swap in another `Container`-style problem as long as it provides `rhs()` and labels.

```{toctree}
:maxdepth: 1

step01_integrate_until_convergence
step02_detect_orbits
step03_calculate_orbits
step04_plots
```

## What you get

- A frequency sweep integrated for many random initial conditions, written to `outputs/simulations.parquet`
- Orbit detection results (orbit labels + mapping back to simulations), written to:
  - `outputs/orbits.parquet`
  - `outputs/sim_orbit.parquet`
- One-period sampled trajectories reconstructed from each attractor, written to `outputs/orbits_from_attractors/`
- A compact per-orbit summary table (including average power), written to `outputs/orbit_data.parquet`

## How it works (step-by-step)

## Step 1 — Integrate until convergence

The first notebook:

- Build an `AttractorFinder` (Diffrax solver + adaptive controller)
- Create a frequency sweep (`fd`) and a batch of random initial conditions
- `vmap` + `jit` the attractor search so it runs efficiently on a single device
- Post-process the raw solver output into a tidy `polars.DataFrame`
- Save the result to `outputs/simulations.parquet`

This is the “batched time integration” part of the example.

## Step 2 — Detect orbits

The second notebook calls `kinamax.integration.core.detect_orbits`, which groups converged attractors into orbit labels and produces:

- `outputs/orbits.parquet`: attractor/orbit metadata and representative attractor states
- `outputs/sim_orbit.parquet`: mapping between simulation runs and detected orbits

Note: the EM model tracks cumulative energy states (`Ev`, `Eh`). Orbit detection focuses on the dynamical variables, so the script zeroes the energy-like states before saving.

## Step 3 — Reconstruct limit cycles

The third notebook:

- Integrate *one drive period* starting from each detected attractor state
- Save a dense set of samples per period (configurable via `samples_per_period`)
- Write one parquet file per `(orbit_label, attractor_label)` in `outputs/orbits_from_attractors/`
- Recompute and store per-orbit energy/power summaries in `outputs/orbit_data.parquet`

## Step 4 — Plot

Open `step04_plots.ipynb` to visualise:

- orbit branches across the frequency sweep
- representative state trajectories
- energy / power metrics derived from the reconstructed cycles

## Running the example

From `docs/examples/batched_time_integration/`, run the first three stages in order:

```bash
python step01_integrate_until_convergence.py
python step02_detect_orbits.py
python step03_calculate_orbits.py
```

Or run the convenience script that executes steps 1–3 in order:

```bash
bash run_steps_1_to_3.sh
```

Then open `step04_plots.ipynb` in Jupyter to generate plots.

## Tips for adapting

- Sweep resolution: edit `fd = jnp.linspace(...)` in `step01_integrate_until_convergence.py`.
- Basin exploration: edit `Nstart` and the initial condition scaling in `step01_integrate_until_convergence.py`.
- Orbit sampling: edit `OrbitCalculator(samples_per_period=...)` in `step03_calculate_orbits.py`.
- Model parameters: override fields when building `H46_EM_Problem` instances (e.g. `Ad`, `Q`, `R`, ...).
