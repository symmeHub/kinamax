# Single time integration (until convergence)

This example is a minimal, **non-batched** version of the workflow shown in
`docs/examples/batched_time_integration/`.

It runs `kinamax.core.AttractorFinder` on a single `kinamax.problems.H46Problem`
(one drive frequency, one initial condition), integrates until convergence of the
subharmonic shooting residuals, and writes a tidy results table to disk.

The executable tutorial version is now the Jupytext notebook
`integrate_until_convergence_single.py`. It stays as a plain Python file, but it
is structured into notebook cells so Sphinx can render it as a narrative tutorial.

```{toctree}
:maxdepth: 1

integrate_until_convergence_single
```

## Run

From `docs/examples/single_time_integration/`:

```bash
python integrate_until_convergence_single.py
```

Outputs are written to `outputs/simulations.parquet`.

## What to tweak

- Drive frequency: edit `fd` in `integrate_until_convergence_single.py`
- Initial condition: edit `init_condition` (or its random seed)
- Convergence criteria: edit `AttractorFinderConfig(convergence_tol=..., subharmonic_factor=...)`
- Integration effort: edit `AttractorFinder.Params(max_periods=..., residuals_per_period=...)`
