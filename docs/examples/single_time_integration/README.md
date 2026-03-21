# Single time integration (until convergence)

This example is a minimal, **non-batched** version of the workflow shown in
`docs/examples/batched_time_integration/`.

It runs `kinamax.core.AttractorFinder` on a single `kinamax.problems.H46Problem`
(one drive frequency, one initial condition), integrates until convergence of the
subharmonic shooting residuals, and writes a tidy results table to disk.

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
