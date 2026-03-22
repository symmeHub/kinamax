# Kinamax

`kinamax` is a JAX/Diffrax toolkit for exploring periodic and subharmonic
responses of nonlinear driven ODEs.

The documentation is built with Sphinx and MyST, so the pages are written in
Markdown while still supporting API autodocumentation.

```{toctree}
:maxdepth: 2
:caption: Contents

api/index
examples/index
```

## What is in the package

- `kinamax.integration.core` contains the attractor search, clustering, and
  post-processing utilities.
- `kinamax.integration.models` contains data-oriented time-integration models
  such as `H46Problem`.
- `kinamax.hbm` contains harmonic-balance helpers for Fourier coefficient
  manipulations.

## Install

Install the package in editable mode:

```bash
pip install -e .
```

Install the documentation dependencies:

```bash
pip install -e .[docs]
```

## Build The Docs

From the repository root:

```bash
make -C docs html
```

The generated site is written to `docs/_build/html`.

## Serve The Docs Locally

To avoid `file://` access issues in the browser, serve the generated site over
HTTP:

```bash
make -C docs html
make -C docs serve
```

By default the local site is available at `http://127.0.0.1:8000`.

You can choose a different host or port:

```bash
make -C docs serve HOST=127.0.0.1 PORT=8123
```

## Publish The Docs

The documentation website is deployed by GitHub Actions with the workflow in
`.github/workflows/docs.yml`.

To validate the documentation locally and then trigger the GitHub Pages
deployment workflow:

```bash
make -C docs html
make -C docs publish
```

Important:

- `make -C docs publish` triggers the `docs.yml` workflow with `gh workflow run`.
- `make -C docs publish` does not rebuild the docs locally.
- GitHub Pages publishes what is on `main`, not your uncommitted local changes.
- Commit and push your changes to `main` before running `make -C docs publish`
  if you want those changes to appear online.
