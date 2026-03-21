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
