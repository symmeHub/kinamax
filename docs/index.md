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

The documentation website can be published locally to the `gh-pages` branch
with `ghp-import`.

To build and publish the current documentation:

```bash
make -C docs publish
```

Important:

- `make -C docs publish` rebuilds the documentation locally before publishing.
- `make -C docs publish` pushes the generated HTML from `docs/_build/html` to the
  `gh-pages` branch with `ghp-import`.
- GitHub Pages must be configured to publish from the `gh-pages` branch.
- The `ghp-import` command is installed with `pip install -e .[docs]`.
