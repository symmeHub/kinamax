# Minimal Makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
HOST          ?= 127.0.0.1
PORT          ?= 8000

.PHONY: help clean doctest html serve publish

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

serve:
	@echo "Serving documentation at http://$(HOST):$(PORT)"
	@cd "$(BUILDDIR)/html" && python -m http.server "$(PORT)" --bind "$(HOST)"

publish: html
	@command -v ghp-import >/dev/null 2>&1 || { echo "ghp-import is required for publish."; exit 1; }
	@echo "Publishing documentation to branch gh-pages"
	@ghp-import -n -p "$(BUILDDIR)/html"
