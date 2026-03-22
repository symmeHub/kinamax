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

publish:
	@command -v gh >/dev/null 2>&1 || { echo "GitHub CLI (gh) is required for publish."; exit 1; }
	@echo "Triggering GitHub Pages deployment from branch main"
	@echo "Only changes already committed and pushed to main will be published."
	@gh workflow run docs.yml --ref main
