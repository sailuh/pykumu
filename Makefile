# Pykumu documentation build
#
# Equivalent to kumu's `devtools::document()` + `pkgdown::build_site()`:
#   - `pdoc` renders docstrings from api/*.py into docs/api/*.html
#   - `jupyter nbconvert` renders notebook/*.ipynb into docs/notebook/*.html
#
# Run from the repo root inside the `pykumu` conda env (see notebook Setup).
# Usage:
#   make docs           # regenerate API pages + every notebook
#   make docs-api       # API pages only
#   make docs-notebook  # notebooks only
#   make clean-docs     # wipe generated output

NOTEBOOKS := $(wildcard notebook/*.ipynb)

.PHONY: docs docs-api docs-notebook clean-docs

docs: docs-api docs-notebook

docs-api:
	pdoc -o docs api

docs-notebook:
	@for nb in $(NOTEBOOKS); do \
		jupyter nbconvert --to html --output-dir docs/notebook $$nb; \
	done

clean-docs:
	rm -rf docs/api docs/api.html docs/index.html docs/search.js docs/notebook
