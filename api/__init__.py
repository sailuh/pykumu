# Pykumu - https://github.com/sailuh/pykumu
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Pykumu API for causal analysis.

This package provides a modular Python API for causal discovery
using the Tetrad library via JPype. Each module exposes thin
wrappers around Tetrad's Java classes:

- `api.tetrad` -- start the JVM and load the Tetrad JAR
- `api.data` -- convert pandas DataFrames into Tetrad data and initialize search state
- `api.score` -- configure scoring functions (e.g., SEM BIC)
- `api.bootstrapping` -- configure bootstrap resampling parameters
- `api.knowledge` -- load domain knowledge constraints
- `api.algorithm` -- run causal search algorithms (FGES, BOSS)
- `api.graph` -- serialize and parse Tetrad graph objects, apply PNEF thresholds

## Usage

Call `api.tetrad.start()` once before using any other module to start the
JVM and load the Tetrad JAR::

    from api import tetrad, data, score, algorithm
    tetrad.start("path/to/tetrad-current.jar")

## Notebook

- [Issue Causal Analysis](notebook/issue_causal_analysis.html) --
  end-to-end causal discovery workflow demonstrating data preparation,
  null-variable bootstrapped search, 1-PNEF threshold derivation,
  and final causal graph inspection.
"""

__all__ = [
    "tetrad",
    "algorithm",
    "bootstrapping",
    "data",
    "graph",
    "knowledge",
    "score",
]
