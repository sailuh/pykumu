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
- `api.independence` -- configure independence tests (e.g., Fisher Z)
- `api.bootstrapping` -- configure bootstrap resampling parameters
- `api.knowledge` -- load domain knowledge constraints
- `api.algorithm` -- run causal search algorithms (FGES, BOSS)
- `api.graph` -- serialize, parse and inspect Tetrad graphs (cycles, bidirectional edges, Markov check)

## Usage

Call `api.tetrad.tetrad_jvm_start()` once before using any other module to start the
JVM and load the Tetrad JAR::

    from api import tetrad, data, score, algorithm
    tetrad.tetrad_jvm_start("path/to/tetrad-current.jar")

## Notebook

- [Issue Causal Analysis (FGES)](notebook/issue_causal_analysis_fges.html) --
  end-to-end causal discovery workflow using FGES.
- [Issue Causal Analysis (BOSS)](notebook/issue_causal_analysis_boss.html) --
  end-to-end causal discovery workflow using BOSS.
"""

__all__ = [
    "tetrad",
    "algorithm",
    "bootstrapping",
    "data",
    "graph",
    "knowledge",
    "score",
    "independence",
]
